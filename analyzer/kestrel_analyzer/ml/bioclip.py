"""BioCLIP zero-shot species classifier for reptiles, amphibians and insects (UK fork).

Model-agnostic: everything model-specific comes from ``config.json`` written by
``tools/uk_models/export_bioclip.py`` next to the ONNX image encoder, and the
species text embeddings come from ``embeddings/<group>_<region>.npy|csv`` written by
``build_bioclip_embeddings.py``. Dropping a BioCLIP 2.5 folder in works the same way.

Folder layout (inside a models_extra search dir)::

    bioclip-2/
      config.json
      image_encoder_fp16.onnx | image_encoder_fp32.onnx
      embeddings/{reptiles,amphibians,insects}_{uk,europe,world}.npy + .csv
      embeddings/negatives.npy + .csv

The ONNX session is created lazily on first use and registered with the shared
:class:`ProviderCoordinator`, so GPU->CPU fallback covers it like every other model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..logging_utils import debug
from .bird_species_inat21 import model_search_dirs, square_context_crop  # noqa: F401 (re-export)

GROUPS = ("reptiles", "amphibians", "insects")
GROUP_ROOTS = {"reptiles": "Reptiles", "amphibians": "Amphibians", "insects": "Insects"}
REGIONS = ("uk", "europe", "world")

# SpeciesNet taxonomy segments (class / order) that send a detection to BioCLIP.
_TAXON_GROUP = {
    "reptilia": "reptiles", "squamata": "reptiles", "testudines": "reptiles", "crocodylia": "reptiles",
    "serpentes": "reptiles", "sauria": "reptiles",
    "amphibia": "amphibians", "anura": "amphibians", "caudata": "amphibians", "urodela": "amphibians",
    "gymnophiona": "amphibians",
    "insecta": "insects", "lepidoptera": "insects", "odonata": "insects", "coleoptera": "insects",
    "hymenoptera": "insects", "diptera": "insects", "hemiptera": "insects", "orthoptera": "insects",
    "mantodea": "insects", "neuroptera": "insects", "trichoptera": "insects",
}
_GENERIC = {"", "animal", "unknown", "no cv result", "blank"}


def group_for_taxonomy(raw: str) -> str | None:
    """'reptiles' | 'amphibians' | 'insects' for a SpeciesNet label, 'any' for a generic
    animal / unknown / blank label, None for anything else (mammals, birds, humans...)."""
    parts = [p.strip().lower() for p in str(raw or "").split(";")]
    taxa = parts[1:6] if len(parts) >= 7 else parts
    for p in taxa:
        if p in _TAXON_GROUP:
            return _TAXON_GROUP[p]
    if "mammalia" in taxa or "aves" in taxa:
        return None
    common = parts[-1] if parts else ""
    if all(not p for p in taxa) and common in _GENERIC:
        return "any"
    if common in _GENERIC and not any(p in ("mammalia", "aves") for p in taxa):
        return "any"
    return None


class TaxonLabel(str):
    """A pipeline ``pred_class`` string that also carries the BioCLIP routing group.

    Behaves exactly like the display string everywhere (comparisons, JSON), so the
    detector's 4-tuple API and ``pred_class[i] == "bird"`` checks are unchanged.
    """

    bioclip_group: str | None
    taxonomy: str

    def __new__(cls, value: str, bioclip_group: str | None = None, taxonomy: str = ""):
        obj = super().__new__(cls, value)
        obj.bioclip_group = bioclip_group
        obj.taxonomy = taxonomy
        return obj


def find_model_dir() -> Path | None:
    """First folder (newest name first) under the model search dirs that holds a
    BioCLIP ``config.json`` and an image encoder."""
    candidates = []
    for d in model_search_dirs():
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            cfg = sub / "config.json"
            if sub.is_dir() and sub.name.lower().startswith("bioclip") and cfg.is_file():
                if any((sub / n).is_file() for n in ("image_encoder_fp16.onnx", "image_encoder_fp32.onnx")):
                    candidates.append(sub)
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0] if candidates else None


def missing_message() -> str:
    folders = "; ".join(str(d) for d in model_search_dirs())
    return ("BioCLIP 2 files not found (folder 'bioclip-2' with config.json, image_encoder_*.onnx and "
            f"embeddings/). Looked in: {folders}. Reptiles, amphibians and insects use SpeciesNet only.")


def _preprocess(image_rgb: np.ndarray, cfg: dict) -> np.ndarray:
    from PIL import Image

    size = int(cfg["input_size"])
    resize = cfg.get("resize") or size
    resize = resize[0] if isinstance(resize, (list, tuple)) else int(resize)
    interp = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}.get(cfg.get("interpolation", "bicubic"),
                                                                        Image.BICUBIC)
    img = Image.fromarray(np.ascontiguousarray(image_rgb[..., :3]).astype(np.uint8), "RGB")
    w, h = img.size
    if w <= h:
        nw, nh = resize, int(resize * h / w)
    else:
        nw, nh = int(resize * w / h), resize
    img = img.resize((nw, nh), interp)
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class SpeciesList:
    """Text embeddings + names for one (group, region) list."""

    def __init__(self, npy: Path, csv_path: Path):
        self.emb = np.load(npy).astype(np.float32)
        with open(csv_path, encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if len(self.rows) != len(self.emb):
            raise ValueError(f"{csv_path.name}: {len(self.rows)} names vs {len(self.emb)} embeddings")


class BioclipClassifier:
    """Zero-shot classifier over the reptile / amphibian / insect lists of one region."""

    def __init__(self, model_dir, coord, region: str = "uk", prefer_fp16: bool = False):
        self.model_dir = Path(model_dir)
        self.cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.coord = coord
        self.region = region if region in REGIONS else "world"
        names = [self.cfg["onnx"]["fp16"], self.cfg["onnx"]["fp32"]]
        if not prefer_fp16:
            names.reverse()
        self.model_path = next((self.model_dir / n for n in names if (self.model_dir / n).is_file()), None)
        if self.model_path is None:
            raise FileNotFoundError(missing_message())
        self.scale = float(self.cfg.get("logit_scale", 100.0))
        self._session = None
        self._lists: dict[str, SpeciesList] = {}
        emb = self.model_dir / "embeddings"
        for g in GROUPS:
            npy, names_csv = emb / f"{g}_{self.region}.npy", emb / f"{g}_{self.region}.csv"
            if npy.is_file() and names_csv.is_file():
                self._lists[g] = SpeciesList(npy, names_csv)
        neg = emb / "negatives.npy"
        self.negatives = np.load(neg).astype(np.float32) if neg.is_file() else np.zeros((0, 1), np.float32)
        if not self._lists:
            raise FileNotFoundError(f"No BioCLIP embeddings for region '{self.region}' in {emb}")
        debug(f"[BioCLIP] {self.model_dir.name} region={self.region} lists="
              f"{ {g: len(l.rows) for g, l in self._lists.items()} } model={self.model_path.name}")

    @property
    def groups(self) -> list[str]:
        return list(self._lists)

    def _sess(self):
        if self._session is None:  # lazy: only pay the memory when a herp/insect shows up
            from .resilient_session import ResilientOnnxSession

            self._session = ResilientOnnxSession("bioclip", self.model_path, self.coord)
        return self._session

    @property
    def providers_used(self):
        return self._session.get_providers() if self._session is not None else []

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        s = self._sess()
        name = s.get_inputs()[0].name
        e = np.asarray(s.run(None, {name: _preprocess(image_rgb, self.cfg)})[0][0], dtype=np.float32)
        return e / max(float(np.linalg.norm(e)), 1e-12)

    def classify_embedding(self, emb: np.ndarray, groups=None, use_negatives: bool = False, top_k: int = 5):
        groups = [g for g in (groups or GROUPS) if g in self._lists]
        if not groups:
            return None
        txt = np.concatenate([self._lists[g].emb for g in groups])
        meta = [(g, r) for g in groups for r in self._lists[g].rows]
        n_species = len(meta)
        if use_negatives and len(self.negatives) and self.negatives.shape[1] == txt.shape[1]:
            txt = np.concatenate([txt, self.negatives])
        logits = self.scale * (txt @ emb)
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        species_p = p[:n_species]
        neg_p = float(p[n_species:].max()) if len(p) > n_species else 0.0
        k = min(top_k, n_species)
        top = np.argsort(species_p)[-k:][::-1]
        # family (scientific) aggregation for the family pill
        fam: dict[str, float] = {}
        for i, (_, r) in enumerate(meta):
            f = r.get("family") or "Unknown Family"
            fam[f] = fam.get(f, 0.0) + float(species_p[i])
        fams = sorted(fam.items(), key=lambda kv: -kv[1])[:k]
        best_g, best_r = meta[top[0]]
        return {
            "top_species_labels": np.array([meta[i][1]["display_name"] for i in top]),
            "top_species_scores": species_p[top].astype(float),
            "top_family_labels": [f for f, _ in fams],
            "top_family_scores": [s for _, s in fams],
            "group": best_g,
            "scientific_name": best_r["scientific_name"],
            "english_name": best_r.get("english_name", ""),
            "negative_score": neg_p,
            "beats_negatives": float(species_p[top[0]]) > neg_p,
        }

    def classify(self, image_rgb, groups=None, use_negatives: bool = False, top_k: int = 5):
        return self.classify_embedding(self.embed(image_rgb), groups, use_negatives, top_k)


__all__ = ["BioclipClassifier", "TaxonLabel", "GROUPS", "GROUP_ROOTS", "REGIONS",
           "find_model_dir", "group_for_taxonomy", "missing_message"]
