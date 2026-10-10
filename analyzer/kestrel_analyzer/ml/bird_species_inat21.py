"""EVA-02 Large iNat21 bird classifier with UK / Europe / World region masks.

Drop-in alternative to :class:`BirdSpeciesClassifier` (same ``classify``
contract) backed by ``timm/eva02_large_patch14_clip_336.merged2b_ft_inat21``
exported to ONNX by ``tools/uk_models/export_eva02.py``.

* The model has 10,000 iNat21 outputs (1,486 birds). The softmax is taken over
  the selected region's bird indices only, so a Grey Heron cannot come out as
  a Great Blue Heron in UK mode.
* Species names returned are Kestrel catalogue names (IOC canonical, see
  :mod:`kestrel_analyzer.bird_names`), so lookups downstream keep working.
* The weights are CC-BY-NC-4.0 and are NOT shipped in the repo; they are
  looked up in :func:`model_search_dirs` (user ``models_extra`` folder first).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..bird_names import INAT21_DATA_DIR, load_inat21_species
from ..logging_utils import debug, error

REGIONS = ("uk", "europe", "world")
MODEL_FILENAMES = {
    "fp16": "eva02_large_inat21_fp16.onnx",
    "fp32": "eva02_large_inat21_fp32.onnx",
}
# timm pretrained_cfg for eva02_large_patch14_clip_336.merged2b_ft_inat21
INPUT_SIZE = 336
CROP_PCT = 1.0
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
# Padding added around the detector box before the square crop (birdtag uses the same).
CONTEXT_PAD = 0.15


class Inat21ModelMissing(FileNotFoundError):
    """Raised when no EVA-02 ONNX file is found; message lists the folders searched."""


def user_models_extra_dir() -> Path:
    """``<user data dir>/models_extra`` (Windows: %LOCALAPPDATA%\\BirdtagDesktop\\models_extra; name from branding.json)."""
    try:
        try:
            from settings_utils import _get_user_data_dir  # type: ignore
        except ImportError:
            from analyzer.settings_utils import _get_user_data_dir  # type: ignore
        return Path(_get_user_data_dir()) / "models_extra"
    except Exception:
        pass
    try:
        from branding import data_dir_name  # type: ignore
    except ImportError:
        from analyzer.branding import data_dir_name  # type: ignore
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / data_dir_name() / "models_extra"


def legacy_models_extra_dirs() -> list[Path]:
    """models_extra under upstream Project Kestrel's data folder (phase 1/2 install path)."""
    try:
        try:
            from settings_utils import _get_legacy_user_data_dirs  # type: ignore
        except ImportError:
            from analyzer.settings_utils import _get_legacy_user_data_dirs  # type: ignore
        return [Path(d) / "models_extra" for d in _get_legacy_user_data_dirs()]
    except Exception:
        return []


def model_search_dirs() -> list[Path]:
    dirs = []
    env = os.environ.get("KESTREL_MODELS_EXTRA_DIR", "").strip()
    if env:
        dirs.append(Path(env))
    dirs += [user_models_extra_dir(), *legacy_models_extra_dirs(), INAT21_DATA_DIR]
    return dirs


def find_model(prefer_fp16: bool = False) -> Path | None:
    order = ("fp16", "fp32") if prefer_fp16 else ("fp32", "fp16")
    for d in model_search_dirs():
        for kind in order:
            p = d / MODEL_FILENAMES[kind]
            if p.is_file():
                return p
    return None


def missing_model_message() -> str:
    names = " or ".join(MODEL_FILENAMES[k] for k in ("fp16", "fp32"))
    folders = "; ".join(str(d) for d in model_search_dirs())
    return (f"UK/Europe/World bird model not found. Put {names} in one of: {folders}. "
            "See FORK-NOTES.md for how to export it.")


def region_rows(region: str, include_rarities: bool = False, rows=None) -> list[dict]:
    """Bird classes allowed in ``region`` ('uk' | 'europe' | 'world')."""
    rows = load_inat21_species() if rows is None else rows
    region = (region or "uk").lower()
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")
    if region == "world":
        return list(rows)
    out = []
    for r in rows:
        rare_ok = include_rarities and bool(r["uk_rarity"])
        if region == "uk" and (r["uk"] or rare_ok):
            out.append(r)
        elif region == "europe" and (r["uk"] or r["europe"] or rare_ok):
            out.append(r)
    return out


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> 1x3x336x336 float32, identical to timm's eval transform.

    timm cfg: Resize(shorter side = floor(336 / crop_pct), bicubic, PIL) ->
    CenterCrop(336) -> ToTensor -> Normalize(CLIP mean/std).
    """
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(image_rgb[..., :3]).astype(np.uint8), "RGB")
    w, h = img.size
    scale = int(INPUT_SIZE / CROP_PCT)
    # torchvision.transforms.functional.resize semantics for an int size
    if w <= h:
        new_w, new_h = scale, int(scale * h / w)
    else:
        new_w, new_h = int(scale * w / h), scale
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = int(round((new_w - INPUT_SIZE) / 2.0))
    top = int(round((new_h - INPUT_SIZE) / 2.0))
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def square_context_crop(box, img: np.ndarray, pad: float = CONTEXT_PAD) -> np.ndarray:
    """Square crop around a pipeline box ``((xmin, ymin), (xmax, ymax))`` with padding.

    The detector box is tight and often tall/wide; timm's centre crop would cut
    the head or tail off. A padded square (clamped to the image) keeps the whole
    bird plus a little context, which is what the model was trained on.
    """
    h, w = img.shape[:2]
    x1, y1 = float(box[0][0]), float(box[0][1])
    x2, y2 = float(box[1][0]), float(box[1][1])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    side = max(1.0, min(side, w, h))
    left = min(max(0.0, cx - side / 2), w - side)
    top = min(max(0.0, cy - side / 2), h - side)
    l, t, s = int(left), int(top), int(side)
    return img[t:t + s, l:l + s]


class Inat21BirdSpeciesClassifier:
    """Same public interface as ``BirdSpeciesClassifier``.

    ``image`` passed to :meth:`classify` is an RGB uint8 array (Kestrel decodes
    with PIL/rawpy, which yield RGB).
    """

    wants_square_context = True

    def __init__(self, model_path, coord, region: str = "uk", include_rarities: bool = False,
                 species_csv: str | None = None):
        from .resilient_session import ResilientOnnxSession

        self.region = region
        self.include_rarities = bool(include_rarities)
        rows = region_rows(region, include_rarities, rows=load_inat21_species(species_csv))
        if not rows:
            raise RuntimeError(f"No iNat21 species data for region {region!r} "
                               f"(missing {INAT21_DATA_DIR / 'species.csv'}?)")
        self.indices = np.array([r["index"] for r in rows], dtype=np.int64)
        self.labels = np.array([r["kestrel_name"] for r in rows])
        families = [r["family_common"] or "Unknown Family" for r in rows]
        _, first = np.unique(families, return_index=True)
        self.family_display_names = [families[i] for i in sorted(first)]
        fam_pos = {f: i for i, f in enumerate(self.family_display_names)}
        self.family_matrix = np.zeros((len(self.family_display_names), len(rows)), dtype=np.float32)
        self.family_matrix[[fam_pos[f] for f in families], np.arange(len(rows))] = 1.0

        self.model_path = str(model_path)
        self.session = ResilientOnnxSession("bird_species_inat21", model_path, coord)
        active = self.session.get_providers()
        self.providers_used = active
        debug(f"[Inat21BirdSpeciesClassifier] region={region} rarities={self.include_rarities} "
              f"classes={len(rows)} model={Path(model_path).name} providers={active}")

    def _logits(self, input_tensor: np.ndarray) -> np.ndarray:
        name = self.session.get_inputs()[0].name
        return np.asarray(self.session.run(None, {name: input_tensor})[0][0], dtype=np.float32)

    def classify(self, image, top_k=5):
        try:
            logits = self._logits(preprocess(image))
        except Exception as e:
            error(f"[Inat21BirdSpeciesClassifier] inference failed: {e}")
            raise
        sub = logits[self.indices]
        sub = sub - sub.max()
        probs = np.exp(sub)
        probs /= probs.sum()

        k = min(top_k, len(probs))
        top = np.argsort(probs)[-k:][::-1]
        fam_probs = self.family_matrix @ probs
        top_f = np.argsort(fam_probs)[-k:][::-1]
        return {
            "top_species_labels": self.labels[top],
            "top_species_scores": probs[top].astype(float),
            "top_family_labels": [self.family_display_names[i] for i in top_f],
            "top_family_scores": fam_probs[top_f].astype(float).tolist(),
        }


__all__ = [
    "Inat21BirdSpeciesClassifier", "Inat21ModelMissing", "REGIONS", "MODEL_FILENAMES",
    "find_model", "missing_model_message", "model_search_dirs", "preprocess",
    "region_rows", "square_context_crop", "user_models_extra_dir",
]
