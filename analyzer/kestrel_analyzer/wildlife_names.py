"""Keyword names for every species Kestrel can store (UK fork).

``keyword_root_names(name)`` -> ``(root, english, latin)`` where root is one of
``Birds``, ``Reptiles``, ``Amphibians``, ``Insects``, ``Mammals``; or None for
names that are not a known species (e.g. "Unknown", "Animal").

Sources, in order:
  1. birds: :func:`kestrel_analyzer.bird_names.keyword_names` (BOU / IOC names)
  2. BioCLIP lists installed next to the BioCLIP model (embeddings/*.csv)
  3. SpeciesNet mammal labels bundled with Kestrel (models/speciesnet/*.labels.*.txt)
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

from .bird_names import keyword_names as _bird_keyword_names
from .config import SPECIESNET_MODEL_DIR

KEYWORD_ROOTS = ("Birds", "Reptiles", "Amphibians", "Insects", "Mammals")
_GROUP_ROOT = {"reptiles": "Reptiles", "amphibians": "Amphibians", "insects": "Insects"}


@functools.lru_cache(maxsize=4)
def _bioclip_index(model_dir: str | None = None) -> dict:
    idx: dict[str, tuple[str, str, str]] = {}
    try:
        if model_dir is None:
            from .ml.bioclip import find_model_dir

            found = find_model_dir()
            model_dir = str(found) if found else None
        if not model_dir:
            return idx
        emb = Path(model_dir) / "embeddings"
        # uk first so UK common names win over Europe/World duplicates
        files = sorted(emb.glob("*_*.csv"), key=lambda p: ("_uk" not in p.stem, "_europe" not in p.stem, p.name))
        for f in files:
            if f.stem == "negatives":
                continue
            with open(f, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    root = _GROUP_ROOT.get(r.get("group", ""))
                    if not root:
                        continue
                    triple = (root, r.get("english_name") or r["scientific_name"], r["scientific_name"])
                    for key in (r.get("display_name"), r.get("english_name"), r.get("scientific_name")):
                        if key:
                            idx.setdefault(key.strip().lower(), triple)
    except Exception:
        return idx
    return idx


@functools.lru_cache(maxsize=1)
def _mammal_index() -> dict:
    """SpeciesNet display name (title-cased common name) -> ('Mammals', English, Latin)."""
    idx: dict[str, tuple[str, str, str]] = {}
    try:
        from .ml.speciesnet_taxonomy import format_species_display_name

        files = sorted(Path(SPECIESNET_MODEL_DIR).glob("*.labels.*.txt"))
        for f in files[:1]:
            for line in f.read_text(encoding="utf-8").splitlines():
                parts = line.split(";")
                if len(parts) < 7 or parts[1].strip().lower() != "mammalia":
                    continue
                genus, epithet, common = parts[4].strip(), parts[5].strip(), parts[6].strip()
                if not (genus and epithet and common) or common.lower() == "human":
                    continue
                english = format_species_display_name(common)
                latin = f"{genus.capitalize()} {epithet.lower()}"
                idx.setdefault(english.lower(), ("Mammals", english, latin))
    except Exception:
        return idx
    return idx


def keyword_root_names(species: str, bioclip_dir: str | None = None):
    if not species or not str(species).strip():
        return None
    birds = _bird_keyword_names(species)
    if birds:
        return ("Birds", birds[0], birds[1])
    key = str(species).strip().lower()
    hit = _bioclip_index(bioclip_dir).get(key)
    if hit:
        return hit
    return _mammal_index().get(key)


@functools.lru_cache(maxsize=8)
def _region_records(region: str, model_dir: str | None = None) -> tuple:
    """Picker records for the BioCLIP lists of one region (empty if not installed)."""
    try:
        if model_dir is None:
            from .ml.bioclip import find_model_dir

            found = find_model_dir()
            model_dir = str(found) if found else None
        if not model_dir:
            return ()
        out = []
        for g, root in _GROUP_ROOT.items():
            f = Path(model_dir) / "embeddings" / f"{g}_{region}.csv"
            if not f.is_file():
                continue
            with open(f, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    out.append({
                        "canonical_common_name": r["display_name"],
                        "scientific_name": r["scientific_name"],
                        "family_sci": r.get("family", ""),
                        "family_common": root,
                        "order": r.get("order", ""),
                        "regions": [],
                        "alpha_4": "",
                        "aliases": [],
                        "is_model_species": True,
                    })
        return tuple(out)
    except Exception:
        return ()


def search_wildlife(query: str, region: str = "uk", limit: int = 10, model_dir: str | None = None) -> list[dict]:
    """Prefix-then-substring search over BioCLIP species (display, scientific name)."""
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []
    recs = _region_records(region, model_dir)
    starts, contains = [], []
    for r in recs:
        names = (r["canonical_common_name"].lower(), r["scientific_name"].lower())
        if any(n.startswith(q) or (" " + q) in n for n in names):
            starts.append(r)
        elif any(q in n for n in names):
            contains.append(r)
        if len(starts) >= limit:
            break
    return (starts + contains)[:limit]


def lookup_wildlife(name: str, region: str = "uk", model_dir: str | None = None) -> dict | None:
    key = (name or "").strip().lower()
    for r in _region_records(region, model_dir):
        if r["canonical_common_name"].lower() == key:
            return r
    return None


def clear_caches() -> None:
    _bioclip_index.cache_clear()
    _mammal_index.cache_clear()
    _region_records.cache_clear()


__all__ = ["KEYWORD_ROOTS", "keyword_root_names", "search_wildlife", "lookup_wildlife", "clear_caches"]
