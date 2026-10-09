"""Bird name tables for the iNat21 (EVA-02) classifier and for keyword writing.

``analyzer/models/birds_inat21/species.csv`` (built by
``tools/uk_models/build_inat21_region_lists.py``) has one row per iNat21 bird
class with:

* ``kestrel_name``  - the name the classifier reports to Kestrel. It is the IOC
  ``canonical_common_name`` from ``birds/birds_global.csv`` whenever the species
  is in Kestrel's catalogue, so the species combobox, ``lookup_birds`` and every
  other name-keyed lookup keep working unchanged.
* ``english_name``  - the name used in keywords: BOU English name for species on
  the British List (``Robin``, ``Grey Heron``), else the IOC name.
* ``scientific_name`` - BOU/IOC binomial written as the Latin keyword.

The BOU names are also exposed as catalogue aliases (``bou_aliases``), so a
user typing "Robin" in the combobox finds "European Robin" and a lookup of a
BOU name resolves to the canonical record instead of failing.

This module has no ONNX / UI dependencies.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

from .config import MODELS_DIR

INAT21_DATA_DIR = MODELS_DIR / "birds_inat21"
SPECIES_CSV = INAT21_DATA_DIR / "species.csv"
GLOBAL_CSV = MODELS_DIR / "birds" / "birds_global.csv"

_INT_COLS = ("index", "uk", "uk_rarity", "europe")


@functools.lru_cache(maxsize=4)
def load_inat21_species(path: str | None = None) -> tuple[dict, ...]:
    """Rows of species.csv (ints converted). Empty tuple if the file is missing."""
    p = Path(path) if path else SPECIES_CSV
    if not p.is_file():
        return ()
    rows = []
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in _INT_COLS:
                r[k] = int(r.get(k) or 0)
            rows.append(r)
    return tuple(rows)


@functools.lru_cache(maxsize=1)
def _global_by_name() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not GLOBAL_CSV.is_file():
        return out
    with open(GLOBAL_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["canonical_common_name"].strip().lower(), r)
    return out


@functools.lru_cache(maxsize=1)
def _name_index() -> dict[str, tuple[str, str]]:
    """lower-cased kestrel/english/iNat names -> (english keyword name, scientific)."""
    idx: dict[str, tuple[str, str]] = {}
    uk_first = sorted(load_inat21_species(), key=lambda r: -(r["uk"] * 2 + r["uk_rarity"]))
    for r in uk_first:
        pair = (r["english_name"], r["scientific_name"])
        for key in (r["kestrel_name"], r["english_name"], r["inat_name"], r["scientific_name"]):
            if key:
                idx.setdefault(key.strip().lower(), pair)
    return idx


def keyword_names(species: str) -> tuple[str, str] | None:
    """Map any species name Kestrel may hold to ``(english, scientific)`` for keywords.

    Accepts the classifier's catalogue name ("European Robin"), a BOU name
    ("Robin"), a scientific name, or any other catalogue name (e.g. an NA-model
    label such as "American Robin"). Returns ``None`` when the name is unknown,
    in which case callers should fall back to the raw name with no Latin.
    """
    if not species:
        return None
    key = species.strip().lower()
    hit = _name_index().get(key)
    if hit:
        return hit
    rec = _global_by_name().get(key)
    if rec:
        return rec["canonical_common_name"], rec["scientific_name"]
    return None


def bou_aliases() -> dict[str, list[str]]:
    """IOC canonical name -> BOU English names that differ from it (catalogue aliases)."""
    out: dict[str, list[str]] = {}
    for r in load_inat21_species():
        if (r["uk"] or r["uk_rarity"]) and r["english_name"] != r["kestrel_name"]:
            out.setdefault(r["kestrel_name"], [])
            if r["english_name"] not in out[r["kestrel_name"]]:
                out[r["kestrel_name"]].append(r["english_name"])
    return out


__all__ = ["INAT21_DATA_DIR", "load_inat21_species", "keyword_names", "bou_aliases"]
