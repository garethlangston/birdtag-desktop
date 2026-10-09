#!/usr/bin/env python
"""Build the iNat21 bird label / region data used by the EVA-02 classifier.

Inputs
  --hf-config  config.json of timm/eva02_large_patch14_clip_336.merged2b_ft_inat21
               (has ``label_names`` = 10,000 scientific names in model order and
               ``label_descriptions`` = "<common name>, <group>")
  --uk-csv     BOU British List (birdtag's uk_birds.csv: scientific_name, english_name,
               category, bbrc_rarity, model_label)
  analyzer/models/birds/birds_global.csv (IOC v15.1, Kestrel's catalogue)

Outputs (analyzer/models/birds_inat21/)
  labels.txt   10,000 model labels (scientific names) in model output order
  species.csv  one row per iNat21 *bird* class (1,486):
      index, inat_name, scientific_name, english_name, kestrel_name, family_common,
      family_sci, uk, uk_rarity, europe
    english_name  BOU English name for UK-list species (matches birdtag tags, e.g. "Robin"),
                  else IOC canonical_common_name, else iNat21 common name, else scientific.
    kestrel_name  Name the classifier reports to Kestrel: IOC canonical name from
                  birds_global.csv when the species is in the catalogue (so lookups and the
                  species combobox keep working), else english_name.
    uk            1 = BOU category A/C and not a BBRC rarity (birdtag default list)
    uk_rarity     1 = on the British List but excluded by default (BBRC rarity / B-only)
    europe        1 = IOC breeding region PAL (Palearctic), AO (Atlantic) or Worldwide, or uk.
                  IOC v15.1 has no finer "Europe" code, so Europe mode = Western Palearctic
                  plus the rest of the Palearctic (e.g. some East-Asian species).
  uk_birds.csv copy of the input UK list (provenance)

The rule for UK membership and the model_label fallback is identical to birdtag's
``load_uk_list``.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "analyzer" / "models" / "birds_inat21"
GLOBAL_CSV = REPO / "analyzer" / "models" / "birds" / "birds_global.csv"
EUROPE_REGIONS = {"PAL", "AO", "Worldwide"}


def uk_category_default(row: dict) -> bool:
    """birdtag rule: category A or C (E/F suffixes ignored) and not a BBRC rarity."""
    cat = row["category"].replace("E", "").replace("F", "")
    return not row["bbrc_rarity"] and ("A" in cat or "C" in cat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-config", required=True, type=Path)
    ap.add_argument("--uk-csv", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    cfg = json.loads(a.hf_config.read_text(encoding="utf-8"))
    labels: list[str] = cfg["label_names"]
    desc: dict[str, str] = cfg["label_descriptions"]
    assert len(labels) == 10000, len(labels)
    bird_idx = [i for i, n in enumerate(labels) if desc.get(n, "").endswith(", Bird")]
    index_of = {n: i for i, n in enumerate(labels)}

    # IOC catalogue indexes
    ioc_by_sci, ioc_by_name = {}, {}
    with open(GLOBAL_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ioc_by_sci.setdefault(r["scientific_name"].strip().lower(), r)
            ioc_by_name.setdefault(r["canonical_common_name"].strip().lower(), r)
            for al in (r.get("aliases") or "").split("|"):
                if al.strip():
                    ioc_by_name.setdefault(al.strip().lower(), r)

    # UK list -> model index (birdtag's scientific_name / model_label fallback)
    uk_default: dict[int, dict] = {}
    uk_rare: dict[int, dict] = {}
    uk_missing = []
    with open(a.uk_csv, encoding="utf-8", newline="") as f:
        uk_rows = list(csv.DictReader(f))
    for r in uk_rows:
        label = r["scientific_name"] if r["scientific_name"] in index_of else (r.get("model_label") or "")
        if label not in index_of:
            if uk_category_default(r):
                uk_missing.append(r["english_name"])
            continue
        target = uk_default if uk_category_default(r) else uk_rare
        target.setdefault(index_of[label], r)

    rows = []
    unmatched = []
    for i in bird_idx:
        inat = labels[i]
        inat_common = desc[inat].rsplit(", ", 1)[0]
        uk_row = uk_default.get(i) or uk_rare.get(i)
        ioc = ioc_by_sci.get(inat.lower())
        if ioc is None and uk_row is not None:
            ioc = ioc_by_sci.get(uk_row["scientific_name"].lower())
        if ioc is None:
            ioc = ioc_by_name.get(inat_common.lower())
        if ioc is None:
            unmatched.append(inat)
        sci = uk_row["scientific_name"] if uk_row else (ioc["scientific_name"] if ioc else inat)
        english = (uk_row["english_name"] if uk_row else
                   (ioc["canonical_common_name"] if ioc else (inat_common or inat)))
        kestrel_name = ioc["canonical_common_name"] if ioc else english
        regions = set((ioc["regions"] if ioc else "").split(","))
        is_uk = i in uk_default
        is_rare = (not is_uk) and i in uk_rare
        rows.append({
            "index": i,
            "inat_name": inat,
            "scientific_name": sci,
            "english_name": english,
            "kestrel_name": kestrel_name,
            "family_common": ioc["family_common"] if ioc else "",
            "family_sci": ioc["family_sci"] if ioc else "",
            "uk": int(is_uk),
            "uk_rarity": int(is_rare),
            "europe": int(is_uk or bool(regions & EUROPE_REGIONS)),
        })

    # kestrel_name must be unique per class or two model outputs would collapse into one
    # catalogue entry; keep the first and fall back to english/scientific for the rest.
    seen = {}
    for r in rows:
        if r["kestrel_name"] in seen:
            r["kestrel_name"] = r["english_name"] if r["english_name"] not in seen else r["scientific_name"]
        seen[r["kestrel_name"]] = r["index"]

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    with open(a.out / "species.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.uk_csv.resolve() != (a.out / "uk_birds.csv").resolve():
        shutil.copyfile(a.uk_csv, a.out / "uk_birds.csv")

    print(f"bird classes: {len(rows)} (indices {bird_idx[0]}..{bird_idx[-1]})")
    print(f"UK default: {sum(r['uk'] for r in rows)}  UK rarities: {sum(r['uk_rarity'] for r in rows)}"
          f"  Europe: {sum(r['europe'] for r in rows)}  World: {len(rows)}")
    print(f"UK default species not in the model: {len(uk_missing)}")
    print(f"iNat21 birds not matched to IOC catalogue: {len(unmatched)} e.g. {unmatched[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
