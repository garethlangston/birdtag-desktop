#!/usr/bin/env python
"""Build reptile / amphibian / insect species lists per region from GBIF.

For each group x region we ask GBIF's occurrence search for a ``speciesKey`` facet
(HUMAN_OBSERVATION records, i.e. photos/sightings, occurrenceStatus=PRESENT), keep
species with at least ``MIN_COUNT[region]`` records (and at most ``CAP`` species,
highest counts first), then fetch each species' accepted name, full taxonomy and
English vernacular name once (cached).

GBIF backbone note: there is no class Reptilia; reptiles are the classes Squamata,
Testudines and Crocodylia. They are fetched separately and written with
class=Reptilia / order=<that class> so prompts follow BioCLIP's Tree-of-Life
taxonomy (Animalia Chordata Reptilia Squamata ...).

Output (``<out>/lists/``): ``<group>_<region>.csv`` with columns
  gbif_key, scientific_name, english_name, kingdom, phylum, class, order, family,
  genus, species, occurrences, group
Raw API responses are cached in ``<out>/gbif_cache/`` so re-runs are offline.

Network use: only this build script talks to GBIF (api.gbif.org); the Kestrel app
never does. User agent identifies the project; requests are throttled.

Usage:
  python build_bioclip_lists.py --out <models>/bioclip-lists [--regions uk europe world]
                                [--groups reptiles amphibians insects]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.gbif.org/v1"
UA = "KestrelUKFork-listbuilder/0.1 (giggaloop@romeos.cc)"
GROUPS = {
    # group: list of (GBIF classKey, order name used in prompts or None to keep GBIF's)
    "reptiles": [(11592253, "Squamata"), (11418114, "Testudines"), (11493978, "Crocodylia")],
    "amphibians": [(131, None)],
    "insects": [(216, None)],
}
GROUP_CLASS = {"reptiles": "Reptilia", "amphibians": "Amphibia", "insects": "Insecta"}
REGIONS = {"uk": {"country": "GB"}, "europe": {"continent": "EUROPE"}, "world": {}}
MIN_COUNT = {"uk": 5, "europe": 20, "world": 50}
# Caps keep text-embedding size/time sane; documented in FORK-NOTES.md.
CAP = {("insects", "world"): 20000, ("insects", "europe"): 10000, ("insects", "uk"): 8000}
DELAY = 0.12  # seconds between requests per worker
WORKERS = 8

_lock = threading.Lock()


def _cached_get(cache: Path, path: str, params: dict | None = None) -> dict:
    url = f"{API}/{path}" + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    key = hashlib.sha1(url.encode()).hexdigest()
    f = cache / key[:2] / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            break
        except Exception as e:  # noqa: BLE001 - retry politely
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    time.sleep(DELAY)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"_url": url, **data}), encoding="utf-8")
    return data


def facet_species(cache, class_key, region, min_count, cap):
    out, offset, page = [], 0, 1000
    while True:
        params = {"classKey": class_key, "basisOfRecord": "HUMAN_OBSERVATION",
                  "occurrenceStatus": "PRESENT", "limit": 0, "facet": "speciesKey",
                  "facetLimit": page, "facetOffset": offset, **REGIONS[region]}
        d = _cached_get(cache, "occurrence/search", params)
        counts = d["facets"][0]["counts"] if d.get("facets") else []
        for c in counts:
            if c["count"] < min_count or (cap and len(out) >= cap):
                return out
            out.append((int(c["name"]), int(c["count"])))
        if len(counts) < page:
            return out
        offset += page


def species_detail(cache, key, vernacular_fallback=True):
    d = _cached_get(cache, f"species/{key}")
    eng = ""
    v = d.get("vernacularName") or ""
    # species/{key}.vernacularName is GBIF's preferred name, usually English; check the
    # vernacular list for an English one when it is missing.
    if v:
        eng = v
    elif vernacular_fallback:
        vn = _cached_get(cache, f"species/{key}/vernacularNames", {"limit": 50})
        for r in vn.get("results", []):
            if r.get("language") == "eng" and r.get("vernacularName"):
                eng = r["vernacularName"]
                if r.get("preferred"):
                    break
    return d, eng


def build(out: Path, groups, regions):
    cache = out / "gbif_cache"
    (out / "lists").mkdir(parents=True, exist_ok=True)
    for group in groups:
        for region in regions:
            keys: dict[int, tuple[int, str | None]] = {}
            for class_key, order in GROUPS[group]:
                for k, n in facet_species(cache, class_key, region, MIN_COUNT[region], CAP.get((group, region))):
                    keys[k] = (n, order)
            cap = CAP.get((group, region))
            ranked = sorted(keys.items(), key=lambda kv: -kv[1][0])[: cap or None]
            print(f"{group}/{region}: {len(ranked)} species (min {MIN_COUNT[region]} records"
                  f"{f', cap {cap}' if cap else ''}); fetching details...", flush=True)
            done = [0]

            def _one(item):
                k, (n, order) = item
                d, eng = species_detail(cache, k, vernacular_fallback=(region == "uk"))
                with _lock:
                    done[0] += 1
                    if done[0] % 500 == 0:
                        print(f"   {done[0]}/{len(ranked)}", flush=True)
                if d.get("taxonomicStatus") not in (None, "ACCEPTED") and d.get("acceptedKey"):
                    pass  # facet keys are accepted species already; keep as is
                return {
                    "gbif_key": k,
                    "scientific_name": d.get("species") or d.get("canonicalName") or d.get("scientificName", ""),
                    "english_name": eng.strip(),
                    "kingdom": d.get("kingdom", "Animalia"),
                    "phylum": d.get("phylum", "Arthropoda" if group == "insects" else "Chordata"),
                    "class": GROUP_CLASS[group],
                    "order": order or d.get("order", ""),
                    "family": d.get("family", ""),
                    "genus": d.get("genus", ""),
                    "species": d.get("species") or d.get("canonicalName", ""),
                    "occurrences": n,
                    "group": group,
                }

            with ThreadPoolExecutor(WORKERS) as ex:
                rows = [r for r in ex.map(_one, ranked) if r["scientific_name"]]
            path = out / "lists" / f"{group}_{region}.csv"
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            named = sum(1 for r in rows if r["english_name"])
            print(f"   -> {path.name}: {len(rows)} rows, {named} with English names", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    a = ap.parse_args()
    build(a.out, a.groups, a.regions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
