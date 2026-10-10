#!/usr/bin/env python
"""Compute BioCLIP text embeddings for the GBIF species lists (build time, PyTorch).

For every ``<lists>/<group>_<region>.csv`` (from build_bioclip_lists.py) this writes
into ``<model_dir>/embeddings/``:
  <group>_<region>.npy   float16, N x D, L2-normalised text embeddings
  <group>_<region>.csv   display_name, english_name, scientific_name, group, order,
                         family, gbif_key, occurrences   (row i <-> embedding i)
  negatives.npy / .csv   background prompts used by the whole-frame route

Prompt format follows BioCLIP's "taxonomic + common name" text type, the one the
BioCLIP authors use for zero-shot evaluation:
  "a photo of Animalia Chordata Reptilia Squamata Lacertidae Zootoca vivipara with common name Common Lizard."
(falls back to the taxonomic string alone when no English name is known).

English names: GBIF vernacular names are tidied (first variant, no parentheses,
Title Case) and overridden by tools/uk_models/data/uk_wildlife_names.csv for common
UK species so keywords use familiar British names.

Usage:
  python build_bioclip_embeddings.py --lists <models>/bioclip-lists/lists \
      --model-dir <models>/bioclip-2 [--only uk europe] [--batch 64]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

OVERRIDES = Path(__file__).with_name("data") / "uk_wildlife_names.csv"
NEGATIVES = [
    "a photo of a plant.", "a photo of a flower.", "a photo of leaves.", "a photo of grass.",
    "a photo of a tree.", "a landscape photo.", "a photo of the sky.", "a photo of water.",
    "a photo of a rock.", "a photo of a person.", "a photo of a building.", "a photo of a car.",
    "a photo of food.", "a photo of a bird.", "a photo of a mammal.", "a photo of a fish.",
    "a photo of a spider.", "a photo of a snail.", "a blurry photo.", "a dark photo.",
]


def tidy_english(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    name = re.split(r"[,;/]", name)[0]
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\s+", " ", name).strip(" -")
    if not name:
        return ""
    words = []
    for w in name.split(" "):
        # Title-case words but keep hyphenated parts and inner capitals sensible.
        words.append("-".join(p[:1].upper() + p[1:].lower() if p else p for p in w.split("-")))
    return " ".join(words)


def taxonomy(row: dict) -> str:
    epithet = row["species"].split(" ", 1)[1] if " " in row["species"] else row["species"]
    parts = [row["kingdom"], row["phylum"], row["class"], row["order"], row["family"], row["genus"], epithet]
    return " ".join(p for p in parts if p)


def load_overrides() -> dict[str, str]:
    if not OVERRIDES.exists():
        return {}
    with open(OVERRIDES, encoding="utf-8") as f:
        return {r["scientific_name"]: r["english_name"] for r in csv.DictReader(f)}


def encode(model, tok, texts, batch, label):
    import torch

    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        with torch.no_grad():
            e = model.encode_text(tok(texts[i:i + batch]))
            e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.numpy().astype(np.float32))
        done = min(i + batch, len(texts))
        rate = done / max(time.time() - t0, 1e-6)
        print(f"   {label}: {done}/{len(texts)}  ({rate:.1f}/s, eta {(len(texts) - done) / rate / 60:.1f} min)",
              flush=True)
    return np.concatenate(out) if out else np.zeros((0, model.text_projection.shape[1]), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path, help="folder with config.json (export_bioclip.py)")
    ap.add_argument("--only", nargs="*", default=None, help="regions to build (default: all lists found)")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import open_clip
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = json.loads((a.model_dir / "config.json").read_text())
    model, _, _ = open_clip.create_model_and_transforms(cfg["model_id"])
    model.eval()
    model.visual = None  # text tower only; frees ~1.2 GB
    tok = open_clip.get_tokenizer(cfg["model_id"])
    out = a.model_dir / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides()

    neg = encode(model, tok, NEGATIVES, a.batch, "negatives")
    np.save(out / "negatives.npy", neg.astype(np.float16))
    with open(out / "negatives.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["prompt"]] + [[p] for p in NEGATIVES])

    for path in sorted(a.lists.glob("*_*.csv")):
        group, region = path.stem.split("_", 1)
        if a.only and region not in a.only:
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        prompts, meta, seen = [], [], {}
        for r in rows:
            eng = overrides.get(r["scientific_name"]) or tidy_english(r["english_name"])
            tax = taxonomy(r)
            prompts.append(f"a photo of {tax} with common name {eng}." if eng else f"a photo of {tax}.")
            display = eng or r["scientific_name"]
            seen[display] = seen.get(display, 0) + 1
            meta.append({"display_name": display, "english_name": eng, "scientific_name": r["scientific_name"],
                         "group": group, "order": r["order"], "family": r["family"],
                         "gbif_key": r["gbif_key"], "occurrences": r["occurrences"]})
        for m in meta:  # make display names unique within the list
            if seen[m["display_name"]] > 1 and m["english_name"]:
                m["display_name"] = f"{m['english_name']} ({m['scientific_name']})"
        emb = encode(model, tok, prompts, a.batch, path.stem)
        np.save(out / f"{path.stem}.npy", emb.astype(np.float16))
        with open(out / f"{path.stem}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(meta[0]))
            w.writeheader()
            w.writerows(meta)
        print(f"{path.stem}: {len(meta)} embeddings -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
