#!/usr/bin/env python
"""Run Kestrel's original NA classifier and the EVA-02 iNat21 classifier on images (CPU).

Classifies the whole image (no detector), so it measures the species models only.
Usage: python tools/uk_models/compare_classifiers.py <image dir> [--regions uk europe world]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2] / "analyzer"
sys.path.insert(0, str(ANALYZER))

from kestrel_analyzer.config import MODELS_DIR, SPECIESCLASSIFIER_LABELS, SPECIESCLASSIFIER_PATH  # noqa: E402
from kestrel_analyzer.image_utils import read_image  # noqa: E402
from kestrel_analyzer.ml.bird_species import BirdSpeciesClassifier  # noqa: E402
from kestrel_analyzer.ml.bird_species_inat21 import Inat21BirdSpeciesClassifier, find_model  # noqa: E402
from kestrel_analyzer.ml.provider_coordinator import ProviderCoordinator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("--regions", nargs="+", default=["uk"])
    ap.add_argument("--fp16", action="store_true", help="prefer the fp16 model file")
    a = ap.parse_args()
    files = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs = {p.name: read_image(str(p)) for p in files}
    coord = ProviderCoordinator(user_gpu_enabled=False)

    results: dict[str, dict[str, str]] = {n: {} for n in imgs}
    na = BirdSpeciesClassifier(str(SPECIESCLASSIFIER_PATH), str(SPECIESCLASSIFIER_LABELS), coord,
                               models_dir=str(MODELS_DIR))
    for n, img in imgs.items():
        r = na.classify(img)
        results[n]["NA"] = f"{r['top_species_labels'][0]} ({r['top_species_scores'][0]:.2f})"
    del na

    model = find_model(prefer_fp16=a.fp16)
    if model is None:
        print("EVA-02 model not found")
        return 1
    print(f"model: {model}")
    for region in a.regions:
        clf = Inat21BirdSpeciesClassifier(model, coord, region=region)
        t0 = time.perf_counter()
        for n, img in imgs.items():
            r = clf.classify(img)
            results[n][region.upper()] = f"{r['top_species_labels'][0]} ({r['top_species_scores'][0]:.2f})"
        print(f"{region}: {len(clf.labels)} classes, {(time.perf_counter() - t0) / len(imgs):.2f} s/image")
        del clf

    cols = ["NA"] + [r.upper() for r in a.regions]
    print("| image | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for n, row in results.items():
        print(f"| {n} | " + " | ".join(row.get(c, "") for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
