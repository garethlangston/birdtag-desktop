#!/usr/bin/env python
"""Export timm EVA-02 Large iNat21 (10,000 classes, 1,486 birds) to ONNX for Kestrel.

Model: timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 (CC-BY-NC-4.0 weights,
non-commercial use only). The weights are NOT part of the Kestrel repo; this script
writes them to an output folder of your choice.

Stages (each runs in its own subprocess so peak memory stays ~3-4 GB):
  reference  timm logits for the test images  -> <out>/parity_reference.npz
  export     torch.onnx.export (legacy tracer, opset 17, dynamic batch) -> eva02_large_inat21_fp32.onnx
  fp16       onnxconverter-common float16 conversion (fp32 I/O kept)    -> eva02_large_inat21_fp16.onnx
  verify     ONNX Runtime (CPU) vs timm: top-1 + max |dprob|; writes README.txt with SHA256

Usage (Linux or Windows, any Python 3.10+ with: torch timm huggingface_hub onnx
onnxruntime onnxconverter-common pillow numpy):
  python tools/uk_models/export_eva02.py --out <folder> --images <dir with .jpg> [--stage all]

On Windows put the result in %LOCALAPPDATA%\ProjectKestrel\models_extra (see FORK-NOTES.md).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

MODEL_ID = "timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
FP32_NAME = "eva02_large_inat21_fp32.onnx"
FP16_NAME = "eva02_large_inat21_fp16.onnx"
OPSET = 17
PROB_TOL = 1e-2


def _images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _load_timm():
    import timm
    import torch

    torch.set_grad_enabled(False)
    model = timm.create_model("hf-hub:" + MODEL_ID, pretrained=True).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    return model, cfg


def stage_reference(out: Path, img_dir: Path) -> None:
    import timm
    import torch
    from PIL import Image

    model, cfg = _load_timm()
    print("timm data config:", cfg)
    tf = timm.data.create_transform(**cfg)
    files = _images(img_dir)
    xs = np.stack([tf(Image.open(p).convert("RGB")).numpy() for p in files]).astype(np.float32)
    with torch.no_grad():
        logits = torch.cat([model(torch.from_numpy(x[None])) for x in xs]).numpy()
    np.savez(out / "parity_reference.npz", inputs=xs, logits=logits,
             names=np.array([p.name for p in files]))
    (out / "data_config.json").write_text(json.dumps({k: v for k, v in cfg.items()}, indent=2, default=list))
    print(f"reference logits for {len(files)} images saved")


def stage_export(out: Path) -> None:
    import timm.layers
    import torch

    # timm's fused attention (F.scaled_dot_product_attention) does not trace with the
    # legacy exporter (is_causal arrives as a tensor). The explicit matmul/softmax path is
    # mathematically identical and exports to plain opset-17 ops.
    timm.layers.set_fused_attn(False)
    model, _ = _load_timm()
    dummy = torch.zeros(1, 3, 336, 336, dtype=torch.float32)
    target = out / FP32_NAME
    import inspect

    with torch.no_grad():
        kwargs = dict(input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=OPSET, do_constant_folding=True)
        has_dynamo_kw = "dynamo" in inspect.signature(torch.onnx.export).parameters
        if not has_dynamo_kw:  # torch < 2.5: legacy tracer is the only exporter
            torch.onnx.export(model, (dummy,), str(target), **kwargs)
        else:
            # Prefer the legacy TorchScript tracer (single file, DirectML-friendly, like
            # Kestrel's mdv1000-cedar export); fall back to the dynamo exporter if it fails.
            try:
                torch.onnx.export(model, (dummy,), str(target), dynamo=False, **kwargs)
            except Exception as e:  # noqa: BLE001 - report and try the other exporter
                print(f"legacy exporter failed ({type(e).__name__}: {e}); retrying with dynamo=True",
                      flush=True)
                gc.collect()
                # The dynamo exporter only implements opset >= 18; asking for 17 makes it
                # down-convert and emit an invalid Split node, so use 18 natively.
                kwargs["opset_version"] = max(OPSET, 18)
                torch.onnx.export(model, (dummy,), str(target), dynamo=True, external_data=False,
                                  **kwargs)
    del model
    gc.collect()
    print(f"exported {target} ({target.stat().st_size / 1e6:.0f} MB)")


def stage_fp16(out: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    m = onnx.load(str(out / FP32_NAME))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
    del m
    gc.collect()
    onnx.save(m16, str(out / FP16_NAME))
    print(f"fp16 saved ({(out / FP16_NAME).stat().st_size / 1e6:.0f} MB)")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _check(path: Path, ref) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    logits = np.concatenate([sess.run(None, {inp: x[None]})[0] for x in ref["inputs"]])
    del sess
    p_ref, p_onnx = _softmax(ref["logits"]), _softmax(logits.astype(np.float32))
    top_ref, top_onnx = ref["logits"].argmax(1), logits.argmax(1)
    res = {
        "top1_identical": bool((top_ref == top_onnx).all()),
        "max_abs_prob_diff": float(np.abs(p_ref - p_onnx).max()),
        "per_image": [(str(n), int(a), int(b), float(p_ref[i, a]), float(p_onnx[i, b]))
                      for i, (n, a, b) in enumerate(zip(ref["names"], top_ref, top_onnx))],
    }
    res["ok"] = res["top1_identical"] and res["max_abs_prob_diff"] <= PROB_TOL
    return res


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_verify(out: Path) -> None:
    ref = dict(np.load(out / "parity_reference.npz"))
    results = {}
    for name in (FP32_NAME, FP16_NAME):
        if (out / name).exists():
            results[name] = _check(out / name, ref)
            r = results[name]
            print(f"{name}: top1_identical={r['top1_identical']} max|dprob|={r['max_abs_prob_diff']:.2e} ok={r['ok']}")
            for row in r["per_image"]:
                print("   ", row)
            gc.collect()
    (out / "parity_report.json").write_text(json.dumps(results, indent=2))

    lines = [
        "EVA-02 Large iNat21 bird/species classifier for the Project Kestrel UK fork",
        "",
        f"Source model : https://huggingface.co/{MODEL_ID}",
        "Architecture : eva02_large_patch14_clip_336, fine-tuned on iNaturalist 2021 (10,000 classes)",
        "Licence      : CC-BY-NC-4.0 (Creative Commons Attribution-NonCommercial 4.0).",
        "               NON-COMMERCIAL USE ONLY. Do not redistribute these files as part of a",
        "               commercial product. Attribution: EVA-02 (Fang et al., BAAI) / timm (Ross Wightman),",
        "               iNaturalist 2021 dataset.",
        f"Exported     : {date.today().isoformat()} with tools/uk_models/export_eva02.py (ONNX opset {OPSET},",
        "               dynamic batch, input 'input' Nx3x336x336 float32 RGB, CLIP mean/std normalised,",
        "               output 'logits' Nx10000)",
        "",
    ]
    for name in (FP32_NAME, FP16_NAME):
        p = out / name
        if p.exists():
            r = results.get(name, {})
            lines.append(f"{name}")
            lines.append(f"  size   : {p.stat().st_size} bytes")
            lines.append(f"  SHA256 : {_sha256(p)}")
            lines.append(f"  parity vs timm (CPU, {len(ref['names'])} test images): top-1 identical="
                         f"{r.get('top1_identical')}, max |prob diff|={r.get('max_abs_prob_diff', float('nan')):.2e}")
            lines.append("")
    lines += [
        "Install: copy the .onnx file(s) into the Kestrel 'models_extra' folder",
        "  (Windows: %LOCALAPPDATA%\\ProjectKestrel\\models_extra) and pick UK / Europe / World under",
        "  Settings > Bird identification model. Kestrel prefers fp16 on GPU (DirectML) and fp32 on CPU.",
    ]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out / "README.txt").read_text())


STAGES = {"reference": stage_reference, "export": stage_export, "fp16": stage_fp16, "verify": stage_verify}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="output folder (outside the git repo)")
    ap.add_argument("--images", type=Path, help="folder of test .jpg images for the parity check")
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--skip-fp16", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.stage == "all":
        if not a.images:
            ap.error("--images is required for --stage all")
        stages = ["reference", "export"] + ([] if a.skip_fp16 else ["fp16"]) + ["verify"]
        for st in stages:
            cmd = [sys.executable, __file__, "--out", str(a.out), "--images", str(a.images), "--stage", st]
            print(f"=== stage {st}", flush=True)
            if subprocess.call(cmd) != 0:
                print(f"stage {st} failed", file=sys.stderr)
                return 1
        return 0
    if a.stage == "reference":
        if not a.images:
            ap.error("--images is required for the reference stage")
        stage_reference(a.out, a.images)
    else:
        STAGES[a.stage](a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
