#!/usr/bin/env python
"""Export a BioCLIP image encoder (open_clip) to ONNX for the Kestrel UK fork.

Default model: hf-hub:imageomics/bioclip-2 (MIT licence, ViT-L/14, 224 px, 768-d).
BioCLIP 2.5 (hf-hub:imageomics/bioclip-2.5-vith14, ViT-H/14) exports the same way:
    python export_bioclip.py --model hf-hub:imageomics/bioclip-2.5-vith14 --name bioclip-2.5 ...
(needs ~8 GB RAM). Kestrel's runtime reads everything model-specific (input size,
mean/std, embedding dim, logit scale, prompt template) from the ``config.json`` this
script writes next to the ONNX files, so a new model folder can simply be dropped in.

Output folder ``<out>/<name>/``:
  image_encoder_fp32.onnx   input 'image' Nx3xSxS float32 RGB normalised -> 'embedding' NxD
                            (L2-normalised inside the graph)
  image_encoder_fp16.onnx   optional (--fp16): half-precision export, fp32 I/O. NOT shipped for
                            BioCLIP 2 yet: ONNX Runtime 1.23 CPU segfaults loading it, so parity
                            could not be checked (onnxconverter-common's post-hoc conversion
                            leaves open_clip's explicit fp32 casts mixed and is rejected).
  config.json               preprocessing + embedding metadata for the Kestrel runtime
  README.txt                source, licence, SHA256, parity results

Text embeddings for the species lists are built separately with
build_bioclip_embeddings.py (text encoder runs in PyTorch at build time only).

Stages run in separate subprocesses to keep peak memory low:
  reference -> export -> [fp16] -> verify
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

FP32 = "image_encoder_fp32.onnx"
FP16 = "image_encoder_fp16.onnx"
OPSET = 17
# Zero-shot labels used for the parity check (top-1 over these must not change).
PARITY_LABELS = [
    "a photo of a bird", "a photo of a frog", "a photo of a toad", "a photo of a newt",
    "a photo of a lizard", "a photo of a snake", "a photo of a slow worm", "a photo of a butterfly",
    "a photo of a dragonfly", "a photo of a damselfly", "a photo of a bumblebee",
    "a photo of a ladybird", "a photo of a heron", "a photo of a robin", "a photo of a telephone box",
    "a photo of a plant", "a photo of a landscape", "a photo of a person",
]
LICENCES = {
    "hf-hub:imageomics/bioclip-2": "MIT",
    "hf-hub:imageomics/bioclip-2.5-vith14": "MIT (check the model card)",
}


def _images(img_dirs):
    out = []
    for d in img_dirs:
        out += sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return out


def _load(model_id):
    import open_clip
    import torch

    torch.set_grad_enabled(False)
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    model.eval()
    return model, preprocess


def _preprocess_cfg(preprocess) -> dict:
    """Pull size / mean / std / interpolation out of the open_clip transform."""
    from torchvision import transforms as T

    cfg = {"input_size": None, "resize": None, "interpolation": "bicubic", "crop_pct": 1.0,
           "mean": None, "std": None}
    for t in preprocess.transforms:
        if isinstance(t, T.Resize):
            cfg["resize"] = t.size if isinstance(t.size, int) else list(t.size)
            cfg["interpolation"] = str(t.interpolation.value).lower()
        elif isinstance(t, T.CenterCrop):
            cfg["input_size"] = list(t.size)[0]
        elif isinstance(t, T.Normalize):
            cfg["mean"], cfg["std"] = list(map(float, t.mean)), list(map(float, t.std))
    return cfg


class _Encoder:
    """torch.nn.Module wrapper: image -> L2-normalised embedding."""

    def __new__(cls, model):
        import torch

        class Enc(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.visual = m.visual

            def forward(self, image):
                f = self.visual(image)
                return f / f.norm(dim=-1, keepdim=True)

        return Enc(model).eval()


def stage_reference(a):
    import open_clip
    import torch
    from PIL import Image

    model, preprocess = _load(a.model)
    tok = open_clip.get_tokenizer(a.model)
    files = _images(a.images)
    xs = np.stack([preprocess(Image.open(p).convert("RGB")).numpy() for p in files]).astype(np.float32)
    enc = _Encoder(model)
    img = torch.cat([enc(torch.from_numpy(x[None])) for x in xs]).numpy()
    txt = model.encode_text(tok(PARITY_LABELS))
    txt = (txt / txt.norm(dim=-1, keepdim=True)).numpy()
    np.savez(a.dir / "parity_reference.npz", inputs=xs, img=img, txt=txt,
             names=np.array([p.name for p in files]))
    cfg = _preprocess_cfg(preprocess)
    cfg.update({
        "model_id": a.model, "name": a.name, "embed_dim": int(img.shape[1]),
        "logit_scale": float(model.logit_scale.exp()),
        "prompt_template": "a photo of {taxonomy} with common name {common}.",
        "prompt_template_no_common": "a photo of {taxonomy}.",
        "licence": LICENCES.get(a.model, "see model card"),
        "onnx": {"fp32": FP32, "fp16": FP16, "input": "image", "output": "embedding"},
    })
    (a.dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print("config:", json.dumps(cfg))


def stage_export(a):
    import torch

    # nn.MultiheadAttention's inference fast path (aten::_native_multi_head_attention)
    # has no ONNX symbolic; the regular path is numerically the same and exports fine.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    model, _ = _load(a.model)
    enc = _Encoder(model)
    del model
    gc.collect()
    size = json.loads((a.dir / "config.json").read_text())["input_size"]
    dummy = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        torch.onnx.export(enc, (dummy,), str(a.dir / FP32), input_names=["image"],
                          output_names=["embedding"],
                          dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)
    print(f"exported {FP32} ({(a.dir / FP32).stat().st_size / 1e6:.0f} MB)")


def stage_fp16(a):
    """fp16 export straight from PyTorch (model.half()).

    Post-hoc onnxconverter-common conversion leaves open_clip's explicit fp32 casts
    (class token, LayerNorm) mixed with fp16 tensors, which ONNX Runtime rejects.
    Exporting a half-precision model types every cast consistently. I/O stay float32.
    """
    import torch

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    model, _ = _load(a.model)
    visual = model.visual.half().eval()
    del model
    gc.collect()

    class Enc16(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.visual = v

        def forward(self, image):
            f = self.visual(image.half()).float()
            return f / f.norm(dim=-1, keepdim=True)

    size = json.loads((a.dir / "config.json").read_text())["input_size"]
    with torch.no_grad():
        torch.onnx.export(Enc16(visual), (torch.zeros(1, 3, size, size),), str(a.dir / FP16),
                          input_names=["image"], output_names=["embedding"],
                          dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)
    print(f"fp16 saved ({(a.dir / FP16).stat().st_size / 1e6:.0f} MB)")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def stage_verify(a):
    import onnxruntime as ort

    ref = dict(np.load(a.dir / "parity_reference.npz"))
    cfg = json.loads((a.dir / "config.json").read_text())
    scale = cfg["logit_scale"]
    ref_top = (ref["img"] @ ref["txt"].T).argmax(1)
    results = {}
    for name in (FP32, FP16):
        p = a.dir / name
        if not p.exists():
            continue
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        emb = np.concatenate([s.run(None, {"image": x[None]})[0] for x in ref["inputs"]]).astype(np.float32)
        del s
        cos = (emb * ref["img"]).sum(1)
        top = (emb @ ref["txt"].T).argmax(1)
        lr = scale * ref["img"] @ ref["txt"].T
        lo = scale * emb @ ref["txt"].T
        pr = np.exp(lr - lr.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        po = np.exp(lo - lo.max(1, keepdims=True)); po /= po.sum(1, keepdims=True)
        r = {"top1_identical": bool((top == ref_top).all()), "min_cosine": float(cos.min()),
             "max_abs_prob_diff": float(np.abs(pr - po).max()),
             "per_image": [(str(n), PARITY_LABELS[t]) for n, t in zip(ref["names"], top)]}
        r["ok"] = r["top1_identical"] and r["max_abs_prob_diff"] <= 1e-2
        results[name] = r
        print(name, {k: v for k, v in r.items() if k != "per_image"})
        for row in r["per_image"]:
            print("   ", row)
    if FP16 in results and not results[FP16]["ok"]:
        print("fp16 failed parity -> removing it (ship fp32 only)")
        (a.dir / FP16).unlink()
    (a.dir / "parity_report.json").write_text(json.dumps(results, indent=2))
    lines = [
        f"BioCLIP image encoder for the Project Kestrel UK fork ({cfg['name']})", "",
        f"Source model : https://huggingface.co/{cfg['model_id'].replace('hf-hub:', '')}",
        f"Licence      : {cfg['licence']} (model weights). BioCLIP: Stevens et al. / Gu et al., Imageomics.",
        f"Exported     : {date.today().isoformat()} with tools/uk_models/export_bioclip.py (ONNX opset {OPSET},",
        f"               input 'image' Nx3x{cfg['input_size']}x{cfg['input_size']} float32 RGB, CLIP mean/std;",
        f"               output 'embedding' Nx{cfg['embed_dim']} L2-normalised)", "",
    ]
    for name in (FP32, FP16):
        p = a.dir / name
        if p.exists():
            r = results[name]
            lines += [name, f"  size   : {p.stat().st_size} bytes", f"  SHA256 : {_sha256(p)}",
                      f"  parity vs open_clip ({len(ref['names'])} images, {len(PARITY_LABELS)} zero-shot labels): "
                      f"top-1 identical={r['top1_identical']}, min cosine={r['min_cosine']:.5f}, "
                      f"max |prob diff|={r['max_abs_prob_diff']:.2e}", ""]
    lines += ["Install: copy this whole folder into <Kestrel user data>/models_extra/ (Windows:",
              "%LOCALAPPDATA%\\ProjectKestrel\\models_extra\\" + cfg["name"] + "\\) together with the",
              "embeddings/ folder produced by build_bioclip_embeddings.py."]
    (a.dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((a.dir / "README.txt").read_text())


STAGES = {"reference": stage_reference, "export": stage_export, "fp16": stage_fp16, "verify": stage_verify}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--name", default="bioclip-2")
    ap.add_argument("--images", nargs="*", type=Path, default=[])
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--fp16", action="store_true", help="also export fp16 (experimental, see docstring)")
    a = ap.parse_args()
    a.dir = a.out / a.name
    a.dir.mkdir(parents=True, exist_ok=True)
    if a.stage == "all":
        for st in ("reference", "export") + (("fp16",) if a.fp16 else ()) + ("verify",):
            print(f"=== {st}", flush=True)
            cmd = [sys.executable, __file__, "--out", str(a.out), "--model", a.model, "--name", a.name,
                   "--stage", st, "--images", *map(str, a.images)]
            if subprocess.call(cmd) != 0:
                return 1
        return 0
    STAGES[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
