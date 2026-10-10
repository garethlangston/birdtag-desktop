#!/usr/bin/env python
"""Generate neutral placeholder artwork for the fork (CC0, no upstream artwork).

Upstream Project Kestrel's logos/icons are proprietary (see the upstream repo's
assets/LICENSE), so the fork ships plain generated placeholders: a rounded square
with the first letter of the product name from analyzer/branding.json.
Run: python tools/branding/make_placeholder_art.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BRAND = json.loads((ROOT / "analyzer" / "branding.json").read_text(encoding="utf-8"))
LETTER = BRAND["product_name"][:1].upper()
BG, FG = (38, 110, 96, 255), (255, 255, 255, 255)


def mark(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 5, fill=BG)
    font = ImageFont.load_default(size=int(size * 0.62))
    box = d.textbbox((0, 0), LETTER, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.text(((size - w) / 2 - box[0], (size - h) / 2 - box[1]), LETTER, font=font, fill=FG)
    return img


def main():
    big = mark(512)
    for p in (ROOT / "analyzer" / "logo.png", ROOT / "assets" / "logo.png"):
        big.resize((256, 256), Image.LANCZOS).save(p)
    for p in (ROOT / "analyzer" / "logo.ico", ROOT / "assets" / "logo.ico"):
        big.save(p, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    # Upstream service logo slot (UI hidden in the fork) -> transparent pixel-art placeholder.
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(ROOT / "analyzer" / "perch-logo.png")
    hero = Image.new("RGB", (800, 450), (32, 40, 52))
    hero.paste(mark(200).convert("RGB"), (300, 125), mark(200))
    hero.save(ROOT / "analyzer" / "whatsnew-bird.jpg", quality=85)
    (ROOT / "assets" / "logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect width="100" height="100" rx="20" fill="#266e60"/>'
        f'<text x="50" y="70" font-family="sans-serif" font-size="62" text-anchor="middle" fill="#fff">{LETTER}</text>'
        "</svg>\n", encoding="utf-8")
    # Installer wizard images (Inno Setup: BMP/PNG); 164x314 and 55x55.
    side = Image.new("RGB", (164, 314), (38, 110, 96))
    side.paste(mark(120).convert("RGB"), (22, 97), mark(120))
    side.save(ROOT / "assets" / "installer-side.png")
    mark(55).convert("RGB").save(ROOT / "assets" / "installer-small.png")
    print("placeholder art written")


if __name__ == "__main__":
    main()
