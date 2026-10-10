#!/usr/bin/env python
"""Download CC-licensed UK reptile / amphibian / insect test photos from Wikimedia Commons.

Writes <out>/<slug>.jpg (long edge <= --max-px) and <out>/SOURCES.txt
(file page, author, licence, URL). Only CC0 / PD / CC-BY / CC-BY-SA files are used.
Usage: python fetch_test_photos.py <out dir> [--max-px 1280]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "KestrelUKFork-testdata/0.1 (giggaloop@romeos.cc)"
API = "https://commons.wikimedia.org/w/api.php"
# slug, search query, (optional) exact File: title preferred
SUBJECTS = [
    ("common_lizard", "Zootoca vivipara", None),
    ("slow_worm", "Anguis fragilis", None),
    ("grass_snake", "Natrix helvetica", None),
    ("common_frog", "Rana temporaria", None),
    ("common_toad", "Bufo bufo", None),
    ("smooth_newt", "Lissotriton vulgaris", None),
    ("peacock_butterfly", "Aglais io", None),
    ("red_admiral", "Vanessa atalanta", None),
    ("emperor_dragonfly", "Anax imperator", None),
    ("banded_demoiselle", "Calopteryx splendens", None),
    ("buff_tailed_bumblebee", "Bombus terrestris", None),
    ("seven_spot_ladybird_macro", "Coccinella septempunctata macro", None),
]
OK_LICENCE = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [0-9.]+)", re.I)


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _strip(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def find(query: str, width: int):
    data = _get({"action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                 "gsrnamespace": 6, "gsrlimit": 15, "prop": "imageinfo",
                 "iiprop": "url|extmetadata|size|mime", "iiurlwidth": width})
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = _strip(meta.get("LicenseShortName", {}).get("value", ""))
        if ii.get("mime") != "image/jpeg" or not OK_LICENCE.match(lic):
            continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 600:
            continue
        return {"title": p["title"], "licence": lic, "author": _strip(meta.get("Artist", {}).get("value", ""))[:80],
                "page": ii.get("descriptionurl"), "url": ii.get("thumburl") or ii.get("url")}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-px", type=int, default=1280)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, query, _ in SUBJECTS:
        hit = find(query, a.max_px)
        if not hit:
            print(f"!! nothing suitable for {query}")
            continue
        req = urllib.request.Request(hit["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if data[:2] != b"\xff\xd8":
            print(f"!! {slug}: not a JPEG")
            continue
        (a.out / f"{slug}.jpg").write_bytes(data)
        rows.append(f"{slug} | {hit['title']} | {hit['licence']} | {hit['author']} | {hit['page']}")
        print(rows[-1])
        time.sleep(1.0)
    (a.out / "SOURCES.txt").write_text(
        "slug | Commons file | licence | author | URL\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
