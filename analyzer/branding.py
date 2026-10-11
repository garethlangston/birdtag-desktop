"""Product branding for this fork, loaded from ``branding.json`` (single source of truth).

The upstream names "Project Kestrel", "Perch" and "Cloud Compute" and their artwork are
proprietary (see assets/LICENSE); this fork must ship under its own name. Everything
user-visible (window title, HTML titles via /branding.json, installer, exe name, user
data folder) reads from here, so renaming is a one-line edit of branding.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_DEFAULTS = {
    "product_name": "Birdtag Desktop",
    "short_name": "Birdtag",
    "exe_name": "BirdtagDesktop",
    "data_dir_name": "BirdtagDesktop",
    "data_dir_name_unix": "birdtag-desktop",
    "legacy_data_dir_names": ["ProjectKestrel", "project-kestrel"],
    "app_id": "io.github.garethlangston.birdtag-desktop",
    "publisher": "Birdtag Desktop contributors",
    "homepage": "",
    "upstream_name": "Project Kestrel",
    "upstream_url": "https://github.com/SanjaySoniLV/ProjectKestrel",
    "upstream_services": False,
}


def _candidates():
    here = Path(__file__).resolve().parent
    yield here / "branding.json"
    base = getattr(sys, "_MEIPASS", None)  # PyInstaller bundle
    if base:
        yield Path(base) / "branding.json"


def load() -> dict:
    data = dict(_DEFAULTS)
    for p in _candidates():
        try:
            data.update({k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                         if not k.startswith("_")})
            break
        except Exception:
            continue
    return data


BRAND = load()
PRODUCT_NAME: str = BRAND["product_name"]
EXE_NAME: str = BRAND["exe_name"]
SHORT_NAME: str = BRAND["short_name"]
UPSTREAM_SERVICES: bool = bool(BRAND.get("upstream_services", False))


def data_dir_name(platform: str | None = None) -> str:
    platform = platform or sys.platform
    if platform.startswith("win") or platform == "darwin":
        return BRAND["data_dir_name"]
    return BRAND["data_dir_name_unix"]


def legacy_data_dir_names() -> list[str]:
    return list(BRAND.get("legacy_data_dir_names") or [])


def user_agent() -> str:
    return f"{EXE_NAME}/1.0"


__all__ = ["BRAND", "PRODUCT_NAME", "SHORT_NAME", "EXE_NAME", "UPSTREAM_SERVICES", "data_dir_name",
           "legacy_data_dir_names", "user_agent"]
