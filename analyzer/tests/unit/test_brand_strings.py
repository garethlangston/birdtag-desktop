"""Fork 6.3.0-uk.2: brand strings live at the source, species names stay intact,
and the Settings model-status path works with no extra models installed.

Regression for two bugs in 6.3.0-uk.1:
 1. js/brand.js rewrote every visible "Kestrel" -> "Birdtag", so "Common Kestrel"
    showed as "Common Birdtag".
 2. The same MutationObserver re-fired forever on text containing "Kestrel" that the
    regex did not change (the missing-model message lists ...\\ProjectKestrel\\models_extra),
    freezing the Settings screen when the extra models were missing.
"""

import ast
import re
import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

pytestmark = pytest.mark.unit
ANALYZER = Path(__file__).resolve().parents[2]
WORD = re.compile(r"\bKestrel\b")

# Lines that may keep the word: comments, upstream-service text, internal identifiers.
UI_ALLOW = [
    re.compile(r"^\s*(//|\*|/\*)"),        # JS comments
    re.compile(r"<!--|-->"),                 # HTML comments (single-line or end)
    re.compile(r"\[Kestrel\]"),              # console log prefix used when reading logs
]


def _html_comment_lines(text):
    """Line numbers inside multi-line <!-- ... --> comments."""
    inside, out = False, set()
    for i, line in enumerate(text.splitlines(), 1):
        if "<!--" in line and "-->" not in line.split("<!--", 1)[1]:
            inside = True
        if inside:
            out.add(i)
        if inside and "-->" in line:
            inside = False
    return out


def _js_block_comment_lines(text):
    inside, out = False, set()
    for i, line in enumerate(text.splitlines(), 1):
        if "/*" in line and "*/" not in line.split("/*", 1)[1]:
            inside = True
        if inside:
            out.add(i)
        if inside and "*/" in line:
            inside = False
    return out


def _ui_offenders():
    files = [ANALYZER / "visualizer.html", ANALYZER / "culling.html"]
    files += [p for p in sorted((ANALYZER / "js").glob("*.js")) if p.name != "brand.js"]
    bad = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        skip = _html_comment_lines(text) | _js_block_comment_lines(text)
        for i, line in enumerate(text.splitlines(), 1):
            if i in skip or not WORD.search(line):
                continue
            if any(p.search(line) for p in UI_ALLOW):
                continue
            bad.append(f"{f.name}:{i}: {line.strip()[:120]}")
    return bad


def test_no_upstream_brand_strings_left_in_ui():
    assert _ui_offenders() == []


# Python: string constants (not docstrings) shown to users.
PY_FILES = ["api_bridge.py", "settings_utils.py", "metadata_writer.py", "visualizer.py", "cli.py",
            "queue_manager.py", "kestrel_analyzer/pipeline.py", "oauth_client.py"]
PY_ALLOW = [
    re.compile(r"Kestrel\|"),                                  # Kestrel|Rating/Species/Family keywords
    re.compile(r"X-Kestrel"),                                  # upstream API headers
    re.compile(r"\[Kestrel\]"),                                # log prefixes
    re.compile(r"Project Kestrel's Terms of Service"),         # upstream service's own terms
    re.compile(r"original Project Kestrel North American"),    # attribution of the NA model
    re.compile(r"Kestrel Bridge Probe"),                       # internal CI probe page title
    re.compile(r"pick up where Kestrel left off|Picking up where Kestrel left off: "),  # cloud logs
]


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                ids.add(id(body[0].value))
    return ids


def test_no_upstream_brand_strings_in_python_messages():
    bad = []
    for rel in PY_FILES:
        src = (ANALYZER / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                v = node.value
                if WORD.search(v) and not any(p.search(v) for p in PY_ALLOW):
                    bad.append(f"{rel}:{node.lineno}: {v[:100]!r}")
    assert bad == []


def test_brand_js_does_not_rewrite_page_text():
    js = (ANALYZER / "js" / "brand.js").read_text(encoding="utf-8")
    code = "\n".join(l for l in js.splitlines() if not l.strip().startswith(("*", "/*", "//")))
    assert "MutationObserver" not in code
    assert "Kestrel" not in code  # no word replacement of any kind
    assert "var BRAND_NAME" in code and "var BRAND_SHORT" in code


# ---------------------------------------------------------------- species names survive

def test_kestrel_species_reach_the_ui_unchanged():
    import api_bridge

    api = api_bridge.Api()
    res = api.search_birds("kestrel", ["PAL", "NA", "AF", "OR"], 20)
    names = {r["canonical_common_name"] for r in res["results"]}
    assert {"Common Kestrel", "Lesser Kestrel", "American Kestrel"} <= names
    looked = api.lookup_birds(["Kestrel", "Common Kestrel"])["map"]
    assert looked["Common Kestrel"]["scientific_name"] == "Falco tinnunculus"


def test_kestrel_keywords():
    from kestrel_analyzer.bird_names import keyword_names
    from metadata_writer import species_keywords

    assert keyword_names("Common Kestrel") == ("Kestrel", "Falco tinnunculus")
    flat, hier = species_keywords("Common Kestrel", 0.95)
    assert hier == ["Birds|Kestrel", "Birds|Kestrel|Falco tinnunculus"]
    assert flat == ["Kestrel", "Falco tinnunculus"]
    assert species_keywords("Lesser Kestrel", 0.95)[1][-1] == "Birds|Lesser Kestrel|Falco naumanni"


# ---------------------------------------------------------------- Settings with no models

@pytest.fixture
def no_extra_models(tmp_path, monkeypatch):
    import settings_utils
    from kestrel_analyzer.ml import bird_species_inat21 as inat
    from kestrel_analyzer import wildlife_names

    monkeypatch.setattr(settings_utils, "_user_data_base", lambda: str(tmp_path))
    monkeypatch.delenv("KESTREL_MODELS_EXTRA_DIR", raising=False)
    monkeypatch.setattr(inat, "INAT21_DATA_DIR", tmp_path / "bundle" / "birds_inat21")
    import branding
    (tmp_path / branding.data_dir_name() / "models_extra").mkdir(parents=True)
    wildlife_names.clear_caches()
    yield tmp_path
    wildlife_names.clear_caches()


def test_settings_status_with_no_models_is_fast_and_explains_fallback(no_extra_models):
    import api_bridge
    import settings_utils

    api = api_bridge.Api()
    t0 = time.perf_counter()
    bird = api.get_bird_model_status("uk")
    wild = api.get_wildlife_model_status("bioclip", "uk")
    na = api.get_bird_model_status("na")
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert bird["success"] and bird["available"] is False
    assert "North America" in bird["message"] and "models_extra" in bird["message"]
    import branding
    assert str(no_extra_models / branding.data_dir_name() / "models_extra") in bird["message"]
    assert wild["success"] and wild["available"] is False and "SpeciesNet" in wild["message"]
    assert na["success"] and na["available"] is True
    # settings load/save still work with nothing installed
    settings_utils.save_persisted_settings({"bird_model_region": "uk", "other_wildlife_model": "bioclip"})
    assert settings_utils.load_persisted_settings()["bird_model_region"] == "uk"


def test_status_calls_never_raise(no_extra_models, monkeypatch):
    import api_bridge
    from kestrel_analyzer.ml import bird_species_inat21 as inat

    def boom(*a, **k):
        raise OSError("disk unplugged")

    monkeypatch.setattr(inat, "find_model", boom)
    res = api_bridge.Api().get_bird_model_status("uk")
    assert res["success"] is False and "disk unplugged" in res["error"]
