"""Fork branding: one source of truth, own data folder, legacy migration (unit)."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import branding  # noqa: E402
import settings_utils  # noqa: E402

pytestmark = pytest.mark.unit
ANALYZER = Path(__file__).resolve().parents[2]
REPO = ANALYZER.parent


def test_branding_json_is_the_source():
    data = json.loads((ANALYZER / "branding.json").read_text(encoding="utf-8"))
    assert branding.PRODUCT_NAME == data["product_name"]
    assert branding.EXE_NAME == data["exe_name"]
    assert "Kestrel" not in data["product_name"] and "Kestrel" not in data["exe_name"]
    assert data["upstream_services"] is False and branding.UPSTREAM_SERVICES is False
    assert len(data["installer_app_guid"]) == 36


def test_packaging_reads_branding():
    spec = (ANALYZER / "ProjectKestrel.spec").read_text(encoding="utf-8")
    assert "branding.json" in spec and "name=_EXE_NAME" in spec
    iss = (REPO / "packaging" / "kestrel_installer.iss").read_text(encoding="utf-8")
    assert "Project Kestrel" not in iss.split("\n", 6)[-1]  # only in the header comment
    assert "org.ProjectKestrel" not in iss and "projectkestrel.org" not in iss
    wf = (REPO / ".github" / "workflows" / "fork-windows-build.yml").read_text(encoding="utf-8")
    assert "analyzer/branding.json" in wf and "secrets." not in wf


def test_ui_loads_brand_script_first():
    html = (ANALYZER / "visualizer.html").read_text(encoding="utf-8")
    assert html.index('src="js/brand.js"') < html.index('src="csv_parser.js"')
    assert '<title>Birdtag Desktop</title>' in html


@pytest.mark.parametrize("platform,name", [("win32", "BirdtagDesktop"), ("darwin", "BirdtagDesktop"),
                                           ("linux", "birdtag-desktop")])
def test_data_dir_name(platform, name):
    assert branding.data_dir_name(platform) == name


def test_user_data_dir_and_legacy_migration(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_utils, "_user_data_base", lambda: str(tmp_path))
    new_dir = Path(settings_utils._get_user_data_dir())
    assert new_dir.name == branding.data_dir_name()
    legacy = tmp_path / branding.legacy_data_dir_names()[0]
    legacy.mkdir()
    (legacy / "settings.json").write_text('{"bird_model_region": "europe"}')
    for other in branding.legacy_data_dir_names()[1:]:
        assert not (tmp_path / other).exists()
    assert settings_utils.migrate_legacy_user_data() == str(legacy)
    assert json.loads((new_dir / "settings.json").read_text())["bird_model_region"] == "europe"
    # second call: already migrated -> no-op
    assert settings_utils.migrate_legacy_user_data() is None


def test_legacy_models_extra_is_searched(tmp_path, monkeypatch):
    from kestrel_analyzer.ml import bird_species_inat21 as inat

    monkeypatch.setattr(settings_utils, "_user_data_base", lambda: str(tmp_path))
    dirs = [str(d) for d in inat.model_search_dirs()]
    assert any(d.endswith(f"{branding.legacy_data_dir_names()[0]}/models_extra") or
               d.endswith(f"{branding.legacy_data_dir_names()[0]}\\models_extra") for d in dirs)
