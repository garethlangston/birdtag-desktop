"""Unit tests for the UK fork's EVA-02 iNat21 bird classifier (no model weights needed).

Covers region masking (UK / Europe / World / rarities), the names it reports
(Kestrel catalogue names, BOU aliases), timm-identical preprocessing, the
square context crop, the classify() contract with a fake ONNX session, and
the pipeline's setting -> classifier resolution.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kestrel_analyzer import bird_names  # noqa: E402
from kestrel_analyzer.bird_catalog import BirdCatalog  # noqa: E402
from kestrel_analyzer.ml import bird_species_inat21 as inat  # noqa: E402

pytestmark = pytest.mark.unit

ROWS = bird_names.load_inat21_species()
BY_SCI = {r["inat_name"]: r for r in ROWS}
ROBIN = BY_SCI["Erithacus rubecula"]
GREY_HERON = BY_SCI["Ardea cinerea"]
GREAT_BLUE_HERON = BY_SCI["Ardea herodias"]


def _indices(region, rarities=False):
    return {r["index"] for r in inat.region_rows(region, rarities)}


class TestRegionMasks:
    def test_world_is_all_1486_bird_classes(self):
        assert len(ROWS) == 1486
        assert _indices("world") == {r["index"] for r in ROWS}
        assert min(_indices("world")) == 3111 and max(_indices("world")) == 4596

    def test_uk_default_matches_birdtag_rule(self):
        uk = _indices("uk")
        assert len(uk) == 258
        assert ROBIN["index"] in uk and GREY_HERON["index"] in uk
        # American species are not on the default UK list
        assert GREAT_BLUE_HERON["index"] not in uk

    def test_rarities_option_adds_bbrc_species(self):
        uk, uk_rare = _indices("uk"), _indices("uk", rarities=True)
        assert uk < uk_rare
        assert all(r["uk_rarity"] for r in ROWS if r["index"] in uk_rare - uk)

    def test_europe_contains_uk_and_more(self):
        eu = _indices("europe")
        assert _indices("uk") < eu < _indices("world")

    def test_unknown_region_rejected(self):
        with pytest.raises(ValueError):
            inat.region_rows("mars")

    def test_labels_file_has_10000_names_in_model_order(self):
        labels = (bird_names.INAT21_DATA_DIR / "labels.txt").read_text(encoding="utf-8").split("\n")
        labels = [l for l in labels if l]
        assert len(labels) == 10000
        assert labels[ROBIN["index"]] == "Erithacus rubecula"


class TestNames:
    def test_bou_names_for_uk_species(self):
        assert ROBIN["english_name"] == "Robin"
        assert GREY_HERON["english_name"] == "Grey Heron"
        assert BY_SCI["Troglodytes troglodytes"]["english_name"] == "Wren"

    def test_reported_names_resolve_in_kestrel_catalogue(self):
        cat = BirdCatalog()
        uk_names = [r["kestrel_name"] for r in inat.region_rows("uk", True)]
        missing = [n for n in uk_names if cat.lookup(n) is None]
        assert missing == []
        assert ROBIN["kestrel_name"] == "European Robin"

    def test_kestrel_names_unique(self):
        names = [r["kestrel_name"] for r in ROWS]
        assert len(names) == len(set(names))

    def test_bou_alias_lookup_and_search(self):
        cat = BirdCatalog()
        assert cat.lookup("Robin").canonical_common_name == "European Robin"
        assert cat.lookup("Wren").canonical_common_name == "Eurasian Wren"
        # an alias never shadows a canonical name
        assert cat.lookup("American Robin").canonical_common_name == "American Robin"
        assert cat.search("Robin", ["PAL"], limit=1)[0].canonical_common_name == "European Robin"

    def test_keyword_names(self):
        assert bird_names.keyword_names("European Robin") == ("Robin", "Erithacus rubecula")
        assert bird_names.keyword_names("Robin") == ("Robin", "Erithacus rubecula")
        assert bird_names.keyword_names("Grey Heron") == ("Grey Heron", "Ardea cinerea")
        # NA-model names fall back to the IOC catalogue
        assert bird_names.keyword_names("American Robin") == ("American Robin", "Turdus migratorius")
        assert bird_names.keyword_names("deer") is None
        assert bird_names.keyword_names("") is None


class TestPreprocess:
    def test_shape_dtype_and_normalisation(self):
        img = np.full((336, 336, 3), 255, dtype=np.uint8)
        x = inat.preprocess(img)
        assert x.shape == (1, 3, 336, 336) and x.dtype == np.float32
        np.testing.assert_allclose(x[0, :, 0, 0], (1.0 - inat.MEAN) / inat.STD, rtol=1e-6)

    def test_channel_order_is_rgb(self):
        img = np.zeros((336, 336, 3), dtype=np.uint8)
        img[..., 0] = 255  # pure red in RGB
        x = inat.preprocess(img)
        assert x[0, 0].mean() > 0 > x[0, 2].mean()

    def test_shorter_side_resized_then_center_cropped(self):
        # 672x1344 (HxW): left/right quarters are black, centre half white.
        img = np.zeros((672, 1344, 3), dtype=np.uint8)
        img[:, 336:1008] = 255
        x = inat.preprocess(img)
        white = (1.0 - inat.MEAN[0]) / inat.STD[0]
        # after resize to 336x672 and centre crop of 336, everything but a
        # couple of bicubic edge pixels is white
        assert np.isclose(x[0, 0, :, 5:-5], white, atol=1e-4).all()

    def test_matches_torchvision_style_reference(self):
        from PIL import Image

        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(400, 500, 3), dtype=np.uint8)
        ref = Image.fromarray(img).resize((int(336 * 500 / 400), 336), Image.BICUBIC)
        left = int(round((ref.width - 336) / 2.0))
        ref = np.asarray(ref.crop((left, 0, left + 336, 336)), dtype=np.float32) / 255.0
        ref = ((ref - inat.MEAN) / inat.STD).transpose(2, 0, 1)
        np.testing.assert_allclose(inat.preprocess(img)[0], ref, atol=1e-6)


class TestSquareContextCrop:
    def test_square_and_padded(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        crop = inat.square_context_crop(((900, 400), (1000, 600)), img)
        assert crop.shape[0] == crop.shape[1] == int(200 * 1.3)

    def test_clamped_to_image(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        crop = inat.square_context_crop(((0, 0), (390, 290)), img)
        assert crop.shape[:2] == (300, 300)


class _FakeSession:
    def __init__(self, kind, path, coord, logits):
        self._logits = logits

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        class _I:
            name = "input"
        return [_I()]

    def run(self, _names, feed):
        assert feed["input"].shape == (1, 3, 336, 336)
        return [self._logits[None]]


@pytest.fixture
def make_clf(monkeypatch):
    from kestrel_analyzer.ml import resilient_session

    def _make(logits, region="uk", rarities=False):
        monkeypatch.setattr(resilient_session, "ResilientOnnxSession",
                            lambda kind, path, coord: _FakeSession(kind, path, coord, logits))
        return inat.Inat21BirdSpeciesClassifier("fake.onnx", coord=None, region=region,
                                                include_rarities=rarities)
    return _make


class TestClassify:
    def test_contract_and_region_softmax(self, make_clf):
        logits = np.zeros(10000, dtype=np.float32)
        logits[GREAT_BLUE_HERON["index"]] = 20.0  # out of region in UK mode
        logits[GREY_HERON["index"]] = 10.0
        logits[ROBIN["index"]] = 8.0
        res = make_clf(logits).classify(np.zeros((50, 80, 3), dtype=np.uint8), top_k=5)
        assert set(res) == {"top_species_labels", "top_species_scores",
                            "top_family_labels", "top_family_scores"}
        assert list(res["top_species_labels"][:2]) == ["Grey Heron", "European Robin"]
        assert len(res["top_species_labels"]) == 5
        assert 0.8 < res["top_species_scores"][0] <= 1.0
        assert res["top_species_scores"][0] >= res["top_species_scores"][1]
        assert res["top_family_labels"][0] == GREY_HERON["family_common"]

    def test_world_mode_sees_all_birds(self, make_clf):
        logits = np.zeros(10000, dtype=np.float32)
        logits[GREAT_BLUE_HERON["index"]] = 20.0
        res = make_clf(logits, region="world").classify(np.zeros((40, 40, 3), dtype=np.uint8))
        assert res["top_species_labels"][0] == GREAT_BLUE_HERON["kestrel_name"]

    def test_wants_square_context(self, make_clf):
        assert make_clf(np.zeros(10000, dtype=np.float32)).wants_square_context is True


class TestModelDiscovery:
    def test_missing_model_message_lists_folders(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KESTREL_MODELS_EXTRA_DIR", str(tmp_path))
        msg = inat.missing_model_message()
        assert str(tmp_path) in msg and "models_extra" in msg

    def test_find_model_prefers_requested_precision(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KESTREL_MODELS_EXTRA_DIR", str(tmp_path))
        monkeypatch.setattr(inat, "user_models_extra_dir", lambda: tmp_path / "none")
        monkeypatch.setattr(inat, "INAT21_DATA_DIR", tmp_path / "none2")
        assert inat.find_model() is None
        (tmp_path / inat.MODEL_FILENAMES["fp32"]).write_bytes(b"x")
        (tmp_path / inat.MODEL_FILENAMES["fp16"]).write_bytes(b"x")
        assert inat.find_model(prefer_fp16=True).name == inat.MODEL_FILENAMES["fp16"]
        assert inat.find_model(prefer_fp16=False).name == inat.MODEL_FILENAMES["fp32"]


class TestPipelineSelection:
    def _pipeline(self, monkeypatch, settings):
        from kestrel_analyzer import pipeline as pl

        monkeypatch.setattr(pl, "load_persisted_settings", lambda: dict(settings))
        return pl.AnalysisPipeline(use_gpu=False)

    def test_default_is_uk(self, monkeypatch, tmp_path):
        monkeypatch.setattr(inat, "find_model", lambda prefer_fp16=False: None)
        p = self._pipeline(monkeypatch, {})
        # UK by default; with no EVA-02 file the builder falls back to NA (tested below)
        assert p._resolve_species_model() == ("uk", False, None)

    def test_explicit_north_america(self, monkeypatch):
        p = self._pipeline(monkeypatch, {"bird_model_region": "na"})
        assert p._resolve_species_model() == ("na", False, None)

    def test_uk_with_rarities(self, monkeypatch, tmp_path):
        monkeypatch.setattr(inat, "find_model", lambda prefer_fp16=False: tmp_path / "m.onnx")
        p = self._pipeline(monkeypatch, {"bird_model_region": "uk", "bird_include_uk_rarities": True})
        assert p._resolve_species_model() == ("uk", True, str(tmp_path / "m.onnx"))

    def test_missing_model_falls_back_to_na_with_message(self, monkeypatch):
        from kestrel_analyzer import pipeline as pl

        built = []
        monkeypatch.setattr(pl, "BirdSpeciesClassifier", lambda *a, **k: built.append(a) or "NA-CLF")
        p = self._pipeline(monkeypatch, {"bird_model_region": "uk"})
        msgs = []
        clf = p._build_species_classifier(("uk", False, None), coord=None, status_cb=msgs.append)
        assert clf == "NA-CLF" and built
        assert msgs and "not found" in msgs[0] and "models_extra" in msgs[0]
