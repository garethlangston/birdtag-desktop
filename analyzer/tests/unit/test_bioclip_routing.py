"""UK fork phase 2: BioCLIP routing, lists, threshold and keyword roots (no real model needed).

A tiny fake BioCLIP folder (4-d embeddings, dummy ONNX file, fake session) exercises
the same code paths as the real 768-d model.
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kestrel_analyzer import wildlife_names  # noqa: E402
from kestrel_analyzer.ml import bioclip, resilient_session  # noqa: E402
from kestrel_analyzer.ml import bird_species_inat21 as inat  # noqa: E402

pytestmark = pytest.mark.unit

SPECIES = {
    "reptiles": [("Common Lizard", "Zootoca vivipara", "Lacertidae", [1, 0, 0, 0])],
    "amphibians": [("Common Frog", "Rana temporaria", "Ranidae", [0, 1, 0, 0])],
    "insects": [("7-spot Ladybird", "Coccinella septempunctata", "Coccinellidae", [0, 0, 1, 0]),
                ("Peacock", "Aglais io", "Nymphalidae", [0, 0, 0.8, 0.6])],
}


@pytest.fixture
def fake_bioclip(tmp_path, monkeypatch):
    d = tmp_path / "models_extra" / "bioclip-2"
    (d / "embeddings").mkdir(parents=True)
    (d / "image_encoder_fp32.onnx").write_bytes(b"fake")
    (d / "config.json").write_text(json.dumps({
        "input_size": 8, "resize": 8, "interpolation": "bicubic", "mean": [0.5] * 3, "std": [0.5] * 3,
        "embed_dim": 4, "logit_scale": 100.0, "name": "bioclip-2",
        "onnx": {"fp32": "image_encoder_fp32.onnx", "fp16": "image_encoder_fp16.onnx"}}))
    for g, rows in SPECIES.items():
        emb = np.array([r[3] for r in rows], np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        np.save(d / "embeddings" / f"{g}_uk.npy", emb.astype(np.float16))
        with open(d / "embeddings" / f"{g}_uk.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["display_name", "english_name", "scientific_name", "group", "order", "family",
                        "gbif_key", "occurrences"])
            for eng, sci, fam, _ in rows:
                w.writerow([eng, eng, sci, g, "", fam, 1, 10])
    np.save(d / "embeddings" / "negatives.npy", np.array([[0, 0, 0, 1]], np.float16))
    (d / "embeddings" / "negatives.csv").write_text("prompt\na photo of a plant.\n")
    monkeypatch.setenv("KESTREL_MODELS_EXTRA_DIR", str(d.parent))
    monkeypatch.setattr(inat, "user_models_extra_dir", lambda: tmp_path / "nothing")
    wildlife_names.clear_caches()
    yield d
    wildlife_names.clear_caches()


class _FakeSession:
    next_embedding = np.array([1, 0, 0, 0], np.float32)

    def __init__(self, *a, **k):
        pass

    def get_inputs(self):
        class _I:
            name = "image"
        return [_I()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _o, feed):
        assert feed["image"].shape == (1, 3, 8, 8)
        return [_FakeSession.next_embedding[None]]


@pytest.fixture
def clf(fake_bioclip, monkeypatch):
    monkeypatch.setattr(resilient_session, "ResilientOnnxSession", _FakeSession)
    return bioclip.BioclipClassifier(fake_bioclip, coord=None, region="uk")


class TestTaxonomyRouting:
    @pytest.mark.parametrize("raw,group", [
        ("x;reptilia;squamata;lacertidae;zootoca;vivipara;common lizard", "reptiles"),
        ("x;reptilia;testudines;;;;turtle order", "reptiles"),
        ("x;amphibia;anura;ranidae;rana;temporaria;common frog", "amphibians"),
        ("x;amphibia;caudata;;;;salamander order", "amphibians"),
        ("x;insecta;lepidoptera;;;;butterfly", "insects"),
        ("x;insecta;odonata;;;;dragonfly", "insects"),
        ("x;mammalia;carnivora;canidae;vulpes;vulpes;red fox", None),
        ("x;aves;;;;;bird", None),
        ("1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal", "any"),
    ])
    def test_group_for_taxonomy(self, raw, group):
        assert bioclip.group_for_taxonomy(raw) == group

    def test_taxon_label_is_a_plain_string(self):
        lab = bioclip.TaxonLabel("Common Frog", "amphibians", "x;amphibia")
        assert lab == "Common Frog" and lab != "bird" and json.dumps([lab]) == '["Common Frog"]'
        assert lab.bioclip_group == "amphibians"

    def test_wrapper_route_hook(self):
        from kestrel_analyzer.ml.speciesnet_sam_hq import SpeciesNetSAMHQWrapper

        w = SpeciesNetSAMHQWrapper.__new__(SpeciesNetSAMHQWrapper)
        w.bioclip_routing = True
        r, lab = w._bioclip_route("ignore", None, "x;amphibia;anura;ranidae;rana;temporaria;common frog")
        assert r == "wildlife" and lab == "Common Frog" and lab.bioclip_group == "amphibians"
        # mammals and birds keep their route
        assert w._bioclip_route("ignore", None, "x;mammalia;carnivora;canidae;vulpes;vulpes;red fox") == ("ignore", None)
        assert w._bioclip_route("bird", "bird", "x;aves;;;;;bird") == ("bird", "bird")
        # generic animal -> BioCLIP over all groups; blank stays ignored
        r, lab = w._bioclip_route("wildlife", "Animal", "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal")
        assert lab.bioclip_group == "any"
        assert w._bioclip_route("ignore", None, "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank") == ("ignore", None)
        w.bioclip_routing = False
        assert w._bioclip_route("ignore", None, "x;amphibia;anura;;;;frog") == ("ignore", None)


class TestBioclipClassifier:
    def test_find_model_dir_and_lists(self, fake_bioclip, clf):
        assert bioclip.find_model_dir() == fake_bioclip
        assert sorted(clf.groups) == ["amphibians", "insects", "reptiles"]

    def test_group_restricted_classification(self, clf):
        _FakeSession.next_embedding = np.array([0, 0, 1, 0], np.float32)
        res = clf.classify(np.zeros((20, 30, 3), np.uint8), groups=["insects"])
        assert res["top_species_labels"][0] == "7-spot Ladybird"
        assert res["group"] == "insects" and res["scientific_name"] == "Coccinella septempunctata"
        assert res["top_family_labels"][0] == "Coccinellidae"
        assert 0.5 < res["top_species_scores"][0] <= 1.0

    def test_negatives_beat_background(self, clf):
        _FakeSession.next_embedding = np.array([0, 0, 0, 1], np.float32)  # "plant"
        res = clf.classify(np.zeros((20, 30, 3), np.uint8), use_negatives=True)
        assert not res["beats_negatives"]
        _FakeSession.next_embedding = np.array([0, 1, 0, 0], np.float32)
        res = clf.classify(np.zeros((20, 30, 3), np.uint8), use_negatives=True)
        assert res["beats_negatives"] and res["top_species_labels"][0] == "Common Frog"

    def test_group_chosen_by_total_probability(self, clf):
        # The lizard is the single closest species, but the two insects are almost as close
        # and together hold more probability -> the insect group (and its best species) wins.
        clf.scale = 10.0
        e = np.array([0.61, 0, 0.6, 0.2], np.float32)  # cos: lizard .694, both insects .683
        e /= np.linalg.norm(e)
        full = clf.classify_embedding(e, groups=["reptiles"])  # sanity: lizard alone
        assert full["top_species_labels"][0] == "Common Lizard"
        res = clf.classify_embedding(e)
        assert res["group"] == "insects"
        assert res["group_scores"]["insects"] > res["group_scores"]["reptiles"]

    def test_preprocess_uses_config(self, fake_bioclip):
        cfg = json.loads((fake_bioclip / "config.json").read_text())
        x = bioclip._preprocess(np.full((16, 32, 3), 255, np.uint8), cfg)
        assert x.shape == (1, 3, 8, 8) and np.allclose(x, 1.0)

    def test_missing_message(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KESTREL_MODELS_EXTRA_DIR", str(tmp_path))
        monkeypatch.setattr(inat, "user_models_extra_dir", lambda: tmp_path / "x")
        assert bioclip.find_model_dir() is None
        assert "bioclip-2" in bioclip.missing_message()


class TestPipelineBioclip:
    def _pipeline(self, monkeypatch, settings):
        from kestrel_analyzer import pipeline as pl

        monkeypatch.setattr(pl, "load_persisted_settings", lambda: dict(settings))
        return pl.AnalysisPipeline(use_gpu=False)

    def test_defaults_bioclip_on_uk_and_threshold(self, monkeypatch, fake_bioclip):
        p = self._pipeline(monkeypatch, {})
        assert p._resolve_bioclip() == (True, str(fake_bioclip), "uk")
        assert p._species_threshold == 0.75

    def test_speciesnet_only_and_na_region(self, monkeypatch, fake_bioclip):
        assert self._pipeline(monkeypatch, {"other_wildlife_model": "speciesnet"})._resolve_bioclip()[0] is False
        p = self._pipeline(monkeypatch, {"bird_model_region": "na", "species_confidence_threshold": 0.6})
        assert p._resolve_bioclip()[2] == "world" and p._species_threshold == 0.6

    def test_detection_and_whole_frame_routes(self, monkeypatch, clf):
        p = self._pipeline(monkeypatch, {})
        p.bioclip_clf = clf
        _FakeSession.next_embedding = np.array([1, 0, 0, 0], np.float32)
        lab = bioclip.TaxonLabel("Lizard", "reptiles", "x;reptilia;squamata;;;;lizard")
        sp, sc, fam, _, route = p._bioclip_for_detection(lab, np.zeros((9, 9, 3), np.uint8), "Lizard", 0.4)
        assert sp == "Common Lizard" and sc > 0.9 and fam == "Lacertidae" and route.startswith("detection")
        # SpeciesNet says "reptile" but it's an amphibian: herp detections use both herp lists
        _FakeSession.next_embedding = np.array([0, 1, 0, 0], np.float32)
        assert p._bioclip_for_detection(lab, np.zeros((9, 9, 3), np.uint8), "Lizard", 0.4)[0] == "Common Frog"
        # but an insect detection never becomes a herp
        ins = bioclip.TaxonLabel("Butterfly", "insects", "x;insecta;lepidoptera;;;;butterfly")
        assert p._bioclip_for_detection(ins, np.zeros((9, 9, 3), np.uint8), "Butterfly", 0.4)[0] in (
            "7-spot Ladybird", "Peacock")
        # plain string label (mammal) is left to SpeciesNet
        assert p._bioclip_for_detection("Red Fox", None, "Red Fox", 0.8)[0] == "Red Fox"
        # whole frame: background wins -> nothing
        _FakeSession.next_embedding = np.array([0, 0, 0, 1], np.float32)
        assert p._bioclip_whole_frame(np.zeros((90, 90, 3), np.uint8)) is None
        _FakeSession.next_embedding = np.array([0, 0, 1, 0], np.float32)
        assert p._bioclip_whole_frame(np.zeros((90, 90, 3), np.uint8))["top_species_labels"][0] == "7-spot Ladybird"
        p._species_threshold = 1.5  # threshold applies to the whole-frame route (fake scores hit 1.0)
        assert p._bioclip_whole_frame(np.zeros((90, 90, 3), np.uint8)) is None


class TestKeywordRoots:
    def test_wildlife_roots(self, fake_bioclip):
        from metadata_writer import species_keywords

        assert species_keywords("Common Lizard", 0.9) == (
            ["Common Lizard", "Zootoca vivipara"],
            ["Reptiles|Common Lizard", "Reptiles|Common Lizard|Zootoca vivipara"])
        assert species_keywords("Common Frog", 0.5)[1] == ["Amphibians|_Check|Common Frog"]
        assert species_keywords("7-spot Ladybird", 0.8)[1][-1] == \
            "Insects|7-spot Ladybird|Coccinella septempunctata"
        assert species_keywords("Animal", 0.99) == ([], [])

    def test_mammals_from_speciesnet_labels(self):
        from metadata_writer import species_keywords

        flat, hier = species_keywords("Red Fox", 0.9)
        assert hier == ["Mammals|Red Fox", "Mammals|Red Fox|Vulpes vulpes"]

    def test_picker_search_and_lookup(self, fake_bioclip):
        res = wildlife_names.search_wildlife("lady", "uk")
        assert res and res[0]["canonical_common_name"] == "7-spot Ladybird"
        assert res[0]["family_common"] == "Insects"
        assert wildlife_names.lookup_wildlife("common frog", "uk")["scientific_name"] == "Rana temporaria"
        assert wildlife_names.search_wildlife("x", "uk") == []
