"""UK fork: species keywords compatible with digiKam / Lightroom / birdtag.

Confident results  -> lr:hierarchicalSubject Birds|<English>|<Latin> (+ Birds|<English>),
                      digiKam:TagsList Birds/<English>/<Latin>, flat dc:subject English + Latin.
Below threshold    -> Birds|_Check|<English> only.
Existing sidecars / JPEGs are merged: keywords Kestrel does not own survive.
"""

from pathlib import Path
import sys

import json

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import metadata_writer as mw  # noqa: E402
from metadata_writer import bird_keywords, merge_keyword_lists, write_xmp_metadata  # noqa: E402

pytestmark = pytest.mark.unit

try:
    import pyexiv2  # noqa: F401
    HAVE_EXIV2 = True
except Exception:  # pragma: no cover
    HAVE_EXIV2 = False
needs_exiv2 = pytest.mark.skipif(not HAVE_EXIV2, reason="pyexiv2 not installed")


def _read(path):
    import pyexiv2
    img = pyexiv2.Image(str(path))
    try:
        return img.read_xmp()
    finally:
        img.close()


class TestBirdKeywords:
    def test_confident_uses_bou_english_and_latin(self):
        flat, hier = bird_keywords("European Robin", 0.93)
        assert flat == ["Robin", "Erithacus rubecula"]
        assert hier == ["Birds|Robin", "Birds|Robin|Erithacus rubecula"]

    def test_below_threshold_is_check_only(self):
        assert bird_keywords("Grey Heron", 0.5) == ([], ["Birds|_Check|Grey Heron"])

    def test_threshold_is_inclusive_and_configurable(self):
        assert bird_keywords("Grey Heron", 0.75)[0]
        assert bird_keywords("Grey Heron", 0.8, threshold=0.9)[1] == ["Birds|_Check|Grey Heron"]

    def test_missing_confidence_counts_as_confident(self):
        assert bird_keywords("Eurasian Wren", None)[1][-1] == "Birds|Wren|Troglodytes troglodytes"

    def test_non_birds_and_unknown_get_nothing(self):
        assert bird_keywords("deer", 0.99) == ([], [])
        assert bird_keywords("Unknown", 0.99) == ([], [])
        assert bird_keywords("", 0.99) == ([], [])


class TestMergeKeywordLists:
    def test_replaces_owned_keeps_foreign(self):
        existing = {
            "Xmp.dc.subject": ["Holiday", "Robin", "Erithacus rubecula", "Kestrel|Rating|3 Star"],
            "Xmp.lr.hierarchicalSubject": ["Places|Norfolk", "Birds|Robin", "Birds|Robin|Erithacus rubecula"],
            "Xmp.digiKam.TagsList": ["Places/Norfolk", "Birds/Robin/Erithacus rubecula"],
        }
        flat, hier = bird_keywords("Eurasian Wren", 0.9)
        out = merge_keyword_lists(existing, ["Kestrel|Rating|4 Star"], flat, hier)
        assert out["Xmp.dc.subject"] == ["Holiday", "Kestrel|Rating|4 Star", "Wren", "Troglodytes troglodytes"]
        assert out["Xmp.lr.hierarchicalSubject"] == [
            "Places|Norfolk", "Birds|Wren", "Birds|Wren|Troglodytes troglodytes"]
        assert out["Xmp.digiKam.TagsList"] == [
            "Places/Norfolk", "Birds/Wren", "Birds/Wren/Troglodytes troglodytes"]

    def test_same_species_is_idempotent(self):
        flat, hier = bird_keywords("European Robin", 0.9)
        first = merge_keyword_lists({}, [], flat, hier)
        assert merge_keyword_lists(first, [], flat, hier) == first

    def test_check_tag_replaced_when_confident(self):
        existing = {"Xmp.lr.hierarchicalSubject": ["Birds|_Check|Grey Heron"],
                    "Xmp.digiKam.TagsList": ["Birds/_Check/Grey Heron"]}
        flat, hier = bird_keywords("Grey Heron", 0.95)
        out = merge_keyword_lists(existing, [], flat, hier)
        assert "Birds|_Check|Grey Heron" not in out["Xmp.lr.hierarchicalSubject"]
        assert "Birds/Grey Heron/Ardea cinerea" in out["Xmp.digiKam.TagsList"]


class TestNewSidecar:
    def test_new_sidecar_has_all_three_keyword_forms(self, tmp_path):
        (tmp_path / "P1.ORF").touch()
        res = write_xmp_metadata(str(tmp_path), [{
            "filename": "P1.ORF", "rating": 4, "culled": "accept", "culled_origin": "manual",
            "species": "European Robin", "species_confidence": 0.97, "family": "Chats", "quality": 0.8}])
        assert res["success"] and res["written"] == 1
        text = (tmp_path / "P1.xmp").read_text(encoding="utf-8")
        assert "<rdf:li>Birds|Robin|Erithacus rubecula</rdf:li>" in text
        assert "<rdf:li>Birds/Robin/Erithacus rubecula</rdf:li>" in text
        assert "<rdf:li>Robin</rdf:li>" in text and "<rdf:li>Erithacus rubecula</rdf:li>" in text
        assert 'xmlns:digiKam="http://www.digikam.org/ns/1.0/"' in text
        # Kestrel's own fields still there
        assert 'kestrel:Species="European Robin"' in text and 'xmp:Rating="4"' in text

    def test_low_confidence_sidecar_check_only(self, tmp_path):
        (tmp_path / "P2.ORF").touch()
        write_xmp_metadata(str(tmp_path), [{
            "filename": "P2.ORF", "rating": 2, "culled": "accept",
            "species": "Grey Heron", "species_confidence": 0.4}])
        text = (tmp_path / "P2.xmp").read_text(encoding="utf-8")
        assert "Birds|_Check|Grey Heron" in text and "Birds/_Check/Grey Heron" in text
        assert "Birds|Grey Heron|" not in text and "<rdf:li>Ardea cinerea</rdf:li>" not in text

    @needs_exiv2
    def test_sidecar_parses_with_exiv2(self, tmp_path):
        (tmp_path / "P3.ORF").touch()
        write_xmp_metadata(str(tmp_path), [{"filename": "P3.ORF", "rating": 3, "culled": "accept",
                                            "species": "Eurasian Blue Tit", "species_confidence": 0.9}])
        x = _read(tmp_path / "P3.xmp")
        assert "Birds|Blue Tit|Cyanistes caeruleus" in x["Xmp.lr.hierarchicalSubject"]
        assert "Birds/Blue Tit/Cyanistes caeruleus" in x["Xmp.digiKam.TagsList"]
        assert {"Blue Tit", "Cyanistes caeruleus"} <= set(x["Xmp.dc.subject"])


USER_SIDECAR = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:lr="http://ns.adobe.com/lightroom/1.0/" xmlns:digiKam="http://www.digikam.org/ns/1.0/"
 xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" photoshop:City="Cley">
<dc:subject><rdf:Bag><rdf:li>Holiday</rdf:li><rdf:li>Robin</rdf:li><rdf:li>Erithacus rubecula</rdf:li></rdf:Bag></dc:subject>
<lr:hierarchicalSubject><rdf:Bag><rdf:li>Places|Norfolk</rdf:li><rdf:li>Birds|Robin</rdf:li><rdf:li>Birds|Robin|Erithacus rubecula</rdf:li></rdf:Bag></lr:hierarchicalSubject>
<digiKam:TagsList><rdf:Seq><rdf:li>Places/Norfolk</rdf:li><rdf:li>Birds/Robin/Erithacus rubecula</rdf:li></rdf:Seq></digiKam:TagsList>
</rdf:Description></rdf:RDF></x:xmpmeta>
<?xpacket end="w"?>
"""


@needs_exiv2
class TestMergeIntoExisting:
    def test_external_sidecar_still_needs_confirmation(self, tmp_path):
        (tmp_path / "P4.ORF").touch()
        (tmp_path / "P4.xmp").write_text(USER_SIDECAR, encoding="utf-8")
        res = write_xmp_metadata(str(tmp_path), [{"filename": "P4.ORF", "rating": 5, "culled": "accept",
                                                  "species": "Eurasian Wren", "species_confidence": 0.9}])
        assert res["skipped_conflicts"] == ["P4.xmp"]
        assert (tmp_path / "P4.xmp").read_text(encoding="utf-8") == USER_SIDECAR

    def test_confirmed_overwrite_merges_and_keeps_user_data(self, tmp_path):
        (tmp_path / "P5.ORF").touch()
        (tmp_path / "P5.xmp").write_text(USER_SIDECAR, encoding="utf-8")
        res = write_xmp_metadata(str(tmp_path), [{"filename": "P5.ORF", "rating": 5, "culled": "accept",
                                                  "culled_origin": "manual", "species": "Eurasian Wren",
                                                  "species_confidence": 0.9}], overwrite_external=True)
        assert res["written"] == 1, res
        x = _read(tmp_path / "P5.xmp")
        assert x["Xmp.photoshop.City"] == "Cley"
        assert x["Xmp.lr.hierarchicalSubject"] == [
            "Places|Norfolk", "Birds|Wren", "Birds|Wren|Troglodytes troglodytes"]
        assert x["Xmp.digiKam.TagsList"] == [
            "Places/Norfolk", "Birds/Wren", "Birds/Wren/Troglodytes troglodytes"]
        assert "Holiday" in x["Xmp.dc.subject"] and "Robin" not in x["Xmp.dc.subject"]
        assert x["Xmp.xmp.Rating"] == "5" and x["Xmp.xmp.Label"] == "Green"
        assert x["Xmp.kestrel.Species"] == "Eurasian Wren"

    def test_rewrite_of_own_sidecar_drops_stale_species(self, tmp_path):
        (tmp_path / "P6.ORF").touch()
        write_xmp_metadata(str(tmp_path), [{"filename": "P6.ORF", "rating": 3, "culled": "accept",
                                            "species": "European Robin", "species_confidence": 0.9}])
        write_xmp_metadata(str(tmp_path), [{"filename": "P6.ORF", "rating": 3, "culled": "accept",
                                            "species": "Unknown"}])
        x = _read(tmp_path / "P6.xmp")
        assert "Xmp.kestrel.Species" not in x
        assert "Xmp.lr.hierarchicalSubject" not in x
        assert "Robin" not in x.get("Xmp.dc.subject", [])

    def test_jpeg_embed_keeps_user_keywords(self, tmp_path):
        import pyexiv2
        from PIL import Image
        jpg = tmp_path / "IMG.jpg"
        Image.new("RGB", (64, 48), (120, 90, 60)).save(jpg, quality=90)
        img = pyexiv2.Image(str(jpg))
        img.modify_xmp({"Xmp.dc.subject": ["Holiday"],
                        "Xmp.lr.hierarchicalSubject": ["Places|Norfolk"]})
        img.close()
        res = write_xmp_metadata(str(tmp_path), [{"filename": "IMG.jpg", "rating": 4, "culled": "accept",
                                                  "species": "Common Blackbird", "species_confidence": 0.95}],
                                 embed_jpeg=True)
        assert res["embedded"] == 1, res
        x = _read(jpg)
        assert x["Xmp.lr.hierarchicalSubject"] == [
            "Places|Norfolk", "Birds|Blackbird", "Birds|Blackbird|Turdus merula"]
        assert "Holiday" in x["Xmp.dc.subject"] and "Blackbird" in x["Xmp.dc.subject"]
        assert x["Xmp.digiKam.TagsList"] == ["Birds/Blackbird", "Birds/Blackbird/Turdus merula"]


def test_default_threshold_matches_browse_filter():
    assert mw.DEFAULT_SPECIES_CONFIDENCE_THRESHOLD == 0.75


@needs_exiv2
class TestExiv2ArrayQuirks:
    """pyexiv2 splits array items on ', ' when reading and writes
    lr:hierarchicalSubject as a Seq; the writer must work around both."""

    def test_comma_in_family_survives_rewrites(self, tmp_path):
        (tmp_path / "P7.ORF").touch()
        entry = {"filename": "P7.ORF", "rating": 3, "culled": "accept", "species": "European Robin",
                 "species_confidence": 0.9, "family": "Chats, Old World Flycatchers"}
        for _ in range(3):
            write_xmp_metadata(str(tmp_path), [entry])
        items = mw._read_keyword_arrays((tmp_path / "P7.xmp").read_text(encoding="utf-8"))
        assert items["Xmp.dc.subject"] == [
            "Kestrel|Rating|3 Star", "Kestrel|Species|European Robin",
            "Kestrel|Family|Chats, Old World Flycatchers", "Robin", "Erithacus rubecula"]

    def test_hierarchical_subject_is_a_bag_after_merge(self, tmp_path):
        (tmp_path / "P8.ORF").touch()
        entry = {"filename": "P8.ORF", "rating": 3, "culled": "accept", "species": "Grey Heron",
                 "species_confidence": 0.9}
        write_xmp_metadata(str(tmp_path), [entry])
        write_xmp_metadata(str(tmp_path), [entry])  # second write goes through the exiv2 merge
        text = (tmp_path / "P8.xmp").read_text(encoding="utf-8")
        hier = text[text.index("<lr:hierarchicalSubject>"):text.index("</lr:hierarchicalSubject>")]
        assert "<rdf:Bag>" in hier and "<rdf:Seq>" not in hier
        assert "Birds|Grey Heron|Ardea cinerea" in hier


@needs_exiv2
class TestManualEditProtection:
    """birdtag rule: a root the user changed since Kestrel's last write is left alone."""

    def _write(self, root, species, conf=0.9, **kw):
        # overwrite_external=True = the user confirmed Kestrel's "sidecar was changed
        # outside Kestrel" dialog; manual-edit protection applies inside that write.
        kw.setdefault("overwrite_external", True)
        return write_xmp_metadata(str(root), [{"filename": "P9.ORF", "rating": 3, "culled": "accept",
                                               "species": species, "species_confidence": conf}], **kw)

    def test_edited_sidecar_still_goes_through_conflict_dialog(self, tmp_path):
        (tmp_path / "P9.ORF").touch()
        self._write(tmp_path, "European Robin")
        self._edit(tmp_path / "P9.xmp", ["Birds|Dunnock"])
        assert self._write(tmp_path, "Eurasian Wren", overwrite_external=False)["skipped_conflicts"] == ["P9.xmp"]

    def _edit(self, path, hier):
        import pyexiv2
        img = pyexiv2.Image(str(path))
        img.modify_xmp({"Xmp.lr.hierarchicalSubject": hier,
                        "Xmp.digiKam.TagsList": [h.replace("|", "/") for h in hier]})
        img.close()

    def test_user_edit_is_kept(self, tmp_path):
        (tmp_path / "P9.ORF").touch()
        self._write(tmp_path, "European Robin")
        state = json.loads((tmp_path / ".kestrel" / "kestrel_keyword_state.json").read_text())
        assert state["P9.xmp"]["Birds"] == ["Birds|Robin", "Birds|Robin|Erithacus rubecula"]
        # user corrects it to Dunnock in digiKam
        self._edit(tmp_path / "P9.xmp", ["Birds|Dunnock", "Birds|Dunnock|Prunella modularis"])
        res = self._write(tmp_path, "Eurasian Wren")
        assert res["kept_manual_edits"] == ["P9.xmp"]
        x = _read(tmp_path / "P9.xmp")
        assert x["Xmp.lr.hierarchicalSubject"] == ["Birds|Dunnock", "Birds|Dunnock|Prunella modularis"]
        assert "Wren" not in x.get("Xmp.dc.subject", [])
        assert x["Xmp.kestrel.Species"] == "Eurasian Wren"  # Kestrel's own field still updates
        # still protected on the next run (record keeps Kestrel's last write)
        assert self._write(tmp_path, "Eurasian Wren")["kept_manual_edits"] == ["P9.xmp"]

    def test_overwrite_option_forces(self, tmp_path):
        (tmp_path / "P9.ORF").touch()
        self._write(tmp_path, "European Robin")
        self._edit(tmp_path / "P9.xmp", ["Birds|Dunnock"])
        res = self._write(tmp_path, "Eurasian Wren", overwrite_manual_edits=True)
        assert res["kept_manual_edits"] == []
        x = _read(tmp_path / "P9.xmp")
        assert x["Xmp.lr.hierarchicalSubject"] == ["Birds|Wren", "Birds|Wren|Troglodytes troglodytes"]

    def test_unedited_root_is_updated_and_other_roots_unaffected(self, tmp_path):
        (tmp_path / "P9.ORF").touch()
        self._write(tmp_path, "European Robin")
        self._write(tmp_path, "Eurasian Wren")  # no user edit -> replaced
        x = _read(tmp_path / "P9.xmp")
        assert x["Xmp.lr.hierarchicalSubject"] == ["Birds|Wren", "Birds|Wren|Troglodytes troglodytes"]

    def test_manual_species_without_score_is_confident(self, tmp_path):
        (tmp_path / "P9.ORF").touch()
        self._write(tmp_path, "Grey Heron", conf=None)
        text = (tmp_path / "P9.xmp").read_text(encoding="utf-8")
        assert "Birds|Grey Heron|Ardea cinerea" in text and "_Check" not in text
