# Birdtag Desktop

**Birdtag Desktop is an unofficial fork of [Project Kestrel](https://github.com/SanjaySoniLV/ProjectKestrel)**
(AGPLv3), the local-first wildlife-photo culling app. It is **not affiliated with,
endorsed by or supported by Project Kestrel**. Please don't report Birdtag Desktop issues upstream.
"Project Kestrel", "Perch" and "Cloud Compute" and their logos are the upstream owner's marks.
This fork doesn't use their artwork, and it hides the upstream hosted services.

Everything Kestrel does still runs 100 % on your computer: bird detection, quality scoring,
scene grouping, culling and XMP export. On top of that, the fork adds:

- **UK / Europe / World bird identification** with EVA-02 Large fine-tuned on iNaturalist 2021
  (1,486 bird species). The answer is restricted to your region's species (UK = the BOU British
  List), and keywords use British names ("Robin", "Grey Heron"). Kestrel's original North
  American model is still available.
- **Reptiles, amphibians and insects** with BioCLIP 2 (zero-shot over GBIF species lists for
  UK / Europe / World), including macro shots the detector misses.
- **digiKam / Lightroom keywords**: `Birds|English|Latin` (plus `Reptiles|…`, `Amphibians|…`,
  `Insects|…`, `Mammals|…`), digiKam `TagsList`, and `…|_Check|Name` below an adjustable
  confidence threshold. Your own keywords and manual species edits are preserved.
- **Privacy defaults**: no telemetry, crash reports, update checks or other network calls unless
  you switch them on; nothing is sent to upstream's servers. See [PRIVACY-FORK.md](PRIVACY-FORK.md).

## Getting it

Windows installers (unsigned) are built by GitHub Actions; see
[FORK-NOTES.md → Installing on Windows](FORK-NOTES.md#installing-on-windows). Windows SmartScreen
will warn because the installer isn't code-signed.

### Model files (not in this repository)

The extra models are installed separately into `%LOCALAPPDATA%\BirdtagDesktop\models_extra\`:

| Model | Licence | Used for |
|---|---|---|
| `eva02_large_inat21_fp16.onnx` / `_fp32.onnx` | **CC-BY-NC-4.0, non-commercial use only** (timm / EVA-02, iNaturalist 2021) | UK / Europe / World birds |
| `bioclip-2/` (image encoder + species embeddings) | MIT (Imageomics BioCLIP 2); species lists from GBIF | reptiles, amphibians, insects |

They can be exported with the scripts in `tools/uk_models/`; see [FORK-NOTES.md](FORK-NOTES.md).
Without them the app falls back to Kestrel's original North American bird model and SpeciesNet.

## Licence and attribution

- Source code: **GNU AGPL v3**, see [LICENSE](LICENSE), as upstream. If you distribute a
  modified build, you must offer its source under the AGPL.
- Upstream: Project Kestrel (Project Kestrel LLC and contributors), https://github.com/SanjaySoniLV/ProjectKestrel.
  Read upstream's README for Kestrel's documentation, development guide and history.
- Placeholder artwork in this fork is generated (CC0); see `assets/README.md`.
- Bird catalogue data: IOC World Bird List (CC-BY 3.0), see `analyzer/models/birds/NOTICES.md`.
  British List data: BOU. Species occurrence data: GBIF.org.

## Development

The same as upstream (pywebview + vanilla JS, Python 3.11):

```bash
pip install -r requirements.txt            # or requirements-windows.txt
python analyzer/visualizer.py              # desktop app
python analyzer/cli.py "/path/to/photos" --no-gpu
pytest analyzer/tests -m unit
```

The product name lives in one place, `analyzer/branding.json`. The fork's changes are described in
[FORK-NOTES.md](FORK-NOTES.md).
