# HanziStyleForge Fusion 2.2

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

A Windows-first, resumable Han glyph reconstruction pipeline that **learns style only from `target.ttf`**, takes **Han structure and character coverage only from `ref.otf`**, rebuilds every Han codepoint covered by the reference font, and verifies preservation of the target font's non-Han glyphs and major OpenType engineering data.

> Status: research/engineering alpha. Generated fonts still require visual review, application testing, and license review before distribution.

## Core contract

```text
fonts/target.ttf  -> STYLE LEARNING ONLY
refs/ref.otf      -> HAN STRUCTURE AND TARGET COVERAGE ONLY
```

Training uses self-reconstruction samples from `target.ttf`:

```text
real target glyph -> style-stripped target proxy -> model -> real target glyph ground truth
```

Generation reads `ref.otf` only after style learning:

```text
reference Han structure -> target-style model -> rebuilt target-style Han glyph
```

The project does not require a manually curated equivalent-glyph list, CN/non-CN classification, or cross-font paired supervision. `hanzistyleforge/contract.py` rejects a dataset if reference-font paths enter training or target-font paths are used as generation structure.

## Main features

- Rebuilds every Han codepoint in the default Unicode `cmap` of `ref.otf`.
- Supports Mainland Chinese, Taiwan, Hong Kong, Japanese, Korean, inherited-form, or other reference glyph standards, provided the desired forms are the reference font's default glyphs.
- Learns global and local target style from multiple real target glyphs.
- Uses a VQ glyph codebook, latent diffusion, deterministic baseline, local real-glyph retrieval, component residuals, a high-resolution refiner, topology gating, and contour refinement.
- Stops automatically if diffusion predictions drift toward the style-stripped structural proxy instead of target ground truth.
- Resumes training, generation, per-glyph refinement, QA, and font building from durable checkpoints.
- Starts the final font from `target.ttf`, then appends/remaps rebuilt Han glyphs.
- Verifies non-Han `cmap`, glyph IDs, outlines, metrics, UVS mappings, layout tables, and hinting-related tables before publishing the output.

## Requirements

Recommended production environment:

```text
Windows 11 64-bit
NVIDIA GPU with 12 GB VRAM or more
Python 3.10-3.14 64-bit
Local SSD
At least 150 GB free disk space
```

Input rules:

- `fonts/target.ttf`: static TrueType font with a `glyf` table; not a variable font.
- `refs/ref.otf`: static TrueType TTF/OTF or static CFF OTF.
- Avoid TTC/OTC files and reference fonts that depend on runtime `locl` substitution to expose the desired regional forms.
- The package contains no fonts and no pretrained weights.

## Quick start on Windows 11

1. Extract the project to a short local path, for example:

   ```text
   C:\FontWork\HanziStyleForge-Fusion
   ```

2. Copy the fonts:

   ```text
   fonts\target.ttf
   refs\ref.otf
   ```

3. Install the isolated CUDA environment:

   ```text
   install_cuda130.bat
   ```

4. Verify the environment, fonts, coverage, configuration, and data-flow contract:

   ```text
   verify_project.bat
   ```

5. Start or resume the complete long-run workflow:

   ```text
   run_months_resilient.bat
   ```

6. Inspect progress without changing state:

   ```text
   run_status.bat
   ```

7. Request a stop after the next durable checkpoint:

   ```text
   request_safe_stop.bat
   ```

   Run `run_months_resilient.bat` again to resume. It clears the completed safe-stop request automatically.

8. Open the generated QA report when available:

   ```text
   open_qa.bat
   ```

## Included Windows launchers

Only the launchers required for normal use are retained:

| File | Purpose |
|---|---|
| `install_cuda130.bat` | Create `.venv`, install dependencies and verify CUDA |
| `verify_project.bat` | Run self-tests and project validation |
| `run_months_resilient.bat` | Start or resume the complete recoverable workflow |
| `request_safe_stop.bat` | Stop after the next durable checkpoint |
| `run_status.bat` | Display read-only progress information |
| `open_qa.bat` | Open the HTML QA report |

Advanced stages remain available through the Python CLI:

```powershell
.venv\Scripts\python.exe hanzistyleforge.py --help
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-train
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-generate
```

## Production configuration

The main configuration is `config_fusion_months_12gb.json`.

Current Style Encoder defaults:

```json
{
  "training": {
    "workers": 4
  },
  "fusion": {
    "style_encoder": {
      "batch_size": 8
    }
  }
}
```

The Style Encoder supports quality-gated early stopping. It trains for at least 100 epochs, then stops after 24 epochs without a significant validation improvement when the recent positive/negative style-similarity metrics remain healthy. Existing compatible checkpoints and `history.csv` files are reused.

Do not change image sizes, model channels, style dimensions, latent channels, or codebook size in the middle of a stage unless you intentionally discard that stage's checkpoints.

## Workflow

```text
check inputs and CUDA
-> render target and reference glyphs
-> enforce data-flow contract
-> build target local-style atlas
-> train Style Encoder
-> train target VQ codebook
-> train deterministic safety baseline
-> train multi-resolution latent diffusion
-> mine hard real-target examples
-> train high-resolution refiner
-> train contour Transformer
-> generate every reference Han glyph
-> topology and style candidate selection
-> resumable per-glyph refinement
-> QA
-> SDF/TrueType vectorization
-> rebuild and verify the final font
```

The process can run for weeks or months. Major stages and individual generated glyphs are checkpointed. Recoverable failures are retried by the resilient launcher; persistent quality-protection failures are not retried automatically.

## Coverage and preservation

Default scope:

```json
{
  "scope": {
    "mode": "reference_han",
    "include_compatibility_ideographs": true
  }
}
```

Every reference Han codepoint must receive a generated or safe-fallback result. If `require_complete=true`, any missing target prevents final publication.

Because the builder protects the original target glyph IDs by appending new glyphs, this limit must hold:

```text
target glyph count + appended Han glyph count < 65,536
```

The final build verifies preservation of target non-Han data, including Latin, Cyrillic, kana, Hangul, numbers, punctuation, symbols, non-Han outlines, metrics, Unicode variation sequences, GSUB/GPOS/GDEF/BASE/kern, and TrueType hinting-related tables.

## Outputs

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\generated\coverage.json
work_hanzistyleforge_fusion_months\generated\selection.csv
work_hanzistyleforge_fusion_months\refined\selection.csv
work_hanzistyleforge_fusion_months\qa\index.html
```

## Research and code references

HanziStyleForge Fusion is an independent implementation. The following public projects and papers informed the design direction. Their source code, pretrained weights, and font datasets are **not vendored or redistributed** by this repository.

| Upstream work | Design ideas referenced | Upstream license/status |
|---|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | Chinese glyph style transfer and content/style conditioning | Apache-2.0 |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | Multi-reference conditioning and diffusion-transformer direction | MIT software license plus upstream font-artifact addendum; check the current upstream terms |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | Denoising diffusion, multi-scale content aggregation, explicit style constraints | No license file was visible in the upstream repository when this release was prepared; do not copy code or weights without permission |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ-VAE plus latent-diffusion font-completion workflow | Apache-2.0 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) / [paper](https://doi.org/10.1609/aaai.v38i15.29577) | Discrete font-token priors and structure-aware enhancement | Check the current repository license before copying code or weights |
| [LF-Font / MX-Font unified repository](https://github.com/clovaai/fewshot-font-generation) | Localized component style, factorization, and multiple experts | MIT; some upstream modules have separate provenance notices |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer-based vector sequence modeling and contour correction | MIT code; the upstream font dataset has separate non-commercial restrictions |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | Component-region transformation and scalable composition | Paper reference; review any associated code/data terms separately |
| [cjk-decomp](https://github.com/amake/cjk-decomp) | Optional decomposition hints for local residual regions | Multi-licensed; this distribution uses the Apache-2.0 option for the bundled data file |

The detailed provenance statement is in [METHOD_REFERENCES.md](METHOD_REFERENCES.md), and redistributed third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Architectural inspiration is not permission to copy an implementation, dataset, font, or model weight. Anyone adding upstream material must preserve its copyright notices and comply with its current license.

## License and font rights

```text
Copyright 2026 feiyangjun_
```

The HanziStyleForge source code and project documentation are licensed under the [Apache License 2.0](LICENSE), except separately identified third-party material.

The project license does not grant rights to any user-supplied font. You are responsible for confirming that the licenses for `target.ttf` and `ref.otf` permit training, modification, derivative font creation, and distribution. Generated fonts and checkpoints may remain subject to one or both input-font licenses.

## Additional documentation

- [Architecture](ARCHITECTURE.md)
- [Data-flow contract](DATA_FLOW.md)
- [Method references](METHOD_REFERENCES.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Test report](TEST_REPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
