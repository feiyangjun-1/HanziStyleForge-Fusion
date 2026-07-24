[简体中文](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [English](README.en.md)

# HanziStyleForge Fusion

An experimental Windows-first Han font reconstruction tool. It learns style from `target.ttf`, takes Han structure from `ref.otf`, and builds an installable TTF font.

> The project is designed for long unattended runs with checkpoint resume, safe stop, and automatic retry.

## What it does

- Learns global and local font style from `fonts/target.ttf`.
- Rebuilds every Han character covered by the default glyphs in `refs/ref.otf`.
- Accepts Mainland Chinese, Taiwan, Hong Kong, Japanese, Korean, inherited, or other reference glyph standards.
- Tries to preserve Latin letters, numbers, symbols, kana, Hangul, and major OpenType data from the target font.
- Automates training, generation, candidate selection, QA, vectorization, and font building.

## How it works

```text
target.ttf: style source
        +
ref.otf: Han structure and coverage
        ↓
Style Encoder → VQ → Diffusion → Refiner / Retrieval / IDS
        ↓
Candidate selection → QA → Outline conversion → TTF
```

The program does not decide which regional glyph form is “more correct.” Final Han structure follows the default Unicode `cmap` glyphs in `ref.otf`.

## Requirements

- Windows 11 64-bit
- NVIDIA GPU with CUDA support
- Python 3.10 or later
- At least 150 GB of free disk space recommended

Input fonts:

```text
fonts\target.ttf
refs\ref.otf
```

Static fonts are recommended. `target.ttf` should contain a TrueType `glyf` table. `ref.otf` may be a static TrueType or static CFF OTF. Variable fonts, TTC, and OTC are not supported.

## Quick start

1. Download or clone this repository.
2. Place the style font at `fonts\target.ttf`.
3. Place the structure reference at `refs\ref.otf`.
4. Install the environment:

   ```text
   install_cuda130.bat
   ```

5. Verify the project:

   ```text
   verify_project.bat
   ```

6. Start or resume the complete pipeline:

   ```text
   run_months_resilient.bat
   ```

7. Request a safe stop:

   ```text
   request_safe_stop.bat
   ```

8. Clear the stop marker before resuming:

   ```text
   clear_safe_stop.bat
   ```

## Outputs

Main outputs:

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\qa\index.html
```

Training data, checkpoints, and generation progress are stored in:

```text
work_hanzistyleforge_fusion_months\
```

Do not delete this directory while training is in progress.

## Before you use it

- A complete run may take days, weeks, or longer.
- The repository does not include fonts, pretrained weights, or third-party font datasets.
- Generated fonts may remain subject to both the `target.ttf` and `ref.otf` licenses.
- Use only fonts that you are allowed to train on, modify, and redistribute.
- This is experimental software. Review the QA page and test the final font before release.

## Research and reference sources

HanziStyleForge Fusion is an independent implementation. The following projects and papers informed its architecture. Their source code, pretrained weights, and font datasets are not bundled in this repository.

| Source | Ideas studied |
|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | Han glyph style transfer and content/style separation |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | Multi-reference style conditioning and diffusion transformers |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | Diffusion generation, multi-scale content aggregation, explicit style constraints |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ representations and conditional latent diffusion |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) | Discrete font tokens and structure-aware enhancement |
| [LF-Font / MX-Font](https://github.com/clovaai/fewshot-font-generation) | Local component style, factorization, and multiple experts |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer vector sequences and contour correction |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | Component-region transforms and scalable composition |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | Unicode IDS component structure and local-region hints |

A citation indicates architectural reference only. It does not grant permission to copy upstream code, weights, data, or fonts. Check the current license and terms of every third-party artifact before use.

## Contributing

Issues and pull requests are welcome. Any contributed third-party code, data, or model must include its source and license information.
