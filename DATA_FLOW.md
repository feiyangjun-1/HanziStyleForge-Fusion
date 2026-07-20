# HanziStyleForge Fusion data-flow contract

The following separation is mandatory:

```text
fonts/target.ttf -> STYLE LEARNING ONLY
refs/ref.otf     -> HAN STRUCTURE AND TARGET COVERAGE ONLY
```

## Training side

Each eligible target Han glyph follows this path:

```text
real target glyph
-> style-stripped target proxy
-> Style Encoder / VQ / deterministic model / latent diffusion / Refiner
-> real target raster, SDF, skeleton and edges as ground truth
```

Training indexes may point only to:

```text
work_hanzistyleforge_fusion_months/cache/target_proxy/
work_hanzistyleforge_fusion_months/cache/target_render/
work_hanzistyleforge_fusion_months/cache/target_aux/
```

There is no reference-to-target same-codepoint supervision.

## Generation side

Every Han codepoint in the reference font's default Unicode `cmap` follows this path:

```text
real reference glyph
-> reference structural proxy
-> target-style models and candidate system
-> one selected target-style output glyph
```

Generation indexes may point only to:

```text
work_hanzistyleforge_fusion_months/cache/ref_proxy/
work_hanzistyleforge_fusion_months/cache/ref_render/
```

## Enforced audit

`hanzistyleforge/contract.py` validates the CSV indexes before training and generation. The audit report is written to:

```text
work_hanzistyleforge_fusion_months/audit/data_flow_contract.json
```

A passing report contains:

```json
{
  "contract": "target-style-only training; ref-structure-only generation",
  "cross_font_training_pairs": 0,
  "passed": true
}
```

Any reference path entering training, target path masquerading as reference structure, or missing required cache family raises `DataFlowContractError` and stops the stage.

## Long-run refinement

Hard-example mining and long-run refinement use only real target glyphs as supervision. Generated reference glyphs, fallback results and earlier AI candidates never become training truth.

## Final font

The builder starts from the target font and appends/remaps only the Han codepoints selected from reference coverage. Non-Han glyphs and protected OpenType engineering data come from the target font and are verified again after saving.
