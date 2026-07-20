# HanziStyleForge Fusion 2.2 architecture

[README](README.md) | [Data-flow contract](DATA_FLOW.md) | [Method references](METHOD_REFERENCES.md)

## 1. Non-negotiable separation

```text
fonts/target.ttf -> style learning only
refs/ref.otf     -> Han structure and generation coverage only
```

The training path is target-font self-reconstruction. The reference font is not a paired target and must never appear in the training index. Generation begins only after the target-style models are available.

## 2. Preparation and audit

The preparation stage:

1. validates static-font requirements and Unicode coverage;
2. renders target glyphs and reference Han glyphs into separate caches;
3. derives style-stripped proxies, signed-distance fields, edges and skeletons;
4. builds target-only training and validation indexes;
5. writes `audit/data_flow_contract.json` and aborts on a path violation.

## 3. Target-style representation

### Style Reference Encoder

A multi-reference encoder learns one global style vector and local expert features from real target glyphs. The production defaults are `batch_size=8` and `training.workers=4`. Its quality-gated early stopping restores state from `history.csv`, so compatible 2.1 checkpoints can continue without restarting.

### Local target atlas

Real target glyphs are indexed for local retrieval. Retrieved target examples provide style evidence only; they never replace the reference font as the structural source for a requested codepoint.

### VQ glyph codebook

A target-only VQ autoencoder learns recurring target-shape and stroke-pattern tokens. It acts as a discrete prior for later latent generation.

## 4. Candidate generation

For each Han codepoint in the default Unicode `cmap` of `ref.otf`, the system constructs several independent candidates:

- deterministic self-reconstruction baseline;
- latent-diffusion candidates from multiple seeds;
- real-target retrieval transfer;
- decomposition/component residual candidate;
- high-resolution refined fusion candidate;
- reference emergency fallback.

The reference image and its proxy carry structure. The learned style vector, local experts, codebook and target atlas carry style.

## 5. Quality protection

The diffusion stage records distance to target truth and distance to the style-stripped proxy. Training stops with a dedicated diagnostic if predictions become systematically more proxy-like than target-like after warm-up.

Inference uses topology checks, stroke occupancy, edge/SDF agreement, target-style similarity, candidate consensus and fallback rules. A single bad glyph cannot terminate the remaining coverage run.

## 6. Contour and TrueType construction

Selected raster/SDF results are refined by a contour-sequence Transformer and converted to quadratic TrueType outlines. The final builder starts from `target.ttf`, appends rebuilt Han glyphs, updates Unicode mappings, and then reopens the output for verification.

Protected non-Han data includes:

- Unicode mappings and variation sequences;
- existing glyph IDs and outlines;
- horizontal and vertical metrics;
- GSUB, GPOS, GDEF, BASE and kern data;
- TrueType hinting-related tables.

The build fails if a protected non-Han comparison changes unexpectedly.

## 7. Recovery model

Major training stages, generation selection, per-glyph refinement, QA and font construction write durable state. `run_months_resilient.bat` retries recoverable process exits. A safe-stop request is honored only after a durable checkpoint and is cleared automatically on the next launch.

## 8. Scope and limits

This is a research/engineering system, not a substitute for type-design review. It does not add hand-authored hinting to generated glyphs, cannot infer language-specific forms hidden behind runtime `locl`, and cannot override the legal terms of either input font.
