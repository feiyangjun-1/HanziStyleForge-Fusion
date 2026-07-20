# Changelog

## 2.2.0-fusion — 2026-07-20

- Added GitHub-ready English, Simplified Chinese, Japanese and Korean README files with language switching.
- Documented research and source-code references, third-party notices, font-rights limitations and the independent-implementation boundary.
- Set project copyright and package author to `feiyangjun_`.
- Standardized runtime, validation and Windows launcher messages in English.
- Reduced normal Windows launchers from thirteen files to six required files.
- Made the resilient launcher clear a completed safe-stop request automatically when restarted.
- Added English architecture, data-flow, contribution, security, citation and upgrade documentation.
- Retained production defaults `training.workers=4` and Style Encoder `batch_size=8`.
- Checkpoint format remains 300; compatible 2.1 training state can resume without model conversion.

## 2.1.0-fusion — 2026-07-20

- Changed the production Style Encoder to `batch_size=8` and `training.workers=4`.
- Enabled three-batch prefetching per worker, persistent workers, deterministic worker seeding and worker-local single-thread CPU limits.
- Enabled cuDNN benchmark, TF32 and high float32 matrix-multiplication precision on supported CUDA devices.
- Added quality-gated Style Encoder early stopping with a 100-epoch minimum, 0.2% significant-improvement threshold, 24-epoch patience and recent positive/negative similarity gates.
- Made early-stop state recoverable from an existing Fusion 2.0 `history.csv`.
- Kept checkpoint format 300 and the target/ref data-flow contract unchanged.

## 2.0.0-fusion — 2026-07-19

- Introduced target-style-only Fusion architecture with local experts, target VQ codebook, multi-resolution latent diffusion, deterministic baseline, retrieval, component residuals, high-resolution refinement and contour transformation.
- Added diffusion style-collapse monitoring, hard-example refinement using only real target glyphs, resumable per-glyph inference and complete reference-Han coverage enforcement.
- Fixed supplementary-plane `cmap` handling and strengthened target non-Han preservation verification.
- Upgraded checkpoint format to 300.

## 1.0.0

- Initial target-style self-reconstruction, reference-structure generation and non-Han preservation release.
