# HanziStyleForge Fusion 2.2 test report

Test date: 2026-07-20  
Checkpoint format: 300

## Scope

These tests validate code paths, data separation, recovery, coverage and font-engineering protection. They do not replace a months-long production run on the user's complete fonts or professional visual review of every generated glyph.

## Static and release checks

- Python source compilation succeeds.
- JSON configuration files parse successfully.
- Production configurations use `training.workers=4` and Style Encoder `batch_size=8`.
- Runtime and retained Windows-launcher messages are English.
- The final project contains only six normal-use `.bat` launchers.
- README language links resolve to English, Simplified Chinese, Japanese and Korean files.
- Project version is 2.2.0; checkpoint format remains 300.
- The data-flow contract module imports and its self-test path is included.

## 2.1 checkpoint compatibility

- Style early-stop state can be reconstructed from a pre-2.2 `history.csv` that lacks explicit counter fields.
- Compatible 2.1 model checkpoints require no conversion because architecture dimensions and checkpoint format are unchanged.
- Documentation and launcher cleanup does not require deleting caches or trained stages.

## Model cascade smoke coverage

The internal tests cover construction, forward/backward flow or sampling for:

- multi-reference `StyleReferenceEncoder` and local experts;
- `MultiScaleContentEncoder`;
- `GlyphVQVAE` encoding, quantization and decoding;
- `LatentDiffusionUNet`, classifier-free guidance and DDIM sampling;
- deterministic multi-task safety baseline;
- `StyleAwareGlyphRefiner`;
- `ContourSequenceTransformer`.

Interfaces at 256, 384 and 512 pixels are covered by the test code.

## Data isolation and collapse protection

Target self-reconstruction indexes contain target-cache paths only. The contract rejects cross-font training pairs. Diffusion history records proxy/target distances and style direction, and the collapse-guard condition is covered by diagnostic tests.

## Coverage and recovery

The smoke path verifies resumable generation state, atomic partial selection files, per-glyph error isolation, emergency fallback and complete-coverage assertions. Randomly initialized short-run models are not used as evidence of visual quality.

## TrueType construction

The self-test constructs target/reference fixtures with non-Han scripts, BMP Han and a supplementary-plane Han codepoint. It checks safe format 4/12 `cmap` behavior, Han replacement/addition, output reopening and protected non-Han mappings, outlines, metrics and tables.

## Known limits

- No months-long validation on the user's complete target/reference fonts was performed in the release environment.
- No professional type designer reviewed every generated glyph.
- Generated Han glyphs do not receive hand-authored TrueType hinting.
- Reference forms exposed only through runtime `locl` are outside the default-cmap workflow.
- Fonts near the 65,535-glyph TrueType limit require special capacity review.
