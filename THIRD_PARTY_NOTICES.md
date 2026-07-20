# Third-party notices

## Architectural references only

HanziStyleForge Fusion independently implements ideas described by zi2zi, zi2zi-JiT, FontDiffuser, HanziGen, VQ-Font, LF-Font, MX-Font, DeepVecFont-v2 and component-composition research. Their source code, pretrained model weights and font datasets are not included in this repository. See `METHOD_REFERENCES.md` for exact links and scope.

## Redistributed CJK decomposition data

- Project: `amake/cjk-decomp`
- Upstream: https://github.com/amake/cjk-decomp
- Included file: `data/cjk-decomp.txt`
- License option selected for this distribution: Apache License 2.0
- Local notice: `data/NOTICE_CJK_DECOMP.txt`
- Purpose: optional semantic/geometric region hints for local residual retrieval and per-glyph refinement

The decomposition data is not used to decide regional correctness. The actual output structure always comes from the user-supplied `refs/ref.otf`.

## Python dependencies

The program installs third-party packages such as PyTorch, fontTools, Pillow, NumPy, OpenCV and tqdm into the local virtual environment. Each package remains under its own license. The repository does not bundle the virtual environment.

## User-supplied fonts and generated artifacts

This repository does not include `fonts/target.ttf` or `refs/ref.otf`. Apache-2.0 for this software does not grant permission to train on, modify, distribute or sublicense any font. Trained weights and generated fonts may be derivative works of one or both input fonts and remain subject to their licenses.
