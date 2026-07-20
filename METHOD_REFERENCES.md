# Method and code references

HanziStyleForge Fusion 2.2 is an independent implementation maintained under the project license. Public research and open-source projects informed the architecture, but this repository does **not** vendor or redistribute their source trees, pretrained weights, or font datasets, except for the separately noticed `data/cjk-decomp.txt` file.

This document distinguishes architectural inspiration from copied code. A paper or repository being cited does not grant permission to copy code, weights, datasets, or fonts outside its current terms.

## Upstream works consulted

| Work | Public source | Ideas studied for this project | Inclusion in this repository |
|---|---|---|---|
| zi2zi | https://github.com/kaonashi-tyc/zi2zi | Chinese glyph style transfer; content/style conditioning | No upstream code, weights or data included |
| zi2zi-JiT | https://github.com/kaonashi-tyc/zi2zi-JiT | Multi-reference conditioning; diffusion-transformer direction | No upstream code, weights or data included |
| FontDiffuser | https://github.com/yeungchenwa/FontDiffuser | Denoising diffusion; multi-scale content aggregation; explicit style constraints | No upstream code, weights or data included; verify upstream permission before reuse |
| HanziGen | https://github.com/wangwenho/HanziGen | VQ representation combined with conditional latent diffusion | No upstream code, weights or data included |
| VQ-Font | https://github.com/Yaomingshuai/VQ-Font and https://doi.org/10.1609/aaai.v38i15.29577 | Discrete font-token priors; structure-aware enhancement | No upstream code, weights or data included; verify current terms before reuse |
| LF-Font / MX-Font | https://github.com/clovaai/fewshot-font-generation | Localized component style; factorization; multiple experts | No upstream code, weights or data included |
| DeepVecFont-v2 | https://github.com/yizhiwang96/deepvecfont-v2 | Transformer vector-sequence modeling; contour correction | No upstream code, weights or data included; upstream dataset terms are separate from code terms |
| Efficient and Scalable Chinese Vector Font Generation via Component Composition | https://arxiv.org/abs/2404.06779 | Component-region transformations and scalable character composition | Paper-level architectural reference only |
| cjk-decomp | https://github.com/amake/cjk-decomp | Optional character-decomposition hints for local regions | `data/cjk-decomp.txt` is redistributed under the Apache-2.0 option with notices |

## Relationship to the implementation

The following HanziStyleForge modules are original project implementations:

- data separation and contract enforcement;
- target-only self-reconstruction datasets;
- style encoder and local experts;
- VQ glyph autoencoder and latent diffusion training;
- deterministic baseline and high-resolution refiner;
- target-glyph retrieval and component residual atlas;
- topology/style candidate selection and collapse guards;
- resumable per-glyph processing;
- contour-sequence refinement and SDF-to-TrueType conversion;
- non-Han preservation and post-build verification.

Similarity in high-level terminology does not imply line-by-line derivation. Contributors who later add third-party material must document the exact files, preserve copyright notices, satisfy the applicable license, and update `THIRD_PARTY_NOTICES.md`.

## License caution

Repository licenses can change, and code, model weights, demonstrations and font datasets from one upstream project may have different terms. Review the exact upstream revision and artifact license before importing anything. The Apache-2.0 license of HanziStyleForge does not relicense third-party material or user-supplied fonts.
