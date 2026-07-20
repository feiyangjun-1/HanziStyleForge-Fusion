from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from .component_atlas import ComponentAtlas
from .contour_polish import contour_features, normalize_contour, resample_closed_contour
from .fusion_diffusion import DiffusionSchedule, ddim_sample
from .fusion_model import (
    ContourSequenceTransformer,
    GlyphVQVAE,
    LatentDiffusionUNet,
    StyleAwareGlyphRefiner,
    StyleReferenceEncoder,
)


def run_fusion_selftest() -> None:
    torch.set_num_threads(1)
    style = StyleReferenceEncoder(base=4, style_dim=32, expert_count=2, heads=2).eval()
    references = torch.rand(1, 2, 1, 32, 32)
    style_output = style(references)
    assert style_output["experts"].shape == (1, 2, 32)

    vq = GlyphVQVAE(base=4, latent_channels=4, embeddings=16).eval()
    target_aux = torch.rand(1, 4, 64, 64)
    with torch.no_grad():
        vq_output = vq(target_aux, update_codebook=False)
    assert vq_output["quantized"].shape == (1, 4, 8, 8)
    assert vq_output["reconstruction"].shape == target_aux.shape

    diffusion = LatentDiffusionUNet(
        latent_channels=4, content_channels=10, base=8, content_base=4,
        style_dim=32, expert_count=2, time_dim=32,
    ).eval()
    proxy = torch.rand(1, 10, 64, 64)
    schedule = DiffusionSchedule.create(8).to("cpu")
    with torch.no_grad():
        latent = ddim_sample(
            diffusion,
            schedule,
            tuple(vq_output["quantized"].shape),
            content_proxy=proxy,
            style_experts=style_output["experts"],
            steps=2,
            generator=torch.Generator().manual_seed(7),
        )
        ink = torch.sigmoid(vq.decode(latent)[:, :1])
    assert ink.shape == (1, 1, 64, 64)

    refiner = StyleAwareGlyphRefiner(in_channels=11, base=4, style_dim=32, expert_count=2).eval()
    with torch.no_grad():
        refined = refiner(torch.cat([ink, proxy], dim=1), style_output["experts"])
    assert refined.shape == (1, 1, 64, 64)

    contour = np.asarray([[8, 8], [56, 8], [56, 56], [8, 56]], dtype=np.float32)
    sampled = resample_closed_contour(contour, 32)
    normalized, _, _ = normalize_contour(sampled)
    features = torch.from_numpy(contour_features(normalized))[None]
    polisher = ContourSequenceTransformer(points=32, hidden=32, layers=1, heads=2).eval()
    with torch.no_grad():
        contour_output = polisher(features)
    assert contour_output["offset"].shape == (1, 32, 2)
    assert contour_output["corner_logits"].shape == (1, 32, 1)

    # Component atlas query shape contract.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        atlas_path = root / "atlas.npz"
        labels_path = root / "labels.json"
        descriptors = np.zeros((2, 10), dtype=np.float32)
        np.savez_compressed(
            atlas_path,
            descriptors=descriptors,
            descriptor_mean=np.zeros(10, dtype=np.float32),
            descriptor_std=np.ones(10, dtype=np.float32),
            residuals=np.zeros((2, 8, 8), dtype=np.int8),
            label_ids=np.zeros(2, dtype=np.int32),
            source_codepoints=np.asarray([0x4E00, 0x4E8C], dtype=np.int32),
            source_xy=np.zeros((2, 2), dtype=np.float32),
        )
        labels_path.write_text('{"labels":["一"],"ranges":{"一":[0,2]}}', encoding="utf-8")
        atlas = ComponentAtlas.load(atlas_path, labels_path)
        indices, distances = atlas.query("一", np.zeros(10, dtype=np.float32), k=2)
        assert len(indices) == 2 and len(distances) == 2


if __name__ == "__main__":
    run_fusion_selftest()
    print("HanziStyleForge Fusion self-test: OK")
