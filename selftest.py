from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from hanzistyleforge.build_font import _map_codepoint
from hanzistyleforge.features import expand_proxy_channels, make_target_aux, split_prediction
from hanzistyleforge.fusion_selftest import run_fusion_selftest
from hanzistyleforge.fusion_training import (
    _create_vq_optimizer,
    _prepare_vq_optimizer_state_dict,
    _repair_vq_optimizer_state_devices,
    _restore_vq_optimizer_backend,
    _style_plateau_state,
    _style_quality_gate,
)
from hanzistyleforge.contract import DataFlowContractError, validate_data_flow_contract
from hanzistyleforge.decomposition import (
    _token_codepoint,
    component_zones,
    load_decompositions,
    parse_ids_expression,
)
from hanzistyleforge.marathon_refine import run_marathon_refinement
from hanzistyleforge.losses import FontLossFinal, VQReconstructionLoss
from hanzistyleforge.inference import _emergency_fallback_row
from hanzistyleforge.model import FontStyleNetFinal, GlyphRefinerFinal, PatchDiscriminatorFinal
from hanzistyleforge.proxy import (
    calibrate_observed_structure_thresholds,
    calibrate_same_structure_thresholds,
    make_content_proxy,
    make_reference_fallbacks,
    proxy_structure_score,
    read_proxy,
    save_ink,
    save_proxy,
)
from hanzistyleforge.report import _load_gray
from hanzistyleforge.retrieval import StyleAtlas, _descriptor, render_retrieval_candidate
from hanzistyleforge.runtime import configure_runtime
from hanzistyleforge.topology import topology_metrics, validate_topology
from hanzistyleforge.vectorize import image_to_ttglyph
from hanzistyleforge.util import save_json, write_csv


def main() -> None:
    runtime = configure_runtime({"training": {"cpu_threads": 1, "interop_threads": 1, "opencv_threads": 1}})
    assert runtime.get("opencv_threads") == 1
    assert runtime.get("torch_threads") == 1
    synthetic_history = [
        {
            "epoch": epoch,
            "val_loss": 0.03 if epoch < 5 else 0.02996,
            "val_positive_similarity": 0.9995,
            "val_negative_similarity": 0.11,
        }
        for epoch in range(1, 15)
    ]
    significant_best, last_significant, stale = _style_plateau_state(
        synthetic_history,
        through_epoch=14,
        minimum_relative_improvement=0.002,
    )
    assert significant_best == 0.03
    assert last_significant == 1 and stale == 13
    quality_ready, quality = _style_quality_gate(
        synthetic_history,
        window=10,
        minimum_positive_similarity=0.999,
        maximum_negative_similarity=0.15,
    )
    assert quality_ready
    assert quality["median_positive_similarity"] >= 0.999
    ink = np.zeros((96, 96), dtype=np.float32)
    ink[14:82, 43:53] = 1.0
    ink[43:53, 14:82] = 1.0
    # Add a closed counter so hole-position and Euler checks are exercised.
    ink[18:38, 18:38] = 1.0
    ink[23:33, 23:33] = 0.0

    proxy4 = make_content_proxy(ink, output_size=64, skeleton_size=64)
    proxy10 = expand_proxy_channels(proxy4)
    target_aux_np = make_target_aux(
        torch.nn.functional.interpolate(
            torch.from_numpy(ink[None, None]), size=(64, 64), mode="bilinear", align_corners=False
        )[0, 0].numpy()
    ).astype(np.float32) / 255.0
    assert proxy4.shape == (64, 64, 4)
    assert proxy10.shape == (64, 64, 10)
    assert target_aux_np.shape == (64, 64, 4)
    assert proxy_structure_score(proxy4, proxy4, analysis_size=64) < 1e-6
    thresholds = calibrate_same_structure_thresholds([proxy4] * 3, analysis_size=64)
    observed = calibrate_observed_structure_thresholds(
        [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.36, 0.38, 0.40] * 3,
        thresholds,
    )
    assert observed["very_strict"] < observed["uncertain"]

    topology_cfg = {
        "maximum_component_delta": 0,
        "maximum_hole_delta": 0,
        "maximum_euler_delta": 0,
        "minimum_endpoint_tolerance": 2,
        "minimum_junction_tolerance": 2,
        "maximum_missing_skeleton_p90": 0.05,
        "maximum_extra_skeleton_p90": 0.05,
        "maximum_hole_centroid_chamfer": 0.08,
        "maximum_zone_skeleton_distance": 0.32,
        "maximum_topology_score": 0.14,
    }
    topology = topology_metrics(ink, ink, size=64)
    assert validate_topology(topology, topology_cfg)["hard_pass"]
    fallbacks = dict(
        make_reference_fallbacks(
            ink,
            {
                "stroke_radius": {"median": 3.0},
                "bbox_width_ratio": {"median": 0.7},
                "bbox_height_ratio": {"median": 0.7},
                "center_x": {"median": 0.5},
                "center_y": {"median": 0.5},
            },
        )
    )
    assert "reference_raw" in fallbacks
    assert parse_ids_expression("⿱⿰日月木").serialize() == "⿱⿰日月木"
    assert _token_codepoint("⑦") == ord("⑦")
    assert _token_codepoint("19968") == 19968
    with tempfile.TemporaryDirectory() as ids_directory:
        ids_path = Path(ids_directory) / "ids.txt"
        ids_path.write_text(
            "# synthetic standard IDS test data\n"
            "U+660E\t明\t⿰日月\n"
            "U+4EAE\t亮\t⿱⿳亠口冖几[G]\t⿱亠兄[TJ]\n"
            "U+4E0D\t不\t⿱一③\n"
            "U+537F\t卿\t⿲𠂎⑦卩[K]\n",
            encoding="utf-8",
        )
        decompositions = load_decompositions(ids_path, region_priority=["G"])
        assert 0x660E in decompositions
        assert decompositions[0x4EAE].regions == ("G",)
        assert decompositions[0x4E0D].sequence == "⿱一③"
        assert decompositions[0x537F].sequence == "⿲𠂎⑦卩"
        zones = component_zones(0x660E, ink, decompositions, fallback_grid=3)
        assert len(zones) >= 2 and all(zone.shape == ink.shape for zone in zones)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        local_ids_path = root / "ids.txt"
        local_ids_path.write_text(
            "# synthetic standard IDS test data\nU+660E\t明\t⿰日月\n",
            encoding="utf-8",
        )
        proxy_path = root / "proxy.png"
        ink_path = root / "ink.png"
        atlas_path = root / "atlas.npz"
        save_proxy(proxy_path, proxy4)
        save_ink(ink_path, ink)
        loaded = read_proxy(proxy_path)
        assert loaded.shape == (64, 64, 4)
        assert _load_gray("", 32).size == (32, 32)
        emergency = _emergency_fallback_row(
            {
                "codepoint": str(0x4E00),
                "unicode": "U+4E00",
                "char": "一",
                "has_target": "0",
                "locl_sensitive": "0",
                "preliminary_status": "missing",
                "ref_path": str(ink_path),
                "target_path": "",
                "ref_proxy_path": str(proxy_path),
                "structure_score": "",
            },
            0x4E00,
            root,
            RuntimeError("self-test"),
        )
        assert emergency["chosen_label"] == "reference_emergency"
        assert Path(emergency["chosen_path"]).is_file()

        patch = loaded[8:56, 8:56]
        descriptor = _descriptor(patch, 0.5, 0.5, descriptor_size=6, position_weight=1.35)
        descriptors = np.stack(
            [descriptor + np.random.default_rng(index).normal(0, 0.01, descriptor.shape) for index in range(64)]
        ).astype(np.float32)
        mean = descriptors.mean(axis=0)
        std = np.maximum(descriptors.std(axis=0), 1e-4)
        normalized = (descriptors - mean) / std
        residuals = np.zeros((64, 32, 32), dtype=np.int8)
        residuals[:, 14:18, :] = 12
        np.savez_compressed(
            atlas_path,
            descriptors=normalized.astype(np.float16),
            descriptor_mean=mean,
            descriptor_std=std,
            residuals=residuals,
            source_codepoints=np.arange(0x4E00, 0x4E00 + 64, dtype=np.int32),
            source_xy=np.full((64, 2), 0.5, dtype=np.float16),
        )
        atlas = StyleAtlas.load(atlas_path, trees=2)
        # Final neural proxies have ten channels; retrieval deliberately slices to
        # the stable first four descriptor channels.
        retrieval, metadata = render_retrieval_candidate(
            proxy10,
            atlas,
            {
                "grid": 2,
                "window_ratio": 0.75,
                "descriptor_size": 6,
                "position_weight": 1.35,
                "minimum_activity": 0.001,
                "knn": 3,
                "flann_checks": 16,
                "strength": 0.8,
            },
        )
        assert retrieval.shape == (64, 64)
        assert metadata["query_count"] > 0

        work = root / "work"
        (work / "generated").mkdir(parents=True)
        (work / "audit").mkdir(parents=True)
        (work / "dataset").mkdir(parents=True)
        write_csv(
            work / "generated" / "selection.csv",
            [{
                "codepoint": 0x660E, "unicode": "U+660E", "char": "明",
                "chosen_source": "fallback", "final_action": "replace",
                "chosen_path": str(ink_path), "ref_path": str(ink_path),
                "ref_proxy_path": str(proxy_path), "notes": "",
            }],
            ["codepoint", "unicode", "char", "chosen_source", "final_action", "chosen_path", "ref_path", "ref_proxy_path", "notes"],
        )
        write_csv(
            work / "audit" / "analysis.csv",
            [{"codepoint": 0x660E, "complexity": 0.5}],
            ["codepoint", "complexity"],
        )
        save_json(work / "dataset" / "style_profile.json", {
            "stroke_radius": {"median": 3.0, "sigma": 1.0},
            "bbox_width_ratio": {"median": 0.7, "sigma": 0.1},
            "bbox_height_ratio": {"median": 0.7, "sigma": 0.1},
            "center_x": {"median": 0.5, "sigma": 0.05},
            "center_y": {"median": 0.5, "sigma": 0.05},
            "ink_ratio": {"median": 0.2, "sigma": 0.1},
        })
        save_json(work / "dataset" / "style_profiles.json", {"global": {
            "stroke_radius": {"median": 3.0, "sigma": 1.0},
            "bbox_width_ratio": {"median": 0.7, "sigma": 0.1},
            "bbox_height_ratio": {"median": 0.7, "sigma": 0.1},
            "center_x": {"median": 0.5, "sigma": 0.05},
            "center_y": {"median": 0.5, "sigma": 0.05},
            "ink_ratio": {"median": 0.2, "sigma": 0.1},
        }, "bins": []})
        refine_cfg = {
            "paths": {"work_dir": str(work)},
            "render": {"threshold": 0.5},
            "topology": topology_cfg | {"analysis_size": 64, "prune_iterations": 1},
            "marathon": {"refine": {
                "enabled": True, "analysis_size": 64, "passes": 1,
                "global_search_trials": 2, "local_sweeps": 1, "zone_grid": 2,
                "save_every_glyphs": 1, "maximum_glyphs": 0,
                "use_component_layout": True,
                "decomposition_file": str(local_ids_path),
                "auto_download": False,
            }},
        }
        refined_summary = run_marathon_refinement(refine_cfg)
        assert refined_summary["output_count"] == 1
        assert (work / "refined" / "selection.csv").is_file()

        glyph = image_to_ttglyph(
            ink_path,
            upm=1000,
            pad=8,
            y_bottom=-120,
            y_top=880,
            config={
                "outline_mode": "sdf_quadratic",
                "minimum_contour_area": 1.0,
                "sdf_upsample": 2,
                "sdf_sigma": 0.5,
                "sdf_levels": [0.0, -0.12, 0.12],
                "curve_simplify": 1.0,
                "corner_angle_degrees": 108.0,
                "maximum_points_per_contour": 128,
            },
        )
        assert int(glyph.numberOfContours) > 0

        # A supplementary-plane Han mapping must create a complete format-12
        # cmap.  It must not corrupt format-4 or hide existing BMP/non-Han text.
        cmap_font_path = root / "supplementary-cmap.ttf"
        cmap_saved_path = root / "supplementary-cmap-saved.ttf"
        builder = FontBuilder(1000, isTTF=True)
        glyph_order = [".notdef", "A", "han"]
        builder.setupGlyphOrder(glyph_order)
        simple_glyphs = {}
        for glyph_name in glyph_order:
            pen = TTGlyphPen(None)
            pen.moveTo((100, 100)); pen.lineTo((900, 100)); pen.lineTo((900, 900)); pen.lineTo((100, 900)); pen.closePath()
            simple_glyphs[glyph_name] = pen.glyph()
        builder.setupGlyf(simple_glyphs)
        builder.setupHorizontalMetrics({name: (1000, 0) for name in glyph_order})
        builder.setupHorizontalHeader(ascent=880, descent=-120)
        builder.setupCharacterMap({0x41: "A", 0x4E00: "han"})
        builder.setupOS2(sTypoAscender=880, sTypoDescender=-120, usWinAscent=900, usWinDescent=140)
        builder.setupNameTable({"familyName": "CmapSelftest", "styleName": "Regular"})
        builder.setupPost(); builder.setupMaxp(); builder.save(cmap_font_path)
        cmap_font = TTFont(cmap_font_path)
        _map_codepoint(cmap_font, 0x20000, "han")
        cmap_font.save(cmap_saved_path); cmap_font.close()
        cmap_verify = TTFont(cmap_saved_path)
        best_cmap = cmap_verify.getBestCmap() or {}
        assert {0x41, 0x4E00, 0x20000}.issubset(best_cmap)
        assert any(table.isUnicode() and table.format == 12 for table in cmap_verify["cmap"].tables)
        assert all(0x20000 not in (getattr(table, "cmap", {}) or {}) for table in cmap_verify["cmap"].tables if table.format == 4)
        cmap_verify.close()

        # Data-flow contract: training must use only target caches and final
        # generation must use only ref caches.  A ref path in the training CSV
        # must be rejected immediately.
        contract_work = root / "contract_work"
        target_proxy_dir = contract_work / "cache" / "target_proxy"
        target_render_dir = contract_work / "cache" / "target_render"
        target_aux_dir = contract_work / "cache" / "target_aux"
        ref_proxy_dir = contract_work / "cache" / "ref_proxy"
        ref_render_dir = contract_work / "cache" / "ref_render"
        for folder in (target_proxy_dir, target_render_dir, target_aux_dir, ref_proxy_dir, ref_render_dir):
            folder.mkdir(parents=True, exist_ok=True)
        for path in (
            target_proxy_dir / "U4E00.png", target_render_dir / "U4E00.png",
            target_aux_dir / "U4E00.png", ref_proxy_dir / "U4E00.png",
            ref_render_dir / "U4E00.png",
        ):
            path.write_bytes(b"selftest")
        (contract_work / "dataset").mkdir(parents=True, exist_ok=True)
        (contract_work / "audit").mkdir(parents=True, exist_ok=True)
        write_csv(
            contract_work / "dataset" / "index.csv",
            [{
                "sample_id": "style-self-U4E00", "codepoint": 0x4E00, "unicode": "U+4E00",
                "char": "一", "split": "train", "mode": "self",
                "proxy_path": str(target_proxy_dir / "U4E00.png"),
                "target_path": str(target_render_dir / "U4E00.png"),
                "target_aux_path": str(target_aux_dir / "U4E00.png"),
                "sample_weight": 1.0, "structure_score": 0.0, "complexity": 1.0,
            }],
            ["sample_id", "codepoint", "unicode", "char", "split", "mode", "proxy_path",
             "target_path", "target_aux_path", "sample_weight", "structure_score", "complexity"],
        )
        write_csv(
            contract_work / "audit" / "analysis.csv",
            [{
                "codepoint": 0x4E00, "unicode": "U+4E00", "char": "一",
                "ref_path": str(ref_render_dir / "U4E00.png"),
                "ref_proxy_path": str(ref_proxy_dir / "U4E00.png"),
            }],
            ["codepoint", "unicode", "char", "ref_path", "ref_proxy_path"],
        )
        contract_cfg = {"paths": {"work_dir": str(contract_work)}}
        contract_report = validate_data_flow_contract(contract_cfg, require_prepared=True, write_report=True)
        assert contract_report["passed"] and contract_report["cross_font_training_pairs"] == 0
        contaminated = list(__import__("csv").DictReader(
            (contract_work / "dataset" / "index.csv").open("r", encoding="utf-8-sig")
        ))
        contaminated[0]["proxy_path"] = str(ref_proxy_dir / "U4E00.png")
        write_csv(
            contract_work / "dataset" / "index.csv", contaminated,
            ["sample_id", "codepoint", "unicode", "char", "split", "mode", "proxy_path",
             "target_path", "target_aux_path", "sample_weight", "structure_score", "complexity"],
        )
        try:
            validate_data_flow_contract(contract_cfg, require_prepared=True, write_report=False)
        except DataFlowContractError:
            pass
        else:
            raise AssertionError("The data-flow contract accepted ref data in the training index.")

    generator = FontStyleNetFinal(base=2).eval()
    x = torch.from_numpy(np.moveaxis(proxy10, -1, 0))[None].float()
    target_aux = torch.from_numpy(np.moveaxis(target_aux_np, -1, 0))[None].float()
    target = target_aux[:, 0:1]
    with torch.no_grad():
        logits = generator(x)
    assert logits.shape == (1, 4, 64, 64)
    ink_logits, sdf_logits, skeleton_logits, edge_logits = split_prediction(logits)
    assert all(head is not None for head in (ink_logits, sdf_logits, skeleton_logits, edge_logits))
    loss, pieces = FontLossFinal()(logits, target, content_proxy=x, target_aux=target_aux)
    assert torch.isfinite(loss)
    for key in ("proxy_skeleton_loss", "sdf_loss", "skeleton_head_loss", "topology_point_loss", "style_signature_loss"):
        assert key in pieces and torch.isfinite(pieces[key])
    vq_loss, vq_pieces = VQReconstructionLoss()(logits, target, target_aux=target_aux)
    assert torch.isfinite(vq_loss)
    for key in ("sdf_loss", "skeleton_head_loss", "edge_head_loss"):
        assert key in vq_pieces and torch.isfinite(vq_pieces[key])

    # Regression: historical VQ checkpoints may contain foreach/fused backend
    # flags and moment tensors with a different memory format. Restore them into
    # the stable single-tensor AdamW path and verify that an update succeeds.
    legacy_model = torch.nn.Conv2d(3, 4, 3, padding=1).to(memory_format=torch.channels_last)
    legacy_optimizer = torch.optim.AdamW(legacy_model.parameters(), foreach=True)
    legacy_input = torch.randn(2, 3, 8, 8).to(memory_format=torch.channels_last)
    legacy_optimizer.zero_grad(set_to_none=True)
    legacy_model(legacy_input).square().mean().backward()
    legacy_optimizer.step()
    legacy_optimizer_state = legacy_optimizer.state_dict()
    for group in legacy_optimizer_state["param_groups"]:
        group["fused"] = True
        group["foreach"] = True
        group["capturable"] = True

    restored_model = torch.nn.Conv2d(3, 4, 3, padding=1).to(memory_format=torch.channels_last)
    restored_optimizer, fused_backend = _create_vq_optimizer(
        restored_model.parameters(),
        learning_rate=1e-3,
        weight_decay=1e-5,
        request_fused=True,
        device=torch.device("cpu"),
    )
    assert not fused_backend
    restored_optimizer.load_state_dict(
        _prepare_vq_optimizer_state_dict(legacy_optimizer_state, fused=fused_backend)
    )
    _restore_vq_optimizer_backend(restored_optimizer, fused=fused_backend)
    _repair_vq_optimizer_state_devices(restored_optimizer)
    assert all(group.get("fused") is False for group in restored_optimizer.param_groups)
    assert all(group.get("foreach") is False for group in restored_optimizer.param_groups)
    assert not getattr(restored_optimizer, "_step_supports_amp_scaling", False)
    restored_optimizer.zero_grad(set_to_none=True)
    restored_model(legacy_input).square().mean().backward()
    restored_optimizer.step()

    generator_ink = torch.sigmoid(ink_logits)
    refiner = GlyphRefinerFinal(base=2).eval()
    with torch.no_grad():
        refined = refiner(torch.cat([generator_ink, x], dim=1))
    assert refined.shape == (1, 4, 64, 64)
    assert torch.isfinite(refined).all()

    discriminator = PatchDiscriminatorFinal(base=4).eval()
    with torch.no_grad():
        score, features = discriminator(generator_ink, return_features=True)
    assert score.ndim == 4 and features
    run_fusion_selftest()
    print("HanziStyleForge Fusion self-test: OK")


if __name__ == "__main__":
    main()
