from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from fontTools.ttLib import TTFont
from tqdm import tqdm

from .charset import is_han_ideograph, select_target_codepoints
from .config import CHECKPOINT_FORMAT_VERSION, validate_config
from .contract import validate_data_flow_contract
from .features import save_target_aux
from .proxy import (
    calibrate_same_structure_thresholds,
    glyph_quality_metrics,
    make_content_proxy,
    read_ink,
    read_proxy,
    robust_location_scale,
    save_proxy,
)
from .render import FontRenderer
from .report import make_audit_contact_sheet
from .retrieval import build_style_atlas
from .util import (
    cp_filename,
    cp_to_char,
    ensure_dir,
    save_codepoints,
    save_json,
    sha256_file,
    write_csv,
)


# The target audit intentionally contains no automatic target/ref same-shape classification.
TARGET_FIELDS = [
    "codepoint",
    "unicode",
    "char",
    "has_target",
    "locl_sensitive",
    "preliminary_status",
    "structure_score",
    "chamfer",
    "dice_distance",
    "grid_distance",
    "projection_distance",
    "component_delta",
    "hole_delta",
    "topology_score",
    "endpoint_delta",
    "junction_delta",
    "reference_endpoints",
    "reference_junctions",
    "target_endpoints",
    "target_junctions",
    "topology_exact",
    "target_ink_ratio",
    "target_border_ink",
    "target_components",
    "target_holes",
    "target_stroke_radius",
    "reference_ink_ratio",
    "reference_components",
    "reference_holes",
    "complexity",
    "ref_path",
    "target_path",
    "ref_proxy_path",
    "target_proxy_path",
    "target_aux_path",
    "notes",
]

STYLE_FIELDS = [
    "codepoint",
    "unicode",
    "char",
    "trainable",
    "ink_ratio",
    "border_ink",
    "components",
    "holes",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "center_x",
    "center_y",
    "stroke_radius",
    "complexity",
    "target_path",
    "proxy_path",
    "aux_path",
    "notes",
]

DATASET_FIELDS = [
    "sample_id",
    "codepoint",
    "unicode",
    "char",
    "split",
    "mode",
    "proxy_path",
    "target_path",
    "target_aux_path",
    "sample_weight",
    "structure_score",
    "complexity",
]


def inspect_font(font_path: str | Path) -> dict[str, Any]:
    path = Path(font_path).resolve()
    font = TTFont(str(path), lazy=False)
    try:
        cmap = font.getBestCmap() or {}
        return {
            "path": str(path),
            "units_per_em": int(font["head"].unitsPerEm),
            "glyph_count": len(font.getGlyphOrder()),
            "unicode_count": len(cmap),
            "han_unicode_count": sum(is_han_ideograph(cp, True) for cp in cmap),
            "non_han_unicode_count": sum(not is_han_ideograph(cp, True) for cp in cmap),
            "outline_type": (
                "TrueType glyf" if "glyf" in font
                else "CFF2" if "CFF2" in font
                else "CFF" if "CFF " in font
                else "unknown"
            ),
            "is_variable": "fvar" in font,
            "tables": list(font.keys()),
        }
    finally:
        font.close()


def check_environment(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_config(cfg)
    target = inspect_font(cfg["paths"]["target_font"])
    reference = inspect_font(cfg["paths"]["reference_font"])
    if target["outline_type"] != "TrueType glyf":
        raise RuntimeError(
            "Automatic font writing requires target.ttf to be a static TrueType glyf font; "
            f"current outline type: {target['outline_type']}."
        )
    if target["is_variable"]:
        raise RuntimeError("Variable fonts cannot be modified directly. Export a static instance first.")
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        torch_info = {"error": str(exc), "cuda_available": False}
    requested_device = str(cfg.get("training", {}).get("device", "cuda")).lower()
    if requested_device.startswith("cuda") and not bool(torch_info.get("cuda_available", False)):
        raise RuntimeError(
            "The current long-run configuration requires CUDA, but PyTorch did not detect an available NVIDIA GPU. "
            "Run install_cuda130.bat and verify that the NVIDIA driver is working."
        )
    if str(cfg.get("build", {}).get("replace_strategy", "new_glyph_and_remap")).lower() in {
        "new_glyph_and_remap", "adaptive_safe"
    }:
        estimated = int(target["glyph_count"]) + int(reference["han_unicode_count"])
        if estimated > 65535:
            raise RuntimeError(
                f"target glyph count {target['glyph_count']} + reference Han glyph count {reference['han_unicode_count']} "
                f"is approximately {estimated}, which exceeds the OpenType limit of 65,535 glyphs. "
                "Use a reference font with a smaller character set that still meets your requirements."
            )
    return {
        "program": "HanziStyleForge Fusion",
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "target": target,
        "reference": reference,
        "torch": torch_info,
        "runtime": cfg.get("_runtime", {}),
    }


def _unwrap_subtable(subtable: Any) -> Any:
    current = subtable
    for _ in range(4):
        extension = getattr(current, "ExtSubTable", None)
        if extension is None:
            break
        current = extension
    return current


def extract_locl_sensitive_codepoints(font_path: str | Path) -> set[int]:
    """Return a report-only set of codepoints touched by a locl feature.

    The result is never used to decide training membership, keeping or
    replacement.  All reference Han characters are rebuilt regardless.
    """
    font = TTFont(str(font_path), lazy=False)
    try:
        if "GSUB" not in font:
            return set()
        table = font["GSUB"].table
        if table.FeatureList is None or table.LookupList is None:
            return set()
        indices: set[int] = set()
        for record in table.FeatureList.FeatureRecord:
            if record.FeatureTag == "locl":
                indices.update(int(index) for index in record.Feature.LookupListIndex)
        glyphs: set[str] = set()
        for index in indices:
            if index >= len(table.LookupList.Lookup):
                continue
            for raw in table.LookupList.Lookup[index].SubTable:
                subtable = _unwrap_subtable(raw)
                mapping = getattr(subtable, "mapping", None)
                if isinstance(mapping, dict):
                    glyphs.update(str(value) for value in mapping)
                    glyphs.update(str(value) for value in mapping.values())
                alternates = getattr(subtable, "alternates", None)
                if isinstance(alternates, dict):
                    glyphs.update(str(value) for value in alternates)
                    for values in alternates.values():
                        glyphs.update(str(value) for value in values)
        cmap = font.getBestCmap() or {}
        return {cp for cp, glyph_name in cmap.items() if glyph_name in glyphs}
    finally:
        font.close()


def _render_and_proxy(
    renderer: FontRenderer,
    codepoint: int,
    render_path: Path,
    proxy_path: Path,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if not render_path.is_file():
        renderer.save_png(codepoint, render_path)
    ink = read_ink(render_path)
    if not proxy_path.is_file():
        render_cfg = cfg["render"]
        proxy = make_content_proxy(
            ink,
            output_size=int(render_cfg["size"]),
            skeleton_size=int(render_cfg["proxy_skeleton_size"]),
            threshold=float(render_cfg["threshold"]),
            canonical_radius_ratio=float(render_cfg["canonical_radius_ratio"]),
            distance_clip_ratio=float(render_cfg["distance_clip_ratio"]),
        )
        save_proxy(proxy_path, proxy)
    else:
        proxy = read_proxy(proxy_path)
    return ink, proxy


def _split(codepoints: list[int], validation_ratio: float, seed: int) -> dict[int, str]:
    values = list(codepoints)
    random.Random(int(seed)).shuffle(values)
    validation_count = max(1, int(round(len(values) * validation_ratio))) if len(values) > 1 else 0
    validation = set(values[:validation_count])
    return {cp: ("val" if cp in validation else "train") for cp in values}


def _style_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "ink_ratio", "bbox_width_ratio", "bbox_height_ratio",
        "center_x", "center_y", "stroke_radius",
    )
    profile: dict[str, Any] = {"sample_count": len(rows)}
    for key in keys:
        median, sigma = robust_location_scale([float(row[key]) for row in rows])
        profile[key] = {"median": median, "sigma": sigma}
    return profile


def _conditional_style_profiles(rows: list[dict[str, Any]], bin_count: int = 10) -> dict[str, Any]:
    global_profile = _style_profile(rows)
    if len(rows) < max(80, bin_count * 12):
        return {"global": global_profile, "edges": [], "bins": []}
    complexities = np.asarray([float(row["complexity"]) for row in rows], dtype=np.float64)
    edges = np.unique(np.quantile(complexities, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) < 3:
        return {"global": global_profile, "edges": [], "bins": []}
    bins: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        lo, hi = float(edges[index]), float(edges[index + 1])
        selected = [
            row for row in rows
            if float(row["complexity"]) >= lo
            and (float(row["complexity"]) <= hi if index == len(edges) - 2 else float(row["complexity"]) < hi)
        ]
        if selected:
            bins.append({"minimum": lo, "maximum": hi, "profile": _style_profile(selected)})
    return {"global": global_profile, "edges": [float(value) for value in edges], "bins": bins}


def _quality_valid(metrics: dict[str, Any], cfg: dict[str, Any]) -> bool:
    analysis = cfg["analysis"]
    return (
        float(analysis["minimum_ink_ratio"]) <= float(metrics["ink_ratio"]) <= float(analysis["maximum_ink_ratio"])
        and float(metrics["border_ink"]) <= float(analysis["maximum_border_ink"])
        and int(metrics["components"]) > 0
        and float(metrics["stroke_radius"]) > 0
    )


def prepare_project(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    validate_config(cfg)
    work = ensure_dir(cfg["paths"]["work_dir"])
    audit_dir = ensure_dir(work / "audit")
    dataset_dir = ensure_dir(work / "dataset")
    cache_dir = ensure_dir(work / "cache")
    ref_render_dir = ensure_dir(cache_dir / "ref_render")
    ref_proxy_dir = ensure_dir(cache_dir / "ref_proxy")
    target_render_dir = ensure_dir(cache_dir / "target_render")
    target_proxy_dir = ensure_dir(cache_dir / "target_proxy")
    target_aux_dir = ensure_dir(cache_dir / "target_aux")
    panels_dir = ensure_dir(audit_dir / "panels")

    target_path = Path(cfg["paths"]["target_font"])
    reference_path = Path(cfg["paths"]["reference_font"])
    fingerprint = {
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "method": "style-only-self-reconstruction-no-cross-font-pairs",
        "target_sha256": sha256_file(target_path),
        "reference_sha256": sha256_file(reference_path),
        "scope": cfg["scope"],
        "render": cfg["render"],
        "analysis": cfg["analysis"],
        "retrieval": cfg.get("retrieval", {}),
        "topology": cfg.get("topology", {}),
    }
    fingerprint_path = audit_dir / "input_fingerprint.json"
    summary_path = audit_dir / "summary.json"
    if (
        not force
        and not bool(cfg["analysis"].get("force_reprepare", False))
        and fingerprint_path.is_file()
        and summary_path.is_file()
        and json.loads(fingerprint_path.read_text(encoding="utf-8")) == fingerprint
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    target_info = inspect_font(target_path)
    if target_info["outline_type"] != "TrueType glyf" or target_info["is_variable"]:
        raise RuntimeError("target.ttf must be a static TrueType glyf font.")

    render_cfg = cfg["render"]
    target_renderer = FontRenderer(
        target_path,
        size=int(render_cfg["size"]),
        pad=int(render_cfg["pad"]),
        antialias=int(render_cfg["antialias"]),
    )
    reference_renderer = FontRenderer(
        reference_path,
        size=int(render_cfg["size"]),
        pad=int(render_cfg["pad"]),
        antialias=int(render_cfg["antialias"]),
    )

    try:
        include_compatibility = bool(cfg["scope"].get("include_compatibility_ideographs", True))
        target_cps = select_target_codepoints(
            reference_renderer.cmap,
            mode=cfg["scope"]["mode"],
            include_compatibility=include_compatibility,
            extra_chars_file=cfg["scope"].get("extra_chars_file", ""),
        )
        all_style_cps = sorted(
            cp for cp in target_renderer.cmap
            if is_han_ideograph(cp, include_compatibility)
        )
        style_cps = list(all_style_cps)
        maximum_style = int(cfg["analysis"].get("maximum_style_glyphs", 0))
        if maximum_style > 0 and len(style_cps) > maximum_style:
            # Deterministic coverage across the ordered Unicode range; this is
            # used only by the packaged fast test configuration.
            positions = np.linspace(0, len(style_cps) - 1, maximum_style, dtype=np.int64)
            style_cps = [style_cps[int(index)] for index in positions]
        target_all_cps = set(target_renderer.cmap)
        target_set = set(target_cps)
        all_style_set = set(all_style_cps)
        non_han_cps = sorted(target_all_cps - all_style_set)
        target_han_outside_reference = sorted(all_style_set - target_set)

        save_codepoints(audit_dir / "target_han_from_reference.txt", target_cps)
        save_codepoints(audit_dir / "target_han_style_source.txt", style_cps)
        save_codepoints(audit_dir / "target_han_outside_reference_preserved.txt", target_han_outside_reference)
        save_codepoints(audit_dir / "non_han_preserved.txt", non_han_cps)

        style_rows: list[dict[str, Any]] = []
        valid_style_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        target_paths: dict[int, tuple[str, str, str]] = {}
        calibration: list[np.ndarray] = []
        maximum_calibration = int(cfg["analysis"].get("calibration_samples", 2048))
        rng = random.Random(int(cfg["training"]["seed"]))

        for cp in tqdm(style_cps, desc="Extracting target Han style", unit="glyph"):
            filename = cp_filename(cp)
            render_path = target_render_dir / filename
            proxy_path = target_proxy_dir / filename
            aux_path = target_aux_dir / filename
            ink, proxy = _render_and_proxy(target_renderer, cp, render_path, proxy_path, cfg)
            if not aux_path.is_file():
                save_target_aux(aux_path, ink, threshold=float(render_cfg["threshold"]))
            metrics = glyph_quality_metrics(ink, threshold=float(render_cfg["threshold"]))
            complexity = float(proxy[..., 1].mean())
            valid = _quality_valid(metrics, cfg)
            row: dict[str, Any] = {
                "codepoint": cp,
                "unicode": f"U+{cp:04X}",
                "char": cp_to_char(cp),
                "trainable": int(valid),
                **metrics,
                "complexity": complexity,
                "target_path": str(render_path.resolve()),
                "proxy_path": str(proxy_path.resolve()),
                "aux_path": str(aux_path.resolve()),
                "notes": "" if valid else "Empty glyph, border contact, or abnormal outline; excluded from style training.",
            }
            style_rows.append(row)
            target_paths[cp] = (
                str(render_path.resolve()), str(proxy_path.resolve()), str(aux_path.resolve())
            )
            if valid:
                valid_style_rows.append(row)
                profile_rows.append({**metrics, "complexity": complexity})
                if len(calibration) < maximum_calibration:
                    calibration.append(proxy)
                else:
                    index = rng.randrange(len(valid_style_rows))
                    if index < maximum_calibration:
                        calibration[index] = proxy

        if len(valid_style_rows) < 256:
            raise RuntimeError(
                f"Only {len(valid_style_rows)} valid target Han glyphs were found. Reliable style learning requires at least 1,000 recommended glyphs."
            )

        split_map = _split(
            [int(row["codepoint"]) for row in valid_style_rows],
            validation_ratio=float(cfg["training"]["validation_ratio"]),
            seed=int(cfg["training"]["seed"]),
        )
        dataset_rows: list[dict[str, Any]] = []
        for row in valid_style_rows:
            cp = int(row["codepoint"])
            dataset_rows.append({
                "sample_id": f"style-self-{cp:06X}",
                "codepoint": cp,
                "unicode": row["unicode"],
                "char": row["char"],
                "split": split_map.get(cp, "train"),
                "mode": "self",
                "proxy_path": row["proxy_path"],
                "target_path": row["target_path"],
                "target_aux_path": row["aux_path"],
                "sample_weight": 1.0,
                "structure_score": 0.0,
                "complexity": row["complexity"],
            })

        locl_sensitive = extract_locl_sensitive_codepoints(reference_path)
        target_rows: list[dict[str, Any]] = []
        for cp in tqdm(target_cps, desc="Preparing reference Han structures", unit="glyph"):
            filename = cp_filename(cp)
            ref_path = ref_render_dir / filename
            ref_proxy_path = ref_proxy_dir / filename
            ref_ink, ref_proxy = _render_and_proxy(reference_renderer, cp, ref_path, ref_proxy_path, cfg)
            reference_metrics = glyph_quality_metrics(ref_ink, threshold=float(render_cfg["threshold"]))
            complexity = float(ref_proxy[..., 1].mean())
            has_target = cp in target_paths
            target_path_value, target_proxy_value, target_aux_value = target_paths.get(cp, ("", "", ""))
            target_metrics: dict[str, Any] = {}
            if has_target:
                try:
                    target_metrics = glyph_quality_metrics(
                        read_ink(target_path_value), threshold=float(render_cfg["threshold"])
                    )
                except Exception:
                    target_metrics = {}
            target_rows.append({
                "codepoint": cp,
                "unicode": f"U+{cp:04X}",
                "char": cp_to_char(cp),
                "has_target": int(has_target),
                "locl_sensitive": int(cp in locl_sensitive),
                "preliminary_status": "rebuild_existing" if has_target else "add_missing",
                "structure_score": "",
                "chamfer": "",
                "dice_distance": "",
                "grid_distance": "",
                "projection_distance": "",
                "component_delta": "",
                "hole_delta": "",
                "topology_score": "",
                "endpoint_delta": "",
                "junction_delta": "",
                "reference_endpoints": "",
                "reference_junctions": "",
                "target_endpoints": "",
                "target_junctions": "",
                "topology_exact": 0,
                "target_ink_ratio": target_metrics.get("ink_ratio", ""),
                "target_border_ink": target_metrics.get("border_ink", ""),
                "target_components": target_metrics.get("components", ""),
                "target_holes": target_metrics.get("holes", ""),
                "target_stroke_radius": target_metrics.get("stroke_radius", ""),
                "reference_ink_ratio": reference_metrics["ink_ratio"],
                "reference_components": reference_metrics["components"],
                "reference_holes": reference_metrics["holes"],
                "complexity": complexity,
                "ref_path": str(ref_path.resolve()),
                "target_path": target_path_value,
                "ref_proxy_path": str(ref_proxy_path.resolve()),
                "target_proxy_path": target_proxy_value,
                "target_aux_path": target_aux_value,
                "notes": "Every Han glyph covered by ref is rebuilt. This table does not classify target/ref glyph equivalence.",
            })
            if len(calibration) < maximum_calibration:
                calibration.append(ref_proxy)

        thresholds = calibrate_same_structure_thresholds(
            calibration,
            seed=int(cfg["training"]["seed"]),
            analysis_size=int(render_cfg["analysis_size"]),
        )
        save_json(audit_dir / "structure_thresholds.json", thresholds)
        save_json(audit_dir / "synthetic_structure_thresholds.json", thresholds)

        style_profiles = _conditional_style_profiles(profile_rows, bin_count=10)
        save_json(dataset_dir / "style_profile.json", style_profiles["global"])
        save_json(dataset_dir / "style_profiles.json", style_profiles)
        write_csv(audit_dir / "style_source.csv", style_rows, STYLE_FIELDS)
        write_csv(audit_dir / "analysis.csv", target_rows, TARGET_FIELDS)
        write_csv(dataset_dir / "index.csv", dataset_rows, DATASET_FIELDS)
        data_flow = validate_data_flow_contract(cfg, require_prepared=True, write_report=True)
        save_codepoints(audit_dir / "style_trainable.txt", [int(row["codepoint"]) for row in valid_style_rows])
        save_codepoints(audit_dir / "target_existing_rebuilt.txt", [int(row["codepoint"]) for row in target_rows if int(row["has_target"])])
        save_codepoints(audit_dir / "target_missing_added.txt", [int(row["codepoint"]) for row in target_rows if not int(row["has_target"])])

        dataset_stats = {
            "method": "style-only self-reconstruction",
            "cross_font_pair_rows": 0,
            "style_samples": len(valid_style_rows),
            "dataset_rows": len(dataset_rows),
            "train_rows": sum(row["split"] == "train" for row in dataset_rows),
            "validation_rows": sum(row["split"] == "val" for row in dataset_rows),
            "style_profile": style_profiles["global"],
            "thresholds": thresholds,
        }
        save_json(dataset_dir / "stats.json", dataset_stats)

        atlas_summary = build_style_atlas(
            cfg,
            dataset_csv=dataset_dir / "index.csv",
            force=force or bool(cfg["analysis"].get("force_reprepare", False)),
        )

        panel_count = int(cfg["analysis"].get("panel_count", 320))
        make_audit_contact_sheet(
            [
                {
                    "codepoint": row["codepoint"], "char": row["char"],
                    "target_path": row["target_path"], "ref_path": "",
                }
                for row in valid_style_rows[:panel_count]
            ],
            panels_dir / "style_training_sample.png",
            cell_size=112,
            rows_per_page=20,
        )
        make_audit_contact_sheet(
            target_rows[:panel_count],
            panels_dir / "ref_targets_sample.png",
            cell_size=112,
            rows_per_page=20,
        )
        make_audit_contact_sheet(
            [row for row in target_rows if not int(row["has_target"])][:160],
            panels_dir / "missing_targets_sample.png",
            cell_size=112,
            rows_per_page=20,
        )

        summary = {
            "program": "HanziStyleForge Fusion",
            "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
            "method": "all valid target Han self-reconstruction; ref content only; no cross-font pairs",
            "automatic_reference_classification": False,
            "cross_font_pair_count": 0,
            "target_style_source_han_count": len(all_style_cps),
            "style_scanned_count": len(style_cps),
            "style_trainable_count": len(valid_style_rows),
            "style_training_coverage": len(valid_style_rows) / max(1, len(style_cps)),
            "target_han_count": len(target_cps),
            "target_existing_rebuild_count": sum(int(row["has_target"]) for row in target_rows),
            "target_missing_add_count": sum(not int(row["has_target"]) for row in target_rows),
            "target_han_outside_reference_preserved_count": len(target_han_outside_reference),
            "non_han_preserved_count": len(non_han_cps),
            "reference_locl_sensitive_report_only_count": len(locl_sensitive & target_set),
            "dataset_row_count": len(dataset_rows),
            "style_profile": style_profiles["global"],
            "style_atlas": atlas_summary,
            "data_flow_contract": data_flow,
            "important_note": (
                "The program does not compare target.ttf with ref.otf to classify equivalent forms or regional standards. "
                "All valid target Han glyphs teach style only; every Han glyph covered by ref.otf is regenerated from reference content structure. "
                "All non-Han codepoints and outlines in target are preserved and verified during font building."
            ),
        }
        save_json(summary_path, summary)
        save_json(fingerprint_path, fingerprint)
        return summary
    finally:
        target_renderer.close()
        reference_renderer.close()
