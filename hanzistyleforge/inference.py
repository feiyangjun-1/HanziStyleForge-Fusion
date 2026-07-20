from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ReferenceProxyDataset
from .config import CHECKPOINT_FORMAT_VERSION
from .contract import validate_data_flow_contract
from .longrun import LongRunGuard
from .features import ink_probability, split_prediction
from .proxy import (
    binary,
    border_ink_ratio,
    component_hole_counts,
    glyph_quality_metrics,
    make_reference_fallbacks,
    proxy_from_binary_for_metrics,
    proxy_structure_metrics,
    read_ink,
    read_proxy,
    save_ink,
    style_distance,
)
from .retrieval import StyleAtlas, render_retrieval_candidate
from .topology import structure_lock_probability, topology_metrics, topology_signature, validate_topology
from .training import load_generator, load_refiner
from .util import (
    cp_filename,
    ensure_dir,
    load_json,
    read_csv,
    save_codepoints,
    save_json,
    sha256_file,
    write_csv,
)


SELECTION_FIELDS = [
    "codepoint",
    "unicode",
    "char",
    "has_target",
    "locl_sensitive",
    "preliminary_status",
    "final_action",
    "chosen_source",
    "chosen_label",
    "chosen_path",
    "ref_path",
    "target_path",
    "ref_proxy_path",
    "target_structure_score",
    "neural_structure_score",
    "retrieval_structure_score",
    "fusion_structure_score",
    "fallback_structure_score",
    "chosen_structure_score",
    "neural_topology_score",
    "retrieval_topology_score",
    "fusion_topology_score",
    "fallback_topology_score",
    "chosen_topology_score",
    "chosen_topology_pass",
    "chosen_component_delta",
    "chosen_hole_delta",
    "chosen_endpoint_delta",
    "chosen_junction_delta",
    "neural_style_score",
    "retrieval_style_score",
    "fusion_style_score",
    "fallback_style_score",
    "neural_confidence",
    "retrieval_confidence",
    "fusion_confidence",
    "chosen_confidence",
    "pseudo_eligible",
    "retrieval_sources",
    "rejection_reasons",
    "notes",
]


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["training"].get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA, but torch.cuda.is_available() is False.")
    if not requested.startswith("cuda"):
        torch.set_num_threads(max(1, min(4, int(cfg["training"].get("cpu_threads", 4)))))
    return torch.device(requested if requested.startswith("cuda") else "cpu")


def _style_profile_for_complexity(profiles: dict[str, Any], complexity: float) -> dict[str, Any]:
    global_profile = profiles.get("global", profiles)
    bins = profiles.get("bins", []) if isinstance(profiles, dict) else []
    for item in bins:
        lo = float(item.get("minimum", -math.inf))
        hi = float(item.get("maximum", math.inf))
        if lo <= complexity <= hi:
            return item.get("profile", global_profile)
    return global_profile


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _ensemble_generator_paths(
    work: Path,
    main_path: Path,
    cfg: dict[str, Any],
    *,
    explicit_checkpoint: bool,
) -> list[Path]:
    """Return main + diverse Marathon checkpoints for candidate generation.

    Explicit checkpoints are isolated diagnostic runs; the standard final run
    uses the ensemble manifest.
    """

    paths = [main_path]
    ensemble_cfg = cfg.get("marathon", {}).get("ensemble", {})
    if explicit_checkpoint or not bool(ensemble_cfg.get("enabled", True)) or not bool(
        ensemble_cfg.get("use_for_inference", True)
    ):
        return paths
    manifest_path = work / "model" / "ensemble" / "manifest.json"
    if not manifest_path.is_file():
        return paths
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return paths
    for item in manifest.get("models", []):
        candidate = Path(item.get("checkpoint", ""))
        if candidate.is_file():
            paths.append(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique[: max(1, int(ensemble_cfg.get("maximum_models", 5)))]


def _mean_optional_heads(
    outputs: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]],
    index: int,
) -> torch.Tensor | None:
    values = [item[index] for item in outputs if item[index] is not None]
    return torch.stack(values, dim=0).mean(dim=0) if values else None


def _probability_heads(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    ink_logits, sdf_logits, skeleton_logits, edge_logits = split_prediction(output)
    return (
        torch.sigmoid(ink_logits),
        torch.sigmoid(sdf_logits) if sdf_logits is not None else None,
        torch.sigmoid(skeleton_logits) if skeleton_logits is not None else None,
        torch.sigmoid(edge_logits) if edge_logits is not None else None,
    )


def _generator_tta(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    variants = [inputs]
    if enabled:
        dilated = inputs.clone()
        eroded = inputs.clone()
        dilated[:, 0:1] = torch.nn.functional.max_pool2d(inputs[:, 0:1], 3, 1, 1)
        eroded[:, 0:1] = -torch.nn.functional.max_pool2d(-inputs[:, 0:1], 3, 1, 1)
        variants.extend([dilated, eroded])
    collected: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]] = []
    for variant in variants:
        collected.append(_probability_heads(model(variant)))
    result: list[torch.Tensor | None] = []
    for head in range(4):
        values = [item[head] for item in collected if item[head] is not None]
        result.append(torch.stack(values, dim=0).mean(dim=0) if values else None)
    assert result[0] is not None
    return result[0], result[1], result[2], result[3]


def _multitask_candidates(
    ink: np.ndarray,
    sdf: np.ndarray | None,
    skeleton: np.ndarray | None,
    threshold: float,
    target_radius: float,
    profile_size: int,
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    sdf_soft = None
    if sdf is not None:
        candidates.append(("neural_sdf_zero", (sdf >= 0.5).astype(np.float32)))
        sdf_soft = 1.0 / (1.0 + np.exp(-np.clip((sdf - 0.5) * 14.0, -20.0, 20.0)))
    skeleton_round = None
    if skeleton is not None:
        radius = max(1, min(12, int(round(float(target_radius) * ink.shape[0] / max(1, int(profile_size))))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        skeleton_round = cv2.dilate((skeleton >= 0.46).astype(np.uint8), kernel).astype(np.float32)
        candidates.append(("neural_skeleton_round", skeleton_round))
    if sdf_soft is not None or skeleton_round is not None:
        consensus = 0.64 * ink
        denominator = 0.64
        if sdf_soft is not None:
            consensus += 0.25 * sdf_soft
            denominator += 0.25
        if skeleton_round is not None:
            consensus += 0.11 * skeleton_round
            denominator += 0.11
        consensus /= denominator
        candidates.append(("neural_multitask_consensus", (consensus >= threshold).astype(np.float32)))
    return candidates

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    src = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    minimum = max(2, int(src.shape[0] * src.shape[1] * 0.00002))
    result = np.zeros_like(src)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            result[labels == index] = 1
    return result


def _quick_candidate_score(mask: np.ndarray, reference_proxy: np.ndarray, analysis_size: int) -> dict[str, Any]:
    cleaned = _clean_mask(mask)
    small = cv2.resize(cleaned.astype(np.float32), (analysis_size, analysis_size), interpolation=cv2.INTER_NEAREST)
    candidate_proxy = proxy_from_binary_for_metrics(small)
    structure = proxy_structure_metrics(candidate_proxy, reference_proxy, analysis_size=analysis_size)
    components, holes = component_hole_counts(small, minimum_area=max(2, analysis_size // 64))
    border = border_ink_ratio(cleaned, border=max(2, cleaned.shape[0] // 128))
    return {
        "mask": cleaned.astype(np.float32),
        "structure": structure,
        "components": components,
        "holes": holes,
        "border": border,
        "ink_ratio": float(cleaned.mean()),
    }


def _style_metrics_at_profile_scale(
    mask: np.ndarray,
    profile: dict[str, Any],
    profile_size: int,
) -> tuple[dict[str, Any], float]:
    analysis = min(160, mask.shape[0])
    small = cv2.resize(mask.astype(np.float32), (analysis, analysis), interpolation=cv2.INTER_AREA)
    metrics = glyph_quality_metrics(small, threshold=0.5)
    metrics["stroke_radius"] = float(metrics["stroke_radius"]) * profile_size / analysis
    return metrics, style_distance(metrics, profile)


def _evaluate_family(
    source: str,
    candidates: list[tuple[str, np.ndarray]],
    reference_proxy: np.ndarray,
    reference_ink: np.ndarray,
    profile: dict[str, Any],
    analysis_size: int,
    profile_size: int,
    maximum_border: float,
    topology_cfg: dict[str, Any],
    source_weight: float,
    reference_signature: Any | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError(f"Candidate family {source} is empty.")
    quick: list[dict[str, Any]] = []
    for label, mask in candidates:
        item = _quick_candidate_score(mask, reference_proxy, analysis_size)
        item["source"] = source
        item["label"] = label
        structure_score = float(item["structure"]["structure_score"])
        topology_penalty = 0.055 * min(4, int(item["structure"]["component_delta"])) + 0.055 * min(
            4, int(item["structure"]["hole_delta"])
        )
        border_penalty = max(0.0, float(item["border"]) - maximum_border) * 10.0
        item["quick_total"] = structure_score + topology_penalty + border_penalty
        quick.append(item)
    quick.sort(key=lambda item: (float(item["quick_total"]), str(item["label"])))
    finalists = quick[: min(max(1, int(topology_cfg.get("finalists_per_family", 2))), len(quick))]
    for item in finalists:
        topo = topology_metrics(
            reference_ink,
            item["mask"],
            size=int(topology_cfg.get("analysis_size", analysis_size)),
            prune_iterations=int(topology_cfg.get("prune_iterations", 1)),
            reference_signature=reference_signature,
        )
        validation = validate_topology(topo, topology_cfg)
        style_metrics, style_score_value = _style_metrics_at_profile_scale(item["mask"], profile, profile_size)
        item["topology"] = topo
        item["validation"] = validation
        item["style_metrics"] = style_metrics
        item["style_score"] = float(style_score_value)
        hard_penalty = 0.0 if validation["hard_pass"] else 0.25 + 0.04 * len(validation["reasons"])
        item["total_score"] = float(source_weight) * (
            float(item["quick_total"])
            + 0.62 * float(topo["topology_score"])
            + 0.075 * float(style_score_value)
            + hard_penalty
        )
    passing = [item for item in finalists if bool(item["validation"]["hard_pass"])]
    pool = passing if passing else finalists
    return min(pool, key=lambda item: (float(item["total_score"]), str(item["label"])))


def _threshold_candidates(
    probability: np.ndarray,
    threshold: float,
    offsets: list[float],
    prefix: str,
) -> list[tuple[str, np.ndarray]]:
    result: list[tuple[str, np.ndarray]] = []
    for offset in offsets:
        value = float(np.clip(threshold + float(offset), 0.20, 0.80))
        result.append((f"{prefix}_threshold_{value:.3f}", (probability >= value).astype(np.float32)))
    base = (probability >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result.append((f"{prefix}_close", cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel).astype(np.float32)))
    result.append((f"{prefix}_open", cv2.morphologyEx(base, cv2.MORPH_OPEN, kernel).astype(np.float32)))
    return result


def _probability_certainty(probability: np.ndarray) -> float:
    ambiguity = 4.0 * probability * (1.0 - probability)
    region = probability > 0.03
    if not np.any(region):
        return 0.0
    return float(1.0 - np.mean(ambiguity[region]))


def _confidence(candidate: dict[str, Any], probability: np.ndarray | None, keep_threshold: float) -> float:
    structure = float(candidate["structure"]["structure_score"])
    topology_score = float(candidate["topology"]["topology_score"])
    style = float(candidate.get("style_score", 1.0))
    topology_delta = (
        int(candidate["topology"]["component_delta"])
        + int(candidate["topology"]["hole_delta"])
        + 0.25 * int(candidate["topology"]["endpoint_delta"])
        + 0.25 * int(candidate["topology"]["junction_delta"])
    )
    certainty = _probability_certainty(probability) if probability is not None else 0.86
    structure_term = math.exp(-structure / max(keep_threshold * 1.8, 1e-4))
    topology_term = math.exp(-5.0 * topology_score) * math.exp(-0.45 * topology_delta)
    style_term = math.exp(-1.45 * style)
    hard = 1.0 if candidate["validation"]["hard_pass"] else 0.35
    return float(np.clip(certainty * structure_term * topology_term * style_term * hard, 0.0, 1.0))


def _family_value(item: dict[str, Any], key: str, default: Any = "") -> Any:
    try:
        return item[key]
    except Exception:
        return default


def _generation_fingerprint(
    analysis_path: Path,
    generator_paths: list[Path],
    refiner_path: Path,
    atlas_path: Path,
    cfg: dict[str, Any],
    output_subdir: str,
) -> str:
    """Build a stable fingerprint so stale partial results are never reused."""

    payload = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "analysis": sha256_file(analysis_path),
        "generators": [sha256_file(path) for path in generator_paths],
        "refiner": sha256_file(refiner_path) if refiner_path.exists() else "",
        "atlas": sha256_file(atlas_path) if atlas_path.exists() else "",
        # The output directory name is intentionally excluded. An accepted
        # adaptation round is renamed from generated_adapt_* to generated; the
        # glyph results remain valid as long as the model/data/config hashes do.
        "render": cfg.get("render", {}),
        "inference": cfg.get("inference", {}),
        "topology": cfg.get("topology", {}),
        "retrieval": cfg.get("retrieval", {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_partial(
    partial_path: Path,
    state_path: Path,
    selection_rows: list[dict[str, Any]],
    fingerprint: str,
    target_count: int,
) -> None:
    """Atomically save resumable generation progress."""

    write_csv(partial_path, selection_rows, SELECTION_FIELDS)
    save_json(
        state_path,
        {
            "version": CHECKPOINT_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "completed_count": len(selection_rows),
            "target_count": target_count,
        },
    )



def _emergency_fallback_row(
    row: dict[str, Any],
    cp: int,
    chosen_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    """Create a structurally safe ref-derived fallback when one glyph raises an error.

    A single malformed outline or third-party library failure must not abort a
    30,000-character run.  The raw ref mask is always available after
    prepare, so it is used as the last-resort candidate and clearly recorded in
    QA/build reports.
    """

    reference_ink = read_ink(row["ref_path"])
    chosen_path = chosen_dir / cp_filename(cp)
    save_ink(chosen_path, reference_ink)
    has_target = bool(int(row.get("has_target", 0)))
    target_score = row.get("structure_score", "") if has_target else ""
    message = f"{type(error).__name__}: {error}"
    if len(message) > 600:
        message = message[:597] + "..."
    return {
        "codepoint": cp,
        "unicode": row.get("unicode", f"U+{cp:04X}"),
        "char": row.get("char", ""),
        "has_target": int(has_target),
        "locl_sensitive": int(row.get("locl_sensitive", 0)),
        "preliminary_status": row.get("preliminary_status", "error"),
        "final_action": "replace" if has_target else "add",
        "chosen_source": "fallback",
        "chosen_label": "reference_emergency",
        "chosen_path": str(chosen_path.resolve()),
        "ref_path": row.get("ref_path", ""),
        "target_path": row.get("target_path", ""),
        "ref_proxy_path": row.get("ref_proxy_path", ""),
        "target_structure_score": target_score,
        "neural_structure_score": "",
        "retrieval_structure_score": "",
        "fusion_structure_score": "",
        "fallback_structure_score": 0.0,
        "chosen_structure_score": 0.0,
        "neural_topology_score": "",
        "retrieval_topology_score": "",
        "fusion_topology_score": "",
        "fallback_topology_score": 0.0,
        "chosen_topology_score": 0.0,
        "chosen_topology_pass": 1,
        "chosen_component_delta": 0,
        "chosen_hole_delta": 0,
        "chosen_endpoint_delta": 0,
        "chosen_junction_delta": 0,
        "neural_style_score": "",
        "retrieval_style_score": "",
        "fusion_style_score": "",
        "fallback_style_score": "",
        "neural_confidence": "",
        "retrieval_confidence": "",
        "fusion_confidence": "",
        "chosen_confidence": 0.0,
        "pseudo_eligible": 0,
        "retrieval_sources": "",
        "rejection_reasons": str({"emergency": [message]}),
        "notes": "Per-glyph processing failed; the original reference structural mask was used automatically and the full workflow continues.",
    }

def generate_and_select(
    cfg: dict[str, Any],
    output_subdir: str = "generated",
    generator_checkpoint: str | Path | None = None,
    refiner_checkpoint: str | Path | None = None,
    generator_calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    validate_data_flow_contract(cfg, require_prepared=True, write_report=True)
    work = Path(cfg["paths"]["work_dir"])
    longrun_guard = LongRunGuard(cfg)
    analysis_path = work / "audit" / "analysis.csv"
    if not analysis_path.exists():
        raise FileNotFoundError("audit/analysis.csv was not found. Run prepare first.")
    rows = read_csv(analysis_path)
    profile = load_json(work / "dataset" / "style_profile.json")
    profiles_path = work / "dataset" / "style_profiles.json"
    style_profiles = load_json(profiles_path) if profiles_path.exists() else {"global": profile, "bins": []}
    thresholds = load_json(work / "audit" / "structure_thresholds.json")
    generator_path = Path(generator_checkpoint) if generator_checkpoint else work / "model" / "generator" / "generator_best.pt"
    if not generator_path.exists():
        raise FileNotFoundError("generator_best.pt was not found. Run train first.")

    generated_dir = ensure_dir(work / output_subdir)
    partial_path = generated_dir / "selection.partial.csv"
    state_path = generated_dir / "generation.state.json"
    completion_path = generated_dir / "generation.completed.json"

    device = _device(cfg)
    amp_enabled = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    generator_paths = _ensemble_generator_paths(
        work, generator_path, cfg, explicit_checkpoint=generator_checkpoint is not None
    )
    generators = [load_generator(path, device)[0] for path in generator_paths]
    refiner_path = Path(refiner_checkpoint) if refiner_checkpoint else work / "model" / "refiner" / "refiner_best.pt"
    refiner = None
    if bool(cfg["refiner"].get("enabled", True)) and refiner_path.exists():
        refiner, _ = load_refiner(refiner_path, device)

    calibration_file = Path(generator_calibration_path) if generator_calibration_path else work / "model" / "generator" / "calibration.json"
    generator_calibration = load_json(calibration_file)
    generator_threshold = float(generator_calibration["threshold"])
    refiner_threshold = generator_threshold
    refiner_calibration_path = work / "model" / "refiner" / "calibration.json"
    if refiner is not None and refiner_calibration_path.exists():
        refiner_threshold = float(load_json(refiner_calibration_path)["threshold"])

    inference_cfg = cfg["inference"]
    topology_cfg = cfg.get("topology", {})
    retrieval_cfg = cfg.get("retrieval", {})
    inference_size = int(inference_cfg["size"])
    analysis_size = int(cfg["render"]["analysis_size"])

    atlas_path = work / "retrieval" / "style_atlas.npz"
    fingerprint = _generation_fingerprint(
        analysis_path,
        generator_paths,
        refiner_path,
        atlas_path,
        cfg,
        output_subdir,
    )

    selection_path = generated_dir / "selection.csv"
    summary_path = generated_dir / "summary.json"
    if completion_path.exists() and selection_path.exists() and summary_path.exists():
        try:
            completion = load_json(completion_path)
            if str(completion.get("fingerprint", "")) == fingerprint:
                existing_rows = read_csv(selection_path)
                if len(existing_rows) == len(rows):
                    return load_json(summary_path)
        except Exception:
            pass

    selection_rows: list[dict[str, Any]] = []
    state_valid = False
    if partial_path.exists() and state_path.exists():
        try:
            state = load_json(state_path)
            state_valid = str(state.get("fingerprint", "")) == fingerprint
            if state_valid:
                selection_rows = read_csv(partial_path)
        except Exception:
            state_valid = False
            selection_rows = []

    if not state_valid:
        partial_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        completion_path.unlink(missing_ok=True)
        selection_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        # Old candidates must not survive a model/config change and appear in QA.
        for source in ("neural", "retrieval", "fusion", "fallback", "chosen"):
            path = generated_dir / source
            if path.exists():
                shutil.rmtree(path)

    source_dirs = {
        source: ensure_dir(generated_dir / source)
        for source in ("neural", "retrieval", "fusion", "fallback", "chosen")
    }

    atlas = None
    if bool(retrieval_cfg.get("enabled", True)):
        if not atlas_path.exists():
            raise FileNotFoundError("retrieval/style_atlas.npz was not found. Run prepare again.")
        atlas = StyleAtlas.load(atlas_path, trees=int(retrieval_cfg.get("flann_trees", 6)))

    processed = {int(item["codepoint"]) for item in selection_rows}
    remaining_rows = [row for row in rows if int(row["codepoint"]) not in processed]
    dataset = ReferenceProxyDataset(remaining_rows, inference_size)
    loader = DataLoader(
        dataset,
        batch_size=int(inference_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    by_cp = {int(row["codepoint"]): row for row in rows}
    offsets = [float(value) for value in inference_cfg["threshold_offsets"]]
    # Retrieval and fusion need fewer threshold variants than the neural output.
    compact_offsets = sorted(set([0.0, offsets[0], offsets[-1], -0.04, 0.04]))
    maximum_border = float(inference_cfg["maximum_border_ink"])
    candidate_weights = {str(k): float(v) for k, v in inference_cfg.get("candidate_weights", {}).items()}
    save_all_family_candidates = bool(inference_cfg.get("save_all_family_candidates", False))
    profile_size = int(cfg["render"]["size"])

    checkpoint_interval = max(1, int(inference_cfg.get("resume_interval", 32)))
    rows_since_checkpoint = 0
    progress = tqdm(
        total=len(rows),
        initial=len(selection_rows),
        desc="HanziStyleForge full generation and topology validation",
        unit="glyph",
    )
    try:
        with torch.no_grad():
            for batch in loader:
                cpu_inputs = batch["input"].float()
                inputs = cpu_inputs.to(device, non_blocking=True)
                with _autocast(device, amp_enabled):
                    member_outputs = [
                        _generator_tta(
                            model, inputs, bool(inference_cfg.get("test_time_augmentation", True))
                        )
                        for model in generators
                    ]
                    generator_ink = _mean_optional_heads(member_outputs, 0)
                    generator_sdf = _mean_optional_heads(member_outputs, 1)
                    generator_skeleton = _mean_optional_heads(member_outputs, 2)
                    generator_edge = _mean_optional_heads(member_outputs, 3)
                    assert generator_ink is not None
                    if refiner is not None:
                        refiner_input = torch.cat([generator_ink, inputs], dim=1)
                        final_ink, final_sdf, final_skeleton, final_edge = _probability_heads(refiner(refiner_input))
                    else:
                        final_ink, final_sdf, final_skeleton, final_edge = (
                            generator_ink, generator_sdf, generator_skeleton, generator_edge
                        )
                member_ink_np = [item[0][:, 0].float().cpu().numpy() for item in member_outputs]
                member_sdf_np = [
                    item[1][:, 0].float().cpu().numpy() if item[1] is not None else None
                    for item in member_outputs
                ]
                member_skeleton_np = [
                    item[2][:, 0].float().cpu().numpy() if item[2] is not None else None
                    for item in member_outputs
                ]
                final_np = final_ink[:, 0].float().cpu().numpy()
                sdf_np = final_sdf[:, 0].float().cpu().numpy() if final_sdf is not None else None
                skeleton_np = final_skeleton[:, 0].float().cpu().numpy() if final_skeleton is not None else None
                edge_np = final_edge[:, 0].float().cpu().numpy() if final_edge is not None else None
                proxy_np = np.moveaxis(cpu_inputs.numpy(), 1, -1)
                cps = [int(value) for value in batch["codepoint"]]

                for local_index, cp in enumerate(cps):
                    try:
                        row = by_cp[cp]
                        has_target = bool(int(row.get("has_target", 0)))
                        profile = _style_profile_for_complexity(
                            style_profiles, float(row.get("complexity", 0.0) or 0.0)
                        )
                        target_radius = float(profile.get("stroke_radius", {}).get("median", 3.0))
                        reference_proxy = proxy_np[local_index]
                        reference_proxy_metrics = read_proxy(row["ref_proxy_path"])
                        reference_ink = read_ink(row["ref_path"])
                        reference_signature = topology_signature(
                            reference_ink,
                            size=int(topology_cfg.get("analysis_size", analysis_size)),
                            prune_iterations=int(topology_cfg.get("prune_iterations", 1)),
                        )
                        neural_probability = final_np[local_index]
                        if bool(topology_cfg.get("structure_lock", True)):
                            neural_probability = structure_lock_probability(
                                neural_probability,
                                reference_proxy,
                                target_stroke_radius=target_radius,
                                profile_size=profile_size,
                                core_strength=float(topology_cfg.get("structure_lock_core_strength", 0.92)),
                                maximum_radius_multiplier=float(topology_cfg.get("structure_lock_radius_multiplier", 2.7)),
                            )
        
                        retrieval_probability = None
                        retrieval_meta: dict[str, Any] = {}
                        if atlas is not None:
                            retrieval_probability, retrieval_meta = render_retrieval_candidate(reference_proxy, atlas, retrieval_cfg)
                            if bool(topology_cfg.get("structure_lock", True)):
                                retrieval_probability = structure_lock_probability(
                                    retrieval_probability,
                                    reference_proxy,
                                    target_stroke_radius=target_radius,
                                    profile_size=profile_size,
                                    core_strength=float(topology_cfg.get("structure_lock_core_strength", 0.92)),
                                    maximum_radius_multiplier=float(topology_cfg.get("structure_lock_radius_multiplier", 2.7)),
                                )
        
                        families: dict[str, dict[str, Any]] = {}
                        family_probabilities: dict[str, np.ndarray | None] = {}
                        neural_candidates = _threshold_candidates(
                            neural_probability, refiner_threshold, offsets, "neural"
                        )
                        neural_candidates.extend(
                            _multitask_candidates(
                                neural_probability,
                                None if sdf_np is None else sdf_np[local_index],
                                None if skeleton_np is None else skeleton_np[local_index],
                                refiner_threshold,
                                target_radius,
                                profile_size,
                            )
                        )
                        # Independent seeds and promoted Marathon snapshots are kept as
                        # separate candidates. Averaging alone can blur rare terminals;
                        # topology gating chooses the strongest individual or consensus result.
                        member_probabilities = [values[local_index] for values in member_ink_np]
                        for member_index, member_probability in enumerate(member_probabilities):
                            prefix = f"ensemble_m{member_index:02d}"
                            neural_candidates.extend(
                                _threshold_candidates(
                                    member_probability, generator_threshold, compact_offsets, prefix
                                )
                            )
                            neural_candidates.extend(
                                _multitask_candidates(
                                    member_probability,
                                    None if member_sdf_np[member_index] is None else member_sdf_np[member_index][local_index],
                                    None if member_skeleton_np[member_index] is None else member_skeleton_np[member_index][local_index],
                                    generator_threshold,
                                    target_radius,
                                    profile_size,
                                )
                            )
                        if len(member_probabilities) > 1:
                            member_stack = np.stack(member_probabilities, axis=0)
                            ensemble_mean = member_stack.mean(axis=0)
                            ensemble_median = np.median(member_stack, axis=0)
                            ensemble_std = member_stack.std(axis=0)
                            neural_candidates.extend(
                                _threshold_candidates(ensemble_mean, generator_threshold, compact_offsets, "ensemble_mean")
                            )
                            neural_candidates.extend(
                                _threshold_candidates(ensemble_median, generator_threshold, compact_offsets, "ensemble_median")
                            )
                            for uncertainty_weight in (0.35, 0.70):
                                conservative = np.clip(ensemble_mean - uncertainty_weight * ensemble_std, 0.0, 1.0)
                                expansive = np.clip(ensemble_mean + uncertainty_weight * ensemble_std, 0.0, 1.0)
                                neural_candidates.append(
                                    (f"ensemble_conservative_{uncertainty_weight:.2f}", (conservative >= generator_threshold).astype(np.float32))
                                )
                                neural_candidates.append(
                                    (f"ensemble_expansive_{uncertainty_weight:.2f}", (expansive >= generator_threshold).astype(np.float32))
                                )
                        families["neural"] = _evaluate_family(
                            "neural",
                            neural_candidates,
                            reference_proxy_metrics,
                            reference_ink,
                            profile,
                            analysis_size,
                            profile_size,
                            maximum_border,
                            topology_cfg,
                            candidate_weights.get("neural", 1.0),
                            reference_signature,
                        )
                        family_probabilities["neural"] = neural_probability
        
                        if retrieval_probability is not None:
                            families["retrieval"] = _evaluate_family(
                                "retrieval",
                                _threshold_candidates(retrieval_probability, 0.5, compact_offsets, "retrieval"),
                                reference_proxy_metrics,
                                reference_ink,
                                profile,
                                analysis_size,
                                profile_size,
                                maximum_border,
                                topology_cfg,
                                candidate_weights.get("retrieval", 0.97),
                                reference_signature,
                            )
                            family_probabilities["retrieval"] = retrieval_probability
                            nw = float(inference_cfg.get("fusion_neural_weight", 0.55))
                            rw = float(inference_cfg.get("fusion_retrieval_weight", 0.45))
                            fusion_probability = (nw * neural_probability + rw * retrieval_probability) / max(nw + rw, 1e-6)
                            if bool(topology_cfg.get("structure_lock", True)):
                                fusion_probability = structure_lock_probability(
                                    fusion_probability,
                                    reference_proxy,
                                    target_stroke_radius=target_radius,
                                    profile_size=profile_size,
                                    core_strength=float(topology_cfg.get("structure_lock_core_strength", 0.92)),
                                    maximum_radius_multiplier=float(topology_cfg.get("structure_lock_radius_multiplier", 2.7)),
                                )
                            families["fusion"] = _evaluate_family(
                                "fusion",
                                _threshold_candidates(fusion_probability, 0.5, compact_offsets, "fusion"),
                                reference_proxy_metrics,
                                reference_ink,
                                profile,
                                analysis_size,
                                profile_size,
                                maximum_border,
                                topology_cfg,
                                candidate_weights.get("fusion", 0.94),
                                reference_signature,
                            )
                            family_probabilities["fusion"] = fusion_probability
        
                        families["fallback"] = _evaluate_family(
                            "fallback",
                            make_reference_fallbacks(
                                reference_ink,
                                profile,
                                threshold=float(cfg["render"]["threshold"]),
                            ),
                            reference_proxy_metrics,
                            reference_ink,
                            profile,
                            analysis_size,
                            profile_size,
                            maximum_border,
                            topology_cfg,
                            candidate_weights.get("fallback", 1.08),
                            reference_signature,
                        )
                        family_probabilities["fallback"] = None
        
                        for source, family in families.items():
                            probability = family_probabilities.get(source)
                            family["confidence"] = _confidence(family, probability, float(thresholds["keep"]))
                            if save_all_family_candidates:
                                save_ink(source_dirs[source] / cp_filename(cp), family["mask"])
        
                        # Topology-pass candidates always outrank failed candidates.
                        # In strict automatic mode, an all-failed set must fall back
                        # to a ref-derived mask instead of accepting an attractive but
                        # structurally unverified neural/retrieval glyph.
                        passing = [family for family in families.values() if family["validation"]["hard_pass"]]
                        if passing:
                            chosen = min(
                                passing,
                                key=lambda family: (float(family["total_score"]), -float(family["confidence"])),
                            )
                        else:
                            chosen = families["fallback"]
                        chosen_source = str(chosen["source"])
                        rejection_reasons = {
                            source: family["validation"]["reasons"]
                            for source, family in families.items()
                            if not family["validation"]["hard_pass"]
                        }
        
                        has_target = bool(int(row["has_target"]))
                        target_score = math.inf
                        final_action = "replace" if has_target else "add"
                        chosen_path = source_dirs["chosen"] / cp_filename(cp)
                        save_ink(chosen_path, chosen["mask"])
                        notes = ""
                        if not chosen["validation"]["hard_pass"]:
                            notes = "All styled candidates failed the hard topology gate; a reference-derived safe fallback was used automatically."

                        confidence_by_source = {source: float(family["confidence"]) for source, family in families.items()}
                        pseudo_eligible = 0

                        def score_for(source: str, kind: str) -> Any:
                            family = families.get(source)
                            if family is None:
                                return ""
                            if kind == "structure":
                                return family["structure"]["structure_score"]
                            if kind == "topology":
                                return family["topology"]["topology_score"]
                            if kind == "style":
                                return family.get("style_score", "")
                            return ""
        
                        selection_rows.append(
                            {
                                "codepoint": cp,
                                "unicode": row["unicode"],
                                "char": row["char"],
                                "has_target": int(has_target),
                                "locl_sensitive": int(row["locl_sensitive"]),
                                "preliminary_status": row["preliminary_status"],
                                "final_action": final_action,
                                "chosen_source": chosen_source,
                                "chosen_label": chosen["label"],
                                "chosen_path": str(Path(chosen_path).resolve()),
                                "ref_path": row["ref_path"],
                                "target_path": row["target_path"],
                                "ref_proxy_path": row["ref_proxy_path"],
                                "target_structure_score": "",
                                "neural_structure_score": score_for("neural", "structure"),
                                "retrieval_structure_score": score_for("retrieval", "structure"),
                                "fusion_structure_score": score_for("fusion", "structure"),
                                "fallback_structure_score": score_for("fallback", "structure"),
                                "chosen_structure_score": chosen["structure"]["structure_score"],
                                "neural_topology_score": score_for("neural", "topology"),
                                "retrieval_topology_score": score_for("retrieval", "topology"),
                                "fusion_topology_score": score_for("fusion", "topology"),
                                "fallback_topology_score": score_for("fallback", "topology"),
                                "chosen_topology_score": chosen["topology"]["topology_score"],
                                "chosen_topology_pass": int(chosen["validation"]["hard_pass"]),
                                "chosen_component_delta": chosen["topology"]["component_delta"],
                                "chosen_hole_delta": chosen["topology"]["hole_delta"],
                                "chosen_endpoint_delta": chosen["topology"]["endpoint_delta"],
                                "chosen_junction_delta": chosen["topology"]["junction_delta"],
                                "neural_style_score": score_for("neural", "style"),
                                "retrieval_style_score": score_for("retrieval", "style"),
                                "fusion_style_score": score_for("fusion", "style"),
                                "fallback_style_score": score_for("fallback", "style"),
                                "neural_confidence": confidence_by_source.get("neural", ""),
                                "retrieval_confidence": confidence_by_source.get("retrieval", ""),
                                "fusion_confidence": confidence_by_source.get("fusion", ""),
                                "chosen_confidence": chosen["confidence"],
                                "pseudo_eligible": pseudo_eligible,
                                "retrieval_sources": ",".join(f"U+{value:04X}" for value in retrieval_meta.get("top_source_codepoints", [])),
                                "rejection_reasons": str(rejection_reasons),
                                "notes": notes,
                            }
                        )
                    except Exception as exc:
                        selection_rows.append(
                            _emergency_fallback_row(
                                by_cp[cp], cp, source_dirs["chosen"], exc
                            )
                        )
                    progress.update(1)
                    rows_since_checkpoint += 1
                    if rows_since_checkpoint >= checkpoint_interval:
                        _write_partial(partial_path, state_path, selection_rows, fingerprint, len(rows))
                        rows_since_checkpoint = 0
                        longrun_guard.checkpoint_boundary()
    finally:
        progress.close()
        if selection_rows and len(selection_rows) < len(rows):
            _write_partial(partial_path, state_path, selection_rows, fingerprint, len(rows))

    write_csv(selection_path, selection_rows, SELECTION_FIELDS)
    partial_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    for source in ("neural", "retrieval", "fusion", "fallback"):
        save_codepoints(
            generated_dir / f"chosen_{source}.txt",
            [int(row["codepoint"]) for row in selection_rows if row["chosen_source"] == source],
        )
    save_codepoints(generated_dir / "added.txt", [int(row["codepoint"]) for row in selection_rows if row["final_action"] == "add"])
    save_codepoints(
        generated_dir / "topology_failed.txt",
        [int(row["codepoint"]) for row in selection_rows if int(row["chosen_topology_pass"]) == 0],
    )

    actions: dict[str, int] = {}
    sources: dict[str, int] = {}
    labels: dict[str, int] = {}
    for item in selection_rows:
        actions[item["final_action"]] = actions.get(item["final_action"], 0) + 1
        sources[item["chosen_source"]] = sources.get(item["chosen_source"], 0) + 1
        label = str(item.get("chosen_label", ""))
        labels[label] = labels.get(label, 0) + 1
    structure_scores = [float(row["chosen_structure_score"]) for row in selection_rows if row["chosen_structure_score"] != ""]
    topology_scores = [float(row["chosen_topology_score"]) for row in selection_rows if row["chosen_topology_score"] != ""]
    summary = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "target_count": len(selection_rows),
        "generator_model_count": len(generators),
        "generator_checkpoints": [str(path.resolve()) for path in generator_paths],
        "actions": actions,
        "sources": sources,
        "chosen_labels": labels,
        "raw_ref_fallback_count": (labels.get("reference_raw", 0) + labels.get("reference_emergency", 0)),
        "topology_pass_count": sum(int(row["chosen_topology_pass"]) for row in selection_rows),
        "topology_failure_count": sum(1 - int(row["chosen_topology_pass"]) for row in selection_rows),
        "pseudo_eligible_count": 0,
        "mean_chosen_structure_score": float(np.mean(structure_scores)) if structure_scores else 0.0,
        "mean_chosen_topology_score": float(np.mean(topology_scores)) if topology_scores else 0.0,
        "selection_csv": str(selection_path.resolve()),
        "replacement_policy": inference_cfg.get("replacement_policy"),
        "important_note": (
            "By default, every Han glyph covered by ref.otf is rebuilt. Cross-font geometric diagnostics do not determine training eligibility or final replacement. "
            "Candidates come from multi-seed and multi-loss ensembles, local real-glyph residual retrieval, fusion, and reference-structure fallback, followed by topology gating."
        ),
    }
    save_json(summary_path, summary)
    save_json(
        completion_path,
        {
            "version": CHECKPOINT_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "target_count": len(selection_rows),
        },
    )
    return summary
