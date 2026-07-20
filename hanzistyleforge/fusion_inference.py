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
from PIL import Image
from tqdm import tqdm

from .component_atlas import ComponentAtlas, render_component_candidate
from .contract import validate_data_flow_contract
from .dataset import expand_proxy_channels
from .decomposition import load_decompositions
from .features import split_prediction
from .fusion_diffusion import DiffusionSchedule, classifier_free_guidance_sample
from .fusion_training import (
    FUSION_CHECKPOINT_VERSION,
    load_diffusion,
    load_direct_baseline,
    load_fusion_refiner,
    load_style_bank,
    load_vqvae,
)
from .inference import (
    SELECTION_FIELDS,
    _confidence,
    _emergency_fallback_row,
    _evaluate_family,
    _multitask_candidates,
    _style_profile_for_complexity,
    _threshold_candidates,
)
from .longrun import LongRunGuard
from .proxy import (
    make_reference_fallbacks,
    read_ink,
    read_proxy,
    save_ink,
)
from .retrieval import StyleAtlas, render_retrieval_candidate
from .topology import structure_lock_probability, topology_signature
from .util import (
    cp_filename,
    durable_replace,
    ensure_dir,
    load_json,
    read_csv,
    save_codepoints,
    save_json,
    sha256_file,
    write_csv,
)


FUSION_SELECTION_FIELDS = SELECTION_FIELDS + [
    "direct_structure_score",
    "diffusion_structure_score",
    "component_structure_score",
    "direct_topology_score",
    "diffusion_topology_score",
    "component_topology_score",
    "direct_style_score",
    "diffusion_style_score",
    "component_style_score",
    "direct_confidence",
    "diffusion_confidence",
    "component_confidence",
    "diffusion_seed_count",
    "diffusion_steps",
    "component_coverage",
    "component_sources",
    "model_disagreement",
]


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg.get("training", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Fusion inference requires CUDA, but torch.cuda.is_available() is False")
    return torch.device(requested if requested.startswith("cuda") else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _read_proxy10(path: str | Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        proxy4 = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    if proxy4.shape[:2] != (size, size):
        proxy4 = np.stack(
            [cv2.resize(proxy4[..., channel], (size, size), interpolation=cv2.INTER_AREA) for channel in range(4)],
            axis=-1,
        )
    return expand_proxy_channels(proxy4.clip(0.0, 1.0)).astype(np.float32)


def _tensor_proxy(proxy: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.moveaxis(proxy, -1, 0)[None].astype(np.float32)).to(device)


def _direct_probability(model: torch.nn.Module | None, proxy: torch.Tensor, amp: bool) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if model is None:
        return None, None, None
    with torch.no_grad(), _autocast(proxy.device, amp):
        output = model(proxy)
        ink_logits, sdf_logits, skeleton_logits, _ = split_prediction(output)
        ink = torch.sigmoid(ink_logits)[0, 0].float().cpu().numpy()
        sdf = torch.sigmoid(sdf_logits)[0, 0].float().cpu().numpy() if sdf_logits is not None else None
        skeleton = torch.sigmoid(skeleton_logits)[0, 0].float().cpu().numpy() if skeleton_logits is not None else None
    return ink, sdf, skeleton


def _style_variants(style_bank: dict[str, Any], count: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    mean = style_bank["mean_experts"].to(device).float()
    groups = style_bank.get("experts")
    values: list[torch.Tensor] = [mean]
    if isinstance(groups, torch.Tensor) and groups.ndim == 3 and len(groups) > 0:
        generator = np.random.default_rng(int(seed))
        order = generator.permutation(len(groups)).tolist()
        for index in order:
            candidate = groups[int(index)].to(device).float()
            # A conservative interpolation preserves the target-font centre
            # while exposing localized style modes learned from disjoint real
            # target glyph groups.
            alpha = 0.50 + 0.35 * ((len(values) - 1) % 3) / 2.0
            values.append((1.0 - alpha) * mean + alpha * candidate)
            if len(values) >= max(1, int(count)):
                break
    while len(values) < max(1, int(count)):
        values.append(mean)
    return [value.unsqueeze(0) for value in values[: max(1, int(count))]]


def _diffusion_probabilities(
    cfg: dict[str, Any],
    codepoint: int,
    proxy_tensor: torch.Tensor,
    diffusion: torch.nn.Module,
    vq: torch.nn.Module,
    refiner: torch.nn.Module | None,
    style_bank: dict[str, Any],
    schedule: DiffusionSchedule,
    *,
    target_radius: float,
    profile_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray | None], list[np.ndarray | None], dict[str, Any]]:
    fusion_cfg = cfg.get("fusion", {})
    inference_cfg = fusion_cfg.get("inference", {})
    device = proxy_tensor.device
    amp = bool(cfg.get("training", {}).get("amp", True) and device.type == "cuda")
    seeds = max(1, int(inference_cfg.get("diffusion_seeds", 6)))
    steps = max(8, int(inference_cfg.get("ddim_steps", 96)))
    guidance = float(inference_cfg.get("guidance_scale", 1.20))
    eta = float(inference_cfg.get("ddim_eta", 0.0))
    style_values = _style_variants(style_bank, seeds, codepoint * 104729 + 17, device)
    latent_h = max(4, proxy_tensor.shape[-2] // 8)
    latent_w = max(4, proxy_tensor.shape[-1] // 8)
    latent_channels = int(fusion_cfg.get("latent_channels", 32))
    probabilities: list[np.ndarray] = []
    sdf_values: list[np.ndarray | None] = []
    skeleton_values: list[np.ndarray | None] = []
    null_style = torch.zeros_like(style_values[0])
    for local in range(seeds):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(codepoint) * 1000003 + local * 104729 + int(cfg.get("training", {}).get("seed", 0)))
        with torch.no_grad(), _autocast(device, amp):
            latent = classifier_free_guidance_sample(
                diffusion,
                schedule,
                (1, latent_channels, latent_h, latent_w),
                content_proxy=proxy_tensor,
                style_experts=style_values[local],
                null_style_experts=null_style,
                guidance_scale=guidance,
                steps=steps,
                eta=eta,
                generator=generator,
            )
            decoded = vq.decode(latent, snap_to_codebook=bool(inference_cfg.get("snap_to_codebook", True)))
            ink = torch.sigmoid(decoded[:, :1])
            sdf = torch.sigmoid(decoded[:, 1:2]) if decoded.shape[1] > 1 else None
            skeleton = torch.sigmoid(decoded[:, 2:3]) if decoded.shape[1] > 2 else None
            if ink.shape[-2:] != proxy_tensor.shape[-2:]:
                ink = torch.nn.functional.interpolate(ink, size=proxy_tensor.shape[-2:], mode="bilinear", align_corners=False)
                if sdf is not None:
                    sdf = torch.nn.functional.interpolate(sdf, size=proxy_tensor.shape[-2:], mode="bilinear", align_corners=False)
                if skeleton is not None:
                    skeleton = torch.nn.functional.interpolate(skeleton, size=proxy_tensor.shape[-2:], mode="bilinear", align_corners=False)
            if refiner is not None:
                refined_logits = refiner(torch.cat([ink, proxy_tensor], dim=1), style_values[local])
                ink = torch.sigmoid(refined_logits)
        probabilities.append(ink[0, 0].float().cpu().numpy().clip(0.0, 1.0))
        sdf_values.append(None if sdf is None else sdf[0, 0].float().cpu().numpy().clip(0.0, 1.0))
        skeleton_values.append(None if skeleton is None else skeleton[0, 0].float().cpu().numpy().clip(0.0, 1.0))
    stack = np.stack(probabilities, axis=0)
    metadata = {
        "seed_count": seeds,
        "steps": steps,
        "mean_disagreement": float(stack.std(axis=0).mean()),
        "maximum_disagreement": float(stack.std(axis=0).max()),
    }
    return probabilities, sdf_values, skeleton_values, metadata


def _fusion_fingerprint(cfg: dict[str, Any], analysis_path: Path) -> str:
    work = Path(cfg["paths"]["work_dir"])
    paths = [
        work / "fusion" / "style" / "best.pt",
        work / "fusion" / "style" / "style_bank.pt",
        work / "fusion" / "vq" / "vq_best.pt",
        work / "fusion" / "diffusion" / "diffusion_best.pt",
        work / "fusion" / "direct" / "generator_best.pt",
        work / "fusion" / "refiner" / "best.pt",
        work / "retrieval" / "style_atlas.npz",
        work / "component_atlas" / "atlas.npz",
    ]
    payload = {
        "version": FUSION_CHECKPOINT_VERSION,
        "analysis": sha256_file(analysis_path),
        "models": {str(path.relative_to(work)): sha256_file(path) if path.is_file() else "" for path in paths},
        "fusion": cfg.get("fusion", {}),
        "topology": cfg.get("topology", {}),
        "retrieval": cfg.get("retrieval", {}),
        "inference": cfg.get("inference", {}),
        "render": cfg.get("render", {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_partial(path: Path, state: Path, rows: list[dict[str, Any]], fingerprint: str, total: int) -> None:
    write_csv(path, rows, FUSION_SELECTION_FIELDS)
    save_json(state, {
        "version": FUSION_CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "completed_count": len(rows),
        "target_count": int(total),
    })


def _save_family(path: Path, mask: np.ndarray) -> None:
    save_ink(path, np.asarray(mask, dtype=np.float32).clip(0.0, 1.0))


def _family_metric(families: dict[str, dict[str, Any]], source: str, kind: str) -> Any:
    item = families.get(source)
    if item is None:
        return ""
    if kind == "structure":
        return item["structure"]["structure_score"]
    if kind == "topology":
        return item["topology"]["topology_score"]
    if kind == "style":
        return item.get("style_score", "")
    if kind == "confidence":
        return item.get("confidence", "")
    return ""


def generate_fusion_and_select(cfg: dict[str, Any], *, output_subdir: str = "generated") -> dict[str, Any]:
    """Generate every Han codepoint covered by ref.otf using the fusion stack.

    The function is deliberately fail-complete: every per-glyph exception is
    converted into a raw ref-derived emergency candidate, and completion is not
    recorded until all analysis rows have one unique selected image.
    """

    validate_data_flow_contract(cfg, require_prepared=True, write_report=True)
    work = Path(cfg["paths"]["work_dir"])
    analysis_path = work / "audit" / "analysis.csv"
    if not analysis_path.is_file():
        raise FileNotFoundError("missing audit/analysis.csv; run prepare first")
    rows = read_csv(analysis_path)
    if not rows:
        raise RuntimeError("analysis target list is empty")
    by_cp = {int(row["codepoint"]): row for row in rows}
    if len(by_cp) != len(rows):
        raise RuntimeError("analysis.csv contains duplicate Han codepoints")

    generated = ensure_dir(work / output_subdir)
    partial_path = generated / "selection.partial.csv"
    state_path = generated / "generation.state.json"
    selection_path = generated / "selection.csv"
    summary_path = generated / "summary.json"
    completion_path = generated / "generation.completed.json"
    fingerprint = _fusion_fingerprint(cfg, analysis_path)

    if completion_path.is_file() and selection_path.is_file() and summary_path.is_file():
        try:
            completed = load_json(completion_path)
            existing = read_csv(selection_path)
            if completed.get("fingerprint") == fingerprint and len(existing) == len(rows):
                return load_json(summary_path)
        except Exception:
            pass

    selection_rows: list[dict[str, Any]] = []
    if partial_path.is_file() and state_path.is_file():
        try:
            state = load_json(state_path)
            if state.get("fingerprint") == fingerprint:
                selection_rows = read_csv(partial_path)
        except Exception:
            selection_rows = []
    if len({int(row["codepoint"]) for row in selection_rows}) != len(selection_rows):
        selection_rows = []
    if not selection_rows:
        partial_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        completion_path.unlink(missing_ok=True)
        for child in generated.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    source_names = ("diffusion", "direct", "retrieval", "component", "fusion", "fallback", "chosen")
    source_dirs = {name: ensure_dir(generated / name) for name in source_names}
    device = _device(cfg)
    amp = bool(cfg.get("training", {}).get("amp", True) and device.type == "cuda")
    diffusion = load_diffusion(cfg, device)
    vq = load_vqvae(cfg, device)
    direct = load_direct_baseline(cfg, device)
    refiner = load_fusion_refiner(cfg, device)
    style_bank = load_style_bank(cfg, device)
    fusion_cfg = cfg.get("fusion", {})
    diffusion_cfg = fusion_cfg.get("diffusion", {})
    schedule = DiffusionSchedule.create(
        int(fusion_cfg.get("diffusion_steps", 1000)),
        schedule=str(diffusion_cfg.get("schedule", "cosine")),
    ).to(device)

    atlas: StyleAtlas | None = None
    atlas_path = work / "retrieval" / "style_atlas.npz"
    if bool(cfg.get("retrieval", {}).get("enabled", True)) and atlas_path.is_file():
        atlas = StyleAtlas.load(atlas_path, trees=int(cfg.get("retrieval", {}).get("flann_trees", 8)))

    component_atlas: ComponentAtlas | None = None
    decompositions: dict[int, Any] = {}
    component_cfg = fusion_cfg.get("component_atlas", {})
    component_atlas_path = work / "component_atlas" / "atlas.npz"
    component_labels_path = work / "component_atlas" / "labels.json"
    decomposition_path = Path(component_cfg.get("decomposition_file", "data/cjk-decomp.txt"))
    if bool(component_cfg.get("enabled", True)) and component_atlas_path.is_file() and component_labels_path.is_file():
        component_atlas = ComponentAtlas.load(component_atlas_path, component_labels_path)
        if decomposition_path.is_file():
            decompositions = load_decompositions(decomposition_path)

    style_profile = load_json(work / "dataset" / "style_profile.json")
    profiles_path = work / "dataset" / "style_profiles.json"
    style_profiles = load_json(profiles_path) if profiles_path.is_file() else {"global": style_profile, "bins": []}
    thresholds_path = work / "audit" / "structure_thresholds.json"
    thresholds = load_json(thresholds_path) if thresholds_path.is_file() else {"keep": 0.05}
    topology_cfg = cfg.get("topology", {})
    fusion_inf = fusion_cfg.get("inference", {})
    normal_inf = cfg.get("inference", {})
    inference_size = int(fusion_inf.get("size", normal_inf.get("size", cfg["render"]["size"])))
    analysis_size = int(cfg["render"].get("analysis_size", 192))
    profile_size = int(cfg["render"].get("size", inference_size))
    maximum_border = float(normal_inf.get("maximum_border_ink", 0.015))
    threshold_offsets = [float(value) for value in fusion_inf.get(
        "threshold_offsets", normal_inf.get("threshold_offsets", [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10])
    )]
    compact_offsets = sorted(set([threshold_offsets[0], -0.04, 0.0, 0.04, threshold_offsets[-1]]))
    weights = {
        "diffusion": 0.90,
        "direct": 1.00,
        "retrieval": 0.98,
        "component": 0.94,
        "fusion": 0.88,
        "fallback": 1.12,
        **{str(key): float(value) for key, value in fusion_inf.get("candidate_weights", {}).items()},
    }
    direct_threshold = 0.5
    calibration = work / "fusion" / "direct" / "calibration.json"
    if calibration.is_file():
        direct_threshold = float(load_json(calibration).get("threshold", 0.5))
    diffusion_threshold = float(fusion_inf.get("threshold", 0.5))
    save_all = bool(fusion_inf.get("save_all_family_candidates", False))
    processed = {int(row["codepoint"]) for row in selection_rows}
    guard = LongRunGuard(cfg)
    progress = tqdm(total=len(rows), initial=len(selection_rows), desc="HanziStyleForge Fusion full generation", unit="glyph")

    try:
        for row in rows:
            cp = int(row["codepoint"])
            if cp in processed:
                continue
            try:
                proxy_np = _read_proxy10(row["ref_proxy_path"], inference_size)
                proxy_tensor = _tensor_proxy(proxy_np, device)
                proxy_metrics = read_proxy(row["ref_proxy_path"])
                reference_ink = read_ink(row["ref_path"])
                if reference_ink.shape != (inference_size, inference_size):
                    reference_ink = cv2.resize(reference_ink, (inference_size, inference_size), interpolation=cv2.INTER_AREA)
                profile = _style_profile_for_complexity(style_profiles, float(row.get("complexity", 0.0) or 0.0))
                target_radius = float(profile.get("stroke_radius", {}).get("median", 3.0))
                reference_signature = topology_signature(
                    reference_ink,
                    size=int(topology_cfg.get("analysis_size", analysis_size)),
                    prune_iterations=int(topology_cfg.get("prune_iterations", 1)),
                )

                diffusion_probs, diffusion_sdf, diffusion_skeleton, diffusion_meta = _diffusion_probabilities(
                    cfg, cp, proxy_tensor, diffusion, vq, refiner, style_bank, schedule,
                    target_radius=target_radius, profile_size=profile_size,
                )
                direct_prob, direct_sdf, direct_skeleton = _direct_probability(direct, proxy_tensor, amp)
                retrieval_prob: np.ndarray | None = None
                retrieval_meta: dict[str, Any] = {}
                if atlas is not None:
                    retrieval_prob, retrieval_meta = render_retrieval_candidate(proxy_np, atlas, cfg.get("retrieval", {}))
                component_prob: np.ndarray | None = None
                component_meta: dict[str, Any] = {}
                if component_atlas is not None and decompositions:
                    component_prob, component_meta = render_component_candidate(
                        cp, proxy_np, component_atlas, decompositions, component_cfg
                    )

                # Apply a conservative structure core lock to every learned
                # candidate. It protects short ref strokes without making the
                # fallback or content proxy the generator's direct output.
                def locked(value: np.ndarray | None) -> np.ndarray | None:
                    if value is None:
                        return None
                    value = np.asarray(value, dtype=np.float32).clip(0.0, 1.0)
                    if bool(topology_cfg.get("structure_lock", True)):
                        value = structure_lock_probability(
                            value,
                            proxy_np,
                            target_stroke_radius=target_radius,
                            profile_size=profile_size,
                            core_strength=float(topology_cfg.get("structure_lock_core_strength", 0.95)),
                            maximum_radius_multiplier=float(topology_cfg.get("structure_lock_radius_multiplier", 2.35)),
                        )
                    return value.clip(0.0, 1.0)

                diffusion_probs = [locked(value) for value in diffusion_probs]
                direct_prob = locked(direct_prob)
                retrieval_prob = locked(retrieval_prob)
                component_prob = locked(component_prob)

                families: dict[str, dict[str, Any]] = {}
                family_probability: dict[str, np.ndarray | None] = {}
                diffusion_candidates: list[tuple[str, np.ndarray]] = []
                for index, probability in enumerate(diffusion_probs):
                    assert probability is not None
                    diffusion_candidates.extend(_threshold_candidates(
                        probability, diffusion_threshold, compact_offsets, f"diffusion_seed_{index:02d}"
                    ))
                    diffusion_candidates.extend(_multitask_candidates(
                        probability,
                        diffusion_sdf[index] if index < len(diffusion_sdf) else None,
                        diffusion_skeleton[index] if index < len(diffusion_skeleton) else None,
                        diffusion_threshold,
                        target_radius,
                        profile_size,
                    ))
                stack = np.stack([value for value in diffusion_probs if value is not None], axis=0)
                diff_mean = stack.mean(axis=0)
                diff_median = np.median(stack, axis=0)
                diff_std = stack.std(axis=0)
                diffusion_candidates.extend(_threshold_candidates(diff_mean, diffusion_threshold, threshold_offsets, "diffusion_mean"))
                diffusion_candidates.extend(_threshold_candidates(diff_median, diffusion_threshold, threshold_offsets, "diffusion_median"))
                for strength in (0.35, 0.70, 1.0):
                    diffusion_candidates.extend(_threshold_candidates(
                        np.clip(diff_mean - strength * diff_std, 0.0, 1.0), diffusion_threshold, [0.0], f"diffusion_conservative_{strength:.2f}"
                    ))
                    diffusion_candidates.extend(_threshold_candidates(
                        np.clip(diff_mean + strength * diff_std, 0.0, 1.0), diffusion_threshold, [0.0], f"diffusion_expansive_{strength:.2f}"
                    ))
                families["diffusion"] = _evaluate_family(
                    "diffusion", diffusion_candidates, proxy_metrics, reference_ink, profile,
                    analysis_size, profile_size, maximum_border, topology_cfg, weights["diffusion"], reference_signature,
                )
                family_probability["diffusion"] = diff_mean

                if direct_prob is not None:
                    direct_candidates = _threshold_candidates(direct_prob, direct_threshold, threshold_offsets, "direct")
                    direct_candidates.extend(_multitask_candidates(
                        direct_prob, direct_sdf, direct_skeleton, direct_threshold, target_radius, profile_size
                    ))
                    families["direct"] = _evaluate_family(
                        "direct", direct_candidates, proxy_metrics, reference_ink, profile,
                        analysis_size, profile_size, maximum_border, topology_cfg, weights["direct"], reference_signature,
                    )
                    family_probability["direct"] = direct_prob

                if retrieval_prob is not None:
                    families["retrieval"] = _evaluate_family(
                        "retrieval", _threshold_candidates(retrieval_prob, 0.5, threshold_offsets, "retrieval"),
                        proxy_metrics, reference_ink, profile, analysis_size, profile_size, maximum_border,
                        topology_cfg, weights["retrieval"], reference_signature,
                    )
                    family_probability["retrieval"] = retrieval_prob

                if component_prob is not None:
                    families["component"] = _evaluate_family(
                        "component", _threshold_candidates(component_prob, 0.5, threshold_offsets, "component"),
                        proxy_metrics, reference_ink, profile, analysis_size, profile_size, maximum_border,
                        topology_cfg, weights["component"], reference_signature,
                    )
                    family_probability["component"] = component_prob

                # Blend only independently generated style probabilities. The
                # weights are normalized per glyph so missing optional modules
                # never change the fail-complete behaviour.
                blend_inputs: list[tuple[float, np.ndarray]] = [(0.52, diff_mean)]
                if direct_prob is not None:
                    blend_inputs.append((0.18, direct_prob))
                if retrieval_prob is not None:
                    blend_inputs.append((0.14, retrieval_prob))
                if component_prob is not None:
                    blend_inputs.append((0.16, component_prob))
                denominator = sum(weight for weight, _ in blend_inputs)
                fusion_prob = sum(weight * value for weight, value in blend_inputs) / max(denominator, 1e-6)
                fusion_prob = locked(fusion_prob)
                assert fusion_prob is not None
                fusion_candidates = _threshold_candidates(fusion_prob, 0.5, threshold_offsets, "fusion_weighted")
                # Pixel median is robust to one weak model family.
                independent = [value for _, value in blend_inputs]
                if len(independent) >= 3:
                    median_probability = np.median(np.stack(independent, axis=0), axis=0)
                    fusion_candidates.extend(_threshold_candidates(median_probability, 0.5, compact_offsets, "fusion_median"))
                families["fusion"] = _evaluate_family(
                    "fusion", fusion_candidates, proxy_metrics, reference_ink, profile,
                    analysis_size, profile_size, maximum_border, topology_cfg, weights["fusion"], reference_signature,
                )
                family_probability["fusion"] = fusion_prob

                families["fallback"] = _evaluate_family(
                    "fallback", make_reference_fallbacks(reference_ink, profile, threshold=float(cfg["render"]["threshold"])),
                    proxy_metrics, reference_ink, profile, analysis_size, profile_size, maximum_border,
                    topology_cfg, weights["fallback"], reference_signature,
                )
                family_probability["fallback"] = None

                for source, family in families.items():
                    family["confidence"] = _confidence(
                        family, family_probability.get(source), float(thresholds.get("keep", 0.05))
                    )
                    if save_all and source in source_dirs:
                        _save_family(source_dirs[source] / cp_filename(cp), family["mask"])

                passing = [family for family in families.values() if family["validation"]["hard_pass"]]
                if passing:
                    chosen = min(passing, key=lambda item: (float(item["total_score"]), -float(item["confidence"])))
                else:
                    chosen = families["fallback"]
                chosen_source = str(chosen["source"])
                chosen_path = source_dirs["chosen"] / cp_filename(cp)
                _save_family(chosen_path, chosen["mask"])
                has_target = bool(int(row.get("has_target", 0)))
                rejection_reasons = {
                    source: family["validation"]["reasons"]
                    for source, family in families.items()
                    if not family["validation"]["hard_pass"]
                }
                notes = ""
                if not chosen["validation"]["hard_pass"]:
                    notes = "All learned candidates failed the hard topology gate; ref-derived fallback selected."

                result: dict[str, Any] = {field: "" for field in FUSION_SELECTION_FIELDS}
                result.update({
                    "codepoint": cp,
                    "unicode": row.get("unicode", f"U+{cp:04X}"),
                    "char": row.get("char", chr(cp)),
                    "has_target": int(has_target),
                    "locl_sensitive": int(row.get("locl_sensitive", 0)),
                    "preliminary_status": row.get("preliminary_status", "rebuild"),
                    "final_action": "replace" if has_target else "add",
                    "chosen_source": chosen_source,
                    "chosen_label": chosen["label"],
                    "chosen_path": str(chosen_path.resolve()),
                    "ref_path": row.get("ref_path", ""),
                    "target_path": row.get("target_path", ""),
                    "ref_proxy_path": row.get("ref_proxy_path", ""),
                    "target_structure_score": "",
                    "chosen_structure_score": chosen["structure"]["structure_score"],
                    "chosen_topology_score": chosen["topology"]["topology_score"],
                    "chosen_topology_pass": int(chosen["validation"]["hard_pass"]),
                    "chosen_component_delta": chosen["topology"]["component_delta"],
                    "chosen_hole_delta": chosen["topology"]["hole_delta"],
                    "chosen_endpoint_delta": chosen["topology"]["endpoint_delta"],
                    "chosen_junction_delta": chosen["topology"]["junction_delta"],
                    "chosen_confidence": chosen["confidence"],
                    "pseudo_eligible": 0,
                    "retrieval_sources": ",".join(f"U+{value:04X}" for value in retrieval_meta.get("top_source_codepoints", [])),
                    "rejection_reasons": str(rejection_reasons),
                    "notes": notes,
                    "diffusion_seed_count": diffusion_meta["seed_count"],
                    "diffusion_steps": diffusion_meta["steps"],
                    "component_coverage": component_meta.get("coverage", ""),
                    "component_sources": ",".join(f"U+{value:04X}" for value in component_meta.get("sources", [])),
                    "model_disagreement": diffusion_meta["mean_disagreement"],
                })
                # Backwards-compatible aggregate columns used by existing QA.
                result.update({
                    "neural_structure_score": _family_metric(families, "diffusion", "structure"),
                    "retrieval_structure_score": _family_metric(families, "retrieval", "structure"),
                    "fusion_structure_score": _family_metric(families, "fusion", "structure"),
                    "fallback_structure_score": _family_metric(families, "fallback", "structure"),
                    "neural_topology_score": _family_metric(families, "diffusion", "topology"),
                    "retrieval_topology_score": _family_metric(families, "retrieval", "topology"),
                    "fusion_topology_score": _family_metric(families, "fusion", "topology"),
                    "fallback_topology_score": _family_metric(families, "fallback", "topology"),
                    "neural_style_score": _family_metric(families, "diffusion", "style"),
                    "retrieval_style_score": _family_metric(families, "retrieval", "style"),
                    "fusion_style_score": _family_metric(families, "fusion", "style"),
                    "fallback_style_score": _family_metric(families, "fallback", "style"),
                    "neural_confidence": _family_metric(families, "diffusion", "confidence"),
                    "retrieval_confidence": _family_metric(families, "retrieval", "confidence"),
                    "fusion_confidence": _family_metric(families, "fusion", "confidence"),
                    "direct_structure_score": _family_metric(families, "direct", "structure"),
                    "diffusion_structure_score": _family_metric(families, "diffusion", "structure"),
                    "component_structure_score": _family_metric(families, "component", "structure"),
                    "direct_topology_score": _family_metric(families, "direct", "topology"),
                    "diffusion_topology_score": _family_metric(families, "diffusion", "topology"),
                    "component_topology_score": _family_metric(families, "component", "topology"),
                    "direct_style_score": _family_metric(families, "direct", "style"),
                    "diffusion_style_score": _family_metric(families, "diffusion", "style"),
                    "component_style_score": _family_metric(families, "component", "style"),
                    "direct_confidence": _family_metric(families, "direct", "confidence"),
                    "diffusion_confidence": _family_metric(families, "diffusion", "confidence"),
                    "component_confidence": _family_metric(families, "component", "confidence"),
                })
                selection_rows.append(result)
            except Exception as exc:
                emergency = _emergency_fallback_row(row, cp, source_dirs["chosen"], exc)
                result = {field: "" for field in FUSION_SELECTION_FIELDS}
                result.update(emergency)
                result["diffusion_seed_count"] = 0
                result["diffusion_steps"] = 0
                selection_rows.append(result)
            processed.add(cp)
            progress.update(1)
            _write_partial(partial_path, state_path, selection_rows, fingerprint, len(rows))
            guard.checkpoint_boundary()
    finally:
        progress.close()
        if selection_rows and len(selection_rows) < len(rows):
            _write_partial(partial_path, state_path, selection_rows, fingerprint, len(rows))

    selected_by_cp = {int(row["codepoint"]): row for row in selection_rows}
    missing = sorted(set(by_cp) - set(selected_by_cp))
    duplicates = len(selection_rows) - len(selected_by_cp)
    absent_files = [cp for cp, row in selected_by_cp.items() if not Path(str(row.get("chosen_path", ""))).is_file()]
    if missing or duplicates or absent_files:
        save_json(generated / "coverage_failure.json", {
            "missing": missing,
            "duplicate_count": duplicates,
            "absent_chosen_files": absent_files,
        })
        raise RuntimeError(
            f"fusion generation incomplete: missing={len(missing)}, duplicates={duplicates}, absent_files={len(absent_files)}"
        )

    selection_rows = [selected_by_cp[int(row["codepoint"])] for row in rows]
    write_csv(selection_path, selection_rows, FUSION_SELECTION_FIELDS)
    partial_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    sources: dict[str, int] = {}
    labels: dict[str, int] = {}
    for item in selection_rows:
        sources[str(item["chosen_source"])] = sources.get(str(item["chosen_source"]), 0) + 1
        labels[str(item["chosen_label"])] = labels.get(str(item["chosen_label"]), 0) + 1
    for source in source_names[:-1]:
        save_codepoints(generated / f"chosen_{source}.txt", [
            int(item["codepoint"]) for item in selection_rows if item["chosen_source"] == source
        ])
    save_codepoints(generated / "added.txt", [
        int(item["codepoint"]) for item in selection_rows if item["final_action"] == "add"
    ])
    save_codepoints(generated / "topology_failed.txt", [
        int(item["codepoint"]) for item in selection_rows if int(item["chosen_topology_pass"]) == 0
    ])
    summary = {
        "version": FUSION_CHECKPOINT_VERSION,
        "method": (
            "target-only localized style encoder + target VQ stroke codebook + multi-scale latent diffusion + "
            "deterministic topology model + generic local retrieval + semantic component residual atlas + "
            "style-aware refiner + hard topology candidate selection"
        ),
        "target_count": len(rows),
        "unique_output_count": len(selected_by_cp),
        "coverage_complete": True,
        "sources": sources,
        "chosen_labels": labels,
        "topology_pass_count": sum(int(item["chosen_topology_pass"]) for item in selection_rows),
        "topology_failure_count": sum(1 - int(item["chosen_topology_pass"]) for item in selection_rows),
        "raw_ref_fallback_count": labels.get("reference_raw", 0) + labels.get("reference_emergency", 0),
        "selection_csv": str(selection_path.resolve()),
        "fingerprint": fingerprint,
        "important_note": (
            "Every Han codepoint in ref.otf receives one output. If all learned candidates fail or a glyph raises an exception, "
            "the raw ref structure is used as a fail-complete fallback. Non-Han glyphs are handled only by the existing protected build stage."
        ),
    }
    save_json(summary_path, summary)
    save_json(generated / "coverage.json", {
        "ref_han_target_count": len(rows),
        "selected_count": len(selection_rows),
        "unique_codepoint_count": len(selected_by_cp),
        "missing_count": 0,
        "chosen_file_missing_count": 0,
        "complete": True,
    })
    save_json(completion_path, {
        "version": FUSION_CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "target_count": len(rows),
    })
    return summary
