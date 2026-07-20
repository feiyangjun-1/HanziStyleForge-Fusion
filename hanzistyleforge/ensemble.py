from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import CHECKPOINT_FORMAT_VERSION
from .longrun import LongRunGuard
from .training import train_generator
from .util import ensure_dir, load_json, save_json, sha256_file


def _apply_recipe(cfg: dict[str, Any], member_index: int) -> str:
    """Create complementary models instead of identical seed-only replicas."""

    weights = cfg.setdefault("loss", {}).setdefault("weights", {})
    recipe = member_index % 3
    if recipe == 1:
        # Topology member: conservative on missing/extra strokes and junctions.
        multipliers = {
            "cldice": 1.30,
            "proxy_skeleton": 1.28,
            "skeleton_head": 1.24,
            "topology_points": 1.28,
            "projection": 1.12,
            "boundary_distance": 1.12,
            "edge": 0.92,
        }
        name = "topology"
    elif recipe == 2:
        # Detail member: favors stroke terminals, curves and SDF boundaries.
        multipliers = {
            "edge": 1.30,
            "edge_head": 1.28,
            "sdf": 1.26,
            "boundary_distance": 1.24,
            "multiscale": 1.18,
            "bce": 0.94,
        }
        name = "detail"
    else:
        # Shape member: slightly more global ink/layout supervision.
        multipliers = {
            "dice": 1.22,
            "projection": 1.22,
            "proxy_layout": 1.30,
            "multiscale": 1.16,
            "cldice": 0.96,
        }
        name = "shape"
    for key, multiplier in multipliers.items():
        if key in weights:
            weights[key] = float(weights[key]) * float(multiplier)
    # Keep the total loss scale close across members.
    total = sum(float(value) for value in weights.values())
    if total > 1e-8:
        for key in list(weights):
            weights[key] = float(weights[key]) / total
    return name


def _snapshot_models(work: Path, maximum: int) -> list[dict[str, Any]]:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    snapshot_dir = work / "marathon" / "snapshots"
    for summary_path in snapshot_dir.glob("cycle_*.json"):
        try:
            summary = load_json(summary_path)
            checkpoint = summary_path.with_suffix(".pt")
            if not checkpoint.is_file():
                checkpoint = Path(summary.get("checkpoint", ""))
            if not checkpoint.is_file():
                continue
            quality = float(summary.get("validation", {}).get("quality", -1e9))
            candidates.append((quality, checkpoint, summary))
        except Exception:
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    result: list[dict[str, Any]] = []
    for quality, checkpoint, summary in candidates[: max(0, int(maximum))]:
        result.append(
            {
                "role": "marathon_snapshot",
                "checkpoint": str(checkpoint.resolve()),
                "calibration": str((work / "model" / "generator" / "calibration.json").resolve()),
                "quality": quality,
                "cycle": int(summary.get("cycle", 0)),
            }
        )
    return result


def run_ensemble_training(cfg: dict[str, Any]) -> dict[str, Any]:
    ensemble_cfg = cfg.get("marathon", {}).get("ensemble", {})
    if not bool(ensemble_cfg.get("enabled", True)):
        return {"enabled": False}

    work = Path(cfg["paths"]["work_dir"])
    active_checkpoint = work / "model" / "generator" / "generator_best.pt"
    active_calibration = work / "model" / "generator" / "calibration.json"
    if not active_checkpoint.is_file():
        raise FileNotFoundError("Ensemble requires completed base training and Marathon main-model training.")

    root = ensure_dir(work / "model" / "ensemble")
    independent_count = max(0, int(ensemble_cfg.get("independent_members", 2)))
    seed_stride = max(1, int(ensemble_cfg.get("seed_stride", 104729)))
    independent: list[dict[str, Any]] = []
    longrun_guard = LongRunGuard(cfg)

    for member_index in range(1, independent_count + 1):
        member_cfg = deepcopy(cfg)
        member_cfg["training"]["seed"] = int(cfg["training"]["seed"]) + seed_stride * member_index
        recipe = _apply_recipe(member_cfg, member_index)
        member_root = ensure_dir(root / f"member_{member_index:02d}_{recipe}")
        result = train_generator(
            member_cfg,
            index_csv=work / "dataset" / "index.csv",
            model_root=member_root,
            phases_override=None,
            init_checkpoint=None,
            resume=True,
        )
        independent.append(
            {
                "role": "independent",
                "member": member_index,
                "recipe": recipe,
                "seed": int(member_cfg["training"]["seed"]),
                "checkpoint": result["checkpoint"],
                "calibration": str((member_root / "calibration.json").resolve()),
                "validation": result.get("validation", {}),
            }
        )
        longrun_guard.checkpoint_boundary()

    models: list[dict[str, Any]] = []
    if bool(ensemble_cfg.get("include_active_model", True)):
        models.append(
            {
                "role": "active_marathon",
                "checkpoint": str(active_checkpoint.resolve()),
                "calibration": str(active_calibration.resolve()),
            }
        )
    models.extend(
        _snapshot_models(work, int(ensemble_cfg.get("include_marathon_snapshots", 2)))
    )
    models.extend(independent)

    # Deduplicate checkpoints by content. Promoted snapshots can equal the active model.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in models:
        path = Path(model["checkpoint"])
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if digest in seen:
            continue
        seen.add(digest)
        item = dict(model)
        item["sha256"] = digest
        unique.append(item)
        if len(unique) >= max(1, int(ensemble_cfg.get("maximum_models", 5))):
            break

    manifest = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "enabled": True,
        "model_count": len(unique),
        "models": unique,
        "independent_training": independent,
    }
    save_json(root / "manifest.json", manifest)
    return manifest


def ensemble_status(cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(cfg["paths"]["work_dir"]) / "model" / "ensemble" / "manifest.json"
    return load_json(path) if path.is_file() else {}
