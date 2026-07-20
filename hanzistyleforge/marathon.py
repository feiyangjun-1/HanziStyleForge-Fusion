from __future__ import annotations

import contextlib
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .analysis import DATASET_FIELDS
from .config import CHECKPOINT_FORMAT_VERSION
from .longrun import LongRunGuard
from .dataset import GlyphStyleDataset
from .features import ink_probability
from .training import load_generator, train_generator
from .util import ensure_dir, load_json, read_csv, save_json, set_seed, sha256_file, write_csv


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["training"].get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Marathon training requires CUDA, but torch.cuda.is_available() is False.")
    return torch.device(requested if requested.startswith("cuda") else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _self_dataset(index_csv: Path, split: str, size: int) -> GlyphStyleDataset:
    dataset = GlyphStyleDataset(index_csv, split, size=size, augment=False)
    dataset.rows = [row for row in dataset.rows if row.get("mode") == "self"]
    if not dataset.rows:
        raise RuntimeError(f"{index_csv} has no self samples for split={split}.")
    return dataset


def _per_sample_metrics(
    checkpoint: Path,
    index_csv: Path,
    cfg: dict[str, Any],
    *,
    split: str,
    size: int,
    batch_size: int,
) -> list[dict[str, float | int]]:
    device = _device(cfg)
    model, _ = load_generator(checkpoint, device)
    dataset = _self_dataset(index_csv, split, size)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(cfg["training"].get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    result: list[dict[str, float | int]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Marathon evaluation {split}", unit="batch", leave=False):
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with _autocast(device, amp):
                probability = ink_probability(model(inputs)).float()
            intersection = (probability * target).flatten(1).sum(1)
            denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
            dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
            l1 = (probability - target).abs().flatten(1).mean(1)
            p_dx = probability[..., :, 1:] - probability[..., :, :-1]
            p_dy = probability[..., 1:, :] - probability[..., :-1, :]
            t_dx = target[..., :, 1:] - target[..., :, :-1]
            t_dy = target[..., 1:, :] - target[..., :-1, :]
            edge = 0.5 * (
                (p_dx.abs() - t_dx.abs()).abs().flatten(1).mean(1)
                + (p_dy.abs() - t_dy.abs()).abs().flatten(1).mean(1)
            )
            hardness = (1.0 - dice) + 0.42 * l1 + 0.18 * edge
            cps = [int(value) for value in batch["codepoint"]]
            for cp, d, pixel, e, hard in zip(
                cps,
                dice.detach().cpu().tolist(),
                l1.detach().cpu().tolist(),
                edge.detach().cpu().tolist(),
                hardness.detach().cpu().tolist(),
            ):
                result.append(
                    {
                        "codepoint": int(cp),
                        "dice": float(d),
                        "l1": float(pixel),
                        "edge": float(e),
                        "hardness": float(hard),
                    }
                )
    return result


def _quality(metrics: list[dict[str, float | int]]) -> dict[str, float | int]:
    dice = np.asarray([float(item["dice"]) for item in metrics], dtype=np.float64)
    l1 = np.asarray([float(item["l1"]) for item in metrics], dtype=np.float64)
    if dice.size == 0:
        return {"count": 0, "mean_dice": 0.0, "p10_dice": 0.0, "mean_l1": 1.0, "quality": -1.0}
    quality = float(dice.mean() + 0.35 * np.quantile(dice, 0.10) - 0.10 * l1.mean())
    return {
        "count": int(dice.size),
        "mean_dice": float(dice.mean()),
        "p10_dice": float(np.quantile(dice, 0.10)),
        "p01_dice": float(np.quantile(dice, 0.01)),
        "mean_l1": float(l1.mean()),
        "quality": quality,
    }


def _make_hard_index(
    base_index: Path,
    metrics: list[dict[str, float | int]],
    output: Path,
    maximum_hard: int,
    maximum_val: int,
    seed: int,
) -> dict[str, Any]:
    rows = read_csv(base_index)
    self_train = {int(row["codepoint"]): row for row in rows if row.get("split") == "train" and row.get("mode") == "self"}
    self_val = [row for row in rows if row.get("split") == "val" and row.get("mode") == "self"]
    ranked = [item for item in sorted(metrics, key=lambda item: (-float(item["hardness"]), int(item["codepoint"]))) if int(item["codepoint"]) in self_train]
    selected = ranked[: min(maximum_hard, len(ranked))]
    hardness_values = np.asarray([float(item["hardness"]) for item in selected], dtype=np.float64)
    low = float(hardness_values.min()) if hardness_values.size else 0.0
    high = float(hardness_values.max()) if hardness_values.size else 1.0
    output_rows: list[dict[str, Any]] = []
    for rank, item in enumerate(selected):
        cp = int(item["codepoint"])
        base = self_train[cp]
        normalized = (float(item["hardness"]) - low) / max(high - low, 1e-8)
        repeats = 1 + int(round(4.0 * normalized))
        for repeat in range(repeats):
            row = dict(base)
            row["sample_id"] = f"hard-{cp:06X}-{rank:05d}-{repeat:02d}"
            row["sample_weight"] = 1.0 + 3.5 * normalized
            output_rows.append(row)
    if maximum_val > 0 and len(self_val) > maximum_val:
        rng = np.random.default_rng(int(seed))
        chosen = sorted(rng.choice(len(self_val), maximum_val, replace=False).tolist())
        self_val = [self_val[index] for index in chosen]
    output_rows.extend(dict(row) for row in self_val)
    write_csv(output, output_rows, DATASET_FIELDS)
    return {
        "selected_hard_codepoints": len(selected),
        "hard_rows_after_repeat": sum(1 for row in output_rows if str(row.get("sample_id", "")).startswith("hard-")),
        "cross_font_pair_rows": 0,
        "validation_rows": len(self_val),
        "total_rows": len(output_rows),
    }


def _copy_promoted(checkpoint: Path, calibration: Path, work: Path) -> None:
    active = ensure_dir(work / "model" / "generator")
    shutil.copy2(checkpoint, active / "generator_best.pt")
    if calibration.is_file():
        shutil.copy2(calibration, active / "calibration.json")


def _trim_snapshots(snapshot_dir: Path, keep: int) -> None:
    summaries = []
    for path in snapshot_dir.glob("cycle_*.json"):
        try:
            summaries.append((float(load_json(path).get("validation", {}).get("quality", -1e9)), path))
        except Exception:
            pass
    summaries.sort(reverse=True, key=lambda item: item[0])
    for _, summary_path in summaries[max(1, int(keep)):]:
        checkpoint_path = summary_path.with_suffix(".pt")
        summary_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)


def run_marathon_training(cfg: dict[str, Any]) -> dict[str, Any]:
    marathon = cfg.get("marathon", {})
    if not bool(marathon.get("enabled", True)):
        return {"enabled": False}
    set_seed(int(cfg["training"]["seed"]) + 1000)
    work = Path(cfg["paths"]["work_dir"])
    base_index = work / "dataset" / "index.csv"
    active_checkpoint = work / "model" / "generator" / "generator_best.pt"
    if not base_index.is_file() or not active_checkpoint.is_file():
        raise FileNotFoundError("Marathon requires completed prepare and base training stages.")
    root = ensure_dir(work / "marathon")
    cycles_dir = ensure_dir(root / "cycles")
    snapshots = ensure_dir(root / "snapshots")
    state_path = root / "state.json"
    fingerprint = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "base_index": sha256_file(base_index),
        "marathon": marathon,
        "base_channels": cfg["training"]["base_channels"],
    }
    previous_state = load_json(state_path) if state_path.is_file() else {}
    longrun_guard = LongRunGuard(cfg)
    if previous_state.get("fingerprint") != fingerprint:
        previous_state = {}

    size = int(marathon.get("hard_eval_size", min(256, int(cfg["inference"]["size"]))))
    eval_batch = max(1, min(6, int(cfg["training"]["phases"][0].get("batch_size", 2))))
    base_validation_metrics = _per_sample_metrics(
        active_checkpoint, base_index, cfg, split="val", size=size, batch_size=eval_batch
    )
    best_validation = _quality(base_validation_metrics)
    best_checkpoint = active_checkpoint
    no_improvement = int(previous_state.get("no_improvement", 0))
    results: list[dict[str, Any]] = list(previous_state.get("cycles", []))
    start_cycle = int(previous_state.get("next_cycle", 1))
    cycles = int(marathon.get("cycles", 18))

    for cycle in range(start_cycle, cycles + 1):
        cycle_dir = ensure_dir(cycles_dir / f"cycle_{cycle:03d}")
        cycle_summary_path = cycle_dir / "cycle_summary.json"
        if cycle_summary_path.is_file():
            summary = load_json(cycle_summary_path)
            results.append(summary)
            if summary.get("promoted"):
                best_checkpoint = Path(summary["checkpoint"])
                best_validation = summary["validation"]
                _copy_promoted(best_checkpoint, Path(summary.get("calibration", "")), work)
            continue

        train_metrics = _per_sample_metrics(
            best_checkpoint, base_index, cfg, split="train", size=size, batch_size=eval_batch
        )
        hard_index = cycle_dir / "hard_index.csv"
        hard_summary = _make_hard_index(
            base_index,
            train_metrics,
            hard_index,
            maximum_hard=int(marathon.get("hard_samples", 6000)),
            maximum_val=int(marathon.get("hard_validation_samples", 512)),
            seed=int(cfg["training"]["seed"]) + cycle,
        )
        learning_rate = max(
            float(marathon.get("minimum_learning_rate", 1.5e-6)),
            float(marathon.get("initial_learning_rate", 1.8e-5))
            * float(marathon.get("learning_rate_decay", 0.88)) ** (cycle - 1),
        )
        last_phase = cfg["training"]["phases"][-1]
        phase = {
            "name": f"hard_cycle_{cycle:03d}",
            "size": int(cfg["inference"]["size"]),
            "epochs": int(marathon.get("epochs_per_cycle", 12)),
            "batch_size": int(last_phase.get("batch_size", 1)),
            "gradient_accumulation": max(1, int(last_phase.get("gradient_accumulation", 1))),
            "learning_rate": learning_rate,
            "minimum_learning_rate": float(marathon.get("minimum_learning_rate", 1.5e-6)),
            "early_stopping_patience": int(marathon.get("epochs_per_cycle", 12)),
            "samples_per_epoch": 0,
            "adversarial": False,
        }
        training = train_generator(
            cfg,
            index_csv=hard_index,
            model_root=cycle_dir / "model",
            phases_override=[phase],
            init_checkpoint=best_checkpoint,
            resume=True,
        )
        candidate_checkpoint = Path(training["checkpoint"])
        candidate_calibration = cycle_dir / "model" / "calibration.json"
        validation_metrics = _per_sample_metrics(
            candidate_checkpoint, base_index, cfg, split="val", size=size, batch_size=eval_batch
        )
        validation = _quality(validation_metrics)
        improvement = float(validation["quality"]) - float(best_validation["quality"])
        promoted = improvement >= float(marathon.get("minimum_dice_improvement", 0.00015))
        if promoted:
            best_checkpoint = candidate_checkpoint
            best_validation = validation
            no_improvement = 0
            _copy_promoted(candidate_checkpoint, candidate_calibration, work)
            snapshot_pt = snapshots / f"cycle_{cycle:03d}.pt"
            snapshot_json = snapshots / f"cycle_{cycle:03d}.json"
            shutil.copy2(candidate_checkpoint, snapshot_pt)
            save_json(snapshot_json, {"cycle": cycle, "validation": validation, "checkpoint": str(snapshot_pt.resolve())})
            _trim_snapshots(snapshots, int(marathon.get("snapshot_keep", 8)))
        else:
            no_improvement += 1

        summary = {
            "cycle": cycle,
            "promoted": promoted,
            "improvement": improvement,
            "learning_rate": learning_rate,
            "checkpoint": str(candidate_checkpoint.resolve()),
            "calibration": str(candidate_calibration.resolve()),
            "validation": validation,
            "best_validation": best_validation,
            "hard_dataset": hard_summary,
            "training": training,
        }
        save_json(cycle_summary_path, summary)
        results.append(summary)
        save_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "next_cycle": cycle + 1,
                "no_improvement": no_improvement,
                "best_checkpoint": str(best_checkpoint.resolve()),
                "best_validation": best_validation,
                "cycles": results,
            },
        )
        longrun_guard.checkpoint_boundary()
        if no_improvement >= int(marathon.get("early_stop_cycles", 6)):
            break

    result = {
        "enabled": True,
        "configured_cycles": cycles,
        "completed_cycles": len(results),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_validation": best_validation,
        "stopped_for_plateau": no_improvement >= int(marathon.get("early_stop_cycles", 6)),
        "cycles": results,
    }
    save_json(root / "summary.json", result)
    return result


def marathon_status(cfg: dict[str, Any]) -> dict[str, Any]:
    from .ensemble import ensemble_status

    def checkpoint_progress(root: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(root.rglob("in_epoch.pt")) + sorted(root.rglob("last.pt")):
            try:
                checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
                items.append({
                    "path": str(path.resolve()),
                    "phase": checkpoint.get("phase_config", {}).get("name", path.parent.name),
                    "epoch": int(checkpoint.get("epoch", 0)),
                    "step_in_epoch": int(checkpoint.get("step_in_epoch", 0)),
                    "global_step": int(checkpoint.get("global_step", 0)),
                    "epoch_complete": bool(checkpoint.get("epoch_complete", True)),
                    "best_loss": float(checkpoint.get("best_loss", 0.0)),
                })
            except Exception as exc:
                items.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
        return items

    work = Path(cfg["paths"]["work_dir"])
    root = work / "marathon"
    refined = work / "refined"
    result: dict[str, Any] = {
        "work_dir": str(work.resolve()),
        "training": load_json(root / "summary.json") if (root / "summary.json").is_file() else {},
        "training_state": load_json(root / "state.json") if (root / "state.json").is_file() else {},
        "refinement": load_json(refined / "summary.json") if (refined / "summary.json").is_file() else {},
        "refinement_progress": load_json(refined / "progress.json") if (refined / "progress.json").is_file() else {},
        "ensemble": ensemble_status(cfg),
        "checkpoint_progress": checkpoint_progress(work / "model"),
    }
    return result
