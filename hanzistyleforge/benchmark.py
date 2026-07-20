from __future__ import annotations

import contextlib
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import CHECKPOINT_FORMAT_VERSION
from .dataset import GlyphStyleDataset
from .features import split_prediction
from .topology import topology_metrics, validate_topology
from .training import load_generator, load_refiner
from .util import atomic_write_text, ensure_dir, load_json, save_json, write_csv


BENCHMARK_FIELDS = [
    "codepoint", "unicode", "char", "dice", "ink_ratio_delta", "sdf_mae",
    "skeleton_dice", "edge_mae", "topology_checked", "topology_pass",
    "topology_score", "component_delta", "hole_delta", "endpoint_delta",
    "junction_delta", "missing_skeleton_p90", "extra_skeleton_p90",
]


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["training"].get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The HanziStyleForge benchmark requires CUDA, but torch.cuda.is_available() is False.")
    return torch.device(requested if requested.startswith("cuda") else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _dice_array(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    intersection = (pred * target).sum(axis=(1, 2))
    denominator = pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2))
    return (2.0 * intersection + 1.0) / (denominator + 1.0)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def run_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = dict(cfg.get("benchmark", {}))
    if not bool(settings.get("enabled", True)):
        return {"enabled": False}
    work = Path(cfg["paths"]["work_dir"])
    index = work / "dataset" / "index.csv"
    if not index.exists():
        raise FileNotFoundError("dataset/index.csv was not found in the current work directory. Run prepare first.")
    generator_path = work / "model" / "generator" / "generator_best.pt"
    if not generator_path.exists():
        raise FileNotFoundError("generator_best.pt was not found. Run train first.")

    size = int(cfg["inference"]["size"])
    dataset = GlyphStyleDataset(index, "val", size=size, augment=False)
    maximum = max(1, min(len(dataset), int(settings.get("maximum_glyphs", 512))))
    if maximum < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, maximum, dtype=np.int64).tolist()
        evaluated = Subset(dataset, indices)
    else:
        evaluated = dataset
    batch_size = max(1, int(settings.get("batch_size", 4)))
    device = _device(cfg)
    loader = DataLoader(
        evaluated,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    generator, _ = load_generator(generator_path, device)
    refiner = None
    refiner_path = work / "model" / "refiner" / "refiner_best.pt"
    if bool(cfg.get("refiner", {}).get("enabled", True)) and refiner_path.exists():
        refiner, _ = load_refiner(refiner_path, device)

    threshold = 0.5
    calibration_path = work / "model" / "generator" / "calibration.json"
    if calibration_path.exists():
        threshold = float(load_json(calibration_path).get("threshold", threshold))
    if refiner is not None:
        ref_calibration = work / "model" / "refiner" / "calibration.json"
        if ref_calibration.exists():
            threshold = float(load_json(ref_calibration).get("threshold", threshold))

    rows: list[dict[str, Any]] = []
    topology_limit = max(0, int(settings.get("topology_sample_count", 256)))
    topology_cfg = cfg["topology"]
    topology_size = int(topology_cfg.get("analysis_size", 144))
    checked = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="HanziStyleForge fixed validation benchmark", unit="batch"):
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_aux = batch["target_aux"].to(device, non_blocking=True)
            with _autocast(device, amp):
                output = generator(inputs)
                ink_logits, sdf_logits, skeleton_logits, edge_logits = split_prediction(output)
                probability = torch.sigmoid(ink_logits)
                if refiner is not None:
                    output = refiner(torch.cat([probability, inputs], dim=1))
                    ink_logits, sdf_logits, skeleton_logits, edge_logits = split_prediction(output)
                    probability = torch.sigmoid(ink_logits)
                sdf_prob = torch.sigmoid(sdf_logits) if sdf_logits is not None else None
                skeleton_prob = torch.sigmoid(skeleton_logits) if skeleton_logits is not None else None
                edge_prob = torch.sigmoid(edge_logits) if edge_logits is not None else None

            pred = (probability[:, 0] >= threshold).float().cpu().numpy().astype(np.uint8)
            truth = (target[:, 0] >= 0.5).float().cpu().numpy().astype(np.uint8)
            dice = _dice_array(pred, truth)
            ink_delta = np.abs(pred.mean(axis=(1, 2)) - truth.mean(axis=(1, 2)))
            aux_np = target_aux.float().cpu().numpy()
            sdf_np = sdf_prob[:, 0].float().cpu().numpy() if sdf_prob is not None else None
            skel_np = skeleton_prob[:, 0].float().cpu().numpy() if skeleton_prob is not None else None
            edge_np = edge_prob[:, 0].float().cpu().numpy() if edge_prob is not None else None
            cps = [int(value) for value in batch["codepoint"]]

            for local, cp in enumerate(cps):
                row: dict[str, Any] = {
                    "codepoint": cp,
                    "unicode": f"U+{cp:04X}",
                    "char": chr(cp),
                    "dice": float(dice[local]),
                    "ink_ratio_delta": float(ink_delta[local]),
                    "sdf_mae": float(np.abs(sdf_np[local] - aux_np[local, 1]).mean()) if sdf_np is not None else "",
                    "skeleton_dice": "",
                    "edge_mae": float(np.abs(edge_np[local] - aux_np[local, 3]).mean()) if edge_np is not None else "",
                    "topology_checked": 0,
                    "topology_pass": "",
                    "topology_score": "",
                    "component_delta": "",
                    "hole_delta": "",
                    "endpoint_delta": "",
                    "junction_delta": "",
                    "missing_skeleton_p90": "",
                    "extra_skeleton_p90": "",
                }
                if skel_np is not None:
                    skel_pred = (skel_np[local] >= 0.45).astype(np.uint8)
                    skel_true = (aux_np[local, 2] >= 0.35).astype(np.uint8)
                    row["skeleton_dice"] = float(_dice_array(skel_pred[None], skel_true[None])[0])
                if checked < topology_limit:
                    metrics = topology_metrics(
                        truth[local].astype(np.float32), pred[local].astype(np.float32),
                        size=topology_size,
                        prune_iterations=int(topology_cfg.get("prune_iterations", 1)),
                    )
                    validation = validate_topology(metrics, topology_cfg)
                    row.update({
                        "topology_checked": 1,
                        "topology_pass": 1 if validation["hard_pass"] else 0,
                        "topology_score": float(metrics["topology_score"]),
                        "component_delta": int(metrics["component_delta"]),
                        "hole_delta": int(metrics["hole_delta"]),
                        "endpoint_delta": int(metrics["endpoint_delta"]),
                        "junction_delta": int(metrics["junction_delta"]),
                        "missing_skeleton_p90": float(metrics["reference_to_candidate_p90"]),
                        "extra_skeleton_p90": float(metrics["candidate_to_reference_p90"]),
                    })
                    checked += 1
                rows.append(row)

    bench_dir = ensure_dir(work / "benchmark")
    write_csv(bench_dir / "glyph_metrics.csv", rows, BENCHMARK_FIELDS)
    dice_values = [float(row["dice"]) for row in rows]
    topology_rows = [row for row in rows if int(row["topology_checked"]) == 1]
    topology_pass_rate = (
        sum(int(row["topology_pass"]) for row in topology_rows) / max(1, len(topology_rows))
    )
    summary = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "enabled": True,
        "note": "This benchmark measures target.ttf self-reconstruction to detect model degradation. It is not a regional glyph-standard or font-engineering compliance certification.",
        "evaluated_glyphs": len(rows),
        "threshold": float(threshold),
        "generator_checkpoint": str(generator_path.resolve()),
        "refiner_checkpoint": str(refiner_path.resolve()) if refiner is not None else "",
        "dice": _quantiles(dice_values),
        "ink_ratio_delta": _quantiles([float(row["ink_ratio_delta"]) for row in rows]),
        "sdf_mae": _quantiles([float(row["sdf_mae"]) for row in rows if row["sdf_mae"] != ""]),
        "skeleton_dice": _quantiles([float(row["skeleton_dice"]) for row in rows if row["skeleton_dice"] != ""]),
        "edge_mae": _quantiles([float(row["edge_mae"]) for row in rows if row["edge_mae"] != ""]),
        "topology_checked": len(topology_rows),
        "topology_pass_rate": float(topology_pass_rate),
        "topology_score": _quantiles([float(row["topology_score"]) for row in topology_rows]),
    }
    min_dice = float(settings.get("minimum_reconstruction_dice", 0.90))
    min_topology = float(settings.get("minimum_topology_pass_rate", 0.94))
    summary["quality_gate"] = {
        "minimum_reconstruction_dice": min_dice,
        "minimum_topology_pass_rate": min_topology,
        "dice_pass": summary["dice"]["mean"] >= min_dice,
        "topology_pass": topology_pass_rate >= min_topology,
    }
    summary["quality_gate"]["overall_pass"] = bool(
        summary["quality_gate"]["dice_pass"] and summary["quality_gate"]["topology_pass"]
    )
    save_json(bench_dir / "summary.json", summary)

    gate = summary["quality_gate"]
    atomic_write_text(
        bench_dir / "index.html",
        "<!doctype html><meta charset='utf-8'><title>HanziStyleForge Benchmark</title>"
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:900px;margin:32px auto}"
        "table{border-collapse:collapse}td,th{padding:7px 12px;border:1px solid #ccc}</style>"
        "<h1>HanziStyleForge fixed validation set</h1>"
        f"<p>{html.escape(summary['note'])}</p>"
        "<table><tr><th>Metric</th><th>Result</th></tr>"
        f"<tr><td>Samples</td><td>{len(rows)}</td></tr>"
        f"<tr><td>Mean Dice</td><td>{summary['dice']['mean']:.4f}</td></tr>"
        f"<tr><td>Topology pass rate</td><td>{topology_pass_rate:.2%}</td></tr>"
        f"<tr><td>Quality gate</td><td>{'PASS' if gate['overall_pass'] else 'WARN'}</td></tr></table>"
        "<p><a href='glyph_metrics.csv'>Per-glyph metrics CSV</a> | <a href='summary.json'>JSON</a></p>",
        encoding="utf-8",
    )
    if not gate["overall_pass"] and not bool(settings.get("warn_only", True)):
        raise RuntimeError(
            f"HanziStyleForge benchmark did not pass: Dice={summary['dice']['mean']:.4f}, topology={topology_pass_rate:.2%}."
        )
    return summary
