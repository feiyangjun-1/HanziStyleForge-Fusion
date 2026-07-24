from __future__ import annotations

import contextlib
import copy
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .component_atlas import build_component_atlas
from .contour_polish import ContourDenoiseDataset, build_contour_cache
from .dataset import GlyphStyleDataset
from .features import ink_probability
from .fusion_dataset import (
    FusionDiffusionDataset,
    FusionRefinerDataset,
    StyleEncoderPretrainDataset,
    TargetStylePool,
    VQGlyphDataset,
    read_ink_image,
)
from .fusion_diffusion import DiffusionSchedule, ExponentialMovingAverage, ddim_sample
from .fusion_model import (
    ContourSequenceTransformer,
    FusionModelSpec,
    GlyphVQVAE,
    LatentDiffusionUNet,
    StyleAwareGlyphRefiner,
    StyleReferenceEncoder,
)
from .longrun import LongRunGuard
from .losses import FontLossFinal, VQReconstructionLoss, batch_binary_dice
from .model import FontStyleNetFinal
from .training import train_generator
from .util import (
    durable_replace,
    ensure_dir,
    load_json,
    read_csv,
    save_json,
    set_seed,
    sha256_file,
    write_csv,
)


FUSION_CHECKPOINT_VERSION = 301


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg.get("training", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Fusion training requires CUDA, but torch.cuda.is_available() is False")
    return torch.device(requested if requested.startswith("cuda") else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _efficient_glyph_criterion(section: dict[str, Any], fallback_weights: dict[str, Any]) -> nn.Module:
    profile = str(section.get("loss_profile", "fast_balanced_v1")).strip().lower()
    weights = section.get("image_loss_weights", section.get("loss_weights", fallback_weights))
    if profile == "legacy_full_v1":
        return FontLossFinal(weights)
    if profile == "fast_balanced_v1":
        return VQReconstructionLoss(weights)
    raise ValueError(f"unsupported glyph loss profile: {profile}")


def _scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=bool(enabled and device.type == "cuda"))


def _is_cuda_context_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "cuda error",
        "cudaerrorunknown",
        "acceleratorerror",
        "device-side assert",
        "illegal memory access",
        "unspecified launch failure",
        "driver shutting down",
    )
    return any(marker in text for marker in markers)


def _write_cuda_recovery_marker(
    phase_dir: Path,
    *,
    phase_name: str,
    epoch: int,
    step: int,
    global_step: int,
    exc: BaseException,
) -> None:
    # Do not touch CUDA here.  Once Windows resets the driver context, even a
    # harmless tensor copy can fail.  The resilient launcher will create a new
    # Python process and resume from the last durable checkpoint.
    try:
        save_json(phase_dir / "CUDA_RECOVERY.json", {
            "stage": phase_name,
            "epoch": int(epoch),
            "batch": int(step),
            "global_step": int(global_step),
            "error": str(exc),
            "action": "restart_process_and_resume_last_durable_checkpoint",
        })
    except Exception:
        pass


def _create_vq_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    request_fused: bool,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, bool]:
    """Create the stable VQ AdamW optimizer.

    VQ checkpoints may have been written by standard, foreach, or fused AdamW
    implementations. PyTorch restores backend flags and moment tensors from the
    checkpoint, and those tensors can retain a device, dtype, or memory format
    that is incompatible with a later fused CUDA kernel. The failure appears at
    the first optimizer step rather than while loading the checkpoint.

    VQ therefore uses the deterministic single-tensor AdamW path. The major VQ
    speedups (fast reconstruction loss, vector-quantizer EMA aggregation,
    channels-last execution, bounded validation, and less frequent checkpoint
    writes) remain enabled, while optimizer restore is reliable across versions.
    """
    del request_fused, device
    parameter_list = list(parameters)
    if not parameter_list:
        raise ValueError("VQ optimizer received no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameter_list,
        lr=float(learning_rate),
        betas=(0.9, 0.99),
        weight_decay=float(weight_decay),
        foreach=False,
        fused=False,
    )
    _restore_vq_optimizer_backend(optimizer, fused=False)
    return optimizer, False


def _restore_vq_optimizer_backend(optimizer: torch.optim.Optimizer, *, fused: bool) -> None:
    """Force the checkpoint-independent VQ AdamW execution path."""
    del fused
    for group in optimizer.param_groups:
        group["fused"] = False
        group["foreach"] = False
        group["capturable"] = False
        group["differentiable"] = False
    # A fused optimizer instance advertises direct GradScaler support. The VQ
    # optimizer is always constructed as standard AdamW, but removing a stale
    # attribute makes this invariant explicit for unusually patched PyTorch
    # builds and prevents GradScaler from forwarding fused-only arguments.
    if hasattr(optimizer, "_step_supports_amp_scaling"):
        try:
            delattr(optimizer, "_step_supports_amp_scaling")
        except (AttributeError, TypeError):
            setattr(optimizer, "_step_supports_amp_scaling", False)


def _prepare_vq_optimizer_state_dict(
    state_dict: dict[str, Any],
    *,
    fused: bool,
) -> dict[str, Any]:
    """Sanitize AdamW backend flags before loading any historical checkpoint."""
    del fused
    prepared = copy.deepcopy(state_dict)
    for group in prepared.get("param_groups", []):
        group["fused"] = False
        group["foreach"] = False
        group["capturable"] = False
        group["differentiable"] = False
    return prepared


def _repair_vq_optimizer_state_devices(optimizer: torch.optim.Optimizer) -> None:
    """Normalize restored standard-AdamW state without requiring fused layout."""
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            state = optimizer.state.get(parameter)
            if not state:
                continue
            for key, value in tuple(state.items()):
                if not torch.is_tensor(value):
                    continue
                if key == "step":
                    # Standard AdamW accepts a host-side scalar step counter.
                    state[key] = value.detach().to(device="cpu", dtype=torch.float32)
                    continue
                target_dtype = parameter.dtype if value.is_floating_point() else value.dtype
                migrated = value.detach().to(device=parameter.device, dtype=target_dtype)
                # Moment tensors loaded from a channels-last checkpoint can have
                # strides that differ from the current parameter. Standard AdamW
                # does not require matching strides, but matching the parameter
                # format avoids implicit copies and keeps future checkpoints tidy.
                if migrated.shape == parameter.shape and migrated.layout == torch.strided:
                    aligned = torch.empty_like(parameter, dtype=target_dtype, memory_format=torch.preserve_format)
                    aligned.copy_(migrated)
                    migrated = aligned
                state[key] = migrated


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    final = Path(path)
    ensure_dir(final.parent)
    temporary = final.with_name(final.stem + ".tmp" + final.suffix)
    torch.save(payload, temporary)
    durable_replace(temporary, final)


def _checkpoint_fingerprint(dataset_path: Path, stage: str, model_spec: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": FUSION_CHECKPOINT_VERSION,
        "dataset_sha256": sha256_file(dataset_path),
        "stage": stage,
        "model_spec": model_spec,
        "phase": phase,
    }


def _compatible(payload: dict[str, Any], fingerprint: dict[str, Any]) -> bool:
    return payload.get("fingerprint") == fingerprint


def _load_resume(directory: Path, fingerprint: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [directory / name for name in ("in_epoch.pt", "last.pt", "best.pt")]
    candidates = [path for path in candidates if path.is_file()]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if _compatible(payload, fingerprint):
                return path, payload
        except Exception:
            continue
    return None, None


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fields: list[str] = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, history, fields)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 15)
    except Exception:
        return ImageFont.load_default()


def _gray_tile(ink: np.ndarray, tile: int) -> Image.Image:
    array = np.rint((1.0 - np.asarray(ink, dtype=np.float32).clip(0.0, 1.0)) * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L").resize((tile, tile), Image.Resampling.LANCZOS).convert("RGB")


def _save_preview(
    output: Path,
    columns: list[tuple[str, np.ndarray]],
    codepoints: list[int],
    *,
    maximum: int = 8,
    tile: int = 144,
) -> None:
    count = min(maximum, len(codepoints), *(len(values) for _, values in columns))
    if count <= 0:
        return
    gap = 16
    label_height = 30
    canvas = Image.new("RGB", (len(columns) * (tile + gap) + gap, count * (tile + label_height + gap) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for row in range(count):
        y = gap + row * (tile + label_height + gap)
        for col, (label, values) in enumerate(columns):
            x = gap + col * (tile + gap)
            canvas.paste(_gray_tile(values[row], tile), (x, y))
            draw.text((x, y + tile + 4), f"{label} U+{int(codepoints[row]):04X}", fill="black", font=font)
    ensure_dir(output.parent)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    canvas.save(temporary, format="PNG")
    durable_replace(temporary, output)


def _loader(dataset, batch_size: int, workers: int, *, shuffle: bool = True) -> DataLoader:
    worker_count = max(0, int(workers))
    options: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": bool(shuffle),
        "num_workers": worker_count,
        "pin_memory": True,
        "persistent_workers": bool(worker_count > 0),
        "drop_last": bool(shuffle and len(dataset) >= batch_size),
    }
    if worker_count > 0:
        # Three prefetched batches per worker keeps the GPU fed without the
        # excessive RAM growth seen with much deeper queues.  The worker
        # callback is module-level and therefore safe with Windows spawn.
        options["prefetch_factor"] = max(2, int(os.environ.get("HSF_PREFETCH_FACTOR", "4")))
        options["worker_init_fn"] = _seed_loader_worker
    return DataLoader(dataset, **options)


def _seed_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # Prevent each image worker from creating another full PyTorch CPU pool.
    torch.set_num_threads(1)


def _history_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _style_plateau_state(
    history: list[dict[str, Any]],
    *,
    through_epoch: int,
    minimum_relative_improvement: float,
) -> tuple[float, int, int]:
    """Reconstruct significant-improvement state from a legacy history CSV.

    Fusion 2.0 checkpoints did not store style early-stop counters.  Rebuilding
    them from history makes the 2.1 upgrade resume-compatible with a running
    Style Encoder instead of forcing a restart.
    """

    significant_best = math.inf
    last_significant_epoch = 0
    stale_epochs = 0
    relative = max(0.0, float(minimum_relative_improvement))
    ordered = sorted(
        history,
        key=lambda row: int(_history_float(row, "epoch", 0.0)),
    )
    for row in ordered:
        epoch = int(_history_float(row, "epoch", 0.0))
        if epoch <= 0 or epoch > int(through_epoch):
            continue
        value = _history_float(row, "val_loss")
        if not math.isfinite(value):
            continue
        threshold = significant_best * (1.0 - relative)
        if not math.isfinite(significant_best) or value < threshold:
            significant_best = value
            last_significant_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs = max(0, epoch - last_significant_epoch)
    return significant_best, last_significant_epoch, stale_epochs


def _style_quality_gate(
    history: list[dict[str, Any]],
    *,
    window: int,
    minimum_positive_similarity: float,
    maximum_negative_similarity: float,
) -> tuple[bool, dict[str, float]]:
    recent = history[-max(1, int(window)) :]
    positives = [
        _history_float(row, "val_positive_similarity")
        for row in recent
        if math.isfinite(_history_float(row, "val_positive_similarity"))
    ]
    negatives = [
        _history_float(row, "val_negative_similarity")
        for row in recent
        if math.isfinite(_history_float(row, "val_negative_similarity"))
    ]
    if not positives or not negatives:
        return False, {
            "median_positive_similarity": math.nan,
            "median_negative_similarity": math.nan,
        }
    positive = float(np.median(np.asarray(positives, dtype=np.float64)))
    negative = float(np.median(np.asarray(negatives, dtype=np.float64)))
    return (
        positive >= float(minimum_positive_similarity)
        and negative <= float(maximum_negative_similarity)
    ), {
        "median_positive_similarity": positive,
        "median_negative_similarity": negative,
    }


def _normalized_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(F.normalize(a, dim=-1), F.normalize(b, dim=-1), dim=-1)


def _expert_diversity(experts: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(experts, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(similarity.shape[-1], device=similarity.device, dtype=similarity.dtype).unsqueeze(0)
    return ((similarity - eye) * (1.0 - eye)).square().mean()


def train_style_encoder(cfg: dict[str, Any]) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    fusion = cfg.get("fusion", {})
    style_cfg = fusion.get("style_encoder", {})
    root = ensure_dir(work / "fusion" / "style")
    summary_path = root / "summary.json"
    index_path = work / "dataset" / "index.csv"
    if summary_path.is_file() and (root / "best.pt").is_file() and (root / "style_bank.pt").is_file():
        try:
            return load_json(summary_path)
        except Exception:
            pass
    device = _device(cfg)
    amp = bool(cfg.get("training", {}).get("amp", True) and device.type == "cuda")
    workers = int(cfg.get("training", {}).get("workers", 0))
    model_spec = {
        "base": int(style_cfg.get("base_channels", 32)),
        "style_dim": int(fusion.get("style_dim", 256)),
        "expert_count": int(fusion.get("expert_count", 8)),
        "heads": int(style_cfg.get("heads", 8)),
        "synthetic_parameters": 7,
    }
    phase = {
        "style_size": int(style_cfg.get("size", 128)),
        "references": int(style_cfg.get("references_per_set", 8)),
        "epochs": int(style_cfg.get("epochs", 160)),
        "batch_size": int(style_cfg.get("batch_size", 4)),
        "learning_rate": float(style_cfg.get("learning_rate", 1.5e-4)),
        "virtual_length": int(style_cfg.get("virtual_length", 16000)),
    }
    fingerprint = _checkpoint_fingerprint(index_path, "style_encoder", model_spec, phase)
    model = StyleReferenceEncoder(
        base=model_spec["base"],
        style_dim=model_spec["style_dim"],
        expert_count=model_spec["expert_count"],
        heads=model_spec["heads"],
        synthetic_parameter_count=7,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase["learning_rate"],
        betas=(0.9, 0.99),
        weight_decay=float(style_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.65, patience=max(3, int(style_cfg.get("lr_patience", 8))), min_lr=1e-6
    )
    scaler = _scaler(device, amp)
    train_dataset = StyleEncoderPretrainDataset(
        index_path,
        style_size=phase["style_size"],
        references_per_set=phase["references"],
        virtual_length=phase["virtual_length"],
        seed=int(cfg["training"].get("seed", 20260719)),
    )
    val_dataset = StyleEncoderPretrainDataset(
        index_path,
        style_size=phase["style_size"],
        references_per_set=phase["references"],
        virtual_length=max(256, phase["batch_size"] * 64),
        seed=int(cfg["training"].get("seed", 20260719)) + 991,
    )
    train_loader = _loader(train_dataset, phase["batch_size"], workers, shuffle=True)
    val_loader = _loader(val_dataset, phase["batch_size"], workers, shuffle=False)
    best = math.inf
    start_epoch = 1
    global_step = 0
    history = _read_history(root / "history.csv")
    _, resume = _load_resume(root, fingerprint)
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scheduler"):
            scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        global_step = int(resume.get("global_step", 0))
        best = float(resume.get("best", math.inf))
    early_cfg = style_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_minimum_epochs = max(1, int(early_cfg.get("minimum_epochs", 100)))
    early_patience = max(1, int(early_cfg.get("patience", 24)))
    early_relative_improvement = max(
        0.0, float(early_cfg.get("minimum_relative_improvement", 0.002))
    )
    early_quality_window = max(1, int(early_cfg.get("quality_window", 20)))
    early_positive_minimum = float(early_cfg.get("positive_similarity_minimum", 0.999))
    early_negative_maximum = float(early_cfg.get("negative_similarity_maximum", 0.15))
    significant_best, last_significant_epoch, stale_epochs = _style_plateau_state(
        history,
        through_epoch=start_epoch - 1,
        minimum_relative_improvement=early_relative_improvement,
    )
    if resume is not None:
        significant_best = float(resume.get("significant_best", significant_best))
        last_significant_epoch = int(
            resume.get("last_significant_epoch", last_significant_epoch)
        )
        stale_epochs = int(resume.get("stale_epochs", stale_epochs))
    guard = LongRunGuard(cfg)
    checkpoint_every = max(20, int(style_cfg.get("checkpoint_every_steps", 100)))
    completed_epoch = start_epoch - 1
    stop_reason = "maximum_epochs"
    early_stopped = False

    def losses(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a = batch["positive_a"].to(device, non_blocking=True)
        b = batch["positive_b"].to(device, non_blocking=True)
        n = batch["negative"].to(device, non_blocking=True)
        positive_parameters = batch["positive_parameters"].to(device)
        negative_parameters = batch["negative_parameters"].to(device)
        out_a = model(a)
        out_b = model(b)
        out_n = model(n)
        similarity_positive = _normalized_cosine(out_a["global"], out_b["global"])
        similarity_negative = _normalized_cosine(out_a["global"], out_n["global"])
        consistency = (1.0 - similarity_positive).mean()
        triplet = F.relu(float(style_cfg.get("contrastive_margin", 0.30)) - similarity_positive + similarity_negative).mean()
        regression = (
            F.smooth_l1_loss(out_a["synthetic"], positive_parameters)
            + F.smooth_l1_loss(out_b["synthetic"], positive_parameters)
            + F.smooth_l1_loss(out_n["synthetic"], negative_parameters)
        ) / 3.0
        diversity = (_expert_diversity(out_a["experts"]) + _expert_diversity(out_b["experts"])) / 2.0
        glyph_variance = F.relu(0.06 - out_a["glyph_tokens"].std(dim=1).mean())
        total = consistency + 1.25 * triplet + 0.65 * regression + 0.06 * diversity + 0.05 * glyph_variance
        return total, {
            "consistency": consistency,
            "triplet": triplet,
            "regression": regression,
            "diversity": diversity,
            "positive_similarity": similarity_positive.mean(),
            "negative_similarity": similarity_negative.mean(),
        }

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = {}
        seen = 0
        for batch in val_loader:
            with _autocast(device, amp):
                loss, pieces = losses(batch)
            count = int(batch["positive_a"].shape[0])
            seen += count
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item()) * count
            for key, value in pieces.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
        return {key: value / max(1, seen) for key, value in totals.items()}

    for epoch in range(start_epoch, phase["epochs"] + 1):
        model.train()
        totals = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"style {epoch:03d}/{phase['epochs']:03d}", unit="batch")
        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, _ = losses(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch["positive_a"].shape[0])
            totals += float(loss.detach().item()) * count
            seen += count
            global_step += 1
            progress.set_postfix(loss=f"{totals/max(1,seen):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if global_step % checkpoint_every == 0:
                payload = {
                    "fingerprint": fingerprint,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch - 1,
                    "global_step": global_step,
                    "best": best,
                    "significant_best": significant_best,
                    "last_significant_epoch": last_significant_epoch,
                    "stale_epochs": stale_epochs,
                    "model_spec": model_spec,
                }
                _atomic_torch_save(payload, root / "in_epoch.pt")
                guard.checkpoint_boundary()
        validation = evaluate()
        scheduler.step(validation["loss"])
        improved = validation["loss"] < best
        if improved:
            best = validation["loss"]
        significant_threshold = significant_best * (1.0 - early_relative_improvement)
        if not math.isfinite(significant_best) or validation["loss"] < significant_threshold:
            significant_best = validation["loss"]
            last_significant_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs = max(0, epoch - last_significant_epoch)
        payload = {
            "fingerprint": fingerprint,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best": best,
            "significant_best": significant_best,
            "last_significant_epoch": last_significant_epoch,
            "stale_epochs": stale_epochs,
            "model_spec": model_spec,
            "validation": validation,
        }
        _atomic_torch_save(payload, root / "last.pt")
        if improved:
            _atomic_torch_save(payload, root / "best.pt")
        (root / "in_epoch.pt").unlink(missing_ok=True)
        history.append({
            "epoch": epoch,
            "train_loss": totals / max(1, seen),
            **{f"val_{key}": value for key, value in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best": int(improved),
            "early_stop_stale_epochs": stale_epochs,
            "early_stop_significant_best": significant_best,
        })
        _write_history(root / "history.csv", history)
        completed_epoch = epoch
        guard.checkpoint_boundary()
        quality_ready, quality_metrics = _style_quality_gate(
            history,
            window=early_quality_window,
            minimum_positive_similarity=early_positive_minimum,
            maximum_negative_similarity=early_negative_maximum,
        )
        if (
            early_enabled
            and epoch >= early_minimum_epochs
            and stale_epochs >= early_patience
            and quality_ready
        ):
            stop_reason = "quality_gated_validation_plateau"
            early_stopped = True
            save_json(root / "EARLY_STOP.json", {
                "stage": "style_encoder",
                "reason": stop_reason,
                "epoch": epoch,
                "maximum_epochs": phase["epochs"],
                "stale_epochs": stale_epochs,
                "last_significant_epoch": last_significant_epoch,
                "significant_best_validation_loss": significant_best,
                "raw_best_validation_loss": best,
                "minimum_relative_improvement": early_relative_improvement,
                "patience": early_patience,
                "minimum_epochs": early_minimum_epochs,
                "quality_window": early_quality_window,
                "positive_similarity_minimum": early_positive_minimum,
                "negative_similarity_maximum": early_negative_maximum,
                **quality_metrics,
            })
            print(
                "Style Encoder early stopping: validation has reached a plateau, "
                f"epoch={epoch}, stale={stale_epochs}, "
                f"positive={quality_metrics['median_positive_similarity']:.6f}, "
                f"negative={quality_metrics['median_negative_similarity']:.6f}"
            )
            break
        if epoch >= 20 and optimizer.param_groups[0]["lr"] <= 1.05e-6:
            stop_reason = "learning_rate_floor"
            early_stopped = True
            save_json(root / "EARLY_STOP.json", {
                "stage": "style_encoder",
                "reason": stop_reason,
                "epoch": epoch,
                "maximum_epochs": phase["epochs"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "raw_best_validation_loss": best,
            })
            break

    best_payload = torch.load(root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    model.eval()
    pool = TargetStylePool(index_path, split="train")
    group_count = max(4, int(style_cfg.get("style_bank_groups", 32)))
    reference_count = phase["references"]
    bank_experts: list[torch.Tensor] = []
    bank_globals: list[torch.Tensor] = []
    with torch.no_grad():
        for group in range(group_count):
            paths = pool.deterministic_paths(reference_count, seed=int(cfg["training"].get("seed", 0)) + group * 101)
            glyphs = np.stack([read_ink_image(path, phase["style_size"]) for path in paths], axis=0)[None, :, None]
            tensor = torch.from_numpy(glyphs.astype(np.float32)).to(device)
            output = model(tensor)
            bank_experts.append(output["experts"][0].float().cpu())
            bank_globals.append(output["global"][0].float().cpu())
    style_bank = {
        "version": FUSION_CHECKPOINT_VERSION,
        "model_spec": model_spec,
        "experts": torch.stack(bank_experts),
        "global": torch.stack(bank_globals),
        "mean_experts": torch.stack(bank_experts).mean(dim=0),
        "mean_global": torch.stack(bank_globals).mean(dim=0),
        "reference_count": reference_count,
        "group_count": group_count,
    }
    _atomic_torch_save(style_bank, root / "style_bank.pt")
    summary = {
        "enabled": True,
        "checkpoint": str((root / "best.pt").resolve()),
        "style_bank": str((root / "style_bank.pt").resolve()),
        "best_validation_loss": best,
        "completed_epoch": completed_epoch,
        "maximum_epochs": phase["epochs"],
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "model_spec": model_spec,
        "method": "target-only synthetic-style contrastive reference encoder",
    }
    save_json(summary_path, summary)
    return summary


def load_style_encoder(cfg: dict[str, Any], device: torch.device | str) -> tuple[StyleReferenceEncoder, dict[str, Any]]:
    path = Path(cfg["paths"]["work_dir"]) / "fusion" / "style" / "best.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    spec = payload["model_spec"]
    model = StyleReferenceEncoder(
        base=int(spec["base"]),
        style_dim=int(spec["style_dim"]),
        expert_count=int(spec["expert_count"]),
        heads=int(spec["heads"]),
        synthetic_parameter_count=int(spec.get("synthetic_parameters", 7)),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), spec


def load_style_bank(cfg: dict[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    path = Path(cfg["paths"]["work_dir"]) / "fusion" / "style" / "style_bank.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in payload.items()}


def _style_experts_from_bank(
    batch: dict[str, Any],
    style_bank: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    groups = style_bank.get("experts")
    if not isinstance(groups, torch.Tensor) or groups.ndim != 3 or groups.shape[0] == 0:
        mean = style_bank["mean_experts"].to(device, non_blocking=True)
        return mean.unsqueeze(0).expand(int(batch_size), -1, -1)
    groups = groups.to(device, non_blocking=True)
    indexes = batch.get("style_index")
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        indexes = torch.remainder(indexes, groups.shape[0])
    else:
        indexes = torch.arange(int(batch_size), device=device, dtype=torch.long) % groups.shape[0]
    return groups.index_select(0, indexes)


def _vq_model(cfg: dict[str, Any]) -> GlyphVQVAE:
    fusion = cfg.get("fusion", {})
    vq = fusion.get("vq", {})
    return GlyphVQVAE(
        in_channels=4,
        out_channels=4,
        base=int(vq.get("base_channels", 48)),
        latent_channels=int(fusion.get("latent_channels", 32)),
        embeddings=int(fusion.get("vq_embeddings", 1024)),
        decay=float(vq.get("codebook_decay", 0.995)),
    )


def _vq_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    fusion = cfg.get("fusion", {})
    vq = fusion.get("vq", {})
    return {
        "base_channels": int(vq.get("base_channels", 48)),
        "latent_channels": int(fusion.get("latent_channels", 32)),
        "embeddings": int(fusion.get("vq_embeddings", 1024)),
        "codebook_decay": float(vq.get("codebook_decay", 0.995)),
    }


def train_vqvae(cfg: dict[str, Any]) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    fusion = cfg.get("fusion", {})
    vq_cfg = fusion.get("vq", {})
    root = ensure_dir(work / "fusion" / "vq")
    index_path = work / "dataset" / "index.csv"
    device = _device(cfg)
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    workers = int(cfg["training"].get("workers", 0))
    model_spec = _vq_spec(cfg)
    model = _vq_model(cfg).to(device)
    channels_last = bool(vq_cfg.get("channels_last", True) and device.type == "cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    loss_profile = str(vq_cfg.get("loss_profile", "fast_balanced_v1")).strip().lower()
    if loss_profile == "legacy_full_v1":
        criterion = FontLossFinal(vq_cfg.get("loss_weights", cfg.get("loss", {}).get("weights", {}))).to(device)
    elif loss_profile == "fast_balanced_v1":
        criterion = VQReconstructionLoss(vq_cfg.get("loss_weights", {})).to(device)
    else:
        raise ValueError(f"unsupported VQ loss_profile: {loss_profile}")
    phases = list(vq_cfg.get("phases", [])) or [{
        "name": "vq256", "size": 256, "epochs": 160, "batch_size": 4,
        "gradient_accumulation": 1, "learning_rate": 2e-4, "patience": 36,
    }]
    previous_best: Path | None = None
    phase_results: list[dict[str, Any]] = []
    guard = LongRunGuard(cfg)

    for phase in phases:
        phase_dir = ensure_dir(root / str(phase["name"]))
        fingerprint = _checkpoint_fingerprint(index_path, "vq", model_spec, dict(phase))
        if (phase_dir / "completed.json").is_file() and (phase_dir / "best.pt").is_file():
            payload = torch.load(phase_dir / "best.pt", map_location=device, weights_only=False)
            if _compatible(payload, fingerprint):
                model.load_state_dict(payload["model"], strict=True)
                previous_best = phase_dir / "best.pt"
                phase_results.append({"name": phase["name"], "checkpoint": str(previous_best.resolve()), "reused": True})
                continue
        resume_path, resume = _load_resume(phase_dir, fingerprint)
        if resume is None and previous_best is not None:
            previous = torch.load(previous_best, map_location=device, weights_only=False)
            model.load_state_dict(previous["model"], strict=True)
        optimizer, fused_optimizer = _create_vq_optimizer(
            model.parameters(),
            learning_rate=float(phase["learning_rate"]),
            weight_decay=float(vq_cfg.get("weight_decay", 1e-5)),
            request_fused=bool(vq_cfg.get("fused_optimizer", True)),
            device=device,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=max(3, int(phase.get("lr_patience", 8))), min_lr=1e-6
        )
        scaler = _scaler(device, amp)
        start_epoch = 1
        best = math.inf
        global_step = 0
        no_improvement = 0
        if resume is not None:
            model.load_state_dict(resume["model"], strict=True)
            restored_optimizer_state = _prepare_vq_optimizer_state_dict(
                resume["optimizer"], fused=fused_optimizer
            )
            optimizer.load_state_dict(restored_optimizer_state)
            # Reapply backend flags and repair legacy state placement. Fused
            # AdamW requires its scalar step tensors on the CUDA device.
            _restore_vq_optimizer_backend(optimizer, fused=fused_optimizer)
            _repair_vq_optimizer_state_devices(optimizer)
            previous_loss_profile = str(resume.get("loss_profile", "legacy_full_v1")).strip().lower()
            loss_profile_changed = previous_loss_profile != loss_profile
            if resume.get("scheduler") and not loss_profile_changed:
                scheduler.load_state_dict(resume["scheduler"])
            if resume.get("scaler"):
                scaler.load_state_dict(resume["scaler"])
            start_epoch = int(resume.get("epoch", 0)) + 1
            # Validation loss scales differ between the legacy generator loss
            # and the VQ-specific fast loss. Keep learned weights and optimizer
            # state, but reset plateau tracking exactly once after an upgrade.
            best = math.inf if loss_profile_changed else float(resume.get("best", math.inf))
            global_step = int(resume.get("global_step", 0))
            no_improvement = 0 if loss_profile_changed else int(resume.get("no_improvement", 0))
        print(
            "VQ optimizer backend: "
            + ("fused AdamW (AMP-compatible)" if fused_optimizer else "standard AdamW")
        )
        size = int(phase["size"])
        train_loader = _loader(
            VQGlyphDataset(index_path, split="train", size=size, augment=True),
            int(phase["batch_size"]), workers, shuffle=True,
        )
        val_loader = _loader(
            VQGlyphDataset(index_path, split="val", size=size, augment=False),
            max(1, int(phase["batch_size"])), workers, shuffle=False,
        )
        accumulation = max(1, int(phase.get("gradient_accumulation", 1)))
        history = _read_history(phase_dir / "history.csv")
        checkpoint_every = max(20, int(vq_cfg.get("checkpoint_every_steps", 240)))
        validation_batches = max(0, int(vq_cfg.get("validation_batches", 64)))
        metric_sync_every = max(4, int(vq_cfg.get("metric_sync_every_steps", 16)))

        # Every VQ resolution uses significant-improvement early stopping.
        # Per-phase values override the shared defaults.
        shared_early_cfg = vq_cfg.get("early_stopping", {})
        early_cfg = {**shared_early_cfg, **phase.get("early_stopping", {})}
        early_enabled = bool(early_cfg.get("enabled", True))
        early_minimum_epochs = max(1, int(early_cfg.get("minimum_epochs", 24)))
        early_patience = max(1, int(early_cfg.get("patience", phase.get("patience", 12))))
        early_relative_improvement = max(
            0.0, float(early_cfg.get("minimum_relative_improvement", 0.001))
        )
        significant_best, last_significant_epoch, stale_epochs = _style_plateau_state(
            history,
            through_epoch=start_epoch - 1,
            minimum_relative_improvement=early_relative_improvement,
        )
        early_stopped = False
        stop_reason = "maximum_epochs"

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            model.eval()
            total_loss = total_dice = total_perplexity = 0.0
            seen = 0
            for batch_index, batch in enumerate(val_loader):
                if validation_batches and batch_index >= validation_batches:
                    break
                target_aux = batch["target_aux"].to(device, non_blocking=True)
                if channels_last:
                    target_aux = target_aux.contiguous(memory_format=torch.channels_last)
                with _autocast(device, amp):
                    output = model(target_aux, update_codebook=False)
                    glyph_loss, _ = criterion(
                        output["reconstruction"], target_aux[:, 0:1], target_aux=target_aux
                    )
                    loss = glyph_loss + float(vq_cfg.get("commitment_weight", 0.25)) * output["commitment"]
                    probability = torch.sigmoid(output["reconstruction"][:, :1])
                count = target_aux.shape[0]
                seen += count
                total_loss += float(loss.item()) * count
                total_dice += float(batch_binary_dice(probability, target_aux[:, 0:1]).mean().item()) * count
                total_perplexity += float(output["perplexity"].item()) * count
            return {
                "loss": total_loss / max(1, seen),
                "dice": total_dice / max(1, seen),
                "perplexity": total_perplexity / max(1, seen),
            }

        completed_before_resume = start_epoch - 1
        if (
            early_enabled
            and completed_before_resume >= early_minimum_epochs
            and stale_epochs >= early_patience
        ):
            early_stopped = True
            stop_reason = "significant_validation_plateau"
            save_json(phase_dir / "EARLY_STOP.json", {
                "stage": str(phase["name"]),
                "reason": stop_reason,
                "epoch": completed_before_resume,
                "maximum_epochs": int(phase["epochs"]),
                "stale_epochs": stale_epochs,
                "last_significant_epoch": last_significant_epoch,
                "significant_best_validation_loss": significant_best,
                "raw_best_validation_loss": best,
                "minimum_relative_improvement": early_relative_improvement,
                "patience": early_patience,
                "minimum_epochs": early_minimum_epochs,
            })
            print(
                f"{phase['name']} early stopping: validation has reached a significant-improvement plateau, "
                f"epoch={completed_before_resume}, stale={stale_epochs}."
            )
            epoch_range: Iterable[int] = ()
        else:
            epoch_range = range(start_epoch, int(phase["epochs"]) + 1)

        for epoch in epoch_range:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = 0.0
            total_perplexity = 0.0
            seen = 0
            metric_seen = 0
            window_loss = torch.zeros((), device=device, dtype=torch.float32)
            window_perplexity = torch.zeros((), device=device, dtype=torch.float32)
            window_seen = 0
            progress = tqdm(train_loader, desc=f"{phase['name']} {epoch:03d}/{int(phase['epochs']):03d}", unit="batch")
            step = 0
            try:
                for step, batch in enumerate(progress, start=1):
                    target_aux = batch["target_aux"].to(device, non_blocking=True)
                    if channels_last:
                        target_aux = target_aux.contiguous(memory_format=torch.channels_last)
                    with _autocast(device, amp):
                        output = model(target_aux, update_codebook=True)
                        glyph_loss, _ = criterion(
                            output["reconstruction"], target_aux[:, 0:1], target_aux=target_aux
                        )
                        loss = glyph_loss + float(vq_cfg.get("commitment_weight", 0.25)) * output["commitment"]
                        scaled = loss / accumulation
                    scaler.scale(scaled).backward()
                    if step % accumulation == 0 or step == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if global_step % checkpoint_every == 0:
                            _atomic_torch_save({
                                "fingerprint": fingerprint, "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                "scaler": scaler.state_dict(), "epoch": epoch - 1, "global_step": global_step,
                                "best": best, "no_improvement": no_improvement, "model_spec": model_spec,
                                "significant_best": significant_best,
                                "last_significant_epoch": last_significant_epoch,
                                "stale_epochs": stale_epochs,
                                "loss_profile": loss_profile, "channels_last": channels_last,
                                "optimizer_backend": "fused" if fused_optimizer else "standard",
                            }, phase_dir / "in_epoch.pt")
                            guard.checkpoint_boundary()
                    count = int(target_aux.shape[0])
                    # Keep metrics on the GPU and synchronize only occasionally.
                    # Per-batch .item() calls serialize the CUDA stream and made
                    # Windows driver resets appear at tqdm formatting sites.
                    window_loss.add_(loss.detach().float() * count)
                    window_perplexity.add_(output["perplexity"].detach().float() * count)
                    window_seen += count
                    seen += count
                    if step % metric_sync_every == 0 or step == len(train_loader):
                        values = torch.stack((window_loss, window_perplexity)).cpu().tolist()
                        totals += float(values[0])
                        total_perplexity += float(values[1])
                        metric_seen += window_seen
                        window_loss.zero_()
                        window_perplexity.zero_()
                        window_seen = 0
                        progress.set_postfix(
                            loss=f"{totals/max(1,metric_seen):.4f}",
                            ppl=f"{total_perplexity/max(1,metric_seen):.0f}",
                        )
                    guard.runtime_boundary()
            except (RuntimeError, torch.AcceleratorError) as exc:
                if _is_cuda_context_error(exc):
                    _write_cuda_recovery_marker(
                        phase_dir,
                        phase_name=str(phase["name"]),
                        epoch=epoch,
                        step=step,
                        global_step=global_step,
                        exc=exc,
                    )
                    raise RuntimeError(
                        "The CUDA driver context was reset during VQ training. "
                        "The resilient launcher will restart the process and resume from "
                        "the last durable checkpoint. If this repeats, reboot Windows and "
                        "use the latest NVIDIA Studio Driver."
                    ) from exc
                raise
            validation = evaluate()
            scheduler.step(validation["loss"])
            improved = validation["loss"] < best
            if improved:
                best = validation["loss"]
                no_improvement = 0
            else:
                no_improvement += 1
            significant_threshold = significant_best * (1.0 - early_relative_improvement)
            if not math.isfinite(significant_best) or validation["loss"] < significant_threshold:
                significant_best = validation["loss"]
                last_significant_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs = max(0, epoch - last_significant_epoch)
            payload = {
                "fingerprint": fingerprint, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
                "best": best, "no_improvement": no_improvement, "model_spec": model_spec,
                "significant_best": significant_best,
                "last_significant_epoch": last_significant_epoch,
                "stale_epochs": stale_epochs,
                "validation": validation, "loss_profile": loss_profile,
                "channels_last": channels_last,
                "optimizer_backend": "fused" if fused_optimizer else "standard",
            }
            _atomic_torch_save(payload, phase_dir / "last.pt")
            if improved:
                _atomic_torch_save(payload, phase_dir / "best.pt")
            (phase_dir / "in_epoch.pt").unlink(missing_ok=True)
            history.append({
                "epoch": epoch, "train_loss": totals / max(1, metric_seen),
                "val_loss": validation["loss"], "val_dice": validation["dice"],
                "val_perplexity": validation["perplexity"],
                "learning_rate": optimizer.param_groups[0]["lr"], "best": int(improved),
                "early_stop_stale_epochs": stale_epochs,
                "early_stop_significant_best": significant_best,
            })
            _write_history(phase_dir / "history.csv", history)
            if epoch == 1 or epoch % int(vq_cfg.get("preview_every", 4)) == 0 or improved:
                model.eval()
                batch = next(iter(val_loader))
                target_aux = batch["target_aux"].to(device, non_blocking=True)
                if channels_last:
                    target_aux = target_aux.contiguous(memory_format=torch.channels_last)
                with torch.no_grad(), _autocast(device, amp):
                    prediction = torch.sigmoid(model(target_aux, update_codebook=False)["reconstruction"][:, :1])
                _save_preview(
                    phase_dir / "previews" / f"epoch_{epoch:03d}.png",
                    [("VQ reconstruction", prediction[:, 0].float().cpu().numpy()),
                     ("target truth", target_aux[:, 0].float().cpu().numpy())],
                    [int(value) for value in batch["codepoint"]],
                )
            guard.checkpoint_boundary()
            if (
                early_enabled
                and epoch >= early_minimum_epochs
                and stale_epochs >= early_patience
            ):
                early_stopped = True
                stop_reason = "significant_validation_plateau"
                save_json(phase_dir / "EARLY_STOP.json", {
                    "stage": str(phase["name"]),
                    "reason": stop_reason,
                    "epoch": epoch,
                    "maximum_epochs": int(phase["epochs"]),
                    "stale_epochs": stale_epochs,
                    "last_significant_epoch": last_significant_epoch,
                    "significant_best_validation_loss": significant_best,
                    "raw_best_validation_loss": best,
                    "minimum_relative_improvement": early_relative_improvement,
                    "patience": early_patience,
                    "minimum_epochs": early_minimum_epochs,
                })
                print(
                    f"{phase['name']} early stopping: validation has reached a significant-improvement plateau, "
                    f"epoch={epoch}, stale={stale_epochs}."
                )
                break
            if not early_enabled and no_improvement >= int(phase.get("patience", 36)):
                stop_reason = "raw_validation_plateau"
                break
        save_json(phase_dir / "completed.json", {
            "fingerprint": fingerprint,
            "best": best,
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "significant_best": significant_best,
            "last_significant_epoch": last_significant_epoch,
            "stale_epochs": stale_epochs,
        })
        previous_best = phase_dir / "best.pt"
        best_payload = torch.load(previous_best, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model"], strict=True)
        phase_results.append({
            "name": phase["name"],
            "checkpoint": str(previous_best.resolve()),
            "best": best,
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "stale_epochs": stale_epochs,
        })

    assert previous_best is not None
    final_path = root / "vq_best.pt"
    shutil.copy2(previous_best, final_path)
    summary = {
        "enabled": True,
        "checkpoint": str(final_path.resolve()),
        "model_spec": model_spec,
        "phases": phase_results,
        "method": "target-only VQ stroke-prior autoencoder",
        "loss_profile": loss_profile,
        "channels_last": channels_last,
    }
    save_json(root / "summary.json", summary)
    return summary


def load_vqvae(cfg: dict[str, Any], device: torch.device | str) -> GlyphVQVAE:
    path = Path(cfg["paths"]["work_dir"]) / "fusion" / "vq" / "vq_best.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = _vq_model(cfg)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def _diffusion_model(cfg: dict[str, Any]) -> LatentDiffusionUNet:
    fusion = cfg.get("fusion", {})
    diffusion = fusion.get("diffusion", {})
    return LatentDiffusionUNet(
        latent_channels=int(fusion.get("latent_channels", 32)),
        content_channels=10,
        base=int(diffusion.get("base_channels", 96)),
        content_base=int(diffusion.get("content_base_channels", 40)),
        style_dim=int(fusion.get("style_dim", 256)),
        expert_count=int(fusion.get("expert_count", 8)),
        time_dim=int(diffusion.get("time_dim", 256)),
    )


def _diffusion_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    fusion = cfg.get("fusion", {})
    diffusion = fusion.get("diffusion", {})
    return {
        "latent_channels": int(fusion.get("latent_channels", 32)),
        "base_channels": int(diffusion.get("base_channels", 96)),
        "content_base_channels": int(diffusion.get("content_base_channels", 40)),
        "style_dim": int(fusion.get("style_dim", 256)),
        "expert_count": int(fusion.get("expert_count", 8)),
        "time_dim": int(diffusion.get("time_dim", 256)),
        "diffusion_steps": int(fusion.get("diffusion_steps", 1000)),
        "schedule": str(diffusion.get("schedule", "cosine")),
    }


def _sample_timesteps(batch: int, schedule: DiffusionSchedule, recon_limit: int, device: torch.device) -> torch.Tensor:
    # Half of the batch is intentionally sampled from low/no-medium noise so
    # decoded x0 supervision remains frequent even for batch size one.
    if random.random() < 0.52:
        return torch.randint(0, min(schedule.timesteps, max(2, recon_limit + 1)), (batch,), device=device)
    return torch.randint(0, schedule.timesteps, (batch,), device=device)


def _min_snr_weight(schedule: DiffusionSchedule, timesteps: torch.Tensor, gamma: float) -> torch.Tensor:
    cumulative = schedule.alphas_cumprod.to(timesteps.device).gather(0, timesteps)
    snr = cumulative / (1.0 - cumulative).clamp_min(1e-8)
    return torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr.clamp_min(1e-8)


def _diffusion_training_loss(
    *,
    model: LatentDiffusionUNet,
    vq: GlyphVQVAE,
    style_encoder: StyleReferenceEncoder,
    style_bank: dict[str, Any],
    schedule: DiffusionSchedule,
    criterion: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    cfg: dict[str, Any],
    amp: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    diffusion_cfg = cfg.get("fusion", {}).get("diffusion", {})
    proxy = batch["proxy"].to(device, non_blocking=True)
    target_aux = batch["target_aux"].to(device, non_blocking=True)
    with torch.no_grad():
        target_latent = vq.encode(target_aux, quantize=True, update_codebook=False)["quantized"]
        experts = _style_experts_from_bank(batch, style_bank, device, target_aux.shape[0])
    if model.training and random.random() < float(diffusion_cfg.get("style_dropout", 0.08)):
        experts = torch.zeros_like(experts)
    recon_limit = int(diffusion_cfg.get("reconstruction_timestep_limit", 300))
    timesteps = _sample_timesteps(target_latent.shape[0], schedule, recon_limit, device)
    noise = torch.randn_like(target_latent)
    noisy = schedule.q_sample(target_latent, timesteps, noise)
    predicted_noise = model(noisy, timesteps, proxy, experts)
    per_sample_mse = (predicted_noise - noise).square().mean(dim=(1, 2, 3))
    snr_weight = _min_snr_weight(schedule, timesteps, float(diffusion_cfg.get("min_snr_gamma", 5.0)))
    noise_loss = (per_sample_mse * snr_weight).mean()
    predicted_start = schedule.predict_start_from_noise(noisy, timesteps, predicted_noise).clamp(-4.5, 4.5)
    latent_loss = F.smooth_l1_loss(predicted_start, target_latent)
    image_loss = torch.zeros((), device=device)
    style_loss = torch.zeros((), device=device)
    dice = torch.zeros((), device=device)
    proxy_target_gap = torch.zeros((), device=device)
    prediction_target_l1 = torch.zeros((), device=device)
    prediction_proxy_l1 = torch.zeros((), device=device)
    style_direction = torch.zeros((), device=device)
    selected = timesteps <= recon_limit
    if bool(selected.any()):
        decoded = vq.decode(predicted_start[selected], snap_to_codebook=False)
        selected_target_aux = target_aux[selected]
        selected_proxy = proxy[selected]
        image_loss, _ = criterion(
            decoded,
            selected_target_aux[:, 0:1],
            content_proxy=selected_proxy,
            target_aux=selected_target_aux,
        )
        predicted_ink = torch.sigmoid(decoded[:, :1])
        dice = batch_binary_dice(predicted_ink, selected_target_aux[:, 0:1]).mean()
        style_size = int(diffusion_cfg.get("style_loss_size", 128))
        predicted_style_input = F.interpolate(predicted_ink, (style_size, style_size), mode="bilinear", align_corners=False)
        target_style_input = F.interpolate(selected_target_aux[:, 0:1], (style_size, style_size), mode="bilinear", align_corners=False)
        predicted_embedding = style_encoder.glyph_encoder(predicted_style_input)
        with torch.no_grad():
            target_embedding = style_encoder.glyph_encoder(target_style_input)
        style_loss = (1.0 - _normalized_cosine(predicted_embedding, target_embedding)).mean()
        proxy_ink = selected_proxy[:, 0:1].clamp(0.0, 1.0)
        target_ink = selected_target_aux[:, 0:1]
        proxy_gap_per_sample = (proxy_ink - target_ink).abs().mean(dim=(1, 2, 3))
        pred_target_per_sample = (predicted_ink - target_ink).abs().mean(dim=(1, 2, 3))
        pred_proxy_per_sample = (predicted_ink - proxy_ink).abs().mean(dim=(1, 2, 3))
        proxy_target_gap = proxy_gap_per_sample.mean()
        prediction_target_l1 = pred_target_per_sample.mean()
        prediction_proxy_l1 = pred_proxy_per_sample.mean()
        style_direction = ((pred_proxy_per_sample - pred_target_per_sample) / proxy_gap_per_sample.clamp_min(1e-5)).mean()
    total = (
        float(diffusion_cfg.get("noise_weight", 1.0)) * noise_loss
        + float(diffusion_cfg.get("latent_weight", 0.15)) * latent_loss
        + float(diffusion_cfg.get("image_weight", 0.42)) * image_loss
        + float(diffusion_cfg.get("style_weight", 0.20)) * style_loss
    )
    return total, {
        "noise": noise_loss.detach(),
        "latent": latent_loss.detach(),
        "image": image_loss.detach(),
        "style": style_loss.detach(),
        "dice": dice.detach(),
        "mean_timestep": timesteps.float().mean().detach(),
        "proxy_target_gap": proxy_target_gap.detach(),
        "prediction_target_l1": prediction_target_l1.detach(),
        "prediction_proxy_l1": prediction_proxy_l1.detach(),
        "style_direction": style_direction.detach(),
    }


@torch.no_grad()
def _evaluate_diffusion(
    *,
    model: LatentDiffusionUNet,
    vq: GlyphVQVAE,
    style_encoder: StyleReferenceEncoder,
    style_bank: dict[str, Any],
    schedule: DiffusionSchedule,
    loader: Iterable,
    device: torch.device,
    cfg: dict[str, Any],
    amp: bool,
    maximum_batches: int = 96,
) -> dict[str, float]:
    model.eval()
    criterion = _efficient_glyph_criterion(cfg.get("fusion", {}).get("diffusion", {}), cfg.get("loss", {}).get("weights", {})).to(device)
    totals: dict[str, float] = {}
    seen = 0
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(987654321)
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            with _autocast(device, amp):
                loss, pieces = _diffusion_training_loss(
                    model=model, vq=vq, style_encoder=style_encoder, style_bank=style_bank, schedule=schedule,
                    criterion=criterion, batch=batch, device=device, cfg=cfg, amp=amp,
                )
            count = int(batch["proxy"].shape[0])
            seen += count
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item()) * count
            for key, value in pieces.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
    finally:
        torch.random.set_rng_state(rng_state)
    return {key: value / max(1, seen) for key, value in totals.items()}


@torch.no_grad()
def _diffusion_preview(
    *,
    model: LatentDiffusionUNet,
    vq: GlyphVQVAE,
    style_encoder: StyleReferenceEncoder,
    style_bank: dict[str, Any],
    schedule: DiffusionSchedule,
    loader: Iterable,
    device: torch.device,
    cfg: dict[str, Any],
    output: Path,
) -> None:
    model.eval()
    batch = next(iter(loader))
    proxy = batch["proxy"].to(device)
    target_aux = batch["target_aux"].to(device)
    count = min(6, proxy.shape[0])
    proxy = proxy[:count]
    target_aux = target_aux[:count]
    preview_batch = dict(batch)
    if isinstance(preview_batch.get("style_index"), torch.Tensor):
        preview_batch["style_index"] = preview_batch["style_index"][:count]
    experts = _style_experts_from_bank(preview_batch, style_bank, device, count)
    latent_height = max(1, proxy.shape[-2] // 8)
    latent_width = max(1, proxy.shape[-1] // 8)
    generator = torch.Generator(device=device).manual_seed(20260719)
    latent = ddim_sample(
        model,
        schedule,
        (count, int(cfg["fusion"].get("latent_channels", 32)), latent_height, latent_width),
        content_proxy=proxy,
        style_experts=experts,
        steps=min(48, int(cfg["fusion"].get("inference", {}).get("ddim_steps", 80))),
        eta=0.0,
        generator=generator,
    )
    decoded = torch.sigmoid(vq.decode(latent, snap_to_codebook=True)[:, :1])
    _save_preview(
        output,
        [
            ("content proxy", proxy[:, 0].float().cpu().numpy()),
            ("diffusion", decoded[:, 0].float().cpu().numpy()),
            ("target truth", target_aux[:, 0].float().cpu().numpy()),
        ],
        [int(value) for value in batch["codepoint"][:count]],
        maximum=count,
    )


def _enforce_diffusion_style_guard(
    cfg: dict[str, Any],
    phase: dict[str, Any],
    epoch: int,
    validation: dict[str, float],
    phase_dir: Path,
) -> None:
    """Stop a months-long run if diffusion drifts toward the style-reduced proxy.

    Validation uses only real target.ttf self-reconstruction samples. A positive
    style_direction means the decoded prediction is closer to target truth than
    to the canonical proxy; a sustained negative value after warm-up indicates
    content-proxy/style collapse rather than useful target-style learning.
    """

    guard = cfg.get("style_guard", {})
    if not bool(guard.get("enabled", True)):
        return
    warmup = max(4, int(guard.get("diffusion_warmup_epochs", guard.get("warmup_epochs", 16))))
    gap = float(validation.get("proxy_target_gap", 0.0))
    direction = float(validation.get("style_direction", 0.0))
    if epoch < warmup or gap < float(guard.get("minimum_proxy_target_gap", 0.035)):
        return
    minimum = float(guard.get("minimum_diffusion_style_direction", guard.get("minimum_style_direction", -0.03)))
    if direction >= minimum:
        return
    report = {
        "status": "diffusion_style_collapse_detected",
        "phase": phase.get("name"),
        "epoch": int(epoch),
        "proxy_target_gap": gap,
        "prediction_target_l1": float(validation.get("prediction_target_l1", 0.0)),
        "prediction_proxy_l1": float(validation.get("prediction_proxy_l1", 0.0)),
        "style_direction": direction,
        "minimum_allowed_style_direction": minimum,
        "explanation": (
            "The latent diffusion decoder is becoming closer to the canonical content proxy than to the real target.ttf truth. "
            "The current phase was stopped before a long run could erase the target style."
        ),
    }
    save_json(phase_dir / "DIFFUSION_STYLE_COLLAPSE_DETECTED.json", report)
    if bool(guard.get("abort_on_collapse", True)):
        raise RuntimeError(
            "STYLE_COLLAPSE_FATAL: latent-diffusion style collapse detected: the prediction is moving toward the style-stripped target structural proxy instead of the target.ttf ground truth. "
            f" phase={phase.get('name')} epoch={epoch} style_direction={direction:.4f}."
            "DIFFUSION_STYLE_COLLAPSE_DETECTED.json was saved and the stage was stopped."
        )


def _save_diffusion_checkpoint(
    path: Path,
    *,
    fingerprint: dict[str, Any],
    model: LatentDiffusionUNet,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best: float,
    no_improvement: int,
    model_spec: dict[str, Any],
    validation: dict[str, float] | None = None,
) -> None:
    _atomic_torch_save({
        "fingerprint": fingerprint,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best": float(best),
        "no_improvement": int(no_improvement),
        "model_spec": model_spec,
        "validation": validation or {},
    }, path)


def _train_diffusion_phase(
    *,
    cfg: dict[str, Any],
    phase: dict[str, Any],
    phase_dir: Path,
    model: LatentDiffusionUNet,
    vq: GlyphVQVAE,
    style_encoder: StyleReferenceEncoder,
    style_bank: dict[str, Any],
    schedule: DiffusionSchedule,
    init_checkpoint: Path | None,
    hard_codepoints: set[int] | None = None,
) -> Path:
    work = Path(cfg["paths"]["work_dir"])
    index_path = work / "dataset" / "index.csv"
    device = _device(cfg)
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    workers = int(cfg["training"].get("workers", 0))
    diffusion_cfg = cfg.get("fusion", {}).get("diffusion", {})
    model_spec = _diffusion_spec(cfg)
    phase_fingerprint = dict(phase)
    if hard_codepoints:
        phase_fingerprint["hard_codepoints_sha256"] = hashlib_codepoints(hard_codepoints)
    fingerprint = _checkpoint_fingerprint(index_path, "latent_diffusion", model_spec, phase_fingerprint)
    ensure_dir(phase_dir)
    completed_path = phase_dir / "completed.json"
    if completed_path.is_file() and (phase_dir / "best.pt").is_file():
        payload = torch.load(phase_dir / "best.pt", map_location=device, weights_only=False)
        if _compatible(payload, fingerprint):
            if payload.get("ema"):
                ema_temp = ExponentialMovingAverage(model, decay=float(diffusion_cfg.get("ema_decay", 0.9999)))
                ema_temp.load_state_dict(payload["ema"])
                ema_temp.copy_to(model)
            else:
                model.load_state_dict(payload["model"], strict=True)
            return phase_dir / "best.pt"
    resume_path, resume = _load_resume(phase_dir, fingerprint)
    if resume is None and init_checkpoint is not None and init_checkpoint.is_file():
        previous = torch.load(init_checkpoint, map_location=device, weights_only=False)
        if previous.get("ema"):
            previous_ema = ExponentialMovingAverage(model)
            previous_ema.load_state_dict(previous["ema"])
            previous_ema.copy_to(model)
        else:
            model.load_state_dict(previous["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(phase["learning_rate"]),
        betas=(0.9, 0.99),
        weight_decay=float(diffusion_cfg.get("weight_decay", 1e-4)),
    )
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.65,
        patience=max(3, int(phase.get("lr_patience", 8))),
        min_lr=float(phase.get("minimum_learning_rate", 5e-7)),
    )
    scaler = _scaler(device, amp)
    ema = ExponentialMovingAverage(model, decay=float(diffusion_cfg.get("ema_decay", 0.9999)))
    start_epoch = 1
    global_step = 0
    best = math.inf
    no_improvement = 0
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
        if resume.get("ema"):
            ema.load_state_dict(resume["ema"])
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scheduler"):
            scheduler_lr.load_state_dict(resume["scheduler"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        global_step = int(resume.get("global_step", 0))
        best = float(resume.get("best", math.inf))
        no_improvement = int(resume.get("no_improvement", 0))
    size = int(phase["size"])
    style_size = int(cfg["fusion"].get("style_encoder", {}).get("size", 128))
    style_refs = int(cfg["fusion"].get("style_encoder", {}).get("inference_references", 12))
    train_dataset = FusionDiffusionDataset(
        index_path, split="train", size=size, style_size=style_size, style_references=style_refs,
        style_groups=int(style_bank.get("group_count", 12)), augment=True, hard_codepoints=hard_codepoints,
        hard_repeat=int(phase.get("hard_repeat", 5)),
        seed=int(cfg["training"].get("seed", 20260719)) + int(phase.get("seed_offset", 0)),
    )
    val_dataset = FusionDiffusionDataset(
        index_path, split="val", size=size, style_size=style_size, style_references=style_refs,
        style_groups=int(style_bank.get("group_count", 12)), augment=False, seed=int(cfg["training"].get("seed", 20260719)) + 8081,
    )
    train_loader = _loader(train_dataset, int(phase["batch_size"]), workers, shuffle=True)
    val_loader = _loader(val_dataset, max(1, int(phase["batch_size"])), workers, shuffle=False)
    criterion = _efficient_glyph_criterion(diffusion_cfg, cfg.get("loss", {}).get("weights", {})).to(device)
    accumulation = max(1, int(phase.get("gradient_accumulation", 1)))
    history = _read_history(phase_dir / "history.csv")
    checkpoint_every = max(20, int(diffusion_cfg.get("checkpoint_every_steps", 120)))
    guard = LongRunGuard(cfg)
    started = time.time()
    for epoch in range(start_epoch, int(phase["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        seen = 0
        progress = tqdm(train_loader, desc=f"{phase['name']} {epoch:03d}/{int(phase['epochs']):03d}", unit="batch")
        for step, batch in enumerate(progress, start=1):
            with _autocast(device, amp):
                loss, pieces = _diffusion_training_loss(
                    model=model, vq=vq, style_encoder=style_encoder, style_bank=style_bank, schedule=schedule,
                    criterion=criterion, batch=batch, device=device, cfg=cfg, amp=amp,
                )
                scaled = loss / accumulation
            scaler.scale(scaled).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(diffusion_cfg.get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1
                if global_step % checkpoint_every == 0:
                    _save_diffusion_checkpoint(
                        phase_dir / "in_epoch.pt", fingerprint=fingerprint, model=model, ema=ema,
                        optimizer=optimizer, scheduler=scheduler_lr, scaler=scaler,
                        epoch=epoch - 1, global_step=global_step, best=best,
                        no_improvement=no_improvement, model_spec=model_spec,
                    )
                    guard.checkpoint_boundary()
            count = int(batch["proxy"].shape[0])
            seen += count
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().item()) * count
            for key, value in pieces.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
            progress.set_postfix(loss=f"{totals['loss']/max(1,seen):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        with ema.average_parameters(model):
            validation = _evaluate_diffusion(
                model=model, vq=vq, style_encoder=style_encoder, style_bank=style_bank, schedule=schedule,
                loader=val_loader, device=device, cfg=cfg, amp=amp,
                maximum_batches=int(diffusion_cfg.get("validation_batches", 96)),
            )
        _enforce_diffusion_style_guard(cfg, phase, epoch, validation, phase_dir)
        scheduler_lr.step(validation["loss"])
        improved = validation["loss"] < best - float(phase.get("minimum_improvement", 1e-5))
        if improved:
            best = validation["loss"]
            no_improvement = 0
        else:
            no_improvement += 1
        _save_diffusion_checkpoint(
            phase_dir / "last.pt", fingerprint=fingerprint, model=model, ema=ema,
            optimizer=optimizer, scheduler=scheduler_lr, scaler=scaler,
            epoch=epoch, global_step=global_step, best=best, no_improvement=no_improvement,
            model_spec=model_spec, validation=validation,
        )
        if improved:
            _save_diffusion_checkpoint(
                phase_dir / "best.pt", fingerprint=fingerprint, model=model, ema=ema,
                optimizer=optimizer, scheduler=scheduler_lr, scaler=scaler,
                epoch=epoch, global_step=global_step, best=best, no_improvement=no_improvement,
                model_spec=model_spec, validation=validation,
            )
        (phase_dir / "in_epoch.pt").unlink(missing_ok=True)
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best": int(improved),
            "elapsed_seconds": int(time.time() - started),
        })
        _write_history(phase_dir / "history.csv", history)
        if epoch == 1 or epoch % int(diffusion_cfg.get("preview_every", 4)) == 0 or improved:
            with ema.average_parameters(model):
                _diffusion_preview(
                    model=model, vq=vq, style_encoder=style_encoder, style_bank=style_bank, schedule=schedule,
                    loader=val_loader, device=device, cfg=cfg,
                    output=phase_dir / "previews" / f"epoch_{epoch:03d}.png",
                )
        guard.checkpoint_boundary()
        if no_improvement >= int(phase.get("patience", 48)):
            break
    save_json(completed_path, {"fingerprint": fingerprint, "best": best})
    payload = torch.load(phase_dir / "best.pt", map_location=device, weights_only=False)
    if payload.get("ema"):
        final_ema = ExponentialMovingAverage(model)
        final_ema.load_state_dict(payload["ema"])
        final_ema.copy_to(model)
    else:
        model.load_state_dict(payload["model"], strict=True)
    return phase_dir / "best.pt"


def hashlib_codepoints(values: set[int]) -> str:
    import hashlib
    payload = ",".join(str(value) for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def mine_hard_codepoints(
    cfg: dict[str, Any],
    model: LatentDiffusionUNet,
    vq: GlyphVQVAE,
    style_encoder: StyleReferenceEncoder,
    style_bank: dict[str, Any],
    schedule: DiffusionSchedule,
    *,
    maximum: int,
    evaluation_size: int,
) -> list[tuple[int, float]]:
    work = Path(cfg["paths"]["work_dir"])
    index_path = work / "dataset" / "index.csv"
    device = _device(cfg)
    style_size = int(cfg["fusion"].get("style_encoder", {}).get("size", 128))
    style_refs = int(cfg["fusion"].get("style_encoder", {}).get("inference_references", 12))
    dataset = FusionDiffusionDataset(
        index_path, split="train", size=evaluation_size, style_size=style_size,
        style_references=style_refs, style_groups=int(style_bank.get("group_count", 12)), augment=False,
        seed=int(cfg["training"].get("seed", 20260719)) + 4567,
    )
    loader = _loader(dataset, 1, int(cfg["training"].get("workers", 0)), shuffle=False)
    model.eval()
    scores: list[tuple[int, float]] = []
    fixed_t = min(schedule.timesteps - 1, int(cfg["fusion"].get("purification", {}).get("evaluation_timestep", 120)))
    for batch in tqdm(loader, desc="Mining hard diffusion glyphs", unit="glyph"):
        proxy = batch["proxy"].to(device)
        target_aux = batch["target_aux"].to(device)
        target_latent = vq.encode(target_aux, quantize=True, update_codebook=False)["quantized"]
        generator = torch.Generator(device=device).manual_seed(int(batch["codepoint"][0]))
        noise = torch.randn(target_latent.shape, device=device, generator=generator)
        timestep = torch.full((1,), fixed_t, device=device, dtype=torch.long)
        noisy = schedule.q_sample(target_latent, timestep, noise)
        experts = _style_experts_from_bank(batch, style_bank, device, proxy.shape[0])
        predicted_noise = model(noisy, timestep, proxy, experts)
        predicted_start = schedule.predict_start_from_noise(noisy, timestep, predicted_noise).clamp(-4.5, 4.5)
        ink = torch.sigmoid(vq.decode(predicted_start, snap_to_codebook=True)[:, :1])
        dice = float(batch_binary_dice(ink, target_aux[:, 0:1]).item())
        scores.append((int(batch["codepoint"][0]), 1.0 - dice))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[: max(1, int(maximum))]


def train_diffusion(cfg: dict[str, Any]) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    fusion = cfg.get("fusion", {})
    diffusion_cfg = fusion.get("diffusion", {})
    root = ensure_dir(work / "fusion" / "diffusion")
    summary_path = root / "summary.json"
    device = _device(cfg)
    style_encoder, _ = load_style_encoder(cfg, device)
    style_bank = load_style_bank(cfg, device)
    vq = load_vqvae(cfg, device)
    for parameter in style_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in vq.parameters():
        parameter.requires_grad_(False)
    model = _diffusion_model(cfg).to(device)
    schedule = DiffusionSchedule.create(
        int(fusion.get("diffusion_steps", 1000)),
        schedule=str(diffusion_cfg.get("schedule", "cosine")),
    ).to(device)
    phases = list(diffusion_cfg.get("phases", [])) or [{
        "name": "latent256", "size": 256, "epochs": 220, "batch_size": 3,
        "gradient_accumulation": 2, "learning_rate": 1.5e-4,
        "minimum_learning_rate": 5e-7, "patience": 52,
    }]
    previous: Path | None = None
    results: list[dict[str, Any]] = []
    for phase in phases:
        phase_dir = root / str(phase["name"])
        previous = _train_diffusion_phase(
            cfg=cfg, phase=phase, phase_dir=phase_dir, model=model, vq=vq,
            style_encoder=style_encoder, style_bank=style_bank, schedule=schedule, init_checkpoint=previous,
        )
        results.append({"name": phase["name"], "checkpoint": str(previous.resolve())})

    purification_cfg = fusion.get("purification", {})
    purification_results: list[dict[str, Any]] = []
    if bool(purification_cfg.get("enabled", True)):
        cycles = max(0, int(purification_cfg.get("cycles", 24)))
        no_improvement = 0
        active_checkpoint = previous
        active_payload = torch.load(active_checkpoint, map_location=device, weights_only=False)
        active_ema = ExponentialMovingAverage(model)
        active_ema.load_state_dict(active_payload["ema"])
        active_ema.copy_to(model)
        best_validation = float(active_payload.get("validation", {}).get("loss", active_payload.get("best", math.inf)))
        for cycle in range(1, cycles + 1):
            cycle_dir = root / "purification" / f"cycle_{cycle:03d}"
            if (cycle_dir / "completed.json").is_file() and (cycle_dir / "best.pt").is_file():
                payload = torch.load(cycle_dir / "best.pt", map_location=device, weights_only=False)
                if payload.get("ema"):
                    ema = ExponentialMovingAverage(model)
                    ema.load_state_dict(payload["ema"])
                    ema.copy_to(model)
                active_checkpoint = cycle_dir / "best.pt"
                best_validation = min(best_validation, float(payload.get("validation", {}).get("loss", math.inf)))
                purification_results.append({"cycle": cycle, "checkpoint": str(active_checkpoint.resolve()), "reused": True})
                continue
            hard = mine_hard_codepoints(
                cfg, model, vq, style_encoder, style_bank, schedule,
                maximum=int(purification_cfg.get("hard_samples", 8000)),
                evaluation_size=int(purification_cfg.get("evaluation_size", 256)),
            )
            hard_set = {cp for cp, _ in hard}
            ensure_dir(cycle_dir)
            save_json(cycle_dir / "hard_samples.json", {
                "count": len(hard),
                "samples": [{"codepoint": cp, "error": error} for cp, error in hard],
            })
            cycle_phase = {
                "name": f"purify_{cycle:03d}",
                "size": int(purification_cfg.get("size", phases[-1]["size"])),
                "epochs": int(purification_cfg.get("epochs_per_cycle", 12)),
                "batch_size": int(purification_cfg.get("batch_size", 1)),
                "gradient_accumulation": int(purification_cfg.get("gradient_accumulation", 8)),
                "learning_rate": max(
                    float(purification_cfg.get("minimum_learning_rate", 3e-7)),
                    float(purification_cfg.get("initial_learning_rate", 1.2e-5))
                    * float(purification_cfg.get("learning_rate_decay", 0.95)) ** (cycle - 1),
                ),
                "minimum_learning_rate": float(purification_cfg.get("minimum_learning_rate", 3e-7)),
                "patience": int(purification_cfg.get("epoch_patience", 8)),
                "hard_repeat": int(purification_cfg.get("hard_repeat", 6)),
                "seed_offset": cycle * 7919,
                "minimum_improvement": float(purification_cfg.get("minimum_improvement", 2e-5)),
            }
            candidate_checkpoint = _train_diffusion_phase(
                cfg=cfg, phase=cycle_phase, phase_dir=cycle_dir, model=model, vq=vq,
                style_encoder=style_encoder, style_bank=style_bank, schedule=schedule, init_checkpoint=active_checkpoint,
                hard_codepoints=hard_set,
            )
            candidate_payload = torch.load(candidate_checkpoint, map_location=device, weights_only=False)
            candidate_validation = float(candidate_payload.get("validation", {}).get("loss", math.inf))
            accepted = candidate_validation < best_validation - float(purification_cfg.get("minimum_cycle_improvement", 1e-5))
            if accepted:
                best_validation = candidate_validation
                active_checkpoint = candidate_checkpoint
                no_improvement = 0
                ema = ExponentialMovingAverage(model)
                ema.load_state_dict(candidate_payload["ema"])
                ema.copy_to(model)
            else:
                no_improvement += 1
                active_payload = torch.load(active_checkpoint, map_location=device, weights_only=False)
                ema = ExponentialMovingAverage(model)
                ema.load_state_dict(active_payload["ema"])
                ema.copy_to(model)
            save_json(cycle_dir / "decision.json", {
                "accepted": accepted,
                "candidate_validation": candidate_validation,
                "active_validation": best_validation,
                "active_checkpoint": str(active_checkpoint.resolve()),
            })
            purification_results.append({
                "cycle": cycle, "accepted": accepted,
                "candidate": str(candidate_checkpoint.resolve()),
                "active": str(active_checkpoint.resolve()),
            })
            if no_improvement >= int(purification_cfg.get("early_stop_cycles", 8)):
                break
        previous = active_checkpoint

    assert previous is not None
    final_path = root / "diffusion_best.pt"
    shutil.copy2(previous, final_path)
    summary = {
        "enabled": True,
        "checkpoint": str(final_path.resolve()),
        "model_spec": _diffusion_spec(cfg),
        "phases": results,
        "purification": purification_results,
        "method": "target-style conditioned VQ latent diffusion with real-target hard-sample purification",
    }
    save_json(summary_path, summary)
    return summary


def load_diffusion(cfg: dict[str, Any], device: torch.device | str) -> LatentDiffusionUNet:
    path = Path(cfg["paths"]["work_dir"]) / "fusion" / "diffusion" / "diffusion_best.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing trained diffusion checkpoint: {path}. Run fusion training before generation."
        )
    payload = torch.load(path, map_location=device, weights_only=False)
    model = _diffusion_model(cfg)
    if payload.get("ema"):
        ema = ExponentialMovingAverage(model)
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model)
    else:
        model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def train_direct_baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    fusion = cfg.get("fusion", {})
    direct_cfg = fusion.get("direct_baseline", {})
    if not bool(direct_cfg.get("enabled", True)):
        return {"enabled": False}
    work = Path(cfg["paths"]["work_dir"])
    model_root = ensure_dir(work / "fusion" / "direct")
    phases = list(direct_cfg.get("phases", [])) or [{
        "name": "direct256", "size": 256, "epochs": 120, "batch_size": 6,
        "gradient_accumulation": 1, "learning_rate": 1.5e-4,
        "minimum_learning_rate": 1e-6, "early_stopping_patience": 36,
        "adversarial": False,
    }]
    direct = copy.deepcopy(cfg)
    direct["training"]["base_channels"] = int(direct_cfg.get("base_channels", 28))
    direct["training"]["ema_decay"] = float(direct_cfg.get("ema_decay", 0.9995))
    direct["adversarial"]["enabled"] = bool(direct_cfg.get("adversarial", False))
    result = train_generator(
        direct,
        index_csv=work / "dataset" / "index.csv",
        model_root=model_root,
        phases_override=phases,
        resume=True,
    )
    result["method"] = "deterministic topology-aware safety baseline"
    save_json(model_root / "fusion_summary.json", result)
    return result


def load_direct_baseline(cfg: dict[str, Any], device: torch.device | str) -> FontStyleNetFinal | None:
    path = Path(cfg["paths"]["work_dir"]) / "fusion" / "direct" / "generator_best.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = payload.get("model_config", {})
    model = FontStyleNetFinal(
        base=int(model_config.get("base_channels", cfg.get("fusion", {}).get("direct_baseline", {}).get("base_channels", 28))),
        in_channels=int(model_config.get("input_channels", 10)),
        out_channels=int(model_config.get("output_channels", 4)),
    )
    if payload.get("ema_state"):
        state = payload["ema_state"].get("shadow", payload["ema_state"])
        model.load_state_dict(state, strict=True)
    elif payload.get("model_state"):
        model.load_state_dict(payload["model_state"], strict=True)
    elif payload.get("model"):
        model.load_state_dict(payload["model"], strict=True)
    else:
        raise RuntimeError(f"unrecognized direct baseline checkpoint: {path}")
    return model.to(device).eval()


def _refiner_model(cfg: dict[str, Any]) -> StyleAwareGlyphRefiner:
    fusion = cfg.get("fusion", {})
    refiner_cfg = fusion.get("refiner", {})
    return StyleAwareGlyphRefiner(
        in_channels=11,
        base=int(refiner_cfg.get("base_channels", 32)),
        style_dim=int(fusion.get("style_dim", 256)),
        expert_count=int(fusion.get("expert_count", 8)),
    )


def train_fusion_refiner(cfg: dict[str, Any]) -> dict[str, Any]:
    fusion = cfg.get("fusion", {})
    ref_cfg = fusion.get("refiner", {})
    if not bool(ref_cfg.get("enabled", True)):
        return {"enabled": False}
    work = Path(cfg["paths"]["work_dir"])
    root = ensure_dir(work / "fusion" / "refiner")
    index_path = work / "dataset" / "index.csv"
    device = _device(cfg)
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    workers = int(cfg["training"].get("workers", 0))
    style_encoder, style_spec = load_style_encoder(cfg, device)
    for parameter in style_encoder.parameters():
        parameter.requires_grad_(False)
    model = _refiner_model(cfg).to(device)
    model_spec = {
        "base_channels": int(ref_cfg.get("base_channels", 32)),
        "style_dim": int(fusion.get("style_dim", 256)),
        "expert_count": int(fusion.get("expert_count", 8)),
    }
    phase = {
        "size": int(ref_cfg.get("size", 512)),
        "style_size": int(fusion.get("style_encoder", {}).get("size", 128)),
        "style_references": int(fusion.get("style_encoder", {}).get("inference_references", 12)),
        "epochs": int(ref_cfg.get("epochs", 160)),
        "batch_size": int(ref_cfg.get("batch_size", 1)),
        "gradient_accumulation": int(ref_cfg.get("gradient_accumulation", 8)),
        "learning_rate": float(ref_cfg.get("learning_rate", 2e-5)),
    }
    fingerprint = _checkpoint_fingerprint(index_path, "fusion_refiner", model_spec, phase)
    summary_path = root / "summary.json"
    if summary_path.is_file() and (root / "best.pt").is_file():
        payload = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
        if _compatible(payload, fingerprint):
            return load_json(summary_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=phase["learning_rate"], betas=(0.9, 0.99), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.65, patience=max(3, int(ref_cfg.get("lr_patience", 8))), min_lr=5e-7
    )
    scaler = _scaler(device, amp)
    ema = ExponentialMovingAverage(model, decay=float(ref_cfg.get("ema_decay", 0.9995)))
    resume_path, resume = _load_resume(root, fingerprint)
    start_epoch = 1
    global_step = 0
    best = math.inf
    no_improvement = 0
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
        if resume.get("ema"):
            ema.load_state_dict(resume["ema"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        global_step = int(resume.get("global_step", 0))
        best = float(resume.get("best", math.inf))
        no_improvement = int(resume.get("no_improvement", 0))
    train_loader = _loader(
        FusionRefinerDataset(
            index_path, split="train", size=phase["size"], style_size=phase["style_size"],
            style_references=phase["style_references"], style_groups=int(style_bank.get("group_count", 12)), augment=True,
            seed=int(cfg["training"].get("seed", 20260719)) + 71,
        ),
        phase["batch_size"], workers, shuffle=True,
    )
    val_loader = _loader(
        FusionRefinerDataset(
            index_path, split="val", size=phase["size"], style_size=phase["style_size"],
            style_references=phase["style_references"], style_groups=int(style_bank.get("group_count", 12)), augment=False,
            seed=int(cfg["training"].get("seed", 20260719)) + 72,
        ),
        phase["batch_size"], workers, shuffle=False,
    )
    criterion = _efficient_glyph_criterion(ref_cfg, cfg.get("loss", {}).get("weights", {})).to(device)
    accumulation = max(1, phase["gradient_accumulation"])
    history = _read_history(root / "history.csv")
    guard = LongRunGuard(cfg)
    checkpoint_every = max(20, int(ref_cfg.get("checkpoint_every_steps", 100)))

    @torch.no_grad()
    def evaluate(evaluation_model: StyleAwareGlyphRefiner) -> dict[str, float]:
        evaluation_model.eval()
        total_loss = total_dice = total_style = 0.0
        seen = 0
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= int(ref_cfg.get("validation_batches", 96)):
                break
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            target_aux = batch["target_aux"].to(device)
            experts = _style_experts_from_bank(batch, style_bank, device, inputs.shape[0])
            with _autocast(device, amp):
                prediction = evaluation_model(inputs, experts)
                glyph_loss, _ = criterion(prediction, target, content_proxy=inputs[:, 1:], target_aux=target_aux)
                probability = torch.sigmoid(prediction)
                style_size = phase["style_size"]
                pred_embedding = style_encoder.glyph_encoder(F.interpolate(probability, (style_size, style_size), mode="bilinear", align_corners=False))
                target_embedding = style_encoder.glyph_encoder(F.interpolate(target, (style_size, style_size), mode="bilinear", align_corners=False))
                style_loss = (1.0 - _normalized_cosine(pred_embedding, target_embedding)).mean()
                loss = glyph_loss + float(ref_cfg.get("style_weight", 0.18)) * style_loss
            count = inputs.shape[0]
            seen += count
            total_loss += float(loss.item()) * count
            total_style += float(style_loss.item()) * count
            total_dice += float(batch_binary_dice(probability, target).mean().item()) * count
        return {"loss": total_loss / max(1, seen), "dice": total_dice / max(1, seen), "style": total_style / max(1, seen)}

    for epoch in range(start_epoch, phase["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"style-refiner {epoch:03d}/{phase['epochs']:03d}", unit="batch")
        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            target_aux = batch["target_aux"].to(device)
            with torch.no_grad():
                experts = _style_experts_from_bank(batch, style_bank, device, inputs.shape[0])
            with _autocast(device, amp):
                prediction = model(inputs, experts)
                glyph_loss, _ = criterion(prediction, target, content_proxy=inputs[:, 1:], target_aux=target_aux)
                probability = torch.sigmoid(prediction)
                pred_embedding = style_encoder.glyph_encoder(F.interpolate(probability, (phase["style_size"], phase["style_size"]), mode="bilinear", align_corners=False))
                with torch.no_grad():
                    target_embedding = style_encoder.glyph_encoder(F.interpolate(target, (phase["style_size"], phase["style_size"]), mode="bilinear", align_corners=False))
                style_loss = (1.0 - _normalized_cosine(pred_embedding, target_embedding)).mean()
                loss = glyph_loss + float(ref_cfg.get("style_weight", 0.18)) * style_loss
                scaled = loss / accumulation
            scaler.scale(scaled).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1
                if global_step % checkpoint_every == 0:
                    _atomic_torch_save({
                        "fingerprint": fingerprint, "model": model.state_dict(), "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch - 1, "global_step": global_step,
                        "best": best, "no_improvement": no_improvement, "model_spec": model_spec,
                    }, root / "in_epoch.pt")
                    guard.checkpoint_boundary()
            count = inputs.shape[0]
            total += float(loss.detach().item()) * count
            seen += count
            progress.set_postfix(loss=f"{total/max(1,seen):.4f}")
        with ema.average_parameters(model):
            validation = evaluate(model)
        scheduler.step(validation["loss"])
        improved = validation["loss"] < best
        if improved:
            best = validation["loss"]
            no_improvement = 0
        else:
            no_improvement += 1
        payload = {
            "fingerprint": fingerprint, "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
            "best": best, "no_improvement": no_improvement, "model_spec": model_spec,
            "validation": validation,
        }
        _atomic_torch_save(payload, root / "last.pt")
        if improved:
            _atomic_torch_save(payload, root / "best.pt")
        (root / "in_epoch.pt").unlink(missing_ok=True)
        history.append({
            "epoch": epoch, "train_loss": total / max(1, seen),
            "val_loss": validation["loss"], "val_dice": validation["dice"],
            "val_style": validation["style"], "learning_rate": optimizer.param_groups[0]["lr"],
            "best": int(improved),
        })
        _write_history(root / "history.csv", history)
        if epoch == 1 or epoch % int(ref_cfg.get("preview_every", 4)) == 0 or improved:
            batch = next(iter(val_loader))
            inputs = batch["input"].to(device)
            with torch.no_grad():
                experts = _style_experts_from_bank(batch, style_bank, device, inputs.shape[0])
                with ema.average_parameters(model):
                    prediction = torch.sigmoid(model(inputs, experts))
            _save_preview(
                root / "previews" / f"epoch_{epoch:03d}.png",
                [("corrupted candidate", inputs[:, 0].cpu().numpy()),
                 ("refined", prediction[:, 0].cpu().numpy()),
                 ("target truth", batch["target"][:, 0].numpy())],
                [int(value) for value in batch["codepoint"]],
            )
        guard.checkpoint_boundary()
        if no_improvement >= int(ref_cfg.get("patience", 44)):
            break
    summary = {
        "enabled": True,
        "checkpoint": str((root / "best.pt").resolve()),
        "best_validation_loss": best,
        "model_spec": model_spec,
        "method": "target-style conditioned high-resolution artifact refiner",
    }
    save_json(summary_path, summary)
    return summary


def load_fusion_refiner(cfg: dict[str, Any], device: torch.device | str) -> StyleAwareGlyphRefiner | None:
    path = Path(cfg["paths"]["work_dir"]) / "fusion" / "refiner" / "best.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    model = _refiner_model(cfg)
    if payload.get("ema"):
        ema = ExponentialMovingAverage(model)
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model)
    else:
        model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def train_contour_polisher(cfg: dict[str, Any]) -> dict[str, Any]:
    fusion = cfg.get("fusion", {})
    contour_cfg = fusion.get("contour_polisher", {})
    if not bool(contour_cfg.get("enabled", True)):
        return {"enabled": False}
    work = Path(cfg["paths"]["work_dir"])
    cache_summary = build_contour_cache(cfg, force=False)
    if not cache_summary.get("enabled"):
        return cache_summary
    root = ensure_dir(work / "fusion" / "contour")
    cache_path = Path(cache_summary["cache_path"])
    device = _device(cfg)
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    spec = {
        "points": int(contour_cfg.get("points", 128)),
        "hidden": int(contour_cfg.get("hidden", 192)),
        "layers": int(contour_cfg.get("layers", 6)),
        "heads": int(contour_cfg.get("heads", 8)),
    }
    phase = {
        "epochs": int(contour_cfg.get("epochs", 180)),
        "batch_size": int(contour_cfg.get("batch_size", 64)),
        "learning_rate": float(contour_cfg.get("learning_rate", 1.5e-4)),
        "virtual_length": int(contour_cfg.get("virtual_length", 160000)),
    }
    fingerprint = {
        "version": FUSION_CHECKPOINT_VERSION,
        "stage": "contour_polisher",
        "cache_sha256": sha256_file(cache_path),
        "spec": spec,
        "phase": phase,
    }
    model = ContourSequenceTransformer(**spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=phase["learning_rate"], betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.65, patience=8, min_lr=5e-7)
    scaler = _scaler(device, amp)
    dataset = ContourDenoiseDataset(cache_path, virtual_length=phase["virtual_length"], seed=int(cfg["training"].get("seed", 0)))
    val_dataset = ContourDenoiseDataset(cache_path, virtual_length=max(512, phase["batch_size"] * 16), seed=int(cfg["training"].get("seed", 0)) + 111)
    loader = _loader(dataset, phase["batch_size"], int(cfg["training"].get("workers", 0)), shuffle=True)
    val_loader = _loader(val_dataset, phase["batch_size"], int(cfg["training"].get("workers", 0)), shuffle=False)
    resume_path, resume = _load_resume(root, fingerprint)
    start_epoch = 1
    best = math.inf
    no_improvement = 0
    global_step = 0
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        best = float(resume.get("best", math.inf))
        no_improvement = int(resume.get("no_improvement", 0))
        global_step = int(resume.get("global_step", 0))
    history = _read_history(root / "history.csv")
    guard = LongRunGuard(cfg)

    def contour_loss(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = batch["features"].to(device)
        offsets = batch["offsets"].to(device)
        corners = batch["corners"].to(device)
        output = model(features)
        offset_loss = F.smooth_l1_loss(output["offset"], offsets)
        predicted_points = features[..., :2] + output["offset"]
        target_points = features[..., :2] + offsets
        pred_curvature = torch.roll(predicted_points, -1, 1) - 2 * predicted_points + torch.roll(predicted_points, 1, 1)
        true_curvature = torch.roll(target_points, -1, 1) - 2 * target_points + torch.roll(target_points, 1, 1)
        curvature_loss = F.smooth_l1_loss(pred_curvature, true_curvature)
        corner_loss = F.binary_cross_entropy_with_logits(output["corner_logits"], corners)
        closure_loss = (predicted_points[:, 0] - predicted_points[:, -1]).norm(dim=1).mean() * 0.01
        total = offset_loss + 0.35 * curvature_loss + 0.12 * corner_loss + closure_loss
        return total, {"offset": offset_loss, "curvature": curvature_loss, "corner": corner_loss}

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = {}
        seen = 0
        for index, batch in enumerate(val_loader):
            if index >= 24:
                break
            with _autocast(device, amp):
                loss, pieces = contour_loss(batch)
            count = batch["features"].shape[0]
            seen += count
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item()) * count
            for key, value in pieces.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
        return {key: value / max(1, seen) for key, value in totals.items()}

    for epoch in range(start_epoch, phase["epochs"] + 1):
        model.train()
        total = 0.0
        seen = 0
        progress = tqdm(loader, desc=f"contour {epoch:03d}/{phase['epochs']:03d}", unit="batch")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, _ = contour_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = batch["features"].shape[0]
            total += float(loss.detach().item()) * count
            seen += count
            global_step += 1
            progress.set_postfix(loss=f"{total/max(1,seen):.5f}")
        validation = evaluate()
        scheduler.step(validation["loss"])
        improved = validation["loss"] < best
        if improved:
            best = validation["loss"]
            no_improvement = 0
        else:
            no_improvement += 1
        payload = {
            "fingerprint": fingerprint, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
            "best": best, "no_improvement": no_improvement, "spec": spec,
            "validation": validation,
        }
        _atomic_torch_save(payload, root / "last.pt")
        if improved:
            _atomic_torch_save(payload, root / "best.pt")
        history.append({
            "epoch": epoch, "train_loss": total / max(1, seen),
            **{f"val_{key}": value for key, value in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"], "best": int(improved),
        })
        _write_history(root / "history.csv", history)
        guard.checkpoint_boundary()
        if no_improvement >= int(contour_cfg.get("patience", 28)):
            break
    summary = {
        "enabled": True,
        "checkpoint": str((root / "best.pt").resolve()),
        "best_validation_loss": best,
        "spec": spec,
        "method": "target-contour Transformer denoiser",
    }
    save_json(root / "training_summary.json", summary)
    return summary


def train_fusion_all(cfg: dict[str, Any]) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    root = ensure_dir(work / "fusion")
    results: dict[str, Any] = {}
    results["component_atlas"] = build_component_atlas(cfg, force=False)
    results["style_encoder"] = train_style_encoder(cfg)
    results["vq"] = train_vqvae(cfg)
    results["direct_baseline"] = train_direct_baseline(cfg)
    results["diffusion"] = train_diffusion(cfg)
    results["refiner"] = train_fusion_refiner(cfg)
    results["contour_polisher"] = train_contour_polisher(cfg)
    summary = {
        "version": FUSION_CHECKPOINT_VERSION,
        "method": (
            "target-only localized style encoder + target VQ stroke prior + multi-scale latent diffusion + "
            "deterministic baseline + semantic component residuals + style-aware refiner + contour Transformer"
        ),
        "stages": results,
    }
    save_json(root / "summary.json", summary)
    return summary
