from __future__ import annotations

import contextlib
import csv
import math
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import make_refiner_loaders, make_style_loaders
from .config import CHECKPOINT_FORMAT_VERSION
from .contract import validate_data_flow_contract
from .longrun import LongRunGuard
from .features import ink_probability
from .losses import FontLossFinal, batch_binary_dice
from .model import FontStyleNetFinal, GlyphRefinerFinal, PatchDiscriminatorFinal, count_parameters
from .report import make_training_preview
from .util import durable_replace, ensure_dir, load_json, save_json, set_seed, sha256_file, write_csv


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.decay = float(decay)
        self.shadow = deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.shadow.state_dict()
        for key, value in target.items():
            current = source[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                value.copy_(current)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow.load_state_dict(state)


def _resolve_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["training"].get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "The configuration requires CUDA, but the current PyTorch installation cannot use it. Run install_cuda130.bat "
            "and confirm that torch.cuda.is_available() is True."
        )
    if not requested.startswith("cuda"):
        torch.set_num_threads(max(1, min(8, int(cfg["training"].get("cpu_threads", 4)))))
    return torch.device(requested if requested.startswith("cuda") else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
    )


def _grad_scaler(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=active)


def _extract_content_proxy(inputs: torch.Tensor) -> torch.Tensor:
    # Generator input: 10 proxy channels. Refiner: candidate + 10 proxy channels.
    return inputs[:, 1:11] if inputs.shape[1] >= 11 else inputs[:, :10]


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_loss: float,
    dataset_hash: str,
    model_kind: str,
    model_config: dict[str, Any],
    phase_config: dict[str, Any],
    discriminator: nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    *,
    step_in_epoch: int = 0,
    global_step: int = 0,
    epoch_complete: bool = True,
    no_improvement: int = 0,
) -> None:
    ensure_dir(path.parent)
    temp = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "model_kind": model_kind,
        "model_config": model_config,
        "phase_config": phase_config,
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "epoch_complete": bool(epoch_complete),
        "no_improvement": int(no_improvement),
        "best_loss": float(best_loss),
        "dataset_hash": dataset_hash,
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    if discriminator is not None:
        payload["discriminator_state"] = discriminator.state_dict()
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer_state"] = discriminator_optimizer.state_dict()
    torch.save(payload, temp)
    durable_replace(temp, path)


def _load_model_state(model: nn.Module, checkpoint: dict[str, Any], prefer_ema: bool = True) -> None:
    state = checkpoint.get("ema_state") if prefer_ema and checkpoint.get("ema_state") else checkpoint["model_state"]
    model.load_state_dict(state)


def load_generator(
    checkpoint_path: str | Path,
    device: torch.device,
    prefer_ema: bool = True,
) -> tuple[FontStyleNetFinal, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = FontStyleNetFinal(
        base=int(config.get("base_channels", 32)),
        in_channels=int(config.get("input_channels", 10)),
        out_channels=int(config.get("output_channels", 4)),
    ).to(device)
    _load_model_state(model, checkpoint, prefer_ema=prefer_ema)
    model.eval()
    return model, checkpoint


def load_refiner(
    checkpoint_path: str | Path,
    device: torch.device,
    prefer_ema: bool = True,
) -> tuple[GlyphRefinerFinal, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = GlyphRefinerFinal(
        base=int(config.get("base_channels", 24)),
        in_channels=int(config.get("input_channels", 11)),
        out_channels=int(config.get("output_channels", 4)),
    ).to(device)
    _load_model_state(model, checkpoint, prefer_ema=prefer_ema)
    model.eval()
    return model, checkpoint


def _metric_keys() -> list[str]:
    return [
        "bce",
        "dice_loss",
        "edge_loss",
        "cldice_loss",
        "proxy_skeleton_loss",
        "proxy_layout_loss",
        "sdf_loss",
        "skeleton_head_loss",
        "edge_head_loss",
        "topology_point_loss",
        "boundary_distance_loss",
        "style_signature_loss",
    ]


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: FontLossFinal,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "proxy_target_gap": 0.0,
        "prediction_target_l1": 0.0,
        "prediction_proxy_l1": 0.0,
        "style_direction": 0.0,
        **{key: 0.0 for key in _metric_keys()},
    }
    seen = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        target_aux = batch["target_aux"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        with _autocast(device, amp_enabled):
            prediction = model(inputs)
            loss, pieces = criterion(
                prediction,
                target,
                weight,
                content_proxy=_extract_content_proxy(inputs),
                target_aux=target_aux,
            )
            probability = ink_probability(prediction)
        count = inputs.shape[0]
        seen += count
        totals["loss"] += float(loss.item()) * count
        totals["dice"] += float(batch_binary_dice(probability, target).mean().item()) * count
        proxy_ink = _extract_content_proxy(inputs)[:, 0:1].clamp(0.0, 1.0)
        proxy_target_gap = (proxy_ink - target).abs().mean(dim=(1, 2, 3))
        pred_target_l1 = (probability - target).abs().mean(dim=(1, 2, 3))
        pred_proxy_l1 = (probability - proxy_ink).abs().mean(dim=(1, 2, 3))
        style_direction = (pred_proxy_l1 - pred_target_l1) / proxy_target_gap.clamp_min(1e-5)
        totals["proxy_target_gap"] += float(proxy_target_gap.mean().item()) * count
        totals["prediction_target_l1"] += float(pred_target_l1.mean().item()) * count
        totals["prediction_proxy_l1"] += float(pred_proxy_l1.mean().item()) * count
        totals["style_direction"] += float(style_direction.mean().item()) * count
        for key in _metric_keys():
            totals[key] += float(pieces[key].item()) * count
    return {key: value / max(1, seen) for key, value in totals.items()}


@torch.no_grad()
def _save_preview(
    model: nn.Module,
    ema_model: nn.Module,
    loader: Iterable,
    device: torch.device,
    amp_enabled: bool,
    output: Path,
) -> None:
    model.eval()
    ema_model.eval()
    batch = next(iter(loader))
    inputs = batch["input"].to(device)
    target = batch["target"].to(device)
    with _autocast(device, amp_enabled):
        raw_probability = ink_probability(model(inputs))
        ema_probability = ink_probability(ema_model(inputs))
    base_index = 1 if inputs.shape[1] >= 11 else 0
    base = inputs[:, base_index].detach().float().cpu().numpy()
    raw_prediction = raw_probability[:, 0].detach().float().cpu().numpy()
    ema_prediction = ema_probability[:, 0].detach().float().cpu().numpy()
    truth = target[:, 0].detach().float().cpu().numpy()
    cps = [int(value) for value in batch["codepoint"]]
    make_training_preview(base, raw_prediction, ema_prediction, truth, cps, output, max_items=10)


def _enforce_style_guard(
    cfg: dict[str, Any],
    phase: dict[str, Any],
    epoch: int,
    validation: dict[str, float],
    phase_dir: Path,
) -> None:
    guard = cfg.get("style_guard", {})
    if not bool(guard.get("enabled", True)):
        return
    warmup = int(guard.get("warmup_epochs", 16))
    if str(phase.get("name", "")).lower() != "foundation256":
        warmup = min(warmup, 5)
    gap = float(validation.get("proxy_target_gap", 0.0))
    direction = float(validation.get("style_direction", 0.0))
    if epoch < warmup or gap < float(guard.get("minimum_proxy_target_gap", 0.035)):
        return
    minimum = float(guard.get("minimum_style_direction", -0.03))
    if direction >= minimum:
        return
    report = {
        "status": "style_collapse_detected",
        "phase": phase.get("name"),
        "epoch": int(epoch),
        "proxy_target_gap": gap,
        "prediction_target_l1": float(validation.get("prediction_target_l1", 0.0)),
        "prediction_proxy_l1": float(validation.get("prediction_proxy_l1", 0.0)),
        "style_direction": direction,
        "minimum_allowed_style_direction": minimum,
        "explanation": (
            "Prediction is becoming closer to the canonical content proxy than to the target.ttf truth. "
            "The run was stopped to prevent reference/proxy-style collapse."
        ),
    }
    save_json(phase_dir / "STYLE_COLLAPSE_DETECTED.json", report)
    if bool(guard.get("abort_on_collapse", True)):
        raise RuntimeError(
            "Style collapse detected: the prediction is moving toward the structural proxy instead of target.ttf. "
            f" phase={phase.get('name')} epoch={epoch} style_direction={direction:.4f}."
            "STYLE_COLLAPSE_DETECTED.json was saved and training was stopped."
        )


def _adversarial_settings(cfg: dict[str, Any], phase: dict[str, Any], refiner: bool) -> dict[str, Any]:
    settings = dict(cfg.get("adversarial", {}))
    enabled = bool(settings.get("enabled", False)) and bool(phase.get("adversarial", not refiner))
    settings["enabled"] = enabled
    return settings


def _discriminator_step(
    discriminator: PatchDiscriminatorFinal,
    optimizer: torch.optim.Optimizer,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    real_score = discriminator(real)
    fake_score = discriminator(fake.detach())
    loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def _generator_adversarial_loss(
    discriminator: PatchDiscriminatorFinal,
    real: torch.Tensor,
    fake: torch.Tensor,
    feature_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fake_score, fake_features = discriminator(fake, return_features=True)
    with torch.no_grad():
        _, real_features = discriminator(real, return_features=True)
    adversarial = -fake_score.mean()
    feature = torch.zeros((), device=fake.device)
    for fake_feature, real_feature in zip(fake_features, real_features):
        feature = feature + F.l1_loss(fake_feature, real_feature)
    feature = feature / max(1, len(fake_features))
    return adversarial, feature * float(feature_weight)


def _run_phase(
    *,
    model: nn.Module,
    model_kind: str,
    model_config: dict[str, Any],
    ema_decay: float,
    index_csv: Path,
    phase: dict[str, Any],
    phase_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
    init_checkpoint: Path | None = None,
    refiner: bool = False,
    resume: bool = True,
) -> Path:
    dataset_hash = sha256_file(index_csv)
    amp_enabled = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    longrun_guard = LongRunGuard(cfg)

    def phase_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
        ensure_dir(phase_dir)
        return (
            ensure_dir(phase_dir / "previews"),
            phase_dir / "history.csv",
            phase_dir / "best.pt",
            phase_dir / "last.pt",
            phase_dir / "in_epoch.pt",
            phase_dir / "completed.json",
        )

    previews, history_path, best_path, last_path, in_epoch_path, completed_path = phase_paths()

    def checkpoint_compatible(checkpoint: dict[str, Any]) -> bool:
        if int(checkpoint.get("version", 0)) != CHECKPOINT_FORMAT_VERSION:
            return False
        if checkpoint.get("dataset_hash") != dataset_hash:
            return False
        if str(checkpoint.get("model_kind", "")) != str(model_kind):
            return False
        if dict(checkpoint.get("model_config", {})) != dict(model_config):
            return False
        previous_phase = dict(checkpoint.get("phase_config", {}))
        for key in ("name", "size", "batch_size", "gradient_accumulation", "learning_rate"):
            if previous_phase.get(key) != phase.get(key):
                return False
        return True

    def newest_resume_checkpoint() -> Path | None:
        candidates = [path for path in (in_epoch_path, last_path, best_path) if path.is_file()]
        candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for path in candidates:
            try:
                checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
                if checkpoint_compatible(checkpoint):
                    return path
            except Exception:
                # A power loss may leave a corrupt newest file. Atomic writes normally avoid this,
                # but falling back to the previous epoch checkpoint is safer for month-long runs.
                continue
        return None

    existing_files = [path for path in (in_epoch_path, last_path, best_path) if path.is_file()]
    existing_checkpoint_path = newest_resume_checkpoint() if resume else None
    if resume and existing_files and existing_checkpoint_path is None:
        if bool(cfg["training"].get("reset_incompatible_checkpoints", True)):
            shutil.rmtree(phase_dir)
            previews, history_path, best_path, last_path, in_epoch_path, completed_path = phase_paths()
        else:
            raise RuntimeError(f"{existing_files[0]} is incompatible with the current HanziStyleForge configuration or the checkpoint is corrupted.")

    if completed_path.exists() and best_path.exists() and resume:
        checkpoint = torch.load(str(best_path), map_location=device, weights_only=False)
        try:
            completed = load_json(completed_path)
        except Exception:
            completed = {}
        if checkpoint_compatible(checkpoint) and int(completed.get("configured_epochs", 0)) >= int(phase["epochs"]):
            _load_model_state(model, checkpoint, prefer_ema=True)
            return best_path
        completed_path.unlink(missing_ok=True)

    if init_checkpoint is not None and init_checkpoint.exists() and not last_path.exists() and not in_epoch_path.exists():
        checkpoint = torch.load(str(init_checkpoint), map_location=device, weights_only=False)
        _load_model_state(model, checkpoint, prefer_ema=True)

    size = int(phase["size"])
    batch_size = int(phase["batch_size"])
    workers = int(cfg["training"].get("workers", 0))
    pin = device.type == "cuda"
    balanced = bool(cfg["training"].get("balanced_sampling", True))
    samples_per_epoch = int(phase.get("samples_per_epoch", cfg["training"].get("samples_per_epoch", 0)))
    if refiner:
        train_loader, val_loader = make_refiner_loaders(
            index_csv,
            size,
            batch_size,
            workers,
            pin,
            balanced=balanced,
            samples_per_epoch=samples_per_epoch,
        )
    else:
        train_loader, val_loader = make_style_loaders(
            index_csv,
            size,
            batch_size,
            workers,
            pin,
            balanced=balanced,
            samples_per_epoch=samples_per_epoch,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(phase["learning_rate"]),
        betas=(0.9, 0.99),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0001)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.55,
        patience=max(2, int(phase.get("early_stopping_patience", 8)) // 3),
        min_lr=float(phase.get("minimum_learning_rate", 8e-7)),
    )
    scaler = _grad_scaler(device, amp_enabled)
    criterion = FontLossFinal(cfg.get("loss", {}).get("weights", {})).to(device)
    ema = ModelEMA(model, decay=float(ema_decay))

    adv_cfg = _adversarial_settings(cfg, phase, refiner)
    discriminator: PatchDiscriminatorFinal | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    if adv_cfg["enabled"]:
        discriminator = PatchDiscriminatorFinal(base=int(adv_cfg.get("base_channels", 24))).to(device)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(adv_cfg.get("learning_rate", 8e-5)),
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

    start_epoch = 1
    best_loss = math.inf
    global_step = 0
    resume_replayed_step = 0
    history: list[dict[str, Any]] = []
    resume_path = newest_resume_checkpoint() if resume else None
    if resume_path is not None:
        checkpoint = torch.load(str(resume_path), map_location=device, weights_only=False)
        if not checkpoint_compatible(checkpoint):
            raise RuntimeError(f"{resume_path} is incompatible with the current training configuration.")
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("ema_state"):
            ema.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        if discriminator is not None and checkpoint.get("discriminator_state"):
            discriminator.load_state_dict(checkpoint["discriminator_state"])
        if discriminator_optimizer is not None and checkpoint.get("discriminator_optimizer_state"):
            discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state"])
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        start_epoch = int(checkpoint["epoch"]) + (1 if epoch_complete else 0)
        resume_replayed_step = 0 if epoch_complete else int(checkpoint.get("step_in_epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", math.inf))
        no_improvement = int(checkpoint.get("no_improvement", 0))
        if history_path.exists():
            with history_path.open("r", encoding="utf-8-sig", newline="") as file:
                history = list(csv.DictReader(file))

    epochs = int(phase["epochs"])
    accumulation = max(1, int(phase.get("gradient_accumulation", 1)))
    patience = int(phase.get("early_stopping_patience", 8))
    no_improvement = int(locals().get("no_improvement", 0))
    checkpoint_every_steps = max(0, int(cfg["training"].get("checkpoint_every_steps", 250)))
    started = time.time()
    adv_start = int(phase.get("adversarial_start_epoch", adv_cfg.get("start_epoch", max(2, epochs // 3))))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "adv_g": 0.0, "adv_d": 0.0, **{key: 0.0 for key in _metric_keys()}}
        seen = 0
        replay_note = f" replay<={resume_replayed_step} steps" if epoch == start_epoch and resume_replayed_step > 0 else ""
        progress = tqdm(train_loader, desc=f"{phase['name']} {epoch:03d}/{epochs:03d}{replay_note}", unit="batch")
        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_aux = batch["target_aux"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            with _autocast(device, amp_enabled):
                prediction = model(inputs)
                loss, pieces = criterion(
                    prediction,
                    target,
                    weight,
                    content_proxy=_extract_content_proxy(inputs),
                    target_aux=target_aux,
                )
                fake = ink_probability(prediction)
                adv_g = torch.zeros((), device=device)
                feature_loss = torch.zeros((), device=device)
                if discriminator is not None and epoch >= adv_start:
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(False)
                    adv_g, feature_loss = _generator_adversarial_loss(
                        discriminator,
                        target,
                        fake,
                        float(adv_cfg.get("feature_matching_weight", 0.35)),
                    )
                    loss = loss + float(adv_cfg.get("weight", 0.008)) * (adv_g + feature_loss)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = step % accumulation == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            adv_d = torch.zeros((), device=device)
            if discriminator is not None and discriminator_optimizer is not None and epoch >= adv_start:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                adv_d = _discriminator_step(discriminator, discriminator_optimizer, target.float(), fake.float())

            count = inputs.shape[0]
            seen += count
            totals["loss"] += float(loss.detach().item()) * count
            totals["adv_g"] += float(adv_g.detach().item()) * count
            totals["adv_d"] += float(adv_d.detach().item()) * count
            for key in _metric_keys():
                totals[key] += float(pieces[key].item()) * count
            progress.set_postfix(loss=f"{totals['loss']/max(1,seen):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if should_step:
                global_step += 1
                if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                    _save_checkpoint(
                        in_epoch_path,
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_loss,
                        dataset_hash,
                        model_kind,
                        model_config,
                        phase,
                        discriminator,
                        discriminator_optimizer,
                        step_in_epoch=step,
                        global_step=global_step,
                        epoch_complete=False,
                        no_improvement=no_improvement,
                    )
                    longrun_guard.checkpoint_boundary()

        validation = _evaluate(ema.shadow, val_loader, criterion, device, amp_enabled)
        _enforce_style_guard(cfg, phase, epoch, validation, phase_dir)
        scheduler.step(validation["loss"])
        improved = validation["loss"] < best_loss - 1e-5
        if improved:
            best_loss = validation["loss"]
            no_improvement = 0
            _save_checkpoint(
                best_path,
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_loss,
                dataset_hash,
                model_kind,
                model_config,
                phase,
                discriminator,
                discriminator_optimizer,
                step_in_epoch=len(train_loader),
                global_step=global_step,
                epoch_complete=True,
                no_improvement=no_improvement,
            )
        else:
            no_improvement += 1
        _save_checkpoint(
            last_path,
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_loss,
            dataset_hash,
            model_kind,
            model_config,
            phase,
            discriminator,
            discriminator_optimizer,
            step_in_epoch=len(train_loader),
            global_step=global_step,
            epoch_complete=True,
            no_improvement=no_improvement,
        )
        in_epoch_path.unlink(missing_ok=True)
        longrun_guard.checkpoint_boundary()

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(1, seen),
            "train_adversarial_g": totals["adv_g"] / max(1, seen),
            "train_adversarial_d": totals["adv_d"] / max(1, seen),
            "val_loss": validation["loss"],
            "val_dice": validation["dice"],
            "val_proxy_target_gap": validation["proxy_target_gap"],
            "val_prediction_target_l1": validation["prediction_target_l1"],
            "val_prediction_proxy_l1": validation["prediction_proxy_l1"],
            "val_style_direction": validation["style_direction"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best": 1 if improved else 0,
            "elapsed_seconds": int(time.time() - started),
        }
        for key in _metric_keys():
            row[f"train_{key}"] = totals[key] / max(1, seen)
            row[f"val_{key}"] = validation[key]
        history.append(row)
        write_csv(history_path, history, fieldnames=list(row.keys()))
        if epoch == 1 or epoch % int(cfg["training"].get("preview_every", 2)) == 0 or improved:
            _save_preview(model, ema.shadow, val_loader, device, amp_enabled, previews / f"epoch_{epoch:03d}.png")
        resume_replayed_step = 0
        if patience > 0 and no_improvement >= patience:
            break

    if not best_path.exists():
        if last_path.exists():
            shutil.copy2(last_path, best_path)
        else:
            raise RuntimeError(f"Stage {phase['name']} did not produce a checkpoint.")
    best = torch.load(str(best_path), map_location=device, weights_only=False)
    _load_model_state(model, best, prefer_ema=True)
    last_checkpoint = torch.load(str(last_path), map_location="cpu", weights_only=False) if last_path.exists() else best
    save_json(
        completed_path,
        {
            "version": CHECKPOINT_FORMAT_VERSION,
            "phase": phase["name"],
            "best_loss": best.get("best_loss"),
            "checkpoint": str(best_path.resolve()),
            "configured_epochs": int(phase["epochs"]),
            "completed_epoch": int(last_checkpoint.get("epoch", 0)),
            "global_step": int(last_checkpoint.get("global_step", 0)),
            "checkpoint_every_steps": checkpoint_every_steps,
            "dataset_hash": dataset_hash,
        },
    )
    return best_path


@torch.no_grad()
def calibrate_threshold(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    thresholds = np.arange(0.30, 0.721, 0.015)
    sums = np.zeros_like(thresholds, dtype=np.float64)
    count = 0
    model.eval()
    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        with _autocast(device, amp_enabled):
            probability = ink_probability(model(inputs))
        for index, threshold in enumerate(thresholds):
            sums[index] += float(batch_binary_dice(probability, target, float(threshold)).sum().item())
        count += inputs.shape[0]
    means = sums / max(1, count)
    best_index = int(np.argmax(means))
    return {
        "threshold": float(thresholds[best_index]),
        "validation_dice": float(means[best_index]),
        "tested": {f"{threshold:.3f}": float(value) for threshold, value in zip(thresholds, means)},
    }


def train_generator(
    cfg: dict[str, Any],
    index_csv: str | Path | None = None,
    model_root: str | Path | None = None,
    phases_override: list[dict[str, Any]] | None = None,
    init_checkpoint: str | Path | None = None,
    resume: bool | None = None,
) -> dict[str, Any]:
    validate_data_flow_contract(cfg, require_prepared=True, write_report=True)
    set_seed(int(cfg["training"]["seed"]))
    device = _resolve_device(cfg)
    work = Path(cfg["paths"]["work_dir"])
    index = Path(index_csv) if index_csv else work / "dataset" / "index.csv"
    if not index.exists():
        raise FileNotFoundError("dataset/index.csv was not found in the current work directory. Run prepare first.")
    root = ensure_dir(model_root or (work / "model" / "generator"))
    model_config = {
        "base_channels": int(cfg["training"]["base_channels"]),
        "input_channels": 10,
        "output_channels": 4,
        "architecture": "topology_multitask_attention_final",
        # These fields are ignored by load_generator but make resume safety strict:
        # checkpoints trained with a different seed/loss/adversarial recipe are not reused.
        "training_seed": int(cfg["training"]["seed"]),
        "loss_weights": {str(k): float(v) for k, v in cfg.get("loss", {}).get("weights", {}).items()},
        "adversarial": dict(cfg.get("adversarial", {})),
    }
    model = FontStyleNetFinal(base=model_config["base_channels"], in_channels=10, out_channels=4).to(device)
    phases = phases_override or list(cfg["training"]["phases"])
    current_init = Path(init_checkpoint) if init_checkpoint else None
    resume_enabled = bool(cfg["training"].get("resume_if_exists", True) if resume is None else resume)
    phase_results: list[dict[str, Any]] = []
    best_path: Path | None = None
    for phase in phases:
        phase_dir = root / str(phase["name"])
        best_path = _run_phase(
            model=model,
            model_kind="generator",
            model_config=model_config,
            ema_decay=float(cfg["training"].get("ema_decay", 0.9995)),
            index_csv=index,
            phase=phase,
            phase_dir=phase_dir,
            cfg=cfg,
            device=device,
            init_checkpoint=current_init,
            refiner=False,
            resume=resume_enabled,
        )
        current_init = best_path
        phase_results.append({"name": phase["name"], "checkpoint": str(best_path.resolve())})

    assert best_path is not None
    final_path = root / "generator_best.pt"
    shutil.copy2(best_path, final_path)
    final_model, _ = load_generator(final_path, device)
    last_phase = phases[-1]
    _, val_loader = make_style_loaders(
        index,
        int(last_phase["size"]),
        max(1, int(last_phase["batch_size"])),
        int(cfg["training"].get("workers", 0)),
        device.type == "cuda",
        balanced=False,
    )
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    calibration = calibrate_threshold(final_model, val_loader, device, amp)
    validation = _evaluate(
        final_model,
        val_loader,
        FontLossFinal(cfg.get("loss", {}).get("weights", {})).to(device),
        device,
        amp,
    )
    save_json(root / "calibration.json", calibration)
    save_json(root / "validation.json", validation)
    result = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint": str(final_path.resolve()),
        "parameters": count_parameters(final_model),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "phases": phase_results,
        "calibration": calibration,
        "validation": validation,
    }
    save_json(root / "summary.json", result)
    return result


def train_refiner(cfg: dict[str, Any]) -> dict[str, Any]:
    if not bool(cfg["refiner"].get("enabled", True)):
        return {"enabled": False}
    set_seed(int(cfg["training"]["seed"]) + 91)
    device = _resolve_device(cfg)
    work = Path(cfg["paths"]["work_dir"])
    index = work / "dataset" / "index.csv"
    root = ensure_dir(work / "model" / "refiner")
    ref_cfg = cfg["refiner"]
    model_config = {
        "base_channels": int(ref_cfg["base_channels"]),
        "input_channels": 11,
        "output_channels": 4,
        "architecture": "multitask_refiner_final",
        "training_seed": int(cfg["training"]["seed"]) + 91,
        "loss_weights": {str(k): float(v) for k, v in cfg.get("loss", {}).get("weights", {}).items()},
        "adversarial": dict(cfg.get("adversarial", {})),
    }
    model = GlyphRefinerFinal(base=model_config["base_channels"], in_channels=11, out_channels=4).to(device)
    phase = {
        "name": "detail_refiner",
        "size": int(ref_cfg["size"]),
        "epochs": int(ref_cfg["epochs"]),
        "batch_size": int(ref_cfg["batch_size"]),
        "gradient_accumulation": int(ref_cfg["gradient_accumulation"]),
        "learning_rate": float(ref_cfg["learning_rate"]),
        "early_stopping_patience": int(ref_cfg["early_stopping_patience"]),
        "samples_per_epoch": int(ref_cfg.get("samples_per_epoch", 0)),
        "adversarial": bool(ref_cfg.get("adversarial", False)),
    }
    best_path = _run_phase(
        model=model,
        model_kind="refiner",
        model_config=model_config,
        ema_decay=float(ref_cfg.get("ema_decay", 0.9995)),
        index_csv=index,
        phase=phase,
        phase_dir=root / "detail_refiner",
        cfg=cfg,
        device=device,
        init_checkpoint=None,
        refiner=True,
        resume=bool(cfg["training"].get("resume_if_exists", True)),
    )
    final_path = root / "refiner_best.pt"
    shutil.copy2(best_path, final_path)
    final_model, _ = load_refiner(final_path, device)
    _, val_loader = make_refiner_loaders(
        index,
        int(ref_cfg["size"]),
        max(1, int(ref_cfg["batch_size"])),
        int(cfg["training"].get("workers", 0)),
        device.type == "cuda",
        balanced=False,
    )
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    calibration = calibrate_threshold(final_model, val_loader, device, amp)
    validation = _evaluate(
        final_model,
        val_loader,
        FontLossFinal(cfg.get("loss", {}).get("weights", {})).to(device),
        device,
        amp,
    )
    save_json(root / "calibration.json", calibration)
    save_json(root / "validation.json", validation)
    result = {
        "enabled": True,
        "checkpoint": str(final_path.resolve()),
        "parameters": count_parameters(final_model),
        "calibration": calibration,
        "validation": validation,
    }
    save_json(root / "summary.json", result)
    return result


def train_all(cfg: dict[str, Any]) -> dict[str, Any]:
    generator = train_generator(cfg)
    refiner = train_refiner(cfg)
    result = {"version": CHECKPOINT_FORMAT_VERSION, "generator": generator, "refiner": refiner}
    save_json(Path(cfg["paths"]["work_dir"]) / "model" / "summary.json", result)
    return result
