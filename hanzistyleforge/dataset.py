from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .features import expand_proxy_channels, target_aux_from_path
from .util import read_csv


def _seed_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def _loader_options(workers: int, pin_memory: bool) -> dict[str, Any]:
    worker_count = max(0, int(workers))
    options: dict[str, Any] = {
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(worker_count > 0),
    }
    if worker_count > 0:
        options["prefetch_factor"] = 3
        options["worker_init_fn"] = _seed_loader_worker
    return options


def _read_proxy4(path: str | Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    if arr.shape[0] != size or arr.shape[1] != size:
        channels = [cv2.resize(arr[..., i], (size, size), interpolation=cv2.INTER_AREA) for i in range(4)]
        arr = np.stack(channels, axis=-1)
    return arr.clip(0.0, 1.0)


def _joint_affine(proxy: np.ndarray, target_aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = target_aux.shape[0]
    scale = random.uniform(0.974, 1.026)
    dx = random.uniform(-1.8, 1.8) * size / 256.0
    dy = random.uniform(-1.8, 1.8) * size / 256.0
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), 0.0, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    proxy_channels = []
    for index in range(proxy.shape[-1]):
        border = 0.5 if index in {2, 5, 6} else 0.0
        proxy_channels.append(
            cv2.warpAffine(proxy[..., index], matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=border)
        )
    target_channels = []
    for index in range(target_aux.shape[-1]):
        border = 0.5 if index == 1 else 0.0
        target_channels.append(
            cv2.warpAffine(target_aux[..., index], matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=border)
        )
    return np.stack(proxy_channels, axis=-1).clip(0, 1), np.stack(target_channels, axis=-1).clip(0, 1)


def _morph_channel(channel: np.ndarray, delta: int) -> np.ndarray:
    if delta == 0:
        return channel
    radius = abs(int(delta))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    if delta > 0:
        return cv2.dilate(channel, kernel)
    return cv2.erode(channel, kernel)


def _proxy_jitter(proxy: np.ndarray) -> np.ndarray:
    result = proxy.copy()
    # Deliberately vary canonical stroke width while preserving the target.
    # This teaches the model that stroke weight belongs to the target font,
    # not to the content reference.
    if random.random() < 0.42:
        delta = random.choice([-1, 0, 0, 1])
        result[..., 0] = _morph_channel((result[..., 0] >= 0.45).astype(np.uint8), delta).astype(np.float32)
    if random.random() < 0.35:
        result[..., 1] = cv2.GaussianBlur(result[..., 1], (0, 0), random.uniform(0.25, 0.75))
    if random.random() < 0.30:
        gain = random.uniform(0.90, 1.10)
        result[..., 3] = np.clip(result[..., 3] * gain, 0.0, 1.0)
    if random.random() < 0.20:
        # Drop orientation cues together rather than damaging the topology.
        result[..., 5:7] = 0.5
    if random.random() < 0.18:
        result[..., 7:9] *= random.uniform(0.65, 0.95)
    return result.clip(0.0, 1.0)


def _corrupt_glyph(clean: np.ndarray) -> np.ndarray:
    image = clean.copy().astype(np.float32)
    size = image.shape[0]
    if random.random() < 0.75:
        image = cv2.GaussianBlur(image, (0, 0), random.uniform(0.30, 1.20) * size / 384.0)
    if random.random() < 0.70:
        threshold = random.uniform(0.37, 0.63)
        hard = (image >= threshold).astype(np.uint8)
        radius = random.choice([-2, -1, 0, 0, 0, 1, 2])
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            hard = cv2.dilate(hard, kernel)
        elif radius < 0:
            rr = abs(radius)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rr * 2 + 1, rr * 2 + 1))
            hard = cv2.erode(hard, kernel)
        image = hard.astype(np.float32)
    if random.random() < 0.45:
        count = random.randint(1, 8)
        for _ in range(count):
            x = random.randrange(size)
            y = random.randrange(size)
            radius = random.randint(1, max(1, size // 160 + 1))
            value = 0.0 if random.random() < 0.65 else 1.0
            cv2.circle(image, (x, y), radius, value, thickness=-1)
    if random.random() < 0.30:
        dx = random.uniform(-1.3, 1.3)
        dy = random.uniform(-1.3, 1.3)
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return image.clip(0.0, 1.0)


class GlyphStyleDataset(Dataset):
    def __init__(self, index_csv: str | Path, split: str, size: int, augment: bool = False) -> None:
        rows = read_csv(index_csv)
        self.rows = [row for row in rows if row.get("split") == split]
        if not self.rows:
            raise RuntimeError(f"The training manifest has no samples for split={split!r}.")
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        proxy = expand_proxy_channels(_read_proxy4(row["proxy_path"], self.size))
        target_aux = target_aux_from_path(
            row["target_path"], row.get("target_aux_path", ""), self.size
        )
        if self.augment:
            if random.random() < 0.72:
                proxy, target_aux = _joint_affine(proxy, target_aux)
            proxy = _proxy_jitter(proxy)
        target = target_aux[..., 0]
        return {
            "input": torch.from_numpy(np.moveaxis(proxy.astype(np.float32), -1, 0)),
            "target": torch.from_numpy(target[None, ...].astype(np.float32)),
            "target_aux": torch.from_numpy(np.moveaxis(target_aux.astype(np.float32), -1, 0)),
            "weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
            "complexity": torch.tensor(float(row.get("complexity", 0.0) or 0.0), dtype=torch.float32),
            "codepoint": int(row["codepoint"]),
            "mode": row.get("mode", "self"),
        }


class GlyphRefinerDataset(Dataset):
    def __init__(self, index_csv: str | Path, split: str, size: int, augment: bool = False) -> None:
        rows = read_csv(index_csv)
        self.rows = [row for row in rows if row.get("split") == split and row.get("mode") == "self"]
        if not self.rows:
            raise RuntimeError(f"The manifest has no self-reconstruction samples for the refiner in split={split!r}.")
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        proxy = expand_proxy_channels(_read_proxy4(row["proxy_path"], self.size))
        target_aux = target_aux_from_path(row["target_path"], row.get("target_aux_path", ""), self.size)
        if self.augment and random.random() < 0.68:
            proxy, target_aux = _joint_affine(proxy, target_aux)
        clean = target_aux[..., 0]
        corrupted = _corrupt_glyph(clean) if self.augment else cv2.GaussianBlur(clean, (0, 0), 0.55)
        inputs = np.concatenate([corrupted[..., None], proxy], axis=-1)
        return {
            "input": torch.from_numpy(np.moveaxis(inputs.astype(np.float32), -1, 0)),
            "target": torch.from_numpy(clean[None, ...].astype(np.float32)),
            "target_aux": torch.from_numpy(np.moveaxis(target_aux.astype(np.float32), -1, 0)),
            "weight": torch.tensor(1.0, dtype=torch.float32),
            "complexity": torch.tensor(float(row.get("complexity", 0.0) or 0.0), dtype=torch.float32),
            "codepoint": int(row["codepoint"]),
            "mode": "refiner",
        }


class ReferenceProxyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], size: int) -> None:
        self.rows = rows
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        proxy = expand_proxy_channels(_read_proxy4(row["ref_proxy_path"], self.size))
        return {
            "input": torch.from_numpy(np.moveaxis(proxy.astype(np.float32), -1, 0)),
            "codepoint": int(row["codepoint"]),
        }


def _balanced_weights(rows: list[dict[str, Any]]) -> list[float]:
    complexities = np.asarray([float(row.get("complexity", 0.0) or 0.0) for row in rows], dtype=np.float64)
    if len(rows) < 10 or float(np.ptp(complexities)) < 1e-8:
        return [1.0] * len(rows)
    edges = np.unique(np.quantile(complexities, [0.0, 0.18, 0.38, 0.60, 0.80, 1.0]))
    if len(edges) < 3:
        return [1.0] * len(rows)
    bins = np.clip(np.digitize(complexities, edges[1:-1], right=True), 0, len(edges) - 2)
    counts = np.bincount(bins, minlength=len(edges) - 1).astype(np.float64)
    weights = 1.0 / np.maximum(counts[bins], 1.0)
    # The upper complexity bins are slightly oversampled because they are the
    # characters most likely to lose short strokes or close counters.
    weights *= 1.0 + 0.16 * bins
    modes = [row.get("mode", "self") for row in rows]
    weights *= np.asarray([1.0 if mode == "self" else 0.55 for mode in modes], dtype=np.float64)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.tolist()


def _make_train_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    balanced: bool,
    samples_per_epoch: int,
) -> DataLoader:
    common = _loader_options(workers, pin_memory)
    if balanced and hasattr(dataset, "rows"):
        rows = getattr(dataset, "rows")
        weights = torch.as_tensor(_balanced_weights(rows), dtype=torch.double)
        count = int(samples_per_epoch) if int(samples_per_epoch) > 0 else len(dataset)
        sampler = WeightedRandomSampler(weights, num_samples=max(1, count), replacement=True)
        return DataLoader(dataset, batch_size=int(batch_size), sampler=sampler, drop_last=False, **common)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False, **common)


def make_style_loaders(
    index_csv: str | Path,
    size: int,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    *,
    balanced: bool = True,
    samples_per_epoch: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train = GlyphStyleDataset(index_csv, "train", size=size, augment=True)
    val = GlyphStyleDataset(index_csv, "val", size=size, augment=False)
    common = _loader_options(workers, pin_memory)
    return (
        _make_train_loader(train, batch_size, workers, pin_memory, balanced, samples_per_epoch),
        DataLoader(val, batch_size=max(1, min(int(batch_size), 10)), shuffle=False, drop_last=False, **common),
    )


def make_refiner_loaders(
    index_csv: str | Path,
    size: int,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    *,
    balanced: bool = True,
    samples_per_epoch: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train = GlyphRefinerDataset(index_csv, "train", size=size, augment=True)
    val = GlyphRefinerDataset(index_csv, "val", size=size, augment=False)
    common = _loader_options(workers, pin_memory)
    return (
        _make_train_loader(train, batch_size, workers, pin_memory, balanced, samples_per_epoch),
        DataLoader(val, batch_size=max(1, min(int(batch_size), 10)), shuffle=False, drop_last=False, **common),
    )
