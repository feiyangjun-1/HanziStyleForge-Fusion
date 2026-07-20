from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import _read_proxy4
from .features import expand_proxy_channels, target_aux_from_path
from .util import read_csv


def read_ink_image(path: str | Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - gray
    if ink.shape != (int(size), int(size)):
        ink = cv2.resize(ink, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    return ink.clip(0.0, 1.0).astype(np.float32)


def _safe_affine(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = image.shape
    return cv2.warpAffine(
        image,
        matrix.astype(np.float32),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).clip(0.0, 1.0)


def _morph(image: np.ndarray, delta: float) -> np.ndarray:
    rounded = int(round(abs(float(delta))))
    if rounded <= 0:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rounded * 2 + 1, rounded * 2 + 1))
    if delta > 0:
        return cv2.dilate(image, kernel).astype(np.float32)
    return cv2.erode(image, kernel).astype(np.float32)


@dataclass(frozen=True)
class SyntheticStyle:
    weight: float
    slant: float
    scale_x: float
    scale_y: float
    blur: float
    roundness: float
    contrast: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.weight, self.slant, self.scale_x, self.scale_y, self.blur, self.roundness, self.contrast],
            dtype=np.float32,
        )


def random_synthetic_style(rng: random.Random, identity_probability: float = 0.38) -> SyntheticStyle:
    if rng.random() < float(identity_probability):
        return SyntheticStyle(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0)
    return SyntheticStyle(
        weight=rng.uniform(-1.8, 2.4),
        slant=rng.uniform(-0.12, 0.12),
        scale_x=rng.uniform(0.94, 1.06),
        scale_y=rng.uniform(0.94, 1.06),
        blur=rng.uniform(0.0, 1.15),
        roundness=rng.uniform(-0.8, 1.2),
        contrast=rng.uniform(0.88, 1.12),
    )


def apply_synthetic_style(ink: np.ndarray, style: SyntheticStyle) -> np.ndarray:
    image = np.asarray(ink, dtype=np.float32).copy()
    height, width = image.shape
    center_x, center_y = width / 2.0, height / 2.0
    matrix = np.asarray(
        [
            [style.scale_x, style.slant, center_x - style.scale_x * center_x - style.slant * center_y],
            [0.0, style.scale_y, center_y - style.scale_y * center_y],
        ],
        dtype=np.float32,
    )
    image = _safe_affine(image, matrix)
    image = _morph(image, style.weight)
    if abs(style.roundness) > 0.15:
        radius = max(1, int(round(abs(style.roundness) * 1.8)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        if style.roundness > 0:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.GaussianBlur(image, (0, 0), 0.25 + 0.25 * radius)
        else:
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    if style.blur > 0.05:
        image = cv2.GaussianBlur(image, (0, 0), style.blur)
    image = np.clip((image - 0.5) * style.contrast + 0.5, 0.0, 1.0)
    return image.astype(np.float32)


class TargetStylePool:
    def __init__(self, index_csv: str | Path, *, split: str | None = None) -> None:
        rows = [row for row in read_csv(index_csv) if row.get("mode") == "self"]
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]
        if not rows:
            raise RuntimeError("target style pool is empty")
        self.rows = rows

    def sample_paths(self, rng: random.Random, count: int, exclude_codepoint: int | None = None) -> list[str]:
        candidates = self.rows
        if exclude_codepoint is not None and len(candidates) > count:
            filtered = [row for row in candidates if int(row["codepoint"]) != int(exclude_codepoint)]
            if len(filtered) >= count:
                candidates = filtered
        if len(candidates) >= count:
            selected = rng.sample(candidates, count)
        else:
            selected = [rng.choice(candidates) for _ in range(count)]
        return [str(row["target_path"]) for row in selected]

    def deterministic_paths(self, count: int, seed: int = 0) -> list[str]:
        rng = random.Random(int(seed))
        return self.sample_paths(rng, min(int(count), max(1, len(self.rows))))


class StyleEncoderPretrainDataset(Dataset):
    """Synthetic pseudo-font task for a target-only style encoder.

    Two disjoint glyph sets receive the same synthetic style transform and form
    a positive pair. A third set receives a different transform. The encoder
    learns stable target-font style factors without ever reading ref.otf.
    """

    def __init__(
        self,
        index_csv: str | Path,
        *,
        style_size: int = 128,
        references_per_set: int = 8,
        virtual_length: int = 12000,
        seed: int = 20260719,
    ) -> None:
        self.pool = TargetStylePool(index_csv, split="train")
        self.style_size = int(style_size)
        self.references_per_set = int(references_per_set)
        self.virtual_length = max(len(self.pool.rows), int(virtual_length))
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.virtual_length

    def _load_set(self, paths: Sequence[str], style: SyntheticStyle) -> np.ndarray:
        glyphs = [apply_synthetic_style(read_ink_image(path, self.style_size), style) for path in paths]
        return np.stack(glyphs, axis=0)[:, None, :, :]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + int(index) * 1000003 + random.randrange(1 << 20))
        positive_style = random_synthetic_style(rng)
        negative_style = random_synthetic_style(rng, identity_probability=0.12)
        count = self.references_per_set
        paths_a = self.pool.sample_paths(rng, count)
        paths_b = self.pool.sample_paths(rng, count)
        paths_negative = self.pool.sample_paths(rng, count)
        return {
            "positive_a": torch.from_numpy(self._load_set(paths_a, positive_style)),
            "positive_b": torch.from_numpy(self._load_set(paths_b, positive_style)),
            "negative": torch.from_numpy(self._load_set(paths_negative, negative_style)),
            "positive_parameters": torch.from_numpy(positive_style.as_array()),
            "negative_parameters": torch.from_numpy(negative_style.as_array()),
        }


class VQGlyphDataset(Dataset):
    def __init__(self, index_csv: str | Path, *, split: str, size: int, augment: bool = False) -> None:
        rows = [row for row in read_csv(index_csv) if row.get("mode") == "self" and row.get("split") == split]
        if not rows:
            raise RuntimeError(f"no VQ samples for split={split}")
        self.rows = rows
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target = target_aux_from_path(row["target_path"], row.get("target_aux_path", ""), self.size)
        if self.augment and random.random() < 0.45:
            dx = random.uniform(-1.2, 1.2) * self.size / 256.0
            dy = random.uniform(-1.2, 1.2) * self.size / 256.0
            scale = random.uniform(0.985, 1.015)
            matrix = cv2.getRotationMatrix2D((self.size / 2.0, self.size / 2.0), 0.0, scale)
            matrix[:, 2] += [dx, dy]
            channels = [
                cv2.warpAffine(
                    target[..., channel], matrix, (self.size, self.size),
                    flags=cv2.INTER_LINEAR, borderValue=0.5 if channel == 1 else 0.0,
                )
                for channel in range(target.shape[-1])
            ]
            target = np.stack(channels, axis=-1).clip(0.0, 1.0)
        return {
            "target_aux": torch.from_numpy(np.moveaxis(target.astype(np.float32), -1, 0)),
            "codepoint": int(row["codepoint"]),
            "complexity": torch.tensor(float(row.get("complexity", 0.0) or 0.0), dtype=torch.float32),
        }


class FusionDiffusionDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        *,
        split: str,
        size: int,
        style_size: int,
        style_references: int,
        augment: bool = False,
        hard_codepoints: set[int] | None = None,
        hard_repeat: int = 4,
        seed: int = 20260719,
    ) -> None:
        rows = [row for row in read_csv(index_csv) if row.get("mode") == "self" and row.get("split") == split]
        if hard_codepoints:
            hard_rows = [row for row in rows if int(row["codepoint"]) in hard_codepoints]
            rows = rows + hard_rows * max(0, int(hard_repeat) - 1)
        if not rows:
            raise RuntimeError(f"no diffusion samples for split={split}")
        self.rows = rows
        self.size = int(size)
        self.style_size = int(style_size)
        self.style_references = int(style_references)
        self.augment = bool(augment)
        self.pool = TargetStylePool(index_csv, split="train")
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _style_refs(self, rng: random.Random, exclude_codepoint: int) -> np.ndarray:
        paths = self.pool.sample_paths(rng, self.style_references, exclude_codepoint=exclude_codepoint)
        glyphs = [read_ink_image(path, self.style_size) for path in paths]
        return np.stack(glyphs, axis=0)[:, None, :, :]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        cp = int(row["codepoint"])
        proxy = expand_proxy_channels(_read_proxy4(row["proxy_path"], self.size))
        target_aux = target_aux_from_path(row["target_path"], row.get("target_aux_path", ""), self.size)
        rng = random.Random(self.seed + index * 65537 + random.randrange(1 << 20))
        if self.augment:
            if rng.random() < 0.55:
                # Content proxy augmentation only. Target style remains the real target font.
                delta = rng.choice([-1, 0, 0, 1])
                if delta:
                    radius = abs(delta)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                    proxy[..., 0] = (
                        cv2.dilate(proxy[..., 0], kernel) if delta > 0 else cv2.erode(proxy[..., 0], kernel)
                    )
                if rng.random() < 0.25:
                    proxy[..., 5:7] = 0.5
            if rng.random() < 0.35:
                dx = rng.uniform(-1.2, 1.2) * self.size / 256.0
                dy = rng.uniform(-1.2, 1.2) * self.size / 256.0
                scale = rng.uniform(0.988, 1.012)
                matrix = cv2.getRotationMatrix2D((self.size / 2.0, self.size / 2.0), 0.0, scale)
                matrix[:, 2] += [dx, dy]
                pchannels = [
                    cv2.warpAffine(
                        proxy[..., channel], matrix, (self.size, self.size), flags=cv2.INTER_LINEAR,
                        borderValue=0.5 if channel in {2, 5, 6} else 0.0,
                    )
                    for channel in range(proxy.shape[-1])
                ]
                tchannels = [
                    cv2.warpAffine(
                        target_aux[..., channel], matrix, (self.size, self.size), flags=cv2.INTER_LINEAR,
                        borderValue=0.5 if channel == 1 else 0.0,
                    )
                    for channel in range(target_aux.shape[-1])
                ]
                proxy = np.stack(pchannels, axis=-1).clip(0.0, 1.0)
                target_aux = np.stack(tchannels, axis=-1).clip(0.0, 1.0)
        return {
            "proxy": torch.from_numpy(np.moveaxis(proxy.astype(np.float32), -1, 0)),
            "target_aux": torch.from_numpy(np.moveaxis(target_aux.astype(np.float32), -1, 0)),
            "style_refs": torch.from_numpy(self._style_refs(rng, cp).astype(np.float32)),
            "codepoint": cp,
            "complexity": torch.tensor(float(row.get("complexity", 0.0) or 0.0), dtype=torch.float32),
        }


class ReferenceFusionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, size: int) -> None:
        self.rows = rows
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        proxy = expand_proxy_channels(_read_proxy4(row["ref_proxy_path"], self.size))
        return {
            "proxy": torch.from_numpy(np.moveaxis(proxy.astype(np.float32), -1, 0)),
            "codepoint": int(row["codepoint"]),
        }


def corrupt_target_ink(clean: np.ndarray, rng: random.Random) -> np.ndarray:
    image = np.asarray(clean, dtype=np.float32).copy()
    size = image.shape[0]
    if rng.random() < 0.85:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.25, 1.4) * size / 384.0)
    if rng.random() < 0.75:
        threshold = rng.uniform(0.38, 0.62)
        hard = (image >= threshold).astype(np.uint8)
        delta = rng.choice([-2, -1, 0, 0, 1, 2])
        if delta:
            radius = abs(delta)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            hard = cv2.dilate(hard, kernel) if delta > 0 else cv2.erode(hard, kernel)
        image = hard.astype(np.float32)
    if rng.random() < 0.45:
        for _ in range(rng.randint(1, 8)):
            x = rng.randrange(size)
            y = rng.randrange(size)
            radius = rng.randint(1, max(1, size // 180 + 1))
            cv2.circle(image, (x, y), radius, 0.0 if rng.random() < 0.7 else 1.0, -1)
    return image.clip(0.0, 1.0)


class FusionRefinerDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        *,
        split: str,
        size: int,
        style_size: int,
        style_references: int,
        augment: bool,
        seed: int = 20260719,
    ) -> None:
        self.rows = [
            row for row in read_csv(index_csv)
            if row.get("mode") == "self" and row.get("split") == split
        ]
        if not self.rows:
            raise RuntimeError(f"no refiner samples for split={split}")
        self.size = int(size)
        self.style_size = int(style_size)
        self.style_references = int(style_references)
        self.augment = bool(augment)
        self.pool = TargetStylePool(index_csv, split="train")
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        cp = int(row["codepoint"])
        proxy = expand_proxy_channels(_read_proxy4(row["proxy_path"], self.size))
        target_aux = target_aux_from_path(row["target_path"], row.get("target_aux_path", ""), self.size)
        rng = random.Random(self.seed + index * 524287 + random.randrange(1 << 20))
        clean = target_aux[..., 0]
        candidate = corrupt_target_ink(clean, rng) if self.augment else cv2.GaussianBlur(clean, (0, 0), 0.55)
        paths = self.pool.sample_paths(rng, self.style_references, exclude_codepoint=cp)
        style_refs = np.stack([read_ink_image(path, self.style_size) for path in paths], axis=0)[:, None]
        model_input = np.concatenate([candidate[..., None], proxy], axis=-1)
        return {
            "input": torch.from_numpy(np.moveaxis(model_input.astype(np.float32), -1, 0)),
            "target": torch.from_numpy(clean[None].astype(np.float32)),
            "target_aux": torch.from_numpy(np.moveaxis(target_aux.astype(np.float32), -1, 0)),
            "style_refs": torch.from_numpy(style_refs.astype(np.float32)),
            "codepoint": cp,
        }
