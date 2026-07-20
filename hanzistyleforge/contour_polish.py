from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .fusion_model import ContourSequenceTransformer
from .util import ensure_dir, read_csv, save_json, sha256_file


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_closed_contour(points: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(source) < 3:
        raise ValueError("a closed contour needs at least three points")
    closed = np.concatenate([source, source[:1]], axis=0)
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(source[:1], count, axis=0)
    samples = np.linspace(0.0, total, int(count), endpoint=False)
    result = np.empty((int(count), 2), dtype=np.float32)
    index = 0
    for i, value in enumerate(samples):
        while index + 1 < len(cumulative) and cumulative[index + 1] <= value:
            index += 1
        index = min(index, len(source) - 1)
        denominator = max(segments[index], 1e-6)
        alpha = (value - cumulative[index]) / denominator
        result[i] = closed[index] * (1.0 - alpha) + closed[index + 1] * alpha
    return result


def normalize_contour(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(points, dtype=np.float32)
    center = source.mean(axis=0)
    centered = source - center
    scale = max(float(np.max(np.abs(centered))), 1e-5)
    return centered / scale, center, scale


def contour_features(normalized: np.ndarray) -> np.ndarray:
    points = np.asarray(normalized, dtype=np.float32)
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    curvature = following - 2.0 * points + previous
    radial = np.linalg.norm(points, axis=1, keepdims=True)
    orientation = np.full((len(points), 1), 1.0 if signed_area(points) >= 0 else -1.0, dtype=np.float32)
    return np.concatenate([points, tangent, curvature, radial, orientation], axis=1).astype(np.float32)


def corner_targets(points: np.ndarray, threshold_degrees: float = 112.0) -> np.ndarray:
    source = np.asarray(points, dtype=np.float32)
    previous = np.roll(source, 1, axis=0) - source
    following = np.roll(source, -1, axis=0) - source
    denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    cosine = np.sum(previous * following, axis=1) / np.maximum(denominator, 1e-6)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return (angles <= float(threshold_degrees)).astype(np.float32)


def extract_contours_from_ink(ink: np.ndarray, minimum_area: float = 4.0) -> list[np.ndarray]:
    mask = (np.asarray(ink, dtype=np.float32) >= 0.5).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output: list[np.ndarray] = []
    for contour in contours:
        if abs(float(cv2.contourArea(contour))) < float(minimum_area):
            continue
        points = contour[:, 0, :].astype(np.float32)
        if len(points) >= 6:
            output.append(points)
    return output


def build_contour_cache(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    contour_cfg = cfg.get("fusion", {}).get("contour_polisher", {})
    output_dir = ensure_dir(work / "contour")
    cache_path = output_dir / "target_contours.npz"
    summary_path = output_dir / "summary.json"
    index_path = work / "dataset" / "index.csv"
    if not bool(contour_cfg.get("enabled", True)):
        summary = {"enabled": False, "reason": "disabled by configuration"}
        save_json(summary_path, summary)
        return summary
    if not index_path.is_file():
        raise FileNotFoundError(f"missing dataset index: {index_path}")
    fingerprint = {
        "dataset": sha256_file(index_path),
        "points": int(contour_cfg.get("points", 128)),
        "maximum_contours": int(contour_cfg.get("maximum_training_contours", 120000)),
    }
    fingerprint_path = output_dir / "fingerprint.json"
    if not force and cache_path.is_file() and summary_path.is_file() and fingerprint_path.is_file():
        try:
            if json.loads(fingerprint_path.read_text(encoding="utf-8")) == fingerprint:
                return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    point_count = max(32, int(contour_cfg.get("points", 128)))
    maximum = max(256, int(contour_cfg.get("maximum_training_contours", 120000)))
    rng = random.Random(int(cfg.get("training", {}).get("seed", 20260719)))
    rows = [row for row in read_csv(index_path) if row.get("mode") == "self" and row.get("split") == "train"]
    stored: list[np.ndarray] = []
    stored_corners: list[np.ndarray] = []
    stored_cps: list[int] = []
    seen = 0
    for row in rows:
        with Image.open(row["target_path"]) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        ink = 1.0 - gray
        for contour in extract_contours_from_ink(ink, minimum_area=max(3.0, ink.shape[0] * ink.shape[1] * 0.000015)):
            sampled = resample_closed_contour(contour, point_count)
            normalized, _, _ = normalize_contour(sampled)
            corners = corner_targets(normalized)
            seen += 1
            if len(stored) < maximum:
                stored.append(normalized.astype(np.float16))
                stored_corners.append(corners.astype(np.uint8))
                stored_cps.append(int(row["codepoint"]))
            else:
                index = rng.randrange(seen)
                if index < maximum:
                    stored[index] = normalized.astype(np.float16)
                    stored_corners[index] = corners.astype(np.uint8)
                    stored_cps[index] = int(row["codepoint"])
    if len(stored) < 64:
        summary = {"enabled": False, "reason": f"only {len(stored)} target contours"}
        save_json(summary_path, summary)
        return summary
    np.savez_compressed(
        cache_path,
        points=np.stack(stored).astype(np.float16),
        corners=np.stack(stored_corners).astype(np.uint8),
        codepoints=np.asarray(stored_cps, dtype=np.int32),
    )
    summary = {
        "enabled": True,
        "method": "target contour sequence denoising",
        "point_count": point_count,
        "stored_contours": len(stored),
        "seen_contours": seen,
        "cache_path": str(cache_path.resolve()),
    }
    save_json(summary_path, summary)
    save_json(fingerprint_path, fingerprint)
    return summary


class ContourDenoiseDataset(Dataset):
    def __init__(self, cache_path: str | Path, *, virtual_length: int = 0, seed: int = 20260719) -> None:
        with np.load(cache_path, allow_pickle=False) as data:
            self.points = np.asarray(data["points"], dtype=np.float32)
            self.corners = np.asarray(data["corners"], dtype=np.float32)
        self.virtual_length = max(len(self.points), int(virtual_length))
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.virtual_length

    @staticmethod
    def _smooth(points: np.ndarray, strength: float) -> np.ndarray:
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)
        return points * (1.0 - strength) + (previous + following) * (0.5 * strength)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(index) * 1009 + random.randrange(1 << 20))
        clean = self.points[index % len(self.points)].copy()
        noisy = clean.copy()
        noise_scale = rng.uniform(0.004, 0.045)
        noisy += rng.normal(0.0, noise_scale, size=noisy.shape).astype(np.float32)
        if rng.random() < 0.65:
            noisy = self._smooth(noisy, float(rng.uniform(0.05, 0.34)))
        if rng.random() < 0.30:
            angle = float(rng.uniform(-0.035, 0.035))
            matrix = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
            noisy = noisy @ matrix.T
        if rng.random() < 0.35:
            noisy *= float(rng.uniform(0.96, 1.04))
        features = contour_features(noisy)
        offsets = clean - noisy
        return {
            "features": torch.from_numpy(features),
            "offsets": torch.from_numpy(offsets.astype(np.float32)),
            "corners": torch.from_numpy(self.corners[index % len(self.points)][:, None].astype(np.float32)),
        }


@dataclass
class LoadedContourPolisher:
    model: ContourSequenceTransformer
    device: torch.device
    points: int

    @torch.no_grad()
    def polish(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sampled = resample_closed_contour(points, self.points)
        normalized, center, scale = normalize_contour(sampled)
        features = torch.from_numpy(contour_features(normalized))[None].to(self.device)
        output = self.model(features)
        polished = normalized + output["offset"][0].cpu().numpy()
        corners = torch.sigmoid(output["corner_logits"])[0, :, 0].cpu().numpy()
        return polished * scale + center, corners


def load_contour_polisher(checkpoint: str | Path, *, device: torch.device | str = "cpu") -> LoadedContourPolisher | None:
    path = Path(checkpoint)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    spec = payload.get("spec", {})
    model = ContourSequenceTransformer(
        points=int(spec.get("points", 128)),
        hidden=int(spec.get("hidden", 192)),
        layers=int(spec.get("layers", 6)),
        heads=int(spec.get("heads", 8)),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return LoadedContourPolisher(model=model, device=torch.device(device), points=model.points)
