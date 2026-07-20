from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import CHECKPOINT_FORMAT_VERSION
from .proxy import read_ink, read_proxy
from .util import ensure_dir, read_csv, save_json, sha256_file


def _window_layout(size: int, grid: int, window_ratio: float) -> list[tuple[int, int, int, int, float, float]]:
    window = max(16, min(size, int(round(size * float(window_ratio)))))
    half = window // 2
    margin = max(half, int(round(size * 0.12)))
    positions = np.linspace(margin, size - margin, max(2, int(grid)))
    result: list[tuple[int, int, int, int, float, float]] = []
    for cy in positions:
        for cx in positions:
            x0 = int(round(cx)) - half
            y0 = int(round(cy)) - half
            x0 = max(0, min(size - window, x0))
            y0 = max(0, min(size - window, y0))
            x1 = x0 + window
            y1 = y0 + window
            result.append((x0, y0, x1, y1, float(cx / size), float(cy / size)))
    return result


def _orientation_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    src = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    angle = np.mod(angle, np.pi)
    indices = np.floor(angle / np.pi * bins).astype(np.int32).clip(0, bins - 1)
    histogram = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        histogram[index] = float(magnitude[indices == index].sum())
    total = float(histogram.sum())
    return histogram / total if total > 1e-6 else histogram


def _descriptor(proxy_patch: np.ndarray, cx: float, cy: float, descriptor_size: int, position_weight: float) -> np.ndarray:
    patch = np.asarray(proxy_patch, dtype=np.float32).clip(0.0, 1.0)
    d = int(descriptor_size)
    resized = np.stack(
        [cv2.resize(patch[..., channel], (d, d), interpolation=cv2.INTER_AREA) for channel in range(4)],
        axis=-1,
    )
    features = [resized.reshape(-1)]
    skeleton = patch[..., 1]
    features.append(_orientation_histogram(skeleton, bins=8))
    h_projection = cv2.resize(skeleton.mean(axis=0)[None, :], (d, 1), interpolation=cv2.INTER_AREA).reshape(-1)
    v_projection = cv2.resize(skeleton.mean(axis=1)[:, None], (1, d), interpolation=cv2.INTER_AREA).reshape(-1)
    features.extend([h_projection, v_projection])
    features.append(
        np.asarray(
            [
                float(patch[..., 0].mean()),
                float(patch[..., 1].mean()),
                float(patch[..., 2].mean()),
                float(patch[..., 3].mean()),
                float(patch[..., 0].std()),
                float(patch[..., 1].std()),
            ],
            dtype=np.float32,
        )
    )
    pw = float(position_weight)
    features.append(np.asarray([cx * pw, cy * pw, cx * cx * pw, cy * cy * pw], dtype=np.float32))
    return np.concatenate(features).astype(np.float32)


def _hann_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(max(3, int(height))).astype(np.float32)
    wx = np.hanning(max(3, int(width))).astype(np.float32)
    window = np.outer(wy, wx)
    return np.maximum(window, 0.03).astype(np.float32)


def _encode_residual(residual: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(residual).clip(-1.0, 1.0) * 127.0).astype(np.int8)


def _decode_residual(residual: np.ndarray) -> np.ndarray:
    return np.asarray(residual, dtype=np.float32) / 127.0


def build_style_atlas(
    cfg: dict[str, Any],
    *,
    dataset_csv: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    retrieval_cfg = cfg.get("retrieval", {})
    if not bool(retrieval_cfg.get("enabled", True)):
        return {"enabled": False}
    work = Path(cfg["paths"]["work_dir"])
    dataset_path = Path(dataset_csv) if dataset_csv else work / "dataset" / "index.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training set not found: {dataset_path}")
    atlas_dir = ensure_dir(work / "retrieval")
    atlas_path = atlas_dir / "style_atlas.npz"
    summary_path = atlas_dir / "summary.json"
    fingerprint = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "dataset_sha256": sha256_file(dataset_path),
        "config": retrieval_cfg,
        "render_size": int(cfg["render"]["size"]),
    }
    fingerprint_path = atlas_dir / "fingerprint.json"
    if (
        not force
        and atlas_path.exists()
        and summary_path.exists()
        and fingerprint_path.exists()
        and json.loads(fingerprint_path.read_text(encoding="utf-8")) == fingerprint
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    rows = [row for row in read_csv(dataset_path) if row.get("mode") == "self"]
    if not rows:
        raise RuntimeError("No self-reconstruction samples are available; the local style atlas cannot be built.")
    rng = random.Random(int(cfg["training"].get("seed", 20260717)))
    maximum = max(64, int(retrieval_cfg.get("maximum_patches", 24000)))
    per_glyph = max(1, int(retrieval_cfg.get("patches_per_glyph", 2)))
    grid = max(2, int(retrieval_cfg.get("grid", 4)))
    window_ratio = float(retrieval_cfg.get("window_ratio", 0.38))
    descriptor_size = max(4, int(retrieval_cfg.get("descriptor_size", 6)))
    stored_patch_size = max(24, int(retrieval_cfg.get("stored_patch_size", 56)))
    position_weight = float(retrieval_cfg.get("position_weight", 1.35))
    minimum_activity = float(retrieval_cfg.get("minimum_activity", 0.012))

    descriptors: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    source_cps: list[int] = []
    source_xy: list[tuple[float, float]] = []
    seen = 0
    for row in rows:
        proxy = read_proxy(row["proxy_path"])
        target = read_ink(row["target_path"])
        if target.shape != proxy.shape[:2]:
            target = cv2.resize(target, (proxy.shape[1], proxy.shape[0]), interpolation=cv2.INTER_AREA)
        size = int(proxy.shape[0])
        windows = _window_layout(size, grid, window_ratio)
        ranked: list[tuple[float, tuple[int, int, int, int, float, float]]] = []
        for window in windows:
            x0, y0, x1, y1, _, _ = window
            activity = float(proxy[y0:y1, x0:x1, 1].mean())
            if activity >= minimum_activity:
                ranked.append((activity, window))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[int, int, int, int, float, float]] = []
        for _, window in ranked:
            if not selected:
                selected.append(window)
            else:
                _, _, _, _, cx, cy = window
                distance = min(math.hypot(cx - item[4], cy - item[5]) for item in selected)
                if distance >= 0.20 or len(ranked) <= per_glyph:
                    selected.append(window)
            if len(selected) >= per_glyph:
                break
        for x0, y0, x1, y1, cx, cy in selected:
            proxy_patch = proxy[y0:y1, x0:x1]
            target_patch = target[y0:y1, x0:x1]
            base_patch = proxy_patch[..., 0]
            descriptor = _descriptor(proxy_patch, cx, cy, descriptor_size, position_weight)
            residual = target_patch - base_patch
            residual = cv2.resize(residual, (stored_patch_size, stored_patch_size), interpolation=cv2.INTER_AREA)
            encoded = _encode_residual(residual)
            seen += 1
            if len(descriptors) < maximum:
                descriptors.append(descriptor)
                residuals.append(encoded)
                source_cps.append(int(row["codepoint"]))
                source_xy.append((cx, cy))
            else:
                index = rng.randrange(seen)
                if index < maximum:
                    descriptors[index] = descriptor
                    residuals[index] = encoded
                    source_cps[index] = int(row["codepoint"])
                    source_xy[index] = (cx, cy)

    if len(descriptors) < 32:
        raise RuntimeError(f"The local style atlas contains only {len(descriptors)} patches, which is insufficient.")
    descriptor_array = np.stack(descriptors).astype(np.float32)
    mean = descriptor_array.mean(axis=0)
    std = descriptor_array.std(axis=0)
    std = np.maximum(std, 1e-4)
    normalized = ((descriptor_array - mean) / std).astype(np.float32)
    residual_array = np.stack(residuals).astype(np.int8)
    np.savez_compressed(
        atlas_path,
        descriptors=normalized.astype(np.float16),
        descriptor_mean=mean.astype(np.float32),
        descriptor_std=std.astype(np.float32),
        residuals=residual_array,
        source_codepoints=np.asarray(source_cps, dtype=np.int32),
        source_xy=np.asarray(source_xy, dtype=np.float16),
    )
    summary = {
        "enabled": True,
        "atlas_path": str(atlas_path.resolve()),
        "patch_count": int(len(descriptors)),
        "seen_patch_count": int(seen),
        "descriptor_dimensions": int(normalized.shape[1]),
        "stored_patch_size": int(stored_patch_size),
        "source_glyph_count": int(len(rows)),
        "method": "local proxy nearest-neighbour residual atlas",
    }
    save_json(summary_path, summary)
    save_json(fingerprint_path, fingerprint)
    return summary


@dataclass
class StyleAtlas:
    descriptors: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    residuals: np.ndarray
    source_codepoints: np.ndarray
    source_xy: np.ndarray
    index: Any

    @classmethod
    def load(cls, path: str | Path, trees: int = 6) -> "StyleAtlas":
        with np.load(path, allow_pickle=False) as data:
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            mean = np.asarray(data["descriptor_mean"], dtype=np.float32)
            std = np.asarray(data["descriptor_std"], dtype=np.float32)
            residuals = np.asarray(data["residuals"], dtype=np.int8)
            source_codepoints = np.asarray(data["source_codepoints"], dtype=np.int32)
            source_xy = np.asarray(data["source_xy"], dtype=np.float32)
        params = dict(algorithm=1, trees=max(1, int(trees)))
        index = cv2.flann_Index(descriptors, params)
        return cls(descriptors, mean, std, residuals, source_codepoints, source_xy, index)

    def query(self, descriptor: np.ndarray, k: int, checks: int = 64) -> tuple[np.ndarray, np.ndarray]:
        normalized = ((np.asarray(descriptor, dtype=np.float32) - self.mean) / self.std).reshape(1, -1)
        count = min(max(1, int(k)), len(self.descriptors))
        indices, distances = self.index.knnSearch(normalized, count, params={"checks": int(checks)})
        return indices[0].astype(np.int64), distances[0].astype(np.float32)


def render_retrieval_candidate(
    proxy: np.ndarray,
    atlas: StyleAtlas,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    content = np.asarray(proxy, dtype=np.float32).clip(0.0, 1.0)
    # The atlas descriptor is intentionally based on the stable four-channel
    # proxy representation. Final inference expands proxies to ten channels for
    # the neural network, so slice here to keep atlas/query dimensions equal.
    if content.ndim != 3 or content.shape[-1] < 4:
        raise ValueError(f"Local retrieval requires at least four proxy channels; received {content.shape}.")
    content = content[..., :4]
    size = int(content.shape[0])
    grid = max(2, int(config.get("grid", 4)))
    window_ratio = float(config.get("window_ratio", 0.38))
    descriptor_size = max(4, int(config.get("descriptor_size", 6)))
    position_weight = float(config.get("position_weight", 1.35))
    minimum_activity = float(config.get("minimum_activity", 0.012))
    k = max(1, int(config.get("knn", 5)))
    checks = max(16, int(config.get("flann_checks", 64)))
    strength = float(config.get("strength", 0.88))
    temperature = max(1e-4, float(config.get("distance_temperature", 1.0)))

    residual_sum = np.zeros((size, size), dtype=np.float32)
    weight_sum = np.zeros((size, size), dtype=np.float32)
    queries = 0
    source_votes: dict[int, float] = {}
    for x0, y0, x1, y1, cx, cy in _window_layout(size, grid, window_ratio):
        patch = content[y0:y1, x0:x1]
        activity = float(patch[..., 1].mean())
        if activity < minimum_activity:
            continue
        descriptor = _descriptor(patch, cx, cy, descriptor_size, position_weight)
        indices, distances = atlas.query(descriptor, k=k, checks=checks)
        scale = max(float(np.median(distances)), 1e-4) * temperature
        weights = np.exp(-distances / scale).astype(np.float32)
        weights /= max(float(weights.sum()), 1e-6)
        retrieved = np.zeros_like(_decode_residual(atlas.residuals[int(indices[0])]))
        for index, weight in zip(indices, weights):
            retrieved += _decode_residual(atlas.residuals[int(index)]) * float(weight)
            cp = int(atlas.source_codepoints[int(index)])
            source_votes[cp] = source_votes.get(cp, 0.0) + float(weight) * activity
        retrieved = cv2.resize(retrieved, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        window = _hann_window(y1 - y0, x1 - x0) * max(0.10, activity)
        residual_sum[y0:y1, x0:x1] += retrieved * window
        weight_sum[y0:y1, x0:x1] += window
        queries += 1

    base = content[..., 0].copy()
    valid = weight_sum > 1e-5
    residual_canvas = np.zeros_like(base)
    residual_canvas[valid] = residual_sum[valid] / weight_sum[valid]
    candidate = np.clip(base + strength * residual_canvas, 0.0, 1.0)
    candidate = cv2.GaussianBlur(candidate, (0, 0), max(0.28, size / 1200.0)).clip(0.0, 1.0)
    top_sources = sorted(source_votes.items(), key=lambda item: item[1], reverse=True)[:8]
    metadata = {
        "query_count": int(queries),
        "top_source_codepoints": [int(item[0]) for item in top_sources],
        "mean_abs_residual": float(np.mean(np.abs(residual_canvas[valid]))) if np.any(valid) else 0.0,
    }
    return candidate.astype(np.float32), metadata
