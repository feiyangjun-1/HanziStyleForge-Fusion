from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .decomposition import Decomposition, decomposition_regions, load_decompositions
from .ids_data import IDSDataError, ensure_decomposition_data
from .proxy import read_ink, read_proxy
from .util import ensure_dir, read_csv, save_json, sha256_file


@dataclass(frozen=True)
class ComponentRegion:
    label: str
    rect: tuple[int, int, int, int]
    depth: int
    path: str


def labeled_component_regions(
    codepoint: int,
    mask: np.ndarray,
    decompositions: dict[int, Decomposition],
    *,
    maximum_depth: int = 3,
    maximum_regions: int = 48,
    minimum_region_size: int = 10,
) -> list[ComponentRegion]:
    records = decomposition_regions(
        int(codepoint),
        np.asarray(mask, dtype=np.float32),
        decompositions,
        maximum_depth=max(1, int(maximum_depth)),
        maximum_regions=max(1, int(maximum_regions)),
        minimum_region_size=max(2, int(minimum_region_size)),
    )
    return [
        ComponentRegion(label=label, rect=rect, depth=depth, path=path)
        for label, rect, depth, path, _ in records
    ]


def _crop_resize(array: np.ndarray, rect: tuple[int, int, int, int], size: int) -> np.ndarray:
    x0, y0, x1, y1 = rect
    crop = np.asarray(array)[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    return cv2.resize(crop.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def _descriptor(proxy_patch: np.ndarray, rect: tuple[int, int, int, int], full_shape: tuple[int, int], size: int) -> np.ndarray:
    if proxy_patch.ndim == 2:
        proxy_patch = proxy_patch[..., None]
    channels = min(4, proxy_patch.shape[-1])
    features = []
    for index in range(channels):
        features.append(cv2.resize(proxy_patch[..., index], (size, size), interpolation=cv2.INTER_AREA).reshape(-1))
    x0, y0, x1, y1 = rect
    h, w = full_shape
    geometry = np.asarray(
        [
            (x1 - x0) / max(1, w),
            (y1 - y0) / max(1, h),
            ((x0 + x1) * 0.5) / max(1, w),
            ((y0 + y1) * 0.5) / max(1, h),
            float(proxy_patch[..., 0].mean()),
            float(proxy_patch[..., 0].std()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([*features, geometry], axis=0).astype(np.float32)


def _label_key(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8", errors="replace")).hexdigest()[:16]


def build_component_atlas(
    cfg: dict[str, Any],
    *,
    dataset_csv: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    fusion_cfg = cfg.get("fusion", {})
    component_cfg = fusion_cfg.get("component_atlas", {})
    enabled = bool(component_cfg.get("enabled", True))
    output_dir = ensure_dir(work / "component_atlas")
    summary_path = output_dir / "summary.json"
    atlas_path = output_dir / "atlas.npz"
    labels_path = output_dir / "labels.json"
    if not enabled:
        summary = {"enabled": False, "reason": "disabled by configuration"}
        save_json(summary_path, summary)
        return summary

    dataset_path = Path(dataset_csv) if dataset_csv is not None else work / "dataset" / "index.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"missing dataset index: {dataset_path}")
    try:
        decomposition_path, ids_status = ensure_decomposition_data(component_cfg)
    except IDSDataError as exc:
        summary = {
            "enabled": False,
            "reason": str(exc),
            "data_source": "cjkvi/cjkvi-ids",
        }
        save_json(summary_path, summary)
        return summary
    if not decomposition_path.is_file():
        summary = {
            "enabled": False,
            "reason": f"missing optional CJKVI IDS file: {decomposition_path}",
            "data_source": "cjkvi/cjkvi-ids",
            "ids_status": ids_status,
        }
        save_json(summary_path, summary)
        return summary

    fingerprint = {
        "dataset": sha256_file(dataset_path),
        "decomposition": sha256_file(decomposition_path),
        "config": component_cfg,
    }
    fingerprint_path = output_dir / "fingerprint.json"
    if (
        not force and atlas_path.is_file() and labels_path.is_file() and summary_path.is_file()
        and fingerprint_path.is_file()
    ):
        try:
            if json.loads(fingerprint_path.read_text(encoding="utf-8")) == fingerprint:
                return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    decompositions = load_decompositions(
        decomposition_path,
        region_priority=component_cfg.get("region_priority", []),
        include_obsolete=bool(component_cfg.get("include_obsolete", False)),
    )
    rows = [row for row in read_csv(dataset_path) if row.get("mode") == "self"]
    rng = random.Random(int(cfg.get("training", {}).get("seed", 20260719)))
    patch_size = max(24, int(component_cfg.get("stored_patch_size", 96)))
    descriptor_size = max(4, int(component_cfg.get("descriptor_size", 8)))
    maximum_total = max(128, int(component_cfg.get("maximum_patches", 180000)))
    maximum_per_label = max(2, int(component_cfg.get("maximum_per_component", 48)))
    maximum_depth = max(1, int(component_cfg.get("maximum_depth", 3)))
    maximum_regions = max(4, int(component_cfg.get("maximum_regions_per_glyph", 40)))
    minimum_activity = float(component_cfg.get("minimum_activity", 0.006))

    records_by_label: dict[str, list[tuple[np.ndarray, np.ndarray, int, tuple[float, float]]]] = {}
    seen_by_label: dict[str, int] = {}
    seen_total = 0
    for row in rows:
        cp = int(row["codepoint"])
        item = decompositions.get(cp)
        if item is None:
            continue
        target = read_ink(row["target_path"])
        proxy = read_proxy(row["proxy_path"])
        regions = labeled_component_regions(
            cp,
            proxy[..., 0],
            decompositions,
            maximum_depth=maximum_depth,
            maximum_regions=maximum_regions,
        )
        for region in regions:
            x0, y0, x1, y1 = region.rect
            proxy_crop = proxy[y0:y1, x0:x1]
            if proxy_crop.size == 0 or float(proxy_crop[..., 1].mean()) < minimum_activity:
                continue
            target_patch = _crop_resize(target, region.rect, patch_size)
            base_patch = _crop_resize(proxy[..., 0], region.rect, patch_size)
            residual = np.clip(target_patch - base_patch, -1.0, 1.0)
            descriptor = _descriptor(proxy_crop, region.rect, proxy.shape[:2], descriptor_size)
            label = region.label
            seen_total += 1
            seen_by_label[label] = seen_by_label.get(label, 0) + 1
            bucket = records_by_label.setdefault(label, [])
            record = (descriptor, residual, cp, ((x0 + x1) / (2 * proxy.shape[1]), (y0 + y1) / (2 * proxy.shape[0])))
            if len(bucket) < maximum_per_label:
                bucket.append(record)
            else:
                replacement = rng.randrange(seen_by_label[label])
                if replacement < maximum_per_label:
                    bucket[replacement] = record

    # Keep the most frequently represented components within the global budget.
    labels = sorted(records_by_label, key=lambda label: (-len(records_by_label[label]), label))
    descriptors: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    label_ids: list[int] = []
    source_cps: list[int] = []
    source_xy: list[tuple[float, float]] = []
    label_table: list[str] = []
    label_ranges: dict[str, list[int]] = {}
    for label in labels:
        if len(descriptors) >= maximum_total:
            break
        start = len(descriptors)
        label_id = len(label_table)
        label_table.append(label)
        for descriptor, residual, cp, xy in records_by_label[label]:
            if len(descriptors) >= maximum_total:
                break
            descriptors.append(descriptor)
            residuals.append(np.rint(residual * 127.0).astype(np.int8))
            label_ids.append(label_id)
            source_cps.append(cp)
            source_xy.append(xy)
        label_ranges[label] = [start, len(descriptors)]

    if len(descriptors) < 32:
        summary = {"enabled": False, "reason": f"only {len(descriptors)} usable component patches"}
        save_json(summary_path, summary)
        return summary

    descriptor_array = np.stack(descriptors).astype(np.float32)
    mean = descriptor_array.mean(axis=0)
    std = np.maximum(descriptor_array.std(axis=0), 1e-4)
    normalized = ((descriptor_array - mean) / std).astype(np.float16)
    np.savez_compressed(
        atlas_path,
        descriptors=normalized,
        descriptor_mean=mean.astype(np.float32),
        descriptor_std=std.astype(np.float32),
        residuals=np.stack(residuals).astype(np.int8),
        label_ids=np.asarray(label_ids, dtype=np.int32),
        source_codepoints=np.asarray(source_cps, dtype=np.int32),
        source_xy=np.asarray(source_xy, dtype=np.float16),
    )
    labels_payload = {"labels": label_table, "ranges": label_ranges}
    labels_path.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "enabled": True,
        "method": "semantic component residual atlas with standard Unicode IDS layouts",
        "data_source": "cjkvi/cjkvi-ids",
        "ids_status": ids_status,
        "decomposition_record_count": len(decompositions),
        "component_label_count": len(label_table),
        "patch_count": len(descriptors),
        "seen_patch_count": seen_total,
        "source_glyph_count": len(rows),
        "atlas_path": str(atlas_path.resolve()),
        "labels_path": str(labels_path.resolve()),
    }
    save_json(summary_path, summary)
    save_json(fingerprint_path, fingerprint)
    return summary


@dataclass
class ComponentAtlas:
    descriptors: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    residuals: np.ndarray
    label_ids: np.ndarray
    source_codepoints: np.ndarray
    source_xy: np.ndarray
    labels: list[str]
    ranges: dict[str, list[int]]

    @classmethod
    def load(cls, atlas_path: str | Path, labels_path: str | Path) -> "ComponentAtlas":
        with np.load(atlas_path, allow_pickle=False) as data:
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            mean = np.asarray(data["descriptor_mean"], dtype=np.float32)
            std = np.asarray(data["descriptor_std"], dtype=np.float32)
            residuals = np.asarray(data["residuals"], dtype=np.int8)
            label_ids = np.asarray(data["label_ids"], dtype=np.int32)
            source_codepoints = np.asarray(data["source_codepoints"], dtype=np.int32)
            source_xy = np.asarray(data["source_xy"], dtype=np.float32)
        payload = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        return cls(
            descriptors=descriptors,
            mean=mean,
            std=std,
            residuals=residuals,
            label_ids=label_ids,
            source_codepoints=source_codepoints,
            source_xy=source_xy,
            labels=list(payload["labels"]),
            ranges={str(key): [int(v[0]), int(v[1])] for key, v in payload["ranges"].items()},
        )

    def query(self, label: str, descriptor: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        bounds = self.ranges.get(str(label))
        if bounds is None or bounds[1] <= bounds[0]:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        start, end = bounds
        query = ((np.asarray(descriptor, dtype=np.float32) - self.mean) / self.std).reshape(1, -1)
        candidates = self.descriptors[start:end]
        distances = np.mean((candidates - query) ** 2, axis=1)
        count = min(max(1, int(k)), len(distances))
        local = np.argpartition(distances, count - 1)[:count]
        local = local[np.argsort(distances[local])]
        return (local + start).astype(np.int64), distances[local].astype(np.float32)


def render_component_candidate(
    codepoint: int,
    ref_proxy: np.ndarray,
    atlas: ComponentAtlas,
    decompositions: dict[int, Decomposition],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    proxy = np.asarray(ref_proxy, dtype=np.float32)
    if proxy.ndim != 3 or proxy.shape[-1] < 4:
        raise ValueError("component candidate requires an HxWx4+ ref proxy")
    base = proxy[..., 0].copy()
    regions = labeled_component_regions(
        int(codepoint),
        base,
        decompositions,
        maximum_depth=max(1, int(config.get("maximum_depth", 3))),
        maximum_regions=max(4, int(config.get("maximum_regions_per_glyph", 40))),
    )
    if not regions:
        return base, {"matched_regions": 0, "total_regions": 0, "coverage": 0.0, "sources": []}
    descriptor_size = max(4, int(config.get("descriptor_size", 8)))
    k = max(1, int(config.get("knn", 5)))
    strength = float(config.get("strength", 0.92))
    temperature = max(1e-4, float(config.get("distance_temperature", 0.85)))
    residual_sum = np.zeros_like(base)
    weight_sum = np.zeros_like(base)
    matched = 0
    source_votes: dict[int, float] = {}
    for region in regions:
        x0, y0, x1, y1 = region.rect
        crop = proxy[y0:y1, x0:x1, :4]
        if crop.size == 0:
            continue
        descriptor = _descriptor(crop, region.rect, proxy.shape[:2], descriptor_size)
        indices, distances = atlas.query(region.label, descriptor, k=k)
        if len(indices) == 0:
            continue
        scale = max(float(np.median(distances)), 1e-5) * temperature
        weights = np.exp(-distances / scale).astype(np.float32)
        weights /= max(float(weights.sum()), 1e-6)
        residual = np.zeros(atlas.residuals.shape[1:], dtype=np.float32)
        for index, weight in zip(indices, weights):
            residual += atlas.residuals[int(index)].astype(np.float32) / 127.0 * float(weight)
            cp = int(atlas.source_codepoints[int(index)])
            source_votes[cp] = source_votes.get(cp, 0.0) + float(weight)
        residual = cv2.resize(residual, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        height, width = y1 - y0, x1 - x0
        wy = np.hanning(max(3, height))[:height] if height > 2 else np.ones(height)
        wx = np.hanning(max(3, width))[:width] if width > 2 else np.ones(width)
        window = np.outer(wy, wx).astype(np.float32)
        window = np.maximum(window, 0.12)
        activity = np.clip(cv2.GaussianBlur(crop[..., 1], (0, 0), max(0.8, min(height, width) / 22.0)), 0.0, 1.0)
        window *= np.maximum(activity, 0.20)
        residual_sum[y0:y1, x0:x1] += residual * window
        weight_sum[y0:y1, x0:x1] += window
        matched += 1
    valid = weight_sum > 1e-5
    canvas = np.zeros_like(base)
    canvas[valid] = residual_sum[valid] / weight_sum[valid]
    candidate = np.clip(base + strength * canvas, 0.0, 1.0)
    candidate = cv2.GaussianBlur(candidate, (0, 0), max(0.25, base.shape[0] / 1800.0))
    sources = [cp for cp, _ in sorted(source_votes.items(), key=lambda item: item[1], reverse=True)[:12]]
    return candidate.astype(np.float32), {
        "matched_regions": matched,
        "total_regions": len(regions),
        "coverage": matched / max(1, len(regions)),
        "sources": sources,
        "mean_abs_residual": float(np.mean(np.abs(canvas[valid]))) if np.any(valid) else 0.0,
    }
