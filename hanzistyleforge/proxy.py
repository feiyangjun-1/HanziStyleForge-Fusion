from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .util import atomic_save_pil


def gray_to_ink(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size and float(arr.max()) > 1.5:
        arr = arr / 255.0
    return (1.0 - arr).clip(0.0, 1.0)


def ink_to_gray(ink: np.ndarray) -> np.ndarray:
    arr = np.asarray(ink, dtype=np.float32).clip(0.0, 1.0)
    return np.rint((1.0 - arr) * 255.0).astype(np.uint8)


def read_ink(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return gray_to_ink(np.asarray(image.convert("L"), dtype=np.uint8))


def save_ink(path: str | Path, ink: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_pil(Image.fromarray(ink_to_gray(ink), mode="L"), p, format="PNG")


def binary(ink: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(ink, dtype=np.float32) >= float(threshold)).astype(np.uint8)


def ink_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def remove_small_components(mask: np.ndarray, minimum_area: int = 3) -> np.ndarray:
    src = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    result = np.zeros_like(src)
    for index in range(1, n):
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(minimum_area):
            result[labels == index] = 1
    return result


def _zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    src = (np.asarray(mask) > 0).astype(np.uint8)
    bbox = ink_bbox(src)
    if bbox is None:
        return src
    x0, y0, x1, y1 = bbox
    margin = 2
    xa, ya = max(0, x0 - margin), max(0, y0 - margin)
    xb, yb = min(src.shape[1], x1 + margin), min(src.shape[0], y1 + margin)
    image = src[ya:yb, xa:xb].copy()

    def neighbours(arr: np.ndarray):
        p = np.pad(arr, 1, mode="constant")
        p2 = p[:-2, 1:-1]
        p3 = p[:-2, 2:]
        p4 = p[1:-1, 2:]
        p5 = p[2:, 2:]
        p6 = p[2:, 1:-1]
        p7 = p[2:, :-2]
        p8 = p[1:-1, :-2]
        p9 = p[:-2, :-2]
        return p2, p3, p4, p5, p6, p7, p8, p9

    for _ in range(max(image.shape) + 8):
        changed = False
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbours(image)
        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        )
        remove = (
            (image == 1)
            & (b >= 2)
            & (b <= 6)
            & (a == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        if np.any(remove):
            image[remove] = 0
            changed = True

        p2, p3, p4, p5, p6, p7, p8, p9 = neighbours(image)
        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        )
        remove = (
            (image == 1)
            & (b >= 2)
            & (b <= 6)
            & (a == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        if np.any(remove):
            image[remove] = 0
            changed = True
        if not changed:
            break

    result = np.zeros_like(src)
    result[ya:yb, xa:xb] = image
    return result


def thin_binary(mask: np.ndarray) -> np.ndarray:
    src = ((np.asarray(mask) > 0).astype(np.uint8) * 255)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        try:
            return (ximgproc.thinning(src) > 0).astype(np.uint8)
        except Exception:
            pass
    return _zhang_suen_thinning(src)


def ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def normalize_bbox(
    ink_or_mask: np.ndarray,
    size: int = 128,
    margin: int = 8,
    threshold: float = 0.5,
) -> np.ndarray:
    src = binary(ink_or_mask, threshold)
    bbox = ink_bbox(src)
    canvas = np.zeros((size, size), dtype=np.float32)
    if bbox is None:
        return canvas
    x0, y0, x1, y1 = bbox
    crop = src[y0:y1, x0:x1]
    h, w = crop.shape
    usable = max(1, int(size) - 2 * int(margin))
    scale = min(usable / max(1, w), usable / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = (resized >= 0.35).astype(np.float32)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def component_hole_counts(mask: np.ndarray, minimum_area: int = 3) -> tuple[int, int]:
    src = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    components = sum(
        1
        for index in range(1, n)
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(minimum_area)
    )

    inverse = (1 - src).astype(np.uint8)
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    border_labels = (
        set(int(x) for x in labels2[0, :])
        | set(int(x) for x in labels2[-1, :])
        | set(int(x) for x in labels2[:, 0])
        | set(int(x) for x in labels2[:, -1])
    )
    holes = sum(
        1
        for index in range(1, n2)
        if index not in border_labels
        and int(stats2[index, cv2.CC_STAT_AREA]) >= int(minimum_area)
    )
    return int(components), int(holes)


def border_ink_ratio(mask_or_ink: np.ndarray, border: int = 3) -> float:
    arr = np.asarray(mask_or_ink, dtype=np.float32)
    border = max(1, min(int(border), min(arr.shape[-2:]) // 4))
    values = np.concatenate(
        [
            arr[:border, :].ravel(),
            arr[-border:, :].ravel(),
            arr[:, :border].ravel(),
            arr[:, -border:].ravel(),
        ]
    )
    return float(values.mean()) if values.size else 0.0


def estimate_stroke_radius(mask_or_ink: np.ndarray, threshold: float = 0.5) -> float:
    mask = binary(mask_or_ink, threshold)
    if not mask.any():
        return 0.0
    skeleton = thin_binary(mask)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    values = distance[skeleton > 0]
    if values.size < 4:
        values = distance[mask > 0]
    return float(np.median(values)) if values.size else 0.0


def _proxy_at_small_size(
    ink: np.ndarray,
    size: int,
    threshold: float,
    canonical_radius: int,
    distance_clip: float,
) -> np.ndarray:
    resized = cv2.resize(
        np.asarray(ink, dtype=np.float32),
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    mask = binary(resized, threshold)
    min_area = max(2, int(round(size * size * 0.000035)))
    mask = remove_small_components(mask, min_area)
    if not mask.any():
        return np.zeros((size, size, 4), dtype=np.float32)

    skeleton = thin_binary(mask)
    canonical = cv2.dilate(
        skeleton,
        ellipse_kernel(canonical_radius),
        iterations=1,
    ).astype(np.float32)
    canonical = np.maximum(canonical, skeleton.astype(np.float32))

    skeleton_blur = cv2.GaussianBlur(
        skeleton.astype(np.float32),
        (0, 0),
        sigmaX=max(0.65, size / 180.0),
    )
    max_value = float(skeleton_blur.max())
    if max_value > 0:
        skeleton_blur /= max_value

    inside = cv2.distanceTransform((canonical > 0).astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((canonical <= 0).astype(np.uint8), cv2.DIST_L2, 5)
    signed = np.clip((inside - outside) / max(float(distance_clip), 1.0), -1.0, 1.0)
    signed = (signed + 1.0) * 0.5

    coarse_size = max(12, min(32, size // 5))
    coarse = cv2.resize(mask.astype(np.float32), (coarse_size, coarse_size), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_LINEAR)
    coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=max(1.0, size / 80.0))
    coarse = coarse.clip(0.0, 1.0)

    return np.stack([canonical, skeleton_blur, signed, coarse], axis=-1).astype(np.float32)


def make_content_proxy(
    ink: np.ndarray,
    output_size: int = 384,
    skeleton_size: int = 160,
    threshold: float = 0.5,
    canonical_radius_ratio: float = 0.0105,
    distance_clip_ratio: float = 0.075,
) -> np.ndarray:
    """Build a style-reduced four-channel content representation.

    Channels are: fixed-width canonical stroke, blurred skeleton, signed
    distance field and coarse occupancy.  The representation preserves the
    character structure but removes most stroke weight, serif and terminal
    details, allowing every valid target glyph to be used for self-training.
    """

    skeleton_size = max(64, min(int(skeleton_size), int(output_size)))
    radius = max(1, int(round(skeleton_size * float(canonical_radius_ratio))))
    distance_clip = max(4.0, skeleton_size * float(distance_clip_ratio))
    small = _proxy_at_small_size(
        ink,
        size=skeleton_size,
        threshold=float(threshold),
        canonical_radius=radius,
        distance_clip=distance_clip,
    )
    if skeleton_size == output_size:
        proxy = small
    else:
        channels = [
            cv2.resize(small[..., i], (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            for i in range(4)
        ]
        proxy = np.stack(channels, axis=-1)
    return np.rint(proxy.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def save_proxy(path: str | Path, proxy_rgba: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_pil(
        Image.fromarray(np.asarray(proxy_rgba, dtype=np.uint8), mode="RGBA"),
        p,
        format="PNG",
    )


def read_proxy(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0


def proxy_base_ink(proxy_rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proxy_rgba, dtype=np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return arr[..., 0].clip(0.0, 1.0)


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    sa = thin_binary(a)
    sb = thin_binary(b)
    if not sa.any() and not sb.any():
        return 0.0
    if not sa.any() or not sb.any():
        return 1.0
    da = cv2.distanceTransform((1 - sa).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((1 - sb).astype(np.uint8), cv2.DIST_L2, 3)
    value = (float(da[sb > 0].mean()) + float(db[sa > 0].mean())) / 2.0
    return value / max(a.shape)


def _dice_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0).astype(np.float32)
    bb = (b > 0).astype(np.float32)
    denom = float(aa.sum() + bb.sum())
    if denom <= 0:
        return 0.0
    return float(1.0 - (2.0 * (aa * bb).sum() + 1.0) / (denom + 1.0))


def proxy_structure_metrics(
    proxy_a: np.ndarray,
    proxy_b: np.ndarray,
    analysis_size: int = 128,
) -> dict[str, float | int]:
    a = normalize_bbox(proxy_base_ink(proxy_a), size=analysis_size, margin=8, threshold=0.35)
    b = normalize_bbox(proxy_base_ink(proxy_b), size=analysis_size, margin=8, threshold=0.35)
    ma = (a > 0.35).astype(np.uint8)
    mb = (b > 0.35).astype(np.uint8)

    chamfer = _symmetric_chamfer(ma, mb)
    dice_distance = _dice_distance(ma, mb)

    grid_size = 16
    ga = cv2.resize(ma.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(mb.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    grid_distance = float(np.mean(np.abs(ga - gb)))

    row_a = ma.mean(axis=1)
    row_b = mb.mean(axis=1)
    col_a = ma.mean(axis=0)
    col_b = mb.mean(axis=0)
    projection_distance = float((np.mean(np.abs(row_a - row_b)) + np.mean(np.abs(col_a - col_b))) / 2.0)

    components_a, holes_a = component_hole_counts(ma, minimum_area=max(2, analysis_size // 64))
    components_b, holes_b = component_hole_counts(mb, minimum_area=max(2, analysis_size // 64))
    component_delta = abs(components_a - components_b)
    hole_delta = abs(holes_a - holes_b)
    topology_penalty = min(0.30, 0.055 * component_delta + 0.040 * hole_delta)

    score = (
        0.42 * chamfer
        + 0.25 * dice_distance
        + 0.19 * grid_distance
        + 0.14 * projection_distance
        + topology_penalty
    )
    return {
        "structure_score": float(score),
        "chamfer": float(chamfer),
        "dice_distance": float(dice_distance),
        "grid_distance": float(grid_distance),
        "projection_distance": float(projection_distance),
        "components_a": int(components_a),
        "components_b": int(components_b),
        "component_delta": int(component_delta),
        "holes_a": int(holes_a),
        "holes_b": int(holes_b),
        "hole_delta": int(hole_delta),
    }


def proxy_structure_score(
    proxy_a: np.ndarray,
    proxy_b: np.ndarray,
    analysis_size: int = 128,
) -> float:
    return float(proxy_structure_metrics(proxy_a, proxy_b, analysis_size)["structure_score"])


def proxy_from_binary_for_metrics(mask: np.ndarray) -> np.ndarray:
    src = (np.asarray(mask) > 0).astype(np.uint8)
    size = src.shape[0]
    skeleton = thin_binary(src)
    canonical = cv2.dilate(skeleton, ellipse_kernel(max(1, size // 96))).astype(np.float32)
    blur = cv2.GaussianBlur(skeleton.astype(np.float32), (0, 0), 0.8)
    if blur.max(initial=0.0) > 0:
        blur /= float(blur.max())
    signed = np.full_like(canonical, 0.5, dtype=np.float32)
    coarse = cv2.resize(src.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.stack([canonical, blur, signed, coarse], axis=-1).clip(0.0, 1.0)


def synthetic_style_variant(
    base_proxy: np.ndarray,
    rng: random.Random,
    analysis_size: int = 128,
) -> np.ndarray:
    base = normalize_bbox(proxy_base_ink(base_proxy), analysis_size, margin=9, threshold=0.35)
    mask = (base > 0.35).astype(np.uint8)
    operation = rng.choice([0, 0, 0, 0, 1])
    if operation > 0:
        mask = cv2.dilate(mask, ellipse_kernel(operation), iterations=1)
    elif operation < 0:
        mask = cv2.erode(mask, ellipse_kernel(abs(operation)), iterations=1)

    scale = rng.uniform(0.985, 1.015)
    dx = rng.uniform(-1.0, 1.0)
    dy = rng.uniform(-1.0, 1.0)
    center = (analysis_size / 2.0, analysis_size / 2.0)
    matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    mask = cv2.warpAffine(mask, matrix, (analysis_size, analysis_size), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), rng.uniform(0.25, 0.55))
        mask = (soft >= rng.uniform(0.45, 0.55)).astype(np.uint8)
    return proxy_from_binary_for_metrics(mask)


def calibrate_same_structure_thresholds(
    proxies: list[np.ndarray],
    seed: int = 20260717,
    analysis_size: int = 128,
) -> dict[str, float]:
    rng = random.Random(int(seed))
    values: list[float] = []
    for proxy in proxies:
        for _ in range(2):
            variant = synthetic_style_variant(proxy, rng, analysis_size=analysis_size)
            values.append(proxy_structure_score(proxy, variant, analysis_size=analysis_size))
    if not values:
        return {
            "very_strict": 0.055,
            "keep": 0.090,
            "sensitive_keep": 0.070,
            "pair": 0.060,
            "uncertain": 0.150,
        }
    arr = np.asarray(values, dtype=np.float64)
    p90 = float(np.quantile(arr, 0.90))
    p97 = float(np.quantile(arr, 0.97))
    p995 = float(np.quantile(arr, 0.995))
    return {
        "very_strict": max(0.025, p90 * 1.05),
        "pair": max(0.035, p90 * 1.12),
        "sensitive_keep": max(0.045, p97 * 1.08),
        "keep": max(0.060, p995 * 1.18),
        "uncertain": max(0.105, p995 * 1.90),
        "calibration_p90": p90,
        "calibration_p97": p97,
        "calibration_p995": p995,
    }



def calibrate_observed_structure_thresholds(
    scores: list[float],
    synthetic: dict[str, float] | None = None,
) -> dict[str, float]:
    """Infer the same-structure score cluster without using a fixed quantile.

    A two-component one-dimensional Gaussian mixture is fitted to all existing
    target-vs-ref scores.  Regional differences normally form the upper
    component while normal style/proportion differences form the lower one.
    The thresholds are only used for candidate selection/reporting; they never
    decide whether a glyph can teach style.
    """

    if not scores:
        return dict(synthetic or {
            "very_strict": 0.18, "pair": 0.23, "sensitive_keep": 0.27,
            "keep": 0.31, "uncertain": 0.39,
        })
    x = np.asarray(scores, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 20:
        q = np.quantile(x, [0.20, 0.40, 0.60, 0.82, 0.95])
        return {
            "very_strict": float(q[0]),
            "pair": float(q[1]),
            "sensitive_keep": float(q[2]),
            "keep": float(q[3]),
            "uncertain": float(q[4]),
            "mixture_used": 0.0,
        }

    mean1, mean2 = float(np.quantile(x, 0.30)), float(np.quantile(x, 0.78))
    sigma = max(float(np.std(x)), 0.015)
    sigma1 = sigma2 = sigma
    weight1, weight2 = 0.70, 0.30
    for _ in range(80):
        p1 = weight1 / sigma1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2)
        p2 = weight2 / sigma2 * np.exp(-0.5 * ((x - mean2) / sigma2) ** 2)
        denom = p1 + p2 + 1e-12
        r1 = p1 / denom
        r2 = p2 / denom
        sum1, sum2 = max(float(r1.sum()), 1e-6), max(float(r2.sum()), 1e-6)
        new_mean1 = float((r1 * x).sum() / sum1)
        new_mean2 = float((r2 * x).sum() / sum2)
        new_sigma1 = max(0.008, float(np.sqrt((r1 * (x - new_mean1) ** 2).sum() / sum1)))
        new_sigma2 = max(0.008, float(np.sqrt((r2 * (x - new_mean2) ** 2).sum() / sum2)))
        mean1, mean2 = new_mean1, new_mean2
        sigma1, sigma2 = new_sigma1, new_sigma2
        weight1, weight2 = sum1 / x.size, sum2 / x.size
    if mean1 > mean2:
        mean1, mean2 = mean2, mean1
        sigma1, sigma2 = sigma2, sigma1
        weight1, weight2 = weight2, weight1

    separation = (mean2 - mean1) / max(math.sqrt((sigma1**2 + sigma2**2) / 2.0), 1e-6)
    q20, q35, q60, q78, q92 = [float(v) for v in np.quantile(x, [0.20, 0.35, 0.60, 0.78, 0.92])]
    if separation < 0.65 or weight1 < 0.12 or weight2 < 0.04:
        pair = q35
        sensitive = q60
        keep = q78
        uncertain = q92
        mixture_used = 0.0
    else:
        pair = min(q60, mean1 + 0.55 * sigma1)
        sensitive = min(q78, mean1 + 1.45 * sigma1)
        keep = min(q92, mean1 + 2.25 * sigma1)
        midpoint = (mean1 + mean2) / 2.0
        uncertain = max(keep + 0.015, min(q92, midpoint + 0.55 * sigma2))
        mixture_used = 1.0

    very_strict = min(pair, q20)
    # Enforce a useful ordering and broad safety limits. These limits are much
    # broad enough for candidate confidence calibration; they never classify the target font by regional compliance.
    pair = max(very_strict + 0.005, pair)
    sensitive = max(pair + 0.010, sensitive)
    keep = max(sensitive + 0.012, keep)
    uncertain = max(keep + 0.018, uncertain)
    return {
        "very_strict": float(very_strict),
        "pair": float(pair),
        "sensitive_keep": float(sensitive),
        "keep": float(keep),
        "uncertain": float(uncertain),
        "mixture_used": mixture_used,
        "lower_mean": float(mean1),
        "lower_sigma": float(sigma1),
        "upper_mean": float(mean2),
        "upper_sigma": float(sigma2),
        "lower_weight": float(weight1),
        "upper_weight": float(weight2),
        "separation": float(separation),
        "synthetic_keep": float((synthetic or {}).get("keep", 0.0)),
    }

def glyph_quality_metrics(ink: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    mask = binary(ink, threshold)
    bbox = ink_bbox(mask)
    components, holes = component_hole_counts(mask, minimum_area=max(2, mask.shape[0] // 128))
    if bbox is None:
        width_ratio = height_ratio = center_x = center_y = 0.0
    else:
        x0, y0, x1, y1 = bbox
        h, w = mask.shape
        width_ratio = (x1 - x0) / max(1, w)
        height_ratio = (y1 - y0) / max(1, h)
        center_x = ((x0 + x1) / 2.0) / max(1, w)
        center_y = ((y0 + y1) / 2.0) / max(1, h)
    return {
        "ink_ratio": float(mask.mean()),
        "border_ink": border_ink_ratio(mask, border=max(2, mask.shape[0] // 128)),
        "components": int(components),
        "holes": int(holes),
        "bbox_width_ratio": float(width_ratio),
        "bbox_height_ratio": float(height_ratio),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "stroke_radius": estimate_stroke_radius(mask),
    }


def robust_location_scale(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    sigma = max(1e-6, 1.4826 * mad)
    return median, sigma


def style_distance(metrics: dict[str, float | int], profile: dict[str, Any]) -> float:
    keys = (
        "ink_ratio",
        "bbox_width_ratio",
        "bbox_height_ratio",
        "center_x",
        "center_y",
        "stroke_radius",
    )
    weights = {
        "ink_ratio": 0.24,
        "bbox_width_ratio": 0.14,
        "bbox_height_ratio": 0.14,
        "center_x": 0.10,
        "center_y": 0.10,
        "stroke_radius": 0.28,
    }
    total = 0.0
    for key in keys:
        info = profile.get(key, {})
        median = float(info.get("median", 0.0))
        sigma = max(float(info.get("sigma", 1.0)), 1e-5)
        z = abs(float(metrics.get(key, 0.0)) - median) / sigma
        total += weights[key] * min(z, 8.0) / 8.0
    return float(total)


def diff_visual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32).clip(0, 1)
    bb = np.asarray(b, dtype=np.float32).clip(0, 1)
    h, w = aa.shape
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    only_a = (aa >= 0.5) & (bb < 0.5)
    only_b = (bb >= 0.5) & (aa < 0.5)
    both = (aa >= 0.5) & (bb >= 0.5)
    image[both] = (35, 35, 35)
    image[only_a] = (220, 55, 55)
    image[only_b] = (40, 95, 220)
    return image


def affine_mask(
    mask: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> np.ndarray:
    src = np.asarray(mask, dtype=np.float32)
    h, w = src.shape
    matrix = np.array(
        [
            [float(scale_x), 0.0, (1.0 - float(scale_x)) * w / 2.0 + float(shift_x)],
            [0.0, float(scale_y), (1.0 - float(scale_y)) * h / 2.0 + float(shift_y)],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(src, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0).clip(0.0, 1.0)


def make_reference_fallbacks(
    reference_ink: np.ndarray,
    style_profile: dict[str, Any],
    threshold: float = 0.5,
) -> list[tuple[str, np.ndarray]]:
    mask = binary(reference_ink, threshold)
    if not mask.any():
        return [("reference", mask.astype(np.float32))]

    target_radius = float(style_profile.get("stroke_radius", {}).get("median", 2.5))
    current_radius = max(0.5, estimate_stroke_radius(mask))
    delta = int(round(target_radius - current_radius))
    adjusted = mask.copy()
    if delta > 0:
        adjusted = cv2.dilate(adjusted, ellipse_kernel(min(delta, 4)), iterations=1)
    elif delta < 0:
        adjusted = cv2.erode(adjusted, ellipse_kernel(min(abs(delta), 3)), iterations=1)

    profile_w = float(style_profile.get("bbox_width_ratio", {}).get("median", 0.78))
    profile_h = float(style_profile.get("bbox_height_ratio", {}).get("median", 0.78))
    q = glyph_quality_metrics(adjusted)
    sx = np.clip(profile_w / max(float(q["bbox_width_ratio"]), 1e-3), 0.90, 1.10)
    sy = np.clip(profile_h / max(float(q["bbox_height_ratio"]), 1e-3), 0.90, 1.10)
    cx = float(style_profile.get("center_x", {}).get("median", 0.50))
    cy = float(style_profile.get("center_y", {}).get("median", 0.50))
    h, w = adjusted.shape
    shift_x = (cx - float(q["center_x"])) * w
    shift_y = (cy - float(q["center_y"])) * h
    adjusted = affine_mask(adjusted, float(sx), float(sy), shift_x, shift_y)

    skeleton = thin_binary(mask)
    radius = max(1, min(10, int(round(target_radius))))
    uniform = cv2.dilate(skeleton, ellipse_kernel(radius), iterations=1).astype(np.float32)
    uniform = affine_mask(uniform, float(sx), float(sy), shift_x, shift_y)

    rounded = cv2.morphologyEx(
        (adjusted >= 0.5).astype(np.uint8),
        cv2.MORPH_CLOSE,
        ellipse_kernel(1),
    ).astype(np.float32)
    return [
        # Keep an untouched reference candidate as the final structural
        # safety net. It is not the preferred style result, but it guarantees
        # that the automatic workflow can preserve the ref topology when all
        # stylized candidates fail validation.
        ("reference_raw", mask.astype(np.float32)),
        ("reference_adjusted", adjusted.astype(np.float32)),
        ("uniform_round", uniform.astype(np.float32)),
        ("reference_rounded", rounded.astype(np.float32)),
    ]
