from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image


_CONTOUR_POLISHER_CACHE: dict[str, object] = {}


def _load_optional_contour_polisher(config: dict[str, Any]):
    checkpoint = str(config.get("contour_polisher_checkpoint", "") or "").strip()
    if not checkpoint or not bool(config.get("use_contour_polisher", False)):
        return None
    path = Path(checkpoint)
    if not path.is_file():
        return None
    key = str(path.resolve())
    if key not in _CONTOUR_POLISHER_CACHE:
        from .contour_polish import load_contour_polisher
        _CONTOUR_POLISHER_CACHE[key] = load_contour_polisher(path, device="cpu")
    return _CONTOUR_POLISHER_CACHE[key]


def _contour_depth(index: int, hierarchy: np.ndarray) -> int:
    depth = 0
    parent = int(hierarchy[0, index, 3])
    while parent >= 0:
        depth += 1
        parent = int(hierarchy[0, parent, 3])
    return depth


def _signed_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _deduplicate(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for x, y in points:
        point = (float(x), float(y))
        if not result or abs(point[0] - result[-1][0]) > 0.01 or abs(point[1] - result[-1][1]) > 0.01:
            result.append(point)
    if len(result) > 1 and np.hypot(result[0][0] - result[-1][0], result[0][1] - result[-1][1]) < 0.01:
        result.pop()
    return result


def _limit_contour(contour: np.ndarray, epsilon: float, maximum_points: int) -> np.ndarray:
    current = max(0.1, float(epsilon))
    approx = cv2.approxPolyDP(contour, epsilon=current, closed=True)
    for _ in range(12):
        if len(approx) <= int(maximum_points):
            break
        current *= 1.28
        approx = cv2.approxPolyDP(contour, epsilon=current, closed=True)
    return approx


def _angle(previous: tuple[float, float], point: tuple[float, float], following: tuple[float, float]) -> float:
    a = np.asarray(previous, dtype=np.float64) - np.asarray(point, dtype=np.float64)
    b = np.asarray(following, dtype=np.float64) - np.asarray(point, dtype=np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-8:
        return 180.0
    cosine = float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _corner_flags(points: list[tuple[float, float]], threshold_degrees: float) -> list[bool]:
    if len(points) < 4:
        return [True] * len(points)
    flags: list[bool] = []
    for index, point in enumerate(points):
        angle = _angle(points[index - 1], point, points[(index + 1) % len(points)])
        flags.append(angle <= float(threshold_degrees))
    # Avoid isolated one-point smooth/corner flicker caused by raster staircases.
    cleaned = flags[:]
    for index in range(len(flags)):
        previous, current, following = flags[index - 1], flags[index], flags[(index + 1) % len(flags)]
        if previous == following and current != previous:
            cleaned[index] = previous
    return cleaned


def _quadratic_path_corner_aware(
    pen: TTGlyphPen,
    points: list[tuple[float, float]],
    corner_angle: float,
) -> None:
    """Write smooth quadratic segments while preserving deliberate corners."""

    if len(points) < 3:
        return
    corners = _corner_flags(points, corner_angle)
    if not any(corners):
        start = ((points[-1][0] + points[0][0]) / 2.0, (points[-1][1] + points[0][1]) / 2.0)
        pen.moveTo(start)
        for index, control in enumerate(points):
            following = points[(index + 1) % len(points)]
            end = ((control[0] + following[0]) / 2.0, (control[1] + following[1]) / 2.0)
            pen.qCurveTo(control, end)
        pen.closePath()
        return

    start_index = next(index for index, value in enumerate(corners) if value)
    ordered = points[start_index:] + points[:start_index]
    ordered_corners = corners[start_index:] + corners[:start_index]
    pen.moveTo(ordered[0])
    index = 1
    count = len(ordered)
    while index < count:
        point = ordered[index]
        if ordered_corners[index]:
            pen.lineTo(point)
            index += 1
            continue
        following_index = (index + 1) % count
        following = ordered[following_index]
        if following_index == 0 or ordered_corners[following_index]:
            pen.qCurveTo(point, following)
            index += 2
        else:
            midpoint = ((point[0] + following[0]) / 2.0, (point[1] + following[1]) / 2.0)
            pen.qCurveTo(point, midpoint)
            index += 1
    pen.closePath()


def _linear_path(pen: TTGlyphPen, points: list[tuple[float, float]]) -> None:
    if len(points) < 3:
        return
    pen.moveTo(points[0])
    for point in points[1:]:
        pen.lineTo(point)
    pen.closePath()


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((mask <= 0).astype(np.uint8), cv2.DIST_L2, 5)
    return inside - outside


def _component_hole_counts(mask: np.ndarray, minimum_area: int = 2) -> tuple[int, int]:
    source = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8)
    components = sum(1 for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area)
    inverse = (1 - source).astype(np.uint8)
    hcount, hlabels, hstats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    border = (
        set(int(v) for v in hlabels[0, :])
        | set(int(v) for v in hlabels[-1, :])
        | set(int(v) for v in hlabels[:, 0])
        | set(int(v) for v in hlabels[:, -1])
    )
    holes = sum(
        1
        for index in range(1, hcount)
        if index not in border and int(hstats[index, cv2.CC_STAT_AREA]) >= minimum_area
    )
    return components, holes


def _choose_sdf_mask(ink: np.ndarray, factor: int, sigma: float, levels: list[float]) -> np.ndarray:
    base = (ink >= 0.5).astype(np.uint8)
    target_components, target_holes = _component_hole_counts(base, minimum_area=2)
    sdf = _signed_distance(base)
    sdf = cv2.resize(sdf, (ink.shape[1] * factor, ink.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
    if sigma > 0:
        sdf = cv2.GaussianBlur(sdf, (0, 0), sigmaX=sigma, sigmaY=sigma)
    best: tuple[float, np.ndarray] | None = None
    target_ratio = float(base.mean())
    for level in levels:
        mask = (sdf >= float(level) * factor).astype(np.uint8)
        components, holes = _component_hole_counts(mask, minimum_area=max(2, factor * factor))
        ratio = float(mask.mean())
        score = 2.0 * abs(components - target_components) + 2.5 * abs(holes - target_holes) + abs(ratio - target_ratio)
        if best is None or score < best[0]:
            best = (score, mask)
    assert best is not None
    return best[1] * 255


def image_to_ttglyph(
    path: str | Path,
    *,
    upm: int,
    pad: int,
    y_bottom: float,
    y_top: float,
    config: dict[str, Any],
):
    gray = np.asarray(read_gray_u8(path), dtype=np.uint8)
    ink = 1.0 - gray.astype(np.float32) / 255.0
    mode = str(config.get("outline_mode", "sdf_quadratic")).lower()
    minimum_area = float(config.get("minimum_contour_area", 2.0))
    simplify = float(config.get("simplify", 0.85))
    curve_simplify = float(config.get("curve_simplify", 1.05))
    maximum_points = int(config.get("maximum_points_per_contour", 320))
    corner_angle = float(config.get("corner_angle_degrees", 112.0))

    if mode == "sdf_quadratic":
        factor = max(1, int(config.get("sdf_upsample", 4)))
        sigma = max(0.0, float(config.get("sdf_sigma", 0.70))) * factor
        levels = [float(value) for value in config.get("sdf_levels", [0.0, -0.18, 0.18])]
        contour_mask = _choose_sdf_mask(ink, factor, sigma, levels)
        epsilon = curve_simplify * factor
        area_scale = factor * factor
    else:
        factor = 1
        contour_mask = (ink >= 0.5).astype(np.uint8) * 255
        epsilon = simplify
        area_scale = 1

    contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pen = TTGlyphPen(None)
    if hierarchy is None:
        return pen.glyph()
    height, width = gray.shape
    usable_x = float(width - 2 * pad)
    usable_y = float(height - 2 * pad)
    if usable_x <= 0 or usable_y <= 0:
        raise ValueError("build.pad does not match the generated image size.")

    for index, contour in enumerate(contours):
        if abs(float(cv2.contourArea(contour))) < minimum_area * area_scale:
            continue
        approx = _limit_contour(contour, epsilon=epsilon, maximum_points=maximum_points)
        if len(approx) < 3:
            continue
        pixel_points = np.asarray(approx[:, 0, :], dtype=np.float32) / float(factor)
        polisher = _load_optional_contour_polisher(config)
        if polisher is not None and len(pixel_points) >= 6:
            try:
                polished, _ = polisher.polish(pixel_points)
                strength = float(np.clip(config.get("contour_polisher_strength", 0.58), 0.0, 1.0))
                # The Transformer is a denoiser, not a topology generator. Blend
                # it conservatively with the SDF contour so counters and outer
                # contour hierarchy remain controlled by the selected raster.
                if polished.shape == pixel_points.shape:
                    pixel_points = (1.0 - strength) * pixel_points + strength * polished
                else:
                    pixel_points = polished
            except Exception:
                # A single unusual contour must never prevent full-font build.
                pass
        raw: list[tuple[float, float]] = []
        for point in pixel_points:
            px = float(point[0])
            py = float(point[1])
            x = (px - pad) / usable_x * upm
            y = y_top - (py - pad) / usable_y * (y_top - y_bottom)
            raw.append((x, y))
        points = _deduplicate(raw)
        if len(points) < 3:
            continue
        is_hole = _contour_depth(index, hierarchy) % 2 == 1
        should_be_ccw = is_hole
        if (_signed_area(points) > 0) != should_be_ccw:
            points.reverse()
        if mode in {"sdf_quadratic", "quadratic_smooth"}:
            _quadratic_path_corner_aware(pen, points, corner_angle=corner_angle)
        else:
            _linear_path(pen, points)
    return pen.glyph()
from .image_cache import read_gray_u8
