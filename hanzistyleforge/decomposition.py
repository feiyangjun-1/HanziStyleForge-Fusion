from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


_RECORD = re.compile(r"^([^:]+):([^()]*)\((.*)\)\s*$")


@dataclass(frozen=True)
class Decomposition:
    codepoint: int
    kind: str
    components: tuple[str, ...]


def _token_codepoint(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        value = int(token)
        return value if value >= 0x3400 else None
    if len(token) == 1:
        return ord(token)
    return None


def load_decompositions(path: str | Path) -> dict[int, Decomposition]:
    p = Path(path)
    if not p.is_file():
        return {}
    result: dict[int, Decomposition] = {}
    with p.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _RECORD.match(line)
            if not match:
                continue
            head, kind, body = match.groups()
            cp = _token_codepoint(head)
            if cp is None:
                continue
            components = tuple(part.strip() for part in body.split(",") if part.strip())
            result[cp] = Decomposition(cp, kind.strip().lower(), components)
    return result


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask) > 0.2)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _projection_splits(mask: np.ndarray, count: int, axis: int) -> list[int]:
    if count <= 1:
        return []
    x0, y0, x1, y1 = _bbox(mask)
    crop = mask[y0:y1, x0:x1]
    projection = crop.sum(axis=0 if axis == 1 else 1).astype(np.float32)
    length = len(projection)
    if length < count * 4:
        return [int(round((i + 1) * length / count)) for i in range(count - 1)]
    smooth = cv2.GaussianBlur(projection.reshape(1, -1), (0, 0), max(1.0, length / 90.0)).ravel()
    splits: list[int] = []
    for i in range(1, count):
        expected = i * length / count
        radius = max(3, int(length / count * 0.28))
        lo = max(2, int(expected - radius))
        hi = min(length - 2, int(expected + radius))
        if hi <= lo:
            position = int(round(expected))
        else:
            position = lo + int(np.argmin(smooth[lo:hi]))
        splits.append(position)
    offset = x0 if axis == 1 else y0
    return [value + offset for value in splits]


def _feather(rect: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return rect.astype(np.float32)
    return cv2.GaussianBlur(rect.astype(np.float32), (0, 0), sigma).clip(0.0, 1.0)


def _region_splits(
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    count: int,
    axis: int,
) -> list[int]:
    x0, y0, x1, y1 = rect
    crop = np.asarray(mask)[y0:y1, x0:x1]
    if crop.size == 0 or count <= 1:
        return []
    projection = crop.sum(axis=0 if axis == 1 else 1).astype(np.float32)
    length = len(projection)
    if length < count * 5:
        local = [int(round(i * length / count)) for i in range(1, count)]
    else:
        smooth = cv2.GaussianBlur(
            projection.reshape(1, -1), (0, 0), max(1.0, length / 90.0)
        ).ravel()
        local = []
        for i in range(1, count):
            expected = i * length / count
            radius = max(3, int(length / count * 0.30))
            lo = max(2, int(expected - radius))
            hi = min(length - 2, int(expected + radius))
            position = int(round(expected)) if hi <= lo else lo + int(np.argmin(smooth[lo:hi]))
            local.append(position)
    offset = x0 if axis == 1 else y0
    return [offset + value for value in local]


def _rect_zone(
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    sigma: float,
) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = rect
    zone = np.zeros((h, w), dtype=np.float32)
    zone[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1.0
    return _feather(zone, sigma)


def _recursive_layout_zones(
    codepoint: int,
    mask: np.ndarray,
    decompositions: dict[int, Decomposition],
    rect: tuple[int, int, int, int],
    zones: list[np.ndarray],
    *,
    depth: int,
    maximum_depth: int,
    maximum_zones: int,
    sigma: float,
) -> None:
    if depth >= maximum_depth or len(zones) >= maximum_zones:
        return
    item = decompositions.get(int(codepoint))
    if item is None:
        return
    x0, y0, x1, y1 = rect
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    kind = item.kind
    count = len(item.components)
    child_rects: list[tuple[int, int, int, int]] = []
    if kind.startswith("a") and 2 <= count <= 5:
        splits = [x0, *_region_splits(mask, rect, count, axis=1), x1]
        child_rects = [(left, y0, right, y1) for left, right in zip(splits[:-1], splits[1:])]
    elif kind.startswith("d") and 2 <= count <= 5:
        splits = [y0, *_region_splits(mask, rect, count, axis=0), y1]
        child_rects = [(x0, top, x1, bottom) for top, bottom in zip(splits[:-1], splits[1:])]
    elif (kind.startswith("s") or kind.startswith("w")) and count >= 2:
        margin_x = max(2, int((x1 - x0) * 0.22))
        margin_y = max(2, int((y1 - y0) * 0.22))
        inner_rect = (x0 + margin_x, y0 + margin_y, x1 - margin_x, y1 - margin_y)
        inner = _rect_zone(mask.shape, inner_rect, sigma)
        outer = (
            _rect_zone(mask.shape, rect, sigma * 0.55) - 0.92 * inner
        ).clip(0.0, 1.0)
        zones.extend([outer, inner])
        inner_cp = _token_codepoint(item.components[-1])
        if inner_cp is not None:
            _recursive_layout_zones(
                inner_cp, mask, decompositions, inner_rect, zones,
                depth=depth + 1, maximum_depth=maximum_depth,
                maximum_zones=maximum_zones, sigma=sigma,
            )
        return
    else:
        return

    for token, child_rect in zip(item.components, child_rects):
        if len(zones) >= maximum_zones:
            break
        zones.append(_rect_zone(mask.shape, child_rect, sigma))
        child_cp = _token_codepoint(token)
        if child_cp is not None:
            _recursive_layout_zones(
                child_cp, mask, decompositions, child_rect, zones,
                depth=depth + 1, maximum_depth=maximum_depth,
                maximum_zones=maximum_zones, sigma=sigma,
            )


def component_zones(
    codepoint: int,
    reference_mask: np.ndarray,
    decompositions: dict[int, Decomposition],
    fallback_grid: int = 3,
    maximum_depth: int = 3,
    maximum_zones: int = 32,
) -> list[np.ndarray]:
    """Return hierarchical soft zones for component-aware local repair.

    Decomposition data selects horizontal/vertical/surround layouts. Region
    boundaries are inferred from valleys in the actual ref glyph, so
    the zones follow the proportions of the selected reference font. Nested
    components are expanded recursively up to ``maximum_depth``.
    """

    mask = np.asarray(reference_mask, dtype=np.float32)
    h, w = mask.shape
    sigma = max(1.0, min(h, w) / 100.0)
    zones: list[np.ndarray] = []
    x0, y0, x1, y1 = _bbox(mask)
    _recursive_layout_zones(
        int(codepoint),
        mask,
        decompositions,
        (x0, y0, x1, y1),
        zones,
        depth=0,
        maximum_depth=max(1, int(maximum_depth)),
        maximum_zones=max(2, int(maximum_zones)),
        sigma=sigma,
    )

    if not zones:
        grid = max(2, int(fallback_grid))
        for gy in range(grid):
            for gx in range(grid):
                zone = np.zeros((h, w), dtype=np.float32)
                left = int(round(gx * w / grid))
                right = int(round((gx + 1) * w / grid))
                top = int(round(gy * h / grid))
                bottom = int(round((gy + 1) * h / grid))
                zone[top:bottom, left:right] = 1.0
                zones.append(_feather(zone, sigma))
    # Keep deterministic order and remove near-duplicates caused by nested aliases.
    unique: list[np.ndarray] = []
    signatures: set[bytes] = set()
    for zone in zones:
        signature = (cv2.resize(zone, (16, 16), interpolation=cv2.INTER_AREA) > 0.35).astype(np.uint8).tobytes()
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(zone.astype(np.float32))
        if len(unique) >= max(2, int(maximum_zones)):
            break
    return unique

