from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


_LEGACY_RECORD = re.compile(r"^([^:]+):([^()]*)\((.*)\)\s*$")
_VARIANT_TAG = re.compile(r"^(.*?)(?:\[([A-Za-z]+)\])?\s*$")
_IDC_ARITY = {
    "⿰": 2,
    "⿱": 2,
    "⿲": 3,
    "⿳": 3,
    "⿴": 2,
    "⿵": 2,
    "⿶": 2,
    "⿷": 2,
    "⿸": 2,
    "⿹": 2,
    "⿺": 2,
    "⿻": 2,
}
_VARIATION_SELECTOR_RANGES = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))


@dataclass(frozen=True)
class IDSNode:
    operator: str | None = None
    token: str = ""
    children: tuple["IDSNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return self.operator is None

    def serialize(self) -> str:
        if self.is_leaf:
            return self.token
        return str(self.operator) + "".join(child.serialize() for child in self.children)


@dataclass(frozen=True)
class Decomposition:
    codepoint: int
    kind: str
    components: tuple[str, ...]
    tree: IDSNode
    sequence: str
    regions: tuple[str, ...] = ()
    source_format: str = "cjkvi-ids"
    variant_count: int = 1


@dataclass(frozen=True)
class _ParsedVariant:
    sequence: str
    regions: tuple[str, ...]
    tree: IDSNode


def _is_variation_selector(character: str) -> bool:
    value = ord(character)
    return any(start <= value <= end for start, end in _VARIATION_SELECTOR_RANGES)


def _token_codepoint(token: str) -> int | None:
    value = str(token).strip()
    if not value or value.startswith("&"):
        return None
    if value.upper().startswith("U+"):
        try:
            return int(value[2:], 16)
        except ValueError:
            return None
    # str.isdigit() is true for Unicode symbols such as circled digits (for
    # example "⑦"), but int("⑦") raises ValueError.  CJKVI IDS deliberately
    # uses these symbols as structural placeholder components, so only plain
    # ASCII decimal text may be interpreted as a numeric codepoint.
    if re.fullmatch(r"[0-9]+", value):
        try:
            number = int(value, 10)
        except ValueError:
            return None
        return number if number >= 0x3400 else None
    characters = [character for character in value if not _is_variation_selector(character)]
    if len(characters) == 1:
        return ord(characters[0])
    return None


def _parse_ids_node(expression: str, index: int = 0) -> tuple[IDSNode, int]:
    length = len(expression)
    while index < length and expression[index].isspace():
        index += 1
    if index >= length:
        raise ValueError("unexpected end of IDS expression")
    character = expression[index]
    if character in _IDC_ARITY:
        arity = _IDC_ARITY[character]
        children: list[IDSNode] = []
        cursor = index + 1
        for _ in range(arity):
            child, cursor = _parse_ids_node(expression, cursor)
            children.append(child)
        return IDSNode(operator=character, children=tuple(children)), cursor
    if character == "&":
        end = expression.find(";", index + 1)
        if end < 0:
            raise ValueError("unterminated IDS entity reference")
        return IDSNode(token=expression[index : end + 1]), end + 1
    cursor = index + 1
    token = character
    while cursor < length and _is_variation_selector(expression[cursor]):
        token += expression[cursor]
        cursor += 1
    return IDSNode(token=token), cursor


def parse_ids_expression(expression: str) -> IDSNode:
    text = str(expression).strip()
    if not text:
        raise ValueError("empty IDS expression")
    node, cursor = _parse_ids_node(text, 0)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor != len(text):
        raise ValueError(f"trailing IDS content at offset {cursor}: {text[cursor:cursor + 16]!r}")
    return node


def _variant_from_field(field: str) -> _ParsedVariant | None:
    match = _VARIANT_TAG.match(str(field).strip())
    if not match:
        return None
    sequence, region_text = match.groups()
    sequence = sequence.strip()
    if not sequence:
        return None
    try:
        tree = parse_ids_expression(sequence)
    except ValueError:
        return None
    regions = tuple(dict.fromkeys(region_text.upper())) if region_text else ()
    return _ParsedVariant(sequence=sequence, regions=regions, tree=tree)


def _choose_variant(
    variants: list[_ParsedVariant],
    region_priority: Iterable[str] | None,
    include_obsolete: bool,
) -> _ParsedVariant | None:
    if not variants:
        return None
    usable = variants if include_obsolete else [item for item in variants if "O" not in item.regions]
    if not usable:
        usable = variants
    priority = [str(value).strip().upper() for value in (region_priority or ()) if str(value).strip()]
    for region in priority:
        for item in usable:
            if region in item.regions:
                return item
    for item in usable:
        if not item.regions:
            return item
    return usable[0]


def _make_decomposition(
    codepoint: int,
    variant: _ParsedVariant,
    *,
    source_format: str,
    variant_count: int,
) -> Decomposition:
    tree = variant.tree
    kind = tree.operator or "leaf"
    components = tuple(child.serialize() for child in tree.children)
    return Decomposition(
        codepoint=int(codepoint),
        kind=kind,
        components=components,
        tree=tree,
        sequence=variant.sequence,
        regions=variant.regions,
        source_format=source_format,
        variant_count=max(1, int(variant_count)),
    )


def _load_cjkvi_ids(
    path: Path,
    *,
    region_priority: Iterable[str] | None,
    include_obsolete: bool,
) -> dict[int, Decomposition]:
    result: dict[int, Decomposition] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3 or not fields[0].upper().startswith("U+"):
                continue
            try:
                codepoint = int(fields[0][2:], 16)
            except ValueError:
                continue
            variants = [item for item in (_variant_from_field(field) for field in fields[2:]) if item is not None]
            selected = _choose_variant(variants, region_priority, include_obsolete)
            if selected is None or selected.tree.is_leaf:
                continue
            result[codepoint] = _make_decomposition(
                codepoint,
                selected,
                source_format="cjkvi-ids",
                variant_count=len(variants),
            )
    return result


def _legacy_operator(kind: str, count: int) -> str:
    value = str(kind).lower().strip()
    if value.startswith("a"):
        return "⿲" if count == 3 else "⿰"
    if value.startswith("d"):
        return "⿳" if count == 3 else "⿱"
    if value.startswith(("s", "w", "st", "sl", "sr")):
        return "⿴"
    return "⿰" if count >= 2 else ""


def _load_legacy(path: Path) -> dict[int, Decomposition]:
    result: dict[int, Decomposition] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _LEGACY_RECORD.match(line)
            if not match:
                continue
            head, kind, body = match.groups()
            codepoint = _token_codepoint(head)
            if codepoint is None:
                continue
            components = tuple(part.strip() for part in body.split(",") if part.strip())
            operator = _legacy_operator(kind, len(components))
            if not operator or _IDC_ARITY.get(operator) != len(components):
                continue
            tree = IDSNode(operator=operator, children=tuple(IDSNode(token=value) for value in components))
            variant = _ParsedVariant(tree.serialize(), (), tree)
            result[codepoint] = _make_decomposition(
                codepoint,
                variant,
                source_format="cjk-decomp-legacy",
                variant_count=1,
            )
    return result


def load_decompositions(
    path: str | Path,
    *,
    region_priority: Iterable[str] | None = None,
    include_obsolete: bool = False,
) -> dict[int, Decomposition]:
    source = Path(path)
    if not source.is_file():
        return {}
    with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first_record = ""
        for line in handle:
            if line.strip() and not line.startswith("#"):
                first_record = line
                break
    if first_record.upper().startswith("U+") and "\t" in first_record:
        return _load_cjkvi_ids(
            source,
            region_priority=region_priority,
            include_obsolete=include_obsolete,
        )
    return _load_legacy(source)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask) > 0.2)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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


def _rect_zone(shape: tuple[int, int], rect: tuple[int, int, int, int], sigma: float) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = rect
    zone = np.zeros((h, w), dtype=np.float32)
    zone[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 1.0
    return _feather(zone, sigma)


def _inset_rect(
    rect: tuple[int, int, int, int],
    operator: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    left = right = max(2, int(round(width * 0.20)))
    top = bottom = max(2, int(round(height * 0.20)))
    small_x = max(1, int(round(width * 0.05)))
    small_y = max(1, int(round(height * 0.05)))
    if operator == "⿵":
        bottom = small_y
    elif operator == "⿶":
        top = small_y
    elif operator == "⿷":
        right = small_x
    elif operator == "⿸":
        right, bottom = small_x, small_y
    elif operator == "⿹":
        left, bottom = small_x, small_y
    elif operator == "⿺":
        right, top = small_x, small_y
    ix0, iy0, ix1, iy1 = x0 + left, y0 + top, x1 - right, y1 - bottom
    if ix1 - ix0 < 4:
        ix0, ix1 = x0 + max(1, width // 4), x1 - max(1, width // 4)
    if iy1 - iy0 < 4:
        iy0, iy1 = y0 + max(1, height // 4), y1 - max(1, height // 4)
    return ix0, iy0, ix1, iy1


def _child_layouts(
    node: IDSNode,
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    sigma: float,
) -> list[tuple[IDSNode, tuple[int, int, int, int], np.ndarray]]:
    if node.is_leaf or not node.children:
        return []
    operator = str(node.operator)
    count = len(node.children)
    x0, y0, x1, y1 = rect
    output: list[tuple[IDSNode, tuple[int, int, int, int], np.ndarray]] = []
    if operator in {"⿰", "⿲"}:
        splits = [x0, *_region_splits(mask, rect, count, axis=1), x1]
        rects = [(left, y0, right, y1) for left, right in zip(splits[:-1], splits[1:])]
    elif operator in {"⿱", "⿳"}:
        splits = [y0, *_region_splits(mask, rect, count, axis=0), y1]
        rects = [(x0, top, x1, bottom) for top, bottom in zip(splits[:-1], splits[1:])]
    elif operator in {"⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺"} and count == 2:
        inner_rect = _inset_rect(rect, operator)
        inner_zone = _rect_zone(mask.shape, inner_rect, sigma)
        outer_zone = (_rect_zone(mask.shape, rect, sigma * 0.55) - 0.92 * inner_zone).clip(0.0, 1.0)
        return [
            (node.children[0], rect, outer_zone.astype(np.float32)),
            (node.children[1], inner_rect, inner_zone.astype(np.float32)),
        ]
    elif operator == "⿻" and count == 2:
        rects = [rect, rect]
    else:
        axis = 1 if (x1 - x0) >= (y1 - y0) else 0
        if axis == 1:
            splits = [x0, *_region_splits(mask, rect, count, axis=1), x1]
            rects = [(left, y0, right, y1) for left, right in zip(splits[:-1], splits[1:])]
        else:
            splits = [y0, *_region_splits(mask, rect, count, axis=0), y1]
            rects = [(x0, top, x1, bottom) for top, bottom in zip(splits[:-1], splits[1:])]
    for child, child_rect in zip(node.children, rects):
        output.append((child, child_rect, _rect_zone(mask.shape, child_rect, sigma)))
    return output


def decomposition_regions(
    codepoint: int,
    mask: np.ndarray,
    decompositions: dict[int, Decomposition],
    *,
    maximum_depth: int = 3,
    maximum_regions: int = 48,
    minimum_region_size: int = 8,
) -> list[tuple[str, tuple[int, int, int, int], int, str, np.ndarray]]:
    source = np.asarray(mask, dtype=np.float32)
    sigma = max(1.0, min(source.shape) / 100.0)
    root_item = decompositions.get(int(codepoint))
    if root_item is None:
        return []
    root_rect = _bbox(source)
    regions: list[tuple[str, tuple[int, int, int, int], int, str, np.ndarray]] = []

    def visit_node(
        node: IDSNode,
        rect: tuple[int, int, int, int],
        depth: int,
        path: str,
        active_codepoints: frozenset[int],
    ) -> None:
        if depth >= maximum_depth or len(regions) >= maximum_regions or node.is_leaf:
            return
        for index, (child, child_rect, zone) in enumerate(_child_layouts(node, source, rect, sigma)):
            if len(regions) >= maximum_regions:
                break
            x0, y0, x1, y1 = child_rect
            if x1 - x0 < minimum_region_size or y1 - y0 < minimum_region_size:
                continue
            label = child.serialize()
            child_path = f"{path}/{index}:{label}"
            regions.append((label, child_rect, depth, child_path, zone.astype(np.float32)))
            if not child.is_leaf:
                visit_node(child, child_rect, depth + 1, child_path, active_codepoints)
                continue
            child_cp = _token_codepoint(child.token)
            if child_cp is None or child_cp in active_codepoints:
                continue
            nested = decompositions.get(child_cp)
            if nested is not None:
                visit_node(
                    nested.tree,
                    child_rect,
                    depth + 1,
                    child_path,
                    active_codepoints | {child_cp},
                )

    visit_node(
        root_item.tree,
        root_rect,
        0,
        f"U+{int(codepoint):04X}",
        frozenset({int(codepoint)}),
    )
    unique: list[tuple[str, tuple[int, int, int, int], int, str, np.ndarray]] = []
    signatures: set[tuple[str, int, int, int, int, bytes]] = set()
    for label, rect, depth, path, zone in regions:
        signature = (
            label,
            *rect,
            (cv2.resize(zone, (12, 12), interpolation=cv2.INTER_AREA) > 0.35).astype(np.uint8).tobytes(),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append((label, rect, depth, path, zone))
        if len(unique) >= maximum_regions:
            break
    return unique


def component_zones(
    codepoint: int,
    reference_mask: np.ndarray,
    decompositions: dict[int, Decomposition],
    fallback_grid: int = 3,
    maximum_depth: int = 3,
    maximum_zones: int = 32,
) -> list[np.ndarray]:
    """Return hierarchical soft zones from standard IDS component layouts."""

    mask = np.asarray(reference_mask, dtype=np.float32)
    regions = decomposition_regions(
        int(codepoint),
        mask,
        decompositions,
        maximum_depth=max(1, int(maximum_depth)),
        maximum_regions=max(2, int(maximum_zones)),
    )
    zones = [zone for _, _, _, _, zone in regions]
    if not zones:
        h, w = mask.shape
        sigma = max(1.0, min(h, w) / 100.0)
        grid = max(2, int(fallback_grid))
        for gy in range(grid):
            for gx in range(grid):
                left = int(round(gx * w / grid))
                right = int(round((gx + 1) * w / grid))
                top = int(round(gy * h / grid))
                bottom = int(round((gy + 1) * h / grid))
                zones.append(_rect_zone(mask.shape, (left, top, right, bottom), sigma))
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
