from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .proxy import binary, component_hole_counts, thin_binary


@dataclass(frozen=True)
class TopologySignature:
    components: int
    holes: int
    endpoints: int
    junctions: int
    skeleton_pixels: int
    endpoint_map: np.ndarray
    junction_map: np.ndarray
    skeleton: np.ndarray

    def to_dict(self) -> dict[str, int]:
        return {
            "components": int(self.components),
            "holes": int(self.holes),
            "endpoints": int(self.endpoints),
            "junctions": int(self.junctions),
            "skeleton_pixels": int(self.skeleton_pixels),
        }


def _resize_mask(mask_or_ink: np.ndarray, size: int) -> np.ndarray:
    source = binary(np.asarray(mask_or_ink), threshold=0.5)
    if source.shape != (size, size):
        source = cv2.resize(source.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return (source > 0).astype(np.uint8)


def _remove_tiny_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    source = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8)
    output = np.zeros_like(source)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(minimum_area):
            output[labels == index] = 1
    return output


def _prune_skeleton(skeleton: np.ndarray, iterations: int) -> np.ndarray:
    """Remove one-pixel spurs without changing the main skeleton topology.

    Chinese glyph raster skeletons often contain tiny anti-aliasing spurs. A
    small number of endpoint-removal rounds makes endpoint/junction statistics
    much more stable across fonts while retaining real stroke branches.
    """

    result = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    for _ in range(max(0, int(iterations))):
        neighbours = cv2.filter2D(result, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = (result > 0) & (neighbours <= 1)
        if not np.any(endpoints):
            break
        candidate = result.copy()
        candidate[endpoints] = 0
        # Do not let pruning erase a tiny standalone dot component entirely.
        if int(candidate.sum()) < max(4, int(result.sum() * 0.85)):
            break
        result = candidate
    return result


def _cluster_count(mask: np.ndarray, minimum_area: int = 1) -> tuple[int, np.ndarray]:
    source = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(source, connectivity=8)
    points: list[tuple[int, int]] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(minimum_area):
            x, y = centroids[index]
            points.append((int(round(x)), int(round(y))))
    point_map = np.zeros_like(source)
    for x, y in points:
        if 0 <= y < source.shape[0] and 0 <= x < source.shape[1]:
            point_map[y, x] = 1
    return len(points), point_map


def _fast_morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    source = (mask > 0).astype(np.uint8)
    skeleton = np.zeros_like(source)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = source.copy()
    # Morphological skeletonization performs the heavy work inside OpenCV and
    # is substantially faster than Python-level Zhang-Suen on full CJK sets.
    for _ in range(max(source.shape) // 2 + 2):
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = cv2.erode(current, element)
        if cv2.countNonZero(current) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def topology_signature(
    mask_or_ink: np.ndarray,
    *,
    size: int = 128,
    prune_iterations: int = 1,
    skeleton_hint: np.ndarray | None = None,
) -> TopologySignature:
    mask = _resize_mask(mask_or_ink, int(size))
    minimum_component = max(2, int(size * size * 0.00008))
    mask = _remove_tiny_components(mask, minimum_component)
    components, holes = component_hole_counts(mask, minimum_area=minimum_component)
    if skeleton_hint is not None:
        hint = np.asarray(skeleton_hint, dtype=np.float32)
        if hint.shape != (int(size), int(size)):
            hint = cv2.resize(hint, (int(size), int(size)), interpolation=cv2.INTER_LINEAR)
        skeleton = (hint >= 0.42).astype(np.uint8)
    else:
        skeleton = _fast_morphological_skeleton(mask)
    skeleton = _prune_skeleton(skeleton, int(prune_iterations))
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoint_pixels = ((skeleton > 0) & (neighbours == 1)).astype(np.uint8)
    junction_pixels = ((skeleton > 0) & (neighbours >= 3)).astype(np.uint8)
    endpoints, endpoint_map = _cluster_count(endpoint_pixels, minimum_area=1)
    # Junctions form multi-pixel clusters, so count clusters rather than pixels.
    junctions, junction_map = _cluster_count(junction_pixels, minimum_area=1)
    return TopologySignature(
        components=int(components),
        holes=int(holes),
        endpoints=int(endpoints),
        junctions=int(junctions),
        skeleton_pixels=int(skeleton.sum()),
        endpoint_map=endpoint_map,
        junction_map=junction_map,
        skeleton=skeleton.astype(np.uint8),
    )


def _directed_distance(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    source_bool = source > 0
    target_bool = target > 0
    if not np.any(source_bool):
        return (0.0, 0.0, 0.0) if not np.any(target_bool) else (1.0, 1.0, 1.0)
    if not np.any(target_bool):
        return 1.0, 1.0, 1.0
    distance = cv2.distanceTransform((~target_bool).astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[source_bool]
    norm = float(max(source.shape))
    return (
        float(np.mean(values) / norm),
        float(np.quantile(values, 0.90) / norm),
        float(np.quantile(values, 0.98) / norm),
    )


def _point_chamfer(source: np.ndarray, target: np.ndarray) -> float:
    if not np.any(source) and not np.any(target):
        return 0.0
    if not np.any(source) or not np.any(target):
        return 1.0
    a_mean, _, _ = _directed_distance(source, target)
    b_mean, _, _ = _directed_distance(target, source)
    return float((a_mean + b_mean) / 2.0)




def _component_and_hole_points(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = (mask > 0).astype(np.uint8)
    component_map = np.zeros_like(source)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(source, connectivity=8)
    minimum = max(2, int(source.size * 0.00008))
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            x, y = centroids[index]
            component_map[int(round(y)), int(round(x))] = 1

    inverse = (1 - source).astype(np.uint8)
    hcount, hlabels, hstats, hcentroids = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    border_labels = (
        set(int(v) for v in hlabels[0, :])
        | set(int(v) for v in hlabels[-1, :])
        | set(int(v) for v in hlabels[:, 0])
        | set(int(v) for v in hlabels[:, -1])
    )
    hole_map = np.zeros_like(source)
    for index in range(1, hcount):
        if index not in border_labels and int(hstats[index, cv2.CC_STAT_AREA]) >= minimum:
            x, y = hcentroids[index]
            hole_map[int(round(y)), int(round(x))] = 1
    return component_map, hole_map


def _zone_distance(reference: np.ndarray, candidate: np.ndarray, grid: int = 8) -> float:
    a = cv2.resize(reference.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    b = cv2.resize(candidate.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    a /= max(float(a.sum()), 1e-6)
    b /= max(float(b.sum()), 1e-6)
    return float(np.abs(a - b).sum() / 2.0)

def topology_metrics(
    reference_mask_or_ink: np.ndarray,
    candidate_mask_or_ink: np.ndarray,
    *,
    size: int = 128,
    prune_iterations: int = 1,
    reference_signature: TopologySignature | None = None,
    candidate_signature: TopologySignature | None = None,
) -> dict[str, Any]:
    reference_mask = _resize_mask(reference_mask_or_ink, int(size))
    candidate_mask = _resize_mask(candidate_mask_or_ink, int(size))
    reference = reference_signature or topology_signature(
        reference_mask, size=int(size), prune_iterations=int(prune_iterations)
    )
    candidate = candidate_signature or topology_signature(
        candidate_mask, size=int(size), prune_iterations=int(prune_iterations)
    )
    ref_to_candidate_mean, ref_to_candidate_p90, ref_to_candidate_p98 = _directed_distance(
        reference.skeleton, candidate.skeleton
    )
    candidate_to_ref_mean, candidate_to_ref_p90, candidate_to_ref_p98 = _directed_distance(
        candidate.skeleton, reference.skeleton
    )
    component_delta = abs(candidate.components - reference.components)
    hole_delta = abs(candidate.holes - reference.holes)
    endpoint_delta = abs(candidate.endpoints - reference.endpoints)
    junction_delta = abs(candidate.junctions - reference.junctions)
    endpoint_chamfer = _point_chamfer(reference.endpoint_map, candidate.endpoint_map)
    junction_chamfer = _point_chamfer(reference.junction_map, candidate.junction_map)
    ref_component_map, ref_hole_map = _component_and_hole_points(reference_mask)
    cand_component_map, cand_hole_map = _component_and_hole_points(candidate_mask)
    component_centroid_chamfer = _point_chamfer(ref_component_map, cand_component_map)
    hole_centroid_chamfer = _point_chamfer(ref_hole_map, cand_hole_map)
    zone_skeleton_distance = _zone_distance(reference.skeleton, candidate.skeleton, grid=8)
    zone_ink_distance = _zone_distance(reference_mask, candidate_mask, grid=8)
    skeleton_length_delta = abs(candidate.skeleton_pixels - reference.skeleton_pixels) / max(
        1, reference.skeleton_pixels
    )
    euler_delta = abs((candidate.components - candidate.holes) - (reference.components - reference.holes))
    score = (
        0.19 * (ref_to_candidate_mean + candidate_to_ref_mean)
        + 0.085 * (ref_to_candidate_p90 + candidate_to_ref_p90)
        + 0.055 * min(1.0, endpoint_chamfer)
        + 0.050 * min(1.0, junction_chamfer)
        + 0.040 * min(1.0, component_centroid_chamfer)
        + 0.060 * min(1.0, hole_centroid_chamfer)
        + 0.060 * min(1.0, zone_skeleton_distance)
        + 0.040 * min(1.0, zone_ink_distance)
        + 0.055 * min(1.0, skeleton_length_delta)
        + 0.085 * min(1.0, component_delta / 3.0)
        + 0.115 * min(1.0, hole_delta / 3.0)
        + 0.040 * min(1.0, euler_delta / 3.0)
        + 0.065 * min(1.0, endpoint_delta / max(3.0, reference.endpoints * 0.35 + 1.0))
        + 0.060 * min(1.0, junction_delta / max(3.0, reference.junctions * 0.35 + 1.0))
    )
    return {
        "topology_score": float(score),
        "component_delta": int(component_delta),
        "hole_delta": int(hole_delta),
        "euler_delta": int(euler_delta),
        "endpoint_delta": int(endpoint_delta),
        "junction_delta": int(junction_delta),
        "endpoint_chamfer": float(endpoint_chamfer),
        "junction_chamfer": float(junction_chamfer),
        "component_centroid_chamfer": float(component_centroid_chamfer),
        "hole_centroid_chamfer": float(hole_centroid_chamfer),
        "zone_skeleton_distance": float(zone_skeleton_distance),
        "zone_ink_distance": float(zone_ink_distance),
        "skeleton_length_delta": float(skeleton_length_delta),
        "reference_to_candidate_mean": float(ref_to_candidate_mean),
        "reference_to_candidate_p90": float(ref_to_candidate_p90),
        "reference_to_candidate_p98": float(ref_to_candidate_p98),
        "candidate_to_reference_mean": float(candidate_to_ref_mean),
        "candidate_to_reference_p90": float(candidate_to_ref_p90),
        "candidate_to_reference_p98": float(candidate_to_ref_p98),
        "reference": reference.to_dict(),
        "candidate": candidate.to_dict(),
    }


def validate_topology(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    reference = metrics.get("reference", {})
    max_endpoint_delta = max(
        int(config.get("minimum_endpoint_tolerance", 2)),
        int(round(float(reference.get("endpoints", 0)) * float(config.get("endpoint_tolerance_ratio", 0.22)))),
    )
    max_junction_delta = max(
        int(config.get("minimum_junction_tolerance", 2)),
        int(round(float(reference.get("junctions", 0)) * float(config.get("junction_tolerance_ratio", 0.24)))),
    )
    reasons: list[str] = []
    if int(metrics["component_delta"]) > int(config.get("maximum_component_delta", 0)):
        reasons.append("component_delta")
    if int(metrics["hole_delta"]) > int(config.get("maximum_hole_delta", 0)):
        reasons.append("hole_delta")
    if int(metrics["endpoint_delta"]) > max_endpoint_delta:
        reasons.append("endpoint_delta")
    if int(metrics["junction_delta"]) > max_junction_delta:
        reasons.append("junction_delta")
    if float(metrics["reference_to_candidate_p90"]) > float(config.get("maximum_missing_skeleton_p90", 0.032)):
        reasons.append("missing_skeleton")
    if float(metrics["candidate_to_reference_p90"]) > float(config.get("maximum_extra_skeleton_p90", 0.036)):
        reasons.append("extra_skeleton")
    if float(metrics.get("hole_centroid_chamfer", 0.0)) > float(config.get("maximum_hole_centroid_chamfer", 0.055)):
        reasons.append("hole_position")
    if float(metrics.get("zone_skeleton_distance", 0.0)) > float(config.get("maximum_zone_skeleton_distance", 0.24)):
        reasons.append("zone_structure")
    if int(metrics.get("euler_delta", 0)) > int(config.get("maximum_euler_delta", 0)):
        reasons.append("euler_delta")
    if float(metrics["topology_score"]) > float(config.get("maximum_topology_score", 0.085)):
        reasons.append("topology_score")
    return {
        "hard_pass": len(reasons) == 0,
        "reasons": reasons,
        "endpoint_tolerance": int(max_endpoint_delta),
        "junction_tolerance": int(max_junction_delta),
    }


def structure_lock_probability(
    probability: np.ndarray,
    reference_proxy: np.ndarray,
    *,
    target_stroke_radius: float,
    profile_size: int,
    core_strength: float = 0.92,
    maximum_radius_multiplier: float = 2.7,
) -> np.ndarray:
    """Project a candidate back onto the ref skeleton without copying ref style.

    The function retains candidate darkness close to the reference centreline,
    inserts a guaranteed inner core, and suppresses isolated material far from
    the ref skeleton. It is deliberately conservative and does not claim to fix
    semantic errors that are absent from the reference proxy.
    """

    candidate = np.asarray(probability, dtype=np.float32).clip(0.0, 1.0)
    size = candidate.shape[0]
    base = np.asarray(reference_proxy[..., 0], dtype=np.float32)
    blurred_skeleton = np.asarray(reference_proxy[..., 1], dtype=np.float32) if reference_proxy.shape[-1] > 1 else base
    if base.shape != candidate.shape:
        base = cv2.resize(base, (size, size), interpolation=cv2.INTER_LINEAR)
        blurred_skeleton = cv2.resize(blurred_skeleton, (size, size), interpolation=cv2.INTER_LINEAR)
    # Proxy channel 1 is already a normalized centreline map, avoiding an
    # expensive thinning operation for each candidate family.
    skeleton = (blurred_skeleton >= 0.42).astype(np.uint8)
    if not np.any(skeleton):
        return candidate
    radius = max(1.0, float(target_stroke_radius) * size / max(1.0, float(profile_size)))
    core_radius = max(1, int(round(radius * 0.62)))
    allow_radius = max(core_radius + 1, int(round(radius * float(maximum_radius_multiplier))))
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_radius * 2 + 1, core_radius * 2 + 1))
    allow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (allow_radius * 2 + 1, allow_radius * 2 + 1))
    core = cv2.dilate(skeleton.astype(np.uint8), core_kernel).astype(np.float32)
    allowed = cv2.dilate(skeleton.astype(np.uint8), allow_kernel).astype(np.float32)
    projected = candidate * allowed
    projected = np.maximum(projected, core * float(core_strength))
    # Keep antialiased borders around the permitted region.
    projected = cv2.GaussianBlur(projected, (0, 0), max(0.35, size / 768.0))
    return projected.clip(0.0, 1.0)
