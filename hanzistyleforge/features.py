from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .proxy import binary, thin_binary
from .util import atomic_save_pil


PROXY_BASE_CHANNELS = 4
PROXY_EXPANDED_CHANNELS = 10
TARGET_AUX_CHANNELS = 4


def _as_float01(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size and float(arr.max()) > 1.5:
        arr = arr / 255.0
    return arr.clip(0.0, 1.0)


def _normalize_signed(values: np.ndarray, activity: np.ndarray) -> np.ndarray:
    scale = float(np.quantile(np.abs(values), 0.98)) if np.any(activity > 0.01) else 1.0
    scale = max(scale, 1e-5)
    signed = np.clip(values / scale, -1.0, 1.0)
    # Neutral value is 0.5 away from an edge.  This makes affine padding safe.
    return (0.5 + 0.5 * signed * activity).astype(np.float32)


def _point_heatmaps(skeleton_like: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skeleton = thin_binary((skeleton_like >= 0.28).astype(np.uint8))
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = ((skeleton > 0) & (neighbours == 1)).astype(np.float32)
    junctions = ((skeleton > 0) & (neighbours >= 3)).astype(np.float32)
    sigma = max(0.65, skeleton.shape[0] / 220.0)
    endpoints = cv2.GaussianBlur(endpoints, (0, 0), sigmaX=sigma)
    junctions = cv2.GaussianBlur(junctions, (0, 0), sigmaX=sigma)
    if float(endpoints.max()) > 0:
        endpoints /= float(endpoints.max())
    if float(junctions.max()) > 0:
        junctions /= float(junctions.max())
    inverse = (skeleton == 0).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    centreline_band = np.exp(-distance / max(1.0, skeleton.shape[0] / 96.0)).astype(np.float32)
    return endpoints.clip(0, 1), junctions.clip(0, 1), centreline_band.clip(0, 1)


def expand_proxy_channels(proxy4: np.ndarray, size: int | None = None) -> np.ndarray:
    """Expand the persistent RGBA proxy to a ten-channel topology-rich tensor.

    The first four channels are the persistent, style-reduced proxy channels:
    canonical stroke, blurred skeleton, normalized signed distance and coarse
    occupancy.  The six derived channels are computed deterministically:
    boundary magnitude, signed horizontal/vertical edge orientation, endpoint
    heatmap, junction heatmap and centreline proximity.
    """

    arr = _as_float01(proxy4)
    if arr.ndim != 3 or arr.shape[2] < PROXY_BASE_CHANNELS:
        raise ValueError(f"content proxy must be HxWx4+, got {arr.shape}")
    arr = arr[..., :PROXY_BASE_CHANNELS]
    if size is not None and arr.shape[:2] != (int(size), int(size)):
        resized: list[np.ndarray] = []
        for index in range(PROXY_BASE_CHANNELS):
            resized.append(cv2.resize(arr[..., index], (int(size), int(size)), interpolation=cv2.INTER_AREA))
        arr = np.stack(resized, axis=-1)

    canonical = arr[..., 0]
    gx = cv2.Sobel(canonical, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(canonical, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_scale = max(float(np.quantile(edge, 0.985)), 1e-5)
    edge = np.clip(edge / edge_scale, 0.0, 1.0)
    gx_norm = _normalize_signed(gx, edge)
    gy_norm = _normalize_signed(gy, edge)
    endpoints, junctions, centreline_band = _point_heatmaps(arr[..., 1])
    return np.stack(
        [
            arr[..., 0],
            arr[..., 1],
            arr[..., 2],
            arr[..., 3],
            edge,
            gx_norm,
            gy_norm,
            endpoints,
            junctions,
            centreline_band,
        ],
        axis=-1,
    ).astype(np.float32)


def make_target_aux(ink: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Create ink/SDF/skeleton/edge targets as an RGBA-compatible uint8 image."""

    source = _as_float01(ink)
    mask = binary(source, threshold)
    if not np.any(mask):
        sdf = np.full_like(source, 0.5, dtype=np.float32)
        skeleton = np.zeros_like(source, dtype=np.float32)
        edge = np.zeros_like(source, dtype=np.float32)
    else:
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        outside = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
        clip = max(4.0, source.shape[0] * 0.055)
        sdf = np.clip((inside - outside) / clip, -1.0, 1.0)
        sdf = (sdf + 1.0) * 0.5
        skeleton_binary = thin_binary(mask)
        skeleton = cv2.GaussianBlur(
            skeleton_binary.astype(np.float32),
            (0, 0),
            sigmaX=max(0.55, source.shape[0] / 300.0),
        )
        if float(skeleton.max()) > 0:
            skeleton /= float(skeleton.max())
        gx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge /= max(float(np.quantile(edge, 0.985)), 1e-5)
        edge = np.clip(edge, 0.0, 1.0)
    aux = np.stack([source, sdf, skeleton, edge], axis=-1)
    return np.rint(aux.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def save_target_aux(path: str | Path, ink: np.ndarray, threshold: float = 0.5) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_pil(
        Image.fromarray(make_target_aux(ink, threshold=threshold), mode="RGBA"),
        p,
        format="PNG",
    )


def read_target_aux(path: str | Path, size: int | None = None) -> np.ndarray:
    p = Path(path)
    with Image.open(p) as image:
        arr = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    if size is not None and arr.shape[:2] != (int(size), int(size)):
        channels: list[np.ndarray] = []
        for index in range(TARGET_AUX_CHANNELS):
            interpolation = cv2.INTER_LINEAR if index != 2 else cv2.INTER_AREA
            channels.append(cv2.resize(arr[..., index], (int(size), int(size)), interpolation=interpolation))
        arr = np.stack(channels, axis=-1)
    return arr.clip(0.0, 1.0).astype(np.float32)


def target_aux_from_path(target_path: str | Path, aux_path: str | Path, size: int, threshold: float = 0.5) -> np.ndarray:
    aux = Path(aux_path) if str(aux_path) else Path()
    if str(aux_path) and aux.is_file():
        return read_target_aux(aux, size=size)
    with Image.open(target_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - gray
    if ink.shape != (int(size), int(size)):
        ink = cv2.resize(ink, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    return make_target_aux(ink, threshold=threshold).astype(np.float32) / 255.0


def split_prediction(output):
    """Return ink logits and optional auxiliary heads from a model output."""

    if isinstance(output, dict):
        return output["ink"], output.get("sdf"), output.get("skeleton"), output.get("edge")
    if output.ndim != 4:
        raise ValueError(f"model output must be NCHW, got {tuple(output.shape)}")
    if output.shape[1] >= 4:
        return output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4]
    return output[:, 0:1], None, None, None


def ink_probability(output):
    import torch

    ink_logits, _, _, _ = split_prediction(output)
    return torch.sigmoid(ink_logits)
