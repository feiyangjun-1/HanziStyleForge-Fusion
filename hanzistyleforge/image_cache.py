from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

_Mode = Literal["gray", "rgba", "aux"]


def _limit_bytes() -> int:
    raw = os.environ.get("HSF_IMAGE_CACHE_MB", "192")
    try:
        value = max(0, int(float(raw)))
    except (TypeError, ValueError):
        value = 192
    return value * 1024 * 1024


_LOCK = threading.RLock()
_CACHE: "OrderedDict[tuple[str, int, int, str, int, int], np.ndarray]" = OrderedDict()
_CACHE_BYTES = 0


def set_cache_limit_mb(value: int) -> None:
    os.environ["HSF_IMAGE_CACHE_MB"] = str(max(0, int(value)))
    _trim()


def clear() -> None:
    global _CACHE_BYTES
    with _LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0


def invalidate(path: str | Path) -> None:
    global _CACHE_BYTES
    resolved = str(Path(path).resolve())
    with _LOCK:
        stale = [key for key in _CACHE if key[0] == resolved]
        for key in stale:
            _CACHE_BYTES -= int(_CACHE.pop(key).nbytes)


def _trim() -> None:
    global _CACHE_BYTES
    limit = _limit_bytes()
    with _LOCK:
        if limit <= 0:
            _CACHE.clear()
            _CACHE_BYTES = 0
            return
        while _CACHE and _CACHE_BYTES > limit:
            _, array = _CACHE.popitem(last=False)
            _CACHE_BYTES -= int(array.nbytes)


def _decode(path: Path, mode: _Mode) -> np.ndarray:
    # np.fromfile + imdecode supports non-ASCII Windows paths while avoiding
    # PIL object creation on the hot path.
    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if mode == "gray" else cv2.IMREAD_UNCHANGED
        decoded = cv2.imdecode(payload, flag)
        if decoded is not None:
            if mode == "gray":
                if decoded.ndim == 3:
                    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                return np.ascontiguousarray(decoded, dtype=np.uint8)
            if decoded.ndim == 2:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGBA)
            elif decoded.shape[2] == 3:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGBA)
            else:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
            return np.ascontiguousarray(decoded, dtype=np.uint8)
    except (OSError, ValueError, cv2.error):
        pass
    with Image.open(path) as image:
        converted = image.convert("L" if mode == "gray" else "RGBA")
        return np.ascontiguousarray(np.asarray(converted, dtype=np.uint8))


def _read(path: str | Path, mode: _Mode, size: int | None) -> np.ndarray:
    global _CACHE_BYTES
    source = Path(path)
    stat = source.stat()
    target = int(size or 0)
    channels = 1 if mode == "gray" else 4
    key = (str(source.resolve()), int(stat.st_mtime_ns), int(stat.st_size), mode, target, channels)
    limit = _limit_bytes()
    if limit > 0:
        with _LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                _CACHE.move_to_end(key)
                return cached
    array = _decode(source, mode)
    if target > 0 and array.shape[:2] != (target, target):
        if mode == "gray":
            array = cv2.resize(array, (target, target), interpolation=cv2.INTER_AREA)
        else:
            interpolations = (
                (cv2.INTER_LINEAR, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_LINEAR)
                if mode == "aux"
                else (cv2.INTER_AREA,) * 4
            )
            array = np.stack(
                [
                    cv2.resize(array[..., channel], (target, target), interpolation=interpolations[channel])
                    for channel in range(4)
                ],
                axis=-1,
            )
        array = np.ascontiguousarray(array, dtype=np.uint8)
    array.setflags(write=False)
    if limit > 0 and int(array.nbytes) <= limit:
        with _LOCK:
            previous = _CACHE.pop(key, None)
            if previous is not None:
                _CACHE_BYTES -= int(previous.nbytes)
            _CACHE[key] = array
            _CACHE_BYTES += int(array.nbytes)
        _trim()
    return array


def read_gray_u8(path: str | Path, size: int | None = None) -> np.ndarray:
    return _read(path, "gray", size)


def read_rgba_u8(path: str | Path, size: int | None = None) -> np.ndarray:
    return _read(path, "rgba", size)


def read_aux_rgba_u8(path: str | Path, size: int | None = None) -> np.ndarray:
    return _read(path, "aux", size)
