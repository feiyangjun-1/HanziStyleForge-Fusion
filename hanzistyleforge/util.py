from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = deep_merge(value, {})
        else:
            result[key] = value
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _flush_and_fsync(handle: Any) -> None:
    """Flush a writable file object all the way to the storage device.

    The project is designed for multi-week runs, so state files must survive a
    process crash or power loss as well as an ordinary Python exception.
    ``fsync`` can still be influenced by the drive/controller cache, but it is
    the strongest portable guarantee available from Python on Windows.
    """

    handle.flush()
    try:
        os.fsync(handle.fileno())
    except (AttributeError, OSError):
        pass


def _fsync_parent_directory(path: Path) -> None:
    """Persist a directory entry on platforms that support directory fsync."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def durable_replace(temp_path: str | Path, final_path: str | Path) -> None:
    """Atomically replace ``final_path`` with an already-written temp file."""

    temp = Path(temp_path)
    final = Path(final_path)
    ensure_dir(final.parent)
    # Re-open and fsync because some third-party writers (Pillow, torch,
    # fontTools) close the file before returning and do not expose the handle.
    try:
        with temp.open("rb") as handle:
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except FileNotFoundError:
        raise
    os.replace(temp, final)
    _fsync_parent_directory(final.parent)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    temp = p.with_name(p.name + ".tmp")
    with temp.open("w", encoding=encoding, newline="") as handle:
        handle.write(text)
        _flush_and_fsync(handle)
    durable_replace(temp, p)


def atomic_save_pil(image: Any, path: str | Path, *, format: str | None = None, **kwargs: Any) -> None:
    """Save a Pillow image through a same-directory temporary file."""

    p = Path(path)
    ensure_dir(p.parent)
    suffix = p.suffix or ".img"
    temp = p.with_name(p.stem + ".tmp" + suffix)
    image.save(temp, format=format, **kwargs)
    if os.environ.get("HSF_DURABLE_IMAGE_WRITES", "0").strip().lower() in {"1", "true", "yes", "on"}:
        durable_replace(temp, p)
    else:
        # Rendered PNGs and generated glyph candidates are reproducible cache
        # artifacts. Atomic replacement is retained, while per-image fsync is
        # skipped to avoid tens of thousands of synchronous disk flushes.
        os.replace(temp, p)


def save_json(path: str | Path, data: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    temp = p.with_name(p.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        _flush_and_fsync(f)
    durable_replace(temp, p)


def cp_label(cp: int) -> str:
    return f"U{cp:04X}" if cp <= 0xFFFF else f"U{cp:06X}"


def cp_filename(cp: int, suffix: str = ".png") -> str:
    return cp_label(cp) + suffix


def cp_to_char(cp: int) -> str:
    try:
        return chr(cp)
    except ValueError:
        return ""


def parse_codepoint_token(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    upper = token.upper()
    if upper.startswith("U+"):
        return int(upper[2:], 16)
    if upper.startswith("U") and all(c in "0123456789ABCDEF" for c in upper[1:]):
        return int(upper[1:], 16)
    if upper.startswith("0X"):
        return int(upper, 16)
    if len(token) == 1:
        return ord(token)
    return None


def load_codepoints(path: str | Path) -> list[int]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8-sig")
    result: list[int] = []
    seen: set[int] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.replace(",", " ").replace(";", " ").split()
        parsed_any = False
        for token in tokens:
            cp = parse_codepoint_token(token)
            if cp is not None:
                parsed_any = True
                if cp not in seen:
                    seen.add(cp)
                    result.append(cp)
        if parsed_any:
            continue
        for ch in stripped:
            cp = ord(ch)
            if cp not in seen:
                seen.add(cp)
                result.append(cp)
    return result


def save_codepoints(path: str | Path, codepoints: Iterable[int]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    lines = []
    for cp in sorted(set(int(x) for x in codepoints)):
        char = cp_to_char(cp)
        lines.append(f"U+{cp:04X}\t{char}")
    atomic_write_text(p, "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def chunks(seq: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def relative_to(path: str | Path, base: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(Path(path).resolve())


def absolute_from(path: str | Path, base: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(base) / p).resolve()


def human_count(value: int) -> str:
    return f"{value:,}"


def robust_median_sigma(values: Sequence[float], floor: float = 1e-6) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, floor
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(1.4826 * mad, floor)
    return med, sigma


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def unique_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}.{i}" in existing:
        i += 1
    return f"{base}.{i}"
