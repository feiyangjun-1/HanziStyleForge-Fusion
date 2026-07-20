from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .proxy import diff_visual, gray_to_ink
from .util import atomic_save_pil, cp_label, ensure_dir


def _missing_image(size: int) -> Image.Image:
    """Return a visible placeholder for a glyph image that does not exist."""
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    inset = max(2, min(4, size // 12))
    right = max(inset, size - inset - 1)
    draw.rectangle((inset, inset, right, right), outline=180, width=2)
    draw.line((inset, inset, right, right), fill=180, width=2)
    draw.line((right, inset, inset, right), fill=180, width=2)
    return image


def _load_gray(path: str | Path | None, size: int) -> Image.Image:
    # Path("") resolves to ".".  The missing-target report intentionally
    # passes an empty path, so checking only exists() attempts to open the
    # current directory and raises PermissionError on Windows.
    raw_path = "" if path is None else str(path).strip()
    if not raw_path:
        return _missing_image(size)

    p = Path(raw_path)
    if not p.is_file():
        return _missing_image(size)

    try:
        with Image.open(p) as source:
            return source.convert("L").resize(
                (size, size), Image.Resampling.LANCZOS
            )
    except (OSError, PermissionError):
        # A corrupt, locked or otherwise unreadable preview must not stop the
        # automatic workflow; the audit sheet can safely show a placeholder.
        return _missing_image(size)


def make_audit_contact_sheet(
    items: list[dict],
    output: str | Path,
    cell_size: int = 128,
    columns: int = 3,
    rows_per_page: int = 20,
) -> list[Path]:
    """Create one or more ref / target / diff audit sheets."""
    output = Path(output)
    ensure_dir(output.parent)
    if not items:
        blank = Image.new("RGB", (800, 120), "white")
        ImageDraw.Draw(blank).text((20, 40), "No items", fill="black")
        atomic_save_pil(blank, output)
        return [output]

    font = ImageFont.load_default()
    label_h = 30
    gap = 8
    row_h = cell_size + label_h + gap
    sheet_w = columns * cell_size + (columns + 1) * gap
    result: list[Path] = []

    for page_index, start in enumerate(range(0, len(items), rows_per_page)):
        page_items = items[start : start + rows_per_page]
        sheet_h = len(page_items) * row_h + gap
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        draw = ImageDraw.Draw(sheet)

        for row_index, item in enumerate(page_items):
            y = gap + row_index * row_h
            ref = _load_gray(item["ref_path"], cell_size)
            target = _load_gray(item.get("target_path", ""), cell_size)
            a = gray_to_ink(np.asarray(target))
            b = gray_to_ink(np.asarray(ref))
            diff = Image.fromarray(diff_visual(a, b), mode="RGB")

            x1 = gap
            x2 = gap * 2 + cell_size
            x3 = gap * 3 + cell_size * 2
            sheet.paste(ref.convert("RGB"), (x1, y))
            sheet.paste(target.convert("RGB"), (x2, y))
            sheet.paste(diff, (x3, y))

            cp = int(item["codepoint"])
            status = str(item.get("status", ""))
            score = item.get("structure_score", "")
            label = f"{cp_label(cp)}  ref | target | diff   {status}   score={score}"
            draw.text((gap, y + cell_size + 6), label, font=font, fill="black")

        page_path = (
            output
            if page_index == 0
            else output.with_name(f"{output.stem}_{page_index + 1:03d}{output.suffix}")
        )
        atomic_save_pil(sheet, page_path)
        result.append(page_path)
    return result


def make_training_preview(
    proxy: np.ndarray,
    raw_prediction: np.ndarray,
    ema_prediction: np.ndarray,
    target: np.ndarray,
    codepoints: list[int],
    output: str | Path,
    max_items: int = 12,
) -> None:
    n = min(max_items, len(codepoints), len(proxy), len(raw_prediction), len(ema_prediction), len(target))
    if n <= 0:
        return
    tile = int(proxy.shape[-1])
    gap = 6
    label_h = 24
    width = 4 * tile + 5 * gap
    height = n * (tile + label_h + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i in range(n):
        y = gap + i * (tile + label_h + gap)
        arrays = (proxy[i], raw_prediction[i], ema_prediction[i], target[i])
        for j, arr in enumerate(arrays):
            gray = np.rint((1.0 - np.asarray(arr).clip(0, 1)) * 255).astype(np.uint8)
            image = Image.fromarray(gray, mode="L").convert("RGB")
            x = gap + j * (tile + gap)
            sheet.paste(image, (x, y))
        draw.text(
            (gap, y + tile + 4),
            f"{cp_label(int(codepoints[i]))}  target proxy | raw prediction | EMA prediction | target truth",
            fill="black",
            font=font,
        )

    output = Path(output)
    ensure_dir(output.parent)
    atomic_save_pil(sheet, output)
