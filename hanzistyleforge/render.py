from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .util import atomic_save_pil, ensure_dir


def _valid_metric_pair(bottom: float, top: float, upm: float) -> bool:
    span = top - bottom
    return top > 0 and bottom <= 0 and 0.70 * upm <= span <= 1.55 * upm


def get_vertical_bounds(font: TTFont, upm: int) -> tuple[float, float, str]:
    candidates: list[tuple[float, float, str]] = []

    if "OS/2" in font:
        os2 = font["OS/2"]
        typo_top = float(getattr(os2, "sTypoAscender", 0))
        typo_bottom = float(getattr(os2, "sTypoDescender", 0))
        if _valid_metric_pair(typo_bottom, typo_top, upm):
            candidates.append((typo_bottom, typo_top, "OS/2.sTypo"))

        win_top = float(getattr(os2, "usWinAscent", 0))
        win_bottom = -float(getattr(os2, "usWinDescent", 0))
        if _valid_metric_pair(win_bottom, win_top, upm):
            candidates.append((win_bottom, win_top, "OS/2.usWin"))

    if "hhea" in font:
        hhea = font["hhea"]
        hhea_top = float(getattr(hhea, "ascent", 0))
        hhea_bottom = float(getattr(hhea, "descent", 0))
        if _valid_metric_pair(hhea_bottom, hhea_top, upm):
            candidates.append((hhea_bottom, hhea_top, "hhea"))

    if candidates:
        bottom, top, source = min(
            candidates,
            key=lambda item: abs((item[1] - item[0]) - upm),
        )
    else:
        bottom, top, source = -0.12 * upm, 0.88 * upm, "fallback"

    # The renderer already has a visible pad. Add only a small metric safety margin.
    margin = upm * 0.008
    return bottom - margin, top + margin, source


@dataclass
class RenderGeometry:
    size: int
    pad: int
    antialias: int
    upm: int
    y_bottom: float
    y_top: float
    metric_source: str
    scale_aa: float
    x_origin_aa: float
    baseline_aa: float
    font_px: int

    @property
    def vertical_span(self) -> float:
        return self.y_top - self.y_bottom

    def pixel_to_font(self, px: float, py: float) -> tuple[float, float]:
        aa = float(self.antialias)
        x_aa = px * aa
        y_aa = py * aa
        x = (x_aa - self.x_origin_aa) / self.scale_aa
        y = (self.baseline_aa - y_aa) / self.scale_aa
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "pad": self.pad,
            "antialias": self.antialias,
            "upm": self.upm,
            "y_bottom": self.y_bottom,
            "y_top": self.y_top,
            "metric_source": self.metric_source,
            "scale_aa": self.scale_aa,
            "x_origin_aa": self.x_origin_aa,
            "baseline_aa": self.baseline_aa,
            "font_px": self.font_px,
        }


class FontRenderer:
    """Render cmap characters into a stable em box.

    PNGs use a white background and black glyph. Internally, ``render_ink``
    returns a float32 ink mask where 1 is foreground and 0 is background.
    """

    def __init__(
        self,
        font_path: str | Path,
        size: int = 256,
        pad: int = 16,
        antialias: int = 4,
    ) -> None:
        self.path = Path(font_path).resolve()
        self.size = int(size)
        self.pad = int(pad)
        self.antialias = max(1, int(antialias))
        self.font = TTFont(str(self.path), lazy=False)
        self.cmap: dict[int, str] = self.font.getBestCmap() or {}
        self.upm = int(self.font["head"].unitsPerEm)
        y_bottom, y_top, metric_source = get_vertical_bounds(self.font, self.upm)

        aa_size = self.size * self.antialias
        aa_pad = self.pad * self.antialias
        usable_aa = aa_size - 2 * aa_pad
        vertical_span = y_top - y_bottom

        # Pillow's font size is the em scale in pixels.
        font_px_float = usable_aa * self.upm / vertical_span
        font_px = max(8, int(round(font_px_float)))
        scale_aa = font_px / float(self.upm)

        em_width_aa = self.upm * scale_aa
        x_origin_aa = aa_pad + max(0.0, (usable_aa - em_width_aa) / 2.0)
        baseline_aa = aa_pad + y_top * scale_aa

        self.geometry = RenderGeometry(
            size=self.size,
            pad=self.pad,
            antialias=self.antialias,
            upm=self.upm,
            y_bottom=y_bottom,
            y_top=y_top,
            metric_source=metric_source,
            scale_aa=scale_aa,
            x_origin_aa=x_origin_aa,
            baseline_aa=baseline_aa,
            font_px=font_px,
        )

        try:
            layout = ImageFont.Layout.BASIC
            self.pil_font = ImageFont.truetype(
                str(self.path),
                size=font_px,
                layout_engine=layout,
            )
        except Exception:
            self.pil_font = ImageFont.truetype(str(self.path), size=font_px)

    def has_codepoint(self, cp: int) -> bool:
        return cp in self.cmap

    def render_gray(self, cp: int) -> np.ndarray:
        if cp not in self.cmap:
            return np.full((self.size, self.size), 255, dtype=np.uint8)

        aa_size = self.size * self.antialias
        image = Image.new("L", (aa_size, aa_size), 255)
        draw = ImageDraw.Draw(image)
        char = chr(cp)

        try:
            draw.text(
                (self.geometry.x_origin_aa, self.geometry.baseline_aa),
                char,
                font=self.pil_font,
                fill=0,
                anchor="ls",
            )
        except (TypeError, ValueError):
            # Fallback for older Pillow builds lacking baseline anchors.
            bbox = self.pil_font.getbbox(char)
            x = self.geometry.x_origin_aa
            y = self.geometry.baseline_aa - bbox[3]
            draw.text((x, y), char, font=self.pil_font, fill=0)

        if self.antialias > 1:
            image = image.resize(
                (self.size, self.size),
                resample=Image.Resampling.LANCZOS,
            )
        return np.asarray(image, dtype=np.uint8)

    def render_ink(self, cp: int) -> np.ndarray:
        gray = self.render_gray(cp)
        return (1.0 - gray.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def save_png(self, cp: int, path: str | Path) -> None:
        p = Path(path)
        ensure_dir(p.parent)
        atomic_save_pil(Image.fromarray(self.render_gray(cp), mode="L"), p, format="PNG")

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "units_per_em": self.upm,
            "unicode_count": len(self.cmap),
            "tables": list(self.font.keys()),
            "geometry": self.geometry.to_dict(),
        }

    def close(self) -> None:
        self.font.close()

    def __enter__(self) -> "FontRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
