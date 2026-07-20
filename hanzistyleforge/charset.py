from __future__ import annotations

from typing import Iterable

from .util import load_codepoints


# Unicode 17.0 Han ideograph blocks.  The program only rebuilds codepoints
# present in the supplied ref font; these ranges prevent Kana,
# Hangul, Latin, Cyrillic, symbols and punctuation from entering the target.
HAN_UNIFIED_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),     # Extension A
    (0x4E00, 0x9FFF),     # Unified Ideographs
    (0x20000, 0x2A6DF),   # Extension B
    (0x2A700, 0x2B73F),   # Extension C
    (0x2B740, 0x2B81D),   # Extension D
    (0x2B820, 0x2CEAD),   # Extension E
    (0x2CEB0, 0x2EBE0),   # Extension F
    (0x2EBF0, 0x2EE5D),   # Extension I
    (0x30000, 0x3134A),   # Extension G
    (0x31350, 0x323AF),   # Extension H
    (0x323B0, 0x33479),   # Extension J
)

HAN_COMPATIBILITY_RANGES: tuple[tuple[int, int], ...] = (
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)


def in_ranges(codepoint: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def is_han_ideograph(codepoint: int, include_compatibility: bool = True) -> bool:
    # U+3007 IDEOGRAPHIC NUMBER ZERO is outside the unified blocks but is
    # conventionally treated as a Han ideograph by fonts and text systems.
    if codepoint == 0x3007:
        return True
    if in_ranges(codepoint, HAN_UNIFIED_RANGES):
        return True
    return include_compatibility and in_ranges(codepoint, HAN_COMPATIBILITY_RANGES)


# Compatibility aliases used by older internal modules.
CJK_UNIFIED_RANGES = HAN_UNIFIED_RANGES
CJK_COMPATIBILITY_RANGES = HAN_COMPATIBILITY_RANGES
is_cjk_ideograph = is_han_ideograph


def select_target_codepoints(
    reference_cmap: dict[int, str],
    mode: str,
    include_compatibility: bool,
    extra_chars_file: str = "",
) -> list[int]:
    reference = set(reference_cmap)
    normalized = str(mode).lower()
    if normalized in {"reference_han", "reference_cjk"}:
        selected = {
            cp for cp in reference
            if is_han_ideograph(cp, include_compatibility)
        }
    elif normalized in {"reference_bmp_han", "reference_bmp_cjk"}:
        selected = {
            cp for cp in reference
            if cp <= 0xFFFF and is_han_ideograph(cp, include_compatibility)
        }
    elif normalized == "chars_file":
        if not extra_chars_file:
            raise ValueError("scope.extra_chars_file must be set when scope.mode=chars_file.")
        selected = set(load_codepoints(extra_chars_file)) & reference
        selected = {
            cp for cp in selected
            if is_han_ideograph(cp, include_compatibility)
        }
    else:
        raise ValueError(
            f"Unknown scope.mode={mode!r}. Available values: reference_han, reference_bmp_han, chars_file"
        )

    if extra_chars_file and normalized != "chars_file":
        selected.update(
            cp for cp in load_codepoints(extra_chars_file)
            if cp in reference and is_han_ideograph(cp, include_compatibility)
        )
    return sorted(selected)
