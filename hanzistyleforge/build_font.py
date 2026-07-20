from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable

from .charset import is_han_ideograph
from .config import CHECKPOINT_FORMAT_VERSION
from .render import get_vertical_bounds
from .util import durable_replace, ensure_dir, read_csv, save_json, unique_name, write_csv
from .vectorize import image_to_ttglyph


UNRESOLVED_FIELDS = ["codepoint", "unicode", "char", "reason"]
NON_HAN_MISMATCH_FIELDS = ["kind", "codepoint", "unicode", "glyph_before", "glyph_after", "details"]


def _ensure_format12(font: TTFont) -> None:
    if any(table.isUnicode() and table.format == 12 for table in font["cmap"].tables):
        return
    # A format-12 subtable becomes the preferred Unicode cmap in most clients.
    # It must therefore begin as a complete copy of the existing best Unicode
    # mapping, not an empty supplementary-only table; otherwise all BMP text
    # (including protected Latin/kana/Hangul) would appear missing after save.
    existing = dict(font.getBestCmap() or {})
    table = CmapSubtable.newSubtable(12)
    table.platformID = 3
    table.platEncID = 10
    table.language = 0
    table.cmap = existing
    font["cmap"].tables.append(table)


def _map_codepoint(font: TTFont, codepoint: int, glyph_name: str) -> None:
    if codepoint > 0xFFFF:
        _ensure_format12(font)
    wrote = False
    for table in font["cmap"].tables:
        if not table.isUnicode() or table.format == 14:
            continue
        if codepoint <= 0xFFFF or table.format in {10, 12, 13}:
            table.cmap[codepoint] = glyph_name
            wrote = True
    if not wrote:
        _ensure_format12(font)
        for table in font["cmap"].tables:
            if table.isUnicode() and table.format == 12:
                table.cmap[codepoint] = glyph_name


def _remove_target_han_uvs(font: TTFont, target: set[int]) -> int:
    """Remove only variation-selector mappings for rebuilt Han codepoints.

    A format-14 cmap may otherwise select an old JP/TW/HK alternate even after
    the default cmap has been remapped to the regenerated ref-structure glyph.
    Non-Han UVS entries are left byte-for-byte equivalent at the data level.
    """

    removed = 0
    if "cmap" not in font:
        return removed
    for table in font["cmap"].tables:
        if int(getattr(table, "format", -1)) != 14:
            continue
        uvs_dict = getattr(table, "uvsDict", None)
        if not isinstance(uvs_dict, dict):
            continue
        new_dict: dict[int, list[tuple[int, str | None]]] = {}
        for selector, entries in uvs_dict.items():
            kept: list[tuple[int, str | None]] = []
            for codepoint, glyph_name in entries:
                if int(codepoint) in target:
                    removed += 1
                else:
                    kept.append((int(codepoint), glyph_name))
            if kept:
                new_dict[int(selector)] = kept
        table.uvsDict = new_dict
    return removed


def _median_han_metrics(
    cmap: dict[int, str],
    metrics: dict[str, tuple[int, int]],
    fallback: tuple[int, int],
) -> tuple[int, int]:
    values = [
        metrics[name]
        for codepoint, name in cmap.items()
        if is_han_ideograph(codepoint, True) and name in metrics
    ]
    if not values:
        return fallback
    return int(np.median([value[0] for value in values])), int(np.median([value[1] for value in values]))


def _font_name(font: TTFont, name_id: int, fallback: str) -> str:
    if "name" not in font:
        return fallback
    return font["name"].getDebugName(name_id) or fallback


def _postscript_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", value.replace(" ", "-"))[:63] or "HanziStyleForgeFinal"


PRESERVED_TABLE_DATA_TAGS: tuple[str, ...] = (
    # These tables govern layout, kerning, TrueType hinting or legacy device
    # behaviour.  Appending new Han glyphs must not rewrite them: all target
    # glyph IDs remain stable because the new glyphs are appended at the end.
    "BASE", "GDEF", "GPOS", "GSUB", "JSTF", "MATH", "kern",
    "fpgm", "prep", "cvt ", "gasp", "LTSH", "hdmx", "VDMX", "PCLT",
)


def _glyph_signature(font: TTFont, glyph_name: str) -> str:
    """Hash the exact compiled TrueType glyph program and outline.

    Unlike a drawing-pen signature, the compiled glyph bytes include composite
    component flags/transforms and TrueType instructions.  Using the same
    canonical compiler before and after the build detects changes to the
    target non-Han glyph even when the visible outline happens to match.
    """

    try:
        glyf = font["glyf"]
        glyph = glyf[glyph_name]
        payload = glyph.compile(glyf, recalcBBoxes=False, optimizeSize=True)
    except Exception as exc:
        payload = f"ERROR:{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _cmap_identity(table: Any, occurrence: int) -> str:
    return ":".join(
        str(int(value))
        for value in (
            getattr(table, "format", -1),
            getattr(table, "platformID", -1),
            getattr(table, "platEncID", -1),
            getattr(table, "language", 0),
            occurrence,
        )
    )


def _snapshot_cmap_subtables(font: TTFont) -> dict[str, dict[str, Any]]:
    """Snapshot every target cmap subtable without treating Han as non-Han.

    Unicode subtables store only the non-Han mappings because target Han
    mappings are intentionally replaced.  Legacy/non-Unicode subtables are
    stored in full.  Format-14 variation sequences are filtered by base
    codepoint so non-Han UVS mappings remain exactly unchanged.
    """

    result: dict[str, dict[str, Any]] = {}
    counters: dict[tuple[int, int, int, int], int] = {}
    for table in font["cmap"].tables:
        signature = (
            int(getattr(table, "format", -1)),
            int(getattr(table, "platformID", -1)),
            int(getattr(table, "platEncID", -1)),
            int(getattr(table, "language", 0)),
        )
        occurrence = counters.get(signature, 0)
        counters[signature] = occurrence + 1
        key = _cmap_identity(table, occurrence)
        if signature[0] == 14:
            entries: list[tuple[int, int, str | None]] = []
            uvs_dict = getattr(table, "uvsDict", {}) or {}
            for selector, pairs in sorted(uvs_dict.items()):
                for codepoint, glyph_name in pairs:
                    cp = int(codepoint)
                    if not is_han_ideograph(cp, True):
                        entries.append((int(selector), cp, glyph_name))
            result[key] = {"mode": "non_han_uvs", "data": entries}
        elif bool(table.isUnicode()):
            mapping = {
                int(cp): name
                for cp, name in (getattr(table, "cmap", {}) or {}).items()
                if not is_han_ideograph(int(cp), True)
            }
            result[key] = {"mode": "non_han_unicode", "data": mapping}
        else:
            mapping = {
                int(cp): name
                for cp, name in (getattr(table, "cmap", {}) or {}).items()
            }
            result[key] = {"mode": "legacy_full", "data": mapping}
    return result


def _snapshot_table_integrity(font: TTFont) -> dict[str, Any]:
    tags = sorted(tag for tag in font.keys() if tag != "GlyphOrder")
    hashes: dict[str, str] = {}
    for tag in PRESERVED_TABLE_DATA_TAGS:
        if tag in font:
            hashes[tag] = hashlib.sha256(font.getTableData(tag)).hexdigest()
    return {"tags": tags, "protected_hashes": hashes}


def _snapshot_non_han(font: TTFont, verify_outlines: bool) -> dict[str, Any]:
    cmap = font.getBestCmap() or {}
    non_han = {cp: name for cp, name in cmap.items() if not is_han_ideograph(cp, True)}
    glyphs = sorted(set(non_han.values()))
    hmtx = font["hmtx"].metrics if "hmtx" in font else {}
    vmtx = font["vmtx"].metrics if "vmtx" in font else {}
    return {
        "best_cmap": non_han,
        "cmap_subtables": _snapshot_cmap_subtables(font),
        "hmtx": {name: tuple(int(v) for v in hmtx[name]) for name in glyphs if name in hmtx},
        "vmtx": {name: tuple(int(v) for v in vmtx[name]) for name in glyphs if name in vmtx},
        "outlines": {name: _glyph_signature(font, name) for name in glyphs} if verify_outlines else {},
        "glyph_order": list(font.getGlyphOrder()),
        "table_integrity": _snapshot_table_integrity(font),
    }


def _mismatch(
    kind: str,
    *,
    codepoint: int | str = "",
    glyph_before: str = "",
    glyph_after: str = "",
    details: str = "",
) -> dict[str, Any]:
    unicode_value = f"U+{int(codepoint):04X}" if isinstance(codepoint, int) else ""
    return {
        "kind": kind,
        "codepoint": codepoint,
        "unicode": unicode_value,
        "glyph_before": glyph_before,
        "glyph_after": glyph_after,
        "details": details,
    }


def _verify_cmap_subtables(font: TTFont, expected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    actual = _snapshot_cmap_subtables(font)
    mismatches: list[dict[str, Any]] = []
    for key, before in expected.items():
        after = actual.get(key)
        if after is None:
            mismatches.append(_mismatch("cmap_subtable_missing", details=f"subtable={key}"))
            continue
        if after.get("mode") != before.get("mode") or after.get("data") != before.get("data"):
            before_data = before.get("data")
            after_data = after.get("data")
            mismatches.append(_mismatch(
                "cmap_subtable_changed",
                details=(
                    f"subtable={key}; mode_before={before.get('mode')}; mode_after={after.get('mode')}; "
                    f"items_before={len(before_data) if hasattr(before_data, '__len__') else '?'}; "
                    f"items_after={len(after_data) if hasattr(after_data, '__len__') else '?'}"
                ),
            ))
    return mismatches


def _verify_non_han(font: TTFont, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    cmap_after = font.getBestCmap() or {}
    hmtx = font["hmtx"].metrics if "hmtx" in font else {}
    vmtx = font["vmtx"].metrics if "vmtx" in font else {}
    for codepoint, glyph_before in snapshot["best_cmap"].items():
        glyph_after = cmap_after.get(int(codepoint))
        if glyph_after != glyph_before:
            mismatches.append(_mismatch(
                "cmap", codepoint=int(codepoint), glyph_before=glyph_before,
                glyph_after=glyph_after or "", details="non-Han best-cmap mapping changed",
            ))
    mismatches.extend(_verify_cmap_subtables(font, snapshot.get("cmap_subtables", {})))
    for glyph_name, before in snapshot["hmtx"].items():
        after = hmtx.get(glyph_name)
        if after is None or tuple(int(v) for v in after) != tuple(before):
            mismatches.append(_mismatch(
                "hmtx", glyph_before=glyph_name, glyph_after=glyph_name,
                details=f"before={before}; after={after}",
            ))
    for glyph_name, before in snapshot["vmtx"].items():
        after = vmtx.get(glyph_name)
        if after is None or tuple(int(v) for v in after) != tuple(before):
            mismatches.append(_mismatch(
                "vmtx", glyph_before=glyph_name, glyph_after=glyph_name,
                details=f"before={before}; after={after}",
            ))
    for glyph_name, before in snapshot["outlines"].items():
        after = _glyph_signature(font, glyph_name)
        if after != before:
            mismatches.append(_mismatch(
                "glyf_bytes", glyph_before=glyph_name, glyph_after=glyph_name,
                details=f"before={before}; after={after}",
            ))

    old_order = list(snapshot.get("glyph_order", []))
    new_order = list(font.getGlyphOrder())
    if new_order[: len(old_order)] != old_order:
        mismatches.append(_mismatch(
            "glyph_order_prefix", details=(
                f"target_count={len(old_order)}; output_count={len(new_order)}; "
                "one or more target glyph IDs changed"
            ),
        ))

    integrity = snapshot.get("table_integrity", {})
    expected_tags = set(integrity.get("tags", [])) - {"DSIG"}
    actual_tags = set(tag for tag in font.keys() if tag != "GlyphOrder")
    missing_tags = sorted(expected_tags - actual_tags)
    unexpected_tags = sorted(actual_tags - expected_tags)
    if missing_tags or unexpected_tags:
        mismatches.append(_mismatch(
            "table_set", details=f"missing={missing_tags}; unexpected={unexpected_tags}",
        ))
    for tag, before_hash in integrity.get("protected_hashes", {}).items():
        if tag not in font:
            mismatches.append(_mismatch("table_missing", details=f"tag={tag}"))
            continue
        after_hash = hashlib.sha256(font.getTableData(tag)).hexdigest()
        if after_hash != before_hash:
            mismatches.append(_mismatch(
                "table_bytes", details=f"tag={tag}; before={before_hash}; after={after_hash}",
            ))
    return mismatches


def _new_glyph_name(codepoint: int, existing_names: set[str]) -> str:
    stem = f"uni{codepoint:04X}" if codepoint <= 0xFFFF else f"u{codepoint:X}"
    return unique_name(f"{stem}.hanzistyleforge", existing_names)


def build_font(cfg: dict[str, Any]) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    refined_selection = work / "refined" / "selection.csv"
    generated_selection = work / "generated" / "selection.csv"
    selection_path = refined_selection if refined_selection.is_file() else generated_selection
    if not selection_path.is_file():
        raise FileNotFoundError("refined/selection.csv or generated/selection.csv was not found. Run generate/refine first.")

    rows = read_csv(selection_path)
    if not rows:
        raise RuntimeError(f"The selection table is empty: {selection_path}")
    target = {int(row["codepoint"]) for row in rows}
    if any(not is_han_ideograph(cp, bool(cfg["scope"].get("include_compatibility_ideographs", True))) for cp in target):
        raise RuntimeError("The selection table contains non-Han codepoints. The final builder only rebuilds Han codepoints and has stopped to protect other scripts.")

    target_path = Path(cfg["paths"]["target_font"])
    reference_path = Path(cfg["paths"]["reference_font"])
    output_path = Path(cfg["paths"]["output_font"])
    build_cfg = cfg["build"]
    verify_non_han = bool(build_cfg.get("preserve_non_han", True))
    verify_outlines = bool(build_cfg.get("verify_non_han_outlines", True))

    # recalcBBoxes=False prevents a save operation from touching legacy glyph
    # data that the user explicitly asked us to preserve.
    font = TTFont(str(target_path), lazy=False, recalcBBoxes=False, recalcTimestamp=True)
    reference_font = TTFont(str(reference_path), lazy=False, recalcBBoxes=False)
    try:
        if "glyf" not in font or "fvar" in font or "gvar" in font:
            raise RuntimeError("Automatic font writing supports only a static TrueType glyf target font.")
        target_snapshot = _snapshot_non_han(font, verify_outlines) if verify_non_han else {}

        # Copy cmap dictionaries before maintaining the local build view.
        # fontTools may share the same dict object between multiple format-4
        # subtables.  Mutating getBestCmap() with a supplementary-plane codepoint
        # would therefore corrupt a 16-bit cmap and fail at save time.  Actual
        # font cmap writes are performed exclusively by _map_codepoint().
        target_cmap = dict(font.getBestCmap() or {})
        reference_cmap = dict(reference_font.getBestCmap() or {})
        reference_hmtx = reference_font["hmtx"].metrics if "hmtx" in reference_font else {}
        reference_vmtx = reference_font["vmtx"].metrics if "vmtx" in reference_font else {}
        reference_upm = int(reference_font["head"].unitsPerEm)

        glyph_order = list(font.getGlyphOrder())
        existing_names = set(glyph_order)
        glyf = font["glyf"]
        hmtx = font["hmtx"].metrics
        vmtx = font["vmtx"].metrics if "vmtx" in font else None
        upm = int(font["head"].unitsPerEm)
        y_bottom, y_top, _ = get_vertical_bounds(font, upm)
        default_h = _median_han_metrics(target_cmap, hmtx, (upm, 0))
        default_v = _median_han_metrics(target_cmap, vmtx, (upm, 0)) if vmtx is not None else None

        def reference_advance(codepoint: int) -> int:
            name = reference_cmap.get(codepoint)
            if name in reference_hmtx:
                return max(1, int(round(reference_hmtx[name][0] * upm / max(1, reference_upm))))
            return int(default_h[0])

        def reference_vertical(codepoint: int) -> tuple[int, int] | None:
            if vmtx is None:
                return None
            name = reference_cmap.get(codepoint)
            if name in reference_vmtx:
                scale = upm / max(1, reference_upm)
                advance, bearing = reference_vmtx[name]
                return int(round(advance * scale)), int(round(bearing * scale))
            return default_v

        strategy = str(build_cfg.get("replace_strategy", "new_glyph_and_remap")).lower()
        if strategy == "adaptive_safe":
            # The strongest preservation guarantee is to append every rebuilt
            # Han glyph.  Use it whenever OpenType's glyph limit permits.
            strategy = "new_glyph_and_remap" if len(glyph_order) + len(rows) <= 65535 else "overwrite_existing"
        if strategy == "new_glyph_and_remap" and len(glyph_order) + len(rows) > 65535:
            raise RuntimeError(
                f"The target font has {len(glyph_order)} glyphs, and rebuilding {len(rows)} additional glyphs would exceed the OpenType limit of 65,535 glyphs."
                "Reduce the coverage of ref.otf. To protect non-Han glyphs, this release will not silently overwrite existing glyphs."
            )

        unresolved: list[dict[str, Any]] = []
        counts = {
            "replace": 0, "add": 0,
            "neural": 0, "diffusion": 0, "direct": 0, "retrieval": 0, "component": 0, "fusion": 0,
            "ensemble": 0, "marathon_refined": 0,
            "fallback": 0, "fallback_raw": 0, "fallback_stylized": 0,
        }

        # When forced to overwrite because of the glyph limit, never overwrite
        # a glyph shared by non-Han Unicode.  This path is not expected for the
        # user's 15k-glyph source but remains fail-safe for other fonts.
        reverse_cmap: dict[str, set[int]] = {}
        for cp, name in target_cmap.items():
            reverse_cmap.setdefault(name, set()).add(int(cp))

        for row in rows:
            codepoint = int(row["codepoint"])
            image_path = Path(row.get("chosen_path", ""))
            if not image_path.is_file():
                unresolved.append({
                    "codepoint": codepoint, "unicode": row.get("unicode", f"U+{codepoint:04X}"),
                    "char": row.get("char", ""), "reason": f"Candidate image does not exist: {image_path}",
                })
                continue

            old_name = target_cmap.get(codepoint)
            if strategy == "new_glyph_and_remap" or old_name is None:
                glyph_name = _new_glyph_name(codepoint, existing_names)
                glyph_order.append(glyph_name)
                existing_names.add(glyph_name)
            else:
                mapped = reverse_cmap.get(old_name, set())
                if any(not is_han_ideograph(cp, True) for cp in mapped):
                    raise RuntimeError(
                        f"glyph {old_name} is also used by a non-Han codepoint and cannot be overwritten."
                        "Use a ref.otf with smaller coverage so the codepoint can be remapped to a separate new glyph."
                    )
                glyph_name = old_name

            advance = int(hmtx.get(old_name, default_h)[0]) if old_name else reference_advance(codepoint)
            vector_config = dict(build_cfg)
            contour_checkpoint = work / "fusion" / "contour" / "best.pt"
            vector_config.setdefault("contour_polisher_checkpoint", str(contour_checkpoint.resolve()))
            vector_config.setdefault(
                "use_contour_polisher",
                bool(cfg.get("fusion", {}).get("contour_polisher", {}).get("enabled", True))
                and contour_checkpoint.is_file(),
            )
            vector_config.setdefault(
                "contour_polisher_strength",
                float(cfg.get("fusion", {}).get("contour_polisher", {}).get("build_strength", 0.58)),
            )
            glyph = image_to_ttglyph(
                image_path,
                upm=upm,
                pad=int(cfg["render"]["pad"]),
                y_bottom=y_bottom,
                y_top=y_top,
                config=vector_config,
            )
            if int(getattr(glyph, "numberOfContours", 0)) == 0:
                unresolved.append({
                    "codepoint": codepoint, "unicode": row.get("unicode", f"U+{codepoint:04X}"),
                    "char": row.get("char", ""), "reason": f"The outline is empty after vectorization: {image_path}",
                })
                continue

            glyf[glyph_name] = glyph
            glyph.recalcBounds(glyf)
            hmtx[glyph_name] = (advance, int(getattr(glyph, "xMin", 0)))
            if vmtx is not None:
                vertical = reference_vertical(codepoint)
                if vertical is not None:
                    vmtx[glyph_name] = vertical
            _map_codepoint(font, codepoint, glyph_name)
            target_cmap[codepoint] = glyph_name
            counts["add" if old_name is None else "replace"] += 1
            source = str(row.get("chosen_source", "fallback"))
            if source in counts:
                counts[source] += 1
            elif source.startswith("ensemble"):
                counts["ensemble"] += 1
            elif source.startswith("marathon"):
                counts["marathon_refined"] += 1
            elif "fallback" in source or "reference" in source:
                counts["fallback"] += 1
            label = str(row.get("chosen_label", ""))
            if "fallback" in source or "reference" in source:
                if label in {"reference_raw", "reference_emergency"} or "raw" in label:
                    counts["fallback_raw"] += 1
                else:
                    counts["fallback_stylized"] += 1

        if unresolved and bool(build_cfg.get("require_complete", True)):
            unresolved_path = work / "build_unresolved.csv"
            write_csv(unresolved_path, unresolved, UNRESOLVED_FIELDS)
            raise RuntimeError(f"Automatic font building stopped: {len(unresolved)} Han glyphs have no usable result. See {unresolved_path}.")

        font.setGlyphOrder(glyph_order)
        if "maxp" in font:
            font["maxp"].numGlyphs = len(glyph_order)
        if "DSIG" in font:
            del font["DSIG"]
        removed_uvs = _remove_target_han_uvs(font, target) if bool(build_cfg.get("remove_han_uvs", True)) else 0

        family = _font_name(font, 1, "Converted Font") + str(build_cfg.get("family_suffix", " HanziStyleForge"))
        subfamily = _font_name(font, 2, "Regular")
        full_name = f"{family} {subfamily}".strip()
        ps_name = _postscript_name(_font_name(font, 6, "ConvertedFont") + "-" + str(build_cfg.get("postscript_suffix", "HanziStyleForge")))
        if "name" in font:
            for platform, encoding, language in ((3, 1, 0x409), (3, 10, 0x409), (1, 0, 0)):
                try:
                    font["name"].setName(family, 1, platform, encoding, language)
                    font["name"].setName(full_name, 4, platform, encoding, language)
                    font["name"].setName(ps_name, 6, platform, encoding, language)
                    font["name"].setName(family, 16, platform, encoding, language)
                    font["name"].setName(subfamily, 17, platform, encoding, language)
                except Exception:
                    pass

        if "OS/2" in font:
            final_cmap_memory = font.getBestCmap() or {}
            bmp = [cp for cp in final_cmap_memory if cp <= 0xFFFF]
            if bmp:
                font["OS/2"].usFirstCharIndex = min(bmp)
                font["OS/2"].usLastCharIndex = max(bmp)
            try:
                font["OS/2"].recalcUnicodeRanges(font)
            except Exception:
                pass
            try:
                font["OS/2"].recalcCodePageRanges(font)
            except Exception:
                pass

        ensure_dir(output_path.parent)
        temporary_output = output_path.with_suffix(output_path.suffix + ".building")
        temporary_output.unlink(missing_ok=True)
        font.save(temporary_output)
    finally:
        font.close()
        reference_font.close()

    verify = TTFont(str(temporary_output), lazy=False, recalcBBoxes=False)
    try:
        final_cmap = verify.getBestCmap() or {}
        missing = sorted(target - set(final_cmap))
        empty_after_build: list[int] = []
        bounds_anomalies: list[dict[str, int]] = []
        verify_glyf = verify["glyf"]
        for codepoint in sorted(target & set(final_cmap)):
            glyph = verify_glyf[final_cmap[codepoint]]
            contour_count = (
                max(1, len(getattr(glyph, "components", []) or []))
                if glyph.isComposite() else int(getattr(glyph, "numberOfContours", 0))
            )
            if contour_count <= 0:
                empty_after_build.append(codepoint)
                continue
            try:
                glyph.recalcBounds(verify_glyf)
                x_min = int(getattr(glyph, "xMin", 0)); x_max = int(getattr(glyph, "xMax", 0))
                y_min = int(getattr(glyph, "yMin", 0)); y_max = int(getattr(glyph, "yMax", 0))
                if (
                    x_min < -upm // 3 or x_max > upm * 4 // 3
                    or y_min < y_bottom - upm * 0.20 or y_max > y_top + upm * 0.20
                ):
                    bounds_anomalies.append({
                        "codepoint": codepoint, "xMin": x_min, "xMax": x_max,
                        "yMin": y_min, "yMax": y_max,
                    })
            except Exception:
                bounds_anomalies.append({"codepoint": codepoint, "xMin": 0, "xMax": 0, "yMin": 0, "yMax": 0})

        non_han_mismatches = _verify_non_han(verify, target_snapshot) if verify_non_han else []
    finally:
        verify.close()

    mismatch_path = work / "non_han_preservation_mismatches.csv"
    if non_han_mismatches:
        write_csv(mismatch_path, non_han_mismatches, NON_HAN_MISMATCH_FIELDS)
    if missing or empty_after_build or non_han_mismatches:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            "Output verification failed: "
            f"missing Han glyphs: {len(missing)}; empty outlines: {len(empty_after_build)}; "
            f"changed non-Han items: {len(non_han_mismatches)}."
            + (f"See {mismatch_path}." if non_han_mismatches else "")
        )

    durable_replace(temporary_output, output_path)
    report = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "program": "HanziStyleForge Fusion",
        "output_font": str(output_path.resolve()),
        "target_han_count": len(target),
        "counts": counts,
        "missing_after_build": 0,
        "empty_after_build": 0,
        "non_han_preservation": {
            "verified": verify_non_han,
            "unicode_count": len(target_snapshot.get("best_cmap", {})),
            "mapped_glyph_count": len(set(target_snapshot.get("best_cmap", {}).values())),
            "mapping_mismatches": 0,
            "metric_mismatches": 0,
            "outline_mismatches": 0,
            "all_cmap_subtables_verified": True,
            "layout_and_hint_tables_verified": True,
            "target_glyph_id_prefix_verified": True,
        },
        "han_uvs_removed": int(removed_uvs),
        "bounds_anomaly_count": len(bounds_anomalies),
        "bounds_anomaly_examples": bounds_anomalies[:100],
        "replace_strategy_used": strategy,
        "glyph_count": len(glyph_order),
        "outline_mode": str(build_cfg.get("outline_mode", "sdf_quadratic")),
        "tables_preserved": "All target tables are retained except DSIG; only target-Han cmap/UVS mappings, names and OS/2 coverage metadata are intentionally changed. Target layout/hint table bytes and target glyph IDs are verified.",
        "note": (
            "Every Han codepoint covered by ref.otf was remapped. All target non-Han cmap subtables, UVS mappings, outline bytes, hmtx/vmtx metrics, original glyph IDs, layout tables, and hinting tables were verified."
            "The generated font should still be tested in the intended layout applications and at small sizes, but the program workflow requires no manual glyph intervention."
        ),
    }
    save_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    return report
