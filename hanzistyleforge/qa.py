from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .image_cache import read_gray_u8
from .util import atomic_save_pil, atomic_write_text, ensure_dir, load_json, read_csv, save_json, write_csv


def _load(path: str, size: int) -> Image.Image:
    p = Path(path) if path else None
    if p is None or not p.is_file():
        image = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(image)
        draw.line((8, 8, size - 8, size - 8), fill=180, width=2)
        draw.line((size - 8, 8, 8, size - 8), fill=180, width=2)
        return image
    try:
        return Image.fromarray(read_gray_u8(p), mode="L").resize((size, size), Image.Resampling.LANCZOS)
    except Exception:
        return Image.new("L", (size, size), 255)


def _make_sheet(rows: list[dict[str, str]], output: Path, *, cell: int = 112, per_page: int = 20) -> list[str]:
    pages: list[str] = []
    for page_index in range(0, len(rows), per_page):
        page_rows = rows[page_index : page_index + per_page]
        width = cell * 3 + 360
        height = max(1, len(page_rows)) * (cell + 24)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, row in enumerate(page_rows):
            y = row_index * (cell + 24)
            cn = _load(row.get("ref_path", ""), cell)
            target = _load(row.get("target_path", ""), cell)
            chosen = _load(row.get("chosen_path", ""), cell)
            canvas.paste(cn.convert("RGB"), (0, y))
            canvas.paste(target.convert("RGB"), (cell, y))
            canvas.paste(chosen.convert("RGB"), (cell * 2, y))
            text = (
                f"{row.get('unicode', '')} {row.get('char', '')}  source={row.get('chosen_source', '')}\n"
                f"structure={row.get('chosen_structure_score', '')}  topology={row.get('chosen_topology_score', '')}\n"
                f"pass={row.get('chosen_topology_pass', '')} confidence={row.get('chosen_confidence', '')}"
            )
            draw.multiline_text((cell * 3 + 10, y + 10), text, fill="black", spacing=5)
            draw.text((4, y + cell + 2), "Ref", fill="black")
            draw.text((cell + 4, y + cell + 2), "Target", fill="black")
            draw.text((cell * 2 + 4, y + cell + 2), "Chosen", fill="black")
        name = f"qa_{page_index // per_page + 1:03d}.png"
        atomic_save_pil(canvas, output / name)
        pages.append(name)
    return pages


def make_qa_report(cfg: dict[str, Any]) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    refined_selection = work / "refined" / "selection.csv"
    selection = refined_selection if refined_selection.exists() else work / "generated" / "selection.csv"
    if not selection.exists():
        raise FileNotFoundError("refined/selection.csv or generated/selection.csv was not found. Run generate/refine first.")
    rows = read_csv(selection)
    qa_dir = ensure_dir(work / "qa")
    benchmark_path = work / "benchmark" / "summary.json"
    benchmark = load_json(benchmark_path) if benchmark_path.exists() else {}
    audit_path = work / "audit" / "summary.json"
    audit = load_json(audit_path) if audit_path.exists() else {}
    failures = [row for row in rows if str(row.get("chosen_topology_pass", "0")) != "1"]
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if str(row.get("chosen_topology_pass", "0")) != "1" else 1,
            float(row.get("chosen_confidence", 0.0) or 0.0),
            -float(row.get("chosen_topology_score", 0.0) or 0.0),
        ),
    )
    sample_limit = int(cfg.get("qa", {}).get("contact_sheet_count", 200))
    pages = _make_sheet(ranked[:sample_limit], qa_dir)
    sources: dict[str, int] = {}
    labels: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}
    low_confidence: list[dict[str, str]] = []
    for row in rows:
        source = row.get("chosen_source", "unknown")
        sources[source] = sources.get(source, 0) + 1
        label = row.get("chosen_label", "")
        labels[label] = labels.get(label, 0) + 1
        if float(row.get("chosen_confidence", 0.0) or 0.0) < 0.75:
            low_confidence.append(row)
        raw_reasons = str(row.get("rejection_reasons", "") or "")
        for reason in ("component_delta", "hole_delta", "endpoint_delta", "junction_delta", "missing_skeleton", "extra_skeleton", "hole_position", "zone_structure", "euler_delta", "topology_score"):
            if reason in raw_reasons:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    summary = {
        "target_count": len(rows),
        "topology_failure_count": len(failures),
        "topology_pass_rate": (len(rows) - len(failures)) / max(1, len(rows)),
        "sources": sources,
        "chosen_labels": labels,
        "raw_ref_fallback_count": (labels.get("reference_raw", 0) + labels.get("reference_emergency", 0)),
        "fallback_rate": sources.get("fallback", 0) / max(1, len(rows)),
        "low_confidence_count": len(low_confidence),
        "rejection_reasons": rejection_reasons,
        "benchmark": benchmark,
        "training_coverage": {
            "target_han_count": audit.get("target_han_count", 0),
            "style_trainable_count": audit.get("style_trainable_count", 0),
            "style_training_coverage": audit.get("style_training_coverage", 0.0),
        },
        "contact_sheet_pages": pages,
    }
    if failures:
        write_csv(qa_dir / "topology_failures.csv", failures, list(failures[0].keys()))
    if low_confidence:
        write_csv(qa_dir / "low_confidence.csv", low_confidence, list(low_confidence[0].keys()))
    save_json(qa_dir / "summary.json", summary)
    links = "\n".join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in pages)
    atomic_write_text(
        qa_dir / "index.html",
        "<!doctype html><meta charset='utf-8'><title>HanziStyleForge QA</title>"
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:960px;margin:30px auto}code{background:#eee;padding:2px 5px}</style>"
        f"<h1>HanziStyleForge automatic QA report</h1><p>Target glyphs: {len(rows)}; topology failures: {len(failures)}; pass rate: {summary['topology_pass_rate']:.2%}</p>"
        f"<p>Candidate sources: <code>{html.escape(str(sources))}</code></p>"
        f"<p>Raw-reference safe fallbacks: <code>{(labels.get('reference_raw', 0) + labels.get('reference_emergency', 0))}</code>; low-confidence glyphs: <code>{len(low_confidence)}</code></p>"
        f"<p>Style-training coverage: <code>{float(audit.get('style_training_coverage', 0.0)):.2%}</code>; "
        f"fixed-validation Dice: <code>{float(benchmark.get('dice', {}).get('mean', 0.0)):.4f}</code></p>"
        f"<p>Main rejection reasons: <code>{html.escape(str(rejection_reasons))}</code></p><ul>{links}</ul>",
        encoding="utf-8",
    )
    return summary
