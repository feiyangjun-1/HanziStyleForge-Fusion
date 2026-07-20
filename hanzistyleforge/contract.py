from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import read_csv, save_json


class DataFlowContractError(RuntimeError):
    """Raised when target-style training and reference-content inference are mixed."""


def _inside(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def validate_data_flow_contract(
    cfg: dict[str, Any],
    *,
    require_prepared: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    """Enforce the central project contract.

    Training rows must use only caches rendered from ``fonts/target.ttf``.
    Generation rows must use only content proxies rendered from ``refs/ref.otf``.
    No cross-font target/ref pair is permitted in the training index.
    """

    work = Path(cfg["paths"]["work_dir"])
    dataset_csv = work / "dataset" / "index.csv"
    audit_csv = work / "audit" / "analysis.csv"
    cache = work / "cache"
    target_proxy_root = cache / "target_proxy"
    target_render_root = cache / "target_render"
    target_aux_root = cache / "target_aux"
    ref_proxy_root = cache / "ref_proxy"
    ref_render_root = cache / "ref_render"

    if require_prepared and (not dataset_csv.is_file() or not audit_csv.is_file()):
        raise DataFlowContractError("The project has not been prepared; dataset/index.csv or audit/analysis.csv is missing.")

    errors: list[str] = []
    training_rows = read_csv(dataset_csv) if dataset_csv.is_file() else []
    generation_rows = read_csv(audit_csv) if audit_csv.is_file() else []

    for index, row in enumerate(training_rows, start=2):
        if row.get("mode", "self") != "self":
            errors.append(f"dataset/index.csv row {index} has a mode other than self.")
        proxy_path = row.get("proxy_path", "")
        target_path = row.get("target_path", "")
        aux_path = row.get("target_aux_path", "")
        if not _inside(proxy_path, target_proxy_root):
            errors.append(f"Training row {index} has a proxy_path outside target_proxy: {proxy_path}")
        if not _inside(target_path, target_render_root):
            errors.append(f"Training row {index} has a target_path outside target_render: {target_path}")
        if not _inside(aux_path, target_aux_root):
            errors.append(f"Training row {index} has a target_aux_path outside target_aux: {aux_path}")
        if _inside(proxy_path, ref_proxy_root) or _inside(target_path, ref_render_root):
            errors.append(f"Training row {index} contains reference data.")

    for index, row in enumerate(generation_rows, start=2):
        ref_path = row.get("ref_path", "")
        ref_proxy_path = row.get("ref_proxy_path", "")
        if not _inside(ref_path, ref_render_root):
            errors.append(f"Generation row {index} has a ref_path outside ref_render: {ref_path}")
        if not _inside(ref_proxy_path, ref_proxy_root):
            errors.append(f"Generation row {index} has a ref_proxy_path outside ref_proxy: {ref_proxy_path}")

    report = {
        "contract": "target-style-only training; ref-structure-only generation",
        "training_rows": len(training_rows),
        "generation_rows": len(generation_rows),
        "cross_font_training_pairs": 0 if not errors else None,
        "training_source": str(target_proxy_root.resolve()),
        "training_truth": str(target_render_root.resolve()),
        "generation_content_source": str(ref_proxy_root.resolve()),
        "passed": not errors,
        "errors": errors[:200],
    }
    if write_report:
        (work / "audit").mkdir(parents=True, exist_ok=True)
        save_json(work / "audit" / "data_flow_contract.json", report)
    if errors:
        preview = "\n".join(errors[:12])
        raise DataFlowContractError(
            "Data-flow contract validation failed. Training must use target only, and generation must use ref only.\n" + preview
        )
    return report
