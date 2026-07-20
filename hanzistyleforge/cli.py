from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from hanzistyleforge import __version__
from hanzistyleforge.analysis import check_environment, prepare_project
from hanzistyleforge.benchmark import run_benchmark
from hanzistyleforge.build_font import build_font
from hanzistyleforge.config import load_config
from hanzistyleforge.contract import validate_data_flow_contract
from hanzistyleforge.ensemble import run_ensemble_training
from hanzistyleforge.inference import generate_and_select
from hanzistyleforge.fusion_training import train_fusion_all
from hanzistyleforge.fusion_inference import generate_fusion_and_select
from hanzistyleforge.ids_data import install_cjkvi_ids
from hanzistyleforge.longrun import SafeStopRequested
from hanzistyleforge.marathon import marathon_status, run_marathon_training
from hanzistyleforge.marathon_refine import run_marathon_refinement
from hanzistyleforge.project_lock import ProjectLock
from hanzistyleforge.qa import make_qa_report
from hanzistyleforge.retrieval import build_style_atlas
from hanzistyleforge.runtime import configure_runtime
from hanzistyleforge.training import train_all, train_generator, train_refiner
from hanzistyleforge.util import save_json


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def command_check(cfg: dict[str, Any]) -> dict[str, Any]:
    result = check_environment(cfg)
    work = Path(cfg["paths"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    save_json(work / "environment.json", result)
    _print(result)
    return result


def command_prepare(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    result = prepare_project(cfg, force=force)
    _print(result)
    print(
        "\nThe training set contains only real target.ttf Han glyph self-reconstruction samples. "
        "The program does not compare target.ttf and ref.otf for classification and does not require a manually curated equivalent-glyph list.\n"
        "ref.otf provides only the Han glyph structures and character coverage to rebuild. Non-Han characters are never reconstruction targets."
    )
    return result


def command_atlas(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    result = build_style_atlas(cfg, force=force)
    _print(result)
    return result


def command_train(cfg: dict[str, Any]) -> dict[str, Any]:
    result = train_all(cfg)
    _print(result)
    return result


def command_marathon(cfg: dict[str, Any]) -> dict[str, Any]:
    result = run_marathon_training(cfg)
    _print(result)
    return result


def command_ensemble(cfg: dict[str, Any]) -> dict[str, Any]:
    result = run_ensemble_training(cfg)
    _print(result)
    return result


def command_generate(cfg: dict[str, Any]) -> dict[str, Any]:
    result = generate_and_select(cfg)
    _print(result)
    return result


def command_refine(cfg: dict[str, Any]) -> dict[str, Any]:
    result = run_marathon_refinement(cfg)
    _print(result)
    return result


def command_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    result = run_benchmark(cfg)
    _print(result)
    return result


def command_qa(cfg: dict[str, Any]) -> dict[str, Any]:
    result = make_qa_report(cfg)
    _print(result)
    return result


def command_build(cfg: dict[str, Any]) -> dict[str, Any]:
    result = build_font(cfg)
    _print(result)
    return result


def command_status(cfg: dict[str, Any]) -> dict[str, Any]:
    result = marathon_status(cfg)
    _print(result)
    return result


def command_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    result = validate_data_flow_contract(cfg, require_prepared=True, write_report=True)
    _print(result)
    return result


def command_ids_install(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    component_cfg = cfg.get("fusion", {}).get("component_atlas", {})
    result = install_cjkvi_ids(
        component_cfg.get("decomposition_file", "data/cjkvi-ids/ids.txt"),
        force=force,
        url=str(component_cfg.get("source_url", "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/86b4d16159f0079437870408f0ca186e529015db/ids.txt")),
        expected_sha256=str(component_cfg.get("source_sha256", "bfc70a8c09f9f5616ebf0543bd6681e67314e9f7ae2307e5ae8c6f15bdc5c6a6")),
    )
    _print(result)
    print(
        "\nThe IDS file was downloaded directly from cjkvi/cjkvi-ids and is not "
        "redistributed as part of HanziStyleForge. Review the upstream CHISE/CJKVI "
        "license terms before redistributing the downloaded file."
    )
    return result


def command_auto_months(cfg: dict[str, Any], force: bool = False) -> None:
    command_check(cfg)
    command_prepare(cfg, force=force)
    command_train(cfg)
    command_marathon(cfg)
    command_ensemble(cfg)
    command_generate(cfg)
    command_refine(cfg)
    command_benchmark(cfg)
    command_qa(cfg)
    command_build(cfg)
    work = Path(cfg["paths"]["work_dir"])
    print(
        f"\nHanziStyleForge completed: {Path(cfg['paths']['output_font']).resolve()}"
        f"\nAutomatic QA: {(work / 'qa' / 'index.html').resolve()}"
        f"\nFixed validation: {(work / 'benchmark' / 'index.html').resolve()}"
        f"\nPer-glyph selection: {(work / 'refined' / 'selection.csv').resolve()}"
    )


def command_fusion_train(cfg: dict[str, Any]) -> dict[str, Any]:
    result = train_fusion_all(cfg)
    _print(result)
    return result


def command_fusion_generate(cfg: dict[str, Any]) -> dict[str, Any]:
    result = generate_fusion_and_select(cfg)
    _print(result)
    return result


def command_fusion_status(cfg: dict[str, Any]) -> dict[str, Any]:
    work = Path(cfg["paths"]["work_dir"])
    def status(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"exists": False}
        if path.suffix.lower() == ".json":
            try:
                return {"exists": True, "data": json.loads(path.read_text(encoding="utf-8"))}
            except Exception as exc:
                return {"exists": True, "error": f"{type(exc).__name__}: {exc}"}
        return {"exists": True, "bytes": path.stat().st_size}
    result = {
        "work_dir": str(work.resolve()),
        "training": status(work / "fusion" / "summary.json"),
        "style": status(work / "fusion" / "style" / "summary.json"),
        "vq": status(work / "fusion" / "vq" / "summary.json"),
        "direct": status(work / "fusion" / "direct" / "summary.json"),
        "diffusion": status(work / "fusion" / "diffusion" / "summary.json"),
        "purification": status(work / "fusion" / "diffusion" / "purification"),
        "refiner": status(work / "fusion" / "refiner" / "summary.json"),
        "component_atlas": status(work / "component_atlas" / "summary.json"),
        "contour": status(work / "fusion" / "contour" / "summary.json"),
        "generation": status(work / "generated" / "generation.state.json"),
        "generation_summary": status(work / "generated" / "summary.json"),
        "coverage": status(work / "generated" / "coverage.json"),
        "refinement": status(work / "refined" / "progress.json"),
        "refinement_summary": status(work / "refined" / "summary.json"),
        "qa": status(work / "qa" / "summary.json"),
        "output_font": status(Path(cfg["paths"]["output_font"])),
    }
    _print(result)
    return result


def command_fusion_auto_months(cfg: dict[str, Any], force: bool = False) -> None:
    command_check(cfg)
    command_prepare(cfg, force=force)
    command_atlas(cfg, force=force)
    command_fusion_train(cfg)
    command_fusion_generate(cfg)
    # Existing long-run pixel/SDF search remains useful as a post-selection
    # purifier and is fully resumable per glyph. It never becomes training data.
    command_refine(cfg)
    command_qa(cfg)
    command_build(cfg)
    work = Path(cfg["paths"]["work_dir"])
    print(
        f"\nHanziStyleForge Fusion completed: {Path(cfg['paths']['output_font']).resolve()}"
        f"\nCoverage: {(work / 'generated' / 'coverage.json').resolve()}"
        f"\nQA: {(work / 'qa' / 'index.html').resolve()}"
        f"\nPer-glyph selection: {(work / 'refined' / 'selection.csv').resolve()}"
    )


def command_clean(cfg: dict[str, Any], keep_render_cache: bool) -> None:
    work = Path(cfg["paths"]["work_dir"])
    if not work.exists():
        print("The work directory does not exist; nothing to clean.")
        return
    if not keep_render_cache:
        shutil.rmtree(work)
        print(f"Deleted: {work}")
        return
    preserve = {"cache", "target_render", "target_proxy", "target_aux", "ref_render", "ref_proxy"}
    for child in work.iterdir():
        if child.name not in preserve:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    print("Render and structural-proxy caches were preserved; training, generation, and build state were removed.")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HanziStyleForge learns Han style only from target.ttf, uses ref.otf for Han structure, "
            "regenerates every Han glyph covered by ref, and verifies preservation of target non-Han content."
        )
    )
    parser.add_argument("--config", default="config_months_12gb.json")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check fonts, PyTorch, CUDA, and target coverage")
    prepare = commands.add_parser("prepare", help="Build the target self-reconstruction training set and reference Han target set")
    prepare.add_argument("--force", action="store_true")
    atlas = commands.add_parser("atlas", help="Build the local real-glyph style atlas")
    atlas.add_argument("--force", action="store_true")
    commands.add_parser("train", help="Progressively train the generator and high-resolution refiner")
    commands.add_parser("train-generator", help="Train the generator only")
    commands.add_parser("train-refiner", help="Train the refiner only")
    commands.add_parser("marathon", help="Run long-term hard-example refinement using only real target glyphs")
    commands.add_parser("ensemble", help="Train and aggregate independent models and strong snapshots")
    commands.add_parser("generate", help="Generate all target Han glyphs from reference structures")
    commands.add_parser("refine", help="Run per-glyph global search and local structural refinement")
    commands.add_parser("benchmark", help="Run the fixed real-target validation set")
    commands.add_parser("qa", help="Generate the automatic QA report")
    commands.add_parser("build", help="Build the TTF and verify complete preservation of target non-Han content")
    commands.add_parser("status", help="Display long-run status without modifying state")
    commands.add_parser("contract", help="Verify the data-flow contract: training reads target only, generation reads ref only")
    ids_install = commands.add_parser("ids-install", help="Download and verify the pinned optional cjkvi/cjkvi-ids data file")
    ids_install.add_argument("--force", action="store_true")
    commands.add_parser("fusion-train", help="Train the style encoder, VQ codebook, latent diffusion model, multi-expert refiner, and contour Transformer")
    commands.add_parser("fusion-generate", help="Generate with multiple models and cover every Han glyph in ref.otf")
    commands.add_parser("fusion-status", help="Display Fusion long-run status without modifying state")
    fusion_auto = commands.add_parser("fusion-auto-months", help="Run the complete resumable Fusion workflow for weeks or months")
    fusion_auto.add_argument("--force", action="store_true")
    auto = commands.add_parser("auto-months", help="Run the legacy deterministic workflow retained as a safety baseline")
    auto.add_argument("--force", action="store_true")
    clean = commands.add_parser("clean", help="Clean the work directory for the current configuration")
    clean.add_argument("--keep-render-cache", action="store_true")
    return parser


def dispatch(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    command = args.command
    if command == "check": command_check(cfg)
    elif command == "prepare": command_prepare(cfg, args.force)
    elif command == "atlas": command_atlas(cfg, args.force)
    elif command == "train": command_train(cfg)
    elif command == "train-generator": _print(train_generator(cfg))
    elif command == "train-refiner": _print(train_refiner(cfg))
    elif command == "marathon": command_marathon(cfg)
    elif command == "ensemble": command_ensemble(cfg)
    elif command == "generate": command_generate(cfg)
    elif command == "refine": command_refine(cfg)
    elif command == "benchmark": command_benchmark(cfg)
    elif command == "qa": command_qa(cfg)
    elif command == "build": command_build(cfg)
    elif command == "status": command_status(cfg)
    elif command == "contract": command_contract(cfg)
    elif command == "ids-install": command_ids_install(cfg, args.force)
    elif command == "fusion-train": command_fusion_train(cfg)
    elif command == "fusion-generate": command_fusion_generate(cfg)
    elif command == "fusion-status": command_fusion_status(cfg)
    elif command == "fusion-auto-months": command_fusion_auto_months(cfg, args.force)
    elif command == "auto-months": command_auto_months(cfg, args.force)
    elif command == "clean": command_clean(cfg, args.keep_render_cache)


def main() -> int:
    args = make_parser().parse_args()
    cfg = load_config(args.config)
    cfg["_runtime"] = configure_runtime(cfg)
    try:
        if args.command in {"check", "status", "fusion-status"}:
            dispatch(args, cfg)
        else:
            with ProjectLock(cfg["paths"]["work_dir"]):
                dispatch(args, cfg)
        return 0
    except SafeStopRequested as exc:
        print(f"\nSafe stop: {exc}\nRun the same launcher again to resume.")
        return 75
    except RuntimeError as exc:
        if str(exc).startswith("STYLE_COLLAPSE_FATAL:"):
            print(f"\n{exc}\nThis is a persistent quality-protection error. Review DIFFUSION_STYLE_COLLAPSE_DETECTED.json; the launcher will not retry automatically.")
            return 76
        raise
    except KeyboardInterrupt:
        print("\nCtrl+C received. The latest completed atomic checkpoint will be restored on the next run.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
