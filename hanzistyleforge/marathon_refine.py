from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .config import CHECKPOINT_FORMAT_VERSION
from .decomposition import component_zones, load_decompositions
from .ids_data import IDSDataError, ensure_decomposition_data
from .longrun import LongRunGuard
from .proxy import (
    affine_mask,
    binary,
    border_ink_ratio,
    glyph_quality_metrics,
    make_reference_fallbacks,
    proxy_from_binary_for_metrics,
    proxy_structure_metrics,
    read_ink,
    read_proxy,
    save_ink,
    style_distance,
)
from .topology import topology_metrics, topology_signature, validate_topology
from .util import cp_filename, ensure_dir, load_json, read_csv, save_json, sha256_file, write_csv


def _profile_for_complexity(profiles: dict[str, Any], complexity: float) -> dict[str, Any]:
    global_profile = profiles.get("global", profiles)
    for item in profiles.get("bins", []):
        if float(item.get("minimum", -math.inf)) <= complexity <= float(item.get("maximum", math.inf)):
            return item.get("profile", global_profile)
    return global_profile


def _morph(mask: np.ndarray, delta: int) -> np.ndarray:
    source = (np.asarray(mask) >= 0.5).astype(np.uint8)
    if delta == 0:
        return source.astype(np.float32)
    radius = min(4, max(1, abs(int(delta))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    result = cv2.dilate(source, kernel) if delta > 0 else cv2.erode(source, kernel)
    return result.astype(np.float32)


def _smooth(mask: np.ndarray, sigma: float, threshold: float) -> np.ndarray:
    if sigma <= 0:
        return (mask >= threshold).astype(np.float32)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return (blurred >= threshold).astype(np.float32)


def _atomic_save_ink(path: Path, mask: np.ndarray) -> None:
    save_ink(path, mask)


def _connect_database(path: Path) -> sqlite3.Connection:
    ensure_dir(path.parent)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS glyph_jobs (
            codepoint INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            score REAL,
            topology_pass INTEGER,
            source TEXT,
            updated_at REAL NOT NULL,
            message TEXT
        )
        """
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def _set_database_fingerprint(connection: sqlite3.Connection, fingerprint: str) -> None:
    row = connection.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
    previous = row[0] if row else ""
    if previous != fingerprint:
        # Config/model/reference changes invalidate every previous per-glyph verdict.
        # Candidate images are overwritten lazily; the database prevents stale reuse.
        connection.execute("DELETE FROM glyph_jobs")
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES('fingerprint',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (fingerprint,),
        )
        connection.commit()


def _job_done(connection: sqlite3.Connection, codepoint: int, output: Path) -> bool:
    row = connection.execute(
        "SELECT status FROM glyph_jobs WHERE codepoint = ?", (int(codepoint),)
    ).fetchone()
    return bool(row and row[0] == "done" and output.is_file())


def _set_job(
    connection: sqlite3.Connection,
    codepoint: int,
    status: str,
    *,
    score: float | None = None,
    topology_pass: bool | None = None,
    source: str = "",
    message: str = "",
) -> None:
    connection.execute(
        """
        INSERT INTO glyph_jobs(codepoint,status,score,topology_pass,source,updated_at,message)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(codepoint) DO UPDATE SET
            status=excluded.status,
            score=excluded.score,
            topology_pass=excluded.topology_pass,
            source=excluded.source,
            updated_at=excluded.updated_at,
            message=excluded.message
        """,
        (
            int(codepoint),
            str(status),
            None if score is None else float(score),
            None if topology_pass is None else int(bool(topology_pass)),
            str(source),
            float(time.time()),
            str(message),
        ),
    )
    connection.commit()


def _candidate_score(
    candidate: np.ndarray,
    reference_ink: np.ndarray,
    reference_proxy: np.ndarray,
    reference_signature: Any,
    profile: dict[str, Any],
    topology_cfg: dict[str, Any],
    analysis_size: int,
) -> tuple[float, dict[str, Any]]:
    mask = (np.asarray(candidate) >= 0.5).astype(np.float32)
    small = cv2.resize(mask, (analysis_size, analysis_size), interpolation=cv2.INTER_NEAREST)
    candidate_proxy = proxy_from_binary_for_metrics(small)
    structure = proxy_structure_metrics(candidate_proxy, reference_proxy, analysis_size=analysis_size)
    topology = topology_metrics(
        reference_ink,
        mask,
        size=int(topology_cfg.get("analysis_size", analysis_size)),
        prune_iterations=int(topology_cfg.get("prune_iterations", 1)),
        reference_signature=reference_signature,
    )
    validation = validate_topology(topology, topology_cfg)
    quality = glyph_quality_metrics(mask)
    style = style_distance(quality, profile)
    border = border_ink_ratio(mask, border=max(2, mask.shape[0] // 128))
    hard_penalty = 0.0 if validation["hard_pass"] else 1.25 + 0.16 * len(validation["reasons"])
    exact_penalty = 0.12 * min(5, abs(int(topology["component_delta"])))
    exact_penalty += 0.14 * min(5, abs(int(topology["hole_delta"])))
    score = (
        float(structure["structure_score"])
        + 0.72 * float(topology["topology_score"])
        + 0.34 * float(style)
        + max(0.0, float(border) - 0.012) * 12.0
        + hard_penalty
        + exact_penalty
    )
    return float(score), {
        "structure": structure,
        "topology": topology,
        "validation": validation,
        "quality": quality,
        "style_score": float(style),
        "border": float(border),
    }


def _global_trial(
    trial: int,
    codepoint: int,
    starts: list[np.ndarray],
    reference_fallback: np.ndarray,
    pass_index: int,
) -> np.ndarray:
    rng = np.random.default_rng((int(codepoint) * 1000003 + int(pass_index) * 10007 + int(trial)) & 0xFFFFFFFF)
    base = starts[int(rng.integers(0, len(starts)))].astype(np.float32)
    scale_range = (0.050, 0.032, 0.018)[min(pass_index, 2)]
    shift_range = (0.018, 0.011, 0.006)[min(pass_index, 2)] * base.shape[0]
    sx = 1.0 + float(rng.uniform(-scale_range, scale_range))
    sy = 1.0 + float(rng.uniform(-scale_range, scale_range))
    dx = float(rng.uniform(-shift_range, shift_range))
    dy = float(rng.uniform(-shift_range, shift_range))
    candidate = affine_mask(base, sx, sy, dx, dy)
    if rng.random() < 0.55:
        alpha = float(rng.uniform(0.03, 0.35 if pass_index == 0 else 0.20))
        candidate = (1.0 - alpha) * candidate + alpha * reference_fallback
    delta_choices = [-2, -1, 0, 0, 0, 1, 2] if pass_index == 0 else [-1, 0, 0, 0, 1]
    candidate = _morph(candidate, int(rng.choice(delta_choices)))
    sigma = float(rng.choice([0.0, 0.0, 0.35, 0.55, 0.80]))
    threshold = float(rng.uniform(0.46, 0.54))
    return _smooth(candidate, sigma, threshold)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _save_state(path: Path, data: dict[str, Any]) -> None:
    save_json(path, data)


def _refine_one(
    *,
    row: dict[str, str],
    output: Path,
    state_path: Path,
    cfg: dict[str, Any],
    style_profiles: dict[str, Any],
    decompositions: dict[int, Any],
) -> tuple[np.ndarray, float, dict[str, Any], str]:
    cp = int(row["codepoint"])
    chosen = read_ink(row["chosen_path"])
    reference = read_ink(row["ref_path"])
    reference_proxy = read_proxy(row["ref_proxy_path"])
    complexity = float(row.get("complexity", 0.0) or 0.0)
    profile = _profile_for_complexity(style_profiles, complexity)
    topology_cfg = cfg.get("topology", {})
    ref_cfg = cfg["marathon"]["refine"]
    analysis_size = int(ref_cfg.get("analysis_size", 192))
    reference_signature = topology_signature(
        reference,
        size=int(topology_cfg.get("analysis_size", analysis_size)),
        prune_iterations=int(topology_cfg.get("prune_iterations", 1)),
    )
    fallback_candidates = make_reference_fallbacks(reference, profile, threshold=float(cfg["render"]["threshold"]))
    evaluated_fallbacks: list[tuple[float, np.ndarray, dict[str, Any], str]] = []
    for label, mask in fallback_candidates:
        score, details = _candidate_score(
            mask, reference, reference_proxy, reference_signature, profile, topology_cfg, analysis_size
        )
        evaluated_fallbacks.append((score, mask, details, label))
    evaluated_fallbacks.sort(key=lambda item: (not item[2]["validation"]["hard_pass"], item[0]))
    fallback_score, fallback, fallback_details, fallback_label = evaluated_fallbacks[0]

    initial: list[tuple[float, np.ndarray, dict[str, Any], str]] = []
    for label, mask in [
        ("selected", chosen),
        ("fallback", fallback),
        ("blend25", 0.75 * chosen + 0.25 * fallback),
        ("blend50", 0.50 * chosen + 0.50 * fallback),
        ("blend75", 0.25 * chosen + 0.75 * fallback),
    ]:
        binary_mask = (mask >= 0.5).astype(np.float32)
        score, details = _candidate_score(
            binary_mask, reference, reference_proxy, reference_signature, profile, topology_cfg, analysis_size
        )
        initial.append((score, binary_mask, details, label))
    initial.sort(key=lambda item: (not item[2]["validation"]["hard_pass"], item[0]))
    best_score, best_mask, best_details, best_label = initial[0]

    state = _load_state(state_path)
    if state.get("fingerprint") == row.get("_refine_fingerprint") and output.is_file():
        loaded = read_ink(output)
        loaded_score, loaded_details = _candidate_score(
            loaded, reference, reference_proxy, reference_signature, profile, topology_cfg, analysis_size
        )
        if loaded_score < best_score:
            best_score, best_mask, best_details, best_label = loaded_score, loaded, loaded_details, "resumed"

    skip_confidence = float(ref_cfg.get("skip_confidence", 0.985))
    if (
        bool(ref_cfg.get("skip_high_confidence_topology_pass", True))
        and bool(best_details["validation"]["hard_pass"])
        and float(row.get("chosen_confidence", 0.0) or 0.0) >= skip_confidence
    ):
        _atomic_save_ink(output, best_mask)
        _save_state(
            state_path,
            {
                "fingerprint": row.get("_refine_fingerprint"),
                "stage": "done",
                "best_score": best_score,
                "best_label": "high_confidence_skip",
                "topology_pass": True,
            },
        )
        return best_mask, best_score, best_details, "high_confidence_skip"

    starts = [item[1] for item in initial]
    passes = max(1, int(ref_cfg.get("passes", 3)))
    trials = max(0, int(ref_cfg.get("global_search_trials", 220)))
    state_interval = max(16, int(ref_cfg.get("state_save_interval_trials", 64)))
    completed_pass = int(state.get("completed_pass", -1)) if state.get("fingerprint") == row.get("_refine_fingerprint") else -1
    completed_trial = int(state.get("completed_trial", -1)) if state.get("fingerprint") == row.get("_refine_fingerprint") else -1
    for pass_index in range(passes):
        if pass_index < completed_pass:
            continue
        start_trial = completed_trial + 1 if pass_index == completed_pass else 0
        if start_trial >= trials:
            continue
        starts[0] = best_mask
        for trial in range(start_trial, trials):
            candidate = _global_trial(trial, cp, starts, fallback, pass_index)
            score, details = _candidate_score(
                candidate, reference, reference_proxy, reference_signature, profile, topology_cfg, analysis_size
            )
            if (details["validation"]["hard_pass"] and not best_details["validation"]["hard_pass"]) or (
                details["validation"]["hard_pass"] == best_details["validation"]["hard_pass"] and score < best_score
            ):
                best_score, best_mask, best_details, best_label = score, candidate, details, f"global_p{pass_index}_t{trial}"
                _atomic_save_ink(output, best_mask)
            if trial % state_interval == state_interval - 1:
                _save_state(
                    state_path,
                    {
                        "fingerprint": row.get("_refine_fingerprint"),
                        "completed_pass": pass_index,
                        "completed_trial": trial,
                        "stage": "global",
                        "best_score": best_score,
                        "best_label": best_label,
                    },
                )
        completed_trial = -1
        _save_state(
            state_path,
            {
                "fingerprint": row.get("_refine_fingerprint"),
                "completed_pass": pass_index,
                "completed_trial": trials - 1,
                "stage": "global_pass_done",
                "best_score": best_score,
                "best_label": best_label,
            },
        )

    zones = component_zones(
        cp,
        reference,
        decompositions,
        fallback_grid=int(ref_cfg.get("zone_grid", 3)),
        maximum_depth=int(ref_cfg.get("component_depth", 3)),
        maximum_zones=int(ref_cfg.get("maximum_component_zones", 32)),
    )
    sweeps = max(0, int(ref_cfg.get("local_sweeps", 4)))
    state = _load_state(state_path)
    resume_local = state.get("fingerprint") == row.get("_refine_fingerprint") and state.get("stage") in {"local", "done"}
    completed_sweep = int(state.get("completed_sweep", -1)) if resume_local else -1
    completed_zone = int(state.get("completed_zone", -1)) if resume_local else -1
    local_state_interval = max(1, int(ref_cfg.get("state_save_interval_zones", 8)))
    for sweep in range(sweeps):
        if sweep < completed_sweep:
            continue
        changed = False
        zone_start = completed_zone + 1 if sweep == completed_sweep else 0
        for zone_index, zone in enumerate(zones):
            if zone_index < zone_start:
                continue
            local_best = (best_score, best_mask, best_details, best_label)
            for weight in (0.18, 0.32, 0.50, 0.68, 0.84, 1.0):
                alpha = (zone * float(weight)).clip(0.0, 1.0)
                candidate = ((1.0 - alpha) * best_mask + alpha * fallback >= 0.5).astype(np.float32)
                score, details = _candidate_score(
                    candidate, reference, reference_proxy, reference_signature, profile, topology_cfg, analysis_size
                )
                if (details["validation"]["hard_pass"] and not local_best[2]["validation"]["hard_pass"]) or (
                    details["validation"]["hard_pass"] == local_best[2]["validation"]["hard_pass"]
                    and score < local_best[0]
                ):
                    local_best = (score, candidate, details, f"zone_s{sweep}_z{zone_index}_w{weight:.2f}")
            if local_best[0] + 1e-8 < best_score or (
                local_best[2]["validation"]["hard_pass"] and not best_details["validation"]["hard_pass"]
            ):
                best_score, best_mask, best_details, best_label = local_best
                _atomic_save_ink(output, best_mask)
                changed = True
            if zone_index % local_state_interval == local_state_interval - 1 or zone_index == len(zones) - 1:
                _save_state(
                    state_path,
                    {
                        "fingerprint": row.get("_refine_fingerprint"),
                        "completed_pass": passes - 1,
                        "completed_trial": trials - 1,
                        "stage": "local",
                        "completed_sweep": sweep,
                        "completed_zone": zone_index,
                        "best_score": best_score,
                        "best_label": best_label,
                    },
                )
        if not changed:
            break

    if not best_details["validation"]["hard_pass"] and fallback_details["validation"]["hard_pass"]:
        best_score, best_mask, best_details, best_label = fallback_score, fallback, fallback_details, f"fallback:{fallback_label}"
    _atomic_save_ink(output, best_mask)
    _save_state(
        state_path,
        {
            "fingerprint": row.get("_refine_fingerprint"),
            "stage": "done",
            "completed_pass": passes - 1,
            "completed_trial": trials - 1,
            "completed_sweep": sweeps - 1,
            "completed_zone": len(zones) - 1,
            "best_score": best_score,
            "best_label": best_label,
            "topology_pass": bool(best_details["validation"]["hard_pass"]),
        },
    )
    return best_mask, best_score, best_details, best_label


def run_marathon_refinement(cfg: dict[str, Any]) -> dict[str, Any]:
    refine_cfg = cfg.get("marathon", {}).get("refine", {})
    if not bool(refine_cfg.get("enabled", True)):
        return {"enabled": False}
    work = Path(cfg["paths"]["work_dir"])
    selection_path = work / "generated" / "selection.csv"
    if not selection_path.is_file():
        raise FileNotFoundError("generated/selection.csv was not found. Run generate first.")
    rows = read_csv(selection_path)
    analysis_rows = {int(row["codepoint"]): row for row in read_csv(work / "audit" / "analysis.csv")}
    profiles_path = work / "dataset" / "style_profiles.json"
    style_profiles = load_json(profiles_path) if profiles_path.is_file() else {"global": load_json(work / "dataset" / "style_profile.json"), "bins": []}
    decomposition_path = Path(refine_cfg.get("decomposition_file", "data/cjkvi-ids/ids.txt"))
    decompositions: dict[int, Any] = {}
    if bool(refine_cfg.get("use_component_layout", True)):
        try:
            decomposition_path, _ = ensure_decomposition_data(refine_cfg)
            decompositions = load_decompositions(
                decomposition_path,
                region_priority=refine_cfg.get("region_priority", []),
                include_obsolete=bool(refine_cfg.get("include_obsolete", False)),
            )
        except IDSDataError as exc:
            print(f"Optional CJKVI IDS refinement hints are unavailable: {exc}")

    refined_dir = ensure_dir(work / "refined")
    chosen_dir = ensure_dir(refined_dir / "chosen")
    states_dir = ensure_dir(refined_dir / "states")
    database = _connect_database(refined_dir / "jobs.sqlite3")
    config_fingerprint = json.dumps(
        {
            "version": CHECKPOINT_FORMAT_VERSION,
            "selection": sha256_file(selection_path),
            "refine": refine_cfg,
            "topology": cfg.get("topology", {}),
            "profiles": sha256_file(profiles_path) if profiles_path.is_file() else "",
            "decomposition": sha256_file(decomposition_path) if decomposition_path.is_file() else "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    fingerprint = __import__("hashlib").sha256(config_fingerprint.encode("utf-8")).hexdigest()
    _set_database_fingerprint(database, fingerprint)
    maximum = int(refine_cfg.get("maximum_glyphs", 0))
    output_rows: list[dict[str, Any]] = []
    counts = {"refined": 0, "fallback": 0, "topology_pass": 0, "failed": 0}
    longrun_guard = LongRunGuard(cfg)

    try:
        iterable = rows if maximum <= 0 else rows[:maximum]
        for source_row in tqdm(iterable, desc="HanziStyleForge per-glyph exhaustive refinement", unit="glyph"):
            row = dict(source_row)
            cp = int(row["codepoint"])
            analysis = analysis_rows.get(cp, {})
            row["complexity"] = analysis.get("complexity", "0")
            output = chosen_dir / cp_filename(cp)
            state_path = states_dir / f"U{cp:06X}.json"
            row["_refine_fingerprint"] = fingerprint
            if _job_done(database, cp, output):
                saved_state = _load_state(state_path)
                saved_label = str(saved_state.get("best_label", "resumed_done"))
                saved_score = float(saved_state.get("best_score", 0.0))
                passed = bool(saved_state.get("topology_pass", True))
                row["chosen_path"] = str(output.resolve())
                row["chosen_source"] = "marathon_fallback" if saved_label.startswith("fallback:") else "marathon_refined"
                row["chosen_label"] = saved_label
                row["chosen_topology_pass"] = int(passed)
                row["chosen_confidence"] = max(0.0, 1.0 - saved_score / 2.0)
                row["pseudo_eligible"] = 0
                output_rows.append(row)
                counts["refined"] += int(not saved_label.startswith("fallback:"))
                counts["fallback"] += int(saved_label.startswith("fallback:"))
                counts["topology_pass"] += int(passed)
                continue
            _set_job(database, cp, "running")
            try:
                _, score, details, label = _refine_one(
                    row=row,
                    output=output,
                    state_path=state_path,
                    cfg=cfg,
                    style_profiles=style_profiles,
                    decompositions=decompositions,
                )
                passed = bool(details["validation"]["hard_pass"])
                row["chosen_path"] = str(output.resolve())
                row["chosen_source"] = "marathon_refined" if not label.startswith("fallback:") else "marathon_fallback"
                row["chosen_label"] = label
                row["chosen_structure_score"] = details["structure"]["structure_score"]
                row["chosen_topology_score"] = details["topology"]["topology_score"]
                row["chosen_topology_pass"] = int(passed)
                row["chosen_component_delta"] = details["topology"]["component_delta"]
                row["chosen_hole_delta"] = details["topology"]["hole_delta"]
                row["chosen_endpoint_delta"] = details["topology"]["endpoint_delta"]
                row["chosen_junction_delta"] = details["topology"]["junction_delta"]
                row["chosen_confidence"] = max(0.0, 1.0 - float(score) / 2.0)
                row["pseudo_eligible"] = 0
                row["notes"] = (row.get("notes", "") + f" | LongRun: {label}; score={score:.6f}").strip(" |")
                output_rows.append(row)
                counts["refined"] += int(not label.startswith("fallback:"))
                counts["fallback"] += int(label.startswith("fallback:"))
                counts["topology_pass"] += int(passed)
                _set_job(database, cp, "done", score=score, topology_pass=passed, source=row["chosen_source"])
            except Exception as exc:
                counts["failed"] += 1
                _set_job(database, cp, "failed", message=f"{type(exc).__name__}: {exc}")
                # Preserve the already selected Final result rather than aborting the whole font.
                output_rows.append(source_row)
            if len(output_rows) % max(1, int(refine_cfg.get("save_every_glyphs", 8))) == 0:
                write_csv(refined_dir / "selection.partial.csv", output_rows, list(source_row.keys()))
                save_json(refined_dir / "progress.json", {"fingerprint": fingerprint, "processed": len(output_rows), "counts": counts})
                longrun_guard.checkpoint_boundary()
    finally:
        database.close()

    # If maximum_glyphs was used, append untouched remainder for a valid build.
    if maximum > 0 and maximum < len(rows):
        output_rows.extend(rows[maximum:])
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(refined_dir / "selection.csv", output_rows, fieldnames)
    summary = {
        "enabled": True,
        "fingerprint": fingerprint,
        "input_count": len(rows),
        "output_count": len(output_rows),
        "counts": counts,
        "selection": str((refined_dir / "selection.csv").resolve()),
        "decomposition_records": len(decompositions),
    }
    save_json(refined_dir / "summary.json", summary)
    save_json(refined_dir / "completed.json", {"fingerprint": fingerprint, "completed": True})
    longrun_guard.checkpoint_boundary()
    return summary
