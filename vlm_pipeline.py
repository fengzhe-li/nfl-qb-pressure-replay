"""End-to-end orchestration pipeline connecting QB Pressure Replay v2 with VLM analysis.

Provides:
- Non-blocking / optional execution of VLM visual inference on selected plays.
- Automatic computation of window-specific tracking-data ground truth for every play.
- Persistent JSON serialization of model checkpoint, revision, config, per-play
  evaluations, and aggregate accuracy/abstention/latency metrics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import argparse
import json
import platform
import sys

from vlm_evaluation import (
    aggregate_evaluation_results,
    aggregate_temporal_evaluation_results,
    analyze_vlm_errors,
    compute_temporal_pass_rush_ground_truth,
    compute_window_ground_truth,
    evaluate_play_prediction,
    evaluate_temporal_play_prediction,
)
from vlm_frame_generator import (
    generate_play_frames,
    select_representative_frame_indices,
)
from vlm_inference import (
    DEFAULT_MODEL_ID,
    DEFAULT_QUESTION,
    DEFAULT_TEMPORAL_QUESTION,
    QwenVLMAnalyzer,
)


def get_runtime_metadata(
    game_id: int,
    model_id: str,
    model_revision: str,
    device: str,
    num_frames: int,
    evaluated_play_ids: Sequence[int],
    experiment_version: str = "v1_baseline",
) -> Dict[str, Any]:
    """Record reproducible hardware, environment, model, and dataset metadata."""
    torch_version = "not_installed"
    transformers_version = "not_installed"
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        pass
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        pass

    return {
        "experiment_version": experiment_version,
        "model_checkpoint": model_id,
        "model_revision": model_revision,
        "inference_device": device,
        "dataset_repository": "https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data",
        "tracking_game_id": int(game_id),
        "frames_per_play": int(num_frames),
        "evaluated_play_ids": [int(pid) for pid in evaluated_play_ids],
        "python_version": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def load_cached_vlm_evaluation(output_folder: Path, game_id: int) -> Optional[Dict[str, Any]]:
    """Load an existing v1 VLM evaluation JSON file for `game_id` if present."""
    out_dir = Path(output_folder)
    candidates = [out_dir / f"vlm_evaluation_{int(game_id)}.json"]
    if out_dir.name == "qb_pressure_outputs":
        candidates.append(
            Path(__file__).resolve().parent / f"example_evaluation_{int(game_id)}.json"
        )
    for eval_path in candidates:
        if eval_path.is_file():
            try:
                return json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def load_cached_temporal_vlm_evaluation(
    output_folder: Path, game_id: int, version_tag: str = "v2"
) -> Optional[Dict[str, Any]]:
    """Load a separately versioned temporal pass-rush VLM evaluation report if present."""
    out_dir = Path(output_folder)
    candidates = [
        out_dir / f"vlm_temporal_evaluation_{int(game_id)}_{version_tag}.json"
    ]
    if out_dir.name == "qb_pressure_outputs":
        candidates.append(
            Path(__file__).resolve().parent
            / f"example_evaluation_temporal_{int(game_id)}_{version_tag}.json"
        )
    for eval_path in candidates:
        if eval_path.is_file():
            try:
                return json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def run_temporal_vlm_experiment(
    payloads: List[Dict[str, Any]],
    output_folder: Path,
    game_id: int,
    vlm_play_ids: Optional[Sequence[int]] = None,
    max_vlm_plays: int = 10,
    num_frames: int = 4,
    model_id: str = DEFAULT_MODEL_ID,
    analyzer: Optional[QwenVLMAnalyzer] = None,
    version_tag: str = "v2",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run genuine temporal pass-rush VLM inference and save to a separately versioned report.

    Does NOT modify the frozen v1 baseline (`example_evaluation_2021090900.json` or
    `vlm_evaluation_2021090900.json`). Writes results to:
    `output_folder / vlm_temporal_evaluation_{game_id}_{version_tag}.json`.
    """
    output_folder = Path(output_folder)
    frames_root = output_folder / f"frames_{int(game_id)}"

    if vlm_play_ids:
        target_play_set = {int(pid) for pid in vlm_play_ids}
    else:
        target_play_set = {
            int(p["play"]) for p in payloads[: max(0, int(max_vlm_plays))]
        }

    if analyzer is None:
        analyzer = QwenVLMAnalyzer(model_id=model_id)

    target_order = [int(p["play"]) for p in payloads if int(p["play"]) in target_play_set]
    temporal_records: List[Dict[str, Any]] = []

    for p in payloads:
        play_id = int(p["play"])
        if play_id not in target_play_set:
            continue

        idx_in_run = target_order.index(play_id) + 1
        if progress_callback:
            progress_callback(
                f"Running Temporal VLM ({version_tag}) on Play {play_id} ({idx_in_run}/{len(target_order)})..."
            )
        play_frame_dir = frames_root / f"play_{play_id}"
        bundle = generate_play_frames(
            p,
            output_dir=play_frame_dir,
            num_frames=num_frames,
        )
        temporal_gt = compute_temporal_pass_rush_ground_truth(
            p, sampled_frame_ids=bundle.sampled_frame_ids
        )
        pred = analyzer.analyze_temporal_frame_bundle(
            bundle, question=DEFAULT_TEMPORAL_QUESTION
        )
        eval_rec = evaluate_temporal_play_prediction(
            parsed_prediction=pred,
            temporal_ground_truth=temporal_gt,
            latency_seconds=pred.get("latency_seconds", 0.0),
            model_name=pred.get("model_name", model_id),
            model_revision=pred.get("model_revision", analyzer.model_revision),
        )
        eval_rec["sampled_frame_ids"] = bundle.sampled_frame_ids
        eval_rec["sampled_times"] = bundle.sampled_times
        temporal_records.append(eval_rec)

    metadata = get_runtime_metadata(
        game_id=game_id,
        model_id=model_id,
        model_revision=analyzer.model_revision,
        device=analyzer.device,
        num_frames=num_frames,
        evaluated_play_ids=[int(r["play_id"]) for r in temporal_records],
        experiment_version=f"temporal_pass_rush_{version_tag}",
    )
    report = aggregate_temporal_evaluation_results(temporal_records, metadata=metadata)

    output_folder.mkdir(parents=True, exist_ok=True)
    versioned_path = output_folder / f"vlm_temporal_evaluation_{int(game_id)}_{version_tag}.json"
    versioned_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def attach_vlm_analysis_to_plays(
    payloads: List[Dict[str, Any]],
    output_folder: Path,
    game_id: int,
    run_vlm: bool = False,
    vlm_play_ids: Optional[Sequence[int]] = None,
    max_vlm_plays: int = 5,
    num_frames: int = 4,
    model_id: str = DEFAULT_MODEL_ID,
    analyzer: Optional[QwenVLMAnalyzer] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Attach window/temporal ground truth, error analysis, and optional VLM analysis to plays.

    If `run_vlm` is False, any previously saved results in
    `output_folder / vlm_evaluation_{game_id}.json` (and versioned temporal evaluations in
    `vlm_temporal_evaluation_{game_id}_v2.json`) are loaded and attached without invoking
    the model. Every play receives its independent tracking-data window & temporal
    ground-truth reference regardless of whether VLM inference was run on that play.
    """
    output_folder = Path(output_folder)
    frames_root = output_folder / f"frames_{int(game_id)}"

    # Load existing cached v1 per-play VLM evaluations
    cached_report = load_cached_vlm_evaluation(output_folder, game_id)
    cached_by_play: Dict[int, Dict[str, Any]] = {}
    if cached_report and isinstance(cached_report.get("per_play_results"), list):
        for item in cached_report["per_play_results"]:
            if "play_id" in item:
                cached_by_play[int(item["play_id"])] = dict(item)

    # Load existing cached v2 temporal per-play evaluations if available
    cached_temporal_report = load_cached_temporal_vlm_evaluation(output_folder, game_id, version_tag="v2")
    temporal_by_play: Dict[int, Dict[str, Any]] = {}
    if cached_temporal_report and isinstance(cached_temporal_report.get("per_play_results"), list):
        for item in cached_temporal_report["per_play_results"]:
            if "play_id" in item:
                temporal_by_play[int(item["play_id"])] = dict(item)

    target_play_set: set[int] = set()
    if run_vlm:
        if vlm_play_ids:
            target_play_set = {int(pid) for pid in vlm_play_ids}
        else:
            for p in payloads[: max(0, int(max_vlm_plays))]:
                target_play_set.add(int(p["play"]))

    if run_vlm and analyzer is None:
        analyzer = QwenVLMAnalyzer(model_id=model_id)

    evaluated_records: List[Dict[str, Any]] = []
    target_order = [int(p["play"]) for p in payloads if int(p["play"]) in target_play_set]

    for p in payloads:
        play_id = int(p["play"])
        frames = p.get("frames", [])
        sampled_indices = select_representative_frame_indices(
            len(frames), num_frames=num_frames
        )
        sampled_frame_ids = [int(frames[idx]["frame"]) for idx in sampled_indices]
        temporal_gt = compute_temporal_pass_rush_ground_truth(
            p, sampled_frame_ids=sampled_frame_ids
        )

        if run_vlm and play_id in target_play_set and analyzer is not None:
            idx_in_run = target_order.index(play_id) + 1
            if progress_callback:
                progress_callback(
                    f"Running VLM analysis on Play {play_id} ({idx_in_run}/{len(target_order)})..."
                )
            play_frame_dir = frames_root / f"play_{play_id}"
            bundle = generate_play_frames(
                p,
                output_dir=play_frame_dir,
                num_frames=num_frames,
            )
            pred = analyzer.analyze_frame_bundle(bundle, question=DEFAULT_QUESTION)
            eval_rec = evaluate_play_prediction(
                parsed_prediction=pred,
                ground_truth=temporal_gt,
                latency_seconds=pred.get("latency_seconds", 0.0),
                model_name=pred.get("model_name", model_id),
                model_revision=pred.get("model_revision", analyzer.model_revision),
            )
            eval_rec["sampled_frame_ids"] = bundle.sampled_frame_ids
            eval_rec["sampled_times"] = bundle.sampled_times
            eval_rec["preview_frames"] = bundle.to_preview_data_uris(max_width=260)
            cached_by_play[play_id] = eval_rec

        if play_id in cached_by_play:
            rec = dict(cached_by_play[play_id])
            rec["ground_truth"] = temporal_gt
            p["vlm_analysis"] = rec
            evaluated_records.append(rec)
        else:
            p["vlm_analysis"] = {
                "game_id": int(game_id),
                "play_id": play_id,
                "model_name": model_id,
                "model_revision": (
                    analyzer.model_revision
                    if analyzer is not None
                    else (
                        cached_report.get("metadata", {}).get("model_revision", "not_run")
                        if cached_report
                        else "not_run"
                    )
                ),
                "outcome_status": "not_analyzed",
                "exact_match": False,
                "sampled_frame_match": False,
                "abstained": False,
                "invalid_output": False,
                "latency_seconds": None,
                "predicted_rusher_id": None,
                "predicted_rusher_tag": None,
                "predicted_rusher_name": None,
                "predicted_rusher_distance_yards": None,
                "distance_error_yards": None,
                "movement_pattern": "",
                "visible_evidence": "",
                "uncertainty": "",
                "explanation": (
                    "Optional VLM inference has not been run on this play yet. "
                    "Tracking-data reference for this play's frame sequence is shown alongside."
                ),
                "sampled_frame_ids": sampled_frame_ids,
                "sampled_times": [float(frames[idx]["time"]) for idx in sampled_indices],
                "ground_truth": temporal_gt,
                "error_analysis": "Not analyzed (run with --vlm to evaluate additional plays).",
            }

        if play_id in temporal_by_play:
            t_rec = dict(temporal_by_play[play_id])
            t_rec["ground_truth"] = temporal_gt
            p["vlm_temporal_analysis"] = t_rec
        else:
            p["vlm_temporal_analysis"] = None

    model_rev = "not_run"
    device_used = "not_run"
    if analyzer is not None:
        model_rev = analyzer.model_revision
        device_used = analyzer.device
    elif cached_report and isinstance(cached_report.get("metadata"), dict):
        model_rev = cached_report["metadata"].get("model_revision", "unknown")
        device_used = cached_report["metadata"].get("inference_device", "unknown")

    metadata = get_runtime_metadata(
        game_id=game_id,
        model_id=model_id,
        model_revision=model_rev,
        device=device_used,
        num_frames=num_frames,
        evaluated_play_ids=[int(r["play_id"]) for r in evaluated_records],
    )
    summary_report = aggregate_evaluation_results(evaluated_records, metadata=metadata)
    summary_report["error_analysis_summary"] = analyze_vlm_errors(summary_report)
    if cached_temporal_report:
        summary_report["temporal_evaluation_v2"] = {
            k: v for k, v in cached_temporal_report.items() if k != "per_play_results"
        }
        summary_report["temporal_evaluation_v2"]["per_play_results"] = [
            {k: v for k, v in r.items() if k != "ground_truth"}
            for r in cached_temporal_report.get("per_play_results", [])
        ]

    if run_vlm and evaluated_records:
        output_folder.mkdir(parents=True, exist_ok=True)
        eval_path = output_folder / f"vlm_evaluation_{int(game_id)}.json"
        compact_records = []
        for r in evaluated_records:
            r_copy = {k: v for k, v in r.items() if k != "preview_frames"}
            compact_records.append(r_copy)
        compact_report = dict(summary_report)
        compact_report["per_play_results"] = compact_records
        eval_path.write_text(json.dumps(compact_report, indent=2), encoding="utf-8")

    return summary_report

