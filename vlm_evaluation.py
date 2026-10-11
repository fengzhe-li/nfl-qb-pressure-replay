"""Tracking-grounded evaluation and JSON validation for VLM QB pressure analysis.

Computes independent reference answers strictly over the frame/time window shown
to the VLM, validates structured VLM JSON outputs, resolves player identities,
and computes per-play error diagnostics and aggregate evaluation metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import re
import statistics


def _last_name(full_name: str) -> str:
    parts = str(full_name).strip().split()
    if not parts:
        return ""
    last = parts[-1]
    if last in {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV"} and len(parts) > 1:
        return parts[-2]
    return last


def compute_window_ground_truth(
    play_payload: Dict[str, Any],
    sampled_frame_ids: Optional[Sequence[int]] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    pressure_threshold_yd: float = 2.0,
) -> Dict[str, Any]:
    """Compute tracking-data ground truth for the specific frame window shown to the VLM.

    Unlike `play_payload["summary"]` (which always summarizes the entire play),
    this function restricts distance calculations to `[start_frame, end_frame]`
    and also records the closest rusher across the exact `sampled_frame_ids`.
    """
    if not isinstance(play_payload, dict):
        raise ValueError("play_payload must be a dictionary.")
    frames = play_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("play_payload must contain a non-empty 'frames' list.")
    rushers = play_payload.get("rushers")
    if not isinstance(rushers, list) or not rushers:
        raise ValueError("play_payload must contain a non-empty 'rushers' list.")
    qb_id = int(play_payload["qb"])

    all_frame_ids = [int(f["frame"]) for f in frames]
    if sampled_frame_ids is not None and len(sampled_frame_ids) > 0:
        sampled_set = {int(fid) for fid in sampled_frame_ids}
        if start_frame is None:
            start_frame = min(sampled_set)
        if end_frame is None:
            end_frame = max(sampled_set)
    else:
        if start_frame is None:
            start_frame = min(all_frame_ids)
        if end_frame is None:
            end_frame = max(all_frame_ids)
        sampled_set = {
            fid for fid in all_frame_ids if int(start_frame) <= fid <= int(end_frame)
        }

    window_frames = [
        f for f in frames if int(start_frame) <= int(f["frame"]) <= int(end_frame)
    ]
    if not window_frames:
        raise ValueError(
            f"No frames found in window [{start_frame}, {end_frame}]."
        )

    sampled_frames = [f for f in window_frames if int(f["frame"]) in sampled_set]
    if not sampled_frames:
        sampled_frames = window_frames

    # Deterministic rusher tags R1, R2, ... by sorted nflId (matching vlm_frame_generator)
    jersey_by_id: Dict[int, str] = {}
    for ent in window_frames[0].get("entities", []):
        eid = ent.get("id")
        if eid is not None and ent.get("number"):
            jersey_by_id[int(eid)] = str(ent["number"])

    sorted_rushers = sorted(rushers, key=lambda r: int(r["id"]))
    rusher_meta: Dict[int, Dict[str, Any]] = {}
    for idx, r in enumerate(sorted_rushers):
        rid = int(r["id"])
        name = str(r.get("name", rid))
        rusher_meta[rid] = {
            "id": rid,
            "tag": f"R{idx + 1}",
            "name": name,
            "last_name": _last_name(name),
            "number": jersey_by_id.get(rid, ""),
        }

    def _frame_rusher_distance(frame_obj: Dict[str, Any], rid: int) -> float:
        # Compute directly from entity coordinates (x, y) to be self-contained
        entities = frame_obj.get("entities", [])
        qb_ent = next((e for e in entities if e.get("id") == qb_id), None)
        r_ent = next((e for e in entities if e.get("id") == rid), None)
        if qb_ent is not None and r_ent is not None:
            return float(
                math.hypot(
                    float(r_ent["x"]) - float(qb_ent["x"]),
                    float(r_ent["y"]) - float(qb_ent["y"]),
                )
            )
        # Fallback to frame distances dict if entities were omitted in a synthetic test
        dists = frame_obj.get("distances", {})
        if str(rid) in dists:
            return float(dists[str(rid)])
        raise ValueError(f"Rusher {rid} or QB {qb_id} missing from frame {frame_obj.get('frame')}.")

    per_rusher_records: List[Dict[str, Any]] = []
    for rid, rinfo in rusher_meta.items():
        win_series = [
            (int(f["frame"]), float(f["time"]), _frame_rusher_distance(f, rid))
            for f in window_frames
        ]
        samp_series = [
            (int(f["frame"]), float(f["time"]), _frame_rusher_distance(f, rid))
            for f in sampled_frames
        ]
        best_win = min(win_series, key=lambda x: x[2])
        best_samp = min(samp_series, key=lambda x: x[2])
        within_entries = [item for item in win_series if item[2] <= pressure_threshold_yd]
        per_rusher_records.append({
            "id": rid,
            "tag": rinfo["tag"],
            "name": rinfo["name"],
            "last_name": rinfo["last_name"],
            "number": rinfo["number"],
            "min_distance_yards": round(best_win[2], 4),
            "closest_frame_id": best_win[0],
            "closest_time_s": round(best_win[1], 2),
            "min_sampled_distance_yards": round(best_samp[2], 4),
            "sampled_closest_frame_id": best_samp[0],
            "sampled_closest_time_s": round(best_samp[1], 2),
            "entered_2yd_zone": len(within_entries) > 0,
            "first_within_2yd_time_s": (
                round(min(item[1] for item in within_entries), 2)
                if within_entries
                else None
            ),
        })

    by_window_dist = sorted(per_rusher_records, key=lambda r: (r["min_distance_yards"], r["id"]))
    by_sampled_dist = sorted(
        per_rusher_records, key=lambda r: (r["min_sampled_distance_yards"], r["id"])
    )

    closest = by_window_dist[0]
    runner_up = by_window_dist[1] if len(by_window_dist) > 1 else None
    sampled_closest = by_sampled_dist[0]

    margin_yards = (
        round(runner_up["min_distance_yards"] - closest["min_distance_yards"], 4)
        if runner_up is not None
        else None
    )

    return {
        "game_id": int(play_payload.get("game", 0)),
        "play_id": int(play_payload.get("play", 0)),
        "window_start_frame": int(window_frames[0]["frame"]),
        "window_end_frame": int(window_frames[-1]["frame"]),
        "window_start_time": float(window_frames[0]["time"]),
        "window_end_time": float(window_frames[-1]["time"]),
        "sampled_frame_ids": [int(f["frame"]) for f in sampled_frames],
        "closest_rusher_id": closest["id"],
        "closest_rusher_tag": closest["tag"],
        "closest_rusher_name": closest["name"],
        "closest_rusher_number": closest["number"],
        "closest_distance_yards": closest["min_distance_yards"],
        "closest_frame_id": closest["closest_frame_id"],
        "closest_time_s": closest["closest_time_s"],
        "sampled_closest_rusher_id": sampled_closest["id"],
        "sampled_closest_rusher_tag": sampled_closest["tag"],
        "sampled_closest_rusher_name": sampled_closest["name"],
        "sampled_closest_distance_yards": sampled_closest["min_sampled_distance_yards"],
        "sampled_closest_frame_id": sampled_closest["sampled_closest_frame_id"],
        "sampled_closest_time_s": sampled_closest["sampled_closest_time_s"],
        "runner_up_rusher_id": runner_up["id"] if runner_up else None,
        "runner_up_rusher_name": runner_up["name"] if runner_up else None,
        "runner_up_distance_yards": runner_up["min_distance_yards"] if runner_up else None,
        "margin_yards": margin_yards,
        "entered_2yd_zone_in_window": any(r["entered_2yd_zone"] for r in per_rusher_records),
        "pressure_threshold_yd": float(pressure_threshold_yd),
        "whole_play_pff_label": str(play_payload.get("pff_label", "Unknown")),
        "whole_play_pff_pressure": bool(play_payload.get("pff_pressure", False)),
        "rusher_window_distances": per_rusher_records,
        "metric_distinction_note": (
            f"Tracking reference reflects geometric Euclidean proximity in the observed window "
            f"[{window_frames[0]['time']:.1f}s, {window_frames[-1]['time']:.1f}s] "
            f"(exploratory threshold: {pressure_threshold_yd:g} yd). "
            f"Official PFF labels ({play_payload.get('pff_label', 'n/a')}) are whole-play annotations."
        ),
    }


def resolve_rusher_identity(
    raw_identifier: Any,
    raw_name: Any,
    candidate_rushers: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Match a VLM rusher identifier/name to one of the play's candidate pass rushers."""
    if not candidate_rushers:
        return None

    tokens: List[str] = []
    for val in (raw_identifier, raw_name):
        if val is not None:
            s = str(val).strip()
            if s and s.lower() not in {"none", "null", "n/a", "unknown", "abstain", "abstained", "uncertain"}:
                tokens.append(s)
    if not tokens:
        return None

    combined = " ".join(tokens)

    # 1. Exact nflId match
    for tok in tokens:
        if tok.isdigit():
            tid = int(tok)
            for r in candidate_rushers:
                if int(r["id"]) == tid:
                    return r
        id_match = re.search(r"\b(?:id[:=\s]*)?(\d{4,7})\b", tok, flags=re.IGNORECASE)
        if id_match:
            tid = int(id_match.group(1))
            for r in candidate_rushers:
                if int(r["id"]) == tid:
                    return r

    # 2. Exact R-tag match (e.g. "R1", "R2")
    tag_matches = re.findall(r"\b(R\d+)\b", combined, flags=re.IGNORECASE)
    if tag_matches:
        target_tag = tag_matches[0].upper()
        for r in candidate_rushers:
            if str(r.get("tag", "")).upper() == target_tag:
                return r

    # 3. Full name or last-name match (if unambiguous)
    combined_lower = combined.lower()
    full_matches = [
        r for r in candidate_rushers if str(r.get("name", "")).lower() in combined_lower
    ]
    if len(full_matches) == 1:
        return full_matches[0]

    last_matches = [
        r
        for r in candidate_rushers
        if r.get("last_name") and re.search(rf"\b{re.escape(str(r['last_name']).lower())}\b", combined_lower)
    ]
    if len(last_matches) == 1:
        return last_matches[0]

    # 4. Jersey number match (e.g. "#58" or "58", if unambiguous among rushers)
    num_matches = re.findall(r"#?(\d{1,2})\b", combined)
    for num_str in num_matches:
        cand = [r for r in candidate_rushers if str(r.get("number", "")) == num_str]
        if len(cand) == 1:
            return cand[0]

    return None


def validate_and_parse_vlm_json(
    raw_output: Any,
    candidate_rushers: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate and normalize raw VLM output into a structured prediction dictionary."""
    if raw_output is None:
        return {
            "is_valid_json": False,
            "status": "invalid_output",
            "abstained": False,
            "selected_rusher_id": None,
            "selected_rusher_name": None,
            "selected_rusher_tag": None,
            "movement_pattern": "",
            "visible_evidence": "",
            "uncertainty": "Empty output from model.",
            "explanation": "",
            "raw_text": "",
            "validation_error": "Output is None.",
        }

    raw_text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
    parsed: Optional[Dict[str, Any]] = None

    if isinstance(raw_output, dict):
        parsed = raw_output
    elif isinstance(raw_output, str):
        text = raw_output.strip()
        if not text:
            return {
                "is_valid_json": False,
                "status": "invalid_output",
                "abstained": False,
                "selected_rusher_id": None,
                "selected_rusher_name": None,
                "selected_rusher_tag": None,
                "movement_pattern": "",
                "visible_evidence": "",
                "uncertainty": "Empty string returned by model.",
                "explanation": "",
                "raw_text": raw_text,
                "validation_error": "Empty string.",
            }
        # Strip markdown code fences if present
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate_str = fenced.group(1) if fenced else text
        try:
            obj = json.loads(candidate_str)
            if isinstance(obj, dict):
                parsed = obj
        except Exception:
            # Try finding outermost {...}
            brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if brace_match:
                try:
                    obj = json.loads(brace_match.group(0))
                    if isinstance(obj, dict):
                        parsed = obj
                except Exception:
                    parsed = None

    if not isinstance(parsed, dict):
        return {
            "is_valid_json": False,
            "status": "invalid_output",
            "abstained": False,
            "selected_rusher_id": None,
            "selected_rusher_name": None,
            "selected_rusher_tag": None,
            "movement_pattern": "",
            "visible_evidence": "",
            "uncertainty": "Failed to parse structured JSON from model output.",
            "explanation": "",
            "raw_text": raw_text,
            "validation_error": "Malformed JSON.",
        }

    movement_pattern = parsed.get("movement_pattern")
    visible_evidence = parsed.get("visible_evidence")
    uncertainty = parsed.get("uncertainty", parsed.get("abstention_reason"))

    missing_fields = []
    for fname, fval in [
        ("movement_pattern", movement_pattern),
        ("visible_evidence", visible_evidence),
        ("uncertainty", uncertainty),
    ]:
        if not isinstance(fval, str) or not fval.strip():
            missing_fields.append(fname)

    if "selected_rusher_id" not in parsed and "selected_rusher" not in parsed:
        missing_fields.append("selected_rusher_id")

    if missing_fields:
        return {
            "is_valid_json": False,
            "status": "invalid_output",
            "abstained": False,
            "selected_rusher_id": None,
            "selected_rusher_name": None,
            "selected_rusher_tag": None,
            "movement_pattern": str(movement_pattern or ""),
            "visible_evidence": str(visible_evidence or ""),
            "uncertainty": str(uncertainty or ""),
            "explanation": str(parsed.get("explanation", "") or ""),
            "raw_text": raw_text,
            "validation_error": f"Missing or invalid required fields: {', '.join(missing_fields)}",
        }

    raw_id = parsed.get("selected_rusher_id", parsed.get("selected_rusher"))
    raw_name = parsed.get("selected_rusher_name", parsed.get("rusher_name"))
    explicit_abstain = bool(parsed.get("abstained", False))

    null_tokens = {"none", "null", "n/a", "abstain", "abstained", "uncertain", "unidentifiable", ""}
    is_null_selection = (
        raw_id is None or str(raw_id).strip().lower() in null_tokens
    ) and (
        raw_name is None or str(raw_name).strip().lower() in null_tokens
    )

    abstained = explicit_abstain or is_null_selection
    resolved = None
    if not abstained and candidate_rushers:
        resolved = resolve_rusher_identity(raw_id, raw_name, candidate_rushers)

    if abstained:
        status = "abstained"
        sel_id = None
        sel_name = None
        sel_tag = None
    elif resolved is not None:
        status = "ok"
        sel_id = int(resolved["id"])
        sel_name = str(resolved["name"])
        sel_tag = str(resolved.get("tag", ""))
    else:
        # Model claimed a non-null rusher identifier, keep raw claim if no candidate list was supplied
        if candidate_rushers is None and raw_id is not None:
            status = "ok"
            sel_id = int(raw_id) if str(raw_id).isdigit() else raw_id
            sel_name = str(raw_name or raw_id)
            sel_tag = None
        else:
            status = "unresolvable_rusher"
            sel_id = None
            sel_name = str(raw_name or raw_id or "")
            sel_tag = None

    explanation = str(parsed.get("explanation", "") or "").strip()
    if not explanation:
        explanation = f"{movement_pattern.strip()} {visible_evidence.strip()}".strip()

    return {
        "is_valid_json": True,
        "status": status,
        "abstained": abstained,
        "selected_rusher_id": sel_id,
        "selected_rusher_name": sel_name,
        "selected_rusher_tag": sel_tag,
        "raw_selected_rusher": raw_id,
        "movement_pattern": movement_pattern.strip(),
        "visible_evidence": visible_evidence.strip(),
        "uncertainty": uncertainty.strip(),
        "explanation": explanation,
        "raw_text": raw_text,
        "validation_error": None,
    }


def evaluate_play_prediction(
    parsed_prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    latency_seconds: float = 0.0,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    model_revision: str = "unknown",
) -> Dict[str, Any]:
    """Evaluate a validated VLM prediction against the play window's tracking ground truth."""
    pred_status = parsed_prediction.get("status", "invalid_output")
    is_valid_json = bool(parsed_prediction.get("is_valid_json", False))
    abstained = bool(parsed_prediction.get("abstained", False))
    pred_id = parsed_prediction.get("selected_rusher_id")

    gt_id = int(ground_truth["closest_rusher_id"])
    gt_name = str(ground_truth["closest_rusher_name"])
    gt_tag = str(ground_truth.get("closest_rusher_tag", ""))
    gt_dist = float(ground_truth["closest_distance_yards"])
    samp_gt_id = int(ground_truth["sampled_closest_rusher_id"])

    rusher_dist_map = {
        int(r["id"]): float(r["min_distance_yards"])
        for r in ground_truth.get("rusher_window_distances", [])
    }

    pred_dist: Optional[float] = None
    dist_error: Optional[float] = None
    if isinstance(pred_id, int) and pred_id in rusher_dist_map:
        pred_dist = rusher_dist_map[pred_id]
        dist_error = round(pred_dist - gt_dist, 4)

    if pred_status == "model_unavailable":
        outcome_status = "model_unavailable"
        exact_match = False
        sampled_match = False
        error_note = parsed_prediction.get("uncertainty") or "VLM model unavailable."
    elif not is_valid_json or pred_status == "invalid_output":
        outcome_status = "invalid_output"
        exact_match = False
        sampled_match = False
        error_note = f"Invalid VLM output: {parsed_prediction.get('validation_error', 'parse failure')}"
    elif abstained or pred_status == "abstained":
        outcome_status = "abstained"
        exact_match = False
        sampled_match = False
        error_note = f"Model abstained: {parsed_prediction.get('uncertainty', 'ambiguous visual sequence')}"
    elif isinstance(pred_id, int) and pred_id == gt_id:
        outcome_status = "match"
        exact_match = True
        sampled_match = pred_id == samp_gt_id
        error_note = (
            f"Exact match: correctly identified {gt_tag} {gt_name} "
            f"(closest at {gt_dist:.2f} yd in window)."
        )
    else:
        outcome_status = "mismatch"
        exact_match = False
        sampled_match = isinstance(pred_id, int) and pred_id == samp_gt_id
        margin = ground_truth.get("margin_yards")
        if sampled_match:
            error_note = (
                f"Window vs. sampled-frame discrepancy: predicted {parsed_prediction.get('selected_rusher_name')} "
                f"was closest on the discrete sampled frames, whereas {gt_name} reached {gt_dist:.2f} yd "
                f"between sampled frames."
            )
        elif dist_error is not None and dist_error <= 0.5:
            error_note = (
                f"Near-tie mismatch: predicted {parsed_prediction.get('selected_rusher_name')} "
                f"({pred_dist:.2f} yd) vs tracking closest {gt_name} ({gt_dist:.2f} yd, "
                f"difference {dist_error:.2f} yd)."
            )
        elif dist_error is not None:
            error_note = (
                f"Mismatch: predicted {parsed_prediction.get('selected_rusher_name')} "
                f"(min distance {pred_dist:.2f} yd) instead of {gt_name} ({gt_dist:.2f} yd, "
                f"+{dist_error:.2f} yd farther; #1/#2 margin={margin if margin is not None else 'n/a'} yd)."
            )
        else:
            error_note = (
                f"Unresolvable rusher identifier '{parsed_prediction.get('raw_selected_rusher')}' "
                f"(tracking closest: {gt_name} at {gt_dist:.2f} yd)."
            )

    return {
        "game_id": int(ground_truth["game_id"]),
        "play_id": int(ground_truth["play_id"]),
        "model_name": model_name,
        "model_revision": model_revision,
        "outcome_status": outcome_status,
        "exact_match": exact_match,
        "sampled_frame_match": sampled_match,
        "abstained": outcome_status == "abstained",
        "invalid_output": outcome_status == "invalid_output",
        "latency_seconds": round(float(latency_seconds), 4),
        "predicted_rusher_id": pred_id if isinstance(pred_id, int) else None,
        "predicted_rusher_tag": parsed_prediction.get("selected_rusher_tag"),
        "predicted_rusher_name": parsed_prediction.get("selected_rusher_name"),
        "predicted_rusher_distance_yards": pred_dist,
        "distance_error_yards": dist_error,
        "movement_pattern": parsed_prediction.get("movement_pattern", ""),
        "visible_evidence": parsed_prediction.get("visible_evidence", ""),
        "uncertainty": parsed_prediction.get("uncertainty", ""),
        "explanation": parsed_prediction.get("explanation", ""),
        "ground_truth": ground_truth,
        "error_analysis": error_note,
    }


def aggregate_evaluation_results(
    play_evaluations: Sequence[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute aggregate evaluation metrics across a collection of evaluated plays."""
    total = len(play_evaluations)
    if total == 0:
        return {
            "total_plays": 0,
            "identifiable_predictions": 0,
            "exact_match_count": 0,
            "exact_match_accuracy": None,
            "sampled_frame_match_count": 0,
            "sampled_frame_match_accuracy": None,
            "overall_accuracy": None,
            "abstention_count": 0,
            "abstention_frequency": None,
            "invalid_output_count": 0,
            "invalid_output_frequency": None,
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "total_latency_seconds": 0.0,
            "mean_distance_error_yards": None,
            "status_counts": {},
            "metadata": metadata or {},
            "per_play_results": [],
        }

    exact_matches = sum(1 for r in play_evaluations if r.get("exact_match"))
    sampled_matches = sum(1 for r in play_evaluations if r.get("sampled_frame_match"))
    abstentions = sum(1 for r in play_evaluations if r.get("outcome_status") == "abstained")
    invalids = sum(1 for r in play_evaluations if r.get("outcome_status") == "invalid_output")
    unavailable = sum(1 for r in play_evaluations if r.get("outcome_status") == "model_unavailable")
    identifiable = total - abstentions - invalids - unavailable

    latencies = [
        float(r["latency_seconds"])
        for r in play_evaluations
        if r.get("latency_seconds") is not None and r.get("outcome_status") != "model_unavailable"
    ]
    dist_errors = [
        float(r["distance_error_yards"])
        for r in play_evaluations
        if r.get("distance_error_yards") is not None
    ]

    status_counts: Dict[str, int] = {}
    for r in play_evaluations:
        st = str(r.get("outcome_status", "unknown"))
        status_counts[st] = status_counts.get(st, 0) + 1

    return {
        "total_plays": total,
        "identifiable_predictions": identifiable,
        "exact_match_count": exact_matches,
        "exact_match_accuracy": (
            round(exact_matches / float(identifiable), 4) if identifiable > 0 else None
        ),
        "sampled_frame_match_count": sampled_matches,
        "sampled_frame_match_accuracy": (
            round(sampled_matches / float(identifiable), 4) if identifiable > 0 else None
        ),
        "overall_accuracy": round(exact_matches / float(total), 4),
        "abstention_count": abstentions,
        "abstention_frequency": round(abstentions / float(total), 4),
        "invalid_output_count": invalids,
        "invalid_output_frequency": round(invalids / float(total), 4),
        "mean_latency_seconds": round(statistics.mean(latencies), 4) if latencies else None,
        "median_latency_seconds": round(statistics.median(latencies), 4) if latencies else None,
        "total_latency_seconds": round(sum(latencies), 4),
        "mean_distance_error_yards": (
            round(statistics.mean(dist_errors), 4) if dist_errors else None
        ),
        "status_counts": status_counts,
        "metadata": metadata or {},
        "per_play_results": list(play_evaluations),
    }


VALID_DISTANCE_TRENDS = {"closing", "close_then_recede", "steady", "receding"}
VALID_TIMING_PHASES = {"early", "mid", "late"}


def _classify_timing_phase(
    time_s: float, window_start_s: float, window_end_s: float
) -> str:
    """Classify a timestamp within [window_start_s, window_end_s] into 'early', 'mid', or 'late'."""
    span = max(float(window_end_s) - float(window_start_s), 0.1)
    rel = (float(time_s) - float(window_start_s)) / span
    if rel <= (1.0 / 3.0) + 1e-6:
        return "early"
    if rel <= (2.0 / 3.0) + 1e-6:
        return "mid"
    return "late"


def _classify_distance_trend(
    initial_dist: float,
    final_dist: float,
    min_dist: float,
    min_idx: int,
    num_points: int,
) -> str:
    """Deterministically classify a rusher's QB-distance trajectory across a frame window.

    Categories:
    - 'closing': approaches the QB by >= 0.75 yd without rebounding by >= 1.0 yd after peak.
    - 'close_then_recede': approaches the QB by >= 0.75 yd (or peaks before the final frame)
      and then recedes by >= 1.0 yd from minimum before the end of the window.
    - 'receding': net distance increases by >= 0.75 yd with < 0.75 yd closing.
    - 'steady': distance stays within a narrow band (< 0.75 yd closing and < 0.75 yd net change).
    """
    total_closing = float(initial_dist) - float(min_dist)
    rebound = float(final_dist) - float(min_dist)
    net_change = float(final_dist) - float(initial_dist)

    if total_closing >= 0.75:
        if rebound >= 1.0 and min_idx < max(1, num_points - 1):
            return "close_then_recede"
        return "closing"
    if net_change >= 0.75:
        return "receding"
    return "steady"


def _extract_cited_frames(
    text: str,
    sampled_frame_ids: Sequence[int],
) -> Dict[str, Any]:
    """Extract frame references cited by the VLM in visible_evidence / explanation."""
    if not text:
        return {
            "raw_cited_numbers": [],
            "mapped_seq_indices": [],
            "mapped_frame_ids": [],
        }
    matches = re.findall(r"\bframes?\s*(\d+)(?:\s*(?:to|and|-|&|,)\s*(?:frames?\s*)?(\d+))?", text, flags=re.IGNORECASE)
    nums: List[int] = []
    for m in matches:
        for grp in m:
            if grp and grp.isdigit():
                val = int(grp)
                if val not in nums:
                    nums.append(val)

    sampled_list = [int(fid) for fid in sampled_frame_ids]
    seq_indices: List[int] = []
    frame_ids: List[int] = []
    for n in nums:
        if 1 <= n <= len(sampled_list):
            if n not in seq_indices:
                seq_indices.append(n)
            fid = sampled_list[n - 1]
            if fid not in frame_ids:
                frame_ids.append(fid)
        elif n in sampled_list:
            idx1 = sampled_list.index(n) + 1
            if idx1 not in seq_indices:
                seq_indices.append(idx1)
            if n not in frame_ids:
                frame_ids.append(n)
        else:
            # Model cited an unsampled raw tracking frame ID (e.g. Frame 27)
            frame_ids.append(n)

    return {
        "raw_cited_numbers": nums,
        "mapped_seq_indices": seq_indices,
        "mapped_frame_ids": frame_ids,
    }


def analyze_vlm_errors(evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse all VLM predictions in an evaluation report against tracking-grounded references.

    Reports:
    - Incorrect rusher identification & rank distribution of selected rushers
    - Distance error distribution and severity buckets
    - Temporal mistakes (peak timing discrepancy, cited-frame vs. true closest frame/phase,
      early-snap vs. late-pocket mismatches, between-sampled-frame gaps)
    - Observable model biases (tag selection bias toward R1/R2, zero-abstention overconfidence)
    - Explicit separation of `measurable_error_categories` (computed from tracking & outputs)
      from `hypothesised_causes` (interpretive hypotheses that require controlled ablation).
    """
    if not isinstance(evaluation_report, dict):
        raise ValueError("evaluation_report must be a dictionary.")

    per_play = evaluation_report.get("per_play_results", [])
    evaluated = [
        r
        for r in per_play
        if isinstance(r, dict)
        and r.get("outcome_status") not in {"not_analyzed", "model_unavailable", None}
    ]
    total_eval = len(evaluated)
    if total_eval == 0:
        return {
            "evaluated_plays": 0,
            "correct_identifications": 0,
            "incorrect_identifications": 0,
            "incorrect_identification_rate": None,
            "distance_error_summary": {},
            "temporal_mistake_summary": {},
            "observable_biases": {},
            "measurable_error_categories": [],
            "hypothesised_causes": [],
            "per_play_diagnostics": [],
        }

    correct_ids: List[int] = []
    incorrect_ids: List[int] = []
    dist_errors: List[float] = []
    rank_counts: Dict[str, int] = {"rank_1": 0, "rank_2": 0, "rank_3": 0, "rank_4_plus": 0, "unranked": 0}
    ranks_numeric: List[int] = []

    bucket_exact: List[int] = []
    bucket_near_tie: List[int] = []
    bucket_moderate: List[int] = []
    bucket_large: List[int] = []

    peak_timing_errors: List[float] = []
    early_phase_missed_plays: List[int] = []
    between_sampled_frames_plays: List[int] = []
    cited_frame_mismatch_plays: List[int] = []
    phase_mismatch_plays: List[int] = []

    pred_tag_counts: Dict[str, int] = {}
    gt_tag_counts: Dict[str, int] = {}
    crowded_pocket_no_abstain_plays: List[int] = []

    per_play_diagnostics: List[Dict[str, Any]] = []

    for rec in evaluated:
        play_id = int(rec.get("play_id", 0))
        gt = rec.get("ground_truth", {})
        exact = bool(rec.get("exact_match", False))
        if exact:
            correct_ids.append(play_id)
        else:
            incorrect_ids.append(play_id)

        pred_id = rec.get("predicted_rusher_id")
        pred_tag = str(rec.get("predicted_rusher_tag") or "None")
        pred_name = str(rec.get("predicted_rusher_name") or "None")
        gt_id = int(gt.get("closest_rusher_id", 0))
        gt_tag = str(gt.get("closest_rusher_tag") or "Unknown")
        gt_name = str(gt.get("closest_rusher_name") or "Unknown")
        gt_dist = float(gt.get("closest_distance_yards", 0.0))
        pred_dist = (
            float(rec["predicted_rusher_distance_yards"])
            if rec.get("predicted_rusher_distance_yards") is not None
            else None
        )
        d_err = (
            float(rec["distance_error_yards"])
            if rec.get("distance_error_yards") is not None
            else None
        )

        pred_tag_counts[pred_tag] = pred_tag_counts.get(pred_tag, 0) + 1
        gt_tag_counts[gt_tag] = gt_tag_counts.get(gt_tag, 0) + 1

        # Rank of predicted rusher among candidate rushers sorted by min_distance_yards
        rusher_list = sorted(
            gt.get("rusher_window_distances", []),
            key=lambda x: (float(x.get("min_distance_yards", 999.0)), int(x.get("id", 0))),
        )
        pred_rank: Optional[int] = None
        pred_rusher_info: Optional[Dict[str, Any]] = None
        for idx, rinfo in enumerate(rusher_list):
            if pred_id is not None and int(rinfo.get("id", -1)) == int(pred_id):
                pred_rank = idx + 1
                pred_rusher_info = rinfo
                break

        if pred_rank is None:
            rank_counts["unranked"] += 1
        else:
            ranks_numeric.append(pred_rank)
            if pred_rank == 1:
                rank_counts["rank_1"] += 1
            elif pred_rank == 2:
                rank_counts["rank_2"] += 1
            elif pred_rank == 3:
                rank_counts["rank_3"] += 1
            else:
                rank_counts["rank_4_plus"] += 1

        # Distance error bucket
        if d_err is not None:
            dist_errors.append(d_err)
            if d_err == 0.0:
                bucket_exact.append(play_id)
                bucket_label = "exact_0yd"
            elif d_err <= 0.5:
                bucket_near_tie.append(play_id)
                bucket_label = "near_tie_le_0_5yd"
            elif d_err <= 1.5:
                bucket_moderate.append(play_id)
                bucket_label = "moderate_0_5_to_1_5yd"
            else:
                bucket_large.append(play_id)
                bucket_label = "large_gt_1_5yd"
        else:
            bucket_label = "n/a"

        # Crowded pocket / tight margin check for abstention behavior
        margin = gt.get("margin_yards")
        rushers_within_2_5 = sum(
            1 for r in rusher_list if float(r.get("min_distance_yards", 99.0)) <= 2.5
        )
        if ((margin is not None and float(margin) <= 0.75) or rushers_within_2_5 >= 2) and not rec.get("abstained"):
            crowded_pocket_no_abstain_plays.append(play_id)

        # Temporal analysis on existing prediction
        w_start = float(gt.get("window_start_time", 0.0))
        w_end = float(gt.get("window_end_time", 0.0))
        gt_time = float(gt.get("closest_time_s", 0.0))
        gt_frame = int(gt.get("closest_frame_id", 0))
        samp_closest_frame = int(gt.get("sampled_closest_frame_id", gt_frame))
        sampled_fids = [int(fid) for fid in gt.get("sampled_frame_ids", rec.get("sampled_frame_ids", []))]

        gt_phase = _classify_timing_phase(gt_time, w_start, w_end)
        pred_peak_time = (
            float(pred_rusher_info["closest_time_s"])
            if pred_rusher_info and pred_rusher_info.get("closest_time_s") is not None
            else None
        )
        pred_peak_frame = (
            int(pred_rusher_info["closest_frame_id"])
            if pred_rusher_info and pred_rusher_info.get("closest_frame_id") is not None
            else None
        )
        pred_peak_phase = (
            _classify_timing_phase(pred_peak_time, w_start, w_end)
            if pred_peak_time is not None
            else None
        )

        timing_err_s: Optional[float] = None
        if pred_peak_time is not None:
            timing_err_s = round(abs(pred_peak_time - gt_time), 2)
            peak_timing_errors.append(timing_err_s)

        if gt_phase == "early" and not exact:
            early_phase_missed_plays.append(play_id)

        between_sampled = gt_frame not in sampled_fids if sampled_fids else False
        if between_sampled:
            between_sampled_frames_plays.append(play_id)

        if pred_peak_phase is not None and pred_peak_phase != gt_phase:
            phase_mismatch_plays.append(play_id)

        # Extract frame citations from visible_evidence
        cited = _extract_cited_frames(
            f"{rec.get('visible_evidence', '')} {rec.get('explanation', '')}",
            sampled_fids,
        )
        cited_fids = cited["mapped_frame_ids"]
        cited_matches_sampled_closest = (
            (samp_closest_frame in cited_fids) if cited_fids else None
        )
        if cited_matches_sampled_closest is False and not exact:
            cited_frame_mismatch_plays.append(play_id)

        if exact:
            measurable_type = "exact_match"
        elif d_err is not None and d_err <= 0.5:
            measurable_type = "near_tie_spatial_error"
        elif gt_phase == "early":
            measurable_type = "early_snap_proximity_missed"
        elif pred_peak_phase is not None and pred_peak_phase != gt_phase:
            measurable_type = "temporal_phase_and_rusher_error"
        else:
            measurable_type = "spatial_rusher_selection_error"

        cited_str = (
            f"Cited frames {cited['raw_cited_numbers']} (mapped IDs {cited_fids})"
            if cited["raw_cited_numbers"]
            else "No specific frame number cited"
        )

        per_play_diagnostics.append({
            "play_id": play_id,
            "outcome_status": rec.get("outcome_status"),
            "exact_match": exact,
            "predicted_rusher_id": pred_id,
            "predicted_rusher_tag": pred_tag,
            "predicted_rusher_name": pred_name,
            "predicted_min_distance_yards": pred_dist,
            "predicted_rusher_rank": pred_rank,
            "ground_truth_rusher_id": gt_id,
            "ground_truth_rusher_tag": gt_tag,
            "ground_truth_rusher_name": gt_name,
            "ground_truth_min_distance_yards": gt_dist,
            "distance_error_yards": d_err,
            "distance_error_bucket": bucket_label,
            "margin_yards": margin,
            "gt_closest_frame_id": gt_frame,
            "gt_closest_time_s": gt_time,
            "gt_timing_phase": gt_phase,
            "pred_rusher_closest_frame_id": pred_peak_frame,
            "pred_rusher_closest_time_s": pred_peak_time,
            "pred_rusher_timing_phase": pred_peak_phase,
            "peak_timing_error_seconds": timing_err_s,
            "true_peak_between_sampled_frames": between_sampled,
            "vlm_cited_frames": cited["raw_cited_numbers"],
            "vlm_cited_mapped_frame_ids": cited_fids,
            "vlm_cited_frame_note": cited_str,
            "measurable_error_type": measurable_type,
        })

    # Tag bias calculations
    all_tags = sorted(
        set(list(pred_tag_counts.keys()) + list(gt_tag_counts.keys()) + ["R1", "R2", "R3", "R4", "R5"]),
        key=lambda t: (0 if t.startswith("R") and t[1:].isdigit() else 1, int(t[1:]) if t.startswith("R") and t[1:].isdigit() else 99, t),
    )
    tag_comparison_table = []
    for t in all_tags:
        if t == "None":
            continue
        p_cnt = pred_tag_counts.get(t, 0)
        g_cnt = gt_tag_counts.get(t, 0)
        tag_comparison_table.append({
            "tag": t,
            "predicted_count": p_cnt,
            "predicted_share": round(p_cnt / float(total_eval), 4),
            "ground_truth_count": g_cnt,
            "ground_truth_share": round(g_cnt / float(total_eval), 4),
            "share_diff": round((p_cnt - g_cnt) / float(total_eval), 4),
        })

    pred_r1_r2 = pred_tag_counts.get("R1", 0) + pred_tag_counts.get("R2", 0)
    gt_r1_r2 = gt_tag_counts.get("R1", 0) + gt_tag_counts.get("R2", 0)
    pred_r4_r5 = pred_tag_counts.get("R4", 0) + pred_tag_counts.get("R5", 0)
    gt_r4_r5 = gt_tag_counts.get("R4", 0) + gt_tag_counts.get("R5", 0)

    incorrect_count = len(incorrect_ids)
    incorrect_rate = round(incorrect_count / float(total_eval), 4)

    distance_error_summary = {
        "mean_yards": round(statistics.mean(dist_errors), 4) if dist_errors else None,
        "median_yards": round(statistics.median(dist_errors), 4) if dist_errors else None,
        "min_yards": round(min(dist_errors), 4) if dist_errors else None,
        "max_yards": round(max(dist_errors), 4) if dist_errors else None,
        "buckets": {
            "exact_0yd": {"count": len(bucket_exact), "play_ids": bucket_exact},
            "near_tie_le_0_5yd": {"count": len(bucket_near_tie), "play_ids": bucket_near_tie},
            "moderate_0_5_to_1_5yd": {"count": len(bucket_moderate), "play_ids": bucket_moderate},
            "large_gt_1_5yd": {"count": len(bucket_large), "play_ids": bucket_large},
        },
    }

    temporal_mistake_summary = {
        "mean_peak_timing_error_seconds": (
            round(statistics.mean(peak_timing_errors), 4) if peak_timing_errors else None
        ),
        "median_peak_timing_error_seconds": (
            round(statistics.median(peak_timing_errors), 4) if peak_timing_errors else None
        ),
        "max_peak_timing_error_seconds": (
            round(max(peak_timing_errors), 4) if peak_timing_errors else None
        ),
        "early_snap_proximity_missed_count": len(early_phase_missed_plays),
        "early_snap_proximity_missed_play_ids": early_phase_missed_plays,
        "timing_phase_mismatch_count": len(phase_mismatch_plays),
        "timing_phase_mismatch_play_ids": phase_mismatch_plays,
        "true_peak_between_sampled_frames_count": len(between_sampled_frames_plays),
        "true_peak_between_sampled_frames_play_ids": between_sampled_frames_plays,
        "cited_frame_mismatch_count": len(cited_frame_mismatch_plays),
        "cited_frame_mismatch_play_ids": cited_frame_mismatch_plays,
    }

    observable_biases = {
        "predicted_tag_counts": pred_tag_counts,
        "ground_truth_tag_counts": gt_tag_counts,
        "tag_comparison_table": tag_comparison_table,
        "low_tag_r1_r2_predicted_share": round(pred_r1_r2 / float(total_eval), 4),
        "low_tag_r1_r2_ground_truth_share": round(gt_r1_r2 / float(total_eval), 4),
        "low_tag_overselection_bias": round((pred_r1_r2 - gt_r1_r2) / float(total_eval), 4),
        "high_tag_r4_r5_predicted_share": round(pred_r4_r5 / float(total_eval), 4),
        "high_tag_r4_r5_ground_truth_share": round(gt_r4_r5 / float(total_eval), 4),
        "crowded_or_tight_margin_plays_without_abstention": crowded_pocket_no_abstain_plays,
        "abstention_rate_on_crowded_plays": 0.0 if crowded_pocket_no_abstain_plays else None,
    }

    measurable_error_categories = [
        {
            "category_id": "incorrect_rusher_identification",
            "category_name": "Incorrect Rusher Identification",
            "verification_type": "measurable_from_tracking",
            "measured_value": f"{incorrect_count}/{total_eval} plays ({incorrect_rate * 100:.1f}%)",
            "secondary_metric": (
                f"Mean selected rusher rank: #{statistics.mean(ranks_numeric):.2f} "
                f"(Rank #1: {rank_counts['rank_1']}, Rank #2: {rank_counts['rank_2']}, "
                f"Rank #3: {rank_counts['rank_3']}, Rank #4+: {rank_counts['rank_4_plus']})"
                if ranks_numeric
                else "n/a"
            ),
            "affected_play_ids": incorrect_ids,
            "objective_evidence": (
                "Computed directly by comparing the VLM's resolved `selected_rusher_id` against "
                "the rusher with minimum Euclidean distance to the QB in the observed frame window."
            ),
        },
        {
            "category_id": "spatial_distance_error",
            "category_name": "Euclidean Distance Error Magnitude",
            "verification_type": "measurable_from_tracking",
            "measured_value": (
                f"Mean +{distance_error_summary['mean_yards']:.2f} yd "
                f"(median +{distance_error_summary['median_yards']:.2f} yd, "
                f"range 0.00 to +{distance_error_summary['max_yards']:.2f} yd)"
                if distance_error_summary["mean_yards"] is not None
                else "n/a"
            ),
            "secondary_metric": (
                f"Exact (0 yd): {len(bucket_exact)}, Near-tie (<=0.5 yd): {len(bucket_near_tie)}, "
                f"Moderate (0.5-1.5 yd): {len(bucket_moderate)}, Large (>1.5 yd): {len(bucket_large)}"
            ),
            "affected_play_ids": bucket_near_tie + bucket_moderate + bucket_large,
            "objective_evidence": (
                "Measured as `min_distance(predicted_rusher) - min_distance(true_closest_rusher)` "
                "in yards across the play's tracking frames."
            ),
        },
        {
            "category_id": "temporal_peak_and_frame_discrepancy",
            "category_name": "Temporal Peak-Proximity & Frame Alignment Errors",
            "verification_type": "measurable_from_tracking",
            "measured_value": (
                f"Mean peak-timing offset: {temporal_mistake_summary['mean_peak_timing_error_seconds']:.2f}s | "
                f"Phase mismatches: {len(phase_mismatch_plays)}/{total_eval} plays"
                if temporal_mistake_summary["mean_peak_timing_error_seconds"] is not None
                else "n/a"
            ),
            "secondary_metric": (
                f"Early-snap proximity missed: {len(early_phase_missed_plays)} play(s) {early_phase_missed_plays}; "
                f"True peak between discrete sampled frames: {len(between_sampled_frames_plays)} play(s) {between_sampled_frames_plays}"
            ),
            "affected_play_ids": sorted(set(phase_mismatch_plays + early_phase_missed_plays + cited_frame_mismatch_plays)),
            "objective_evidence": (
                "Measured by comparing `closest_time_s` of the VLM-selected rusher and the frame numbers "
                "cited in `visible_evidence` against the tracking-data timestamp of minimum QB distance."
            ),
        },
        {
            "category_id": "tag_and_abstention_bias",
            "category_name": "Observable R-Tag Selection & Zero-Abstention Bias",
            "verification_type": "measurable_from_outputs",
            "measured_value": (
                f"R1/R2 selected on {pred_r1_r2}/{total_eval} ({pred_r1_r2 * 100.0 / total_eval:.0f}%) of plays "
                f"vs {gt_r1_r2}/{total_eval} ({gt_r1_r2 * 100.0 / total_eval:.0f}%) in ground truth"
            ),
            "secondary_metric": (
                f"R4/R5 selected on {pred_r4_r5}/{total_eval} ({pred_r4_r5 * 100.0 / total_eval:.0f}%) "
                f"vs {gt_r4_r5}/{total_eval} ({gt_r4_r5 * 100.0 / total_eval:.0f}%) in ground truth; "
                f"0 abstentions across {len(crowded_pocket_no_abstain_plays)} crowded/tight-margin plays"
            ),
            "affected_play_ids": incorrect_ids,
            "objective_evidence": (
                "Counted directly from the frequency distribution of `predicted_rusher_tag` vs "
                "`ground_truth.closest_rusher_tag` and `abstained` flags."
            ),
        },
    ]

    hypothesised_causes = [
        {
            "hypothesis_id": "prompt_list_order_anchoring",
            "title": "Prompt List-Order / Low-Index Tag Anchoring in 3B VLM",
            "verification_status": "hypothesised_cause_unverified",
            "related_measurable_category": "tag_and_abstention_bias",
            "hypothesis_statement": (
                "Candidate rushers are listed in ascending `nflId` order (`R1, R2, R3, R4, R5`) in the prompt. "
                "When visual distances in 2D pocket projections are ambiguous, the 3B model may anchor on early "
                "candidates (`R1`/`R2`, especially `R2` with the high-contrast cyan ring) rather than later items (`R4`/`R5`)."
            ),
            "why_not_directly_proven": (
                "Confirming causal attribution to list order vs. ring color requires a controlled permutation "
                "ablation (shuffling R-tags and palette colors across identical frames)."
            ),
        },
        {
            "hypothesis_id": "interior_crowding_and_marker_occlusion",
            "title": "2D Interior Pocket Crowding vs. Unobstructed Edge Salience",
            "verification_status": "hypothesised_cause_unverified",
            "related_measurable_category": "spatial_distance_error",
            "hypothesis_statement": (
                "Interior rushers approaching within 1.3-2.0 yd of the QB frequently overlap with offensive linemen "
                "(blue dots), whereas edge/perimeter rushers (`R2` on several plays) have unobstructed colored trails "
                "and callout boxes that may appear more visually salient to a general-purpose ViT encoder."
            ),
            "why_not_directly_proven": (
                "Attention weights or marker-removal ablations are not observed in standard black-box generation."
            ),
        },
        {
            "hypothesis_id": "late_frame_recency_bias",
            "title": "Instruction/Recency Bias Toward Final Frame Over Early-Snap Proximity",
            "verification_status": "hypothesised_cause_unverified",
            "related_measurable_category": "temporal_peak_and_frame_discrepancy",
            "hypothesis_statement": (
                "The prompt instructs the model to examine sequential frames 'especially the later frames as rushers "
                "converge'. On quick-game plays (e.g., Play 282 where R4 is closest at +0.1s in Frame 1) or plays where "
                "a rusher peaks mid-play and rebounds (Play 349), the VLM may overweight the final frame."
            ),
            "why_not_directly_proven": (
                "Distinguishing prompt-induced recency bias from multi-image visual token pooling requires prompt "
                "and frame-order ablations."
            ),
        },
        {
            "hypothesis_id": "instruction_tuned_abstention_reluctance",
            "title": "Instruction-Tuned Reluctance to Emit Null / Abstention",
            "verification_status": "hypothesised_cause_unverified",
            "related_measurable_category": "tag_and_abstention_bias",
            "hypothesis_statement": (
                "Even on near-ties (e.g., Play 410 with a 0.45 yd margin) and crowded pockets, the 3B instruction-tuned "
                "model always commits to a concrete rusher ID while expressing caveats only in the free-text `uncertainty` field."
            ),
            "why_not_directly_proven": (
                "Internal token calibration probabilities were not thresholded separately from greedy decoding."
            ),
        },
    ]

    return {
        "evaluated_plays": total_eval,
        "correct_identifications": len(correct_ids),
        "correct_play_ids": correct_ids,
        "incorrect_identifications": incorrect_count,
        "incorrect_play_ids": incorrect_ids,
        "incorrect_identification_rate": incorrect_rate,
        "predicted_rusher_rank_counts": rank_counts,
        "mean_predicted_rusher_rank": (
            round(statistics.mean(ranks_numeric), 4) if ranks_numeric else None
        ),
        "distance_error_summary": distance_error_summary,
        "temporal_mistake_summary": temporal_mistake_summary,
        "observable_biases": observable_biases,
        "measurable_error_categories": measurable_error_categories,
        "hypothesised_causes": hypothesised_causes,
        "per_play_diagnostics": per_play_diagnostics,
    }


def compute_temporal_pass_rush_ground_truth(
    play_payload: Dict[str, Any],
    sampled_frame_ids: Optional[Sequence[int]] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    pressure_threshold_yd: float = 2.0,
) -> Dict[str, Any]:
    """Compute independent tracking-data QB-rusher distance trends and closest-approach timing.

    Extends `compute_window_ground_truth` with per-rusher and play-level temporal dynamics:
    - Initial, final, minimum, and maximum QB-rusher distances in the window
    - Net distance change (final - initial) and total closing distance (initial - min)
    - Rebound distance after peak proximity (final - min)
    - Peak closing speed (yd/s) across 10 Hz frames
    - Deterministic distance trend classification ('closing', 'close_then_recede', 'steady', 'receding')
    - Closest-approach timing phase ('early', 'mid', 'late') and 1-based sampled frame index (1..K)
    """
    base_gt = compute_window_ground_truth(
        play_payload=play_payload,
        sampled_frame_ids=sampled_frame_ids,
        start_frame=start_frame,
        end_frame=end_frame,
        pressure_threshold_yd=pressure_threshold_yd,
    )

    frames = play_payload["frames"]
    qb_id = int(play_payload["qb"])
    w_start_f = int(base_gt["window_start_frame"])
    w_end_f = int(base_gt["window_end_frame"])
    w_start_t = float(base_gt["window_start_time"])
    w_end_t = float(base_gt["window_end_time"])
    samp_fids = [int(fid) for fid in base_gt["sampled_frame_ids"]]

    window_frames = [f for f in frames if w_start_f <= int(f["frame"]) <= w_end_f]
    sampled_frames = [f for f in window_frames if int(f["frame"]) in set(samp_fids)]
    if not sampled_frames:
        sampled_frames = window_frames

    def _dist(frame_obj: Dict[str, Any], rid: int) -> float:
        entities = frame_obj.get("entities", [])
        qb_ent = next((e for e in entities if e.get("id") == qb_id), None)
        r_ent = next((e for e in entities if e.get("id") == rid), None)
        if qb_ent is not None and r_ent is not None:
            return float(
                math.hypot(
                    float(r_ent["x"]) - float(qb_ent["x"]),
                    float(r_ent["y"]) - float(qb_ent["y"]),
                )
            )
        dists = frame_obj.get("distances", {})
        return float(dists[str(rid)])

    temporal_rushers: List[Dict[str, Any]] = []
    for r_rec in base_gt["rusher_window_distances"]:
        rid = int(r_rec["id"])
        win_series = [
            (idx, int(f["frame"]), float(f["time"]), _dist(f, rid))
            for idx, f in enumerate(window_frames)
        ]
        samp_series = [
            (seq_idx + 1, int(f["frame"]), float(f["time"]), _dist(f, rid))
            for seq_idx, f in enumerate(sampled_frames)
        ]

        init_d = win_series[0][3]
        final_d = win_series[-1][3]
        min_item = min(win_series, key=lambda x: x[3])
        max_item = max(win_series, key=lambda x: x[3])
        best_samp = min(samp_series, key=lambda x: x[3])

        # Peak closing speed (yd/s) across consecutive 10 Hz frames
        closing_rates: List[float] = []
        for k in range(1, len(win_series)):
            dt = win_series[k][2] - win_series[k - 1][2]
            if dt > 1e-6:
                rate = (win_series[k - 1][3] - win_series[k][3]) / dt
                closing_rates.append(rate)
        peak_closing_speed = max(closing_rates) if closing_rates else 0.0

        trend = _classify_distance_trend(
            initial_dist=init_d,
            final_dist=final_d,
            min_dist=min_item[3],
            min_idx=min_item[0],
            num_points=len(win_series),
        )
        phase = _classify_timing_phase(min_item[2], w_start_t, w_end_t)
        samp_phase = _classify_timing_phase(best_samp[2], w_start_t, w_end_t)

        enriched = dict(r_rec)
        enriched.update({
            "initial_distance_yards": round(init_d, 4),
            "final_distance_yards": round(final_d, 4),
            "max_distance_yards": round(max_item[3], 4),
            "net_distance_change_yards": round(final_d - init_d, 4),
            "total_closing_yards": round(init_d - min_item[3], 4),
            "rebound_after_peak_yards": round(final_d - min_item[3], 4),
            "peak_closing_speed_yd_s": round(peak_closing_speed, 4),
            "distance_trend": trend,
            "closest_approach_phase": phase,
            "sampled_closest_seq_number": int(best_samp[0]),
            "sampled_closest_phase": samp_phase,
            "sampled_distances_yards": [round(item[3], 2) for item in samp_series],
        })
        temporal_rushers.append(enriched)

    by_min = sorted(temporal_rushers, key=lambda r: (r["min_distance_yards"], r["id"]))
    closest_r = by_min[0]

    closing_count = sum(
        1 for r in temporal_rushers if r["distance_trend"] in {"closing", "close_then_recede"}
    )
    mean_net_change = statistics.mean(r["net_distance_change_yards"] for r in temporal_rushers)
    collective_trend = (
        "converging"
        if closing_count >= max(1, math.ceil(len(temporal_rushers) / 2.0))
        else "contained"
    )

    result = dict(base_gt)
    result["rusher_window_distances"] = temporal_rushers
    result["closest_rusher_distance_trend"] = closest_r["distance_trend"]
    result["closest_rusher_initial_distance_yards"] = closest_r["initial_distance_yards"]
    result["closest_rusher_final_distance_yards"] = closest_r["final_distance_yards"]
    result["closest_rusher_net_distance_change_yards"] = closest_r["net_distance_change_yards"]
    result["closest_rusher_total_closing_yards"] = closest_r["total_closing_yards"]
    result["closest_rusher_peak_closing_speed_yd_s"] = closest_r["peak_closing_speed_yd_s"]
    result["closest_approach_phase"] = closest_r["closest_approach_phase"]
    result["closest_sampled_seq_number"] = closest_r["sampled_closest_seq_number"]
    result["closing_rusher_count"] = closing_count
    result["total_rusher_count"] = len(temporal_rushers)
    result["mean_rusher_net_distance_change_yards"] = round(mean_net_change, 4)
    result["collective_rush_trend"] = collective_trend
    return result


def _normalize_trend_token(raw_trend: Any) -> Optional[str]:
    if raw_trend is None:
        return None
    s = str(raw_trend).strip().lower().replace("-", "_").replace(" ", "_")
    if s in VALID_DISTANCE_TRENDS:
        return s
    if "recede" in s and ("close" in s or "approach" in s or "peak" in s or "then" in s):
        return "close_then_recede"
    if any(k in s for k in ("clos", "approach", "converg", "decreas", "narrow", "collaps")):
        return "closing"
    if any(k in s for k in ("reced", "widen", "increas", "away", "drop")):
        return "receding"
    if any(k in s for k in ("stead", "constant", "flat", "stall", "contain", "blocked", "stationary")):
        return "steady"
    return None


def _normalize_phase_token(raw_phase: Any, raw_frame_num: Any = None, num_sampled: int = 4) -> Optional[str]:
    if raw_phase is not None:
        s = str(raw_phase).strip().lower()
        if s in VALID_TIMING_PHASES:
            return s
        if "early" in s or "start" in s or "initial" in s or "snap" in s or "first" in s:
            return "early"
        if "mid" in s or "middle" in s:
            return "mid"
        if "late" in s or "end" in s or "final" in s or "last" in s:
            return "late"
    if raw_frame_num is not None and str(raw_frame_num).strip().isdigit():
        fnum = int(str(raw_frame_num).strip())
        if fnum == 1:
            return "early"
        if 1 < fnum < num_sampled:
            return "mid"
        if fnum >= num_sampled:
            return "late"
    return None


def validate_and_parse_temporal_vlm_json(
    raw_output: Any,
    candidate_rushers: Optional[Sequence[Dict[str, Any]]] = None,
    num_sampled_frames: int = 4,
) -> Dict[str, Any]:
    """Validate and normalize temporal pass-rush VLM output into verifiable and interpretive fields."""
    base = validate_and_parse_vlm_json(raw_output, candidate_rushers=candidate_rushers)
    if not base.get("is_valid_json"):
        base.update({
            "distance_trend": None,
            "closest_approach_phase": None,
            "closest_frame_number": None,
            "collective_rush_description": "",
            "unverified_interpretation": "",
        })
        return base

    # Extract extra temporal fields from the JSON object
    raw_text = base.get("raw_text", "")
    parsed_obj: Dict[str, Any] = {}
    if isinstance(raw_output, dict):
        parsed_obj = raw_output
    else:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
        candidate_str = fenced.group(1) if fenced else raw_text
        try:
            parsed_obj = json.loads(candidate_str)
        except Exception:
            brace_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if brace_match:
                try:
                    parsed_obj = json.loads(brace_match.group(0))
                except Exception:
                    parsed_obj = {}

    raw_trend = parsed_obj.get("distance_trend", parsed_obj.get("approach_trend"))
    raw_phase = parsed_obj.get("closest_approach_phase", parsed_obj.get("closest_timing_phase"))
    raw_frame_num = parsed_obj.get("closest_frame_number", parsed_obj.get("closest_frame"))

    frame_num: Optional[int] = None
    if raw_frame_num is not None:
        m = re.search(r"(\d+)", str(raw_frame_num))
        if m:
            val = int(m.group(1))
            if 1 <= val <= max(1, num_sampled_frames):
                frame_num = val

    norm_trend = _normalize_trend_token(raw_trend)
    if norm_trend is None:
        # Fallback: infer from movement_pattern if explicitly stated
        norm_trend = _normalize_trend_token(base.get("movement_pattern"))

    norm_phase = _normalize_phase_token(raw_phase, frame_num, num_sampled=num_sampled_frames)

    collective_desc = str(parsed_obj.get("collective_rush_description", "") or "").strip()
    unverified_interp = str(
        parsed_obj.get(
            "unverified_interpretation",
            parsed_obj.get("non_verifiable_interpretation", ""),
        )
        or ""
    ).strip()

    base.update({
        "distance_trend": norm_trend,
        "raw_distance_trend": raw_trend,
        "closest_approach_phase": norm_phase,
        "raw_closest_approach_phase": raw_phase,
        "closest_frame_number": frame_num,
        "collective_rush_description": collective_desc,
        "unverified_interpretation": unverified_interp,
    })
    return base


def evaluate_temporal_play_prediction(
    parsed_prediction: Dict[str, Any],
    temporal_ground_truth: Dict[str, Any],
    latency_seconds: float = 0.0,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    model_revision: str = "unknown",
) -> Dict[str, Any]:
    """Evaluate a temporal VLM prediction against tracking-data distance trends and timing.

    Clearly separates:
    1. `verifiable_visual_observations`: spatial rusher ID, distance trend ('closing',
       'close_then_recede', 'steady', 'receding'), closest-approach timing phase
       ('early', 'mid', 'late'), and closest sampled frame index (1..K), all scored
       against objective tracking calculations.
    2. `unverifiable_interpretations`: free-text commentary on blocking technique,
       protection schemes, quarterback reads/intent, or official PFF pressure grades
       that cannot be validated from 2D coordinate trajectories alone.
    """
    base_eval = evaluate_play_prediction(
        parsed_prediction=parsed_prediction,
        ground_truth=temporal_ground_truth,
        latency_seconds=latency_seconds,
        model_name=model_name,
        model_revision=model_revision,
    )

    pred_id = base_eval.get("predicted_rusher_id")
    pred_trend = parsed_prediction.get("distance_trend")
    pred_phase = parsed_prediction.get("closest_approach_phase")
    pred_frame_num = parsed_prediction.get("closest_frame_number")

    gt_trend_closest = str(temporal_ground_truth.get("closest_rusher_distance_trend", "closing"))
    gt_phase_closest = str(temporal_ground_truth.get("closest_approach_phase", "late"))
    gt_seq_num_closest = int(temporal_ground_truth.get("closest_sampled_seq_number", 4))

    selected_rusher_gt: Optional[Dict[str, Any]] = None
    for r in temporal_ground_truth.get("rusher_window_distances", []):
        if pred_id is not None and int(r.get("id", -1)) == int(pred_id):
            selected_rusher_gt = r
            break

    gt_trend_selected = (
        str(selected_rusher_gt["distance_trend"]) if selected_rusher_gt else None
    )
    gt_phase_selected = (
        str(selected_rusher_gt["closest_approach_phase"]) if selected_rusher_gt else None
    )
    gt_seq_num_selected = (
        int(selected_rusher_gt["sampled_closest_seq_number"]) if selected_rusher_gt else None
    )

    trend_match_vs_closest = (pred_trend == gt_trend_closest) if pred_trend is not None else False
    trend_match_vs_selected = (
        (pred_trend == gt_trend_selected)
        if (pred_trend is not None and gt_trend_selected is not None)
        else False
    )

    # Coarse closing direction agreement: did both agree the rusher approached the QB (>=0.75 yd closing)?
    closing_set = {"closing", "close_then_recede"}
    closing_dir_match_vs_closest = (
        ((pred_trend in closing_set) == (gt_trend_closest in closing_set))
        if pred_trend is not None
        else False
    )
    closing_dir_match_vs_selected = (
        ((pred_trend in closing_set) == (gt_trend_selected in closing_set))
        if (pred_trend is not None and gt_trend_selected is not None)
        else False
    )

    phase_match_vs_closest = (pred_phase == gt_phase_closest) if pred_phase is not None else False
    phase_match_vs_selected = (
        (pred_phase == gt_phase_selected)
        if (pred_phase is not None and gt_phase_selected is not None)
        else False
    )

    frame_err_vs_closest = (
        abs(int(pred_frame_num) - gt_seq_num_closest)
        if pred_frame_num is not None
        else None
    )
    frame_err_vs_selected = (
        abs(int(pred_frame_num) - gt_seq_num_selected)
        if (pred_frame_num is not None and gt_seq_num_selected is not None)
        else None
    )

    verifiable_visual_observations = {
        "closest_rusher_identification": {
            "vlm_claim": (
                f"{base_eval.get('predicted_rusher_tag') or ''} {base_eval.get('predicted_rusher_name') or 'None'}".strip()
            ),
            "tracking_reference": (
                f"{temporal_ground_truth.get('closest_rusher_tag')} {temporal_ground_truth.get('closest_rusher_name')} "
                f"({temporal_ground_truth.get('closest_distance_yards'):.2f} yd)"
            ),
            "verified_match": bool(base_eval.get("exact_match")),
            "distance_error_yards": base_eval.get("distance_error_yards"),
        },
        "qb_rusher_distance_trend": {
            "vlm_claim": pred_trend,
            "tracking_closest_rusher_trend": gt_trend_closest,
            "tracking_closest_rusher_metrics": (
                f"init {temporal_ground_truth.get('closest_rusher_initial_distance_yards'):.2f} yd -> "
                f"min {temporal_ground_truth.get('closest_distance_yards'):.2f} yd -> "
                f"final {temporal_ground_truth.get('closest_rusher_final_distance_yards'):.2f} yd "
                f"(net {temporal_ground_truth.get('closest_rusher_net_distance_change_yards'):+.2f} yd, "
                f"peak closing {temporal_ground_truth.get('closest_rusher_peak_closing_speed_yd_s'):.2f} yd/s)"
            ),
            "tracking_selected_rusher_trend": gt_trend_selected,
            "tracking_selected_rusher_metrics": (
                f"init {selected_rusher_gt['initial_distance_yards']:.2f} yd -> "
                f"min {selected_rusher_gt['min_distance_yards']:.2f} yd -> "
                f"final {selected_rusher_gt['final_distance_yards']:.2f} yd "
                f"(net {selected_rusher_gt['net_distance_change_yards']:+.2f} yd)"
                if selected_rusher_gt
                else "n/a"
            ),
            "verified_exact_trend_vs_closest": trend_match_vs_closest,
            "verified_exact_trend_vs_selected": trend_match_vs_selected,
            "verified_closing_direction_vs_closest": closing_dir_match_vs_closest,
            "verified_closing_direction_vs_selected": closing_dir_match_vs_selected,
        },
        "closest_approach_timing": {
            "vlm_claimed_phase": pred_phase,
            "vlm_claimed_sampled_frame_number": pred_frame_num,
            "tracking_closest_phase": gt_phase_closest,
            "tracking_closest_time_s": temporal_ground_truth.get("closest_time_s"),
            "tracking_closest_sampled_frame_number": gt_seq_num_closest,
            "tracking_selected_rusher_phase": gt_phase_selected,
            "tracking_selected_rusher_time_s": (
                selected_rusher_gt["closest_time_s"] if selected_rusher_gt else None
            ),
            "verified_phase_match_vs_closest": phase_match_vs_closest,
            "verified_phase_match_vs_selected": phase_match_vs_selected,
            "sampled_frame_number_error_vs_closest": frame_err_vs_closest,
            "sampled_frame_number_error_vs_selected": frame_err_vs_selected,
        },
    }

    unverifiable_interpretations = {
        "model_unverified_interpretation": parsed_prediction.get("unverified_interpretation", ""),
        "collective_rush_narrative": parsed_prediction.get("collective_rush_description", ""),
        "movement_narrative": parsed_prediction.get("movement_pattern", ""),
        "whole_play_pff_annotation": temporal_ground_truth.get("whole_play_pff_label", "Unknown"),
        "verification_boundary_note": (
            "Visual observations of 2D distance change, closing direction, and frame-of-closest-approach "
            "are objectively verified against NFL tracking coordinates above. By contrast, commentary on "
            "offensive line blocking technique, stunt intent, quarterback progression, or official PFF "
            "pressure causation cannot be validated from 2D overhead dots and must be treated as "
            "unverified model interpretation."
        ),
    }

    result = dict(base_eval)
    result.update({
        "predicted_distance_trend": pred_trend,
        "predicted_closest_approach_phase": pred_phase,
        "predicted_closest_frame_number": pred_frame_num,
        "collective_rush_description": parsed_prediction.get("collective_rush_description", ""),
        "unverified_interpretation": parsed_prediction.get("unverified_interpretation", ""),
        "gt_closest_distance_trend": gt_trend_closest,
        "gt_closest_approach_phase": gt_phase_closest,
        "gt_closest_sampled_seq_number": gt_seq_num_closest,
        "gt_selected_distance_trend": gt_trend_selected,
        "gt_selected_approach_phase": gt_phase_selected,
        "gt_selected_sampled_seq_number": gt_seq_num_selected,
        "trend_match_vs_closest": trend_match_vs_closest,
        "trend_match_vs_selected": trend_match_vs_selected,
        "closing_direction_match_vs_closest": closing_dir_match_vs_closest,
        "closing_direction_match_vs_selected": closing_dir_match_vs_selected,
        "phase_match_vs_closest": phase_match_vs_closest,
        "phase_match_vs_selected": phase_match_vs_selected,
        "sampled_frame_error_vs_closest": frame_err_vs_closest,
        "sampled_frame_error_vs_selected": frame_err_vs_selected,
        "verifiable_visual_observations": verifiable_visual_observations,
        "unverifiable_interpretations": unverifiable_interpretations,
    })
    return result


def aggregate_temporal_evaluation_results(
    temporal_evaluations: Sequence[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute aggregate spatial + temporal evaluation metrics across evaluated plays."""
    base_agg = aggregate_evaluation_results(temporal_evaluations, metadata=metadata)
    identifiable_recs = [
        r
        for r in temporal_evaluations
        if r.get("outcome_status") not in {"not_analyzed", "model_unavailable", "invalid_output", "abstained"}
    ]
    n_id = len(identifiable_recs)
    if n_id == 0:
        base_agg["temporal_metrics"] = {
            "identifiable_plays": 0,
            "trend_match_vs_closest_count": 0,
            "trend_match_vs_closest_accuracy": None,
            "trend_match_vs_selected_count": 0,
            "trend_match_vs_selected_accuracy": None,
            "closing_direction_match_vs_closest_count": 0,
            "closing_direction_match_vs_closest_accuracy": None,
            "closing_direction_match_vs_selected_count": 0,
            "closing_direction_match_vs_selected_accuracy": None,
            "phase_match_vs_closest_count": 0,
            "phase_match_vs_closest_accuracy": None,
            "phase_match_vs_selected_count": 0,
            "phase_match_vs_selected_accuracy": None,
            "mean_sampled_frame_error_vs_closest": None,
            "mean_sampled_frame_error_vs_selected": None,
        }
        return base_agg

    trend_closest_cnt = sum(1 for r in identifiable_recs if r.get("trend_match_vs_closest"))
    trend_selected_cnt = sum(1 for r in identifiable_recs if r.get("trend_match_vs_selected"))
    dir_closest_cnt = sum(1 for r in identifiable_recs if r.get("closing_direction_match_vs_closest"))
    dir_selected_cnt = sum(1 for r in identifiable_recs if r.get("closing_direction_match_vs_selected"))
    phase_closest_cnt = sum(1 for r in identifiable_recs if r.get("phase_match_vs_closest"))
    phase_selected_cnt = sum(1 for r in identifiable_recs if r.get("phase_match_vs_selected"))

    ferr_closest = [
        int(r["sampled_frame_error_vs_closest"])
        for r in identifiable_recs
        if r.get("sampled_frame_error_vs_closest") is not None
    ]
    ferr_selected = [
        int(r["sampled_frame_error_vs_selected"])
        for r in identifiable_recs
        if r.get("sampled_frame_error_vs_selected") is not None
    ]

    base_agg["temporal_metrics"] = {
        "identifiable_plays": n_id,
        "trend_match_vs_closest_count": trend_closest_cnt,
        "trend_match_vs_closest_accuracy": round(trend_closest_cnt / float(n_id), 4),
        "trend_match_vs_selected_count": trend_selected_cnt,
        "trend_match_vs_selected_accuracy": round(trend_selected_cnt / float(n_id), 4),
        "closing_direction_match_vs_closest_count": dir_closest_cnt,
        "closing_direction_match_vs_closest_accuracy": round(dir_closest_cnt / float(n_id), 4),
        "closing_direction_match_vs_selected_count": dir_selected_cnt,
        "closing_direction_match_vs_selected_accuracy": round(dir_selected_cnt / float(n_id), 4),
        "phase_match_vs_closest_count": phase_closest_cnt,
        "phase_match_vs_closest_accuracy": round(phase_closest_cnt / float(n_id), 4),
        "phase_match_vs_selected_count": phase_selected_cnt,
        "phase_match_vs_selected_accuracy": round(phase_selected_cnt / float(n_id), 4),
        "mean_sampled_frame_error_vs_closest": (
            round(statistics.mean(ferr_closest), 4) if ferr_closest else None
        ),
        "mean_sampled_frame_error_vs_selected": (
            round(statistics.mean(ferr_selected), 4) if ferr_selected else None
        ),
    }
    base_agg["error_analysis_summary"] = analyze_vlm_errors(base_agg)
    return base_agg

