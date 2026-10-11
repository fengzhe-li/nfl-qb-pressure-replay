"""Automated test suite for QB Pressure Replay v2 and VLM analysis extensions.

Covers:
1. Frame generation (sequential images, dimensions, saved files, data URIs)
2. Visual-input / ground-truth separation (strict stripping of distances, nearest, PFF, outcomes)
3. Output JSON validation (valid JSON, markdown-fenced JSON, abstention, malformed JSON, missing fields)
4. Correct tracking-data reference calculation (window-specific ground truth vs. whole-play summary)
5. VLM result matching (exact match, mismatch, near-tie, abstention, invalid output)
6. Missing or invalid inputs (error handling across generator, evaluator, and replay builder)
7. Model-unavailable fallback (graceful degradation when model cannot load or errors)
8. Preservation of original replay functionality (prepare_play, build_replay, HTML output, excluded CSV)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import math
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qb_pressure_replay as qbr
from vlm_evaluation import (
    aggregate_evaluation_results,
    aggregate_temporal_evaluation_results,
    analyze_vlm_errors,
    compute_temporal_pass_rush_ground_truth,
    compute_window_ground_truth,
    evaluate_play_prediction,
    evaluate_temporal_play_prediction,
    resolve_rusher_identity,
    validate_and_parse_temporal_vlm_json,
    validate_and_parse_vlm_json,
)
from vlm_frame_generator import (
    FORBIDDEN_ENTITY_KEYS,
    FORBIDDEN_FRAME_KEYS,
    FORBIDDEN_PLAY_KEYS,
    FrameSequenceBundle,
    generate_play_frames,
    sanitize_play_for_visual_input,
    select_representative_frame_indices,
    verify_no_ground_truth_leakage,
)
from vlm_inference import (
    DEFAULT_MODEL_ID,
    QwenVLMAnalyzer,
    build_temporal_vlm_prompt,
    build_vlm_prompt,
)
from telemetry_comparison import (
    CROSS_PLAY_DISCLAIMER,
    QUANTITY_DISTINCTION_NOTE,
    NormalizedPlayTelemetryTrace,
    align_cross_play_traces,
    build_compact_cross_game_play_payload,
    build_cross_play_comparison,
    build_normalized_play_trace,
    build_play_telemetry_bundle,
    compute_distance_chart_render_state,
    compute_head_to_head_gap_analysis,
    compute_multi_play_player_analytics,
    compute_player_telemetry_series,
    compute_segment_head_to_head_comparison,
    compute_timeline_segments,
    evaluate_cross_play_at_time,
    extract_play_rusher_records,
    hit_test_distance_chart_point,
    interpolate_visual_frame,
    resolve_play_rusher,
    seek_timeline_position,
    validate_cross_game_record_identity,
)
from vlm_pipeline import attach_vlm_analysis_to_plays, run_temporal_vlm_experiment


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture()
def synthetic_play_payload() -> Dict[str, Any]:
    """Create a realistic multi-frame play payload where early-window and late-window closest rushers differ."""
    # QB at (30.0, 26.65)
    # Early frames (frames 10, 11): Rusher 101 (Alpha) is closer (3.0 yd) than Rusher 202 (Beta, 6.0 yd)
    # Late frames (frames 12, 13): Rusher 202 (Beta) rushes in to 1.5 yd while Rusher 101 stays at 3.5 yd
    frames = []
    specs = [
        # (frameId, time_s, r101_x, r101_y, r202_x, r202_y)
        (10, 0.0, 33.0, 26.65, 36.0, 26.65),
        (11, 0.1, 32.8, 26.65, 34.5, 26.65),
        (12, 0.2, 33.2, 26.65, 32.0, 26.65),
        (13, 0.3, 33.5, 26.65, 31.5, 26.65),
    ]
    for fid, t, x1, y1, x2, y2 in specs:
        d1 = math.hypot(x1 - 30.0, y1 - 26.65)
        d2 = math.hypot(x2 - 30.0, y2 - 26.65)
        nearest = 101 if d1 <= d2 else 202
        frames.append({
            "frame": fid,
            "time": t,
            "entities": [
                {"id": 99, "x": 30.0, "y": 26.65, "team": "TB", "name": "Tom Brady", "number": "12", "role": "Pass"},
                {"id": 101, "x": x1, "y": y1, "team": "DAL", "name": "Alpha Rusher", "number": "90", "role": "Pass Rush"},
                {"id": 202, "x": x2, "y": y2, "team": "DAL", "name": "Beta Defender", "number": "94", "role": "Pass Rush"},
                {"id": 303, "x": 29.0, "y": 25.0, "team": "TB", "name": "Left Tackle", "number": "76", "role": "Pass Block"},
                {"id": None, "x": 30.2, "y": 26.65, "team": "football", "name": "Football", "number": "", "role": "Football"},
            ],
            "distances": {"101": d1, "202": d2},
            "nearest": nearest,
        })

    return {
        "game": 2021090900,
        "play": 1001,
        "qb": 99,
        "offense": "TB",
        "defense": "DAL",
        "description": "(Shotgun) T.Brady pass incomplete [B.Defender].",
        "outcome": "I",
        "outcome_label": "Incomplete",
        "yards": 0.0,
        "end_event": "pass_forward",
        "end_label": "Pass released",
        "duration": 0.3,
        "frames": frames,
        "label": "Q1 | Test play | Incomplete | PFF: Hurry",
        "headline": "Tom Brady had 0.3 s before the pass released.",
        "summary": {
            "closest_yards": 1.5,
            "closest_name": "Beta Defender",
            "closest_id": 202,
            "closest_time": 0.3,
            "first_pressure": {"time": 0.3, "id": 202, "name": "Beta Defender"},
        },
        "rusher_stats": [
            {
                "id": 101,
                "name": "Alpha Rusher",
                "closest_yards": 2.8,
                "first_within": None,
                "pff_hit": False,
                "pff_hurry": False,
                "pff_sack": False,
                "pff_pressure": False,
            },
            {
                "id": 202,
                "name": "Beta Defender",
                "closest_yards": 1.5,
                "first_within": 0.3,
                "pff_hit": False,
                "pff_hurry": True,
                "pff_sack": False,
                "pff_pressure": True,
            },
        ],
        "pff_label": "Hurry",
        "pff_pressure": True,
        "flipped": False,
        "view": {"x0": 16.0, "x1": 46.0, "los": 31.0, "qb_y": 26.65},
        "rushers": [
            {"id": 202, "name": "Beta Defender"},
            {"id": 101, "name": "Alpha Rusher"},
        ],
        "pff_html": "<table></table>",
    }


# ---------------------------------------------------------------------------
# 1. Frame Generation Tests
# ---------------------------------------------------------------------------

def test_frame_generation_creates_valid_images_and_files(
    synthetic_play_payload: Dict[str, Any], tmp_path: Path
) -> None:
    bundle = generate_play_frames(
        synthetic_play_payload,
        output_dir=tmp_path / "frames",
        num_frames=3,
        width=640,
        height=480,
    )
    assert isinstance(bundle, FrameSequenceBundle)
    assert len(bundle.images) == 3
    assert len(bundle.image_paths) == 3
    assert bundle.sampled_frame_ids == [10, 12, 13]
    assert bundle.window_start_frame == 10
    assert bundle.window_end_frame == 13

    for img, path in zip(bundle.images, bundle.image_paths):
        assert img.size == (640, 480)
        assert path.is_file()
        assert path.stat().st_size > 1000

    uris = bundle.to_preview_data_uris(max_width=200)
    assert len(uris) == 3
    assert all(u.startswith("data:image/png;base64,") for u in uris)


# ---------------------------------------------------------------------------
# 2. Visual-Input / Ground-Truth Separation Tests
# ---------------------------------------------------------------------------

def test_visual_input_strictly_strips_ground_truth_and_pff(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    sanitized = sanitize_play_for_visual_input(synthetic_play_payload)
    assert verify_no_ground_truth_leakage(sanitized) is True

    for forbidden in FORBIDDEN_PLAY_KEYS:
        assert forbidden not in sanitized

    for f in sanitized["frames"]:
        for forbidden in FORBIDDEN_FRAME_KEYS:
            assert forbidden not in f
        for e in f["entities"]:
            for forbidden in FORBIDDEN_ENTITY_KEYS:
                assert forbidden not in e

    # Candidate rushers must be ordered deterministically by nflId (101 then 202), NOT by closeness
    assert [r["id"] for r in sanitized["candidate_rushers"]] == [101, 202]
    assert [r["tag"] for r in sanitized["candidate_rushers"]] == ["R1", "R2"]

    # Prompt must not contain distances, PFF outcomes, or play descriptions
    prompt = build_vlm_prompt(
        sanitized["candidate_rushers"], sampled_times=[0.0, 0.1, 0.2, 0.3]
    )
    assert "1.5" not in prompt
    assert "2.8" not in prompt
    assert "Hurry" not in prompt
    assert "Alpha Rusher" in prompt
    assert "Beta Defender" in prompt


def test_leakage_detector_catches_injected_ground_truth(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    sanitized = sanitize_play_for_visual_input(synthetic_play_payload)
    sanitized["frames"][0]["distances"] = {"101": 3.0}
    with pytest.raises(AssertionError, match="Forbidden key 'distances'"):
        verify_no_ground_truth_leakage(sanitized)

    bad_rushers = [{"tag": "R1", "id": 101, "name": "Alpha", "closest_yards": 2.8}]
    with pytest.raises(ValueError, match="forbidden key 'closest_yards'"):
        build_vlm_prompt(bad_rushers)


# ---------------------------------------------------------------------------
# 3. Output JSON Validation Tests
# ---------------------------------------------------------------------------

def test_json_validation_valid_and_fenced_and_invalid(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    sanitized = sanitize_play_for_visual_input(synthetic_play_payload)
    candidates = sanitized["candidate_rushers"]

    # Valid JSON wrapped in markdown code fence, identifying by tag "R2"
    raw_fenced = """```json
    {
      "selected_rusher_id": "R2",
      "selected_rusher_name": "Beta Defender",
      "abstained": false,
      "movement_pattern": "Direct edge collapse into the pocket from the right.",
      "visible_evidence": "R2 #94 reaches the yellow QB marker in Frame 4.",
      "uncertainty": "Low uncertainty.",
      "explanation": "Beta Defender closes directly on the quarterback by the final frame."
    }
    ```"""
    parsed = validate_and_parse_vlm_json(raw_fenced, candidate_rushers=candidates)
    assert parsed["is_valid_json"] is True
    assert parsed["status"] == "ok"
    assert parsed["abstained"] is False
    assert parsed["selected_rusher_id"] == 202
    assert parsed["selected_rusher_tag"] == "R2"

    # Valid abstention JSON
    raw_abstain = json.dumps({
        "selected_rusher_id": None,
        "selected_rusher_name": None,
        "abstained": True,
        "movement_pattern": "Multiple rushers congestion around the pocket.",
        "visible_evidence": "R1 and R2 appear equidistant in early frames.",
        "uncertainty": "Cannot distinguish closest rusher due to overlap.",
    })
    parsed_abs = validate_and_parse_vlm_json(raw_abstain, candidate_rushers=candidates)
    assert parsed_abs["is_valid_json"] is True
    assert parsed_abs["status"] == "abstained"
    assert parsed_abs["abstained"] is True
    assert parsed_abs["selected_rusher_id"] is None

    # Invalid malformed text
    parsed_bad = validate_and_parse_vlm_json("Not a JSON object at all", candidate_rushers=candidates)
    assert parsed_bad["is_valid_json"] is False
    assert parsed_bad["status"] == "invalid_output"

    # Missing required fields
    parsed_incomplete = validate_and_parse_vlm_json(
        json.dumps({"selected_rusher_id": 101}), candidate_rushers=candidates
    )
    assert parsed_incomplete["is_valid_json"] is False
    assert parsed_incomplete["status"] == "invalid_output"


# ---------------------------------------------------------------------------
# 4. Window-Specific Ground-Truth Reference Calculation Tests
# ---------------------------------------------------------------------------

def test_window_ground_truth_matches_observed_window_not_whole_play(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    # Whole-play closest is Rusher 202 (1.5 yd at frame 13)
    full_gt = compute_window_ground_truth(synthetic_play_payload)
    assert full_gt["closest_rusher_id"] == 202
    assert math.isclose(full_gt["closest_distance_yards"], 1.5, rel_tol=1e-4)
    assert full_gt["entered_2yd_zone_in_window"] is True

    # Restrict observed window to early frames [10, 11] where Rusher 101 (2.8 yd) is closer than Rusher 202 (4.5 yd)
    early_gt = compute_window_ground_truth(
        synthetic_play_payload, sampled_frame_ids=[10, 11], start_frame=10, end_frame=11
    )
    assert early_gt["closest_rusher_id"] == 101
    assert early_gt["closest_rusher_name"] == "Alpha Rusher"
    assert math.isclose(early_gt["closest_distance_yards"], 2.8, rel_tol=1e-4)
    assert early_gt["entered_2yd_zone_in_window"] is False
    assert early_gt["whole_play_pff_label"] == "Hurry"


# ---------------------------------------------------------------------------
# 5. VLM Result Matching & Aggregation Tests
# ---------------------------------------------------------------------------

def test_evaluation_matching_and_aggregate_metrics(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    gt = compute_window_ground_truth(synthetic_play_payload)
    candidates = sanitize_play_for_visual_input(synthetic_play_payload)["candidate_rushers"]

    # 1. Exact match
    p_match = validate_and_parse_vlm_json({
        "selected_rusher_id": 202,
        "movement_pattern": "Edge rush.",
        "visible_evidence": "Closest at frame 13.",
        "uncertainty": "None.",
    }, candidate_rushers=candidates)
    e1 = evaluate_play_prediction(p_match, gt, latency_seconds=2.5)
    assert e1["outcome_status"] == "match"
    assert e1["exact_match"] is True
    assert e1["distance_error_yards"] == 0.0

    # 2. Mismatch
    p_mismatch = validate_and_parse_vlm_json({
        "selected_rusher_id": 101,
        "movement_pattern": "Interior push.",
        "visible_evidence": "Moved toward pocket.",
        "uncertainty": "Moderate.",
    }, candidate_rushers=candidates)
    e2 = evaluate_play_prediction(p_mismatch, gt, latency_seconds=3.0)
    assert e2["outcome_status"] == "mismatch"
    assert e2["exact_match"] is False
    assert math.isclose(e2["distance_error_yards"], 2.8 - 1.5, rel_tol=1e-3)

    # 3. Abstained
    p_abs = validate_and_parse_vlm_json({
        "selected_rusher_id": None,
        "abstained": True,
        "movement_pattern": "Crowded line.",
        "visible_evidence": "Occluded.",
        "uncertainty": "High.",
    }, candidate_rushers=candidates)
    e3 = evaluate_play_prediction(p_abs, gt, latency_seconds=2.0)
    assert e3["outcome_status"] == "abstained"
    assert e3["abstained"] is True

    # 4. Invalid output
    p_inv = validate_and_parse_vlm_json("bad json", candidate_rushers=candidates)
    e4 = evaluate_play_prediction(p_inv, gt, latency_seconds=1.5)
    assert e4["outcome_status"] == "invalid_output"
    assert e4["invalid_output"] is True

    agg = aggregate_evaluation_results([e1, e2, e3, e4])
    assert agg["total_plays"] == 4
    assert agg["identifiable_predictions"] == 2
    assert agg["exact_match_count"] == 1
    assert agg["exact_match_accuracy"] == 0.5
    assert agg["abstention_count"] == 1
    assert agg["abstention_frequency"] == 0.25
    assert agg["invalid_output_count"] == 1
    assert agg["invalid_output_frequency"] == 0.25
    assert agg["mean_latency_seconds"] == 2.25


# ---------------------------------------------------------------------------
# 6. Missing or Invalid Inputs Tests
# ---------------------------------------------------------------------------

def test_missing_or_invalid_inputs_raise_clear_errors(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="non-empty 'frames'"):
        sanitize_play_for_visual_input({"qb": 99, "frames": []})

    with pytest.raises(ValueError, match="Invalid frame window"):
        select_representative_frame_indices(total_frames=5, num_frames=3, start_idx=4, end_idx=2)

    with pytest.raises(ValueError, match="No frames found in window"):
        compute_window_ground_truth(synthetic_play_payload, start_frame=999, end_frame=1005)


# ---------------------------------------------------------------------------
# 7. Model-Unavailable Fallback Tests (using mock/stub only in unit test)
# ---------------------------------------------------------------------------

def test_model_unavailable_fallback_preserves_workflow(
    synthetic_play_payload: Dict[str, Any], tmp_path: Path
) -> None:
    bundle = generate_play_frames(synthetic_play_payload, num_frames=2)
    analyzer = QwenVLMAnalyzer(model_id="nonexistent/offline-model-id", device="unavailable")
    analyzer._load_error = "Simulated offline environment"

    result = analyzer.analyze_frame_bundle(bundle)
    assert result["status"] == "model_unavailable"
    assert result["selected_rusher_id"] is None

    gt = compute_window_ground_truth(synthetic_play_payload, sampled_frame_ids=bundle.sampled_frame_ids)
    eval_rec = evaluate_play_prediction(result, gt)
    assert eval_rec["outcome_status"] == "model_unavailable"
    assert eval_rec["ground_truth"]["closest_rusher_id"] == 202


# ---------------------------------------------------------------------------
# 8. Preservation of Original Replay Functionality & Pipeline Integration Tests
# ---------------------------------------------------------------------------

def test_resolve_rusher_identity_supports_id_tag_name_and_jersey(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    candidates = sanitize_play_for_visual_input(synthetic_play_payload)["candidate_rushers"]

    # Match by integer ID
    assert resolve_rusher_identity(101, None, candidates)["id"] == 101
    # Match by R-tag
    assert resolve_rusher_identity("R2", None, candidates)["id"] == 202
    # Match by last name
    assert resolve_rusher_identity(None, "Defender", candidates)["id"] == 202
    # Match by jersey number
    assert resolve_rusher_identity("#90", None, candidates)["id"] == 101
    # Abstention / null tokens return None
    assert resolve_rusher_identity("null", "abstained", candidates) is None
    assert resolve_rusher_identity(999999, "Nonexistent Player", candidates) is None


def test_pipeline_attach_and_cache_persistence(
    synthetic_play_payload: Dict[str, Any], tmp_path: Path
) -> None:
    payload_copy = json.loads(json.dumps(synthetic_play_payload))

    def stub_generate_fn(images, prompt):
        return json.dumps({
            "selected_rusher_id": 202,
            "selected_rusher_tag": "R2",
            "selected_rusher_name": "Beta Defender",
            "abstained": False,
            "movement_pattern": "Direct edge collapse into the pocket.",
            "visible_evidence": "R2 is touching the yellow QB circle in the final frame.",
            "uncertainty": "Low uncertainty.",
            "explanation": "Beta Defender converges closest to the QB at the end of the sequence.",
        })

    stub_analyzer = QwenVLMAnalyzer(
        model_id=DEFAULT_MODEL_ID, custom_generate_fn=stub_generate_fn
    )
    summary = attach_vlm_analysis_to_plays(
        payloads=[payload_copy],
        output_folder=tmp_path,
        game_id=2021090900,
        run_vlm=True,
        max_vlm_plays=1,
        num_frames=3,
        analyzer=stub_analyzer,
    )
    assert summary["total_plays"] == 1
    assert summary["exact_match_count"] == 1
    assert summary["exact_match_accuracy"] == 1.0
    assert payload_copy["vlm_analysis"]["outcome_status"] == "match"
    assert (tmp_path / "vlm_evaluation_2021090900.json").is_file()

    # Second pass with run_vlm=False should automatically reload cached evaluation from disk
    payload_reloaded = json.loads(json.dumps(synthetic_play_payload))
    summary_cached = attach_vlm_analysis_to_plays(
        payloads=[payload_reloaded],
        output_folder=tmp_path,
        game_id=2021090900,
        run_vlm=False,
    )
    assert summary_cached["total_plays"] == 1
    assert payload_reloaded["vlm_analysis"]["outcome_status"] == "match"
    assert payload_reloaded["vlm_analysis"]["predicted_rusher_id"] == 202


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="Dataset folder not present")
def test_original_replay_functionality_preserved(tmp_path: Path) -> None:
    html_path, play_count = qbr.build_replay(DATA_DIR, "2021090900", tmp_path, run_vlm=False)
    assert play_count == 95
    assert html_path.is_file()
    excluded_path = tmp_path / "excluded_2021090900.csv"
    assert excluded_path.is_file()

    html_text = html_path.read_text(encoding="utf-8")
    # Verify original elements and controls remain intact alongside AI and Error Analysis panels
    for required_token in [
        'id="field"',
        'id="chart"',
        'id="playSelect"',
        'id="pffTable"',
        'id="gameTable"',
        'id="valid"',
        'id="aiPanel"',
        'id="vlmErrorPanel"',
        "Pressure zone = within 2 yd of the QB",
    ]:
        assert required_token in html_text


def test_real_vlm_evaluation_artifact_integrity() -> None:
    eval_file = Path(__file__).resolve().parent.parent / "example_evaluation_2021090900.json"
    if not eval_file.is_file():
        pytest.skip("example_evaluation_2021090900.json not yet generated")
    report = json.loads(eval_file.read_text(encoding="utf-8"))
    # Verify the frozen 10-play baseline is preserved intact
    assert report["total_plays"] == 10
    assert report["exact_match_count"] == 2
    assert report["exact_match_accuracy"] == 0.2
    assert math.isclose(report["mean_distance_error_yards"], 1.6007, rel_tol=1e-3)
    assert report["metadata"]["model_checkpoint"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert len(report["metadata"]["model_revision"]) >= 10
    assert report["metadata"]["tracking_game_id"] == 2021090900
    assert len(report["per_play_results"]) == report["total_plays"]
    for item in report["per_play_results"]:
        assert item["outcome_status"] in {"match", "mismatch", "abstained", "invalid_output"}
        assert "ground_truth" in item
        assert item["ground_truth"]["closest_distance_yards"] > 0


# ---------------------------------------------------------------------------
# 9. VLM Error Analysis Tests (Feature 1)
# ---------------------------------------------------------------------------

def test_vlm_error_analysis_on_frozen_baseline() -> None:
    eval_file = Path(__file__).resolve().parent.parent / "example_evaluation_2021090900.json"
    if not eval_file.is_file():
        pytest.skip("example_evaluation_2021090900.json not present")
    report = json.loads(eval_file.read_text(encoding="utf-8"))
    err = analyze_vlm_errors(report)

    assert err["evaluated_plays"] == 10
    assert err["correct_identifications"] == 2
    assert err["incorrect_identifications"] == 8
    assert err["incorrect_identification_rate"] == 0.8
    assert err["correct_play_ids"] == [187, 456]

    # Distance error buckets
    buckets = err["distance_error_summary"]["buckets"]
    assert buckets["exact_0yd"]["play_ids"] == [187, 456]
    assert buckets["near_tie_le_0_5yd"]["play_ids"] == [410]
    assert buckets["moderate_0_5_to_1_5yd"]["play_ids"] == [97, 137, 282]
    assert buckets["large_gt_1_5yd"]["play_ids"] == [349, 687, 1267, 1506]

    # Temporal mistakes
    tm = err["temporal_mistake_summary"]
    assert tm["early_snap_proximity_missed_play_ids"] == [282, 1267]
    assert 349 in tm["timing_phase_mismatch_play_ids"]

    # Observable biases (R2 selected 7/10 times, R1/R2 selected 80% vs 30% in GT)
    ob = err["observable_biases"]
    assert ob["predicted_tag_counts"]["R2"] == 7
    assert ob["low_tag_r1_r2_predicted_share"] == 0.8
    assert ob["low_tag_r1_r2_ground_truth_share"] == 0.3
    assert ob["low_tag_overselection_bias"] == 0.5

    # Explicit distinction between measurable error categories and hypothesised causes
    assert len(err["measurable_error_categories"]) >= 4
    for mcat in err["measurable_error_categories"]:
        assert mcat["verification_type"].startswith("measurable")
        assert "objective_evidence" in mcat

    assert len(err["hypothesised_causes"]) >= 3
    for hyp in err["hypothesised_causes"]:
        assert hyp["verification_status"] == "hypothesised_cause_unverified"
        assert "why_not_directly_proven" in hyp


# ---------------------------------------------------------------------------
# 10. Temporal Pass-Rush Ground Truth & Evaluation Tests (Feature 2)
# ---------------------------------------------------------------------------

def test_temporal_pass_rush_ground_truth_trends_and_timing(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    t_gt = compute_temporal_pass_rush_ground_truth(
        synthetic_play_payload, sampled_frame_ids=[10, 11, 12, 13]
    )
    # Closest rusher is 202 (Beta Defender), closing from 6.0 yd to 1.5 yd at frame 13 (+0.3s)
    assert t_gt["closest_rusher_id"] == 202
    assert t_gt["closest_rusher_distance_trend"] == "closing"
    assert math.isclose(t_gt["closest_rusher_initial_distance_yards"], 6.0, rel_tol=1e-4)
    assert math.isclose(t_gt["closest_rusher_final_distance_yards"], 1.5, rel_tol=1e-4)
    assert math.isclose(t_gt["closest_rusher_net_distance_change_yards"], -4.5, rel_tol=1e-4)
    assert math.isclose(t_gt["closest_rusher_total_closing_yards"], 4.5, rel_tol=1e-4)
    assert t_gt["closest_rusher_peak_closing_speed_yd_s"] >= 20.0
    assert t_gt["closest_approach_phase"] == "late"
    assert t_gt["closest_sampled_seq_number"] == 4

    # Rusher 101 (Alpha Rusher) stayed between 2.8 yd and 3.5 yd (< 0.75 yd closing/receding -> steady)
    # Closest at t=0.1s out of [0.0s, 0.3s] (rel = 1/3 -> 'early')
    r101 = next(r for r in t_gt["rusher_window_distances"] if r["id"] == 101)
    assert r101["distance_trend"] == "steady"
    assert r101["closest_approach_phase"] == "early"
    assert r101["sampled_closest_seq_number"] == 2

    # Modify Rusher 101 to close to 1.8 yd at frame 12 (t=0.2s, rel=2/3 -> 'mid') and recede to 4.0 yd at frame 13
    play_mod = json.loads(json.dumps(synthetic_play_payload))
    play_mod["frames"][0]["entities"][1]["x"] = 35.0  # d = 5.0 yd at t=0.0
    play_mod["frames"][1]["entities"][1]["x"] = 33.0  # d = 3.0 yd at t=0.1
    play_mod["frames"][2]["entities"][1]["x"] = 31.8  # d = 1.8 yd at t=0.2 (mid)
    play_mod["frames"][3]["entities"][1]["x"] = 34.0  # d = 4.0 yd at t=0.3 (receded +2.2 yd)
    t_gt_mod = compute_temporal_pass_rush_ground_truth(play_mod, sampled_frame_ids=[10, 11, 12, 13])
    r101_mod = next(r for r in t_gt_mod["rusher_window_distances"] if r["id"] == 101)
    assert r101_mod["distance_trend"] == "close_then_recede"
    assert r101_mod["closest_approach_phase"] == "mid"
    assert r101_mod["sampled_closest_seq_number"] == 3


def test_temporal_vlm_evaluation_separates_observations_from_interpretations(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    sanitized = sanitize_play_for_visual_input(synthetic_play_payload)
    candidates = sanitized["candidate_rushers"]

    # Prompt check: no ground truth leaked
    t_prompt = build_temporal_vlm_prompt(candidates, sampled_times=[0.0, 0.1, 0.2, 0.3])
    assert "distance_trend" in t_prompt
    assert "unverified_interpretation" in t_prompt
    assert "1.5" not in t_prompt

    t_gt = compute_temporal_pass_rush_ground_truth(
        synthetic_play_payload, sampled_frame_ids=[10, 11, 12, 13]
    )

    raw_temporal_json = json.dumps({
        "selected_rusher_id": 202,
        "selected_rusher_tag": "R2",
        "selected_rusher_name": "Beta Defender",
        "abstained": False,
        "distance_trend": "closing",
        "closest_approach_phase": "late",
        "closest_frame_number": 4,
        "collective_rush_description": "R2 converges rapidly from the right while R1 stalls near the line.",
        "movement_pattern": "R2 moves continuously toward the yellow QB dot from Frame 1 to Frame 4.",
        "visible_evidence": "In Frame 4 (+0.3s), R2 #94 reaches its shortest separation to QB.",
        "unverified_interpretation": "The offensive tackle lost outside leverage against a speed-to-power rush.",
        "uncertainty": "Low uncertainty.",
        "explanation": "Beta Defender (R2) closes steadily across all 4 frames, peaking in Frame 4.",
    })

    parsed = validate_and_parse_temporal_vlm_json(
        raw_temporal_json, candidate_rushers=candidates, num_sampled_frames=4
    )
    assert parsed["is_valid_json"] is True
    assert parsed["distance_trend"] == "closing"
    assert parsed["closest_approach_phase"] == "late"
    assert parsed["closest_frame_number"] == 4
    assert "outside leverage" in parsed["unverified_interpretation"]

    eval_rec = evaluate_temporal_play_prediction(parsed, t_gt, latency_seconds=1.8)
    assert eval_rec["exact_match"] is True
    assert eval_rec["trend_match_vs_closest"] is True
    assert eval_rec["trend_match_vs_selected"] is True
    assert eval_rec["closing_direction_match_vs_closest"] is True
    assert eval_rec["phase_match_vs_closest"] is True
    assert eval_rec["sampled_frame_error_vs_closest"] == 0

    # Check explicit separation of verifiable visual observations vs unverifiable interpretations
    vo = eval_rec["verifiable_visual_observations"]
    ui = eval_rec["unverifiable_interpretations"]
    assert vo["closest_rusher_identification"]["verified_match"] is True
    assert vo["qb_rusher_distance_trend"]["verified_exact_trend_vs_closest"] is True
    assert vo["closest_approach_timing"]["verified_phase_match_vs_closest"] is True
    assert "outside leverage" in ui["model_unverified_interpretation"]
    assert "cannot be validated" in ui["verification_boundary_note"]

    agg = aggregate_temporal_evaluation_results([eval_rec])
    assert agg["temporal_metrics"]["identifiable_plays"] == 1
    assert agg["temporal_metrics"]["trend_match_vs_closest_accuracy"] == 1.0
    assert agg["temporal_metrics"]["phase_match_vs_closest_accuracy"] == 1.0
    assert agg["temporal_metrics"]["mean_sampled_frame_error_vs_closest"] == 0.0


def test_versioned_temporal_v2_evaluation_artifact_integrity() -> None:
    v2_file = Path(__file__).resolve().parent.parent / "example_evaluation_temporal_2021090900_v2.json"
    if not v2_file.is_file():
        pytest.skip("example_evaluation_temporal_2021090900_v2.json not yet generated")
    report = json.loads(v2_file.read_text(encoding="utf-8"))
    assert report["total_plays"] == 10
    assert report["metadata"]["experiment_version"] == "temporal_pass_rush_v2"
    assert report["metadata"]["model_checkpoint"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert "temporal_metrics" in report
    assert "error_analysis_summary" in report
    for item in report["per_play_results"]:
        assert "verifiable_visual_observations" in item
        assert "unverifiable_interpretations" in item
        assert item["predicted_distance_trend"] in {
            "closing",
            "close_then_recede",
            "steady",
            "receding",
            None,
        }
        assert item["predicted_closest_approach_phase"] in {"early", "mid", "late", None}


# ---------------------------------------------------------------------------
# 11. Dynamic Player Comparison & Synchronized Telemetry Tests (Phase 3)
# ---------------------------------------------------------------------------

def test_player_selection_and_id_mapping(synthetic_play_payload: Dict[str, Any]) -> None:
    # Resolve by integer nflId
    r_int = resolve_play_rusher(synthetic_play_payload, 101)
    assert r_int["id"] == 101
    assert r_int["tag"] == "R1"
    assert r_int["name"] == "Alpha Rusher"

    # Resolve by string ID and R-tag (ordered deterministically by nflId: 101 -> R1, 202 -> R2)
    r_str = resolve_play_rusher(synthetic_play_payload, "202")
    assert r_str["id"] == 202
    assert r_str["tag"] == "R2"

    r_tag = resolve_play_rusher(synthetic_play_payload, "r2")
    assert r_tag["id"] == 202
    assert r_tag["name"] == "Beta Defender"

    # Resolve by full name and partial name
    assert resolve_play_rusher(synthetic_play_payload, "Alpha Rusher")["id"] == 101
    assert resolve_play_rusher(synthetic_play_payload, "Defender")["id"] == 202

    # Invalid or missing identifiers raise ValueError
    with pytest.raises(ValueError, match="not found"):
        resolve_play_rusher(synthetic_play_payload, 999999)
    with pytest.raises(ValueError, match="cannot be None"):
        resolve_play_rusher(synthetic_play_payload, None)


def test_distance_speed_and_closing_rate_distinction(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    """Verify distance, speed, acceleration, and closing rate, and prove absolute speed != closing rate."""
    play = json.loads(json.dumps(synthetic_play_payload))
    # Attach explicit tracking s, a, dir, o to entities
    # Rusher 101: moves tangentially around the QB at constant radius 3.0 yd with high ground speed s = 8.5 yd/s!
    # Because radius to QB (30.0, 26.65) stays 3.0 yd, its closing rate toward the QB must be 0.0 yd/s while speed is 8.5 yd/s.
    angles = [0.0, 0.25, 0.50, 0.75]
    for idx, f in enumerate(play["frames"]):
        theta = angles[idx]
        r101_ent = next(e for e in f["entities"] if e["id"] == 101)
        r101_ent["x"] = 30.0 + 3.0 * math.cos(theta)
        r101_ent["y"] = 26.65 + 3.0 * math.sin(theta)
        r101_ent["s"] = 8.5
        r101_ent["a"] = 2.1
        r101_ent["dir"] = 180.0
        r101_ent["o"] = 270.0

        # Rusher 202: rushes radially straight at QB from 6.0 yd -> 4.5 yd -> 3.0 yd -> 1.5 yd at 15.0 yd/s
        r202_ent = next(e for e in f["entities"] if e["id"] == 202)
        r202_ent["s"] = 15.0
        r202_ent["a"] = 3.4

    s101 = compute_player_telemetry_series(play, 101, pressure_threshold_yd=2.0)
    s202 = compute_player_telemetry_series(play, 202, pressure_threshold_yd=2.0)

    # Check Rusher 101: high scalar ground speed (8.5 yd/s) but zero QB closing rate (0.0 yd/s)
    for k, smp in enumerate(s101["samples"]):
        assert math.isclose(smp["distance_yd"], 3.0, abs_tol=1e-3)
        assert math.isclose(smp["distance_change_from_snap_yd"], 0.0, abs_tol=1e-3)
        assert math.isclose(smp["speed_yd_s"], 8.5, abs_tol=1e-3)
        assert math.isclose(smp["acceleration_yd_s2"], 2.1, abs_tol=1e-3)
        assert math.isclose(smp["closing_rate_yd_s"], 0.0, abs_tol=1e-3)
        assert smp["speed_yd_s"] != smp["closing_rate_yd_s"]

    # Check Rusher 202: radial approach toward QB -> closing rate matches radial decrease (-Δd/Δt)
    assert math.isclose(s202["samples"][1]["closing_rate_yd_s"], 15.0, rel_tol=1e-3)
    assert math.isclose(s202["samples"][2]["closing_rate_yd_s"], 25.0, rel_tol=1e-3)
    assert math.isclose(s202["samples"][3]["closing_rate_yd_s"], 5.0, rel_tol=1e-3)
    assert s202["summary"]["entered_2yd_zone"] is True
    assert s202["summary"]["first_within_2yd_time_s"] == 0.2

    # Head-to-head gap analysis
    gap = compute_head_to_head_gap_analysis(play, 101, 202, pressure_threshold_yd=2.0)
    assert QUANTITY_DISTINCTION_NOTE in gap["quantity_distinction_note"]
    # Frame 0: d_101 = 3.0, d_202 = 6.0 -> diff = -3.0 yd (101 is closer)
    assert math.isclose(gap["frames"][0]["distance_diff_a_minus_b_yd"], -3.0, abs_tol=1e-3)
    assert gap["frames"][0]["closer_player_id"] == 101
    # Frame 3: d_101 = 3.0, d_202 = 1.5 -> diff = +1.5 yd (202 is closer)
    assert math.isclose(gap["frames"][3]["distance_diff_a_minus_b_yd"], 1.5, abs_tol=1e-3)
    assert gap["frames"][3]["closer_player_id"] == 202
    assert math.isclose(gap["frames"][3]["speed_diff_a_minus_b_yd_s"], 8.5 - 15.0, abs_tol=1e-3)


def test_timeline_segmentation_sectors(synthetic_play_payload: Dict[str, Any]) -> None:
    """Verify S1/S2/S3 timeline segmentation boundaries and per-player sector statistics."""
    # Expand synthetic play to 6 frames (0.0s .. 0.5s) for clean 3-sector split (2 frames per sector)
    play = json.loads(json.dumps(synthetic_play_payload))
    play["frames"] = []
    for k in range(6):
        fid = 10 + k
        t = round(0.1 * k, 2)
        # Rusher 101 stays at 4.0 yd; Rusher 202 closes from 6.5 yd by 1.0 yd per frame -> 1.5 yd at k=5
        d101 = 4.0
        d202 = 6.5 - 1.0 * k
        play["frames"].append({
            "frame": fid,
            "time": t,
            "entities": [
                {"id": 99, "x": 30.0, "y": 26.65, "team": "TB", "name": "Tom Brady", "number": "12", "role": "Pass"},
                {"id": 101, "x": 30.0 + d101, "y": 26.65, "s": 4.0, "a": 1.0, "team": "DAL", "name": "Alpha Rusher", "number": "90", "role": "Pass Rush"},
                {"id": 202, "x": 30.0 + d202, "y": 26.65, "s": 10.0, "a": 2.5, "team": "DAL", "name": "Beta Defender", "number": "94", "role": "Pass Rush"},
            ],
            "distances": {"101": d101, "202": d202},
            "nearest": 101 if d101 <= d202 else 202,
        })

    segments = compute_timeline_segments(play, num_segments=3, pressure_threshold_yd=2.0)
    assert len(segments) == 3
    assert [s["segment_code"] for s in segments] == ["S1", "S2", "S3"]
    assert segments[0]["start_frame_index"] == 0 and segments[0]["end_frame_index"] == 1
    assert segments[1]["start_frame_index"] == 2 and segments[1]["end_frame_index"] == 3
    assert segments[2]["start_frame_index"] == 4 and segments[2]["end_frame_index"] == 5

    # Check Rusher 202 in S1 (frames 0..1: 6.5 -> 5.5 yd) vs S3 (frames 4..5: 2.5 -> 1.5 yd, enters 2-yd zone at +0.5s)
    s1_202 = segments[0]["per_player"]["202"]
    assert math.isclose(s1_202["initial_distance_yards"], 6.5, abs_tol=1e-3)
    assert math.isclose(s1_202["final_distance_yards"], 5.5, abs_tol=1e-3)
    assert math.isclose(s1_202["net_distance_change_yards"], -1.0, abs_tol=1e-3)
    assert s1_202["entered_2yd_zone"] is False

    s3_202 = segments[2]["per_player"]["202"]
    assert math.isclose(s3_202["initial_distance_yards"], 2.5, abs_tol=1e-3)
    assert math.isclose(s3_202["final_distance_yards"], 1.5, abs_tol=1e-3)
    assert s3_202["entered_2yd_zone"] is True
    assert s3_202["first_within_2yd_time_s"] == 0.5


def test_cross_play_time_alignment_and_disclaimer(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    play_a = json.loads(json.dumps(synthetic_play_payload))
    play_b = json.loads(json.dumps(synthetic_play_payload))
    play_b["play"] = 2002
    play_b["game"] = 2021091200

    trace_a = build_normalized_play_trace(play_a, 101)
    trace_b = build_normalized_play_trace(play_b, 202)
    assert isinstance(trace_a, NormalizedPlayTelemetryTrace)
    assert trace_a.normalized_times_u[0] == 0.0
    assert trace_a.normalized_times_u[-1] == 1.0
    assert CROSS_PLAY_DISCLAIMER in trace_a.equivalency_disclaimer

    aligned_u = align_cross_play_traces(trace_a, trace_b, alignment_mode="normalized_u", num_grid_points=5)
    assert aligned_u["comparison_scope"] == "cross_game_players"
    assert len(aligned_u["aligned_points"]) == 5
    assert aligned_u["aligned_points"][0]["grid_value"] == 0.0
    assert aligned_u["aligned_points"][-1]["grid_value"] == 1.0
    assert CROSS_PLAY_DISCLAIMER in aligned_u["equivalency_disclaimer"]

    aligned_sec = align_cross_play_traces(trace_a, trace_b, alignment_mode="seconds_since_snap", num_grid_points=4)
    assert aligned_sec["alignment_mode"] == "seconds_since_snap"
    assert len(aligned_sec["aligned_points"]) == 4


def test_missing_tracking_data_and_discontinuity_handling(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    """Verify missing player frames and frame-gap discontinuities do not invent movement."""
    play = json.loads(json.dumps(synthetic_play_payload))
    # Remove Rusher 101 from frame index 1 (missing tracking observation)
    play["frames"][1]["entities"] = [
        e for e in play["frames"][1]["entities"] if e.get("id") != 101
    ]
    # Introduce a frame gap discontinuity between frame index 2 (frame 12, t=0.2) and frame index 3 (frame 18, t=0.8)
    play["frames"][3]["frame"] = 18
    play["frames"][3]["time"] = 0.8

    s101 = compute_player_telemetry_series(play, 101)
    assert s101["samples"][1]["missing_data"] is True
    assert s101["samples"][1]["distance_yd"] is None
    assert s101["samples"][1]["closing_rate_yd_s"] is None
    assert s101["summary"]["missing_frame_count"] == 1

    # Frame index 2 follows a missing frame -> prev_valid was reset, so closing_rate is 0.0 (not invented across gap)
    assert s101["samples"][2]["missing_data"] is False
    assert s101["samples"][2]["closing_rate_yd_s"] == 0.0

    # Frame index 3 has a 0.6s / 6-frame jump -> flagged as discontinuity, closing_rate is 0.0
    assert s101["samples"][3]["discontinuity"] is True
    assert s101["samples"][3]["closing_rate_yd_s"] == 0.0
    assert s101["summary"]["discontinuity_count"] >= 1


def test_frame_interpolation_boundaries_and_scientific_integrity(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    """Verify visual interpolation clamps bounds, interpolates contiguous frames, refuses across gaps, and preserves measured telemetry."""
    frames = json.loads(json.dumps(synthetic_play_payload["frames"]))

    # 1. Out-of-bounds clamping (< 0 and > max_index)
    low = interpolate_visual_frame(frames, -2.5)
    assert low["clamped_fractional_index"] == 0.0
    assert low["is_interpolated_visual"] is False
    assert low["measured_frame_id"] == 10

    high = interpolate_visual_frame(frames, 99.0)
    assert high["clamped_fractional_index"] == 3.0
    assert high["is_interpolated_visual"] is False
    assert high["measured_frame_id"] == 13

    # 2. Mid-frame visual interpolation (alpha = 0.5 between frame 0 and frame 1)
    mid = interpolate_visual_frame(frames, 0.5)
    assert mid["is_interpolated_visual"] is True
    assert mid["discontinuity_detected"] is False
    r202_vis = next(e for e in mid["visual_entities"] if e.get("id") == 202)
    # Frame 0 x=36.0, Frame 1 x=34.5 -> midpoint x=35.25
    assert math.isclose(r202_vis["x"], 35.25, abs_tol=1e-4)
    # Scientific telemetry must come strictly from measured frame (round(0.5) -> frame index 0 or 1), NOT interpolated
    meas_idx = mid["measured_frame_index"]
    assert mid["measured_distances"] == frames[meas_idx]["distances"]
    assert mid["measured_nearest_id"] == frames[meas_idx]["nearest"]

    # 3. Discontinuity across frame gap refuses visual interpolation
    frames_gap = json.loads(json.dumps(frames))
    frames_gap[1]["frame"] = 15
    frames_gap[1]["time"] = 0.5
    gap_interp = interpolate_visual_frame(frames_gap, 0.5)
    assert gap_interp["discontinuity_detected"] is True
    assert gap_interp["is_interpolated_visual"] is False
    r202_no_interp = next(e for e in gap_interp["visual_entities"] if e.get("id") == 202)
    assert math.isclose(r202_no_interp["x"], 36.0, abs_tol=1e-4)


def test_playback_seeking_by_time_frame_and_fraction(
    synthetic_play_payload: Dict[str, Any]
) -> None:
    frames = synthetic_play_payload["frames"]

    # Seek by exact frameId
    s_fid = seek_timeline_position(frames, target_frame_id=12)
    assert s_fid["exact_frame_index"] == 2
    assert s_fid["frame_id"] == 12
    assert math.isclose(s_fid["time_s"], 0.2, abs_tol=1e-4)

    # Seek by time (0.15s is halfway between index 1 at 0.1s and index 2 at 0.2s)
    s_time = seek_timeline_position(frames, target_time_s=0.15)
    assert math.isclose(s_time["fractional_frame_index"], 1.5, abs_tol=1e-3)
    assert s_time["exact_frame_index"] in {1, 2}

    # Seek by normalized fraction (u = 1.0 -> last frame; u = -0.5 -> clamped to first frame)
    s_end = seek_timeline_position(frames, target_fraction=1.0)
    assert s_end["exact_frame_index"] == 3
    assert s_end["frame_id"] == 13

    s_neg = seek_timeline_position(frames, target_fraction=-0.5)
    assert s_neg["exact_frame_index"] == 0
    assert s_neg["frame_id"] == 10


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="Dataset folder not present")
def test_telemetry_bundle_and_dashboard_integration(tmp_path: Path) -> None:
    html_path, play_count = qbr.build_replay(DATA_DIR, "2021090900", tmp_path, run_vlm=False)
    assert play_count == 95
    html_text = html_path.read_text(encoding="utf-8")

    # Verify all new Dynamic Player Comparison & Synchronized Telemetry UI elements exist
    for token in [
        'id="comparisonSection"',
        'id="cmpEnable"',
        'id="playerASelect"',
        'id="playerBSelect"',
        'id="swapPlayers"',
        'id="presetClosestPair"',
        'id="presetVlmVsGt"',
        'id="cmpTrails"',
        'id="cmpVectors"',
        'id="liveGapPanel"',
        'id="telemetryChart"',
        'id="sectorPanel"',
        'id="smoothToggle"',
        'id="speedSelect"',
        'id="stepPrev"',
        'id="stepNext"',
        "requestAnimationFrame",
    ]:
        assert token in html_text

    # Extract embedded JSON DATA from HTML and verify telemetry integrity on real tracking plays
    prefix = "const DATA="
    start_pos = html_text.index(prefix) + len(prefix)
    end_pos = html_text.index(";const allPlays=DATA.plays", start_pos)
    embedded_data = json.loads(html_text[start_pos:end_pos])

    first_play = embedded_data["plays"][0]
    assert "telemetry" in first_play
    tel = first_play["telemetry"]
    assert tel["default_player_a_id"] is not None
    assert tel["default_player_b_id"] is not None
    assert len(tel["segments"]) == 3
    for r_id_str, r_series in tel["rushers"].items():
        assert len(r_series["distance_yd"]) == len(first_play["frames"])
        assert len(r_series["speed_yd_s"]) == len(first_play["frames"])
        assert len(r_series["acceleration_yd_s2"]) == len(first_play["frames"])
        assert len(r_series["closing_rate_yd_s"]) == len(first_play["frames"])
        assert r_series["summary"]["max_speed_yd_s"] > 0

    # Ensure VLM sanitizer strips the new telemetry & entity kinematics so VLM leakage prevention remains 100% intact
    sanitized = sanitize_play_for_visual_input(first_play)
    assert verify_no_ground_truth_leakage(sanitized) is True
    assert "telemetry" not in sanitized
    for ent in sanitized["frames"][0]["entities"]:
        assert "s" not in ent and "a" not in ent


# ==============================================================================
# PHASE 2 TESTS: PROGRESSIVE DISTANCE CHART, ADVANCED GAP/SECTORS, CROSS-PLAY,
# AND MULTI-PLAY PLAYER ANALYTICS
# ==============================================================================


def test_progressive_reveal_vs_full_trajectory_and_axis_stability(
    synthetic_play_payload: Dict[str, Any],
) -> None:
    """Verify Progressive Reveal vs Full Trajectory modes, no future reveal, axis stability, and tooltip hit-testing."""
    # At fractional index 1.5 (between frame_index 1 at 0.1s and frame_index 2 at 0.2s)
    prog_mid = compute_distance_chart_render_state(
        synthetic_play_payload, fractional_index=1.5, mode="progressive"
    )
    full_mid = compute_distance_chart_render_state(
        synthetic_play_payload, fractional_index=1.5, mode="full"
    )
    prog_start = compute_distance_chart_render_state(
        synthetic_play_payload, fractional_index=0.0, mode="progressive"
    )
    prog_end = compute_distance_chart_render_state(
        synthetic_play_payload, fractional_index=3.0, mode="progressive"
    )

    # 1. Axis stability across all playback positions and both modes
    assert prog_start["axes"] == prog_mid["axes"] == prog_end["axes"] == full_mid["axes"]
    assert prog_mid["axes"]["x_min_s"] == 0.0
    assert prog_mid["axes"]["x_max_s"] == 0.3
    assert prog_mid["axes"]["pressure_threshold_yd"] == 2.0

    # 2. Progressive Reveal must NOT reveal future tracking points beyond lower_frame_index (1)
    for curve in prog_mid["rusher_curves"]:
        meas_pts = curve["measured_points"]
        assert len(meas_pts) == 2  # frame_index 0 and 1 only
        assert all(pt["time_s"] <= 0.1 for pt in meas_pts)
        # Interpolated tip is clearly distinguished from measured points
        tip = curve["interpolated_tip"]
        assert tip is not None
        assert tip["is_interpolated"] is True
        assert math.isclose(tip["time_s"], 0.15, abs_tol=1e-4)
        assert curve["moving_marker"]["is_interpolated"] is True

    # 3. Full Trajectory mode reveals all 4 frames even when playback is at 1.5
    for curve in full_mid["rusher_curves"]:
        assert len(curve["measured_points"]) == 4

    # 4. At playback completion (index 3.0), Progressive Reveal shows the completed trajectory
    assert prog_end["is_play_completed"] is True
    for curve in prog_end["rusher_curves"]:
        assert len(curve["measured_points"]) == 4
        assert curve["interpolated_tip"] is None

    # 5. Tooltip hit-testing respects Progressive Reveal visibility (does not match unrevealed future points)
    # At fractional_index=1.0 (t=0.1s), querying a future point at t=0.3s, d=1.5yd must return None in progressive mode
    future_hit_prog = hit_test_distance_chart_point(
        synthetic_play_payload,
        query_time_s=0.3,
        query_distance_yd=1.5,
        mode="progressive",
        current_fractional_index=1.0,
        time_tolerance_s=0.05,
        distance_tolerance_yd=0.5,
    )
    assert future_hit_prog is None

    # The same point IS hit-testable in Full Trajectory mode (or once playback reaches index 3.0)
    future_hit_full = hit_test_distance_chart_point(
        synthetic_play_payload,
        query_time_s=0.3,
        query_distance_yd=1.5,
        mode="full",
        current_fractional_index=1.0,
        time_tolerance_s=0.05,
        distance_tolerance_yd=0.5,
    )
    assert future_hit_full is not None
    assert future_hit_full["player_id"] == 202
    assert math.isclose(future_hit_full["distance_yd"], 1.5, abs_tol=1e-2)


def test_advanced_gap_analysis_and_sector_head_to_head(
    synthetic_play_payload: Dict[str, Any],
) -> None:
    """Verify Feature C live gap/advantage fields and Feature D sector head-to-head comparison."""
    gap = compute_head_to_head_gap_analysis(synthetic_play_payload, "R1", "R2")
    adv = gap["advantage_summary"]

    # R1 starts 3.0 yd closer than R2 (-3.0 yd), ends 2.0 yd farther than R2 (+2.0 yd) -> net shift +5.0 yd
    assert math.isclose(adv["initial_gap_a_minus_b_yd"], -3.0, abs_tol=1e-4)
    assert math.isclose(adv["final_gap_a_minus_b_yd"], 2.0, abs_tol=1e-4)
    assert math.isclose(adv["net_gap_shift_yd"], 5.0, abs_tol=1e-4)
    assert adv["total_lead_changes"] == 1
    assert adv["player_a_closer_frames"] == 2
    assert adv["player_b_closer_frames"] == 2

    # Check live status statement and running minimums so far at frame 0 vs frame 3
    f0 = gap["frames"][0]
    assert "Player A (Alpha Rusher) is currently 3.00 yards closer" in f0["status_statement"]
    assert math.isclose(f0["min_distance_so_far_b_yd"], 6.0, abs_tol=1e-4)
    assert math.isclose(f0["time_of_min_so_far_b_s"], 0.0, abs_tol=1e-4)

    f3 = gap["frames"][3]
    assert "Player B (Beta Defender) is currently 2.00 yards closer" in f3["status_statement"]
    assert math.isclose(f3["min_distance_so_far_b_yd"], 1.5, abs_tol=1e-4)
    assert math.isclose(f3["time_of_min_so_far_b_s"], 0.3, abs_tol=1e-4)
    assert f3["advantage_lead_changes_so_far"] == 1

    # Sector head-to-head comparison
    segments = compute_timeline_segments(synthetic_play_payload, num_segments=3)
    h2h = compute_segment_head_to_head_comparison(segments, 101, 202)
    assert len(h2h["segments"]) == 3
    assert h2h["largest_closing_rate_gap_segment"] in {"S1", "S2", "S3"}
    assert h2h["largest_net_distance_swing_segment"] in {"S1", "S2", "S3"}

    # Check 2-yard zone entry tracking in S3 for Player 202 (enters <=2.0 yd at t=0.2s)
    s3_b = segments[2]["per_player"]["202"]
    assert s3_b["entered_2yd_zone"] is True
    assert s3_b["frames_within_2yd_zone"] >= 1
    assert "max_acceleration_yd_s2" in s3_b
    assert "mean_acceleration_yd_s2" in s3_b


def test_cross_play_alignment_and_unequal_duration_without_extrapolation(
    synthetic_play_payload: Dict[str, Any],
) -> None:
    """Verify Feature E cross-play alignment at ball_snap (t=0.0s) and unequal duration handling without extrapolation."""
    play_short = json.loads(json.dumps(synthetic_play_payload))
    play_short["game"] = 2021090900
    play_short["play"] = 1001

    # Create a longer second play in another game (6 frames, t=0.0s .. 0.5s)
    play_long = json.loads(json.dumps(synthetic_play_payload))
    play_long["game"] = 2021091200
    play_long["play"] = 2002
    play_long["duration"] = 0.5
    for extra_idx, (fid, t_s, b_x) in enumerate([(14, 0.4, 31.2), (15, 0.5, 31.0)], start=4):
        f_copy = json.loads(json.dumps(play_long["frames"][-1]))
        f_copy["frame"] = fid
        f_copy["time"] = t_s
        for ent in f_copy["entities"]:
            if ent.get("id") == 202:
                ent["x"] = b_x
        f_copy["distances"]["202"] = round(b_x - 30.0, 2)
        play_long["frames"].append(f_copy)

    cross = build_cross_play_comparison(play_short, 101, play_long, 202)
    assert cross["same_game"] is False
    assert cross["unequal_duration"] is True
    assert cross["shorter_play_side"] == "A"
    assert math.isclose(cross["duration_a_s"], 0.3, abs_tol=1e-4)
    assert math.isclose(cross["duration_b_s"], 0.5, abs_tol=1e-4)
    assert math.isclose(cross["shared_timeline_duration_s"], 0.5, abs_tol=1e-4)

    # At t=0.0s (ball_snap), both plays are active and aligned
    eval_snap = evaluate_cross_play_at_time(cross, 0.0)
    assert eval_snap["both_plays_active"] is True
    assert eval_snap["play_a_ended"] is False
    assert eval_snap["play_b_ended"] is False
    assert eval_snap["distance_diff_a_minus_b_yd"] is not None

    # At t=0.5s, Play A (0.3s duration) has ended; Play B is still active.
    # Play A must NOT have an active sample and distance_diff_a_minus_b_yd must be None (no extrapolation).
    eval_late = evaluate_cross_play_at_time(cross, 0.5)
    assert eval_late["play_a_ended"] is True
    assert eval_late["play_b_ended"] is False
    assert eval_late["both_plays_active"] is False
    assert eval_late["play_a_active_sample"] is None
    assert eval_late["play_b_active_sample"] is not None
    assert eval_late["distance_diff_a_minus_b_yd"] is None

    # Verify compact cross-game play payload retains QB + rushers and strips non-rusher/non-QB players
    compact = build_compact_cross_game_play_payload(play_long)
    assert compact["game"] == 2021091200
    assert len(compact["frames"]) == len(play_long["frames"])


def test_multi_play_analytics_denominators_filters_and_cross_game_identity(
    synthetic_play_payload: Dict[str, Any],
) -> None:
    """Verify Feature F multi-play aggregation, explicit denominators, contextual filters, and cross-game composite keys."""
    p1 = json.loads(json.dumps(synthetic_play_payload))
    p1["game"] = 2021090900
    p1["play"] = 97
    p1["down"] = 1
    p1["formation"] = "SHOTGUN"
    p1["outcome"] = "C"

    # Second play in a different game with the SAME numeric play_id (97) to prove composite key handling
    p2 = json.loads(json.dumps(synthetic_play_payload))
    p2["game"] = 2021091200
    p2["play"] = 97
    p2["down"] = 3
    p2["formation"] = "EMPTY"
    p2["outcome"] = "IN"

    records = extract_play_rusher_records([p1, p2])
    assert len(records) == 4  # 2 rushers * 2 plays

    id_check = validate_cross_game_record_identity(records)
    assert id_check["valid"] is True
    assert id_check["distinct_games"] == 2
    assert 97 in id_check["colliding_play_ids_across_games"]

    # Duplicate (game_id, play_id, nfl_id) must raise ValueError
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cross_game_record_identity(records + [records[0]])

    # Unfiltered analytics across both games
    analytics_all = compute_multi_play_player_analytics(records, min_plays=1)
    denom = analytics_all["denominators"]
    assert denom["total_games_in_scope"] == 2
    assert denom["total_eligible_plays_in_scope"] == 2
    assert denom["total_rusher_snaps_in_scope"] == 4

    by_id = {p["nfl_id"]: p for p in analytics_all["players"]}
    assert by_id[202]["eligible_plays"] == 2
    assert math.isclose(by_id[202]["mean_min_distance_yd"], 1.5, abs_tol=1e-4)
    assert math.isclose(by_id[202]["median_min_distance_yd"], 1.5, abs_tol=1e-4)
    assert by_id[202]["within_2yd_pct"] == 100.0
    assert by_id[202]["pff_hurry_count"] == 2

    # Filter by 3rd down only -> 1 eligible play in scope
    analytics_3rd = compute_multi_play_player_analytics(
        records, filters={"down": 3}, min_plays=1
    )
    assert analytics_3rd["denominators"]["total_eligible_plays_in_scope"] == 1
    assert all(p["eligible_plays"] == 1 for p in analytics_3rd["players"])


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="Dataset folder not present")
def test_phase2_dashboard_serialization_and_cross_game_bundle(tmp_path: Path) -> None:
    """Verify Phase 2 HTML dashboard contains all required DOM sections and cross-game / multi-play data."""
    html_path, play_count = qbr.build_replay(DATA_DIR, "2021090900", tmp_path, run_vlm=False)
    assert play_count == 95
    html_text = html_path.read_text(encoding="utf-8")

    # Verify all Phase 2 UI elements exist in generated HTML
    for token in [
        'id="workspaceNav"',
        'id="distChartMode"',
        'id="telMetricDist"',
        'id="telMetricSpeed"',
        'id="telMetricAccel"',
        'id="telMetricGap"',
        'id="telMetricClosing"',
        'id="sectorChart"',
        'id="crossPlaySection"',
        'id="cpGameA"',
        'id="cpPlayA"',
        'id="cpPlayerA"',
        'id="cpGameB"',
        'id="cpPlayB"',
        'id="cpPlayerB"',
        'id="crossFieldCanvas"',
        'id="crossTelemetryCanvas"',
        'id="multiPlaySection"',
        'id="mpGameFilter"',
        'id="mpSearch"',
        'id="analyticsDistCanvas"',
        'id="mpPlayerTable"',
        'id="mpDrilldownPanel"',
    ]:
        assert token in html_text

    prefix = "const DATA="
    start_pos = html_text.index(prefix) + len(prefix)
    end_pos = html_text.index(";const allPlays=DATA.plays", start_pos)
    embedded_data = json.loads(html_text[start_pos:end_pos])

    assert len(embedded_data["available_games"]) >= 2
    assert "2021090900" in embedded_data["cross_game_plays"]
    assert "2021091200" in embedded_data["cross_game_plays"]
    assert len(embedded_data["cross_game_plays"]["2021090900"]) == 95
    assert len(embedded_data["cross_game_plays"]["2021091200"]) == 71
    assert len(embedded_data["multi_play_records"]) > 500
    assert embedded_data["multi_play_default_analytics"]["denominators"]["total_games_in_scope"] == 2




