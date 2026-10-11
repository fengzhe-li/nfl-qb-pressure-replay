"""Dynamic player comparison, synchronized tracking metrics, proximity comparison, and play phase segmentation.

Provides tracking-grounded calculations for NFL pass-rusher motion and performance analysis:
- Per-rusher time series (QB distance, distance change from snap, speed, acceleration,
  closing rate toward QB, orientation, and direction)
- Head-to-head Player A vs. Player B proximity comparison across a shared timeline
- Objective play phase analysis (S1/S2/S3 timeline segment breakdown)
- Frame interpolation boundary and discontinuity handling (separating visual interpolation
  from 10 Hz measured tracking observations)
- Cross-play normalized tracking alignment structures
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import statistics


QUANTITY_DISTINCTION_NOTE = (
    "Physical quantity distinction: (1) Absolute player speed (s, yd/s) is scalar ground "
    "speed along the player's movement vector; (2) Relative closing rate toward the QB "
    "(-Δd/Δt, yd/s) is the rate at which Euclidean QB-rusher separation decreases "
    "(positive = closing on QB, negative = receding); (3) Distance to the QB (d, yd) is "
    "instantaneous 2D Euclidean separation. Geometric proximity advantage does not "
    "automatically indicate superior overall football performance (e.g., contain assignments, "
    "double-team occupation, or stunt setups)."
)

CROSS_PLAY_DISCLAIMER = (
    "Cross-play alignment aligns trajectories by time since ball_snap (seconds) or "
    "normalized play fraction (u in [0, 1]). Aligning snap times does NOT make two plays "
    "tactically or physically equivalent because dropback depth, pass protection scheme, "
    "and time-to-release/sack vary across plays."
)


def resolve_play_rusher(
    play_payload: Dict[str, Any],
    player_identifier: Any,
) -> Dict[str, Any]:
    """Resolve a pass rusher in `play_payload` by nflId, R-tag (R1..Rn), or player name."""
    if not isinstance(play_payload, dict):
        raise ValueError("play_payload must be a dictionary.")
    rushers = play_payload.get("rushers")
    if not isinstance(rushers, list) or not rushers:
        raise ValueError("play_payload contains no pass rushers.")

    sorted_by_id = sorted(rushers, key=lambda r: int(r["id"]))
    tag_map = {f"R{idx + 1}": r for idx, r in enumerate(sorted_by_id)}
    id_to_tag = {int(r["id"]): f"R{idx + 1}" for idx, r in enumerate(sorted_by_id)}

    if player_identifier is None:
        raise ValueError("player_identifier cannot be None.")

    # 1. Integer ID match
    if isinstance(player_identifier, int) or (
        isinstance(player_identifier, str) and player_identifier.strip().isdigit()
    ):
        target_id = int(str(player_identifier).strip())
        for r in rushers:
            if int(r["id"]) == target_id:
                out = dict(r)
                out["id"] = target_id
                out["tag"] = id_to_tag.get(target_id, "")
                return out

    # 2. R-tag match (R1, R2, ...)
    s_ident = str(player_identifier).strip()
    if s_ident.upper() in tag_map:
        r = tag_map[s_ident.upper()]
        out = dict(r)
        out["id"] = int(r["id"])
        out["tag"] = s_ident.upper()
        return out

    # 3. Exact or substring name match
    s_lower = s_ident.lower()
    for r in rushers:
        name = str(r.get("name", ""))
        if name.lower() == s_lower or s_lower in name.lower():
            rid = int(r["id"])
            out = dict(r)
            out["id"] = rid
            out["tag"] = id_to_tag.get(rid, "")
            return out

    raise ValueError(
        f"Pass rusher '{player_identifier}' not found in play {play_payload.get('play')}."
    )


def compute_player_telemetry_series(
    play_payload: Dict[str, Any],
    player_id: Any,
    pressure_threshold_yd: float = 2.0,
    max_contiguous_dt_s: float = 0.15,
) -> Dict[str, Any]:
    """Compute 10 Hz tracking telemetry time series for a single pass rusher on a play.

    Uses actual tracking fields (`frameId`, `time`, `x`, `y`, `s`, `a`, `dir`, `o`)
    wherever present, and handles missing frames or frame-gap discontinuities without
    inventing player movement.
    """
    rusher_info = resolve_play_rusher(play_payload, player_id)
    rid = int(rusher_info["id"])
    frames = play_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("play_payload must contain a non-empty 'frames' list.")
    qb_id = int(play_payload["qb"])

    samples: List[Dict[str, Any]] = []
    initial_dist: Optional[float] = None
    prev_valid: Optional[Dict[str, Any]] = None
    jersey_number = ""

    for f_idx, frame_obj in enumerate(frames):
        fid = int(frame_obj["frame"])
        t_s = round(float(frame_obj["time"]), 4)
        entities = frame_obj.get("entities", [])
        qb_ent = next((e for e in entities if e.get("id") == qb_id), None)
        r_ent = next((e for e in entities if e.get("id") == rid), None)

        if qb_ent is None or r_ent is None:
            # Missing tracking observation in this frame: record None without inventing movement
            samples.append({
                "frame_index": f_idx,
                "frame_id": fid,
                "time_s": t_s,
                "missing_data": True,
                "discontinuity": True,
                "x": None,
                "y": None,
                "qb_x": float(qb_ent["x"]) if qb_ent is not None else None,
                "qb_y": float(qb_ent["y"]) if qb_ent is not None else None,
                "distance_yd": None,
                "distance_change_from_snap_yd": None,
                "speed_yd_s": None,
                "acceleration_yd_s2": None,
                "closing_rate_yd_s": None,
                "dir_deg": None,
                "o_deg": None,
                "within_pressure_zone": False,
            })
            prev_valid = None
            continue

        if not jersey_number and r_ent.get("number"):
            jersey_number = str(r_ent["number"])

        rx, ry = float(r_ent["x"]), float(r_ent["y"])
        qx, qy = float(qb_ent["x"]), float(qb_ent["y"])
        dist_yd = float(math.hypot(rx - qx, ry - qy))
        if initial_dist is None:
            initial_dist = dist_yd
        dist_change_snap = dist_yd - initial_dist

        # Check continuity with previous valid frame
        is_discontinuous = False
        dt = 0.0
        if prev_valid is not None:
            dt = t_s - float(prev_valid["time_s"])
            frame_step = fid - int(prev_valid["frame_id"])
            if dt <= 1e-6 or dt > max_contiguous_dt_s or frame_step > 1:
                is_discontinuous = True

        # Speed (`s` in yd/s from tracking data; fallback to finite-difference only if omitted in synthetic fixture)
        raw_s = r_ent.get("s")
        if raw_s is not None and not (isinstance(raw_s, float) and math.isnan(raw_s)):
            speed_yd_s = float(raw_s)
            s_source = "tracking"
        elif prev_valid is not None and not is_discontinuous and dt > 1e-6:
            dx = rx - float(prev_valid["x"])
            dy = ry - float(prev_valid["y"])
            speed_yd_s = float(math.hypot(dx, dy) / dt)
            s_source = "derived_displacement"
        else:
            speed_yd_s = 0.0
            s_source = "fallback_zero"

        # Acceleration (`a` in yd/s^2 from tracking data; fallback to finite-difference if omitted)
        raw_a = r_ent.get("a")
        if raw_a is not None and not (isinstance(raw_a, float) and math.isnan(raw_a)):
            accel_yd_s2 = float(raw_a)
        elif prev_valid is not None and not is_discontinuous and dt > 1e-6 and prev_valid.get("speed_yd_s") is not None:
            accel_yd_s2 = float(abs(speed_yd_s - float(prev_valid["speed_yd_s"])) / dt)
        else:
            accel_yd_s2 = 0.0

        # Relative closing rate toward QB: -Δd / Δt (positive = closing toward QB)
        if prev_valid is not None and not is_discontinuous and dt > 1e-6:
            closing_rate_yd_s = float((float(prev_valid["distance_yd"]) - dist_yd) / dt)
        else:
            closing_rate_yd_s = 0.0

        raw_dir = r_ent.get("dir")
        dir_deg = (
            round(float(raw_dir), 2)
            if raw_dir is not None and not (isinstance(raw_dir, float) and math.isnan(raw_dir))
            else None
        )
        raw_o = r_ent.get("o")
        o_deg = (
            round(float(raw_o), 2)
            if raw_o is not None and not (isinstance(raw_o, float) and math.isnan(raw_o))
            else None
        )

        sample = {
            "frame_index": f_idx,
            "frame_id": fid,
            "time_s": t_s,
            "missing_data": False,
            "discontinuity": is_discontinuous,
            "x": round(rx, 4),
            "y": round(ry, 4),
            "qb_x": round(qx, 4),
            "qb_y": round(qy, 4),
            "distance_yd": round(dist_yd, 4),
            "distance_change_from_snap_yd": round(dist_change_snap, 4),
            "speed_yd_s": round(speed_yd_s, 4),
            "speed_source": s_source,
            "acceleration_yd_s2": round(accel_yd_s2, 4),
            "closing_rate_yd_s": round(closing_rate_yd_s, 4),
            "dir_deg": dir_deg,
            "o_deg": o_deg,
            "within_pressure_zone": dist_yd <= pressure_threshold_yd,
        }
        samples.append(sample)
        prev_valid = sample

    # Compute running minimum distance & 2-yd entry so far across samples
    running_min_d: Optional[float] = None
    running_min_t: Optional[float] = None
    running_first_2yd_t: Optional[float] = None
    for smp in samples:
        d_cur = smp["distance_yd"]
        if d_cur is not None:
            if running_min_d is None or float(d_cur) < running_min_d:
                running_min_d = float(d_cur)
                running_min_t = float(smp["time_s"])
            if smp["within_pressure_zone"] and running_first_2yd_t is None:
                running_first_2yd_t = float(smp["time_s"])
        smp["min_distance_so_far_yd"] = (
            round(running_min_d, 4) if running_min_d is not None else None
        )
        smp["time_of_min_so_far_s"] = running_min_t
        smp["entered_2yd_so_far"] = running_first_2yd_t is not None
        smp["first_within_2yd_so_far_s"] = running_first_2yd_t

    valid_samples = [s for s in samples if not s["missing_data"]]
    if not valid_samples:
        raise ValueError(f"Pass rusher {rid} has no valid tracking observations in play.")

    closest_sample = min(valid_samples, key=lambda s: float(s["distance_yd"]))
    within_samples = [s for s in valid_samples if s["within_pressure_zone"]]
    speeds = [float(s["speed_yd_s"]) for s in valid_samples]
    accels = [float(s["acceleration_yd_s2"]) for s in valid_samples]
    closing_rates = [
        float(s["closing_rate_yd_s"])
        for idx, s in enumerate(valid_samples)
        if idx > 0 and not s["discontinuity"]
    ]
    if not closing_rates:
        closing_rates = [0.0]

    return {
        "player_id": rid,
        "tag": rusher_info.get("tag", ""),
        "name": str(rusher_info.get("name", rid)),
        "number": jersey_number,
        "samples": samples,
        "summary": {
            "initial_distance_yards": round(float(valid_samples[0]["distance_yd"]), 4),
            "final_distance_yards": round(float(valid_samples[-1]["distance_yd"]), 4),
            "net_distance_change_yards": round(
                float(valid_samples[-1]["distance_yd"]) - float(valid_samples[0]["distance_yd"]), 4
            ),
            "closest_distance_yards": round(float(closest_sample["distance_yd"]), 4),
            "closest_time_s": float(closest_sample["time_s"]),
            "closest_frame_id": int(closest_sample["frame_id"]),
            "closest_frame_index": int(closest_sample["frame_index"]),
            "entered_2yd_zone": len(within_samples) > 0,
            "first_within_2yd_time_s": (
                float(within_samples[0]["time_s"]) if within_samples else None
            ),
            "max_speed_yd_s": round(max(speeds), 4),
            "mean_speed_yd_s": round(statistics.mean(speeds), 4),
            "max_acceleration_yd_s2": round(max(accels), 4),
            "mean_acceleration_yd_s2": round(statistics.mean(accels), 4),
            "peak_closing_rate_yd_s": round(max(closing_rates), 4),
            "mean_closing_rate_yd_s": round(statistics.mean(closing_rates), 4),
            "missing_frame_count": len(samples) - len(valid_samples),
            "discontinuity_count": sum(1 for s in valid_samples if s["discontinuity"]),
        },
    }


def compute_head_to_head_gap_analysis(
    play_payload: Dict[str, Any],
    player_a_id: Any,
    player_b_id: Any,
    pressure_threshold_yd: float = 2.0,
) -> Dict[str, Any]:
    """Compute synchronized player tracking metrics and proximity comparison between Player A and Player B."""
    series_a = compute_player_telemetry_series(
        play_payload, player_a_id, pressure_threshold_yd=pressure_threshold_yd
    )
    series_b = compute_player_telemetry_series(
        play_payload, player_b_id, pressure_threshold_yd=pressure_threshold_yd
    )

    frames_gap: List[Dict[str, Any]] = []
    initial_gap: Optional[float] = None
    prev_closer_id: Optional[int] = None
    lead_changes = 0
    a_closer_frames = 0
    b_closer_frames = 0
    tied_frames = 0
    max_a_adv_yd = 0.0
    max_a_adv_time_s = 0.0
    max_b_adv_yd = 0.0
    max_b_adv_time_s = 0.0

    id_a = int(series_a["player_id"])
    id_b = int(series_b["player_id"])
    name_a = series_a["name"]
    name_b = series_b["name"]

    for sa, sb in zip(series_a["samples"], series_b["samples"]):
        f_idx = int(sa["frame_index"])
        fid = int(sa["frame_id"])
        t_s = float(sa["time_s"])

        da = sa["distance_yd"]
        db = sb["distance_yd"]
        sa_spd = sa["speed_yd_s"]
        sb_spd = sb["speed_yd_s"]
        ca_rate = sa["closing_rate_yd_s"]
        cb_rate = sb["closing_rate_yd_s"]

        if da is not None and db is not None:
            dist_diff = round(float(da) - float(db), 4)
            if initial_gap is None:
                initial_gap = dist_diff
            gap_shift = round(dist_diff - initial_gap, 4)

            if abs(dist_diff) < 1e-6:
                closer_id = None
                tied_frames += 1
                status_stmt = (
                    f"Player A ({name_a}) and Player B ({name_b}) are currently "
                    f"equidistant ({float(da):.2f} yards) from the quarterback."
                )
            elif dist_diff < 0:
                closer_id = id_a
                a_closer_frames += 1
                adv = abs(dist_diff)
                if adv > max_a_adv_yd:
                    max_a_adv_yd = round(adv, 4)
                    max_a_adv_time_s = t_s
                status_stmt = (
                    f"Player A ({name_a}) is currently {adv:.2f} yards closer "
                    f"to the quarterback than Player B ({name_b})."
                )
            else:
                closer_id = id_b
                b_closer_frames += 1
                adv = abs(dist_diff)
                if adv > max_b_adv_yd:
                    max_b_adv_yd = round(adv, 4)
                    max_b_adv_time_s = t_s
                status_stmt = (
                    f"Player B ({name_b}) is currently {adv:.2f} yards closer "
                    f"to the quarterback than Player A ({name_a})."
                )

            if closer_id is not None:
                if prev_closer_id is not None and closer_id != prev_closer_id:
                    lead_changes += 1
                prev_closer_id = closer_id
        else:
            dist_diff = None
            gap_shift = None
            closer_id = None
            status_stmt = "Distance comparison unavailable at this frame due to missing observation."

        spd_diff = (
            round(float(sa_spd) - float(sb_spd), 4)
            if (sa_spd is not None and sb_spd is not None)
            else None
        )
        close_diff = (
            round(float(ca_rate) - float(cb_rate), 4)
            if (ca_rate is not None and cb_rate is not None)
            else None
        )

        frames_gap.append({
            "frame_index": f_idx,
            "frame_id": fid,
            "time_s": t_s,
            "distance_diff_a_minus_b_yd": dist_diff,
            "gap_shift_from_snap_yd": gap_shift,
            "speed_diff_a_minus_b_yd_s": spd_diff,
            "closing_rate_diff_a_minus_b_yd_s": close_diff,
            "closer_player_id": closer_id,
            "advantage_lead_changes_so_far": lead_changes,
            "min_distance_so_far_a_yd": sa["min_distance_so_far_yd"],
            "time_of_min_so_far_a_s": sa["time_of_min_so_far_s"],
            "min_distance_so_far_b_yd": sb["min_distance_so_far_yd"],
            "time_of_min_so_far_b_s": sb["time_of_min_so_far_s"],
            "status_statement": status_stmt,
            "player_a": sa,
            "player_b": sb,
        })

    valid_gaps = [fg["distance_diff_a_minus_b_yd"] for fg in frames_gap if fg["distance_diff_a_minus_b_yd"] is not None]
    final_gap = valid_gaps[-1] if valid_gaps else None

    return {
        "game_id": int(play_payload.get("game", 0)),
        "play_id": int(play_payload.get("play", 0)),
        "pressure_threshold_yd": float(pressure_threshold_yd),
        "player_a": {
            "player_id": series_a["player_id"],
            "tag": series_a["tag"],
            "name": series_a["name"],
            "number": series_a["number"],
            "summary": series_a["summary"],
        },
        "player_b": {
            "player_id": series_b["player_id"],
            "tag": series_b["tag"],
            "name": series_b["name"],
            "number": series_b["number"],
            "summary": series_b["summary"],
        },
        "advantage_summary": {
            "initial_gap_a_minus_b_yd": initial_gap,
            "final_gap_a_minus_b_yd": final_gap,
            "net_gap_shift_yd": (
                round(final_gap - initial_gap, 4)
                if (initial_gap is not None and final_gap is not None)
                else None
            ),
            "max_player_a_advantage_yd": max_a_adv_yd,
            "max_player_a_advantage_time_s": max_a_adv_time_s,
            "max_player_b_advantage_yd": max_b_adv_yd,
            "max_player_b_advantage_time_s": max_b_adv_time_s,
            "total_lead_changes": lead_changes,
            "player_a_closer_frames": a_closer_frames,
            "player_b_closer_frames": b_closer_frames,
            "tied_frames": tied_frames,
            "player_a_closer_time_s": round(a_closer_frames * 0.1, 2),
            "player_b_closer_time_s": round(b_closer_frames * 0.1, 2),
        },
        "frames": frames_gap,
        "quantity_distinction_note": QUANTITY_DISTINCTION_NOTE,
    }


def compute_timeline_segments(
    play_payload: Dict[str, Any],
    player_ids: Optional[Sequence[Any]] = None,
    num_segments: int = 3,
    pressure_threshold_yd: float = 2.0,
) -> List[Dict[str, Any]]:
    """Divide a play into objective, contiguous timeline segments (sectors) and compute per-player stats.

    Uses equal-index / equal-duration partitioning across the play's tracking frames
    without subjective football grading or composite scores.
    """
    frames = play_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("play_payload must contain a non-empty 'frames' list.")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive.")

    if player_ids is None:
        player_ids = [int(r["id"]) for r in play_payload.get("rushers", [])]

    series_by_id: Dict[int, Dict[str, Any]] = {}
    for pid in player_ids:
        s = compute_player_telemetry_series(
            play_payload, pid, pressure_threshold_yd=pressure_threshold_yd
        )
        series_by_id[int(s["player_id"])] = s

    n_frames = len(frames)
    actual_segments = min(num_segments, n_frames)
    sector_titles = [
        "Snap & Initial Burst",
        "Pocket Development",
        "Late Rush & Finish",
    ]

    segments: List[Dict[str, Any]] = []
    for seg_idx in range(actual_segments):
        start_i = int(round(seg_idx * n_frames / float(actual_segments)))
        end_i = int(round((seg_idx + 1) * n_frames / float(actual_segments))) - 1
        start_i = max(0, min(start_i, n_frames - 1))
        end_i = max(start_i, min(end_i, n_frames - 1))

        f_start = frames[start_i]
        f_end = frames[end_i]
        code = f"S{seg_idx + 1}"
        title = (
            sector_titles[seg_idx]
            if seg_idx < len(sector_titles)
            else f"Phase {seg_idx + 1}"
        )

        per_player_stats: Dict[str, Dict[str, Any]] = {}
        for pid, p_series in series_by_id.items():
            all_samples = p_series["samples"]
            seg_samples = [
                smp
                for smp in all_samples[start_i : end_i + 1]
                if not smp["missing_data"]
            ]
            if not seg_samples:
                per_player_stats[str(pid)] = {
                    "player_id": pid,
                    "name": p_series["name"],
                    "tag": p_series["tag"],
                    "missing_segment": True,
                }
                continue

            init_d = float(seg_samples[0]["distance_yd"])
            final_d = float(seg_samples[-1]["distance_yd"])
            best_s = min(seg_samples, key=lambda x: float(x["distance_yd"]))
            within_s = [x for x in seg_samples if x["within_pressure_zone"]]
            spds = [float(x["speed_yd_s"]) for x in seg_samples]
            accs = [float(x["acceleration_yd_s2"]) for x in seg_samples]

            # Track proximity-zone entry and exit events within this segment
            zone_entries: List[float] = []
            zone_exits: List[float] = []
            prev_in_zone = (
                bool(all_samples[start_i - 1]["within_pressure_zone"])
                if start_i > 0 and not all_samples[start_i - 1]["missing_data"]
                else False
            )
            for smp in seg_samples:
                cur_in_zone = bool(smp["within_pressure_zone"])
                if cur_in_zone and not prev_in_zone:
                    zone_entries.append(float(smp["time_s"]))
                elif not cur_in_zone and prev_in_zone:
                    zone_exits.append(float(smp["time_s"]))
                prev_in_zone = cur_in_zone

            seg_dt = float(seg_samples[-1]["time_s"]) - float(seg_samples[0]["time_s"])
            if seg_dt > 1e-6:
                mean_close = (init_d - final_d) / seg_dt
            else:
                mean_close = float(seg_samples[-1]["closing_rate_yd_s"])
            peak_close = max(float(x["closing_rate_yd_s"]) for x in seg_samples)

            per_player_stats[str(pid)] = {
                "player_id": pid,
                "name": p_series["name"],
                "tag": p_series["tag"],
                "missing_segment": False,
                "initial_distance_yards": round(init_d, 4),
                "final_distance_yards": round(final_d, 4),
                "net_distance_change_yards": round(final_d - init_d, 4),
                "max_speed_yd_s": round(max(spds), 4),
                "mean_speed_yd_s": round(statistics.mean(spds), 4),
                "max_acceleration_yd_s2": round(max(accs), 4),
                "mean_acceleration_yd_s2": round(statistics.mean(accs), 4),
                "min_acceleration_yd_s2": round(min(accs), 4),
                "mean_closing_rate_yd_s": round(mean_close, 4),
                "peak_closing_rate_yd_s": round(peak_close, 4),
                "closest_distance_yards": round(float(best_s["distance_yd"]), 4),
                "closest_time_s": float(best_s["time_s"]),
                "closest_frame_id": int(best_s["frame_id"]),
                "entered_2yd_zone": len(within_s) > 0,
                "first_within_2yd_time_s": (
                    float(within_s[0]["time_s"]) if within_s else None
                ),
                "zone_entry_times_s": zone_entries,
                "zone_exit_times_s": zone_exits,
                "frames_within_2yd_zone": len(within_s),
            }

        segments.append({
            "segment_index": seg_idx + 1,
            "segment_code": code,
            "segment_name": f"{code}: {title}",
            "start_frame_index": start_i,
            "end_frame_index": end_i,
            "start_frame_id": int(f_start["frame"]),
            "end_frame_id": int(f_end["frame"]),
            "start_time_s": round(float(f_start["time"]), 2),
            "end_time_s": round(float(f_end["time"]), 2),
            "duration_s": round(float(f_end["time"]) - float(f_start["time"]), 2),
            "per_player": per_player_stats,
        })

    return segments


def compute_segment_head_to_head_comparison(
    segments: Sequence[Dict[str, Any]],
    player_a_id: Any,
    player_b_id: Any,
) -> Dict[str, Any]:
    """Compare Player A and Player B across play segments and identify where largest measured differences developed."""
    sa_key = str(int(player_a_id))
    sb_key = str(int(player_b_id))

    segment_comparisons: List[Dict[str, Any]] = []
    max_closing_diff = -1.0
    max_closing_seg = None
    max_swing_diff = -1.0
    max_swing_seg = None
    max_speed_diff = -1.0
    max_speed_seg = None

    for seg in segments:
        pa = seg.get("per_player", {}).get(sa_key, {})
        pb = seg.get("per_player", {}).get(sb_key, {})
        if not pa or not pb or pa.get("missing_segment") or pb.get("missing_segment"):
            continue

        code = seg["segment_code"]
        net_diff = round(
            float(pa["net_distance_change_yards"]) - float(pb["net_distance_change_yards"]), 4
        )
        min_d_diff = round(
            float(pa["closest_distance_yards"]) - float(pb["closest_distance_yards"]), 4
        )
        mean_s_diff = round(
            float(pa["mean_speed_yd_s"]) - float(pb["mean_speed_yd_s"]), 4
        )
        max_s_diff = round(
            float(pa["max_speed_yd_s"]) - float(pb["max_speed_yd_s"]), 4
        )
        close_diff = round(
            float(pa["mean_closing_rate_yd_s"]) - float(pb["mean_closing_rate_yd_s"]), 4
        )
        acc_diff = round(
            float(pa["max_acceleration_yd_s2"]) - float(pb["max_acceleration_yd_s2"]), 4
        )

        if abs(close_diff) > max_closing_diff:
            max_closing_diff = abs(close_diff)
            max_closing_seg = code
        if abs(net_diff) > max_swing_diff:
            max_swing_diff = abs(net_diff)
            max_swing_seg = code
        if abs(mean_s_diff) > max_speed_diff:
            max_speed_diff = abs(mean_s_diff)
            max_speed_seg = code

        segment_comparisons.append({
            "segment_index": seg["segment_index"],
            "segment_code": code,
            "segment_name": seg["segment_name"],
            "start_frame_index": seg["start_frame_index"],
            "end_frame_index": seg["end_frame_index"],
            "start_time_s": seg["start_time_s"],
            "end_time_s": seg["end_time_s"],
            "net_distance_change_diff_a_minus_b_yd": net_diff,
            "min_distance_diff_a_minus_b_yd": min_d_diff,
            "mean_speed_diff_a_minus_b_yd_s": mean_s_diff,
            "max_speed_diff_a_minus_b_yd_s": max_s_diff,
            "mean_closing_rate_diff_a_minus_b_yd_s": close_diff,
            "max_acceleration_diff_a_minus_b_yd_s2": acc_diff,
            "player_a": pa,
            "player_b": pb,
        })

    return {
        "player_a_id": int(player_a_id),
        "player_b_id": int(player_b_id),
        "segments": segment_comparisons,
        "largest_closing_rate_gap_segment": max_closing_seg,
        "largest_net_distance_swing_segment": max_swing_seg,
        "largest_speed_gap_segment": max_speed_seg,
        "interpretation_disclaimer": (
            "Measured segment statistics describe kinematic trajectory differences within "
            "each time window and do not label a player as tactically better or worse."
        ),
    }


def interpolate_visual_frame(
    frames: Sequence[Dict[str, Any]],
    fractional_index: float,
    max_frame_gap: int = 1,
    max_time_gap_s: float = 0.15,
) -> Dict[str, Any]:
    """Interpolate entity positions visually between adjacent frames while preserving exact tracking telemetry.

    Enforces:
    - Clamping to valid `[0, len(frames) - 1]` bounds.
    - No interpolation across frame gaps (`frameId[i+1] - frameId[i] > max_frame_gap`) or
      time discontinuities (`time[i+1] - time[i] > max_time_gap_s`).
    - No invented movement if a player is missing in the next frame.
    - Scientific telemetry (`distances`, `nearest`, `s`, `a`) remains strictly tied to
      the measured 10 Hz tracking frame (`measured_frame_index`).
    """
    if not frames:
        raise ValueError("frames must be non-empty.")

    n = len(frames)
    clamped = max(0.0, min(float(fractional_index), float(n - 1)))
    i0 = int(math.floor(clamped))
    i1 = min(i0 + 1, n - 1)
    alpha = clamped - i0
    meas_idx = int(round(clamped))
    meas_idx = max(0, min(meas_idx, n - 1))

    f0 = frames[i0]
    f1 = frames[i1]
    f_meas = frames[meas_idx]

    frame_gap = int(f1["frame"]) - int(f0["frame"])
    time_gap = float(f1["time"]) - float(f0["time"])
    discontinuity = (i0 != i1) and (
        frame_gap > max_frame_gap or time_gap > max_time_gap_s or time_gap <= 0.0
    )

    if alpha <= 1e-6 or i0 == i1 or discontinuity:
        vis_entities = [dict(e, interpolated=False) for e in f0.get("entities", [])]
        vis_time = float(f0["time"])
        is_interp = False
    else:
        ent1_by_key: Dict[Any, Dict[str, Any]] = {}
        for e1 in f1.get("entities", []):
            key = "football" if e1.get("role") == "Football" else e1.get("id")
            if key is not None:
                ent1_by_key[key] = e1

        vis_entities = []
        for e0 in f0.get("entities", []):
            key = "football" if e0.get("role") == "Football" else e0.get("id")
            e1 = ent1_by_key.get(key)
            if e1 is not None:
                vx = (1.0 - alpha) * float(e0["x"]) + alpha * float(e1["x"])
                vy = (1.0 - alpha) * float(e0["y"]) + alpha * float(e1["y"])
                ent_copy = dict(e0)
                ent_copy["x"] = round(vx, 4)
                ent_copy["y"] = round(vy, 4)
                ent_copy["interpolated"] = True
                vis_entities.append(ent_copy)
            else:
                # Entity missing in next frame: do not invent movement
                ent_copy = dict(e0)
                ent_copy["interpolated"] = False
                vis_entities.append(ent_copy)
        vis_time = round((1.0 - alpha) * float(f0["time"]) + alpha * float(f1["time"]), 4)
        is_interp = True

    return {
        "requested_fractional_index": fractional_index,
        "clamped_fractional_index": round(clamped, 4),
        "lower_frame_index": i0,
        "upper_frame_index": i1,
        "alpha": round(alpha, 4),
        "measured_frame_index": meas_idx,
        "measured_frame_id": int(f_meas["frame"]),
        "measured_time_s": float(f_meas["time"]),
        "visual_time_s": vis_time,
        "is_interpolated_visual": is_interp,
        "discontinuity_detected": discontinuity,
        "visual_entities": vis_entities,
        "measured_distances": dict(f_meas.get("distances", {})),
        "measured_nearest_id": f_meas.get("nearest"),
        "observation_integrity_note": (
            "Visual coordinates are linearly interpolated between adjacent 10 Hz frames "
            "for smooth animation only; all scientific telemetry readouts come from "
            f"measured tracking frame {f_meas['frame']} (+{float(f_meas['time']):.1f}s)."
        ),
    }


def seek_timeline_position(
    frames: Sequence[Dict[str, Any]],
    target_time_s: Optional[float] = None,
    target_frame_id: Optional[int] = None,
    target_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve a user timeline seek request (by time, tracking frameId, or [0,1] fraction) to exact and fractional indices."""
    if not frames:
        raise ValueError("frames must be non-empty.")
    n = len(frames)
    t_start = float(frames[0]["time"])
    t_end = float(frames[-1]["time"])
    duration = max(t_end - t_start, 0.0)

    if target_frame_id is not None:
        fid_int = int(target_frame_id)
        best_idx = min(
            range(n), key=lambda idx: abs(int(frames[idx]["frame"]) - fid_int)
        )
        return {
            "exact_frame_index": best_idx,
            "fractional_frame_index": float(best_idx),
            "frame_id": int(frames[best_idx]["frame"]),
            "time_s": float(frames[best_idx]["time"]),
        }

    if target_fraction is not None:
        u = max(0.0, min(1.0, float(target_fraction)))
        target_time_s = t_start + u * duration

    if target_time_s is None:
        raise ValueError(
            "One of target_time_s, target_frame_id, or target_fraction must be provided."
        )

    t_clamped = max(t_start, min(t_end, float(target_time_s)))
    best_idx = min(
        range(n), key=lambda idx: abs(float(frames[idx]["time"]) - t_clamped)
    )

    # Fractional index between surrounding frames
    frac_idx = float(best_idx)
    for k in range(n - 1):
        t0 = float(frames[k]["time"])
        t1 = float(frames[k + 1]["time"])
        if t0 <= t_clamped <= t1 and (t1 - t0) > 1e-6:
            frac_idx = k + (t_clamped - t0) / (t1 - t0)
            break

    return {
        "exact_frame_index": best_idx,
        "fractional_frame_index": round(frac_idx, 4),
        "frame_id": int(frames[best_idx]["frame"]),
        "time_s": float(frames[best_idx]["time"]),
    }


@dataclass
class NormalizedPlayTelemetryTrace:
    """Future-ready cross-play / cross-game telemetry trace for a single pass rusher."""

    game_id: int
    play_id: int
    player_id: int
    player_name: str
    player_tag: str
    play_duration_s: float
    end_event: str
    timestamps_s: List[float] = field(default_factory=list)
    normalized_times_u: List[float] = field(default_factory=list)
    distances_yd: List[Optional[float]] = field(default_factory=list)
    speeds_yd_s: List[Optional[float]] = field(default_factory=list)
    accelerations_yd_s2: List[Optional[float]] = field(default_factory=list)
    closing_rates_yd_s: List[Optional[float]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    equivalency_disclaimer: str = CROSS_PLAY_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_normalized_play_trace(
    play_payload: Dict[str, Any],
    player_id: Any,
    pressure_threshold_yd: float = 2.0,
) -> NormalizedPlayTelemetryTrace:
    """Build a reusable `NormalizedPlayTelemetryTrace` for future cross-play or cross-game comparisons."""
    series = compute_player_telemetry_series(
        play_payload, player_id, pressure_threshold_yd=pressure_threshold_yd
    )
    samples = series["samples"]
    t0 = float(samples[0]["time_s"])
    t1 = float(samples[-1]["time_s"])
    span = max(t1 - t0, 0.1)

    times_s = [float(s["time_s"]) for s in samples]
    norm_u = [round((float(s["time_s"]) - t0) / span, 4) for s in samples]

    return NormalizedPlayTelemetryTrace(
        game_id=int(play_payload.get("game", 0)),
        play_id=int(play_payload.get("play", 0)),
        player_id=int(series["player_id"]),
        player_name=str(series["name"]),
        player_tag=str(series["tag"]),
        play_duration_s=round(t1 - t0, 4),
        end_event=str(play_payload.get("end_event", "unknown")),
        timestamps_s=times_s,
        normalized_times_u=norm_u,
        distances_yd=[s["distance_yd"] for s in samples],
        speeds_yd_s=[s["speed_yd_s"] for s in samples],
        accelerations_yd_s2=[s["acceleration_yd_s2"] for s in samples],
        closing_rates_yd_s=[s["closing_rate_yd_s"] for s in samples],
        summary=series["summary"],
    )


def align_cross_play_traces(
    trace_a: NormalizedPlayTelemetryTrace,
    trace_b: NormalizedPlayTelemetryTrace,
    alignment_mode: str = "normalized_u",
    num_grid_points: int = 11,
) -> Dict[str, Any]:
    """Align two `NormalizedPlayTelemetryTrace` objects (across plays or games) on a shared grid.

    Supports:
    - `alignment_mode="normalized_u"`: grid `u in [0.0, 1.0]` (fraction of play duration)
    - `alignment_mode="seconds_since_snap"`: grid `t in [0.0, min(duration_a, duration_b)]`
    """
    if num_grid_points < 2:
        raise ValueError("num_grid_points must be >= 2.")
    if alignment_mode not in {"normalized_u", "seconds_since_snap"}:
        raise ValueError("alignment_mode must be 'normalized_u' or 'seconds_since_snap'.")

    def _sample_nearest(
        axis_vals: Sequence[float],
        metric_vals: Sequence[Optional[float]],
        target: float,
    ) -> Optional[float]:
        valid_pairs = [
            (ax, mv) for ax, mv in zip(axis_vals, metric_vals) if mv is not None
        ]
        if not valid_pairs:
            return None
        best = min(valid_pairs, key=lambda item: abs(item[0] - target))
        return round(float(best[1]), 4)

    if alignment_mode == "normalized_u":
        grid = [round(k / float(num_grid_points - 1), 4) for k in range(num_grid_points)]
        axis_a = trace_a.normalized_times_u
        axis_b = trace_b.normalized_times_u
    else:
        common_dur = max(0.1, min(trace_a.play_duration_s, trace_b.play_duration_s))
        grid = [
            round(k * common_dur / float(num_grid_points - 1), 4)
            for k in range(num_grid_points)
        ]
        axis_a = trace_a.timestamps_s
        axis_b = trace_b.timestamps_s

    aligned_points: List[Dict[str, Any]] = []
    for g in grid:
        da = _sample_nearest(axis_a, trace_a.distances_yd, g)
        db = _sample_nearest(axis_b, trace_b.distances_yd, g)
        sa = _sample_nearest(axis_a, trace_a.speeds_yd_s, g)
        sb = _sample_nearest(axis_b, trace_b.speeds_yd_s, g)
        aligned_points.append({
            "grid_value": g,
            "trace_a_distance_yd": da,
            "trace_b_distance_yd": db,
            "distance_diff_a_minus_b_yd": (
                round(da - db, 4) if (da is not None and db is not None) else None
            ),
            "trace_a_speed_yd_s": sa,
            "trace_b_speed_yd_s": sb,
        })

    same_player = trace_a.player_id == trace_b.player_id
    same_game = trace_a.game_id == trace_b.game_id

    return {
        "comparison_scope": (
            "same_player_across_plays"
            if same_player
            else ("cross_game_players" if not same_game else "two_players_across_plays")
        ),
        "alignment_mode": alignment_mode,
        "trace_a_meta": {
            "game_id": trace_a.game_id,
            "play_id": trace_a.play_id,
            "player_id": trace_a.player_id,
            "player_name": trace_a.player_name,
            "duration_s": trace_a.play_duration_s,
            "end_event": trace_a.end_event,
        },
        "trace_b_meta": {
            "game_id": trace_b.game_id,
            "play_id": trace_b.play_id,
            "player_id": trace_b.player_id,
            "player_name": trace_b.player_name,
            "duration_s": trace_b.play_duration_s,
            "end_event": trace_b.end_event,
        },
        "aligned_points": aligned_points,
        "equivalency_disclaimer": CROSS_PLAY_DISCLAIMER,
    }


def compute_distance_chart_render_state(
    play_payload: Dict[str, Any],
    fractional_index: float,
    mode: str = "progressive",
    pressure_threshold_yd: float = 2.0,
    max_frame_gap: int = 1,
    max_time_gap_s: float = 0.15,
) -> Dict[str, Any]:
    """Compute the render state for the right-side Pass-rusher Distance to QB chart.

    Supports:
    - `mode="progressive"` (Default): Displays measured trajectories only up to `lower_frame_index`
      (`i0`) plus a distinct visually interpolated leading segment when `0 < alpha < 1` and
      contiguous. Never reveals future tracking values beyond current playback time.
    - `mode="full"`: Preserves full-play distance curves while keeping the synchronized
      playback time cursor and current moving point markers.
    - Guarantees stable X and Y axis bounds (`x_min_s`, `x_max_s`, `y_min_yd`, `y_max_yd`)
      across all playback positions and both modes.
    """
    if mode not in {"progressive", "full"}:
        raise ValueError("mode must be 'progressive' or 'full'.")
    frames = play_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("play_payload must contain a non-empty 'frames' list.")

    interp_state = interpolate_visual_frame(
        frames,
        fractional_index,
        max_frame_gap=max_frame_gap,
        max_time_gap_s=max_time_gap_s,
    )
    i0 = int(interp_state["lower_frame_index"])
    i1 = int(interp_state["upper_frame_index"])
    alpha = float(interp_state["alpha"])
    meas_idx = int(interp_state["measured_frame_index"])
    vis_time = float(interp_state["visual_time_s"])
    is_interp = bool(interp_state["is_interpolated_visual"])
    discont = bool(interp_state["discontinuity_detected"])

    # Stable X and Y axis limits derived from the entire play
    all_dists = [
        float(v)
        for ff in frames
        for v in ff.get("distances", {}).values()
        if v is not None
    ]
    max_y = max([5.0] + all_dists) * 1.1
    duration = max(float(play_payload.get("duration", frames[-1]["time"])), 0.1)

    sorted_rushers = sorted(play_payload.get("rushers", []), key=lambda r: int(r["id"]))
    rusher_curves: List[Dict[str, Any]] = []
    for idx, r in enumerate(sorted_rushers):
        rid = int(r["id"])
        r_key = str(rid)
        tag = f"R{idx + 1}"

        max_k = i0 if mode == "progressive" else (len(frames) - 1)
        measured_points: List[Dict[str, Any]] = []
        for k in range(max_k + 1):
            d_val = frames[k].get("distances", {}).get(r_key)
            if d_val is None:
                continue
            measured_points.append({
                "frame_index": k,
                "frame_id": int(frames[k]["frame"]),
                "time_s": float(frames[k]["time"]),
                "distance_yd": round(float(d_val), 4),
                "is_interpolated": False,
            })

        d0 = frames[i0].get("distances", {}).get(r_key)
        d1 = frames[i1].get("distances", {}).get(r_key)
        d_meas = frames[meas_idx].get("distances", {}).get(r_key)

        interpolated_tip: Optional[Dict[str, Any]] = None
        if is_interp and not discont and d0 is not None and d1 is not None:
            vis_d = round((1.0 - alpha) * float(d0) + alpha * float(d1), 4)
            interpolated_tip = {
                "time_s": vis_time,
                "visual_distance_yd": vis_d,
                "is_interpolated": True,
                "measured_reference_frame_id": int(frames[meas_idx]["frame"]),
                "measured_reference_distance_yd": (
                    round(float(d_meas), 4) if d_meas is not None else None
                ),
            }
            marker_vis_d = vis_d
        else:
            marker_vis_d = round(float(d0), 4) if d0 is not None else None

        rusher_curves.append({
            "player_id": rid,
            "tag": tag,
            "name": str(r.get("name", rid)),
            "measured_points": measured_points,
            "interpolated_tip": interpolated_tip,
            "moving_marker": {
                "time_s": vis_time,
                "visual_distance_yd": marker_vis_d,
                "measured_frame_id": int(frames[meas_idx]["frame"]),
                "measured_time_s": float(frames[meas_idx]["time"]),
                "measured_distance_yd": (
                    round(float(d_meas), 4) if d_meas is not None else None
                ),
                "is_interpolated": is_interp and not discont,
            },
        })

    return {
        "mode": mode,
        "axes": {
            "x_min_s": 0.0,
            "x_max_s": round(duration, 4),
            "y_min_yd": 0.0,
            "y_max_yd": round(max_y, 4),
            "pressure_threshold_yd": float(pressure_threshold_yd),
        },
        "cursor_time_s": vis_time,
        "measured_frame_index": meas_idx,
        "measured_frame_id": int(frames[meas_idx]["frame"]),
        "measured_time_s": float(frames[meas_idx]["time"]),
        "is_interpolated_position": is_interp and not discont,
        "is_play_completed": i0 >= (len(frames) - 1),
        "rusher_curves": rusher_curves,
    }


def hit_test_distance_chart_point(
    play_payload: Dict[str, Any],
    query_time_s: float,
    query_distance_yd: float,
    mode: str = "progressive",
    current_fractional_index: Optional[float] = None,
    time_tolerance_s: float = 0.25,
    distance_tolerance_yd: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """Hit-test the distance chart for hover tooltips, respecting Progressive Reveal visibility."""
    frames = play_payload.get("frames", [])
    if not frames:
        return None
    n = len(frames)
    if mode == "progressive" and current_fractional_index is not None:
        max_idx = max(0, min(n - 1, int(math.floor(float(current_fractional_index)))))
    else:
        max_idx = n - 1

    sorted_rushers = sorted(play_payload.get("rushers", []), key=lambda r: int(r["id"]))
    tag_map = {int(r["id"]): f"R{idx + 1}" for idx, r in enumerate(sorted_rushers)}
    name_map = {int(r["id"]): str(r.get("name", r["id"])) for r in sorted_rushers}

    best_hit: Optional[Dict[str, Any]] = None
    best_norm_dist = 1e9

    for k in range(max_idx + 1):
        ff = frames[k]
        t_s = float(ff["time"])
        dt = abs(t_s - float(query_time_s))
        if dt > time_tolerance_s:
            continue
        for r_key, d_val in ff.get("distances", {}).items():
            if d_val is None:
                continue
            dd = abs(float(d_val) - float(query_distance_yd))
            if dd > distance_tolerance_yd:
                continue
            norm_score = math.hypot(
                dt / max(time_tolerance_s, 1e-6),
                dd / max(distance_tolerance_yd, 1e-6),
            )
            if norm_score < best_norm_dist:
                best_norm_dist = norm_score
                rid = int(r_key)
                ent = next((e for e in ff.get("entities", []) if e.get("id") == rid), None)
                best_hit = {
                    "player_id": rid,
                    "tag": tag_map.get(rid, ""),
                    "name": name_map.get(rid, str(rid)),
                    "number": str(ent.get("number", "")) if ent else "",
                    "frame_index": k,
                    "frame_id": int(ff["frame"]),
                    "time_s": round(t_s, 2),
                    "distance_yd": round(float(d_val), 2),
                    "speed_yd_s": (
                        round(float(ent["s"]), 2)
                        if ent and ent.get("s") is not None
                        else None
                    ),
                }

    return best_hit


def build_cross_play_comparison(
    play_a_payload: Dict[str, Any],
    player_a_id: Any,
    play_b_payload: Dict[str, Any],
    player_b_id: Any,
    pressure_threshold_yd: float = 2.0,
) -> Dict[str, Any]:
    """Align and compare two plays (same-game or cross-game) along a shared elapsed-time timeline since ball_snap.

    Strictly enforces:
    - Timeline alignment at official `ball_snap` (`t = 0.0 s`).
    - No extrapolation of trajectories or telemetry beyond a play's final observed frame when durations differ.
    - Explicit `play_a_ended` and `play_b_ended` status flags at every step.
    - LOS-relative and initial-QB-relative pocket coordinates (`rel_x_from_los`, `rel_y_from_qb0`)
      for normalized dual-play trajectory comparison.
    """
    series_a = compute_player_telemetry_series(
        play_a_payload, player_a_id, pressure_threshold_yd=pressure_threshold_yd
    )
    series_b = compute_player_telemetry_series(
        play_b_payload, player_b_id, pressure_threshold_yd=pressure_threshold_yd
    )

    samples_a = series_a["samples"]
    samples_b = series_b["samples"]
    dur_a = round(float(samples_a[-1]["time_s"]) - float(samples_a[0]["time_s"]), 4)
    dur_b = round(float(samples_b[-1]["time_s"]) - float(samples_b[0]["time_s"]), 4)
    shared_dur = max(dur_a, dur_b)
    n_steps = max(len(samples_a), len(samples_b))

    los_a = float(play_a_payload.get("view", {}).get("los", samples_a[0]["qb_x"] or 0.0))
    qb_y0_a = float(play_a_payload.get("view", {}).get("qb_y", samples_a[0]["qb_y"] or 26.65))
    los_b = float(play_b_payload.get("view", {}).get("los", samples_b[0]["qb_x"] or 0.0))
    qb_y0_b = float(play_b_payload.get("view", {}).get("qb_y", samples_b[0]["qb_y"] or 26.65))

    def _enrich_rel(smp: Dict[str, Any], los_val: float, qby0_val: float) -> Dict[str, Any]:
        out = dict(smp)
        if smp["x"] is not None and smp["y"] is not None:
            out["rel_x_from_los"] = round(float(smp["x"]) - los_val, 4)
            out["rel_y_from_qb0"] = round(float(smp["y"]) - qby0_val, 4)
        else:
            out["rel_x_from_los"] = None
            out["rel_y_from_qb0"] = None
        if smp["qb_x"] is not None and smp["qb_y"] is not None:
            out["qb_rel_x_from_los"] = round(float(smp["qb_x"]) - los_val, 4)
            out["qb_rel_y_from_qb0"] = round(float(smp["qb_y"]) - qby0_val, 4)
        else:
            out["qb_rel_x_from_los"] = None
            out["qb_rel_y_from_qb0"] = None
        return out

    enriched_a = [_enrich_rel(s, los_a, qb_y0_a) for s in samples_a]
    enriched_b = [_enrich_rel(s, los_b, qb_y0_b) for s in samples_b]

    elapsed_steps: List[Dict[str, Any]] = []
    for k in range(n_steps):
        a_ended = k >= len(enriched_a)
        b_ended = k >= len(enriched_b)
        sa_active = None if a_ended else enriched_a[k]
        sb_active = None if b_ended else enriched_b[k]
        sa_held = enriched_a[-1] if a_ended else enriched_a[k]
        sb_held = enriched_b[-1] if b_ended else enriched_b[k]

        t_s = (
            float(sa_active["time_s"])
            if sa_active is not None
            else (float(sb_active["time_s"]) if sb_active is not None else round(k * 0.1, 4))
        )

        if sa_active is not None and sb_active is not None:
            da = sa_active["distance_yd"]
            db = sb_active["distance_yd"]
            spd_a = sa_active["speed_yd_s"]
            spd_b = sb_active["speed_yd_s"]
            cr_a = sa_active["closing_rate_yd_s"]
            cr_b = sb_active["closing_rate_yd_s"]
            d_diff = round(float(da) - float(db), 4) if (da is not None and db is not None) else None
            s_diff = round(float(spd_a) - float(spd_b), 4) if (spd_a is not None and spd_b is not None) else None
            c_diff = round(float(cr_a) - float(cr_b), 4) if (cr_a is not None and cr_b is not None) else None
            both_active = True
        else:
            # Never extrapolate telemetry differences after either play has ended
            d_diff = None
            s_diff = None
            c_diff = None
            both_active = False

        elapsed_steps.append({
            "step_index": k,
            "elapsed_time_s": round(t_s, 4),
            "play_a_ended": a_ended,
            "play_b_ended": b_ended,
            "both_plays_active": both_active,
            "play_a_active_sample": sa_active,
            "play_b_active_sample": sb_active,
            "play_a_display_sample": sa_held,
            "play_b_display_sample": sb_held,
            "distance_diff_a_minus_b_yd": d_diff,
            "speed_diff_a_minus_b_yd_s": s_diff,
            "closing_rate_diff_a_minus_b_yd_s": c_diff,
        })

    same_game = int(play_a_payload.get("game", 0)) == int(play_b_payload.get("game", 0))
    same_player = int(series_a["player_id"]) == int(series_b["player_id"])
    team_a = str(play_a_payload.get("defense", ""))
    team_b = str(play_b_payload.get("defense", ""))
    same_team = team_a == team_b and bool(team_a)

    if abs(dur_a - dur_b) <= 1e-4:
        shorter_side = "equal"
    elif dur_a < dur_b:
        shorter_side = "A"
    else:
        shorter_side = "B"

    return {
        "same_game": same_game,
        "same_player": same_player,
        "same_team": same_team,
        "unequal_duration": abs(dur_a - dur_b) > 1e-4,
        "shorter_play_side": shorter_side,
        "duration_a_s": dur_a,
        "duration_b_s": dur_b,
        "shared_timeline_duration_s": shared_dur,
        "total_steps": n_steps,
        "play_a_meta": {
            "game_id": int(play_a_payload.get("game", 0)),
            "play_id": int(play_a_payload.get("play", 0)),
            "offense": str(play_a_payload.get("offense", "")),
            "defense": team_a,
            "end_event": str(play_a_payload.get("end_event", "")),
            "end_label": str(play_a_payload.get("end_label", "")),
            "duration_s": dur_a,
            "player_id": series_a["player_id"],
            "player_tag": series_a["tag"],
            "player_name": series_a["name"],
            "player_number": series_a["number"],
            "summary": series_a["summary"],
        },
        "play_b_meta": {
            "game_id": int(play_b_payload.get("game", 0)),
            "play_id": int(play_b_payload.get("play", 0)),
            "offense": str(play_b_payload.get("offense", "")),
            "defense": team_b,
            "end_event": str(play_b_payload.get("end_event", "")),
            "end_label": str(play_b_payload.get("end_label", "")),
            "duration_s": dur_b,
            "player_id": series_b["player_id"],
            "player_tag": series_b["tag"],
            "player_name": series_b["name"],
            "player_number": series_b["number"],
            "summary": series_b["summary"],
        },
        "elapsed_steps": elapsed_steps,
        "equivalency_disclaimer": CROSS_PLAY_DISCLAIMER,
    }


def evaluate_cross_play_at_time(
    cross_comparison: Dict[str, Any],
    elapsed_time_s: float,
) -> Dict[str, Any]:
    """Evaluate a prebuilt `cross_comparison` at a specific elapsed time `t >= 0` without extrapolating ended plays."""
    steps = cross_comparison.get("elapsed_steps", [])
    if not steps:
        raise ValueError("cross_comparison contains no elapsed_steps.")
    shared_dur = float(cross_comparison["shared_timeline_duration_s"])
    t_clamped = max(0.0, min(shared_dur, float(elapsed_time_s)))
    nearest_step = min(
        steps, key=lambda st: abs(float(st["elapsed_time_s"]) - t_clamped)
    )
    return {
        "requested_elapsed_time_s": elapsed_time_s,
        "clamped_elapsed_time_s": round(t_clamped, 4),
        "step_index": nearest_step["step_index"],
        "measured_elapsed_time_s": nearest_step["elapsed_time_s"],
        "play_a_ended": nearest_step["play_a_ended"],
        "play_b_ended": nearest_step["play_b_ended"],
        "both_plays_active": nearest_step["both_plays_active"],
        "play_a_active_sample": nearest_step["play_a_active_sample"],
        "play_b_active_sample": nearest_step["play_b_active_sample"],
        "play_a_display_sample": nearest_step["play_a_display_sample"],
        "play_b_display_sample": nearest_step["play_b_display_sample"],
        "distance_diff_a_minus_b_yd": nearest_step["distance_diff_a_minus_b_yd"],
        "speed_diff_a_minus_b_yd_s": nearest_step["speed_diff_a_minus_b_yd_s"],
        "closing_rate_diff_a_minus_b_yd_s": nearest_step["closing_rate_diff_a_minus_b_yd_s"],
    }


def validate_cross_game_record_identity(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate that cross-game analytical records use unique composite keys (game_id, play_id, nfl_id).

    Prevents playId collisions across different games when aggregating player-level statistics.
    """
    seen_composite = set()
    play_ids_by_game: Dict[int, set] = {}
    players_seen = set()

    for rec in records:
        gid = int(rec["game_id"])
        pid = int(rec["play_id"])
        nid = int(rec["nfl_id"])
        comp = (gid, pid, nid)
        if comp in seen_composite:
            raise ValueError(
                f"Duplicate composite key (game_id, play_id, nfl_id) detected: {comp}"
            )
        seen_composite.add(comp)
        play_ids_by_game.setdefault(gid, set()).add(pid)
        players_seen.add(nid)

    # Check whether any numeric play_id appears in more than one game (proving composite key is required)
    all_pids: Dict[int, List[int]] = {}
    for gid, pset in play_ids_by_game.items():
        for pid in pset:
            all_pids.setdefault(pid, []).append(gid)
    colliding_play_ids = {pid: gids for pid, gids in all_pids.items() if len(gids) > 1}

    return {
        "valid": True,
        "total_records": len(records),
        "distinct_games": len(play_ids_by_game),
        "distinct_players": len(players_seen),
        "colliding_play_ids_across_games": colliding_play_ids,
    }


def extract_play_rusher_records(
    play_payloads: Sequence[Dict[str, Any]],
    pressure_threshold_yd: float = 2.0,
) -> List[Dict[str, Any]]:
    """Extract per-play, per-rusher analytical records across one or more games.

    Validates composite key uniqueness `(game_id, play_id, nfl_id)` and preserves
    contextual metadata (down, formation, outcome, team, opponent, PFF labels).
    """
    records: List[Dict[str, Any]] = []
    seen_keys = set()

    for play in play_payloads:
        game_id = int(play.get("game", 0))
        play_id = int(play.get("play", 0))
        offense = str(play.get("offense", ""))
        defense = str(play.get("defense", ""))
        quarter = play.get("quarter")
        down = play.get("down")
        yards_to_go = play.get("yards_to_go")
        formation = str(play.get("formation", "UNKNOWN") or "UNKNOWN")
        outcome = str(play.get("outcome", ""))
        outcome_label = str(play.get("outcome_label", ""))
        dropback_type = str(play.get("dropback_type", "UNKNOWN") or "UNKNOWN")
        play_action = play.get("play_action")
        coverage = str(play.get("pff_pass_coverage", "UNKNOWN") or "UNKNOWN")
        coverage_type = str(play.get("pff_pass_coverage_type", "UNKNOWN") or "UNKNOWN")
        duration = float(play.get("duration", 0.0))
        end_label = str(play.get("end_label", ""))

        r_stats_map = {
            int(rs["id"]): rs for rs in play.get("rusher_stats", []) if "id" in rs
        }
        tel = play.get("telemetry")
        if tel and "rushers" in tel:
            rusher_series_map = tel["rushers"]
        else:
            rusher_series_map = {}
            for r in play.get("rushers", []):
                rid = int(r["id"])
                s_obj = compute_player_telemetry_series(
                    play, rid, pressure_threshold_yd=pressure_threshold_yd
                )
                rusher_series_map[str(rid)] = s_obj

        for r in sorted(play.get("rushers", []), key=lambda x: int(x["id"])):
            rid = int(r["id"])
            comp_key = (game_id, play_id, rid)
            if comp_key in seen_keys:
                raise ValueError(
                    f"Duplicate (gameId, playId, nflId) record detected: {comp_key}"
                )
            seen_keys.add(comp_key)

            r_ser = rusher_series_map.get(str(rid), {})
            r_sum = r_ser.get("summary", {})
            rs = r_stats_map.get(rid, {})

            hit = bool(rs.get("pff_hit", False))
            hurry = bool(rs.get("pff_hurry", False))
            sack = bool(rs.get("pff_sack", False))
            pff_press = bool(rs.get("pff_pressure", hit or hurry or sack))
            if sack:
                pff_cat = "Sack"
            elif hit:
                pff_cat = "Hit"
            elif hurry:
                pff_cat = "Hurry"
            else:
                pff_cat = "No PFF pressure"

            min_d = round(
                float(r_sum.get("closest_distance_yards", rs.get("closest_yards", 0.0))), 4
            )
            t_close = round(float(r_sum.get("closest_time_s", 0.0)), 4)
            init_d = round(float(r_sum.get("initial_distance_yards", 0.0)), 4)
            first_2yd = (
                round(float(r_sum["first_within_2yd_time_s"]), 4)
                if r_sum.get("first_within_2yd_time_s") is not None
                else rs.get("first_within")
            )
            records.append({
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": rid,
                "player_name": str(r.get("name", rid)),
                "name": str(r.get("name", rid)),
                "player_tag": str(r_ser.get("tag", "")),
                "jersey_number": str(r_ser.get("number", "")),
                "number": str(r_ser.get("number", "")),
                "team": defense,
                "defense": defense,
                "opponent": offense,
                "offense": offense,
                "quarter": int(quarter) if quarter is not None else None,
                "down": int(down) if down is not None else None,
                "yards_to_go": int(yards_to_go) if yards_to_go is not None else None,
                "formation": formation,
                "outcome": outcome,
                "outcome_label": outcome_label,
                "dropback_type": dropback_type,
                "play_action": bool(play_action) if play_action is not None else None,
                "coverage": coverage,
                "coverage_type": coverage_type,
                "play_duration_s": round(duration, 2),
                "end_label": end_label,
                "pff_role": "Pass Rush",
                "pff_hit": hit,
                "pff_hurry": hurry,
                "pff_sack": sack,
                "pff_pressure": pff_press,
                "pff_label": pff_cat,
                "min_distance_yd": min_d,
                "closest_distance_yards": min_d,
                "time_to_closest_s": t_close,
                "closest_time_s": t_close,
                "initial_distance_yd": init_d,
                "init_distance_yards": init_d,
                "final_distance_yd": round(float(r_sum.get("final_distance_yards", 0.0)), 4),
                "net_distance_change_yd": round(float(r_sum.get("net_distance_change_yards", 0.0)), 4),
                "mean_speed_yd_s": round(float(r_sum.get("mean_speed_yd_s", 0.0)), 4),
                "max_speed_yd_s": round(float(r_sum.get("max_speed_yd_s", 0.0)), 4),
                "mean_closing_rate_yd_s": round(float(r_sum.get("mean_closing_rate_yd_s", 0.0)), 4),
                "peak_closing_rate_yd_s": round(float(r_sum.get("peak_closing_rate_yd_s", 0.0)), 4),
                "max_acceleration_yd_s2": round(float(r_sum.get("max_acceleration_yd_s2", 0.0)), 4),
                "entered_2yd_zone": bool(
                    r_sum.get("entered_2yd_zone", rs.get("first_within") is not None)
                ),
                "first_within_2yd_s": first_2yd,
                "first_within_2yd_time_s": first_2yd,
            })

    return records


def _quantile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return round(float(sorted_vals[0]), 4)
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return round((1.0 - frac) * float(sorted_vals[lo]) + frac * float(sorted_vals[hi]), 4)


def compute_multi_play_player_analytics(
    records: Sequence[Dict[str, Any]],
    filters: Optional[Dict[str, Any]] = None,
    min_plays: int = 1,
) -> Dict[str, Any]:
    """Compute multi-play and player-level statistics with explicit denominators and distributions."""
    filters = filters or {}

    def _play_context_matches(rec: Dict[str, Any]) -> bool:
        if filters.get("game_id") not in (None, "", "all"):
            g_filter = filters["game_id"]
            if isinstance(g_filter, (list, tuple, set)):
                if int(rec["game_id"]) not in {int(x) for x in g_filter}:
                    return False
            elif int(rec["game_id"]) != int(g_filter):
                return False
        if filters.get("team") not in (None, "", "all"):
            if str(rec["team"]).upper() != str(filters["team"]).upper():
                return False
        if filters.get("opponent") not in (None, "", "all"):
            if str(rec["opponent"]).upper() != str(filters["opponent"]).upper():
                return False
        if filters.get("down") not in (None, "", "all"):
            if rec.get("down") != int(filters["down"]):
                return False
        if filters.get("formation") not in (None, "", "all"):
            if str(rec.get("formation", "")).upper() != str(filters["formation"]).upper():
                return False
        if filters.get("outcome") not in (None, "", "all"):
            out_f = str(filters["outcome"]).upper()
            if (
                str(rec.get("outcome", "")).upper() != out_f
                and str(rec.get("outcome_label", "")).upper() != out_f
            ):
                return False
        return True

    def _rusher_matches(rec: Dict[str, Any]) -> bool:
        if not _play_context_matches(rec):
            return False
        if filters.get("nfl_id") not in (None, "", "all"):
            if int(rec["nfl_id"]) != int(filters["nfl_id"]):
                return False
        if filters.get("player_search"):
            q = str(filters["player_search"]).strip().lower()
            if (
                q not in str(rec["player_name"]).lower()
                and q not in str(rec["nfl_id"])
                and q != str(rec.get("jersey_number", "")).lower()
            ):
                return False
        pff_f = str(filters.get("pff_filter", "all") or "all").lower()
        if pff_f == "any_pressure" and not rec.get("pff_pressure"):
            return False
        if pff_f == "sack" and not rec.get("pff_sack"):
            return False
        if pff_f == "hit" and not rec.get("pff_hit"):
            return False
        if pff_f == "hurry" and not rec.get("pff_hurry"):
            return False
        if pff_f == "no_pressure" and rec.get("pff_pressure"):
            return False
        return True

    context_records = [r for r in records if _play_context_matches(r)]
    filtered_records = [r for r in records if _rusher_matches(r)]

    # Explicit denominators across context-matching plays
    distinct_games = sorted({int(r["game_id"]) for r in context_records})
    distinct_plays = sorted({(int(r["game_id"]), int(r["play_id"])) for r in context_records})
    team_play_sets: Dict[str, set] = {}
    for r in context_records:
        t_key = str(r["team"])
        team_play_sets.setdefault(t_key, set()).add((int(r["game_id"]), int(r["play_id"])))
    team_eligible_play_counts = {t: len(pset) for t, pset in team_play_sets.items()}

    # Group filtered records by (nfl_id)
    by_player: Dict[int, List[Dict[str, Any]]] = {}
    for r in filtered_records:
        by_player.setdefault(int(r["nfl_id"]), []).append(r)

    player_summaries: List[Dict[str, Any]] = []
    for nfl_id, p_recs in by_player.items():
        n = len(p_recs)
        if n < max(1, int(min_plays)):
            continue
        p_name = str(p_recs[0]["player_name"])
        p_team = str(p_recs[0]["team"])
        p_num = str(p_recs[0].get("jersey_number", ""))
        p_games = sorted({int(r["game_id"]) for r in p_recs})
        team_denom = max(1, team_eligible_play_counts.get(p_team, n))

        min_dists = sorted(float(r["min_distance_yd"]) for r in p_recs)
        t_closests = sorted(float(r["time_to_closest_s"]) for r in p_recs)
        mean_closes = sorted(float(r["mean_closing_rate_yd_s"]) for r in p_recs)
        peak_closes = sorted(float(r["peak_closing_rate_yd_s"]) for r in p_recs)
        max_spds = sorted(float(r["max_speed_yd_s"]) for r in p_recs)

        within_cnt = sum(1 for r in p_recs if r["entered_2yd_zone"])
        arrivals = [
            float(r["first_within_2yd_s"])
            for r in p_recs
            if r["entered_2yd_zone"] and r.get("first_within_2yd_s") is not None
        ]
        hits = sum(1 for r in p_recs if r["pff_hit"])
        hurries = sum(1 for r in p_recs if r["pff_hurry"])
        sacks = sum(1 for r in p_recs if r["pff_sack"])
        press_cnt = sum(1 for r in p_recs if r["pff_pressure"])

        player_summaries.append({
            "nfl_id": nfl_id,
            "player_name": p_name,
            "jersey_number": p_num,
            "team": p_team,
            "games_count": len(p_games),
            "game_ids": p_games,
            "eligible_plays": n,
            "team_eligible_plays_denominator": team_denom,
            "snap_share_pct": round(100.0 * n / float(team_denom), 2),
            "mean_min_distance_yd": round(statistics.mean(min_dists), 4),
            "median_min_distance_yd": round(statistics.median(min_dists), 4),
            "p25_min_distance_yd": _quantile(min_dists, 0.25),
            "p75_min_distance_yd": _quantile(min_dists, 0.75),
            "best_min_distance_yd": round(min_dists[0], 4),
            "mean_time_to_closest_s": round(statistics.mean(t_closests), 4),
            "median_time_to_closest_s": round(statistics.median(t_closests), 4),
            "mean_closing_rate_yd_s": round(statistics.mean(mean_closes), 4),
            "median_closing_rate_yd_s": round(statistics.median(mean_closes), 4),
            "peak_closing_rate_yd_s": round(max(peak_closes), 4),
            "mean_max_speed_yd_s": round(statistics.mean(max_spds), 4),
            "peak_speed_yd_s": round(max(max_spds), 4),
            "within_2yd_count": within_cnt,
            "within_2yd_pct": round(100.0 * within_cnt / float(n), 2),
            "mean_arrival_2yd_s": (
                round(statistics.mean(arrivals), 4) if arrivals else None
            ),
            "pff_hit_count": hits,
            "pff_hurry_count": hurries,
            "pff_sack_count": sacks,
            "pff_pressure_count": press_cnt,
            "pff_pressure_rate_pct": round(100.0 * press_cnt / float(n), 2),
            "distributions": {
                "min_distances_yd": [round(float(r["min_distance_yd"]), 3) for r in p_recs],
                "times_to_closest_s": [round(float(r["time_to_closest_s"]), 2) for r in p_recs],
                "mean_closing_rates_yd_s": [
                    round(float(r["mean_closing_rate_yd_s"]), 3) for r in p_recs
                ],
                "play_refs": [
                    {
                        "game_id": int(r["game_id"]),
                        "play_id": int(r["play_id"]),
                        "down": r["down"],
                        "formation": r["formation"],
                        "outcome_label": r["outcome_label"],
                        "min_distance_yd": round(float(r["min_distance_yd"]), 2),
                        "time_to_closest_s": round(float(r["time_to_closest_s"]), 1),
                        "pff_label": r["pff_label"],
                    }
                    for r in p_recs
                ],
            },
        })

    # Default sort: lowest median min QB distance, then most eligible plays
    player_summaries.sort(
        key=lambda p: (float(p["median_min_distance_yd"]), -int(p["eligible_plays"]))
    )

    return {
        "denominators": {
            "total_games_in_scope": len(distinct_games),
            "game_ids_in_scope": distinct_games,
            "total_eligible_plays_in_scope": len(distinct_plays),
            "total_rusher_snaps_in_scope": len(filtered_records),
            "team_eligible_play_counts": team_eligible_play_counts,
            "min_plays_threshold": max(1, int(min_plays)),
            "eligibility_rule": (
                "Eligible plays require 1 PFF Passer ('Pass'), >=1 PFF Pass Rusher ('Pass Rush'), "
                "an official 'ball_snap' event, and a supported play endpoint "
                "('pass_forward', 'qb_sack', 'qb_strip_sack', or 'run') with validated 10 Hz frame timing."
            ),
        },
        "players": player_summaries,
        "non_causal_disclaimer": (
            "Descriptive statistics only: QB proximity, closing rates, and PFF label frequencies "
            "depend on pass-protection schemes, double-teams, stunt assignments, and QB dropback depth, "
            "and do not establish causal claims about individual player ability or offensive-line quality."
        ),
    }


def build_compact_cross_game_play_payload(play_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact play representation (QB + pass rushers + telemetry) for cross-game comparison without embedding all 22 players per frame."""
    qb_id = int(play_payload["qb"])
    rusher_ids = {int(r["id"]) for r in play_payload.get("rushers", [])}
    keep_ids = rusher_ids | {qb_id}

    compact_frames: List[Dict[str, Any]] = []
    for ff in play_payload.get("frames", []):
        compact_ents = [
            e
            for e in ff.get("entities", [])
            if e.get("role") == "Football" or (e.get("id") is not None and int(e["id"]) in keep_ids)
        ]
        compact_frames.append({
            "frame": int(ff["frame"]),
            "time": float(ff["time"]),
            "entities": compact_ents,
            "distances": dict(ff.get("distances", {})),
            "nearest": ff.get("nearest"),
        })

    return {
        "game": int(play_payload["game"]),
        "play": int(play_payload["play"]),
        "qb": qb_id,
        "offense": play_payload.get("offense", ""),
        "defense": play_payload.get("defense", ""),
        "quarter": play_payload.get("quarter"),
        "down": play_payload.get("down"),
        "yards_to_go": play_payload.get("yards_to_go"),
        "formation": play_payload.get("formation", ""),
        "description": play_payload.get("description", ""),
        "outcome": play_payload.get("outcome", ""),
        "outcome_label": play_payload.get("outcome_label", ""),
        "yards": play_payload.get("yards", 0.0),
        "end_event": play_payload.get("end_event", ""),
        "end_label": play_payload.get("end_label", ""),
        "duration": float(play_payload.get("duration", 0.0)),
        "label": play_payload.get("label", ""),
        "headline": play_payload.get("headline", ""),
        "summary": play_payload.get("summary", {}),
        "rusher_stats": play_payload.get("rusher_stats", []),
        "pff_label": play_payload.get("pff_label", ""),
        "pff_pressure": bool(play_payload.get("pff_pressure", False)),
        "view": play_payload.get("view", {}),
        "rushers": play_payload.get("rushers", []),
        "frames": compact_frames,
        "telemetry": play_payload.get("telemetry"),
    }


def build_play_telemetry_bundle(
    play_payload: Dict[str, Any],
    pressure_threshold_yd: float = 2.0,
    num_segments: int = 3,
) -> Dict[str, Any]:
    """Build compact per-play telemetry and sector bundle to embed in the HTML replay payload."""
    rushers = play_payload.get("rushers", [])
    sorted_rushers = sorted(rushers, key=lambda r: int(r["id"]))

    per_rusher_series: Dict[str, Any] = {}
    for r in sorted_rushers:
        rid = int(r["id"])
        s = compute_player_telemetry_series(
            play_payload, rid, pressure_threshold_yd=pressure_threshold_yd
        )
        per_rusher_series[str(rid)] = {
            "player_id": rid,
            "nfl_id": rid,
            "tag": s["tag"],
            "name": s["name"],
            "number": s["number"],
            "summary": s["summary"],
            "distance_yd": [smp["distance_yd"] for smp in s["samples"]],
            "distance_change_from_snap_yd": [
                smp["distance_change_from_snap_yd"] for smp in s["samples"]
            ],
            "min_distance_so_far_yd": [
                smp["min_distance_so_far_yd"] for smp in s["samples"]
            ],
            "time_of_min_so_far_s": [
                smp["time_of_min_so_far_s"] for smp in s["samples"]
            ],
            "speed_yd_s": [smp["speed_yd_s"] for smp in s["samples"]],
            "acceleration_yd_s2": [smp["acceleration_yd_s2"] for smp in s["samples"]],
            "closing_rate_yd_s": [smp["closing_rate_yd_s"] for smp in s["samples"]],
            "discontinuity": [smp["discontinuity"] for smp in s["samples"]],
        }

    segments = compute_timeline_segments(
        play_payload,
        player_ids=[int(r["id"]) for r in sorted_rushers],
        num_segments=num_segments,
        pressure_threshold_yd=pressure_threshold_yd,
    )

    # Default head-to-head pair: closest rusher vs runner-up rusher (or first two rushers)
    by_closest = sorted(
        sorted_rushers,
        key=lambda r: float(
            per_rusher_series[str(int(r["id"]))]["summary"]["closest_distance_yards"]
        ),
    )
    default_a = int(by_closest[0]["id"]) if by_closest else None
    default_b = (
        int(by_closest[1]["id"])
        if len(by_closest) > 1
        else (default_a if default_a is not None else None)
    )

    return {
        "times_s": [float(ff["time"]) for ff in play_payload.get("frames", [])],
        "default_player_a_id": default_a,
        "default_player_b_id": default_b,
        "rushers": per_rusher_series,
        "segments": segments,
        "quantity_distinction_note": QUANTITY_DISTINCTION_NOTE,
    }

