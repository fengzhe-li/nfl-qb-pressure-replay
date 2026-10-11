"""Standalone visual replay frame generator for VLM-assisted QB pressure analysis.

Reuses the coordinate conventions and visual styling of QB Pressure Replay v2 while
strictly stripping all ground-truth QB-rusher distances, nearest-rusher indicators,
play descriptions/outcomes, and PFF scouting annotations so that visual inputs are
completely separated from evaluation references.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import io
import math

from PIL import Image, ImageDraw, ImageFont

# Same palette as QB Pressure Replay v2 HTML Canvas
RUSHER_PALETTE = [
    "#ff9800",
    "#00e5ff",
    "#e040fb",
    "#76ff03",
    "#ff4081",
    "#b388ff",
    "#1de9b6",
    "#ffd180",
]

FORBIDDEN_PLAY_KEYS = frozenset({
    "distances",
    "nearest",
    "summary",
    "rusher_stats",
    "pff_label",
    "pff_pressure",
    "pff_html",
    "pff_hit",
    "pff_hurry",
    "pff_sack",
    "headline",
    "label",
    "description",
    "outcome",
    "outcome_label",
    "yards",
    "telemetry",
})

FORBIDDEN_FRAME_KEYS = frozenset({
    "distances",
    "nearest",
})

FORBIDDEN_ENTITY_KEYS = frozenset({
    "distance",
    "distances",
    "pff_hit",
    "pff_hurry",
    "pff_sack",
    "pff_pressure",
    "closest_yards",
    "first_within",
})


@dataclass
class FrameSequenceBundle:
    """Container for generated sequential replay frames and sanitized metadata."""

    game_id: int
    play_id: int
    images: List[Image.Image]
    image_paths: List[Path] = field(default_factory=list)
    sampled_frame_indices: List[int] = field(default_factory=list)
    sampled_frame_ids: List[int] = field(default_factory=list)
    sampled_times: List[float] = field(default_factory=list)
    window_start_frame: int = 0
    window_end_frame: int = 0
    window_start_time: float = 0.0
    window_end_time: float = 0.0
    candidate_rushers: List[Dict[str, Any]] = field(default_factory=list)
    sanitized_play: Dict[str, Any] = field(default_factory=dict)

    def to_preview_data_uris(self, max_width: int = 320) -> List[str]:
        """Encode generated frames as compact PNG data URIs for optional dashboard preview."""
        uris: List[str] = []
        for img in self.images:
            if img.width > max_width:
                ratio = max_width / float(img.width)
                resized = img.resize(
                    (max_width, max(1, int(round(img.height * ratio)))),
                    Image.Resampling.LANCZOS,
                )
            else:
                resized = img
            buf = io.BytesIO()
            resized.save(buf, format="PNG", optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            uris.append(f"data:image/png;base64,{encoded}")
        return uris


def _last_name(full_name: str) -> str:
    parts = str(full_name).strip().split()
    if not parts:
        return ""
    last = parts[-1]
    if last in {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV"} and len(parts) > 1:
        return parts[-2]
    return last


def select_representative_frame_indices(
    total_frames: int,
    num_frames: int = 4,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> List[int]:
    """Select evenly spaced sequential frame indices across [start_idx, end_idx].

    Selection depends only on frame index bounds (never on QB-rusher distances).
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if end_idx is None:
        end_idx = total_frames - 1
    if start_idx < 0 or end_idx >= total_frames or start_idx > end_idx:
        raise ValueError(
            f"Invalid frame window [{start_idx}, {end_idx}] for total_frames={total_frames}."
        )
    span = end_idx - start_idx + 1
    if num_frames == 1 or span == 1:
        return [end_idx]
    actual_count = min(num_frames, span)
    indices: List[int] = []
    for k in range(actual_count):
        pos = start_idx + int(round(k * (end_idx - start_idx) / float(actual_count - 1)))
        if not indices or pos != indices[-1]:
            indices.append(pos)
    return indices


def sanitize_play_for_visual_input(
    play_payload: Dict[str, Any],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a copy of the play payload stripped of all ground-truth and PFF fields."""
    if not isinstance(play_payload, dict):
        raise ValueError("play_payload must be a dictionary.")
    frames = play_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("play_payload must contain a non-empty 'frames' list.")
    if "qb" not in play_payload or play_payload["qb"] is None:
        raise ValueError("play_payload is missing 'qb' identifier.")
    if "view" not in play_payload or not isinstance(play_payload["view"], dict):
        raise ValueError("play_payload is missing 'view' coordinate bounds.")
    rushers = play_payload.get("rushers")
    if not isinstance(rushers, list) or not rushers:
        raise ValueError("play_payload must contain a non-empty 'rushers' list.")

    if end_idx is None:
        end_idx = len(frames) - 1
    if start_idx < 0 or end_idx >= len(frames) or start_idx > end_idx:
        raise ValueError(f"Invalid frame index range [{start_idx}, {end_idx}].")

    window_frames = frames[start_idx : end_idx + 1]

    # Map jersey numbers from the first frame in the window
    jersey_by_id: Dict[int, str] = {}
    for ent in window_frames[0].get("entities", []):
        eid = ent.get("id")
        if eid is not None and ent.get("number"):
            jersey_by_id[int(eid)] = str(ent["number"])

    # Sort pass rushers deterministically by nflId (never by distance!)
    sorted_rushers = sorted(rushers, key=lambda r: int(r["id"]))
    candidate_rushers: List[Dict[str, Any]] = []
    for idx, r in enumerate(sorted_rushers):
        rid = int(r["id"])
        name = str(r.get("name", rid))
        number = jersey_by_id.get(rid, "")
        tag = f"R{idx + 1}"
        color = RUSHER_PALETTE[idx % len(RUSHER_PALETTE)]
        candidate_rushers.append({
            "tag": tag,
            "id": rid,
            "name": name,
            "last_name": _last_name(name),
            "number": number,
            "color": color,
        })

    sanitized_frames: List[Dict[str, Any]] = []
    for f in window_frames:
        clean_entities: List[Dict[str, Any]] = []
        for e in f.get("entities", []):
            clean_entities.append({
                "id": None if e.get("id") is None else int(e["id"]),
                "x": float(e["x"]),
                "y": float(e["y"]),
                "team": str(e.get("team", "")),
                "name": str(e.get("name", "")),
                "number": str(e.get("number", "")),
                "role": str(e.get("role", "Unknown")),
            })
        sanitized_frames.append({
            "frame": int(f["frame"]),
            "time": float(f["time"]),
            "entities": clean_entities,
        })

    view = play_payload["view"]
    return {
        "game": int(play_payload.get("game", 0)),
        "play": int(play_payload.get("play", 0)),
        "qb": int(play_payload["qb"]),
        "offense": str(play_payload.get("offense", "OFF")),
        "defense": str(play_payload.get("defense", "DEF")),
        "flipped": bool(play_payload.get("flipped", False)),
        "view": {
            "x0": float(view["x0"]),
            "x1": float(view["x1"]),
            "los": float(view["los"]),
            "qb_y": float(view["qb_y"]),
        },
        "candidate_rushers": candidate_rushers,
        "frames": sanitized_frames,
    }


def verify_no_ground_truth_leakage(sanitized_play: Dict[str, Any]) -> bool:
    """Verify that a sanitized play object contains no forbidden distance or PFF keys."""
    for key in sanitized_play:
        if key in FORBIDDEN_PLAY_KEYS:
            raise AssertionError(f"Forbidden key '{key}' found in sanitized play.")
    for f in sanitized_play.get("frames", []):
        for key in f:
            if key in FORBIDDEN_FRAME_KEYS:
                raise AssertionError(f"Forbidden key '{key}' found in sanitized frame.")
        for e in f.get("entities", []):
            for key in e:
                if key in FORBIDDEN_ENTITY_KEYS:
                    raise AssertionError(f"Forbidden key '{key}' found in sanitized entity.")
    for r in sanitized_play.get("candidate_rushers", []):
        for key in r:
            if key in FORBIDDEN_ENTITY_KEYS:
                raise AssertionError(f"Forbidden key '{key}' found in candidate rusher.")
    return True


def _load_fonts() -> Dict[str, ImageFont.ImageFont]:
    """Load readable sans-serif fonts with safe fallback to PIL default font."""
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    regular_path = None
    bold_path = None
    for cand in font_candidates:
        if Path(cand).is_file():
            if "Bold" in cand and bold_path is None:
                bold_path = cand
            elif regular_path is None:
                regular_path = cand
    bold_path = bold_path or regular_path
    regular_path = regular_path or bold_path

    if regular_path:
        try:
            return {
                "title": ImageFont.truetype(bold_path, 14),
                "subtitle": ImageFont.truetype(regular_path, 12),
                "label": ImageFont.truetype(bold_path, 12),
                "marker": ImageFont.truetype(bold_path, 10),
                "small": ImageFont.truetype(regular_path, 11),
            }
        except Exception:
            pass
    default_font = ImageFont.load_default()
    return {
        "title": default_font,
        "subtitle": default_font,
        "label": default_font,
        "marker": default_font,
        "small": default_font,
    }


def _compute_pocket_viewport(
    sanitized_play: Dict[str, Any],
    aspect_ratio: float,
) -> Tuple[float, float, float, float]:
    """Compute a pocket-focused viewport (x0, x1, y0, y1) in field yards.

    Encompasses the QB, all pass rushers across the window, and the line of scrimmage
    with generous padding while preserving exact 1:1 yard aspect ratio.
    """
    qb_id = sanitized_play["qb"]
    rusher_ids = {r["id"] for r in sanitized_play["candidate_rushers"]}
    los = float(sanitized_play["view"]["los"])
    qb_y0 = float(sanitized_play["view"]["qb_y"])

    xs = [los]
    ys = [qb_y0]
    for f in sanitized_play["frames"]:
        for e in f["entities"]:
            if e["id"] == qb_id or e["id"] in rusher_ids:
                xs.append(float(e["x"]))
                ys.append(float(e["y"]))

    min_x, max_x = min(xs) - 5.5, max(xs) + 6.5
    min_y, max_y = min(ys) - 4.5, max(ys) + 4.5

    span_x = max(max_x - min_x, 18.0)
    span_y = max(max_y - min_y, 14.0)

    # Match canvas aspect ratio so 1 yard in X equals 1 yard in Y
    if span_x / span_y < aspect_ratio:
        span_x = span_y * aspect_ratio
    else:
        span_y = span_x / aspect_ratio

    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cy = min(max(cy, span_y / 2.0), 53.3 - span_y / 2.0)

    return cx - span_x / 2.0, cx + span_x / 2.0, cy - span_y / 2.0, cy + span_y / 2.0


def _boxes_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    pad: float = 2.0,
) -> bool:
    return not (
        a[2] + pad < b[0]
        or a[0] - pad > b[2]
        or a[3] + pad < b[1]
        or a[1] - pad > b[3]
    )


def _render_single_frame(
    sanitized_play: Dict[str, Any],
    window_frame_idx: int,
    seq_position: int,
    seq_total: int,
    width: int = 560,
    height: int = 420,
) -> Image.Image:
    """Render one replay frame with two-pass non-overlapping callout labels."""
    fonts = _load_fonts()
    img = Image.new("RGB", (width, height), "#172338")
    draw = ImageDraw.Draw(img)

    field_left = 20
    field_right = width - 20
    field_top = 36
    field_bottom = height - 68
    field_w = field_right - field_left
    field_h = field_bottom - field_top

    x0, x1, y0, y1 = _compute_pocket_viewport(
        sanitized_play, aspect_ratio=field_w / float(max(field_h, 1))
    )
    scale = field_w / max(x1 - x0, 1.0)

    def px(x: float) -> float:
        return field_left + (x - x0) * scale

    def py(y: float) -> float:
        return field_bottom - (y - y0) * scale

    def in_bounds(x_px: float, y_px: float) -> bool:
        return (field_left <= x_px <= field_right) and (field_top <= y_px <= field_bottom)

    # Turf background
    draw.rectangle([field_left, field_top, field_right, field_bottom], fill="#194f3a")

    # 5-yard vertical lines and 1-yard tick marks
    start_x_grid = int(math.ceil(x0 / 5.0) * 5)
    for x_yd in range(start_x_grid, int(math.floor(x1)) + 1, 5):
        gx = px(float(x_yd))
        if field_left <= gx <= field_right:
            draw.line([(gx, field_top), (gx, field_bottom)], fill="#4c7c60", width=1)

    # Sidelines (y = 0 and y = 53.3)
    for y_side in (0.0, 53.3):
        sy = py(y_side)
        if field_top <= sy <= field_bottom:
            draw.line([(field_left, sy), (field_right, sy)], fill="#ffffff", width=2)

    # Line of scrimmage
    los_px = px(float(sanitized_play["view"]["los"]))
    if field_left <= los_px <= field_right:
        draw.line([(los_px, field_top), (los_px, field_bottom)], fill="#ffd54f", width=2)
        draw.text((los_px + 4, field_top + 4), "LOS", fill="#ffd54f", font=fonts["label"])

    frames = sanitized_play["frames"]
    frame_data = frames[window_frame_idx]
    qb_id = sanitized_play["qb"]
    offense_team = sanitized_play["offense"]
    rusher_map = {r["id"]: r for r in sanitized_play["candidate_rushers"]}

    # Pass 0: Draw movement trails from start of window up to current frame
    trail_start = max(0, window_frame_idx - 15)
    tracked_ids = [qb_id] + [r["id"] for r in sanitized_play["candidate_rushers"]]
    for tid in tracked_ids:
        pts = []
        for prev_f in frames[trail_start : window_frame_idx + 1]:
            ent = next((e for e in prev_f["entities"] if e["id"] == tid), None)
            if ent is not None:
                pts.append((px(ent["x"]), py(ent["y"])))
        if len(pts) >= 2:
            t_col = "#fef08a" if tid == qb_id else rusher_map[tid]["color"]
            draw.line(pts, fill=t_col, width=2)

    # Sort entities so QB and Pass Rushers render on top of linemen
    def _rank(e: Dict[str, Any]) -> int:
        if e["role"] == "Football":
            return 2
        if e["id"] == qb_id:
            return 5
        if e["role"] == "Pass Rush":
            return 4
        return 1

    sorted_entities = sorted(frame_data["entities"], key=_rank)
    circle_obstacles: List[Tuple[float, float, float, float]] = []
    callouts_to_place: List[Dict[str, Any]] = []

    # Pass 1: Draw all entity markers
    for e in sorted_entities:
        X = px(e["x"])
        Y = py(e["y"])
        if not in_bounds(X, Y):
            continue
        eid = e["id"]
        is_qb = eid == qb_id
        is_rusher = eid in rusher_map
        is_ball = e["role"] == "Football"

        if is_ball:
            col = "#c88b55"
            r = 5
            draw.ellipse([X - r, Y - r, X + r, Y + r], fill=col)
        elif is_qb:
            r = 12
            draw.ellipse([X - r, Y - r, X + r, Y + r], fill="#ffeb3b", outline="#101827", width=2)
            draw.text((X - 8, Y - 6), "QB", fill="#101827", font=fonts["marker"])
            circle_obstacles.append((X - r, Y - r, X + r, Y + r))
            num_str = f" #{e['number']}" if e.get("number") else ""
            callouts_to_place.append({
                "x": X,
                "y": Y,
                "r": r,
                "text": f"QB{num_str}",
                "color": "#ffeb3b",
                "priority": 0,
            })
        elif is_rusher:
            rmeta = rusher_map[eid]
            ring_color = rmeta["color"]
            r = 11
            draw.ellipse([X - r, Y - r, X + r, Y + r], fill="#101827", outline=ring_color, width=3)
            draw.text((X - 7, Y - 6), rmeta["tag"], fill=ring_color, font=fonts["marker"])
            circle_obstacles.append((X - r, Y - r, X + r, Y + r))
            num_str = f"#{rmeta['number']} " if rmeta["number"] else ""
            callouts_to_place.append({
                "x": X,
                "y": Y,
                "r": r,
                "text": f"{rmeta['tag']} {num_str}{rmeta['last_name']}",
                "color": ring_color,
                "priority": 1,
            })
        else:
            col = "#4299ff" if e["team"] == offense_team else "#f56b72"
            r = 6
            draw.ellipse([X - r, Y - r, X + r, Y + r], fill=col)
            circle_obstacles.append((X - r, Y - r, X + r, Y + r))

    # Pass 2: Place non-overlapping callout labels with leader lines
    placed_boxes: List[Tuple[float, float, float, float]] = list(circle_obstacles)
    callouts_to_place.sort(key=lambda c: c["priority"])

    # Candidate offsets (dx, dy) around a marker
    candidate_offsets = [
        (16, -8),
        (16, 10),
        (16, -24),
        (-115, -8),
        (-115, 10),
        (-115, -24),
        (-45, -28),
        (-45, 16),
        (24, 22),
        (24, -32),
        (-125, 22),
        (-125, -32),
    ]

    for item in callouts_to_place:
        X, Y = item["x"], item["y"]
        lbl = item["text"]
        tw = max(42.0, len(lbl) * 6.8 + 8.0)
        th = 16.0

        best_box = None
        best_penalty = float("inf")
        for dx, dy in candidate_offsets:
            bx0 = min(max(X + dx, field_left + 3), field_right - tw - 3)
            by0 = min(max(Y + dy, field_top + 3), field_bottom - th - 3)
            box = (bx0, by0, bx0 + tw, by0 + th)
            penalty = 0.0
            for obs in placed_boxes:
                if _boxes_overlap(box, obs, pad=2.0):
                    penalty += 1000.0
            # Prefer closer label positions when overlap-free
            penalty += 0.05 * math.hypot((bx0 + tw / 2.0) - X, (by0 + th / 2.0) - Y)
            if penalty < best_penalty:
                best_penalty = penalty
                best_box = box
                if penalty < 10.0:
                    break

        assert best_box is not None
        placed_boxes.append(best_box)
        bx0, by0, bx1, by1 = best_box
        # Draw leader line from circle edge toward label box center
        cx_box = 0.5 * (bx0 + bx1)
        cy_box = 0.5 * (by0 + by1)
        draw.line([(X, Y), (cx_box, cy_box)], fill=item["color"], width=1)
        draw.rectangle([bx0, by0, bx1, by1], fill="#101827", outline=item["color"], width=1)
        draw.text((bx0 + 4, by0 + 1), lbl, fill=item["color"], font=fonts["label"])

    # Header bar
    header_left = (
        f"Play {sanitized_play['play']} | Frame {seq_position}/{seq_total} "
        f"(ID {frame_data['frame']}, +{frame_data['time']:.1f}s)"
    )
    draw.text((field_left, 10), header_left, fill="#ffffff", font=fonts["title"])
    draw.text(
        (width - 230, 12),
        "Offense -> +x (right)",
        fill="#cbd5e1",
        font=fonts["subtitle"],
    )

    # Yard numbers along bottom of turf
    start_x_num = int(math.ceil(x0 / 5.0) * 5)
    for x_yd in range(start_x_num, int(math.floor(x1)) + 1, 5):
        if x_yd < 10 or x_yd > 110:
            continue
        n = x_yd - 10
        yd_label = str(n if n <= 50 else 100 - n)
        draw.text((px(float(x_yd)) - 6, field_bottom + 2), yd_label, fill="#cbd5e1", font=fonts["small"])

    # Candidate pass-rusher legend bar at the bottom
    legend_y0 = field_bottom + 18
    draw.rectangle([field_left, legend_y0, field_right, height - 4], fill="#101827")
    rx = field_left + 6
    ry = legend_y0 + 4
    for rmeta in sanitized_play["candidate_rushers"]:
        num_part = f"#{rmeta['number']} " if rmeta["number"] else ""
        item_text = f"{rmeta['tag']}:{num_part}{rmeta['last_name']}({rmeta['id']})"
        if rx + len(item_text) * 6.2 > field_right - 8:
            rx = field_left + 6
            ry += 15
        draw.ellipse([rx, ry + 3, rx + 8, ry + 11], fill=rmeta["color"])
        draw.text((rx + 11, ry), item_text, fill=rmeta["color"], font=fonts["small"])
        rx += int(len(item_text) * 6.2) + 16

    return img


def generate_play_frames(
    play_payload: Dict[str, Any],
    output_dir: Optional[Path] = None,
    num_frames: int = 4,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    width: int = 560,
    height: int = 420,
) -> FrameSequenceBundle:
    """Generate a sequence of sanitized replay images for VLM analysis."""
    sanitized = sanitize_play_for_visual_input(
        play_payload, start_idx=start_idx, end_idx=end_idx
    )
    verify_no_ground_truth_leakage(sanitized)

    window_frames = sanitized["frames"]
    rel_indices = select_representative_frame_indices(
        len(window_frames), num_frames=num_frames, start_idx=0, end_idx=len(window_frames) - 1
    )
    abs_indices = [start_idx + idx for idx in rel_indices]

    images: List[Image.Image] = []
    image_paths: List[Path] = []
    sampled_frame_ids: List[int] = []
    sampled_times: List[float] = []

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for seq_idx, rel_idx in enumerate(rel_indices, start=1):
        f_obj = window_frames[rel_idx]
        img = _render_single_frame(
            sanitized_play=sanitized,
            window_frame_idx=rel_idx,
            seq_position=seq_idx,
            seq_total=len(rel_indices),
            width=width,
            height=height,
        )
        images.append(img)
        sampled_frame_ids.append(int(f_obj["frame"]))
        sampled_times.append(float(f_obj["time"]))

        if output_dir is not None:
            out_path = (
                output_dir
                / f"game_{sanitized['game']}_play_{sanitized['play']}_seq_{seq_idx:02d}_frame_{f_obj['frame']}.png"
            )
            img.save(out_path, format="PNG")
            image_paths.append(out_path)

    return FrameSequenceBundle(
        game_id=sanitized["game"],
        play_id=sanitized["play"],
        images=images,
        image_paths=image_paths,
        sampled_frame_indices=abs_indices,
        sampled_frame_ids=sampled_frame_ids,
        sampled_times=sampled_times,
        window_start_frame=int(window_frames[0]["frame"]),
        window_end_frame=int(window_frames[-1]["frame"]),
        window_start_time=float(window_frames[0]["time"]),
        window_end_time=float(window_frames[-1]["time"]),
        candidate_rushers=sanitized["candidate_rushers"],
        sanitized_play=sanitized,
    )
