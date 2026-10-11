"""QB Pressure Replay v2. Requires pandas and numpy; no Jupyter required.

GUI:  python qb_pressure_replay_v2.py
CLI:  python qb_pressure_replay_v2.py --id 2021090900 [--data DATA_FOLDER] [--no-open]
"""
from pathlib import Path
import argparse
import html
import json
import os
import sys
import threading
import webbrowser

BASE = Path(__file__).resolve().parent
FPS = 10
PRESSURE_YD = 2.0   # our own definition: a rusher within 2 yards of the QB = "pressure zone"

OUTCOMES = {"C": "Complete", "I": "Incomplete", "S": "Sack", "IN": "Interception", "R": "Scramble"}
END_LABELS = {"pass_forward": "Pass released", "qb_sack": "Sack",
              "qb_strip_sack": "Strip sack", "run": "QB run"}


def esc(value):
    return html.escape(str(value))


def mean_or_none(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def is_data_folder(path):
    return all((path / name).is_file() for name in
               ["games.csv", "plays.csv", "players.csv", "pffScoutingData.csv"]) and (path / "tracking").is_dir()


def find_data_folder():
    for candidate in [BASE / "data", BASE, BASE.parent / "data", BASE.parent]:
        if is_data_folder(candidate):
            return candidate
    return None


def clean_id(text):
    return text.strip().replace("tracking_", "").removesuffix(".csv")


def pff_table_html(rows):
    head = ("<table><tr><th>Rusher</th><th>Hit</th><th>Hurry</th><th>Sack</th>"
            "<th>Closest to QB (yd)</th><th>First within %.0f yd (s)</th></tr>" % PRESSURE_YD)
    body = ""
    for r in rows:
        flagged = r["pff_hit"] or r["pff_hurry"] or r["pff_sack"]
        cells = ""
        for key in ["pff_hit", "pff_hurry", "pff_sack"]:
            cells += '<td class="yes">Yes</td>' if r[key] else "<td>No</td>"
        first = "-" if r["first_within"] is None else "%.1f" % r["first_within"]
        body += '<tr%s><td>%s</td>%s<td>%.1f</td><td>%s</td></tr>' % (
            ' class="hl"' if flagged else "", esc(r["name"]), cells, r["closest_yards"], first)
    return head + body + "</table>"


def prepare_play(play_id):
    raw = tracking.loc[tracking.playId.eq(play_id)].copy()
    roles = scouting.loc[
        scouting.gameId.eq(GAME_ID) & scouting.playId.eq(play_id)
    ].copy()
    qbs = roles.loc[roles.pff_role.eq("Pass"), "nflId"].unique()
    rushers = roles.loc[roles.pff_role.eq("Pass Rush"), "nflId"].unique()
    if len(qbs) != 1 or len(rushers) == 0:
        raise ValueError("This play requires one passer and at least one pass rusher.")
    qb_id = int(qbs[0])
    rusher_ids = [int(i) for i in rushers]
    snaps = raw.loc[raw.event.eq("ball_snap"), "frameId"]
    if snaps.empty:
        raise ValueError("No official ball_snap event. Choose another play.")
    start = int(snaps.min())
    endings = raw.loc[
        raw.frameId.ge(start) & raw.event.isin(
            ["pass_forward", "qb_sack", "qb_strip_sack", "run"]
        ), ["frameId", "event"]
    ].drop_duplicates().sort_values("frameId")
    if endings.empty:
        raise ValueError("No supported release/sack/run event. Choose another play.")
    end = int(endings.iloc[0].frameId)
    end_event = str(endings.iloc[0].event)
    end_label = END_LABELS.get(end_event, end_event)
    segment = raw.loc[raw.frameId.between(start, end)].copy()
    if segment.duplicated(["frameId", "nflId"]).any():
        raise ValueError("Duplicate entity-frame keys detected.")
    timestamps = pd.to_datetime(segment.time, utc=True)
    stamp_span = (timestamps.max() - timestamps.min()).total_seconds()
    duration = (end - start) / FPS
    if abs(stamp_span - duration) > 1.1:
        raise ValueError("Frame timing disagrees with the 10 FPS assumption.")

    # Priority 2: make every play go left -> right (offense always moves toward +x)
    flipped = False
    if "playDirection" in segment.columns:
        flipped = bool(segment.playDirection.astype(str).str.lower().eq("left").any())
    if flipped:
        segment["x"] = 120 - segment["x"]
        segment["y"] = 160 / 3 - segment["y"]
        if "dir" in segment.columns:
            segment["dir"] = (segment["dir"] + 180.0) % 360.0
        if "o" in segment.columns:
            segment["o"] = (segment["o"] + 180.0) % 360.0

    names = players.set_index("nflId").displayName.to_dict()
    role_map = roles.set_index("nflId").pff_role.to_dict()
    meta = game_plays.loc[game_plays.playId.eq(play_id)].iloc[0]
    qb_name = names.get(qb_id, "The QB")
    frame_ids = sorted(segment.frameId.unique())
    frames, distance_rows = [], []
    los, qb_y0 = None, None
    for frame_id in frame_ids:
        group = segment.loc[segment.frameId.eq(frame_id)]
        qb = group.loc[group.nflId.eq(qb_id)]
        if len(qb) != 1:
            raise ValueError("Missing or ambiguous quarterback frame.")
        q = qb.iloc[0]
        if los is None:
            ball = group.loc[group.team.eq("football")]
            los = float(ball.x.iloc[0]) if len(ball) else float(q.x)
            qb_y0 = float(q.y)
        rr = group.loc[group.nflId.isin(rushers)].copy()
        if len(rr) != len(rushers):
            raise ValueError("A pass rusher is missing from a frame.")
        rr["distance"] = np.hypot(rr.x - q.x, rr.y - q.y)
        nearest = int(rr.loc[rr.distance.idxmin(), "nflId"])
        time_s = (int(frame_id) - start) / FPS
        entities = []
        for row in group.itertuples():
            is_ball = row.team == "football"
            entity_id = None if is_ball else int(row.nflId)
            s_val = getattr(row, "s", None)
            a_val = getattr(row, "a", None)
            dir_val = getattr(row, "dir", None)
            o_val = getattr(row, "o", None)
            entities.append({
                "id": entity_id, "x": float(row.x), "y": float(row.y),
                "s": None if s_val is None or pd.isna(s_val) else round(float(s_val), 3),
                "a": None if a_val is None or pd.isna(a_val) else round(float(a_val), 3),
                "dir": None if dir_val is None or pd.isna(dir_val) else round(float(dir_val), 1),
                "o": None if o_val is None or pd.isna(o_val) else round(float(o_val), 1),
                "team": str(row.team),
                "name": "Football" if is_ball else names.get(entity_id, str(entity_id)),
                "number": "" if pd.isna(row.jerseyNumber) else str(int(row.jerseyNumber)),
                "role": "Football" if is_ball else role_map.get(entity_id, "Unknown")
            })
        distances = {str(int(r.nflId)): float(r.distance) for r in rr.itertuples()}
        frames.append({"frame": int(frame_id), "time": time_s,
                       "entities": entities, "distances": distances, "nearest": nearest})
        for key, value in distances.items():
            distance_rows.append({"frameId": int(frame_id), "seconds_after_snap": time_s,
                                  "nflId": int(key), "distance_yards": value})

    # PFF labels (whole play)
    pff = roles.loc[roles.nflId.isin(rushers),
                    ["nflId", "pff_hit", "pff_hurry", "pff_sack"]].copy()
    pff["nflId"] = pff.nflId.astype(int)
    for col in ["pff_hit", "pff_hurry", "pff_sack"]:
        pff[col] = pff[col].fillna(0)
    pff["name"] = pff.nflId.map(names)
    pff_idx = pff.set_index("nflId")
    if pff.pff_sack.sum() > 0:
        pff_label = "Sack"
    elif pff.pff_hit.sum() > 0:
        pff_label = "Hit"
    elif pff.pff_hurry.sum() > 0:
        pff_label = "Hurry"
    else:
        pff_label = "No PFF pressure"
    pff_pressure = bool(pff[["pff_hit", "pff_hurry", "pff_sack"]].sum().sum() > 0)

    # Priority 1: summary numbers
    dist_df = pd.DataFrame(distance_rows)
    closest = dist_df.loc[dist_df.distance_yards.idxmin()]
    near = dist_df.loc[dist_df.distance_yards <= PRESSURE_YD].sort_values("seconds_after_snap")
    first_pressure = None
    if len(near):
        fp = near.iloc[0]
        first_pressure = {"time": float(fp.seconds_after_snap), "id": int(fp.nflId),
                          "name": names.get(int(fp.nflId), str(int(fp.nflId)))}
    closest_name = names.get(int(closest.nflId), str(int(closest.nflId)))
    summary = {"closest_yards": float(closest.distance_yards), "closest_name": closest_name,
               "closest_id": int(closest.nflId), "closest_time": float(closest.seconds_after_snap),
               "first_pressure": first_pressure}

    if first_pressure:
        headline = ("%s had %.1f s before the %s. %s reached the pressure zone (%g yd) at %.1f s; "
                    "closest approach was %.1f yd by %s." % (
                        qb_name, duration, end_label.lower(), first_pressure["name"], PRESSURE_YD,
                        first_pressure["time"], summary["closest_yards"], closest_name))
    else:
        headline = ("%s had %.1f s before the %s. No rusher got within %g yd; the closest was %s at %.1f yd." % (
            qb_name, duration, end_label.lower(), PRESSURE_YD, closest_name, summary["closest_yards"]))

    rusher_stats = []
    for rid in rusher_ids:
        sub = dist_df.loc[dist_df.nflId.eq(rid)]
        within = sub.loc[sub.distance_yards <= PRESSURE_YD]
        hit = bool(pff_idx.at[rid, "pff_hit"] > 0) if rid in pff_idx.index else False
        hurry = bool(pff_idx.at[rid, "pff_hurry"] > 0) if rid in pff_idx.index else False
        sack = bool(pff_idx.at[rid, "pff_sack"] > 0) if rid in pff_idx.index else False
        rusher_stats.append({
            "id": rid, "name": names.get(rid, str(rid)),
            "closest_yards": float(sub.distance_yards.min()),
            "first_within": None if within.empty else float(within.seconds_after_snap.min()),
            "pff_hit": hit, "pff_hurry": hurry, "pff_sack": sack,
            "pff_pressure": hit or hurry or sack})

    quarter = int(meta["quarter"]) if "quarter" in meta.index and pd.notna(meta["quarter"]) else None
    down = int(meta["down"]) if "down" in meta.index and pd.notna(meta["down"]) else None
    yards_to_go = int(meta["yardsToGo"]) if "yardsToGo" in meta.index and pd.notna(meta["yardsToGo"]) else None
    formation = str(meta["offenseFormation"]) if "offenseFormation" in meta.index and pd.notna(meta["offenseFormation"]) else "UNKNOWN"
    defenders_in_box = int(meta["defendersInBox"]) if "defendersInBox" in meta.index and pd.notna(meta["defendersInBox"]) else None
    dropback_type = str(meta["dropBackType"]) if "dropBackType" in meta.index and pd.notna(meta["dropBackType"]) else "UNKNOWN"
    play_action = bool(meta["pff_playAction"]) if "pff_playAction" in meta.index and pd.notna(meta["pff_playAction"]) else False
    pff_coverage = str(meta["pff_passCoverage"]) if "pff_passCoverage" in meta.index and pd.notna(meta["pff_passCoverage"]) else "UNKNOWN"
    pff_coverage_type = str(meta["pff_passCoverageType"]) if "pff_passCoverageType" in meta.index and pd.notna(meta["pff_passCoverageType"]) else "UNKNOWN"
    pass_result = str(meta.passResult)
    outcome_label = OUTCOMES.get(pass_result, pass_result if pass_result != "nan" else "n/a")
    desc = str(meta.playDescription)
    label = "%s%s | %s | PFF: %s" % (
        "Q%d | " % int(quarter) if quarter is not None else "",
        desc if len(desc) <= 85 else desc[:82] + "...", outcome_label, pff_label)

    payload = {"game": GAME_ID, "play": int(play_id), "qb": qb_id,
               "quarter": quarter, "down": down, "yards_to_go": yards_to_go,
               "formation": formation, "defenders_in_box": defenders_in_box,
               "dropback_type": dropback_type, "play_action": play_action,
               "pff_pass_coverage": pff_coverage, "pff_pass_coverage_type": pff_coverage_type,
               "offense": str(meta.possessionTeam), "defense": str(meta.defensiveTeam),
               "description": desc, "outcome": pass_result, "outcome_label": outcome_label,
               "yards": float(meta.playResult), "end_event": end_event, "end_label": end_label,
               "duration": duration, "frames": frames, "label": label, "headline": headline,
               "summary": summary, "rusher_stats": rusher_stats, "pff_label": pff_label,
               "pff_pressure": pff_pressure, "flipped": flipped,
               "view": {"x0": los - 14, "x1": los + 30, "los": los, "qb_y": qb_y0},
               "rushers": [{"id": i, "name": names.get(i, str(i))} for i in rusher_ids],
               "pff_html": pff_table_html(rusher_stats)}
    from telemetry_comparison import build_play_telemetry_bundle
    payload["telemetry"] = build_play_telemetry_bundle(payload, pressure_threshold_yd=PRESSURE_YD)
    return payload, dist_df, pff


HTML_TEMPLATE = r"""<!doctype html><html lang="en" translate="no"><head><meta charset="utf-8"><meta name="google" content="notranslate"><title>QB Pressure Replay — NFL Player Comparison &amp; Performance Analytics</title><style>
body{font:15px Arial;background:#101827;color:#eef2ff;margin:20px}h2{margin:0 0 10px}h3{margin:24px 0 6px}
button{padding:8px 14px;margin:4px;border:0;border-radius:5px;cursor:pointer;background:#243c5c;color:#eef2ff;font-weight:bold}
button:hover{background:#31517a}select,input[type="text"],input[type="number"]{padding:4px 6px;margin-right:10px;background:#1b2a44;color:#eef2ff;border:1px solid #33435f;border-radius:4px}
input[type="range"]{width:95%}canvas{width:100%;background:#172338;border-radius:8px}
.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:14px}.note{color:#bdc8db;font-size:13px;line-height:1.45}
#status{margin:10px 0;font-weight:bold;background:#162238;padding:8px 12px;border-radius:6px;border-left:4px solid #00e5ff}
#headline{font-size:17px;font-weight:bold;margin:10px 0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}
.card{background:#1b2a44;border-radius:8px;padding:10px 12px}.card .k{color:#9fb0cc;font-size:12px;text-transform:uppercase}
.card .v{font-size:22px;font-weight:bold;margin:3px 0}.card .s{color:#bdc8db;font-size:13px}
table{border-collapse:collapse;margin:6px 0;width:100%}th,td{border:1px solid #33435f;padding:5px 10px;text-align:left}
th{background:#1b2a44}#gameTable th,#mpPlayerTable th{cursor:pointer}tr.hl td{background:#3a2a1a}.yes{color:#ff8a65;font-weight:bold}
.ai-panel,.cmp-panel{background:#162238;border:1px solid #2e4369;border-radius:8px;padding:14px;margin:8px 0}
.cmp-controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827;padding:10px 12px;border-radius:6px;margin-bottom:10px}
.cmp-pill-a{display:inline-block;padding:2px 8px;border-radius:4px;background:rgba(0,229,255,0.16);color:#00e5ff;border:1px solid #00e5ff;font-weight:bold;font-size:12px}
.cmp-pill-b{display:inline-block;padding:2px 8px;border-radius:4px;background:rgba(255,171,0,0.16);color:#ffab00;border:1px solid #ffab00;font-weight:bold;font-size:12px}
.gap-banner{background:#101827;border-left:4px solid #e040fb;padding:8px 12px;border-radius:6px;margin-bottom:10px;font-size:15px;font-weight:bold;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.gap-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0}
.gap-card{background:#101827;border:1px solid #2b3d5e;border-radius:8px;padding:10px 12px}
.gap-card.player-a{border-top:3px solid #00e5ff}.gap-card.player-b{border-top:3px solid #ffab00}.gap-card.delta{border-top:3px solid #e040fb}
.gap-metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-top:8px;font-size:13px}
.gap-metric{background:#172338;padding:6px 8px;border-radius:5px}.gap-metric .lbl{color:#9fb0cc;font-size:11px;text-transform:uppercase}
.gap-metric .val{font-size:16px;font-weight:bold;margin-top:2px}
.sector-row{cursor:pointer}.sector-row:hover td{background:#213454}.sector-active td{background:#1e3a5f}
.ai-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.ai-badge{display:inline-block;padding:4px 10px;border-radius:4px;font-size:12px;font-weight:bold;text-transform:uppercase}
.ai-badge.match{background:#1b5e20;color:#a5d6a7;border:1px solid #2e7d32}
.ai-badge.mismatch{background:#5f2120;color:#ffab91;border:1px solid #c62828}
.ai-badge.abstained{background:#311b92;color:#d1c4e9;border:1px solid #5e35b1}
.ai-badge.invalid_output{background:#4a148c;color:#f8bbd0;border:1px solid #880e4f}
.ai-badge.not_analyzed,.ai-badge.model_unavailable{background:#263238;color:#b0bec5;border:1px solid #455a64}
.ai-details{background:#101827;border-radius:6px;padding:10px 12px;margin-top:8px;font-size:14px;line-height:1.45}
.ai-details p{margin:5px 0}.ai-frame-btn{padding:4px 10px;margin:2px 4px 2px 0;background:#243c5c;color:#eef2ff;font-size:12px}
.playback-bar{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:6px 0}
.ws-nav{display:flex;flex-wrap:wrap;gap:6px;background:#162238;padding:8px 12px;border-radius:8px;border:1px solid #2e4369;margin:10px 0;position:sticky;top:6px;z-index:15}
.ws-tab{padding:6px 12px;margin:0;background:#1b2a44;color:#bdc8db;border:1px solid #33435f;border-radius:5px;font-size:13px}
.ws-tab.active{background:#0288d1;color:#fff;border-color:#00e5ff}
.ended-badge{display:inline-block;padding:2px 8px;border-radius:4px;background:#5f2120;color:#ffab91;border:1px solid #ef5350;font-weight:bold;font-size:11px;text-transform:uppercase}
.active-badge{display:inline-block;padding:2px 8px;border-radius:4px;background:#1b5e20;color:#a5d6a7;border:1px solid #4caf50;font-weight:bold;font-size:11px;text-transform:uppercase}
#tip{position:fixed;display:none;background:rgba(0,0,0,.88);padding:6px 10px;border-radius:4px;font-size:12px;pointer-events:none;z-index:30;border:1px solid #4a5d7e}
@media(max-width:900px){.grid,.cards,.gap-grid{grid-template-columns:1fr}}
</style></head><body>
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
 <h2>QB Pressure Replay &mdash; NFL Player Comparison &amp; Performance Analytics</h2>
 <span class="note">10 Hz Tracking Data &middot; Smooth 60 FPS Replay &middot; Player Tracking Metrics &amp; Play Phase Analysis &middot; VLM Verification</span>
</div>
<div class="ws-nav" id="workspaceNav">
 <span style="align-self:center;font-weight:bold;font-size:13px;color:#9fb0cc;margin-right:4px">Workspace View:</span>
 <button type="button" class="ws-tab active" data-ws="all">Show All Sections</button>
 <button type="button" class="ws-tab" data-ws="replay">1. Replay &amp; Distance Chart</button>
 <button type="button" class="ws-tab" data-ws="comparison">2. Player Motion &amp; Phase Analysis</button>
 <button type="button" class="ws-tab" data-ws="crossplay">3. Cross-Play Comparison</button>
 <button type="button" class="ws-tab" data-ws="multiplay">4. Multi-Play Analytics</button>
 <button type="button" class="ws-tab" data-ws="vlm">5. VLM Analysis &amp; Errors</button>
</div>

<div id="secReplayGroup" class="ws-section" data-sec="replay">
<label>Sort plays: <select id="sort"><option value="game">Game order</option><option value="time">Longest time to throw</option><option value="closest">Closest rusher</option><option value="arrival">Earliest pressure</option></select></label>
<label>Show: <select id="filter"><option value="all">All plays</option><option value="near">Rusher within pressure zone</option><option value="pff">PFF hit / hurry / sack</option><option value="vlm">Plays with VLM AI inference</option></select></label><br><br>
<label>Select play: <select id="playSelect" style="max-width:80%"></select></label>
<div id="desc" style="margin-top:10px"></div><div id="headline"></div><div class="cards" id="cards"></div><div id="status"></div>
<div class="playback-bar">
 <button id="play" type="button">Play</button><button id="pause" type="button">Pause</button><button id="reset" type="button">Reset</button>
 <button id="stepPrev" type="button" title="Step back 1 exact 10 Hz tracking frame">-1 Frame</button>
 <button id="stepNext" type="button" title="Step forward 1 exact 10 Hz tracking frame">+1 Frame</button>
 <label style="margin-left:8px">Speed: <select id="speedSelect"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1.0x (10 FPS data)</option><option value="2">2.0x</option></select></label>
 <label style="cursor:pointer"><input id="smoothToggle" type="checkbox" checked style="width:auto"> Smooth 60 FPS visual interpolation (measurements locked to 10 Hz frames)</label>
 <label style="margin-left:auto">Distance Chart Mode: <select id="distChartMode"><option value="progressive" selected>Mode 1: Progressive Reveal (Default)</option><option value="full">Mode 2: Full Trajectory</option></select></label>
</div>
<input id="slider" type="range" min="0" value="0" step="0.02">
<div class="grid"><canvas id="field" width="780" height="560"></canvas><canvas id="chart" width="650" height="440" style="cursor:ew-resize" title="Hover points for details; click or drag to scrub timeline"></canvas></div>
<p class="note">Field is zoomed to the action and flipped so the offense always moves left to right. Blue: offense. Red: defense (large dots with a colored ring: pass rushers, same color as on the chart). Yellow: quarterback. Brown: football. White ring: nearest pass rusher. Hover a dot for details. Pressure zone = within __TH__ yd of the QB (our own definition, not an official stat). Timing assumes 10 frames/second. Replay stops at release, sack, or run. On the right chart, <strong>Progressive Reveal</strong> plots trajectories only up to the current playback time (with dashed interpolated tips between 10 Hz samples) and displays the full curve when playback finishes; switch to <strong>Full Trajectory</strong> to inspect the whole play at any time.</p>
</div>

<div id="secComparisonGroup" class="ws-section" data-sec="comparison">
<h3 id="comparisonSection">Player Motion &amp; Performance Analysis</h3>
<div class="cmp-panel">
 <div class="cmp-controls">
  <label style="cursor:pointer;font-weight:bold"><input id="cmpEnable" type="checkbox" checked style="width:auto"> Enable Head-to-Head Comparison</label>
  <label><span class="cmp-pill-a">Player A</span> <select id="playerASelect"></select></label>
  <label><span class="cmp-pill-b">Player B</span> <select id="playerBSelect"></select></label>
  <button id="swapPlayers" type="button">Swap A &harr; B</button>
  <button id="presetClosestPair" type="button" title="Select the 2 closest pass rushers on this play">Closest 2 Rushers</button>
  <button id="presetVlmVsGt" type="button" style="display:none;background:#4a148c" title="Compare VLM predicted rusher vs Tracking closest rusher">Compare VLM Pick vs. Tracking Closest</button>
  <label style="cursor:pointer"><input id="cmpTrails" type="checkbox" checked style="width:auto"> Movement trails</label>
  <label style="cursor:pointer"><input id="cmpVectors" type="checkbox" checked style="width:auto"> QB distance vectors</label>
 </div>
 <div class="cmp-controls" style="margin-top:-4px;font-size:13px">
  <strong style="color:#9fb0cc">Player Tracking Metrics (toggle without losing playback position):</strong>
  <label style="cursor:pointer"><input id="telMetricDist" type="checkbox" checked style="width:auto"> 1. QB Distance (yd)</label>
  <label style="cursor:pointer"><input id="telMetricSpeed" type="checkbox" checked style="width:auto"> 2. Player Speed (yd/s)</label>
  <label style="cursor:pointer"><input id="telMetricAccel" type="checkbox" checked style="width:auto"> 3. Acceleration (yd/s&sup2;)</label>
  <label style="cursor:pointer"><input id="telMetricGap" type="checkbox" checked style="width:auto"> 4. A&ndash;B Distance Gap (&Delta;d, yd)</label>
  <label style="cursor:pointer"><input id="telMetricClosing" type="checkbox" checked style="width:auto"> 5. QB Closing Rate (-&Delta;d/&Delta;t, yd/s)</label>
 </div>
 <div id="liveGapPanel"></div>
 <canvas id="telemetryChart" width="1360" height="660" style="cursor:ew-resize;margin-top:8px" title="Click or drag anywhere on the tracking metrics timeline to seek"></canvas>
 <p class="note" style="margin:6px 0"><strong>Player Tracking Metrics &amp; Physical Quantity Distinctions:</strong> (1) <strong>QB Distance (d, yd)</strong> is instantaneous 2D Euclidean separation between player and QB; (2) <strong>Player Speed (s, yd/s)</strong> is absolute scalar ground speed along the player's movement vector; (3) <strong>Player Acceleration (a, yd/s&sup2;)</strong> is scalar tracking acceleration; (4) <strong>A&ndash;B Distance Gap (&Delta;d = d<sub>A</sub> - d<sub>B</sub>, yd)</strong> is relative proximity advantage (&lt;0 means A closer); (5) <strong>Relative Closing Rate (-&Delta;d/&Delta;t, yd/s)</strong> is the rate at which QB separation decreases (positive = closing on QB, negative = moving away). A high ground speed does not imply a high QB closing rate if the rusher is looping laterally around the arc. <strong>Football interpretation disclaimer:</strong> Geometric QB proximity advantage does not automatically indicate better overall football performance (e.g., edge contain discipline, occupying double-teams, or stunt setups).</p>
 <div id="sectorPanel" style="margin-top:12px"></div>
</div>
</div>

<div id="secCrossPlayGroup" class="ws-section" data-sec="crossplay">
<h3 id="crossPlaySection">Cross-Play Comparison &mdash; Multi-Play &amp; Cross-Game Pass-Rusher Analysis (Aligned at ball_snap t = 0.0s)</h3>
<div class="cmp-panel">
 <div class="cmp-controls">
  <span class="cmp-pill-a">Slot A</span>
  <label>Game A: <select id="cpGameA"></select></label>
  <label>Play A: <select id="cpPlayA" style="max-width:320px"></select></label>
  <label>Rusher A: <select id="cpPlayerA"></select></label>
  <button id="cpSyncCurrentPlay" type="button" title="Copy current main replay play &amp; Player A into Slot A">Use Current Replay Play</button>
 </div>
 <div class="cmp-controls">
  <span class="cmp-pill-b">Slot B</span>
  <label>Game B: <select id="cpGameB"></select></label>
  <label>Play B: <select id="cpPlayB" style="max-width:320px"></select></label>
  <label>Rusher B: <select id="cpPlayerB"></select></label>
  <button id="cpPresetSamePlayer" type="button" title="Compare the same pass rusher across two different plays">Preset: Same Player Across 2 Plays</button>
  <button id="cpPresetDiffPlayers" type="button" title="Compare two different pass rushers across two plays">Preset: Different Rushers Across Plays</button>
  <button id="cpPresetCrossGame" type="button" title="Compare rusher trajectories across two different NFL games">Preset: Cross-Game (DAL@TB vs PHI@ATL)</button>
 </div>
 <div class="cmp-controls" style="justify-content:space-between">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;width:100%">
   <strong>Shared Snap-Aligned Clock (t = 0.0s at ball_snap):</strong>
   <span id="cpClockLabel" style="color:#00e5ff;font-weight:bold">+0.00 s</span>
   <input id="cpTimeSlider" type="range" min="0" max="5" step="0.05" value="0" style="width:360px">
   <span id="cpStatusBadges"></span>
  </div>
 </div>
 <div id="crossSummaryCards" class="cards"></div>
 <div class="grid">
  <canvas id="crossFieldCanvas" width="680" height="440"></canvas>
  <canvas id="crossTelemetryCanvas" width="680" height="440" style="cursor:ew-resize" title="Click or drag to scrub shared post-snap timeline"></canvas>
 </div>
 <p class="note"><strong>Cross-Play Alignment &amp; Unequal Duration Rules:</strong> Both plays are aligned at <code>ball_snap</code> (<code>t = 0.0 s</code>) on the shared playback clock. Spatial paths on the left canvas are normalized relative to each play's Line of Scrimmage (&Delta;X from LOS) and QB snap lateral position (&Delta;Y from QB at snap) so rush arcs across different field positions or games can be directly compared. When one play ends earlier (e.g., quick pass release at +2.1s vs extended dropback at +4.2s), the shorter play's trajectory stops at its final observed tracking sample and displays an explicit <strong>PLAY ENDED</strong> indicator &mdash; trajectories are never extrapolated beyond observed tracking frames.</p>
</div>
</div>

<div id="secMultiPlayGroup" class="ws-section" data-sec="multiplay">
<h3 id="multiPlaySection">Multi-Play &amp; Player-Level Pass-Rush Analytics</h3>
<div class="cmp-panel">
 <div class="cmp-controls">
  <label>Search Player: <input id="mpSearch" type="text" placeholder="e.g. Barrett, Parsons, Jarrett..." style="width:170px"></label>
  <label>Game: <select id="mpGameFilter"><option value="all">All Loaded Games</option></select></label>
  <label>Def Team: <select id="mpTeamFilter"><option value="all">All Defenses</option></select></label>
  <label>Opponent: <select id="mpOffenseFilter"><option value="all">All Offenses</option></select></label>
  <label>Down: <select id="mpDownFilter"><option value="all">All Downs</option><option value="1">1st Down</option><option value="2">2nd Down</option><option value="3">3rd Down</option><option value="4">4th Down</option></select></label>
  <label>Formation: <select id="mpFormationFilter"><option value="all">All Formations</option></select></label>
  <label>Outcome: <select id="mpOutcomeFilter"><option value="all">All Outcomes</option><option value="C">Complete</option><option value="I">Incomplete</option><option value="S">Sack</option><option value="IN">Interception</option><option value="R">Scramble</option></select></label>
  <label>PFF Role: <select id="mpPffFilter"><option value="all">All Pass-Rush Snaps</option><option value="pressure">With PFF Hit/Hurry/Sack</option><option value="no_pressure">Without PFF Pressure</option><option value="within_2yd">Reached &le;2 yd Zone</option></select></label>
  <label>Min Plays: <select id="mpMinPlays"><option value="1" selected>Min 1 play</option><option value="3">Min 3 plays</option><option value="5">Min 5 plays</option><option value="10">Min 10 plays</option></select></label>
  <button id="mpResetFilters" type="button">Reset Filters</button>
 </div>
 <div id="mpDenominatorBanner" class="ai-details" style="margin-bottom:10px;border-left:4px solid #00e5ff"></div>
 <canvas id="analyticsDistCanvas" width="1360" height="360" style="margin:6px 0"></canvas>
 <p class="note"><strong>Distribution &amp; Denominator Note:</strong> Every metric reports its exact sample denominator (evaluated pass-rush snaps matching the active filters). Both <strong>mean</strong> and <strong>median [IQR]</strong> minimum QB distances are reported alongside time-to-closest-approach and QB closing-rate distributions. No arbitrary composite player score is constructed. Click any column header below to sort, or click any player row to inspect their individual play-by-play breakdown.</p>
 <div id="mpPlayerTable"></div>
 <div id="mpDrilldownPanel" style="margin-top:10px"></div>
</div>
</div>

<div id="secVlmGroup" class="ws-section" data-sec="vlm">
<h3>Pass rushers on this play</h3><div id="pffTable"></div>
<p class="note">PFF hit / hurry / sack labels describe the whole play, not individual frames.</p>
<h3>AI Visual Analysis (Optional VLM vs. Tracking Verification)</h3>
<div id="aiPanel" class="ai-panel"></div>
<p class="note">VLM predictions are generated solely from sequential overhead replay images (without numerical distances, nearest-rusher rings, or PFF labels) and verified against independent tracking-data Euclidean proximity over the observed frame window. Model explanations are descriptive interpretations, not verified ground truth.</p>
<h3>VLM Error Analysis &amp; Temporal Pass-Rush Evaluation Summary</h3>
<div id="vlmErrorPanel" class="ai-panel"></div>
<p class="note">Measurable error categories are computed objectively from tracking coordinates and parsed model outputs. Hypothesised causes represent unverified interpretive explanations of 3B VLM behavior.</p>
<h3>Whole game: pass-rusher ranking (click a header to sort)</h3><div id="gameTable"></div>
<p class="note">Small sample sizes: a rusher with only a few plays should not be over-interpreted.</p>
<h3>Does the distance measure mean something?</h3><div id="valid"></div>
<p class="note">Descriptive comparison only; it does not establish causation. Dataset: regional event data (check it matches the competition dataset).</p>
</div>
<div id="tip"></div>
<script>
const DATA=__DATA__;const allPlays=DATA.plays,TH=DATA.threshold;
const PAL=['#ff9800','#00e5ff','#e040fb','#76ff03','#ff4081','#b388ff','#1de9b6','#ffd180'];
const COL_A='#00e5ff',COL_B='#ffab00';
let d=null,i=0,playhead=0,rafId=null,lastRafTs=null,playSpeed=1.0,smoothMode=true,order=[],hits=[],chartHits=[],sectorBarHits=[];
let playerAId=null,playerBId=null,isScrubbing=false,sectorMetricKey='net_distance_change_yards';
const $=id=>document.getElementById(id);const slider=$('slider'),select=$('playSelect');
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function lastName(n){const p=String(n).split(' ');let l=p[p.length-1];if(/^(Jr\.?|Sr\.?|II|III|IV)$/.test(l)&&p.length>1)l=p[p.length-2];return l;}
function colorOf(id){const k=d.rushers.findIndex(r=>r.id===id);return PAL[(k<0?0:k)%PAL.length];}
function card(k,v,s){return `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`;}
function stopPlayback(){if(rafId!==null){cancelAnimationFrame(rafId);rafId=null;}lastRafTs=null;}
function syncPlayhead(pos){
 if(!d||!d.frames.length)return;
 const maxIdx=d.frames.length-1;
 const clamped=Math.max(0,Math.min(maxIdx,Number(pos)||0));
 playhead=smoothMode?clamped:Math.round(clamped);
 i=Math.max(0,Math.min(maxIdx,Math.round(playhead)));
 slider.value=String(playhead);
}
function tickPlayback(ts){
 if(!d){stopPlayback();return;}
 if(lastRafTs===null)lastRafTs=ts;
 const dtSec=Math.min(0.1,(ts-lastRafTs)/1000);
 lastRafTs=ts;
 const maxIdx=d.frames.length-1;
 const nextPos=playhead+dtSec*10.0*playSpeed;
 if(nextPos>=maxIdx){
  syncPlayhead(maxIdx);
  render();
  stopPlayback();
  return;
 }
 playhead=nextPos;
 i=Math.max(0,Math.min(maxIdx,Math.round(playhead)));
 slider.value=String(smoothMode?playhead:i);
 render();
 rafId=requestAnimationFrame(tickPlayback);
}
function startPlayback(){
 if(!d||rafId!==null)return;
 if(playhead>=d.frames.length-1)syncPlayhead(0);
 lastRafTs=null;
 rafId=requestAnimationFrame(tickPlayback);
}
function getVisualFrameState(){
 if(!d||!d.frames.length)return null;
 const maxIdx=d.frames.length-1;
 const pos=smoothMode?Math.max(0,Math.min(maxIdx,playhead)):i;
 const i0=Math.floor(pos),i1=Math.min(i0+1,maxIdx),alpha=pos-i0;
 const f0=d.frames[i0],f1=d.frames[i1],fMeas=d.frames[i];
 const frameGap=f1.frame-f0.frame,timeGap=f1.time-f0.time;
 const discont=(i0!==i1)&&(frameGap>1||timeGap>0.15||timeGap<=0);
 if(!smoothMode||alpha<=1e-5||i0===i1||discont){
  return {fMeas:fMeas,visEntities:f0.entities,visTime:f0.time,isInterp:false,discont:discont,pos:i0};
 }
 const map1=new Map();
 for(const e1 of f1.entities){map1.set(e1.role==='Football'?'ball':e1.id,e1);}
 const visEntities=f0.entities.map(e0=>{
  const e1=map1.get(e0.role==='Football'?'ball':e0.id);
  if(!e1)return e0;
  return Object.assign({},e0,{x:(1-alpha)*e0.x+alpha*e1.x,y:(1-alpha)*e0.y+alpha*e1.y});
 });
 return {fMeas:fMeas,visEntities:visEntities,visTime:(1-alpha)*f0.time+alpha*f1.time,isInterp:true,discont:false,pos:pos};
}
function buildOrder(){
 const f=$('filter').value,s=$('sort').value;
 let idx=allPlays.map((p,k)=>k).filter(k=>{const p=allPlays[k];if(f==='pff')return p.pff_pressure;if(f==='near')return p.summary.first_pressure!==null;if(f==='vlm')return p.vlm_analysis&&p.vlm_analysis.outcome_status!=='not_analyzed';return true;});
 const keys={time:k=>-allPlays[k].duration,closest:k=>allPlays[k].summary.closest_yards,arrival:k=>{const a=allPlays[k].summary.first_pressure;return a===null?1e9:a.time;}};
 if(keys[s])idx.sort((a,b)=>keys[s](a)-keys[s](b));
 order=idx;select.innerHTML='';
 order.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=allPlays[k].label;select.appendChild(o);});
 selectPlay();}
function populateComparisonSelectors(){
 const selA=$('playerASelect'),selB=$('playerBSelect'),vlmBtn=$('presetVlmVsGt');
 if(!selA||!selB||!d)return;
 const T=d.telemetry||{},tRushers=T.rushers||{};
 const sorted=[...d.rushers].sort((a,b)=>a.id-b.id);
 selA.innerHTML='';selB.innerHTML='';
 sorted.forEach((r,idx)=>{
  const tr=tRushers[String(r.id)]||{};
  const tag=tr.tag||`R${idx+1}`;
  const minD=tr.summary?tr.summary.closest_distance_yards:null;
  const lbl=`${tag}: ${r.name}${tr.number?' #'+tr.number:''}${minD!=null?' (min '+minD.toFixed(2)+' yd)':''}`;
  const oA=document.createElement('option');oA.value=String(r.id);oA.textContent=lbl;selA.appendChild(oA);
  const oB=document.createElement('option');oB.value=String(r.id);oB.textContent=lbl;selB.appendChild(oB);
 });
 const defA=T.default_player_a_id||(sorted[0]&&sorted[0].id);
 const defB=T.default_player_b_id||(sorted[1]?sorted[1].id:defA);
 playerAId=defA;playerBId=defB;
 if(playerAId!=null)selA.value=String(playerAId);
 if(playerBId!=null)selB.value=String(playerBId);
 if(vlmBtn){
  const A=d.vlm_analysis;
  if(A&&A.predicted_rusher_id&&A.ground_truth&&A.ground_truth.closest_rusher_id){
   vlmBtn.style.display='inline-block';
   vlmBtn.textContent=A.exact_match?'Compare VLM Match Pair':'Compare VLM Pick vs. Tracking Closest';
  }else{
   vlmBtn.style.display='none';
  }
 }
}
function compareVlmVsTracking(seekToClosest){
 if(!d||!d.vlm_analysis)return;
 const A=d.vlm_analysis,gt=A.ground_truth||{};
 const predId=A.predicted_rusher_id,gtId=gt.closest_rusher_id||d.summary.closest_id;
 if(!predId||!gtId)return;
 $('cmpEnable').checked=true;
 playerAId=Number(predId);
 playerBId=Number(gtId);
 if(playerAId===playerBId&&d.telemetry&&d.telemetry.default_player_b_id&&d.telemetry.default_player_b_id!==playerAId){
  playerBId=Number(d.telemetry.default_player_b_id);
 }
 $('playerASelect').value=String(playerAId);
 $('playerBSelect').value=String(playerBId);
 if(seekToClosest&&gt.closest_frame_id){
  stopPlayback();
  const idx=d.frames.findIndex(f=>f.frame===gt.closest_frame_id);
  if(idx>=0)syncPlayhead(idx);
 }
 render();
}
function renderLiveGapAndSectors(){
 const gapEl=$('liveGapPanel'),secEl=$('sectorPanel'),telCanvas=$('telemetryChart');
 if(!gapEl||!secEl||!telCanvas)return;
 if(!d||!d.telemetry||!$('cmpEnable').checked){
  gapEl.innerHTML='<div class="note">Head-to-head comparison is paused. Check "Enable Head-to-Head Comparison" above to view Player Proximity Comparison, synchronized Player Tracking Metrics, and Play Phase Analysis.</div>';
  secEl.innerHTML='';
  telCanvas.style.display='none';
  return;
 }
 telCanvas.style.display='block';
 const T=d.telemetry,rA=T.rushers[String(playerAId)],rB=T.rushers[String(playerBId)];
 if(!rA||!rB)return;
 const dA=rA.distance_yd[i],dB=rB.distance_yd[i];
 const chgA=rA.distance_change_from_snap_yd[i],chgB=rB.distance_change_from_snap_yd[i];
 const sA=rA.speed_yd_s[i],sB=rB.speed_yd_s[i];
 const aA=rA.acceleration_yd_s2[i],aB=rB.acceleration_yd_s2[i];
 const cA=rA.closing_rate_yd_s[i],cB=rB.closing_rate_yd_s[i];
 const minSoFarA=rA.min_distance_so_far_yd?rA.min_distance_so_far_yd[i]:dA;
 const minSoFarB=rB.min_distance_so_far_yd?rB.min_distance_so_far_yd[i]:dB;
 const tMinSoFarA=rA.time_of_min_so_far_s?rA.time_of_min_so_far_s[i]:d.frames[i].time;
 const tMinSoFarB=rB.time_of_min_so_far_s?rB.time_of_min_so_far_s[i]:d.frames[i].time;
 const sumA=rA.summary||{},sumB=rB.summary||{};
 const dDiff=(dA!=null&&dB!=null)?(dA-dB):null;
 const sDiff=(sA!=null&&sB!=null)?(sA-sB):null;
 const cDiff=(cA!=null&&cB!=null)?(cA-cB):null;
 const initGap=(rA.distance_yd[0]!=null&&rB.distance_yd[0]!=null)?(rA.distance_yd[0]-rB.distance_yd[0]):null;
 const gapShift=(dDiff!=null&&initGap!=null)?(dDiff-initGap):null;
 let leadChangesSoFar=0,prevAdv=null,framesACloser=0,framesBCloser=0;
 for(let k=0;k<=i;k++){
  const da=rA.distance_yd[k],db=rB.distance_yd[k];
  if(da==null||db==null)continue;
  const adv=Math.abs(da-db)<1e-4?'tied':(da<db?'A':'B');
  if(adv==='A')framesACloser++;else if(adv==='B')framesBCloser++;
  if(adv==='A'||adv==='B'){if(prevAdv!==null&&prevAdv!==adv)leadChangesSoFar++;prevAdv=adv;}
 }
 const closerTxt=dDiff===null?'n/a':(Math.abs(dDiff)<1e-4?'Tied distance to QB':(dDiff<0?`${rA.tag} ${lastName(rA.name)} closer by ${Math.abs(dDiff).toFixed(2)} yd`:`${rB.tag} ${lastName(rB.name)} closer by ${Math.abs(dDiff).toFixed(2)} yd`));
 const statusStmt=dDiff===null?'Distance comparison unavailable at this frame.':(Math.abs(dDiff)<1e-4?`Player A (${rA.tag} ${rA.name}) and Player B (${rB.tag} ${rB.name}) are currently equidistant from the quarterback (${dA.toFixed(2)} yd).`:(dDiff<0?`Player A (${rA.tag} ${rA.name}) is currently ${Math.abs(dDiff).toFixed(2)} yards closer to the quarterback than Player B (${rB.tag} ${rB.name}).`:`Player B (${rB.tag} ${rB.name}) is currently ${Math.abs(dDiff).toFixed(2)} yards closer to the quarterback than Player A (${rA.tag} ${rA.name}).`));
 const closerCol=dDiff===null?'#eef2ff':(dDiff<0?COL_A:(dDiff>0?COL_B:'#eef2ff'));
 const fMeas=d.frames[i];
 const overallNr=d.rushers.find(r=>r.id===fMeas.nearest);
 const fmt=(v,u,p=2)=>v==null?'n/a':`${v.toFixed(p)}${u}`;
 const fmtSigned=(v,u,p=2)=>v==null?'n/a':`${v>=0?'+':''}${v.toFixed(p)}${u}`;
 const zSoFarA=(sumA.first_within_2yd_time_s!=null&&sumA.first_within_2yd_time_s<=fMeas.time+1e-6)?`+${sumA.first_within_2yd_time_s.toFixed(1)}s`:'Not yet';
 const zSoFarB=(sumB.first_within_2yd_time_s!=null&&sumB.first_within_2yd_time_s<=fMeas.time+1e-6)?`+${sumB.first_within_2yd_time_s.toFixed(1)}s`:'Not yet';

 gapEl.innerHTML=`<div class="gap-banner" style="border-left-color:${closerCol}">`
  +`<span style="color:${closerCol}">${esc(statusStmt)}</span>`
  +`<span class="note">Lead changes so far: <strong>${leadChangesSoFar}</strong> | Time closer so far: A ${(framesACloser*0.1).toFixed(1)}s vs B ${(framesBCloser*0.1).toFixed(1)}s | Snap-to-now gap shift: <strong>${fmtSigned(gapShift,' yd')}</strong></span>`
  +`</div>`
  +`<div class="gap-grid">`
  +`<div class="gap-card player-a">`
  +`<div style="display:flex;justify-content:space-between;align-items:center"><span class="cmp-pill-a">Player A &middot; ${esc(rA.tag)}</span><span class="note">Measured Frame #${fMeas.frame} (+${fMeas.time.toFixed(1)}s)</span></div>`
  +`<div style="font-size:17px;font-weight:bold;margin-top:4px">${esc(rA.name)}${rA.number?' #'+esc(rA.number):''} ${fMeas.nearest===playerAId?'<span style="color:#fff;background:#1b5e20;padding:1px 6px;border-radius:4px;font-size:11px">NEAREST TO QB</span>':''}</div>`
  +`<div class="gap-metrics">`
  +`<div class="gap-metric"><div class="lbl">Current QB Dist (d)</div><div class="val" style="color:${COL_A}">${fmt(dA,' yd')}</div><div class="note">From snap (&Delta;d): ${fmtSigned(chgA,' yd')}</div></div>`
  +`<div class="gap-metric"><div class="lbl">Min Dist Achieved So Far</div><div class="val" style="color:${COL_A}">${fmt(minSoFarA,' yd')}</div><div class="note">at +${(tMinSoFarA||0).toFixed(1)}s (Play min: ${fmt(sumA.closest_distance_yards,' yd')})</div></div>`
  +`<div class="gap-metric"><div class="lbl">Speed (s) &amp; Accel (a)</div><div class="val">${fmt(sA,' yd/s')}</div><div class="note">Accel: ${fmt(aA,' yd/s²')}</div></div>`
  +`<div class="gap-metric"><div class="lbl">Closing Rate / 2-yd Zone</div><div class="val">${fmtSigned(cA,' yd/s')}</div><div class="note">&le;${TH}yd so far: ${zSoFarA} (Play: ${sumA.first_within_2yd_time_s!=null?'+'+sumA.first_within_2yd_time_s.toFixed(1)+'s':'None'})</div></div>`
  +`</div></div>`
  +`<div class="gap-card player-b">`
  +`<div style="display:flex;justify-content:space-between;align-items:center"><span class="cmp-pill-b">Player B &middot; ${esc(rB.tag)}</span><span class="note">Measured Frame #${fMeas.frame} (+${fMeas.time.toFixed(1)}s)</span></div>`
  +`<div style="font-size:17px;font-weight:bold;margin-top:4px">${esc(rB.name)}${rB.number?' #'+esc(rB.number):''} ${fMeas.nearest===playerBId?'<span style="color:#fff;background:#1b5e20;padding:1px 6px;border-radius:4px;font-size:11px">NEAREST TO QB</span>':''}</div>`
  +`<div class="gap-metrics">`
  +`<div class="gap-metric"><div class="lbl">Current QB Dist (d)</div><div class="val" style="color:${COL_B}">${fmt(dB,' yd')}</div><div class="note">From snap (&Delta;d): ${fmtSigned(chgB,' yd')}</div></div>`
  +`<div class="gap-metric"><div class="lbl">Min Dist Achieved So Far</div><div class="val" style="color:${COL_B}">${fmt(minSoFarB,' yd')}</div><div class="note">at +${(tMinSoFarB||0).toFixed(1)}s (Play min: ${fmt(sumB.closest_distance_yards,' yd')})</div></div>`
  +`<div class="gap-metric"><div class="lbl">Speed (s) &amp; Accel (a)</div><div class="val">${fmt(sB,' yd/s')}</div><div class="note">Accel: ${fmt(aB,' yd/s²')}</div></div>`
  +`<div class="gap-metric"><div class="lbl">Closing Rate / 2-yd Zone</div><div class="val">${fmtSigned(cB,' yd/s')}</div><div class="note">&le;${TH}yd so far: ${zSoFarB} (Play: ${sumB.first_within_2yd_time_s!=null?'+'+sumB.first_within_2yd_time_s.toFixed(1)+'s':'None'})</div></div>`
  +`</div></div>`
  +`<div class="gap-card delta">`
  +`<div style="display:flex;justify-content:space-between;align-items:center"><span style="color:#e040fb;font-weight:bold;font-size:12px;text-transform:uppercase">Player Proximity Comparison (A vs B)</span><span class="note">Play Nearest: ${esc(overallNr?overallNr.name:'-')}</span></div>`
  +`<div style="font-size:16px;font-weight:bold;margin-top:4px;color:${closerCol}">${esc(closerTxt)}</div>`
  +`<div class="gap-metrics">`
  +`<div class="gap-metric"><div class="lbl">QB Distance Gap (d<sub>A</sub> - d<sub>B</sub>)</div><div class="val" style="color:${closerCol}">${fmtSigned(dDiff,' yd')}</div><div class="note">Shift from snap: ${fmtSigned(gapShift,' yd')}</div></div>`
  +`<div class="gap-metric"><div class="lbl">Speed Diff (s<sub>A</sub> - s<sub>B</sub>)</div><div class="val">${fmtSigned(sDiff,' yd/s')}</div><div class="note">Scalar ground speed delta</div></div>`
  +`<div class="gap-metric"><div class="lbl">Closing Rate Diff (A - B)</div><div class="val">${fmtSigned(cDiff,' yd/s')}</div><div class="note">Relative QB approach delta</div></div>`
  +`<div class="gap-metric"><div class="lbl">Advantage Shifts (So Far)</div><div class="val">${leadChangesSoFar} lead change${leadChangesSoFar===1?'':'s'}</div><div class="note">Min so far gap: ${fmtSigned((minSoFarA||0)-(minSoFarB||0),' yd')}</div></div>`
  +`</div></div>`
  +`</div>`;

 // Play Phase Analysis & Interactive Segment Comparison (S1 / S2 / S3)
 const segs=T.segments||[];
 let maxNetDiff=-1,maxNetSeg='',maxSpdDiff=-1,maxSpdSeg='';
 segs.forEach(seg=>{
  const pA=(seg.per_player&&seg.per_player[String(playerAId)])||{};
  const pB=(seg.per_player&&seg.per_player[String(playerBId)])||{};
  const nd=Math.abs((pA.net_distance_change_yards||0)-(pB.net_distance_change_yards||0));
  const sd=Math.abs((pA.mean_speed_yd_s||0)-(pB.mean_speed_yd_s||0));
  if(nd>maxNetDiff){maxNetDiff=nd;maxNetSeg=seg.segment_name;}
  if(sd>maxSpdDiff){maxSpdDiff=sd;maxSpdSeg=seg.segment_name;}
 });
 const segRows=segs.map(seg=>{
  const pA=(seg.per_player&&seg.per_player[String(playerAId)])||{};
  const pB=(seg.per_player&&seg.per_player[String(playerBId)])||{};
  const active=i>=seg.start_frame_index&&i<=seg.end_frame_index;
  const fmtZone=p=>{
   if(!p.entered_2yd_zone)return 'None (0 fr)';
   const ent=(p.zone_entry_times_s&&p.zone_entry_times_s.length)?p.zone_entry_times_s.map(t=>'+'+t.toFixed(1)+'s').join(','):('+'+(p.first_within_2yd_time_s||0).toFixed(1)+'s');
   const ex=(p.zone_exit_times_s&&p.zone_exit_times_s.length)?' / Exit '+p.zone_exit_times_s.map(t=>'+'+t.toFixed(1)+'s').join(','):'';
   return `In ${ent}${ex} (${p.frames_within_2yd_zone||1} fr)`;
  };
  const isLargest=seg.segment_name===maxNetSeg;
  return `<tr class="sector-row${active?' sector-active':''}" data-sidx="${seg.start_frame_index}" title="Click to jump replay to ${esc(seg.segment_name)} (+${seg.start_time_s.toFixed(1)}s)">`
   +`<td><strong>${esc(seg.segment_name)}</strong> ${isLargest?'<span style="color:#ffd54f;font-size:11px">[Largest &Delta;d Divergence]</span>':''}<br><span class="note">+${seg.start_time_s.toFixed(1)}s &ndash; +${seg.end_time_s.toFixed(1)}s (Frames #${seg.start_frame_id}&ndash;#${seg.end_frame_id})</span></td>`
   +`<td><span style="color:${COL_A}">A:</span> ${fmt(pA.initial_distance_yards,'')}&rarr;${fmt(pA.final_distance_yards,' yd')} (${fmtSigned(pA.net_distance_change_yards,' yd')})<br><span style="color:${COL_B}">B:</span> ${fmt(pB.initial_distance_yards,'')}&rarr;${fmt(pB.final_distance_yards,' yd')} (${fmtSigned(pB.net_distance_change_yards,' yd')})</td>`
   +`<td><span style="color:${COL_A}">A:</span> max ${fmt(pA.max_speed_yd_s,'')} / mean ${fmt(pA.mean_speed_yd_s,' yd/s')}<br><span style="color:${COL_B}">B:</span> max ${fmt(pB.max_speed_yd_s,'')} / mean ${fmt(pB.mean_speed_yd_s,' yd/s')}</td>`
   +`<td><span style="color:${COL_A}">A:</span> max ${fmt(pA.max_acceleration_yd_s2,'')} / mean ${fmt(pA.mean_acceleration_yd_s2,' yd/s²')}<br><span style="color:${COL_B}">B:</span> max ${fmt(pB.max_acceleration_yd_s2,'')} / mean ${fmt(pB.mean_acceleration_yd_s2,' yd/s²')}</td>`
   +`<td><span style="color:${COL_A}">A:</span> ${fmtSigned(pA.mean_closing_rate_yd_s,' yd/s')}<br><span style="color:${COL_B}">B:</span> ${fmtSigned(pB.mean_closing_rate_yd_s,' yd/s')}</td>`
   +`<td><span style="color:${COL_A}">A:</span> ${fmt(pA.closest_distance_yards,' yd')} (+${(pA.closest_time_s||0).toFixed(1)}s)<br><span style="color:${COL_B}">B:</span> ${fmt(pB.closest_distance_yards,' yd')} (+${(pB.closest_time_s||0).toFixed(1)}s)</td>`
   +`<td><span style="color:${COL_A}">A:</span> ${fmtZone(pA)}<br><span style="color:${COL_B}">B:</span> ${fmtZone(pB)}</td>`
   +`</tr>`;
 }).join('');
 secEl.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">`
  +`<h4 style="margin:4px 0">Play Phase Analysis (Timeline Segment Comparison &mdash; Click any phase row or bar to seek replay)</h4>`
  +`<label class="note">Play Phase Metric: <select id="sectorMetricSel">`
  +`<option value="net_distance_change_yards"${sectorMetricKey==='net_distance_change_yards'?' selected':''}>Net QB-Distance Change &Delta;d (yd)</option>`
  +`<option value="mean_closing_rate_yd_s"${sectorMetricKey==='mean_closing_rate_yd_s'?' selected':''}>Mean QB Closing Rate -&Delta;d/&Delta;t (yd/s)</option>`
  +`<option value="mean_speed_yd_s"${sectorMetricKey==='mean_speed_yd_s'?' selected':''}>Mean Player Speed s (yd/s)</option>`
  +`<option value="max_acceleration_yd_s2"${sectorMetricKey==='max_acceleration_yd_s2'?' selected':''}>Max Acceleration a (yd/s&sup2;)</option>`
  +`<option value="closest_distance_yards"${sectorMetricKey==='closest_distance_yards'?' selected':''}>Phase Minimum QB Distance (yd)</option>`
  +`</select></label></div>`
  +`<div class="note" style="margin-bottom:6px"><strong>Largest Measured Play Phase Differences:</strong> Largest net QB-distance divergence developed in <strong>${esc(maxNetSeg)}</strong> (${maxNetDiff.toFixed(2)} yd difference); largest mean speed divergence occurred in <strong>${esc(maxSpdSeg)}</strong> (${maxSpdDiff.toFixed(2)} yd/s difference).</div>`
  +`<canvas id="sectorChart" width="1360" height="185" style="cursor:pointer;margin:4px 0" title="Click any phase bar to jump replay to that play segment"></canvas>`
  +`<table><tr><th>Play Phase (Time &amp; Frame Window)</th><th>Init &rarr; Final QB Dist (&Delta;d)</th><th>Max / Mean Speed (s)</th><th>Max / Mean Accel (a)</th><th>Mean Closing Rate (-&Delta;d/&Delta;t)</th><th>Phase Closest Approach</th><th>2-Yd Zone Entry / Exit Events</th></tr>${segRows}</table>`;
 const smSel=$('sectorMetricSel');
 if(smSel)smSel.onchange=()=>{sectorMetricKey=smSel.value;renderLiveGapAndSectors();};
 secEl.querySelectorAll('.sector-row').forEach(tr=>tr.onclick=()=>{
  const sidx=Number(tr.dataset.sidx);
  stopPlayback();syncPlayhead(sidx);render();
 });
 renderSectorChart(segs,rA,rB);
}
function renderSectorChart(segs,rA,rB){
 const c=$('sectorChart');if(!c||!segs||!segs.length)return;
 const ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);
 sectorBarHits=[];
 const padL=75,padR=30,topY=30,botY=H-34,plotW=W-padL-padR,plotH=botY-topY;
 const metricLabels={
  net_distance_change_yards:'Net QB-Distance Change Δd (yd, negative = closed on QB)',
  mean_closing_rate_yd_s:'Mean QB Closing Rate -Δd/Δt (yd/s, positive = closing on QB)',
  mean_speed_yd_s:'Mean Scalar Speed s (yd/s)',
  max_acceleration_yd_s2:'Peak Scalar Acceleration a (yd/s²)',
  closest_distance_yards:'Phase Minimum QB Distance (yd)'
 };
 const vals=[];
 segs.forEach(s=>{
  const pA=(s.per_player&&s.per_player[String(playerAId)])||{};
  const pB=(s.per_player&&s.per_player[String(playerBId)])||{};
  if(pA[sectorMetricKey]!=null)vals.push(pA[sectorMetricKey]);
  if(pB[sectorMetricKey]!=null)vals.push(pB[sectorMetricKey]);
 });
 const signed=sectorMetricKey==='net_distance_change_yards'||sectorMetricKey==='mean_closing_rate_yd_s';
 const maxAbs=Math.max(1.5,...vals.map(v=>Math.abs(v)))*1.18;
 const yMap=v=>signed?((topY+botY)/2-(v/maxAbs)*(plotH/2)):(botY-(v/maxAbs)*plotH);
 const yZero=yMap(0);
 ctx.strokeStyle='#4a5d7e';ctx.lineWidth=1;
 ctx.beginPath();ctx.moveTo(padL,topY);ctx.lineTo(padL,botY);ctx.lineTo(padL+plotW,botY);ctx.stroke();
 if(signed){
  ctx.strokeStyle='rgba(255,255,255,0.3)';ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(padL,yZero);ctx.lineTo(padL+plotW,yZero);ctx.stroke();ctx.setLineDash([]);
 }
 ctx.fillStyle='#eef2ff';ctx.font='bold 12px Arial';
 ctx.fillText(metricLabels[sectorMetricKey]||sectorMetricKey,padL+8,17);
 const colW=plotW/segs.length;
 segs.forEach((seg,idx)=>{
  const pA=(seg.per_player&&seg.per_player[String(playerAId)])||{};
  const pB=(seg.per_player&&seg.per_player[String(playerBId)])||{};
  const vA=pA[sectorMetricKey]||0,vB=pB[sectorMetricKey]||0;
  const cx=padL+idx*colW+colW/2,bw=Math.min(72,colW*0.26);
  const active=i>=seg.start_frame_index&&i<=seg.end_frame_index;
  if(active){
   ctx.fillStyle='rgba(0,229,255,0.08)';
   ctx.fillRect(padL+idx*colW+6,topY,colW-12,plotH);
  }
  [[vA,COL_A,cx-bw-4,rA.tag],[vB,COL_B,cx+4,rB.tag]].forEach(([v,col,bx,tag])=>{
   const vy=yMap(v),yTop=Math.min(yZero,vy),barH=Math.max(2,Math.abs(vy-yZero));
   ctx.fillStyle=col;ctx.fillRect(bx,yTop,bw,barH);
   ctx.fillStyle='#eef2ff';ctx.font='bold 11px Arial';ctx.textAlign='center';
   ctx.fillText(`${tag}: ${v>=0&&signed?'+':''}${v.toFixed(2)}`,bx+bw/2,v>=0?yTop-5:yTop+barH+12);
  });
  ctx.fillStyle=active?'#00e5ff':'#9fb0cc';ctx.font='bold 12px Arial';ctx.textAlign='center';
  ctx.fillText(`${seg.segment_name} (+${seg.start_time_s.toFixed(1)}s–+${seg.end_time_s.toFixed(1)}s)`,cx,H-8);
  sectorBarHits.push({x0:padL+idx*colW,x1:padL+(idx+1)*colW,sidx:seg.start_frame_index});
 });
 ctx.textAlign='left';
 c.onclick=ev=>{
  const rect=c.getBoundingClientRect(),mx=(ev.clientX-rect.left)*(c.width/rect.width);
  const hit=sectorBarHits.find(h=>mx>=h.x0&&mx<=h.x1);
  if(hit){stopPlayback();syncPlayhead(hit.sidx);render();}
 };
}
function renderTelemetryChart(visTime){
 const c=$('telemetryChart');if(!c||!d||!d.telemetry||!$('cmpEnable').checked)return;
 const ctx=c.getContext('2d'),W=c.width;
 const T=d.telemetry,rA=T.rushers[String(playerAId)],rB=T.rushers[String(playerBId)];
 if(!rA||!rB)return;
 const diffs=d.frames.map((_,k)=>(rA.distance_yd[k]!=null&&rB.distance_yd[k]!=null)?(rA.distance_yd[k]-rB.distance_yd[k]):null);
 const maxDist=Math.max(5,...rA.distance_yd.filter(v=>v!=null),...rB.distance_yd.filter(v=>v!=null))*1.08;
 const maxSpd=Math.max(6,...rA.speed_yd_s.filter(v=>v!=null),...rB.speed_yd_s.filter(v=>v!=null))*1.1;
 const maxAcc=Math.max(5,...rA.acceleration_yd_s2.filter(v=>v!=null),...rB.acceleration_yd_s2.filter(v=>v!=null))*1.1;
 const maxAbsDiff=Math.max(2.0,...diffs.filter(v=>v!=null).map(v=>Math.abs(v)))*1.12;
 const maxAbsClose=Math.max(3.0,...rA.closing_rate_yd_s.filter(v=>v!=null).map(v=>Math.abs(v)),...rB.closing_rate_yd_s.filter(v=>v!=null).map(v=>Math.abs(v)))*1.12;

 const allDefs=[
  {id:'dist',on:!$('telMetricDist')||$('telMetricDist').checked,title:'1. QB Distance (yd)',signed:false,maxV:maxDist,sA:rA.distance_yd,sB:rB.distance_yd,hasTh:true},
  {id:'speed',on:!$('telMetricSpeed')||$('telMetricSpeed').checked,title:'2. Player Speed s (yd/s)',signed:false,maxV:maxSpd,sA:rA.speed_yd_s,sB:rB.speed_yd_s},
  {id:'accel',on:!$('telMetricAccel')||$('telMetricAccel').checked,title:'3. Player Acceleration a (yd/s²)',signed:false,maxV:maxAcc,sA:rA.acceleration_yd_s2,sB:rB.acceleration_yd_s2},
  {id:'gap',on:!$('telMetricGap')||$('telMetricGap').checked,title:'4. Relative QB-Distance Diff Δd = d_A - d_B (yd)',signed:true,maxV:maxAbsDiff,diffOnly:diffs},
  {id:'closing',on:!$('telMetricClosing')||$('telMetricClosing').checked,title:'5. QB-Relative Closing Rate -Δd/Δt (yd/s, >0 closing on QB)',signed:true,maxV:maxAbsClose,sA:rA.closing_rate_yd_s,sB:rB.closing_rate_yd_s}
 ];
 const activeDefs=allDefs.filter(t=>t.on);
 if(!activeDefs.length){ctx.clearRect(0,0,W,c.height);ctx.fillStyle='#9fb0cc';ctx.font='14px Arial';ctx.fillText('Select at least one Player Tracking Metric checkbox above.',30,40);return;}
 const trackH=116,gapH=20,topPad=26,botPad=32;
 const desiredH=topPad+activeDefs.length*trackH+(activeDefs.length-1)*gapH+botPad;
 if(c.height!==desiredH)c.height=desiredH;
 const H=c.height;ctx.clearRect(0,0,W,H);
 const padL=82,padR=28,plotW=W-padL-padR,dur=Math.max(d.duration,0.1);
 const tx=t=>padL+(t/dur)*plotW;
 const tracks=activeDefs.map((def,idx)=>{
  const top=topPad+idx*(trackH+gapH),plotTop=top+20,bot=top+trackH;
  const yFn=def.signed?(v=>((plotTop+bot)/2)-(v/def.maxV)*((bot-plotTop)/2)):(v=>bot-(v/def.maxV)*(bot-plotTop));
  return Object.assign({},def,{top,plotTop,bot,yFn});
 });

 // Sector background bands & vertical boundaries
 const segs=T.segments||[];
 segs.forEach((seg,idx)=>{
  const xStart=tx(seg.start_time_s),xEnd=tx(seg.end_time_s);
  if(idx%2===1){
   ctx.fillStyle='rgba(255,255,255,0.025)';
   tracks.forEach(tr=>ctx.fillRect(xStart,tr.top,xEnd-xStart,tr.bot-tr.top));
  }
  ctx.fillStyle='#9fb0cc';ctx.font='bold 11px Arial';ctx.textAlign='center';
  ctx.fillText(seg.segment_name,(xStart+xEnd)/2,15);
  if(idx>0){
   ctx.strokeStyle='rgba(159,176,204,0.28)';ctx.lineWidth=1;ctx.setLineDash([3,3]);
   tracks.forEach(tr=>{ctx.beginPath();ctx.moveTo(xStart,tr.top);ctx.lineTo(xStart,tr.bot);ctx.stroke();});
   ctx.setLineDash([]);
  }
 });
 ctx.textAlign='left';

 function plotSeries(vals,discArr,yMap,color,lineW){
  ctx.strokeStyle=color;ctx.lineWidth=lineW;ctx.beginPath();
  let penDown=false;
  for(let k=0;k<d.frames.length;k++){
   const v=vals[k];
   if(v==null||discArr[k]){penDown=false;}
   if(v==null)continue;
   const x=tx(d.frames[k].time),y=yMap(v);
   if(!penDown){ctx.moveTo(x,y);penDown=true;}else{ctx.lineTo(x,y);}
  }
  ctx.stroke();
 }

 const fTime=d.frames[i].time,measX=tx(fTime);
 tracks.forEach((tr,idx)=>{
  const {top,bot,yFn}=tr;
  if(tr.hasTh){
   ctx.fillStyle='rgba(255,82,82,0.12)';
   ctx.fillRect(padL,yFn(TH),plotW,bot-yFn(TH));
   ctx.strokeStyle='#ff5252';ctx.lineWidth=1;ctx.setLineDash([5,4]);
   ctx.beginPath();ctx.moveTo(padL,yFn(TH));ctx.lineTo(padL+plotW,yFn(TH));ctx.stroke();ctx.setLineDash([]);
  }
  if(tr.signed){
   const yZero=yFn(0);
   ctx.strokeStyle='rgba(255,255,255,0.35)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
   ctx.beginPath();ctx.moveTo(padL,yZero);ctx.lineTo(padL+plotW,yZero);ctx.stroke();ctx.setLineDash([]);
  }
  if(tr.diffOnly){
   const yZero=yFn(0);
   for(let k=0;k<d.frames.length-1;k++){
    if(diffs[k]==null||diffs[k+1]==null||rA.discontinuity[k+1]||rB.discontinuity[k+1])continue;
    const x0=tx(d.frames[k].time),x1=tx(d.frames[k+1].time),vMid=(diffs[k]+diffs[k+1])/2;
    ctx.fillStyle=vMid<0?'rgba(0,229,255,0.18)':'rgba(255,171,0,0.18)';
    ctx.beginPath();ctx.moveTo(x0,yZero);ctx.lineTo(x0,yFn(diffs[k]));ctx.lineTo(x1,yFn(diffs[k+1]));ctx.lineTo(x1,yZero);ctx.closePath();ctx.fill();
   }
   plotSeries(diffs,d.frames.map((_,k)=>rA.discontinuity[k]||rB.discontinuity[k]),yFn,'#e040fb',2.4);
   if(diffs[i]!=null){ctx.fillStyle='#e040fb';ctx.beginPath();ctx.arc(measX,yFn(diffs[i]),4.5,0,7);ctx.fill();}
  }else{
   plotSeries(tr.sA,rA.discontinuity,yFn,COL_A,2.4);
   plotSeries(tr.sB,rB.discontinuity,yFn,COL_B,2.4);
   if(tr.sA[i]!=null){ctx.fillStyle=COL_A;ctx.beginPath();ctx.arc(measX,yFn(tr.sA[i]),4.5,0,7);ctx.fill();}
   if(tr.sB[i]!=null){ctx.fillStyle=COL_B;ctx.beginPath();ctx.arc(measX,yFn(tr.sB[i]),4.5,0,7);ctx.fill();}
  }
  ctx.strokeStyle='#4a5d7e';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(padL,top);ctx.lineTo(padL,bot);ctx.lineTo(padL+plotW,bot);ctx.stroke();
  ctx.fillStyle='#eef2ff';ctx.font='bold 12px Arial';ctx.fillText(tr.title,padL+6,top+13);
  ctx.font='11px Arial';ctx.fillStyle='#9fb0cc';ctx.textAlign='right';
  const ticks=tr.signed?[-tr.maxV,0,tr.maxV]:[0,tr.maxV/2,tr.maxV];
  ticks.forEach(tv=>{ctx.fillText((tv>0&&tr.signed?'+':'')+tv.toFixed(1),padL-6,yFn(tv)+4);});
  ctx.textAlign='left';
  if(idx===0){
   ctx.fillStyle='#00e5ff';ctx.font='11px Arial';ctx.fillText(`■ Player A: ${rA.tag} ${rA.name}`,padL+330,top+13);
   ctx.fillStyle='#ffab00';ctx.fillText(`■ Player B: ${rB.tag} ${rB.name}`,padL+610,top+13);
   if(tr.hasTh){ctx.fillStyle='#ff5252';ctx.fillText(`-- ${TH} yd proximity threshold`,padL+890,top+13);}
  }
  if(tr.id==='gap'){
   ctx.fillStyle='#9fb0cc';ctx.font='11px Arial';ctx.fillText(`Δd < 0: Player A closer to QB (Cyan) | Δd > 0: Player B closer to QB (Amber)`,padL+420,top+13);
  }
 });

 // Shared X-axis time ticks at bottom
 ctx.fillStyle='#eef2ff';ctx.font='12px Arial';ctx.textAlign='center';
 const nTicks=Math.min(10,Math.max(4,Math.floor(dur*2)));
 for(let k=0;k<=nTicks;k++){
  const tt=(dur*k)/nTicks,xx=tx(tt);
  ctx.fillText(`+${tt.toFixed(1)}s`,xx,H-18);
 }
 ctx.fillText('Seconds after ball_snap (shared timeline — click or drag to scrub)',padL+plotW/2,H-3);
 ctx.textAlign='left';

 // Synchronized vertical time cursor across all visible tracks
 const curX=tx(visTime);
 tracks.forEach(tr=>{
  ctx.strokeStyle='#ffffff';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
  ctx.beginPath();ctx.moveTo(curX,tr.top);ctx.lineTo(curX,tr.bot);ctx.stroke();ctx.setLineDash([]);
 });
}
function renderAIPanel(){
 const panel=$('aiPanel');if(!panel)return;
 if(!d||!d.vlm_analysis){panel.innerHTML='<div class="note">AI Visual Analysis not available for this play.</div>';return;}
 const A=d.vlm_analysis,T=d.vlm_temporal_analysis,gt=A.ground_truth||{},st=A.outcome_status||'not_analyzed';
 const statusLabels={match:'Match (Verified by Tracking)',mismatch:'Mismatch vs. Tracking',abstained:'Abstained (Uncertain)',invalid_output:'Invalid Model Output',model_unavailable:'Model Unavailable (Fallback Active)',not_analyzed:'Not Analyzed (Optional AI)'};
 const statusText=statusLabels[st]||st;
 const vlmRusher=A.predicted_rusher_name?`${A.predicted_rusher_tag?A.predicted_rusher_tag+' ':''}${A.predicted_rusher_name}`:(st==='abstained'?'Abstained':st==='not_analyzed'?'Not run':'None');
 const vlmSub=A.predicted_rusher_distance_yards!=null?`Window min distance: ${A.predicted_rusher_distance_yards.toFixed(2)} yd`:(st==='abstained'?'Model declined to guess':'No rusher selected');
 const gtRusher=gt.closest_rusher_name?`${gt.closest_rusher_tag?gt.closest_rusher_tag+' ':''}${gt.closest_rusher_name}`:d.summary.closest_name;
 const gtDist=gt.closest_distance_yards!=null?`${gt.closest_distance_yards.toFixed(2)} yd at +${(gt.closest_time_s||0).toFixed(1)}s`:`${d.summary.closest_yards.toFixed(2)} yd`;
 const winSpan=gt.window_start_time!=null?`Window [+${gt.window_start_time.toFixed(1)}s to +${gt.window_end_time.toFixed(1)}s] (frames ${gt.window_start_frame}-${gt.window_end_frame})`:'Full play window';
 const revShort=A.model_revision&&A.model_revision.length>10?A.model_revision.slice(0,10):A.model_revision||'n/a';
 const infSub=A.latency_seconds!=null?`Inference latency: ${A.latency_seconds.toFixed(2)} s (rev: ${revShort})`:`Status: ${st.replace(/_/g,' ')}`;
 let aggHtml='';
 if(DATA.vlm_summary&&DATA.vlm_summary.total_plays>0){
  const VS=DATA.vlm_summary;
  const acc=VS.exact_match_accuracy!=null?(VS.exact_match_accuracy*100).toFixed(1)+'%':'n/a';
  const abst=VS.abstention_frequency!=null?(VS.abstention_frequency*100).toFixed(1)+'%':'0.0%';
  const inv=VS.invalid_output_frequency!=null?(VS.invalid_output_frequency*100).toFixed(1)+'%':'0.0%';
  const mlat=VS.mean_latency_seconds!=null?VS.mean_latency_seconds.toFixed(2)+'s':'n/a';
  aggHtml=`<div class="note" style="margin-bottom:8px"><strong>Frozen Baseline v1 (${VS.total_plays} plays):</strong> Exact-match accuracy: <strong>${acc}</strong> (${VS.exact_match_count}/${VS.identifiable_predictions} identifiable) | Abstention rate: ${abst} | Invalid JSON rate: ${inv} | Mean latency: ${mlat}</div>`;
 }
 const frameBtns=(A.sampled_frame_ids||[]).map((fid,idx)=>{
  const t=(A.sampled_times&&A.sampled_times[idx]!=null)?` (+${A.sampled_times[idx].toFixed(1)}s)`:'';
  return `<button type="button" class="ai-frame-btn" data-fid="${fid}">Frame ${idx+1}: #${fid}${t}</button>`;
 }).join('');
 const cmpJumpBtn=(A.predicted_rusher_id&&gt.closest_rusher_id)?`<button type="button" id="aiCompareBtn" style="padding:4px 10px;background:#4a148c;font-size:12px">Compare VLM Pick (${esc(A.predicted_rusher_tag||'A')}) vs. Tracking Closest (${esc(gt.closest_rusher_tag||'B')}) in Player Tracking Metrics</button>`:'';
 let tempHtml='';
 if(gt.closest_rusher_distance_trend){
  const netChg=gt.closest_rusher_net_distance_change_yards!=null?(gt.closest_rusher_net_distance_change_yards>=0?'+':'')+gt.closest_rusher_net_distance_change_yards.toFixed(2)+' yd':'n/a';
  const pkSpd=gt.closest_rusher_peak_closing_speed_yd_s!=null?gt.closest_rusher_peak_closing_speed_yd_s.toFixed(2)+' yd/s':'n/a';
  tempHtml+=`<div class="ai-details" style="margin-top:8px;border-left:3px solid #00e5ff">`
   +`<p><strong>Independent Tracking Temporal Reference (Play ${d.play}):</strong> Closest rusher (${esc(gtRusher)}) trend: <strong>${esc(gt.closest_rusher_distance_trend)}</strong> (init ${gt.closest_rusher_initial_distance_yards!=null?gt.closest_rusher_initial_distance_yards.toFixed(2):'-'} yd &rarr; min ${gt.closest_distance_yards.toFixed(2)} yd &rarr; final ${gt.closest_rusher_final_distance_yards!=null?gt.closest_rusher_final_distance_yards.toFixed(2):'-'} yd; net change ${netChg}; peak closing speed ${pkSpd}) | Closest-approach timing phase: <strong>${esc(gt.closest_approach_phase||'n/a')}</strong> (+${(gt.closest_time_s||0).toFixed(1)}s, Sampled Frame #${gt.closest_sampled_seq_number||'-'}) | Collective rush: <strong>${esc(gt.collective_rush_trend||'n/a')}</strong> (${gt.closing_rusher_count||0}/${gt.total_rusher_count||0} rushers closing).</p>`;
  if(T&&T.outcome_status!=='not_analyzed'){
   const ui=T.unverifiable_interpretations||{};
   tempHtml+=`<p style="margin-top:6px"><strong>Temporal Pass-Rush Experiment (v2 VLM vs. Tracking):</strong></p>`
    +`<table><tr><th>Dimension</th><th>VLM Visual Claim (v2)</th><th>Tracking Reference</th><th>Objective Verification</th></tr>`
    +`<tr><td>Closest Rusher</td><td>${esc((T.predicted_rusher_tag||'')+' '+(T.predicted_rusher_name||'None'))}</td><td>${esc(gtRusher)} (${gt.closest_distance_yards.toFixed(2)} yd)</td><td>${T.exact_match?'<span style="color:#81c784">Exact Match</span>':'<span style="color:#ff8a65">Mismatch (+'+(T.distance_error_yards||0).toFixed(2)+' yd)</span>'}</td></tr>`
    +`<tr><td>QB-Rusher Distance Trend</td><td><code>${esc(T.predicted_distance_trend||'n/a')}</code></td><td>Closest: <code>${esc(T.gt_closest_distance_trend||'n/a')}</code> | Selected: <code>${esc(T.gt_selected_distance_trend||'n/a')}</code></td><td>vs Closest: <strong>${T.trend_match_vs_closest?'Match':'Mismatch'}</strong> (Closing dir: ${T.closing_direction_match_vs_closest?'Match':'Mismatch'}) | vs Selected: <strong>${T.trend_match_vs_selected?'Match':'Mismatch'}</strong></td></tr>`
    +`<tr><td>Closest-Approach Timing</td><td>Phase: <code>${esc(T.predicted_closest_approach_phase||'n/a')}</code> (Frame #${T.predicted_closest_frame_number||'-'})</td><td>Closest: <code>${esc(T.gt_closest_approach_phase||'n/a')}</code> (Frame #${T.gt_closest_sampled_seq_number||'-'})</td><td>Phase vs Closest: <strong>${T.phase_match_vs_closest?'Match':'Mismatch'}</strong> | Frame offset: ${T.sampled_frame_error_vs_closest!=null?T.sampled_frame_error_vs_closest+' frame(s)':'n/a'}</td></tr>`
    +`</table>`
    +`<p><strong>Verifiable visual observation (v2):</strong> ${esc(T.movement_pattern||'')} ${esc(T.visible_evidence||'')}</p>`
    +(T.collective_rush_description?`<p><strong>Collective pass-rush observation (v2):</strong> ${esc(T.collective_rush_description)}</p>`:'')
    +(ui.model_unverified_interpretation?`<p><strong>Unverifiable football interpretation (not validated by 2D tracking):</strong> <em>${esc(ui.model_unverified_interpretation)}</em></p>`:'')
    +`<p class="note">${esc(ui.verification_boundary_note||'')}</p>`;
  }
  tempHtml+=`</div>`;
 }
 panel.innerHTML=aggHtml
  +`<div class="ai-header"><div><strong>Play ${d.play} Visual Verification</strong> &middot; <span class="note">${esc(winSpan)}</span></div><div>${cmpJumpBtn} <span class="ai-badge ${esc(st)}">${esc(statusText)}</span></div></div>`
  +`<div class="cards">`
  +card('VLM-selected rusher (v1)',vlmRusher,vlmSub)
  +card('Tracking reference (window)',gtRusher,`${gtDist} | 2-yd zone: ${gt.entered_2yd_zone_in_window?'Yes':'No'}`)
  +card('Verification outcome',statusText,A.error_analysis||'Compared against window tracking data')
  +card('Model & inference status',A.model_name||'Qwen/Qwen2.5-VL-3B-Instruct',infSub)
  +`</div>`
  +`<div class="ai-details">`
  +`<p><strong>Model explanation (v1):</strong> ${esc(A.explanation||'n/a')}</p>`
  +(A.movement_pattern?`<p><strong>Observed movement pattern:</strong> ${esc(A.movement_pattern)}</p>`:'')
  +(A.visible_evidence?`<p><strong>Visible evidence:</strong> ${esc(A.visible_evidence)}</p>`:'')
  +(A.uncertainty?`<p><strong>Uncertainty / abstention note:</strong> ${esc(A.uncertainty)}</p>`:'')
  +`<p class="note" style="margin-top:6px">${esc(gt.metric_distinction_note||'Tracking proximity is geometric distance in the observed window; PFF labels apply to the whole play.')}</p>`
  +(frameBtns?`<div style="margin-top:6px"><span class="note">Frames shown to VLM (click to seek): </span>${frameBtns}</div>`:'')
  +`</div>`+tempHtml;
 panel.querySelectorAll('.ai-frame-btn').forEach(btn=>btn.onclick=()=>{
  const fid=Number(btn.dataset.fid);if(!d)return;
  stopPlayback();
  const idx=d.frames.findIndex(f=>f.frame===fid);
  if(idx>=0){syncPlayhead(idx);render();}
 });
 const aiCmp=$('aiCompareBtn');
 if(aiCmp)aiCmp.onclick=()=>{compareVlmVsTracking(true);window.scrollTo({top:$('comparisonSection').offsetTop-16,behavior:'smooth'});};
}
let errFilter='all';
function renderVLMErrorSummary(){
 const ep=$('vlmErrorPanel');if(!ep)return;
 const VS=DATA.vlm_summary||{},EA=VS.error_analysis_summary;
 if(!EA||!EA.evaluated_plays){ep.innerHTML='<div class="note">No evaluated VLM predictions available for error analysis.</div>';return;}
 const DE=EA.distance_error_summary||{},TM=EA.temporal_mistake_summary||{},OB=EA.observable_biases||{};
 const TV2=VS.temporal_evaluation_v2,TM2=TV2?TV2.temporal_metrics:null;
 let kpiHtml=`<div class="cards">`
  +card('Incorrect Rusher ID (v1)',`${EA.incorrect_identifications}/${EA.evaluated_plays} (${(EA.incorrect_identification_rate*100).toFixed(0)}%)`,`Mean selected rusher rank: #${(EA.mean_predicted_rusher_rank||0).toFixed(2)}`)
  +card('Distance Error (v1)',`+${(DE.mean_yards||0).toFixed(2)} yd mean`,`Median +${(DE.median_yards||0).toFixed(2)} yd | Max +${(DE.max_yards||0).toFixed(2)} yd`)
  +card('Low-Tag R1/R2 Bias (v1)',`${((OB.low_tag_r1_r2_predicted_share||0)*100).toFixed(0)}% vs ${((OB.low_tag_r1_r2_ground_truth_share||0)*100).toFixed(0)}% GT`,`R2 chosen on ${(OB.predicted_tag_counts&&OB.predicted_tag_counts.R2)||0}/${EA.evaluated_plays} plays; 0 abstentions`)
  +card('Temporal Mistakes (v1)',`${(TM.mean_peak_timing_error_seconds||0).toFixed(2)} s mean offset`,`Early-snap missed: ${TM.early_snap_proximity_missed_count||0} | Cited-frame mismatch: ${TM.cited_frame_mismatch_count||0}`);
 if(TM2&&TM2.identifiable_plays>0){
  kpiHtml+=card('Temporal v2 Rusher Match',`${TV2.exact_match_count}/${TM2.identifiable_plays} (${(TV2.exact_match_accuracy*100).toFixed(0)}%)`,`Mean dist error: +${(TV2.mean_distance_error_yards||0).toFixed(2)} yd`)
   +card('Temporal v2 Trend Accuracy',`${(TM2.closing_direction_match_vs_selected_accuracy*100).toFixed(0)}% closing dir`,`Exact 4-class trend: ${(TM2.trend_match_vs_selected_accuracy*100).toFixed(0)}% (selected) | ${(TM2.trend_match_vs_closest_accuracy*100).toFixed(0)}% (closest)`)
   +card('Temporal v2 Timing Phase',`${(TM2.phase_match_vs_closest_accuracy*100).toFixed(0)}% phase match`,`vs Selected rusher phase: ${(TM2.phase_match_vs_selected_accuracy*100).toFixed(0)}%`)
   +card('Temporal v2 Frame Error',`${(TM2.mean_sampled_frame_error_vs_closest||0).toFixed(2)} frames`,`Mean sampled frame index offset (1..4)`);
 }
 kpiHtml+=`</div>`;
 const measRows=(EA.measurable_error_categories||[]).map(c=>`<tr><td><strong>${esc(c.category_name)}</strong><br><span class="note">Measurable (${esc(c.verification_type)})</span></td><td><strong>${esc(c.measured_value)}</strong><br><span class="note">${esc(c.secondary_metric)}</span></td><td>${esc((c.affected_play_ids||[]).join(', '))}</td><td>${esc(c.objective_evidence)}</td></tr>`).join('');
 const hypRows=(EA.hypothesised_causes||[]).map(h=>`<tr><td><strong>${esc(h.title)}</strong><br><span class="note" style="color:#ffd54f">Unverified Hypothesis</span></td><td>${esc(h.related_measurable_category)}</td><td>${esc(h.hypothesis_statement)}</td><td>${esc(h.why_not_directly_proven)}</td></tr>`).join('');
 const tagRows=(OB.tag_comparison_table||[]).map(t=>`<tr><td><strong>${esc(t.tag)}</strong></td><td>${t.predicted_count} (${(t.predicted_share*100).toFixed(0)}%)</td><td>${t.ground_truth_count} (${(t.ground_truth_share*100).toFixed(0)}%)</td><td>${t.share_diff>=0?'+':''}${(t.share_diff*100).toFixed(0)} pp</td></tr>`).join('');
 const diag=(EA.per_play_diagnostics||[]).filter(r=>{
  if(errFilter==='exact')return r.exact_match;
  if(errFilter==='moderate')return !r.exact_match&&r.distance_error_yards!=null&&r.distance_error_yards<=1.5;
  if(errFilter==='large')return r.distance_error_yards!=null&&r.distance_error_yards>1.5;
  if(errFilter==='temporal')return r.gt_timing_phase==='early'||(r.peak_timing_error_seconds!=null&&r.peak_timing_error_seconds>=0.5);
  return true;
 });
 const t2Map={};if(TV2&&TV2.per_play_results)TV2.per_play_results.forEach(r=>{t2Map[r.play_id]=r;});
 const diagRows=diag.map(r=>{
  const t2=t2Map[r.play_id];
  const t2Cell=t2?`Trend: <code>${esc(t2.predicted_distance_trend||'-')}</code> (GT: <code>${esc(t2.gt_closest_distance_trend||'-')}</code>) | Phase: <code>${esc(t2.predicted_closest_approach_phase||'-')}</code> (GT: <code>${esc(t2.gt_closest_approach_phase||'-')}</code>)`:'n/a';
  return `<tr class="err-play-row" data-pid="${r.play_id}" style="cursor:pointer" title="Click to load Play ${r.play_id} and compare VLM pick vs Tracking closest in Player Tracking Metrics"><td><strong>Play ${r.play_id}</strong></td><td>${esc(r.predicted_rusher_tag)} ${esc(r.predicted_rusher_name)} (${r.predicted_min_distance_yards!=null?r.predicted_min_distance_yards.toFixed(2)+' yd':'-'}, #${r.predicted_rusher_rank||'-'})</td><td>${esc(r.ground_truth_rusher_tag)} ${esc(r.ground_truth_rusher_name)} (${r.ground_truth_min_distance_yards.toFixed(2)} yd)</td><td>${r.distance_error_yards!=null?'+'+r.distance_error_yards.toFixed(2)+' yd':'-'} <span class="note">(${esc(r.distance_error_bucket)})</span></td><td>GT +${r.gt_closest_time_s.toFixed(1)}s (${esc(r.gt_timing_phase)}) vs Pred +${r.pred_rusher_closest_time_s!=null?r.pred_rusher_closest_time_s.toFixed(1)+'s':'-'} (&Delta;=${r.peak_timing_error_seconds!=null?r.peak_timing_error_seconds.toFixed(1)+'s':'-'})</td><td>${t2Cell}</td></tr>`;
 }).join('');
 ep.innerHTML=kpiHtml
  +`<div class="ai-details" style="margin-top:10px">`
  +`<h4 style="margin:4px 0">1. Measurable Error Categories (Verified from Tracking Coordinates &amp; Outputs)</h4>`
  +`<table><tr><th>Measurable Category</th><th>Measured Value &amp; Breakdown</th><th>Affected Plays</th><th>Objective Tracking Evidence</th></tr>${measRows}</table>`
  +`<h4 style="margin:10px 0 4px 0">2. Hypothesised Causes (Interpretive Hypotheses — Not Directly Proven)</h4>`
  +`<table><tr><th>Hypothesis</th><th>Linked Category</th><th>Hypothesised Mechanism</th><th>Why Unverified Without Ablation</th></tr>${hypRows}</table>`
  +`<h4 style="margin:10px 0 4px 0">3. Observable Candidate R-Tag Selection Bias (Listed in ID Order R1..R5)</h4>`
  +`<table><tr><th>Candidate Tag</th><th>VLM Selected (v1)</th><th>Tracking Ground Truth</th><th>Selection Bias (&Delta;)</th></tr>${tagRows}</table>`
  +`<div style="margin:12px 0 6px 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">`
  +`<h4 style="margin:0">4. Interactive Per-Play Error &amp; Temporal Pass-Rush Diagnostics (click any row to load play &amp; compare in Player Tracking Metrics)</h4>`
  +`<label>Filter error category: <select id="errFilterSel">`
  +`<option value="all"${errFilter==='all'?' selected':''}>All evaluated plays (${EA.evaluated_plays})</option>`
  +`<option value="exact"${errFilter==='exact'?' selected':''}>Exact matches (0.00 yd)</option>`
  +`<option value="moderate"${errFilter==='moderate'?' selected':''}>Near-tie &amp; moderate errors (&le; 1.5 yd)</option>`
  +`<option value="large"${errFilter==='large'?' selected':''}>Large distance errors (&gt; 1.5 yd)</option>`
  +`<option value="temporal"${errFilter==='temporal'?' selected':''}>Early-snap / temporal peak offsets</option>`
  +`</select></label></div>`
  +`<table><tr><th>Play</th><th>VLM Selected (v1, Min Dist, Rank)</th><th>Tracking Closest (Min Dist)</th><th>Distance Error</th><th>Peak Proximity Timing (v1)</th><th>Temporal Pass-Rush (v2 vs. GT)</th></tr>${diagRows}</table>`
  +`</div>`;
 const sel=$('errFilterSel');if(sel)sel.onchange=()=>{errFilter=sel.value;renderVLMErrorSummary();};
 ep.querySelectorAll('.err-play-row').forEach(tr=>tr.onclick=()=>{
  const pid=Number(tr.dataset.pid);
  const k=allPlays.findIndex(p=>p.play===pid);
  if(k<0)return;
  if(!order.includes(k)){$('filter').value='all';buildOrder();}
  select.value=String(k);selectPlay();
  compareVlmVsTracking(true);
  window.scrollTo({top:$('desc').offsetTop-20,behavior:'smooth'});
 });
}
function selectPlay(){
 stopPlayback();i=0;playhead=0;
 if(!order.length){d=null;$('desc').textContent='No plays match this filter.';$('headline').textContent='';$('cards').innerHTML='';$('status').textContent='';$('pffTable').innerHTML='';if($('aiPanel'))$('aiPanel').innerHTML='';if($('liveGapPanel'))$('liveGapPanel').innerHTML='';if($('sectorPanel'))$('sectorPanel').innerHTML='';
  $('field').getContext('2d').clearRect(0,0,780,560);$('chart').getContext('2d').clearRect(0,0,650,440);if($('telemetryChart'))$('telemetryChart').getContext('2d').clearRect(0,0,1360,540);return;}
 d=allPlays[Number(select.value)];slider.max=d.frames.length-1;slider.step=smoothMode?'0.02':'1';slider.value='0';
 $('desc').textContent=`Game ${d.game} | Play ${d.play} | ${d.offense} vs ${d.defense} | Result: ${d.outcome_label} | Yards: ${d.yards}. ${d.description}`;
 $('headline').textContent=d.headline;
 const S=d.summary,fp=S.first_pressure;
 $('cards').innerHTML=card('Time to '+d.end_label.toLowerCase(),d.duration.toFixed(1)+' s','after the snap')
  +card('Closest rusher to QB',S.closest_yards.toFixed(1)+' yd',S.closest_name+' at '+S.closest_time.toFixed(1)+' s')
  +card('Pressure arrival (within '+TH+' yd)',fp?fp.time.toFixed(1)+' s':'None',fp?fp.name+' got within '+TH+' yd':'No rusher got within '+TH+' yd')
  +card('Result',d.outcome_label,d.yards+' yd | PFF: '+d.pff_label);
 $('pffTable').innerHTML=d.pff_html;
 populateComparisonSelectors();
 renderAIPanel();
 render();}
function render(){
 if(!d)return;
 const vf=getVisualFrameState();if(!vf)return;
 const f=vf.fMeas,visEntities=vf.visEntities,visTime=vf.visTime,S=d.summary;
 const nr=d.rushers.find(r=>r.id===f.nearest);
 const interpBadge=vf.isInterp?` | Visual interp +${visTime.toFixed(2)}s (tracking metrics locked to 10 Hz Frame #${f.frame})`:` | Exact 10 Hz Frame #${f.frame}`;
 $('status').textContent=`Measured +${f.time.toFixed(1)} s after snap | Frame ${f.frame}${interpBadge} | Nearest rusher: ${nr.name} (${f.distances[String(f.nearest)].toFixed(2)} yd) | Endpoint: ${d.end_label}`;
 const c=$('field'),ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);
 const x0=d.view.x0,x1=d.view.x1,s=(W-50)/(x1-x0),half=(H-60)/s/2;
 let yc=Math.min(Math.max(d.view.qb_y,half),53.3-half);if(half>26.65)yc=26.65;const ylo=yc-half;
 const px=x=>25+(x-x0)*s,py=y=>H-30-(y-ylo)*s;
 ctx.fillStyle='#194f3a';ctx.fillRect(25,30,W-50,H-60);
 ctx.save();ctx.beginPath();ctx.rect(25,30,W-50,H-60);ctx.clip();
 ctx.lineWidth=1;ctx.strokeStyle='rgba(160,200,175,0.45)';
 for(let x=Math.ceil(x0/5)*5;x<=x1;x+=5){ctx.beginPath();ctx.moveTo(px(x),30);ctx.lineTo(px(x),H-30);ctx.stroke();}
 ctx.strokeStyle='#fff';ctx.lineWidth=2;[0,53.3].forEach(y=>{ctx.beginPath();ctx.moveTo(25,py(y));ctx.lineTo(W-25,py(y));ctx.stroke();});
 ctx.strokeStyle='#ffd54f';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(px(d.view.los),30);ctx.lineTo(px(d.view.los),H-30);ctx.stroke();
 ctx.fillStyle='#ffd54f';ctx.font='bold 12px Arial';ctx.fillText('Line of scrimmage',px(d.view.los)+5,46);
 const cmpOn=$('cmpEnable')&&$('cmpEnable').checked;
 const showTrails=cmpOn&&$('cmpTrails')&&$('cmpTrails').checked;
 const showVecs=cmpOn&&$('cmpVectors')&&$('cmpVectors').checked;
 const qbVis=visEntities.find(e=>e.id===d.qb),nrVis=visEntities.find(e=>e.id===f.nearest);
 // Historical movement trails for Player A and Player B
 if(showTrails&&d.telemetry){
  [[playerAId,COL_A],[playerBId,COL_B]].forEach(([pid,col])=>{
   if(!pid)return;
   const rTel=d.telemetry.rushers[String(pid)];
   const disc=rTel?rTel.discontinuity:[];
   ctx.strokeStyle=col;ctx.lineWidth=3;ctx.globalAlpha=0.75;ctx.beginPath();
   let started=false;
   const maxHist=Math.floor(vf.pos);
   for(let k=0;k<=maxHist;k++){
    if(disc[k])started=false;
    const ent=d.frames[k].entities.find(e=>e.id===pid);
    if(!ent){started=false;continue;}
    if(!started){ctx.moveTo(px(ent.x),py(ent.y));started=true;}else{ctx.lineTo(px(ent.x),py(ent.y));}
   }
   const curEnt=visEntities.find(e=>e.id===pid);
   if(curEnt&&started&&!vf.discont)ctx.lineTo(px(curEnt.x),py(curEnt.y));
   ctx.stroke();ctx.globalAlpha=1.0;
  });
 }
 // Nearest rusher dashed white line
 if(qbVis&&nrVis){
  ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(px(qbVis.x),py(qbVis.y));ctx.lineTo(px(nrVis.x),py(nrVis.y));ctx.stroke();ctx.setLineDash([]);
 }
 const placedFieldLabels=[];
 // Optional QB-to-player distance vectors for Player A and Player B
 if(showVecs&&qbVis){
  [[playerAId,COL_A,'A'],[playerBId,COL_B,'B']].forEach(([pid,col,lbl])=>{
   if(!pid)return;
   const pVis=visEntities.find(e=>e.id===pid);
   const measDist=f.distances[String(pid)];
   if(!pVis||measDist==null)return;
   const xQ=px(qbVis.x),yQ=py(qbVis.y),xP=px(pVis.x),yP=py(pVis.y);
   ctx.strokeStyle=col;ctx.lineWidth=2;ctx.setLineDash([6,4]);
   ctx.beginPath();ctx.moveTo(xQ,yQ);ctx.lineTo(xP,yP);ctx.stroke();ctx.setLineDash([]);
   const mx=(xQ+xP)/2,my=(yQ+yP)/2+(lbl==='A'?-12:14);
   const txt=`${lbl}: ${measDist.toFixed(2)} yd`;
   ctx.font='bold 11px Arial';
   const tw=ctx.measureText(txt).width+8;
   ctx.fillStyle='rgba(16,24,39,0.88)';ctx.fillRect(mx-tw/2,my-9,tw,15);
   ctx.fillStyle=col;ctx.fillText(txt,mx-tw/2+4,my+2);
   placedFieldLabels.push({x:mx-tw/2,y:my+2,w:tw});
  });
 }
 const rank=e=>e.role==='Football'?5:(e.id===d.qb?4:(e.role==='Pass Rush'?3:1));
 hits=[];
 for(const e of [...visEntities].sort((a,b)=>rank(a)-rank(b))){
  const X=px(e.x),Y=py(e.y),isQB=e.id===d.qb,isR=e.role==='Pass Rush',isB=e.role==='Football';
  const isA=cmpOn&&e.id===playerAId,isBPlayer=cmpOn&&e.id===playerBId;
  let col,r;
  if(isB){col='#c88b55';r=5;}else if(isQB){col='#ffeb3b';r=9;}else if(isR){col='#f56b72';r=8;}else{col=e.team===d.offense?'#4299ff':'#f56b72';r=6;}
  ctx.beginPath();ctx.arc(X,Y,r,0,7);ctx.fillStyle=col;ctx.fill();
  if(isR){ctx.lineWidth=3;ctx.strokeStyle=isA?COL_A:(isBPlayer?COL_B:colorOf(e.id));ctx.stroke();}
  if(isA||isBPlayer){
   ctx.beginPath();ctx.arc(X,Y,r+4,0,7);ctx.lineWidth=2.5;ctx.strokeStyle=isA?COL_A:COL_B;ctx.stroke();
  }
  if(isR&&e.id===f.nearest){ctx.beginPath();ctx.arc(X,Y,r+7,0,7);ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();}
  if(isR||isQB){
   const tagPrefix=isA?'[A] ':(isBPlayer?'[B] ':'');
   const txt=isQB?'QB':tagPrefix+lastName(e.name);
   ctx.font='bold 12px Arial';
   const tw=ctx.measureText(txt).width;
   const candOffsets=[[12,-6],[12,14],[-tw-12,-6],[-tw-12,14],[0,-16],[0,22]];
   let lx=X+12,ly=Y-6;
   for(const [ox,oy] of candOffsets){
    const cx=X+ox,cy=Y+oy;
    const clash=placedFieldLabels.some(b=>Math.abs(cx-b.x)<Math.max(tw,b.w)*0.75&&Math.abs(cy-b.y)<13);
    if(!clash){lx=cx;ly=cy;break;}
   }
   placedFieldLabels.push({x:lx,y:ly,w:tw});
   ctx.fillStyle=isA?COL_A:(isBPlayer?COL_B:'#fff');
   ctx.fillText(txt,lx,ly);
  }
  const measEnt=f.entities.find(me=>me.id===e.id)||e;
  let t=isB?'Football':`${e.name}${e.number?' #'+e.number:''} | ${e.role} | ${e.team===d.offense?'offense':'defense'}`;
  if(isR&&f.distances[String(e.id)]!=null)t+=` | ${f.distances[String(e.id)].toFixed(2)} yd from QB`;
  if(measEnt.s!=null)t+=` | s=${measEnt.s.toFixed(2)} yd/s`;
  if(measEnt.a!=null)t+=` | a=${measEnt.a.toFixed(2)} yd/s²`;
  hits.push({x:X,y:Y,t:t});}
 ctx.restore();
 ctx.fillStyle='#fff';ctx.font='13px Arial';ctx.textAlign='center';ctx.fillText('Offense moves left to right  (yards downfield)',W/2,20);
 for(let x=Math.ceil(x0/10)*10;x<=x1;x+=10){if(x<10||x>110)continue;const n=x-10;ctx.fillText(String(n<=50?n:100-n),px(x),H-10);}
 ctx.textAlign='left';
 // Feature A: Interactive Distance Chart Upgrade (#chart — Progressive Reveal vs Full Trajectory)
 const z=$('chart').getContext('2d');z.clearRect(0,0,650,440);
 const chartMode=($('distChartMode')&&$('distChartMode').value)||'progressive';
 const maxY=Math.max(5,...d.frames.flatMap(ff=>Object.values(ff.distances)))*1.1;
 const tx=t=>55+t/Math.max(d.duration,.1)*565,ty=v=>290-v/maxY*245;
 z.textAlign='left';z.font='bold 12px Arial';z.fillStyle='#eef2ff';
 z.fillText(`Pass-rusher distance to QB (yd) — ${chartMode==='progressive'?'Progressive Reveal':'Full Trajectory'}`,56,22);
 z.fillStyle='rgba(255,82,82,0.13)';z.fillRect(55,ty(TH),565,290-ty(TH));
 z.strokeStyle='#65748b';z.lineWidth=1;z.beginPath();z.moveTo(55,40);z.lineTo(55,290);z.lineTo(620,290);z.stroke();
 z.font='12px Arial';z.fillStyle='#fff';
 for(let k=0;k<=4;k++){const v=maxY*k/4;z.fillText(v.toFixed(1),8,ty(v)+4);const tt=d.duration*k/4;z.fillText(tt.toFixed(1),tx(tt)-8,310);}
 z.fillText('Seconds after snap (hover points for details; click/drag to seek)',155,330);
 z.strokeStyle='#ff5252';z.lineWidth=1.5;z.setLineDash([6,4]);z.beginPath();z.moveTo(55,ty(TH));z.lineTo(620,ty(TH));z.stroke();z.setLineDash([]);
 z.fillStyle='#ff5252';z.fillText('Pressure zone ('+TH+' yd)',480,ty(TH)-5);
 const maxPlotIdx=chartMode==='progressive'?Math.floor(vf.pos):(d.frames.length-1);
 const i0=Math.floor(vf.pos),i1=Math.min(d.frames.length-1,i0+1),alpha=vf.pos-i0;
 chartHits=[];
 const placedChartY=[];
 d.rushers.forEach((r,k)=>{
  const ridStr=String(r.id);
  const rTel=d.telemetry&&d.telemetry.rushers?d.telemetry.rushers[ridStr]:null;
  const tag=(rTel&&rTel.tag)||`R${k+1}`;
  const disc=rTel?rTel.discontinuity:[];
  const isSelA=cmpOn&&r.id===playerAId,isSelB=cmpOn&&r.id===playerBId,key=r.id===S.closest_id;
  const col=isSelA?COL_A:(isSelB?COL_B:PAL[k%PAL.length]);
  z.globalAlpha=(isSelA||isSelB||key)?1:0.45;
  z.strokeStyle=col;
  z.lineWidth=(isSelA||isSelB||key)?3.2:1.6;
  z.beginPath();
  let started=false;
  for(let j=0;j<=maxPlotIdx;j++){
   const ff=d.frames[j],v=ff.distances[ridStr];
   if(v==null||disc[j]){started=false;}
   if(v==null)continue;
   const x=tx(ff.time),y=ty(v);
   if(!started){z.moveTo(x,y);started=true;}else{z.lineTo(x,y);}
   chartHits.push({x,y,t:`${tag} ${r.name} | Frame #${ff.frame} (+${ff.time.toFixed(1)}s) | QB Dist: ${v.toFixed(2)} yd (Measured 10 Hz)`});
  }
  z.stroke();
  // Moving point marker & dashed visual interpolation segment
  const d0=d.frames[i0].distances[ridStr],d1=d.frames[i1].distances[ridStr],dMeas=f.distances[ridStr];
  if(d0!=null){
   const xMeas=tx(f.time),yMeas=ty(dMeas!=null?dMeas:d0);
   z.fillStyle=col;z.beginPath();z.arc(xMeas,yMeas,(isSelA||isSelB||key)?5.2:3.8,0,7);z.fill();
   z.strokeStyle='#101827';z.lineWidth=1.2;z.stroke();
   if(vf.isInterp&&d1!=null&&!(disc[i1])){
    const dVis=(1-alpha)*d0+alpha*d1,xVis=tx(visTime),yVis=ty(dVis);
    if(chartMode==='progressive'){
     z.strokeStyle=col;z.lineWidth=2;z.setLineDash([3,3]);
     z.beginPath();z.moveTo(tx(d.frames[i0].time),ty(d0));z.lineTo(xVis,yVis);z.stroke();z.setLineDash([]);
    }
    z.strokeStyle='#fff';z.lineWidth=1.5;z.beginPath();z.arc(xVis,yVis,4.2,0,7);z.stroke();
   }
   if((isSelA||isSelB||key)&&dMeas!=null){
    z.font='bold 10px Arial';z.fillStyle=col;
    const lblX=Math.min(565,xMeas+7);
    let lblY=Math.max(52,Math.min(284,yMeas+(isSelA?-7:(isSelB?11:2))));
    while(placedChartY.some(py=>Math.abs(lblY-py)<11)&&lblY<274){lblY+=11;}
    placedChartY.push(lblY);
    z.fillText(`${tag} ${dMeas.toFixed(2)}yd`,lblX,lblY);
   }
  }
  z.globalAlpha=1;
 });
 if(S.first_pressure&&(chartMode==='full'||visTime+1e-6>=S.first_pressure.time)){
  z.fillStyle='#fff';z.beginPath();z.arc(tx(S.first_pressure.time),ty(TH),5,0,7);z.fill();
 }
 z.strokeStyle='#ffa726';z.lineWidth=2;z.beginPath();z.moveTo(tx(d.duration),40);z.lineTo(tx(d.duration),290);z.stroke();
 z.fillStyle='#ffa726';z.textAlign='right';z.fillText(d.end_label,tx(d.duration)-4,52);z.textAlign='left';
 d.rushers.forEach((r,k)=>{
  const isSelA=cmpOn&&r.id===playerAId,isSelB=cmpOn&&r.id===playerBId,key=r.id===S.closest_id;
  z.fillStyle=isSelA?COL_A:(isSelB?COL_B:PAL[k%PAL.length]);
  z.font=((isSelA||isSelB||key)?'bold ':'')+'11px Arial';
  const curD=f.distances[String(r.id)];
  const suffix=(key?' (closest)':'')+(isSelA?' [A]':'')+(isSelB?' [B]':'')+(curD!=null?` ${curD.toFixed(2)}yd`:'');
  z.fillText(r.name+suffix,65+(k%2)*280,352+Math.floor(k/2)*16);
 });
 z.font='13px Arial';z.strokeStyle='#fff';z.lineWidth=1.5;z.setLineDash([4,4]);z.beginPath();z.moveTo(tx(visTime),40);z.lineTo(tx(visTime),290);z.stroke();z.setLineDash([]);
 renderLiveGapAndSectors();
 renderTelemetryChart(visTime);
 if(typeof syncCrossPlayFromMainClock==='function')syncCrossPlayFromMainClock(visTime);
}
$('field').onmousemove=ev=>{const c=$('field'),r=c.getBoundingClientRect(),k=c.width/r.width,mx=(ev.clientX-r.left)*k,my=(ev.clientY-r.top)*k;
 let best=null,bd=16*16;for(const h of hits){const dd=(h.x-mx)**2+(h.y-my)**2;if(dd<bd){bd=dd;best=h;}}
 const tip=$('tip');if(best){tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.textContent=best.t;}else tip.style.display='none';};
$('field').onmouseleave=()=>{$('tip').style.display='none';};
$('chart').onmousemove=ev=>{
 if(isScrubbing)return;
 const c=$('chart'),r=c.getBoundingClientRect(),mx=(ev.clientX-r.left)*(c.width/r.width),my=(ev.clientY-r.top)*(c.height/r.height);
 let best=null,bd=14*14;for(const h of chartHits){const dd=(h.x-mx)**2+(h.y-my)**2;if(dd<bd){bd=dd;best=h;}}
 const tip=$('tip');if(best){tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.textContent=best.t;}else tip.style.display='none';
};
$('chart').onmouseleave=()=>{$('tip').style.display='none';};
// Interactive timeline scrubbing on both #chart and #telemetryChart
function bindTimelineScrub(canvasEl,padLeft,plotWidth){
 if(!canvasEl)return;
 function seekFromEvent(ev){
  if(!d||!d.frames.length)return;
  const rect=canvasEl.getBoundingClientRect();
  const cx=(ev.clientX-rect.left)*(canvasEl.width/rect.width);
  const u=Math.max(0,Math.min(1,(cx-padLeft)/plotWidth));
  stopPlayback();
  syncPlayhead(u*(d.frames.length-1));
  render();
 }
 canvasEl.addEventListener('mousedown',ev=>{isScrubbing=true;seekFromEvent(ev);});
 window.addEventListener('mousemove',ev=>{if(isScrubbing&&ev.target===canvasEl)seekFromEvent(ev);});
 window.addEventListener('mouseup',()=>{isScrubbing=false;});
}
bindTimelineScrub($('chart'),55,565);
bindTimelineScrub($('telemetryChart'),82,1360-82-28);
// Feature E: Cross-Play & Cross-Game Comparison (ball_snap t=0.0s aligned, unequal duration handling without extrapolation)
const CG_PLAYS=Object.assign({},DATA.cross_game_plays||{});
const AVAIL_GAMES=DATA.available_games||[{game_id:DATA.game,label:String(DATA.game)}];
let cpTime=0.0,cpSyncLock=false;
function ensureGameLoaded(gid,cb){
 const key=String(gid);
 if(CG_PLAYS[key]){cb(CG_PLAYS[key]);return;}
 if(window.__QB_CROSS_GAME_BUNDLES&&window.__QB_CROSS_GAME_BUNDLES[key]){
  CG_PLAYS[key]=window.__QB_CROSS_GAME_BUNDLES[key];cb(CG_PLAYS[key]);return;
 }
 const gm=AVAIL_GAMES.find(g=>String(g.game_id)===key);
 if(!gm||!gm.bundle_js){cb([]);return;}
 const sc=document.createElement('script');
 sc.src=gm.bundle_js;
 sc.onload=()=>{
  if(window.__QB_CROSS_GAME_BUNDLES&&window.__QB_CROSS_GAME_BUNDLES[key]){
   CG_PLAYS[key]=window.__QB_CROSS_GAME_BUNDLES[key];
  }
  cb(CG_PLAYS[key]||[]);
 };
 sc.onerror=()=>cb([]);
 document.head.appendChild(sc);
}
function getCpPlay(gameSelId,playSelId){
 const g=$(gameSelId)&&$(gameSelId).value,p=$(playSelId)&&Number($(playSelId).value);
 const list=CG_PLAYS[String(g)]||[];
 return list.find(x=>x.play===p)||list[0]||null;
}
function populateCpPlayAndRusher(slot,keepPlayId,keepPlayerId){
 const gSel=$(`cpGame${slot}`),pSel=$(`cpPlay${slot}`),rSel=$(`cpPlayer${slot}`);
 if(!gSel||!pSel||!rSel)return;
 ensureGameLoaded(gSel.value,playsList=>{
  pSel.innerHTML='';
  playsList.forEach(p=>{
   const o=document.createElement('option');o.value=String(p.play);
   o.textContent=`Play ${p.play}: ${p.offense}@${p.defense} (${p.duration.toFixed(1)}s, ${p.outcome_label}) — ${p.label.slice(0,68)}`;
   pSel.appendChild(o);
  });
  if(keepPlayId&&playsList.some(p=>p.play===Number(keepPlayId)))pSel.value=String(keepPlayId);
  else if(slot==='B'&&playsList.length>1&&gSel.value===($('cpGameA')&&$('cpGameA').value)){
   const pAVal=Number($('cpPlayA')&&$('cpPlayA').value);
   const alt=playsList.find(p=>p.play!==pAVal)||playsList[0];
   if(alt)pSel.value=String(alt.play);
  }
  populateCpRushersOnly(slot,keepPlayerId);
 });
}
function populateCpRushersOnly(slot,keepPlayerId){
 const p=getCpPlay(`cpGame${slot}`,`cpPlay${slot}`),rSel=$(`cpPlayer${slot}`);
 if(!p||!rSel)return;
 const T=p.telemetry||{},tRush=T.rushers||{};
 const sorted=[...(p.rushers||[])].sort((a,b)=>a.id-b.id);
 rSel.innerHTML='';
 sorted.forEach((r,idx)=>{
  const tr=tRush[String(r.id)]||{},sum=tr.summary||{};
  const o=document.createElement('option');o.value=String(r.id);
  o.textContent=`${tr.tag||('R'+(idx+1))}: ${r.name}${tr.number?' #'+tr.number:''} (min ${sum.closest_distance_yards!=null?sum.closest_distance_yards.toFixed(2):'-'} yd)`;
  rSel.appendChild(o);
 });
 if(keepPlayerId&&sorted.some(r=>r.id===Number(keepPlayerId)))rSel.value=String(keepPlayerId);
 else if(T.default_player_a_id)rSel.value=String(T.default_player_a_id);
 renderCrossPlay();
}
function evalCrossSlotAtTime(p,rid,tSec){
 if(!p||!p.telemetry||!p.telemetry.rushers)return null;
 const r=p.telemetry.rushers[String(rid)]||Object.values(p.telemetry.rushers)[0];
 if(!r)return null;
 const times=p.telemetry.times_s||p.frames.map(f=>f.time);
 const dur=times.length?times[times.length-1]:p.duration;
 const ended=tSec>dur+1e-6;
 const clampedT=Math.max(0,Math.min(tSec,dur));
 let idx=0,bestD=1e9;
 for(let k=0;k<times.length;k++){const dd=Math.abs(times[k]-clampedT);if(dd<bestD){bestD=dd;idx=k;}}
 const f=p.frames[idx]||{};
 const qbEnt=(f.entities||[]).find(e=>e.id===p.qb);
 const rEnt=(f.entities||[]).find(e=>e.id===Number(r.player_id||r.nfl_id||rid));
 return {
  play:p,rusher:r,idx:idx,ended:ended,evalTime:times[idx]||0,duration:dur,
  dist:r.distance_yd[idx],speed:r.speed_yd_s[idx],accel:r.acceleration_yd_s2[idx],closing:r.closing_rate_yd_s[idx],
  minSoFar:r.min_distance_so_far_yd?r.min_distance_so_far_yd[idx]:r.distance_yd[idx],
  qbEnt:qbEnt,rEnt:rEnt
 };
}
function syncCrossPlayFromMainClock(visTime){
 if(cpSyncLock)return;
 cpTime=Math.max(0,Number(visTime)||0);
 const sl=$('cpTimeSlider');
 if(sl){
  const pA=getCpPlay('cpGameA','cpPlayA'),pB=getCpPlay('cpGameB','cpPlayB');
  const maxDur=Math.max(pA?pA.duration:3,pB?pB.duration:3,0.5);
  sl.max=String(maxDur.toFixed(2));
  sl.value=String(Math.min(cpTime,maxDur));
 }
 renderCrossPlay();
}
function renderCrossPlay(){
 const pA=getCpPlay('cpGameA','cpPlayA'),pB=getCpPlay('cpGameB','cpPlayB');
 if(!pA||!pB)return;
 const ridA=Number($('cpPlayerA')&&$('cpPlayerA').value),ridB=Number($('cpPlayerB')&&$('cpPlayerB').value);
 const maxDur=Math.max(pA.duration||0.1,pB.duration||0.1);
 const sl=$('cpTimeSlider');if(sl){sl.max=String(maxDur.toFixed(2));}
 const stA=evalCrossSlotAtTime(pA,ridA,cpTime),stB=evalCrossSlotAtTime(pB,ridB,cpTime);
 if(!stA||!stB)return;
 if($('cpClockLabel'))$('cpClockLabel').textContent=`+${cpTime.toFixed(2)} s (Shared Range: 0.0s – +${maxDur.toFixed(1)}s)`;
 if($('cpStatusBadges')){
  const bA=stA.ended?`<span class="ended-badge">Slot A: PLAY ENDED (+${stA.duration.toFixed(1)}s ${esc(pA.end_label)})</span>`:`<span class="active-badge">Slot A: Active (+${stA.evalTime.toFixed(1)}s / ${stA.duration.toFixed(1)}s)</span>`;
  const bB=stB.ended?`<span class="ended-badge">Slot B: PLAY ENDED (+${stB.duration.toFixed(1)}s ${esc(pB.end_label)})</span>`:`<span class="active-badge">Slot B: Active (+${stB.evalTime.toFixed(1)}s / ${stB.duration.toFixed(1)}s)</span>`;
  $('cpStatusBadges').innerHTML=bA+' '+bB;
 }
 const sumA=stA.rusher.summary||{},sumB=stB.rusher.summary||{};
 const bothActive=!stA.ended&&!stB.ended;
 const gapNow=(bothActive&&stA.dist!=null&&stB.dist!=null)?(stA.dist-stB.dist):null;
 const gapTxt=gapNow!==null?`${gapNow>=0?'+':''}${gapNow.toFixed(2)} yd (${gapNow<0?'Slot A closer':'Slot B closer'})`:'One play ended (no extrapolation)';
 if($('crossSummaryCards')){
  $('crossSummaryCards').innerHTML=
   card(`Slot A: ${stA.rusher.name} (G${pA.game} P${pA.play})`,`Cur: ${stA.dist!=null?stA.dist.toFixed(2)+' yd':'n/a'}${stA.ended?' [ENDED]':''}`,`Play min: ${(sumA.closest_distance_yards||0).toFixed(2)} yd at +${(sumA.closest_time_s||0).toFixed(1)}s | Max spd: ${(sumA.max_speed_yd_s||0).toFixed(2)} yd/s`)
   +card(`Slot B: ${stB.rusher.name} (G${pB.game} P${pB.play})`,`Cur: ${stB.dist!=null?stB.dist.toFixed(2)+' yd':'n/a'}${stB.ended?' [ENDED]':''}`,`Play min: ${(sumB.closest_distance_yards||0).toFixed(2)} yd at +${(sumB.closest_time_s||0).toFixed(1)}s | Max spd: ${(sumB.max_speed_yd_s||0).toFixed(2)} yd/s`)
   +card('Snap-Aligned Distance Gap (A - B)',gapTxt,`Play Min Diff: ${((sumA.closest_distance_yards||0)-(sumB.closest_distance_yards||0)>=0?'+':'')+((sumA.closest_distance_yards||0)-(sumB.closest_distance_yards||0)).toFixed(2)} yd`)
   +card('Play Durations & Outcomes',`A: ${stA.duration.toFixed(1)}s vs B: ${stB.duration.toFixed(1)}s`,`A: ${pA.outcome_label} (${pA.pff_label}) | B: ${pB.outcome_label} (${pB.pff_label})`);
 }
 // Render #crossFieldCanvas (LOS-normalized spatial overlay)
 const fc=$('crossFieldCanvas');
 if(fc){
  const ctx=fc.getContext('2d'),W=fc.width,H=fc.height;ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#194f3a';ctx.fillRect(25,28,W-50,H-54);
  const xMin=-14,xMax=16,yMin=-16,yMax=16;
  const px=rx=>25+((rx-xMin)/(xMax-xMin))*(W-50);
  const py=ry=>H-26-((ry-yMin)/(yMax-yMin))*(H-54);
  ctx.strokeStyle='rgba(160,200,175,0.35)';ctx.lineWidth=1;
  for(let rx=-10;rx<=15;rx+=5){ctx.beginPath();ctx.moveTo(px(rx),28);ctx.lineTo(px(rx),H-26);ctx.stroke();ctx.fillStyle='#d1fae5';ctx.font='11px Arial';ctx.fillText(`${rx>0?'+':''}${rx}yd`,px(rx)-10,H-10);}
  for(let ry=-15;ry<=15;ry+=5){ctx.beginPath();ctx.moveTo(25,py(ry));ctx.lineTo(W-25,py(ry));ctx.stroke();}
  ctx.strokeStyle='#ffd54f';ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(px(0),28);ctx.lineTo(px(0),H-26);ctx.stroke();
  ctx.fillStyle='#ffd54f';ctx.font='bold 11px Arial';ctx.fillText('LOS (0 yd)',px(0)+4,42);
  ctx.fillStyle='#eef2ff';ctx.font='bold 12px Arial';ctx.fillText('LOS & Snap-QB Normalized Rush Paths (ΔX vs LOS, ΔY vs Snap QB)',30,18);
  [[stA,COL_A,'A'],[stB,COL_B,'B']].forEach(([st,col,lbl])=>{
   const P=st.play,los=P.view?P.view.los:50,qbY0=P.view?P.view.qb_y:26.65,rid=Number(st.rusher.player_id||st.rusher.nfl_id);
   // Trail up to st.idx
   ctx.strokeStyle=col;ctx.lineWidth=3;ctx.beginPath();
   let started=false;
   for(let k=0;k<=st.idx;k++){
    const ent=(P.frames[k].entities||[]).find(e=>e.id===rid);
    if(!ent){started=false;continue;}
    const X=px(ent.x-los),Y=py(ent.y-qbY0);
    if(!started){ctx.moveTo(X,Y);started=true;}else{ctx.lineTo(X,Y);}
   }
   ctx.stroke();
   // QB marker and Rusher marker at st.idx
   if(st.qbEnt){
    const qX=px(st.qbEnt.x-los),qY=py(st.qbEnt.y-qbY0);
    ctx.fillStyle='#ffeb3b';ctx.beginPath();ctx.arc(qX,qY,6.5,0,7);ctx.fill();
    ctx.strokeStyle=col;ctx.lineWidth=2;ctx.stroke();
    ctx.fillStyle=col;ctx.font='bold 10px Arial';ctx.fillText(`QB_${lbl}`,qX+8,qY+(lbl==='A'?-5:11));
    if(st.rEnt){
     const rX=px(st.rEnt.x-los),rY=py(st.rEnt.y-qbY0);
     ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
     ctx.beginPath();ctx.moveTo(qX,qY);ctx.lineTo(rX,rY);ctx.stroke();ctx.setLineDash([]);
     ctx.fillStyle=col;ctx.beginPath();ctx.arc(rX,rY,8,0,7);ctx.fill();
     ctx.strokeStyle='#101827';ctx.lineWidth=1.5;ctx.stroke();
     const statusTag=st.ended?' [PLAY ENDED]':'';
     ctx.fillStyle=col;ctx.font='bold 11px Arial';
     ctx.fillText(`[${lbl}] ${lastName(st.rusher.name)} (${(st.dist||0).toFixed(2)}yd)${statusTag}`,rX+10,rY+(lbl==='A'?-6:12));
    }
   }
  });
 }
 // Render #crossTelemetryCanvas (Aligned at ball_snap t=0.0s, no extrapolation past play end)
 const tc=$('crossTelemetryCanvas');
 if(tc){
  const ctx=tc.getContext('2d'),W=tc.width,H=tc.height;ctx.clearRect(0,0,W,H);
  const padL=64,padR=22,plotW=W-padL-padR;
  const tx=t=>padL+(t/maxDur)*plotW;
  const rA=stA.rusher,rB=stB.rusher;
  const tA=pA.telemetry.times_s||pA.frames.map(f=>f.time),tB=pB.telemetry.times_s||pB.frames.map(f=>f.time);
  const maxD=Math.max(5,...rA.distance_yd.filter(v=>v!=null),...rB.distance_yd.filter(v=>v!=null))*1.1;
  const maxS=Math.max(6,...rA.speed_yd_s.filter(v=>v!=null),...rB.speed_yd_s.filter(v=>v!=null))*1.1;
  const maxC=Math.max(3,...rA.closing_rate_yd_s.filter(v=>v!=null).map(v=>Math.abs(v)),...rB.closing_rate_yd_s.filter(v=>v!=null).map(v=>Math.abs(v)))*1.12;
  const subTracks=[
   {top:24,bot:138,title:'1. QB Distance (yd — stops at play end)',maxV:maxD,signed:false,vA:rA.distance_yd,vB:rB.distance_yd,hasTh:true},
   {top:158,bot:272,title:'2. Scalar Ground Speed s (yd/s)',maxV:maxS,signed:false,vA:rA.speed_yd_s,vB:rB.speed_yd_s},
   {top:292,bot:404,title:'3. QB Closing Rate -Δd/Δt (yd/s)',maxV:maxC,signed:true,vA:rA.closing_rate_yd_s,vB:rB.closing_rate_yd_s}
  ];
  subTracks.forEach(tr=>{
   const plotTop=tr.top+18;
   const yFn=tr.signed?(v=>((plotTop+tr.bot)/2)-(v/tr.maxV)*((tr.bot-plotTop)/2)):(v=>tr.bot-(v/tr.maxV)*(tr.bot-plotTop));
   ctx.strokeStyle='#4a5d7e';ctx.lineWidth=1;
   ctx.beginPath();ctx.moveTo(padL,plotTop);ctx.lineTo(padL,tr.bot);ctx.lineTo(padL+plotW,tr.bot);ctx.stroke();
   if(tr.hasTh){
    ctx.fillStyle='rgba(255,82,82,0.12)';ctx.fillRect(padL,yFn(TH),plotW,tr.bot-yFn(TH));
    ctx.strokeStyle='#ff5252';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(padL,yFn(TH));ctx.lineTo(padL+plotW,yFn(TH));ctx.stroke();ctx.setLineDash([]);
   }
   if(tr.signed){
    ctx.strokeStyle='rgba(255,255,255,0.3)';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(padL,yFn(0));ctx.lineTo(padL+plotW,yFn(0));ctx.stroke();ctx.setLineDash([]);
   }
   [[tA,tr.vA,COL_A,stA],[tB,tr.vB,COL_B,stB]].forEach(([tArr,vArr,col,st])=>{
    ctx.strokeStyle=col;ctx.lineWidth=2.4;ctx.beginPath();
    let started=false;
    for(let k=0;k<tArr.length;k++){
     const v=vArr[k];if(v==null){started=false;continue;}
     const X=tx(tArr[k]),Y=yFn(v);
     if(!started){ctx.moveTo(X,Y);started=true;}else{ctx.lineTo(X,Y);}
    }
    ctx.stroke();
    // Moving dot or End marker
    const curV=vArr[st.idx];
    if(curV!=null){
     const mX=tx(st.evalTime),mY=yFn(curV);
     ctx.fillStyle=col;ctx.beginPath();ctx.arc(mX,mY,4.8,0,7);ctx.fill();
    }
    const endX=tx(st.duration);
    ctx.strokeStyle=col;ctx.lineWidth=1.2;ctx.setLineDash([2,2]);
    ctx.beginPath();ctx.moveTo(endX,tr.top);ctx.lineTo(endX,tr.bot);ctx.stroke();ctx.setLineDash([]);
   });
   ctx.fillStyle='#eef2ff';ctx.font='bold 11px Arial';ctx.fillText(tr.title,padL+6,tr.top+12);
   ctx.fillStyle='#9fb0cc';ctx.font='10px Arial';ctx.textAlign='right';
   const ticks=tr.signed?[-tr.maxV,0,tr.maxV]:[0,tr.maxV/2,tr.maxV];
   ticks.forEach(tv=>ctx.fillText((tv>0&&tr.signed?'+':'')+tv.toFixed(1),padL-5,yFn(tv)+3));
   ctx.textAlign='left';
  });
  // Shared cursor
  const curX=tx(Math.min(cpTime,maxDur));
  subTracks.forEach(tr=>{
   ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
   ctx.beginPath();ctx.moveTo(curX,tr.top);ctx.lineTo(curX,tr.bot);ctx.stroke();ctx.setLineDash([]);
  });
  ctx.fillStyle='#eef2ff';ctx.font='11px Arial';ctx.textAlign='center';
  for(let k=0;k<=5;k++){const tt=(maxDur*k)/5;ctx.fillText(`+${tt.toFixed(1)}s`,tx(tt),H-8);}
  ctx.textAlign='left';
 }
}
function initCrossPlay(){
 const gA=$('cpGameA'),gB=$('cpGameB');if(!gA||!gB)return;
 gA.innerHTML='';gB.innerHTML='';
 AVAIL_GAMES.forEach(g=>{
  const o1=document.createElement('option');o1.value=String(g.game_id);o1.textContent=g.label;gA.appendChild(o1);
  const o2=document.createElement('option');o2.value=String(g.game_id);o2.textContent=g.label;gB.appendChild(o2);
 });
 gA.value=String(DATA.game);
 gB.value=String(DATA.game);
 populateCpPlayAndRusher('A',d?d.play:null,playerAId);
 const altPlay=(allPlays.find(p=>d&&p.play!==d.play)||allPlays[0]);
 populateCpPlayAndRusher('B',altPlay?altPlay.play:null,null);
 gA.onchange=()=>populateCpPlayAndRusher('A',null,null);
 gB.onchange=()=>populateCpPlayAndRusher('B',null,null);
 $('cpPlayA').onchange=()=>populateCpRushersOnly('A',null);
 $('cpPlayB').onchange=()=>populateCpRushersOnly('B',null);
 $('cpPlayerA').onchange=renderCrossPlay;
 $('cpPlayerB').onchange=renderCrossPlay;
 if($('cpTimeSlider'))$('cpTimeSlider').oninput=()=>{
  cpTime=Number($('cpTimeSlider').value)||0;
  if(d&&d.frames.length){
   cpSyncLock=true;
   stopPlayback();
   const u=Math.min(1,cpTime/Math.max(d.duration,0.1));
   syncPlayhead(u*(d.frames.length-1));
   render();
   cpSyncLock=false;
  }
  renderCrossPlay();
 };
 if($('crossTelemetryCanvas')){
  $('crossTelemetryCanvas').addEventListener('mousedown',ev=>{
   const c=$('crossTelemetryCanvas'),r=c.getBoundingClientRect();
   const cx=(ev.clientX-r.left)*(c.width/r.width);
   const pA=getCpPlay('cpGameA','cpPlayA'),pB=getCpPlay('cpGameB','cpPlayB');
   const maxDur=Math.max(pA?pA.duration:1,pB?pB.duration:1,0.1);
   const u=Math.max(0,Math.min(1,(cx-64)/(c.width-86)));
   cpTime=u*maxDur;
   if($('cpTimeSlider'))$('cpTimeSlider').value=String(cpTime);
   renderCrossPlay();
  });
 }
 if($('cpSyncCurrentPlay'))$('cpSyncCurrentPlay').onclick=()=>{
  if(!d)return;
  gA.value=String(d.game);
  populateCpPlayAndRusher('A',d.play,playerAId);
 };
 if($('cpPresetSamePlayer'))$('cpPresetSamePlayer').onclick=()=>{
  const list=CG_PLAYS[String(DATA.game)]||[];
  const targetId=playerAId||41233;
  const matching=list.filter(p=>(p.rushers||[]).some(r=>r.id===Number(targetId)));
  if(matching.length>=2){
   gA.value=String(DATA.game);gB.value=String(DATA.game);
   populateCpPlayAndRusher('A',matching[0].play,targetId);
   populateCpPlayAndRusher('B',matching[1].play,targetId);
  }
 };
 if($('cpPresetDiffPlayers'))$('cpPresetDiffPlayers').onclick=()=>{
  const list=CG_PLAYS[String(DATA.game)]||[];
  if(list.length>=2){
   gA.value=String(DATA.game);gB.value=String(DATA.game);
   populateCpPlayAndRusher('A',list[0].play,list[0].telemetry.default_player_a_id);
   populateCpPlayAndRusher('B',list[1].play,list[1].telemetry.default_player_b_id);
  }
 };
 if($('cpPresetCrossGame'))$('cpPresetCrossGame').onclick=()=>{
  const otherGame=AVAIL_GAMES.find(g=>g.game_id!==DATA.game)||AVAIL_GAMES[0];
  gA.value=String(DATA.game);
  gB.value=String(otherGame.game_id);
  populateCpPlayAndRusher('A',d?d.play:null,playerAId);
  populateCpPlayAndRusher('B',null,null);
 };
}

// Feature F: Multi-Play & Player-Level Pass-Rush Analytics
const MP_RECS=DATA.multi_play_records||[];
let mpSortKey='mean_min_distance_yards',mpSortDir=1,mpSelectedPlayerId=null;
function pctQuantile(sortedArr,q){
 if(!sortedArr.length)return null;
 if(sortedArr.length===1)return sortedArr[0];
 const pos=(sortedArr.length-1)*q,lo=Math.floor(pos),hi=Math.min(sortedArr.length-1,lo+1),w=pos-lo;
 return sortedArr[lo]*(1-w)+sortedArr[hi]*w;
}
function filterMpRecords(){
 const q=(($('mpSearch')&&$('mpSearch').value)||'').trim().toLowerCase();
 const g=$('mpGameFilter')?$('mpGameFilter').value:'all';
 const tm=$('mpTeamFilter')?$('mpTeamFilter').value:'all';
 const off=$('mpOffenseFilter')?$('mpOffenseFilter').value:'all';
 const dn=$('mpDownFilter')?$('mpDownFilter').value:'all';
 const fm=$('mpFormationFilter')?$('mpFormationFilter').value:'all';
 const oc=$('mpOutcomeFilter')?$('mpOutcomeFilter').value:'all';
 const pf=$('mpPffFilter')?$('mpPffFilter').value:'all';
 return MP_RECS.filter(r=>{
  if(q&&!String(r.name).toLowerCase().includes(q))return false;
  if(g!=='all'&&String(r.game_id)!==g)return false;
  if(tm!=='all'&&String(r.defense)!==tm)return false;
  if(off!=='all'&&String(r.offense)!==off)return false;
  if(dn!=='all'&&String(r.down)!==dn)return false;
  if(fm!=='all'&&String(r.formation)!==fm)return false;
  if(oc!=='all'&&String(r.outcome)!==oc)return false;
  if(pf==='pressure'&&!r.pff_pressure)return false;
  if(pf==='no_pressure'&&r.pff_pressure)return false;
  if(pf==='within_2yd'&&!r.entered_2yd_zone)return false;
  return true;
 });
}
function renderMultiPlayAnalytics(){
 const banner=$('mpDenominatorBanner'),tbl=$('mpPlayerTable'),drill=$('mpDrilldownPanel');
 if(!banner||!tbl)return;
 const recs=filterMpRecords();
 const minPlays=Number(($('mpMinPlays')&&$('mpMinPlays').value)||1);
 const playSet=new Set(recs.map(r=>`${r.game_id}_${r.play_id}`));
 const gameSet=new Set(recs.map(r=>r.game_id));
 const byPlayer=new Map();
 recs.forEach(r=>{
  if(!byPlayer.has(r.nfl_id))byPlayer.set(r.nfl_id,[]);
  byPlayer.get(r.nfl_id).push(r);
 });
 const rows=[];
 byPlayer.forEach((pRecs,pid)=>{
  const n=pRecs.length;if(n<minPlays)return;
  const r0=pRecs[0];
  const minDists=pRecs.map(x=>x.closest_distance_yards).filter(v=>v!=null).sort((a,b)=>a-b);
  const tClosests=pRecs.map(x=>x.closest_time_s).filter(v=>v!=null).sort((a,b)=>a-b);
  const arrs=pRecs.map(x=>x.first_within_2yd_time_s).filter(v=>v!=null).sort((a,b)=>a-b);
  const meanCloses=pRecs.map(x=>x.mean_closing_rate_yd_s).filter(v=>v!=null).sort((a,b)=>a-b);
  const maxSpds=pRecs.map(x=>x.max_speed_yd_s).filter(v=>v!=null);
  const wCount=pRecs.filter(x=>x.entered_2yd_zone).length;
  const pffCount=pRecs.filter(x=>x.pff_pressure).length;
  const hitsC=pRecs.filter(x=>x.pff_hit).length,hurC=pRecs.filter(x=>x.pff_hurry).length,sackC=pRecs.filter(x=>x.pff_sack).length;
  const mean=arr=>arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:null;
  const q25=pctQuantile(minDists,0.25),med=pctQuantile(minDists,0.5),q75=pctQuantile(minDists,0.75);
  rows.push({
   nfl_id:pid,name:r0.name,number:r0.number,team:r0.defense,plays:n,low_sample:n<5,
   mean_min_distance_yards:mean(minDists),median_min_distance_yards:med,
   iqr_min_distance_yards:(q75!=null&&q25!=null)?(q75-q25):0,q25_min:q25,q75_min:q75,
   best_min:minDists[0],worst_min:minDists[minDists.length-1],
   within_count:wCount,pct_within_2yd:(wCount/n)*100,
   mean_arrival_s:mean(arrs),median_arrival_s:pctQuantile(arrs,0.5),
   mean_time_to_closest_s:mean(tClosests),median_time_to_closest_s:pctQuantile(tClosests,0.5),
   mean_closing_rate_yd_s:mean(meanCloses),median_closing_rate_yd_s:pctQuantile(meanCloses,0.5),
   mean_max_speed_yd_s:mean(maxSpds),
   pff_count:pffCount,pff_rate_pct:(pffCount/n)*100,hits:hitsC,hurries:hurC,sacks:sackC,
   records:pRecs
  });
 });
 rows.sort((a,b)=>{
  const va=a[mpSortKey],vb=b[mpSortKey];
  if(va==null)return 1;if(vb==null)return -1;
  return typeof va==='string'?mpSortDir*va.localeCompare(vb):mpSortDir*(va-vb);
 });
 banner.innerHTML=`<strong>Explicit Sample Denominators:</strong> Matching Games: <strong>${gameSet.size}</strong> | Matching Unique Plays: <strong>${playSet.size}</strong> | Filtered Pass-Rush Snap Observations: <strong>${recs.length}</strong> | Qualifying Pass Rushers (&ge;${minPlays} play${minPlays===1?'':'s'}): <strong>${rows.length}</strong> of ${byPlayer.size}. <span class="note">(Players with &lt;5 plays are flagged with [Low Sample] so small denominators are not over-interpreted.)</span>`;
 renderAnalyticsDistCanvas(rows,recs);
 const cols=[
  ['name','Player (Team)'],
  ['plays','Evaluated Plays (Denominator)'],
  ['mean_min_distance_yards','Mean Min QB Dist (yd)'],
  ['median_min_distance_yards','Median Min Dist [IQR] (yd)'],
  ['pct_within_2yd',`% Within ${TH} yd (Count/Plays)`],
  ['mean_time_to_closest_s','Time to Closest (Mean / Med s)'],
  ['mean_closing_rate_yd_s','QB Closing Rate (Mean / Med yd/s)'],
  ['mean_max_speed_yd_s','Mean Peak Speed (yd/s)'],
  ['pff_rate_pct','PFF Pressure Rate (H/Hu/S)']
 ];
 const f=(v,p=2)=>v==null?'-':v.toFixed(p);
 tbl.innerHTML=`<table><tr>`+cols.map(([k,lbl])=>`<th data-mpk="${k}">${lbl}${k===mpSortKey?(mpSortDir<0?' &#9660;':' &#9650;'):''}</th>`).join('')+`</tr>`
  +rows.map(r=>{
   const sel=r.nfl_id===mpSelectedPlayerId;
   return `<tr class="sector-row${sel?' sector-active':''}" data-mpid="${r.nfl_id}" title="Click to inspect ${esc(r.name)} play-by-play breakdown">`
    +`<td><strong>${esc(r.name)}</strong>${r.number?' #'+esc(r.number):''} (${esc(r.team)}) ${r.low_sample?'<span style="color:#ffd54f;font-size:11px">[Low Sample &lt;5]</span>':''}</td>`
    +`<td><strong>${r.plays}</strong> play${r.plays===1?'':'s'}</td>`
    +`<td><strong>${f(r.mean_min_distance_yards,2)} yd</strong> <span class="note">(min ${f(r.best_min,2)})</span></td>`
    +`<td><strong>${f(r.median_min_distance_yards,2)} yd</strong> <span class="note">[IQR ${f(r.iqr_min_distance_yards,2)}: ${f(r.q25_min,2)}&ndash;${f(r.q75_min,2)}]</span></td>`
    +`<td><strong>${f(r.pct_within_2yd,1)}%</strong> <span class="note">(${r.within_count}/${r.plays}; arr ${f(r.mean_arrival_s,1)}s)</span></td>`
    +`<td>${f(r.mean_time_to_closest_s,2)}s / ${f(r.median_time_to_closest_s,2)}s</td>`
    +`<td>${r.mean_closing_rate_yd_s>=0?'+':''}${f(r.mean_closing_rate_yd_s,2)} / ${r.median_closing_rate_yd_s>=0?'+':''}${f(r.median_closing_rate_yd_s,2)} yd/s</td>`
    +`<td>${f(r.mean_max_speed_yd_s,2)} yd/s</td>`
    +`<td><strong>${f(r.pff_rate_pct,1)}%</strong> <span class="note">(${r.pff_count}/${r.plays}: ${r.hits}H/${r.hurries}Hu/${r.sacks}S)</span></td>`
    +`</tr>`;
  }).join('')+`</table>`;
 tbl.querySelectorAll('th').forEach(th=>th.onclick=()=>{
  const k=th.dataset.mpk;
  if(k===mpSortKey)mpSortDir=-mpSortDir;
  else{mpSortKey=k;mpSortDir=(k==='name'||k.includes('distance')||k.includes('time'))?1:-1;}
  renderMultiPlayAnalytics();
 });
 tbl.querySelectorAll('tr[data-mpid]').forEach(tr=>tr.onclick=()=>{
  mpSelectedPlayerId=Number(tr.dataset.mpid);
  renderMultiPlayAnalytics();
 });
 if(drill){
  const selRow=rows.find(r=>r.nfl_id===mpSelectedPlayerId)||rows[0];
  if(!selRow){drill.innerHTML='';return;}
  const pRows=selRow.records.map(pr=>`<tr class="sector-row" data-dg="${pr.game_id}" data-dp="${pr.play_id}" data-dr="${pr.nfl_id}" title="Click to load Game ${pr.game_id} Play ${pr.play_id} into replay / cross-play comparison">`
   +`<td>Game ${pr.game_id} &middot; <strong>Play ${pr.play_id}</strong> (Q${pr.quarter||'-'} D${pr.down||'-'}&amp;${pr.yards_to_go||'-'})</td>`
   +`<td>${esc(pr.offense)} @ ${esc(pr.defense)} (${esc(pr.formation)})</td>`
   +`<td>${pr.init_distance_yards!=null?pr.init_distance_yards.toFixed(2):'-'} &rarr; <strong>${pr.closest_distance_yards.toFixed(2)} yd</strong> (at +${pr.closest_time_s.toFixed(1)}s)</td>`
   +`<td>${pr.entered_2yd_zone?'<span class="yes">Yes (+'+pr.first_within_2yd_time_s.toFixed(1)+'s)</span>':'No'}</td>`
   +`<td>${pr.mean_closing_rate_yd_s>=0?'+':''}${pr.mean_closing_rate_yd_s.toFixed(2)} yd/s (pk ${pr.peak_closing_rate_yd_s.toFixed(2)})</td>`
   +`<td>${pr.max_speed_yd_s.toFixed(2)} yd/s</td>`
   +`<td>${esc(pr.outcome_label)} (${pr.pff_pressure?'<span class="yes">'+esc(pr.pff_label)+'</span>':'none'})</td>`
   +`</tr>`).join('');
  drill.innerHTML=`<div class="ai-details"><h4 style="margin:2px 0 6px 0">Play-by-Play Breakdown for ${esc(selRow.name)} (${selRow.plays} evaluated pass-rush snaps &mdash; click any play to load)</h4>`
   +`<table><tr><th>Game &amp; Play (Situation)</th><th>Matchup &amp; Formation</th><th>Init &rarr; Min QB Distance</th><th>&le;${TH} yd Zone</th><th>Mean / Peak Closing Rate</th><th>Peak Speed</th><th>Outcome &amp; PFF</th></tr>${pRows}</table></div>`;
  drill.querySelectorAll('tr[data-dp]').forEach(tr=>tr.onclick=()=>{
   const gid=Number(tr.dataset.dg),pid=Number(tr.dataset.dp),rid=Number(tr.dataset.dr);
   if(gid===DATA.game){
    const k=allPlays.findIndex(p=>p.play===pid);
    if(k>=0){
     if(!order.includes(k)){$('filter').value='all';buildOrder();}
     select.value=String(k);selectPlay();
     playerAId=rid;if($('playerASelect'))$('playerASelect').value=String(rid);
     render();
     window.scrollTo({top:$('desc').offsetTop-20,behavior:'smooth'});
    }
   }else{
    if($('cpGameB')){$('cpGameB').value=String(gid);populateCpPlayAndRusher('B',pid,rid);}
    window.scrollTo({top:$('crossPlaySection').offsetTop-16,behavior:'smooth'});
   }
  });
 }
}
function renderAnalyticsDistCanvas(rows,recs){
 const c=$('analyticsDistCanvas');if(!c)return;
 const ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);
 if(!rows.length){ctx.fillStyle='#9fb0cc';ctx.font='14px Arial';ctx.fillText('No players match the current filter criteria.',30,40);return;}
 // Left panel: Top 10 Rushers Min QB Distance Box/IQR + Mean/Median Plot
 const topR=rows.slice(0,10);
 const padL=185,plotW=460,topY=34,botY=H-32;
 const maxD=Math.max(8,...topR.map(r=>r.worst_min||8))*1.05;
 const txD=v=>padL+(v/maxD)*plotW;
 ctx.fillStyle='#eef2ff';ctx.font='bold 12px Arial';
 ctx.fillText('A. Rusher Min QB Distance Distribution (Box = Q25–Median–Q75, Circle = Mean, Whiskers = Min–Max)',16,18);
 ctx.strokeStyle='#4a5d7e';ctx.lineWidth=1;
 ctx.beginPath();ctx.moveTo(padL,topY);ctx.lineTo(padL,botY);ctx.lineTo(padL+plotW,botY);ctx.stroke();
 ctx.strokeStyle='#ff5252';ctx.setLineDash([4,4]);
 ctx.beginPath();ctx.moveTo(txD(TH),topY);ctx.lineTo(txD(TH),botY);ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle='#ff5252';ctx.font='10px Arial';ctx.fillText(`${TH}yd zone`,txD(TH)+3,topY+10);
 const rowH=(botY-topY)/Math.max(1,topR.length);
 topR.forEach((r,idx)=>{
  const cy=topY+(idx+0.5)*rowH;
  ctx.fillStyle='#eef2ff';ctx.font='11px Arial';ctx.textAlign='right';
  ctx.fillText(`${lastName(r.name)} (${r.team}, n=${r.plays})`,padL-8,cy+4);
  ctx.textAlign='left';
  const xMin=txD(r.best_min),xMax=txD(r.worst_min),x25=txD(r.q25_min),xMed=txD(r.median_min_distance_yards),x75=txD(r.q75_min),xMean=txD(r.mean_min_distance_yards);
  ctx.strokeStyle='#9fb0cc';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(xMin,cy);ctx.lineTo(xMax,cy);ctx.stroke();
  ctx.fillStyle='rgba(0,229,255,0.25)';ctx.strokeStyle='#00e5ff';
  ctx.fillRect(x25,cy-7,Math.max(2,x75-x25),14);ctx.strokeRect(x25,cy-7,Math.max(2,x75-x25),14);
  ctx.strokeStyle='#ffd54f';ctx.lineWidth=2.5;
  ctx.beginPath();ctx.moveTo(xMed,cy-8);ctx.lineTo(xMed,cy+8);ctx.stroke();
  ctx.fillStyle='#ffab00';ctx.beginPath();ctx.arc(xMean,cy,4,0,7);ctx.fill();
 });
 ctx.fillStyle='#9fb0cc';ctx.font='11px Arial';ctx.textAlign='center';
 for(let v=0;v<=maxD;v+=2){ctx.fillText(`${v}yd`,txD(v),H-10);}

 // Right panel: Time-to-Closest & Closing Rate Histograms across filtered player-play snaps
 const hPadL=710,hW=290,h2PadL=1045,h2W=290;
 function drawMiniHist(x0,w,title,vals,bins,minV,maxV,unit,col){
  ctx.fillStyle='#eef2ff';ctx.font='bold 12px Arial';ctx.textAlign='left';
  ctx.fillText(title,x0,18);
  ctx.strokeStyle='#4a5d7e';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(x0,topY);ctx.lineTo(x0,botY);ctx.lineTo(x0+w,botY);ctx.stroke();
  const counts=new Array(bins).fill(0);
  vals.forEach(v=>{
   const b=Math.max(0,Math.min(bins-1,Math.floor(((v-minV)/(maxV-minV))*bins)));
   counts[b]++;
  });
  const maxC=Math.max(1,...counts),bw=w/bins;
  counts.forEach((c,b)=>{
   const bh=(c/maxC)*(botY-topY-18),bx=x0+b*bw+2,by=botY-bh;
   ctx.fillStyle=col;ctx.fillRect(bx,by,bw-4,bh);
   if(c>0){ctx.fillStyle='#eef2ff';ctx.font='10px Arial';ctx.textAlign='center';ctx.fillText(String(c),bx+(bw-4)/2,by-3);}
  });
  ctx.fillStyle='#9fb0cc';ctx.font='10px Arial';ctx.textAlign='center';
  for(let k=0;k<=4;k++){const v=minV+((maxV-minV)*k)/4;ctx.fillText(`${v.toFixed(1)}${unit}`,x0+(w*k)/4,H-10);}
  ctx.textAlign='left';
 }
 drawMiniHist(hPadL,hW,`B. Time to Closest Approach (n=${recs.length} snaps)`,recs.map(r=>r.closest_time_s),8,0,6,'s','rgba(0,229,255,0.65)');
 drawMiniHist(h2PadL,h2W,`C. Mean QB Closing Rate -Δd/Δt (n=${recs.length} snaps)`,recs.map(r=>r.mean_closing_rate_yd_s),8,-2,4,'yd/s','rgba(255,171,0,0.65)');
}
function initMultiPlayAnalytics(){
 const gSel=$('mpGameFilter'),tSel=$('mpTeamFilter'),oSel=$('mpOffenseFilter'),fSel=$('mpFormationFilter');
 if(!gSel)return;
 const games=[...new Set(MP_RECS.map(r=>r.game_id))].sort();
 const teams=[...new Set(MP_RECS.map(r=>r.defense))].sort();
 const offs=[...new Set(MP_RECS.map(r=>r.offense))].sort();
 const forms=[...new Set(MP_RECS.map(r=>r.formation).filter(x=>x&&x!=='UNKNOWN'))].sort();
 games.forEach(g=>{const gm=AVAIL_GAMES.find(x=>x.game_id===g);const o=document.createElement('option');o.value=String(g);o.textContent=gm?gm.label:String(g);gSel.appendChild(o);});
 teams.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;tSel.appendChild(o);});
 offs.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;oSel.appendChild(o);});
 forms.forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent=f;fSel.appendChild(o);});
 ['mpGameFilter','mpTeamFilter','mpOffenseFilter','mpDownFilter','mpFormationFilter','mpOutcomeFilter','mpPffFilter','mpMinPlays'].forEach(id=>{if($(id))$(id).onchange=renderMultiPlayAnalytics;});
 if($('mpSearch'))$('mpSearch').oninput=renderMultiPlayAnalytics;
 if($('mpResetFilters'))$('mpResetFilters').onclick=()=>{
  if($('mpSearch'))$('mpSearch').value='';
  ['mpGameFilter','mpTeamFilter','mpOffenseFilter','mpDownFilter','mpFormationFilter','mpOutcomeFilter','mpPffFilter'].forEach(id=>{if($(id))$(id).value='all';});
  if($('mpMinPlays'))$('mpMinPlays').value='1';
  renderMultiPlayAnalytics();
 };
 renderMultiPlayAnalytics();
}
function initWorkspaceNav(){
 const nav=$('workspaceNav');if(!nav)return;
 nav.querySelectorAll('.ws-tab').forEach(btn=>btn.onclick=()=>{
  const target=btn.dataset.ws;
  nav.querySelectorAll('.ws-tab').forEach(b=>b.classList.toggle('active',b===btn));
  document.querySelectorAll('.ws-section').forEach(sec=>{
   sec.style.display=(target==='all'||sec.dataset.sec===target)?'block':'none';
  });
 });
}

// whole-game table
const COLS=[['name','Rusher'],['plays','Plays'],['avg_closest','Avg closest (yd)'],['pct_within','% plays within '+TH+' yd'],['avg_arrival','Avg arrival (s)'],['pff_plays','Plays with PFF hit/hurry/sack']];
let sk='pct_within',sd=-1;
function renderGame(){
 const rows=[...DATA.rushers];
 rows.sort((a,b)=>{const x=a[sk],y=b[sk];if(x===null)return 1;if(y===null)return -1;return typeof x==='string'?sd*x.localeCompare(y):sd*(x-y);});
 const f=(v,n)=>v===null?'-':v.toFixed(n);
 $('gameTable').innerHTML='<table><tr>'+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}${c[0]===sk?(sd<0?' &#9660;':' &#9650;'):''}</th>`).join('')+'</tr>'
  +rows.map(r=>`<tr><td>${esc(r.name)}</td><td>${r.plays}</td><td>${f(r.avg_closest,1)}</td><td>${f(r.pct_within,0)}%</td><td>${f(r.avg_arrival,1)}</td><td>${r.pff_plays}</td></tr>`).join('')+'</table>';
 $('gameTable').querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(k===sk)sd=-sd;else{sk=k;sd=k==='name'?1:-1;}renderGame();});}
function renderValid(){
 const V=DATA.validation,f=(v,n)=>v===null?'-':v.toFixed(n);
 $('valid').innerHTML='<table><tr><th>Group of plays</th><th>Plays</th><th>With PFF hit/hurry/sack</th><th>Share</th></tr>'
  +V.groups.map(g=>`<tr><td>${esc(g.group)}</td><td>${g.plays}</td><td>${g.pff_plays}</td><td>${f(g.pff_pct,0)}${g.pff_pct===null?'':'%'}</td></tr>`).join('')+'</table>'
  +`<p>Average closest rusher distance: ${f(V.avg_closest_pff,1)} yd on plays with a PFF pressure label vs ${f(V.avg_closest_nopff,1)} yd on plays without.</p>`;}
$('sort').onchange=buildOrder;$('filter').onchange=buildOrder;select.onchange=selectPlay;
slider.oninput=()=>{stopPlayback();syncPlayhead(Number(slider.value));render();};
$('play').onclick=startPlayback;
$('pause').onclick=stopPlayback;
$('reset').onclick=()=>{stopPlayback();syncPlayhead(0);render();};
$('stepPrev').onclick=()=>{if(!d)return;stopPlayback();syncPlayhead(Math.max(0,i-1));render();};
$('stepNext').onclick=()=>{if(!d)return;stopPlayback();syncPlayhead(Math.min(d.frames.length-1,i+1));render();};
$('speedSelect').onchange=()=>{playSpeed=Number($('speedSelect').value)||1.0;};
$('smoothToggle').onchange=()=>{smoothMode=$('smoothToggle').checked;slider.step=smoothMode?'0.02':'1';syncPlayhead(playhead);render();};
if($('distChartMode'))$('distChartMode').onchange=render;
['telMetricDist','telMetricSpeed','telMetricAccel','telMetricGap','telMetricClosing'].forEach(id=>{if($(id))$(id).onchange=render;});
$('cmpEnable').onchange=render;
$('cmpTrails').onchange=render;
$('cmpVectors').onchange=render;
$('playerASelect').onchange=()=>{playerAId=Number($('playerASelect').value);render();};
$('playerBSelect').onchange=()=>{playerBId=Number($('playerBSelect').value);render();};
$('swapPlayers').onclick=()=>{const tmp=playerAId;playerAId=playerBId;playerBId=tmp;$('playerASelect').value=String(playerAId);$('playerBSelect').value=String(playerBId);render();};
$('presetClosestPair').onclick=()=>{if(!d||!d.telemetry)return;playerAId=d.telemetry.default_player_a_id;playerBId=d.telemetry.default_player_b_id;$('playerASelect').value=String(playerAId);$('playerBSelect').value=String(playerBId);render();};
$('presetVlmVsGt').onclick=()=>compareVlmVsTracking(true);
renderVLMErrorSummary();renderGame();renderValid();buildOrder();initCrossPlay();initMultiPlayAnalytics();initWorkspaceNav();
</script></body></html>"""


def build_replay(data_folder, tracking_id, output_folder, run_vlm=False, vlm_play_ids=None,
                 max_vlm_plays=5, num_vlm_frames=4, vlm_model=None, analyzer=None, progress_callback=None):
    global np, pd
    import numpy as np
    import pandas as pd
    from telemetry_comparison import (
        build_compact_cross_game_play_payload,
        compute_multi_play_player_analytics,
        extract_play_rusher_records,
    )
    global games, plays, players, scouting, tracking, game_plays, GAME_ID
    primary_game_id = int(tracking_id)
    GAME_ID = primary_game_id
    tracking_path = data_folder / "tracking" / f"tracking_{GAME_ID}.csv"
    if not tracking_path.is_file():
        raise FileNotFoundError(f"Tracking file not found: {tracking_path.name}")
    games = pd.read_csv(data_folder / "games.csv")
    plays = pd.read_csv(data_folder / "plays.csv")
    players = pd.read_csv(data_folder / "players.csv")
    scouting = pd.read_csv(data_folder / "pffScoutingData.csv")
    if plays.duplicated(["gameId", "playId"]).any() or scouting.duplicated(["gameId", "playId", "nflId"]).any():
        raise ValueError("Duplicate source keys detected.")
    tracking = pd.read_csv(tracking_path)
    game_plays = plays.loc[plays.gameId.eq(GAME_ID)].copy()
    payloads, rejected = [], []
    for play_id in game_plays.playId:
        try:
            payload, _, _ = prepare_play(int(play_id))
            payloads.append(payload)
        except ValueError as error:
            rejected.append({"playId": int(play_id), "reason": str(error)})
    if not payloads:
        raise ValueError("No supported plays found in this game.")

    # Priority 5: whole-game rusher table
    agg = {}
    for p in payloads:
        for r in p["rusher_stats"]:
            a = agg.setdefault(r["id"], {"id": r["id"], "name": r["name"], "plays": 0,
                                         "closest": [], "within": 0, "arrivals": [], "pff_plays": 0})
            a["plays"] += 1
            a["closest"].append(r["closest_yards"])
            if r["first_within"] is not None:
                a["within"] += 1
                a["arrivals"].append(r["first_within"])
            if r["pff_pressure"]:
                a["pff_plays"] += 1
    rusher_rows = [{"id": a["id"], "name": a["name"], "plays": a["plays"],
                    "avg_closest": mean_or_none(a["closest"]),
                    "pct_within": 100.0 * a["within"] / a["plays"],
                    "avg_arrival": mean_or_none(a["arrivals"]),
                    "pff_plays": a["pff_plays"]} for a in agg.values()]

    # Priority 6: does "within 2 yd" line up with PFF labels?
    def group(name, ps):
        n = len(ps)
        k = sum(1 for p in ps if p["pff_pressure"])
        return {"group": name, "plays": n, "pff_plays": k, "pff_pct": None if n == 0 else 100.0 * k / n}
    near = [p for p in payloads if p["summary"]["first_pressure"] is not None]
    far = [p for p in payloads if p["summary"]["first_pressure"] is None]
    validation = {
        "groups": [group("A rusher got within %g yd of the QB" % PRESSURE_YD, near),
                   group("No rusher got within %g yd" % PRESSURE_YD, far)],
        "avg_closest_pff": mean_or_none(p["summary"]["closest_yards"] for p in payloads if p["pff_pressure"]),
        "avg_closest_nopff": mean_or_none(p["summary"]["closest_yards"] for p in payloads if not p["pff_pressure"]),
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    vlm_summary = None
    try:
        from vlm_inference import DEFAULT_MODEL_ID
        from vlm_pipeline import attach_vlm_analysis_to_plays

        vlm_summary = attach_vlm_analysis_to_plays(
            payloads=payloads,
            output_folder=output_folder,
            game_id=GAME_ID,
            run_vlm=bool(run_vlm),
            vlm_play_ids=vlm_play_ids,
            max_vlm_plays=max_vlm_plays,
            num_frames=num_vlm_frames,
            model_id=vlm_model or DEFAULT_MODEL_ID,
            analyzer=analyzer,
            progress_callback=progress_callback,
        )
    except Exception as vlm_err:
        vlm_summary = {"total_plays": 0, "error": str(vlm_err)}

    # Build Cross-Game & Multi-Play Analytics bundles across local tracking_*.csv files
    def _game_meta(gid: int, play_count: int, embedded: bool) -> dict:
        row = games.loc[games.gameId.eq(gid)]
        if len(row):
            r0 = row.iloc[0]
            vis = str(r0["visitorTeamAbbr"]) if "visitorTeamAbbr" in r0.index and pd.notna(r0["visitorTeamAbbr"]) else ""
            home = str(r0["homeTeamAbbr"]) if "homeTeamAbbr" in r0.index and pd.notna(r0["homeTeamAbbr"]) else ""
            gdate = str(r0["gameDate"]) if "gameDate" in r0.index and pd.notna(r0["gameDate"]) else ""
            wk = int(r0["week"]) if "week" in r0.index and pd.notna(r0["week"]) else None
            lbl = f"{gid} ({vis} @ {home}{f', Wk {wk}' if wk is not None else ''} — {play_count} plays)"
            return {"game_id": gid, "visitor": vis, "home": home, "date": gdate, "week": wk,
                    "play_count": play_count, "embedded": embedded,
                    "bundle_js": f"cross_game_bundle_{gid}.js", "label": lbl}
        return {"game_id": gid, "visitor": "", "home": "", "date": "", "week": None,
                "play_count": play_count, "embedded": embedded,
                "bundle_js": f"cross_game_bundle_{gid}.js", "label": f"{gid} ({play_count} plays)"}

    primary_compact = [build_compact_cross_game_play_payload(p) for p in payloads]
    cross_game_plays = {str(primary_game_id): primary_compact}
    available_games = [_game_meta(primary_game_id, len(primary_compact), True)]
    all_compact_for_analytics = list(primary_compact)

    cache_dir = BASE / "qb_pressure_outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    primary_cache_path = cache_dir / f"cross_game_cache_{primary_game_id}.json"
    try:
        primary_cache_path.write_text(json.dumps(primary_compact, allow_nan=False), encoding="utf-8")
    except Exception:
        pass

    saved_tracking = tracking
    saved_game_plays = game_plays
    extra_tracking_files = sorted((data_folder / "tracking").glob("tracking_*.csv"))
    extra_embedded_count = 0
    for t_file in extra_tracking_files:
        stem = t_file.stem.replace("tracking_", "")
        if not stem.isdigit():
            continue
        gid = int(stem)
        if gid == primary_game_id:
            continue
        cache_path = cache_dir / f"cross_game_cache_{gid}.json"
        sec_compact = None
        if cache_path.is_file() and cache_path.stat().st_mtime >= t_file.stat().st_mtime:
            try:
                sec_compact = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                sec_compact = None
        if sec_compact is None:
            try:
                GAME_ID = gid
                tracking = pd.read_csv(t_file)
                game_plays = plays.loc[plays.gameId.eq(gid)].copy()
                sec_compact = []
                for pid in game_plays.playId:
                    try:
                        p_sec, _, _ = prepare_play(int(pid))
                        sec_compact.append(build_compact_cross_game_play_payload(p_sec))
                    except ValueError:
                        pass
                cache_path.write_text(json.dumps(sec_compact, allow_nan=False), encoding="utf-8")
            except Exception:
                sec_compact = []
        if sec_compact:
            bundle_js_path = output_folder / f"cross_game_bundle_{gid}.js"
            try:
                js_body = (
                    "window.__QB_CROSS_GAME_BUNDLES=window.__QB_CROSS_GAME_BUNDLES||{};"
                    f"window.__QB_CROSS_GAME_BUNDLES['{gid}']="
                    + json.dumps(sec_compact, allow_nan=False).replace("<", "\\u003c")
                    + ";"
                )
                bundle_js_path.write_text(js_body, encoding="utf-8")
            except Exception:
                pass
            embed_inline = extra_embedded_count < 2
            if embed_inline:
                cross_game_plays[str(gid)] = sec_compact
                extra_embedded_count += 1
            available_games.append(_game_meta(gid, len(sec_compact), embed_inline))
            all_compact_for_analytics.extend(sec_compact)

    GAME_ID = primary_game_id
    tracking = saved_tracking
    game_plays = saved_game_plays

    multi_play_records = extract_play_rusher_records(all_compact_for_analytics, pressure_threshold_yd=PRESSURE_YD)
    multi_play_default_analytics = compute_multi_play_player_analytics(
        multi_play_records, filters=None, min_plays=1
    )

    data = {
        "plays": payloads,
        "rushers": rusher_rows,
        "validation": validation,
        "game": GAME_ID,
        "threshold": PRESSURE_YD,
        "vlm_summary": vlm_summary,
        "available_games": available_games,
        "cross_game_plays": cross_game_plays,
        "multi_play_records": multi_play_records,
        "multi_play_default_analytics": multi_play_default_analytics,
    }
    document = (HTML_TEMPLATE
                .replace("__TH__", "%g" % PRESSURE_YD)
                .replace("__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c")))
    path = output_folder / f"QB_Replay_{GAME_ID}.html"
    path.write_text(document, encoding="utf-8")
    pd.DataFrame(rejected, columns=["playId", "reason"]).to_csv(
        output_folder / f"excluded_{GAME_ID}.csv", index=False
    )
    return path, len(payloads)


def open_file(path):
    """Open the HTML with the default program. os.startfile is more reliable than webbrowser on Windows/Edge."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path.resolve()))
        else:
            webbrowser.open(path.resolve().as_uri())
        return True
    except Exception:
        return False


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    print("Starting QB Pressure Replay window...", flush=True)
    root = tk.Tk()
    root.title("QB Pressure Replay")
    root.geometry("620x350")
    root.minsize(540, 320)
    root.lift()
    root.attributes("-topmost", True)
    root.after(1500, lambda: root.attributes("-topmost", False))
    folder = find_data_folder()
    folder_var = tk.StringVar(value=str(folder) if folder else "")
    id_var = tk.StringVar(value="2021090900")
    vlm_var = tk.BooleanVar(value=False)
    vlm_count_var = tk.StringVar(value="3")
    status_var = tk.StringVar(value="Enter a tracking ID, then click Open Replay.")
    ttk.Label(root, text="QB Pressure Replay", font=("Arial", 18, "bold")).pack(pady=10)
    ttk.Label(root, text="Data folder (automatically detected when placed beside the script)").pack()
    row = ttk.Frame(root)
    row.pack(fill="x", padx=18, pady=6)
    ttk.Entry(row, textvariable=folder_var).pack(side="left", fill="x", expand=True)

    def browse():
        chosen = filedialog.askdirectory(title="Select the folder containing the four CSV files and tracking")
        if chosen:
            folder_var.set(chosen)
    ttk.Button(row, text="Browse", command=browse).pack(side="right", padx=5)
    ttk.Label(root, text="Tracking ID or filename").pack(pady=(6, 0))
    entry = ttk.Entry(root, textvariable=id_var, width=35)
    entry.pack(pady=6)
    entry.focus_set()

    vlm_row = ttk.Frame(root)
    vlm_row.pack( pady=2)
    ttk.Checkbutton(vlm_row, text="Run optional VLM visual analysis (max plays:", variable=vlm_var).pack(side="left")
    ttk.Entry(vlm_row, textvariable=vlm_count_var, width=4).pack(side="left", padx=2)
    ttk.Label(vlm_row, text=")").pack(side="left")

    ttk.Label(root, textvariable=status_var, wraplength=580).pack(pady=6)
    results = []

    def finish():
        if not results:
            root.after(100, finish)
            return
        ok, result = results.pop()
        button.config(state="normal")
        if ok:
            path, count = result
            status_var.set(f"Ready: {count} plays. Opening replay in your browser.")
            if not open_file(path):
                messagebox.showinfo("Replay saved", f"Open this file in a browser:\n{path}")
        else:
            status_var.set("Unable to generate replay. Check the folder and tracking ID.")
            messagebox.showerror("Replay error", str(result))

    def run():
        data = Path(folder_var.get())
        if not is_data_folder(data):
            messagebox.showerror("Data folder", "Choose a folder containing games.csv, plays.csv, players.csv, pffScoutingData.csv and tracking.")
            return
        text = clean_id(id_var.get())
        if not text.isdigit():
            messagebox.showerror("Tracking ID", "Enter a numeric tracking ID, for example 2021090900.")
            return
        max_plays = int(vlm_count_var.get()) if vlm_count_var.get().strip().isdigit() else 3
        button.config(state="disabled")
        status_var.set("Loading and validating plays. Please wait...")

        def update_status(msg):
            root.after(0, lambda: status_var.set(msg))

        def worker():
            try:
                results.append((True, build_replay(
                    data, text, BASE / "qb_pressure_outputs",
                    run_vlm=bool(vlm_var.get()), max_vlm_plays=max_plays,
                    progress_callback=update_status,
                )))
            except Exception as error:
                results.append((False, error))
        threading.Thread(target=worker, daemon=True).start()
        root.after(100, finish)
    button = ttk.Button(root, text="Open Replay", command=run)
    button.pack(pady=5)
    root.bind("<Return>", lambda event: run() if str(button["state"]) != "disabled" else None)
    root.mainloop()


def run_cli(args):
    data = Path(args.data) if args.data else find_data_folder()
    if data is None or not is_data_folder(data):
        sys.exit("Data folder not found. Use --data to point to the folder with games.csv, plays.csv, "
                 "players.csv, pffScoutingData.csv and tracking/.")
    text = clean_id(args.id)
    if not text.isdigit():
        sys.exit("--id must be numeric, for example 2021090900.")
    out = Path(args.out) if args.out else BASE / "qb_pressure_outputs"
    vlm_play_ids = None
    if getattr(args, "vlm_plays", None):
        vlm_play_ids = [int(x.strip()) for x in args.vlm_plays.split(",") if x.strip().isdigit()]
    path, count = build_replay(
        data,
        text,
        out,
        run_vlm=bool(getattr(args, "vlm", False) or vlm_play_ids),
        vlm_play_ids=vlm_play_ids,
        max_vlm_plays=int(getattr(args, "vlm_max_plays", 5)),
        num_vlm_frames=int(getattr(args, "vlm_frames", 4)),
        vlm_model=getattr(args, "vlm_model", None),
        progress_callback=lambda msg: print(msg, flush=True),
    )
    print(f"Done: {count} plays written to {path}")
    if not args.no_open and not open_file(path):
        print("Could not open a browser automatically. Open the HTML file manually.")


def main():
    parser = argparse.ArgumentParser(description="QB Pressure Replay")
    parser.add_argument("--id", help="tracking/game ID, e.g. 2021090900 (skips the window)")
    parser.add_argument("--data", help="data folder (default: auto-detect)")
    parser.add_argument("--out", help="output folder (default: qb_pressure_outputs next to the script)")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument("--vlm", action="store_true", help="run optional VLM visual analysis on selected plays")
    parser.add_argument("--vlm-plays", help="comma-separated play IDs to analyze with VLM, e.g. 55,97,137")
    parser.add_argument("--vlm-max-plays", type=int, default=5, help="max plays to analyze when --vlm is enabled")
    parser.add_argument("--vlm-frames", type=int, default=4, help="sequential frames per play for VLM input")
    parser.add_argument("--vlm-model", default=None, help="VLM checkpoint ID (default: Qwen/Qwen2.5-VL-3B-Instruct)")
    args = parser.parse_args()
    if args.id:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()