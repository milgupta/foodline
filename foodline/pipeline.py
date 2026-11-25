"""`run.py process`: footage → trays.csv, per_minute.csv, stoppages.csv, thumbnails."""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import cv2
import pandas as pd

from . import config as config_mod
from . import events
from .blur import privacy_blur
from .count import count_segment
from .geometry import resize_to_width
from .ingest import list_segments, read_frame_at


def _progress(label):
    t0 = time.time()

    def show(frac):
        el = time.time() - t0
        eta = el / frac - el if frac > 0.01 else 0
        sys.stdout.write(f"\r  {label}: {frac * 100:5.1f}%  ({el:4.0f}s, ~{eta:4.0f}s left) ")
        sys.stdout.flush()
    return show


def _keep_reviewed(new: pd.DataFrame, old_path: Path) -> pd.DataFrame:
    """Re-running keeps causes/notes already typed in for the same stoppage."""
    new = new.copy()
    new["cause"], new["note"] = "", ""
    if not old_path.exists() or new.empty:
        return new
    old = pd.read_csv(old_path, parse_dates=["start"], keep_default_na=False)
    for i, row in new.iterrows():
        near = old[(old["start"] - row["start"]).abs() <= pd.Timedelta(seconds=3)]
        if len(near):
            new.at[i, "cause"] = near.iloc[0].get("cause", "")
            new.at[i, "note"] = near.iloc[0].get("note", "")
    return new


def run(config_path, start=None, debug_video=False) -> Path:
    cfg = config_mod.load(config_path)
    cam_name = config_mod.count_camera(cfg)
    cam = cfg["cameras"][cam_name]
    calib = cam["calib"]
    if not calib.get("count_line"):
        raise SystemExit(f"{cam_name} isn't calibrated. Run: python run.py calibrate {config_path}")

    segs = list_segments(cam["footage"], start or cam.get("start"))
    day = f"{segs[0].start:%Y-%m-%d}"
    out = config_mod.run_dir(cfg, day)
    total_s = sum(s.duration_s for s in segs)
    print(f"{cfg['plant']} · {cfg['line']} · {cam_name}: {len(segs)} file(s), "
          f"{total_s / 60:.1f} min of footage from {segs[0].start:%Y-%m-%d %H:%M}")

    writer = None
    by_seg = []
    for i, seg in enumerate(segs, 1):
        if debug_video and writer is None:
            ok_frame = read_frame_at(seg.path, 0)
            if ok_frame is not None:
                h, w = resize_to_width(ok_frame, int(cfg["process"]["width"])).shape[:2]
                writer = cv2.VideoWriter(str(out / f"debug_{cam_name}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), float(cfg["process"]["fps"]), (w, h))
        crossings = count_segment(seg, calib, cfg, _progress(f"[{i}/{len(segs)}] {seg.path.name}"), writer)
        print(f"→ {len(crossings)} crossings")
        by_seg.append((seg, crossings))
    if writer is not None:
        writer.release()

    trays = events.tray_table(by_seg, cfg["counting"]["merge_ratio"])
    spans = events.footage_spans(segs)
    pm = events.per_minute(trays, spans)
    stops = events.find_stoppages(trays, spans, cfg)
    stops = _keep_reviewed(stops, out / "stoppages.csv")

    trays.to_csv(out / "trays.csv", index=False)
    pm.to_csv(out / "per_minute.csv", index=False)
    stops.to_csv(out / "stoppages.csv", index=False)

    thumbs = out / "thumbs"
    thumbs.mkdir(exist_ok=True)
    seg_by_name = {s.path.name: s for s in segs}
    for _, s in stops.iterrows():
        seg = seg_by_name[s["segment"]]
        t = float(s["offset_s"]) + min(8.0, s["gap_s"] / 2)
        frame = read_frame_at(seg.path, min(t, seg.duration_s - 0.5))
        if frame is not None:
            frame = privacy_blur(resize_to_width(frame, 640), calib.get("belt_zone"))
            cv2.imwrite(str(thumbs / f"{s['id']}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

    meta = {
        "plant": cfg["plant"], "line": cfg["line"], "camera": cam_name, "day": day,
        "segments": [{"file": s.path.name, "start": s.start.isoformat(), "duration_s": round(s.duration_s, 1)}
                     for s in segs],
        "footage_s": round(total_s, 1),
        "spans": [[a.isoformat(), b.isoformat()] for a, b in spans],
        "trays": int(trays["n"].sum()) if len(trays) else 0,
        "merged_events": int((trays["n"] > 1).sum()) if len(trays) else 0,
        "cycle_s": stops.attrs.get("cycle_s", events.normal_cycle_s(trays)),
        "threshold_s": stops.attrs.get("threshold_s"),
        "tray_area_frac": trays.attrs.get("tray_area_frac"),
        "method": "v0: background subtraction + blob tracking (no trained model)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    lost = stops["lost_s"].sum() if len(stops) else 0.0
    print(f"\n{meta['trays']} trays · {len(stops)} stoppages · {lost / 60:.1f} min lost "
          f"(≈{cfg['currency']}{events.money(lost, cfg):,.0f})")
    print(f"Results in {out}")
    if len(stops):
        print("Next: open stoppages.csv, fill in the `cause` column "
              f"({', '.join(config_mod.CAUSES)}), then run `report`.")
    return out


def latest_run(cfg) -> Path:
    base = cfg["_path"].parent.parent / "runs" / cfg["_name"]
    runs = sorted(p for p in base.glob("*") if (p / "meta.json").exists()) if base.exists() else []
    if not runs:
        raise SystemExit(f"No processed footage for {cfg['_name']} yet. Run `process` first.")
    return runs[-1]
