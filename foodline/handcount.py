"""Hand-count trays to check the camera's count.

`run.py handcount`: plays the count camera's footage with the count line drawn.
  space = a tray just crossed      b = undo last      p = pause
  ] / [ = faster / slower          q = save and quit
Saves handcount.csv (+ the window watched) in the run folder.

`run.py evaluate`: compares the camera count with the hand count over the
watched window and writes eval.json, which the report quotes.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config as config_mod
from .geometry import resize_to_width, to_px
from .ingest import list_segments
from .pipeline import latest_run


def play(config_path, minutes: float | None = None, start_s: float = 0.0) -> Path:
    cfg = config_mod.load(config_path)
    cam = cfg["cameras"][config_mod.count_camera(cfg)]
    segs = list_segments(cam["footage"], cam.get("start"))
    rdir = config_mod.run_dir(cfg, f"{segs[0].start:%Y-%m-%d}")
    line = cam["calib"].get("count_line")

    marks: list[datetime] = []
    speed, paused = 1.0, False
    watched_from = segs[0].start + timedelta(seconds=start_s)
    watched_to = watched_from
    limit = watched_from + timedelta(minutes=minutes) if minutes else None
    win = "Hand count - space = tray crossed, q = save"
    quit_all = False
    for si, seg in enumerate(segs):
        cap = cv2.VideoCapture(str(seg.path))
        if si == 0 and start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        frame = None
        while True:
            t_wall = time.time()
            if not paused or frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                now = seg.start + timedelta(seconds=pos)
                watched_to = max(watched_to, now)
            img = resize_to_width(frame, 1100)
            h, w = img.shape[:2]
            if line:
                a, b = to_px(line, w, h)
                cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 2)
            cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(img, f"{now:%H:%M:%S}  counted {len(marks)}  speed {speed:.2g}x{'  PAUSED' if paused else ''}",
                        (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, img)
            delay = max(1, int(1000 / (seg.fps * speed) - (time.time() - t_wall) * 1000))
            k = cv2.waitKey(delay) & 0xFF
            if k == ord(" "):
                marks.append(now)
            elif k == ord("b") and marks:
                marks.pop()
            elif k == ord("p"):
                paused = not paused
            elif k == ord("]"):
                speed = min(4.0, speed * 1.5)
            elif k == ord("["):
                speed = max(0.25, speed / 1.5)
            elif k == ord("q"):
                quit_all = True
                break
            if limit and now >= limit:
                quit_all = True
                break
        cap.release()
        if quit_all:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    pd.DataFrame({"time": marks}).to_csv(rdir / "handcount.csv", index=False)
    (rdir / "handcount.json").write_text(json.dumps(
        {"from": watched_from.isoformat(), "to": watched_to.isoformat()}))
    print(f"Hand count: {len(marks)} trays, {watched_from:%H:%M:%S}–{watched_to:%H:%M:%S} → {rdir / 'handcount.csv'}")
    return rdir


def compare(auto_times, hand_times, tolerance_s: float = 2.0) -> int:
    """Greedy one-to-one matching of hand marks to camera counts within a tolerance."""
    auto = sorted(auto_times)
    used = np.zeros(len(auto), bool)
    matched = 0
    for h in sorted(hand_times):
        best, best_d = None, tolerance_s
        for i, a in enumerate(auto):
            if used[i]:
                continue
            d = abs((a - h).total_seconds())
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            used[best] = True
            matched += 1
    return matched


def evaluate(config_path, truth: str | None = None, tolerance_s: float = 2.0, run: str | None = None) -> dict:
    cfg = config_mod.load(config_path)
    rdir = Path(run) if run else latest_run(cfg)
    trays = pd.read_csv(rdir / "trays.csv", parse_dates=["time"])
    auto = [t for t, n in zip(trays["time"], trays["n"]) for _ in range(int(n))]

    if truth:
        hand = pd.read_csv(truth, parse_dates=["time"])["time"].tolist()
        lo, hi = None, None
        meta = json.loads((rdir / "meta.json").read_text())
        lo = datetime.fromisoformat(meta["spans"][0][0])
        hi = datetime.fromisoformat(meta["spans"][-1][1])
    else:
        hc = rdir / "handcount.csv"
        if not hc.exists():
            raise SystemExit("No hand count yet. Run `handcount` first, or pass --truth.")
        hand = pd.read_csv(hc, parse_dates=["time"])["time"].tolist()
        w = json.loads((rdir / "handcount.json").read_text())
        lo, hi = datetime.fromisoformat(w["from"]), datetime.fromisoformat(w["to"])

    # Ignore the last couple of seconds, where a press may land after the window closes.
    auto_w = [a for a in auto if lo <= a <= hi]
    hand_w = [h for h in hand if lo <= h <= hi]
    matched = compare(auto_w, hand_w, tolerance_s)
    hand_total, auto_total = len(hand_w), len(auto_w)
    err = 100.0 * (auto_total - hand_total) / hand_total if hand_total else float("nan")
    minutes = (hi - lo).total_seconds() / 60
    result = {
        "hand_total": hand_total, "auto_total": auto_total, "error_pct": round(err, 2),
        "matched": matched, "recall_pct": round(100 * matched / hand_total, 1) if hand_total else None,
        "tolerance_s": tolerance_s, "window": f"{minutes:.0f} min ({lo:%H:%M:%S}–{hi:%H:%M:%S})",
        "source": "ground truth file" if truth else "hand count",
        "passes_2pct": bool(hand_total and abs(err) <= 2.0),
    }
    (rdir / "eval.json").write_text(json.dumps(result, indent=2))
    print(f"Camera {auto_total} vs {'truth' if truth else 'hand'} {hand_total}: {err:+.1f}% "
          f"({'PASS' if result['passes_2pct'] else 'FAIL'} ±2%), {matched}/{hand_total} matched within {tolerance_s}s")
    return result
