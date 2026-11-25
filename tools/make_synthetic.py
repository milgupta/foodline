#!/usr/bin/env python3
"""Make a synthetic overhead food-line video with known ground truth.

Trays ride a belt left to right past three stations with workers whose arms
reach onto the belt. Staged events (times in seconds):
   80–112  empty bin: no trays placed           → stoppage
  190–230  belt stopped (trays freeze in place)  → stoppage
  ~290     a 9 s gap                             → NOT a stoppage (under 15 s)
  4 times  two trays placed touching             → counted as 2 from blob size

Writes footage/synthetic/cam3_20261128_140000.mp4 and truth.csv (the time each
tray's centre crossed x = 82% of the frame, where the count line goes).

  python tools/make_synthetic.py [--minutes 6] [--seed 7]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

W, H, FPS = 1280, 720, 15
BELT_Y0, BELT_Y1 = 250, 470
SPEED = 110.0            # px/s while the belt runs
TRAY_W, TRAY_H = 140, 100
COUNT_X = 0.82 * W
START = datetime(2026, 11, 28, 14, 0, 0)
EMPTY_BIN = (80, 112)
BELT_STOP = (190, 230)
SHORT_GAP = (288, 294)
STATIONS = [250, 450, 650]
FOODS = [(235, 235, 240), (60, 160, 70), (40, 90, 150), (30, 120, 200), (90, 60, 140), (70, 180, 220)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "footage" / "synthetic"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cam3_{START:%Y%m%d_%H%M%S}.mp4"
    duration = a.minutes * 60

    # Spawn plan: (time, n_touching)
    spawns, t, pair_at = [], 2.0, {40, 140, 250, 320}
    while t < duration:
        in_bin = EMPTY_BIN[0] <= t < EMPTY_BIN[1]
        in_stop = BELT_STOP[0] <= t < BELT_STOP[1]
        in_short = SHORT_GAP[0] <= t < SHORT_GAP[1]
        if not (in_bin or in_stop or in_short):
            pair = any(abs(t - p) < 1.6 for p in pair_at) and not any(abs(s[0] - t) < 30 and s[1] == 2 for s in spawns)
            spawns.append((t, 2 if pair else 1))
        t += 3.0 + rng.uniform(-0.3, 0.3)

    # Base scene
    floor = np.full((H, W, 3), (70, 75, 80), np.uint8)
    cv2.rectangle(floor, (0, BELT_Y0 - 12), (W, BELT_Y0), (150, 150, 155), -1)
    cv2.rectangle(floor, (0, BELT_Y1), (W, BELT_Y1 + 12), (150, 150, 155), -1)
    floor = cv2.add(floor, rng.integers(0, 12, (H, W, 3), dtype=np.uint8))

    trays = []      # dict(x, y, foods, crossed)
    truth = []
    belt_pos = 0.0
    spawn_i = 0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n_frames = int(duration * FPS)
    workers = [(sx, side, rng.uniform(0, 6.28)) for sx in STATIONS for side in (-1, 1)]

    for f in range(n_frames):
        t = f / FPS
        running = not (BELT_STOP[0] <= t < BELT_STOP[1])
        dx = SPEED / FPS if running else 0.0
        belt_pos += dx

        while spawn_i < len(spawns) and spawns[spawn_i][0] <= t:
            _, n = spawns[spawn_i]
            for k in range(n):
                trays.append({
                    "x": -TRAY_W / 2 - k * TRAY_W,   # touching trays: no gap
                    "y": (BELT_Y0 + BELT_Y1) / 2 + rng.uniform(-12, 12),
                    "foods": [FOODS[i] for i in rng.choice(len(FOODS), 3, replace=False)],
                    "crossed": False,
                })
            spawn_i += 1

        frame = floor.copy()
        belt = np.full((BELT_Y1 - BELT_Y0, W, 3), (196, 188, 172), np.int16)
        stripes = ((np.arange(W) + belt_pos) % 64 < 3) * 5
        belt += stripes[None, :, None].astype(np.int16)
        frame[BELT_Y0:BELT_Y1] = np.clip(belt, 0, 255).astype(np.uint8)

        for tr in trays:
            tr["x"] += dx
            if not tr["crossed"] and tr["x"] >= COUNT_X:
                tr["crossed"] = True
                truth.append(START + timedelta(seconds=t))
            x0, y0 = int(tr["x"] - TRAY_W / 2), int(tr["y"] - TRAY_H / 2)
            cv2.rectangle(frame, (x0, y0), (x0 + TRAY_W, y0 + TRAY_H), (55, 58, 62), -1)
            cv2.rectangle(frame, (x0, y0), (x0 + TRAY_W, y0 + TRAY_H), (35, 35, 38), 2)
            cw = (TRAY_W - 16) // 3
            for i, col in enumerate(tr["foods"]):
                cx0 = x0 + 6 + i * (cw + 2)
                cv2.rectangle(frame, (cx0, y0 + 8), (cx0 + cw, y0 + TRAY_H - 8), col, -1)
        trays = [tr for tr in trays if tr["x"] < W + TRAY_W]

        # Workers: body off the belt, arm reaching onto its edge.
        for sx, side, ph in workers:
            by = BELT_Y0 - 70 if side < 0 else BELT_Y1 + 70
            cv2.ellipse(frame, (sx, by), (55, 40), 0, 0, 360, (120, 130, 170), -1)
            cv2.circle(frame, (sx, by + side * 28), 24, (90, 120, 180), -1)
            reach = 45 * (0.5 + 0.5 * np.sin(t * 2.2 + ph))
            ax = int(sx + 40 * np.sin(t * 1.3 + ph))
            ay = int((BELT_Y0 + reach) if side < 0 else (BELT_Y1 - reach))
            cv2.line(frame, (sx, by), (ax, ay), (110, 150, 205), 18)

        noise = rng.normal(0, 3, frame.shape) + rng.uniform(-2, 2)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    pd.DataFrame({"time": truth}).to_csv(out / "truth.csv", index=False)
    print(f"Wrote {path} ({duration / 60:.1f} min, {len(truth)} trays crossed the count line)")
    print(f"Ground truth: {out / 'truth.csv'}")


if __name__ == "__main__":
    main()
