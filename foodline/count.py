"""v0 tray counter: background subtraction + blob tracking + count line.

No trained model yet (that's Phase 1). Anything that moves inside the belt zone
and crosses the count line is counted. Works when the count line sits past the
last station, where only trays move. Blobs much bigger than a normal tray are
treated as several trays touching; that split happens later in events.py, once
the normal tray size is known from the whole video.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import along, resize_to_width, side, to_px, zone_mask
from .ingest import Segment


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    area: float
    side: int
    missed: int = 0
    counted: bool = False
    max_area: float = 0.0
    trail: list = field(default_factory=list)


@dataclass
class Crossing:
    t: float          # seconds from segment start
    track: int
    area_frac: float  # blob area / frame area, at its largest
    direction: int    # +1 / -1 relative to the count line


def _prime_background(cap, mog, width, mask, n=25):
    """Feed MOG2 a median of frames from across the video, so trays sitting on the
    belt in the first frame aren't learnt as background."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < n * 2:
        return
    frames = []
    for idx in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            frames.append(resize_to_width(f, width))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(frames) >= 5:
        bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
        for _ in range(10):
            mog.apply(bg, learningRate=1.0 if _ == 0 else 0.5)


def count_segment(seg: Segment, calib: dict, cfg: dict, progress=None, debug_writer=None) -> list[Crossing]:
    pc, cc = cfg["process"], cfg["counting"]
    if not calib.get("count_line"):
        raise ValueError("Camera has no count line. Run `python run.py calibrate <config>` first.")

    cap = cv2.VideoCapture(str(seg.path))
    if not cap.isOpened():
        raise IOError(f"Can't open {seg.path}")
    width = int(pc["width"])
    step = max(1, int(round(seg.fps / float(pc["fps"]))))

    ok, first = cap.read()
    if not ok:
        return []
    first = resize_to_width(first, width)
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    a, b = to_px(calib["count_line"], w, h)
    mask = zone_mask(calib.get("belt_zone"), w, h)
    frame_area = float(w * h)
    min_area = cc["min_area_frac"] * frame_area
    max_area = cc["max_area_frac"] * frame_area
    match_dist = cc["match_dist_frac"] * w
    k = max(3, (w // 80) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mog = cv2.createBackgroundSubtractorMOG2(
        history=int(cc["mog_history"]), varThreshold=float(cc["mog_var_threshold"]), detectShadows=True)
    _prime_background(cap, mog, width, mask)

    tracks: list[Track] = []
    crossings: list[Crossing] = []
    next_id = 1
    idx = -1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        idx += 1
        if idx % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / seg.fps
        frame = resize_to_width(frame, width)

        fg = mog.apply(frame)
        fg = np.where(fg > 200, 255, 0).astype(np.uint8)  # drop shadows (127)
        if mask is not None:
            fg = cv2.bitwise_and(fg, mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, open_kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                m = cv2.moments(c)
                if m["m00"]:
                    dets.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area, cv2.boundingRect(c)))

        # Greedy nearest-neighbour matching, closest pairs first.
        pairs = sorted(
            ((np.hypot(tr.cx - d[0], tr.cy - d[1]), ti, di)
             for ti, tr in enumerate(tracks) for di, d in enumerate(dets)),
            key=lambda x: x[0])
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if dist > match_dist or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, (cx, cy, area, _) = tracks[ti], dets[di]
            s = side((cx, cy), a, b)
            if (not tr.counted and s and tr.side and s != tr.side
                    and -0.1 <= along(((tr.cx + cx) / 2, (tr.cy + cy) / 2), a, b) <= 1.1):
                tr.counted = True
                # Size at the moment of crossing: a pair that separated earlier
                # must not be counted as a merged blob too.
                crossings.append(Crossing(t, tr.id, area / frame_area, s))
            tr.cx, tr.cy, tr.area, tr.missed = cx, cy, area, 0
            tr.max_area = max(tr.max_area, area)
            tr.side = s or tr.side
            tr.trail = (tr.trail + [(int(cx), int(cy))])[-12:]

        for ti, tr in enumerate(tracks):
            if ti not in used_t:
                tr.missed += 1
        tracks = [tr for tr in tracks if tr.missed <= cc["max_missed"]]
        for di, (cx, cy, area, _) in enumerate(dets):
            if di not in used_d:
                tracks.append(Track(next_id, cx, cy, area, side((cx, cy), a, b), max_area=area,
                                    trail=[(int(cx), int(cy))]))
                next_id += 1

        if debug_writer is not None:
            debug_writer.write(_debug_frame(frame, fg, a, b, mask, dets, tracks, len(crossings), seg, t))
        if progress and total:
            progress(min(1.0, idx / total))

    cap.release()
    return crossings


def _debug_frame(frame, fg, a, b, mask, dets, tracks, n, seg, t):
    img = frame.copy()
    if mask is not None:
        dim = (img * 0.45).astype(np.uint8)
        img = np.where(mask[..., None] > 0, img, dim)
    tint = np.zeros_like(img)
    tint[..., 1] = fg
    img = cv2.addWeighted(img, 1.0, tint, 0.35, 0)
    cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 2)
    for _, _, _, (x, y, w, h) in dets:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 1)
    for tr in tracks:
        col = (0, 220, 0) if tr.counted else (255, 255, 255)
        if len(tr.trail) > 1:
            cv2.polylines(img, [np.array(tr.trail, np.int32)], False, col, 2)
        cv2.putText(img, str(tr.id), (int(tr.cx) + 6, int(tr.cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    stamp = (seg.start.timestamp() + t)
    from datetime import datetime
    label = f"{datetime.fromtimestamp(stamp):%H:%M:%S}   trays counted: {n}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img
