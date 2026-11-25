"""Draw the count line and belt zone for one camera.

Interactive (opens a window on a frame of the footage):
  1. click 2 points: the COUNT LINE, across the belt where finished trays pass
  2. click 3+ points around the BELT ZONE (only motion inside it is counted)
  keys:  Enter = next step / save    u = undo point    r = restart    q = quit without saving

Non-interactive (scripts, tests):  --line x1,y1,x2,y2  --zone x,y,x,y,...  (0..1 coordinates)
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config as config_mod
from .ingest import list_segments, read_frame_at

STEPS = [
    ("COUNT LINE: click 2 points across the belt where finished trays pass", 2),
    ("BELT ZONE: click around the belt (3+ points). Enter = save, Enter now = whole frame", 0),
]


def _draw(frame, line, zone, step):
    img = frame.copy()
    h, w = img.shape[:2]
    if len(zone) >= 2:
        pts = np.array([[x * w, y * h] for x, y in zone], np.int32)
        overlay = img.copy()
        if len(zone) >= 3:
            cv2.fillPoly(overlay, [pts], (255, 200, 0))
            img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.polylines(img, [pts], len(zone) >= 3, (255, 200, 0), 2)
    for x, y in zone:
        cv2.circle(img, (int(x * w), int(y * h)), 4, (255, 200, 0), -1)
    for x, y in line:
        cv2.circle(img, (int(x * w), int(y * h)), 6, (0, 0, 255), -1)
    if len(line) == 2:
        cv2.line(img, (int(line[0][0] * w), int(line[0][1] * h)),
                 (int(line[1][0] * w), int(line[1][1] * h)), (0, 0, 255), 3)
    msg = STEPS[step][0] if step < len(STEPS) else "Calibration: red = count line, blue = belt zone"
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(img, msg, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def interactive(frame) -> tuple[list, list] | None:
    disp_w = min(1280, frame.shape[1])
    frame = cv2.resize(frame, (disp_w, int(frame.shape[0] * disp_w / frame.shape[1])))
    h, w = frame.shape[:2]
    state = {"step": 0, "line": [], "zone": []}

    def on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        p = [round(x / w, 4), round(y / h, 4)]
        if state["step"] == 0 and len(state["line"]) < 2:
            state["line"].append(p)
            if len(state["line"]) == 2:
                state["step"] = 1
        elif state["step"] == 1:
            state["zone"].append(p)

    win = "Calibrate - foodline"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            cv2.imshow(win, _draw(frame, state["line"], state["zone"], state["step"]))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                return None
            if k == ord("r"):
                state.update(step=0, line=[], zone=[])
            if k == ord("u"):
                if state["step"] == 1 and state["zone"]:
                    state["zone"].pop()
                elif state["line"]:
                    state["line"].pop()
                    state["step"] = 0
            if k in (13, 10) and state["step"] == 1:
                if 0 < len(state["zone"]) < 3:
                    continue  # need 3+ points, or none
                return state["line"], state["zone"]
    finally:
        cv2.destroyWindow(win)
        cv2.waitKey(1)


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.replace(" ", "").split(",") if v]


def run(config_path, camera: str | None, at_s: float, line: str | None, zone: str | None) -> None:
    cfg = config_mod.load(config_path)
    camera = camera or config_mod.count_camera(cfg)
    if camera not in cfg["cameras"]:
        raise SystemExit(f"No camera '{camera}' in config. Cameras: {', '.join(cfg['cameras'])}")
    cam = cfg["cameras"][camera]
    seg = list_segments(cam["footage"], cam.get("start"))[0]
    frame = read_frame_at(seg.path, min(at_s, max(0.0, seg.duration_s - 1)))
    if frame is None:
        raise SystemExit(f"Couldn't read a frame from {seg.path}")

    if line:
        v = _floats(line)
        if len(v) != 4:
            raise SystemExit("--line needs x1,y1,x2,y2")
        line_pts = [v[0:2], v[2:4]]
        z = _floats(zone) if zone else []
        if z and (len(z) % 2 or len(z) < 6):
            raise SystemExit("--zone needs 3+ x,y pairs")
        zone_pts = [z[i:i + 2] for i in range(0, len(z), 2)]
    else:
        result = interactive(frame)
        if result is None:
            print("Not saved.")
            return
        line_pts, zone_pts = result

    data = {"count_line": line_pts, "belt_zone": zone_pts, "frame_size": [seg.width, seg.height]}
    path = config_mod.save_calibration(cfg["_path"], camera, data)
    preview = _draw(frame, line_pts, zone_pts, len(STEPS))
    out = path.with_name(f"{cfg['_name']}.{camera}.calib.jpg")
    cv2.imwrite(str(out), preview)
    print(f"Saved {camera} calibration → {path.name} (preview: {out.name})")
