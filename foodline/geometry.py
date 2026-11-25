"""Small geometry helpers shared by calibration, counting and drawing."""
from __future__ import annotations

import cv2
import numpy as np


def to_px(points, w: int, h: int) -> np.ndarray:
    return np.array([[p[0] * w, p[1] * h] for p in points], dtype=np.float32)


def zone_mask(zone, w: int, h: int) -> np.ndarray | None:
    if not zone or len(zone) < 3:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [to_px(zone, w, h).astype(np.int32)], 255)
    return mask


def side(p, a, b) -> int:
    """Which side of the line a→b point p is on: +1, -1, or 0 on the line."""
    c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return 1 if c > 0 else (-1 if c < 0 else 0)


def along(p, a, b) -> float:
    """Projection of p onto a→b as a fraction of its length (0 at a, 1 at b)."""
    ab = np.subtract(b, a)
    denom = float(ab @ ab) or 1.0
    return float(np.subtract(p, a) @ ab) / denom


def resize_to_width(frame, width: int):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
