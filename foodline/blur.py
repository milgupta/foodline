"""Privacy blur for any frame that leaves the team.

Default: everything outside the belt zone is heavily blurred, so workers are
unrecognisable and only trays on the belt stay sharp. If an OpenCV YuNet face
model is present at models/face_detection_yunet.onnx, faces inside the zone are
blurred too (download it from the opencv_zoo GitHub repo; it's not bundled).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import zone_mask

YUNET = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet.onnx"
_detector = None


def _faces(img):
    global _detector
    if not YUNET.exists() or not hasattr(cv2, "FaceDetectorYN"):
        return []
    h, w = img.shape[:2]
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.6)
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(img)
    return [] if faces is None else [f[:4].astype(int) for f in faces]


def _heavy_blur(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 3)


def privacy_blur(img, belt_zone=None):
    h, w = img.shape[:2]
    out = img.copy()
    mask = zone_mask(belt_zone, w, h)
    if mask is not None:
        soft = cv2.GaussianBlur(mask, (0, 0), 4)[..., None] / 255.0
        out = (out * soft + _heavy_blur(img) * (1 - soft)).astype(np.uint8)
    for x, y, fw, fh in _faces(img):
        x0, y0 = max(0, x - fw // 4), max(0, y - fh // 4)
        x1, y1 = min(w, x + fw + fw // 4), min(h, y + fh + fh // 4)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = _heavy_blur(out[y0:y1, x0:x1])
    return out
