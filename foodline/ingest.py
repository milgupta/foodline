"""Turn a camera's footage (one file or a folder of segments) into a timeline.

Wall-clock start time per file, in order of preference:
  1. an explicit start (config `start:` or `--start`), for phone videos
  2. the recorder's filename pattern, e.g. cam1_20261222_060000.mp4
  3. the file's creation_time metadata (phones write this, in UTC)
  4. the file's modification time
When an explicit start is given for a folder, later files are assumed to follow
the first back to back.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
FNAME_RE = re.compile(r"(\d{8})_(\d{6})")


@dataclass
class Segment:
    path: Path
    start: datetime
    duration_s: float
    fps: float
    width: int
    height: int

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


def probe(path: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Can't open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or fps > 240:
        fps = 30.0
    duration = frames / fps if frames > 0 else _ffprobe_duration(path)
    return fps, duration, w, h


def _ffprobe(path: Path, entries: str) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_entries", entries, str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return json.loads(out.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def _ffprobe_duration(path: Path) -> float:
    d = _ffprobe(path, "format=duration").get("format", {}).get("duration")
    return float(d) if d else 0.0


def _start_from_metadata(path: Path) -> datetime | None:
    tag = _ffprobe(path, "format_tags=creation_time").get("format", {}).get("tags", {}).get("creation_time")
    if not tag:
        return None
    try:
        dt = datetime.fromisoformat(tag.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().replace(tzinfo=None)  # local wall-clock time


def parse_start(text: str | datetime | None) -> datetime | None:
    if text is None or isinstance(text, datetime):
        return text
    return datetime.fromisoformat(str(text).strip())


def list_segments(footage: Path, start: str | datetime | None = None) -> list[Segment]:
    footage = Path(footage)
    if footage.is_dir():
        files = sorted(p for p in footage.iterdir() if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."))
    elif footage.exists():
        files = [footage]
    else:
        raise FileNotFoundError(f"Footage not found: {footage}")
    if not files:
        raise FileNotFoundError(f"No video files in {footage}")

    explicit = parse_start(start)
    segs: list[Segment] = []
    for i, p in enumerate(files):
        fps, dur, w, h = probe(p)
        if explicit is not None:
            t0 = explicit if i == 0 else segs[-1].end
        else:
            m = FNAME_RE.search(p.stem)
            if m:
                t0 = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            else:
                t0 = _start_from_metadata(p) or datetime.fromtimestamp(p.stat().st_mtime)
        segs.append(Segment(p, t0, dur, fps, w, h))
    segs.sort(key=lambda s: s.start)
    return segs


def read_frame_at(path: Path, seconds: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
