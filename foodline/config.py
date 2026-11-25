"""Load a plant config (YAML) plus its calibration sidecar.

A config describes one line: crew size, wage, cameras and where their footage
lives. Calibration (count line and belt zone per camera, drawn with
`run.py calibrate`) is stored next to it in `<config>.calib.yaml`, so
re-calibrating never rewrites the hand-edited config or its comments.

All zone coordinates are normalized to 0..1 of the frame width/height, so they
survive resizing and different processing resolutions.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "plant": "Unnamed plant",
    "line": "Line 1",
    "crew_size": 4,                 # people on the line (for $ lost)
    "loaded_rate_per_hour": 27.0,   # wage + benefits + overhead, $/hr per person
    "shift_hours": 8,
    "work_days_per_year": 250,
    "currency": "$",
    "process": {
        "fps": 5,                   # frames analysed per second of video
        "width": 960,               # frames are resized to this width first
    },
    "counting": {
        "min_area_frac": 0.002,     # smallest blob that can be a tray (fraction of frame)
        "max_area_frac": 0.25,      # largest blob (bigger = lighting change, ignore)
        "match_dist_frac": 0.15,    # max move between analysed frames (fraction of width)
        "max_missed": 3,            # frames a track may vanish before it is dropped
        "mog_history": 300,
        "mog_var_threshold": 32,
        "merge_ratio": 1.6,         # blob > 1.6x a normal tray = several trays touching
    },
    "stoppage": {
        "gap_multiple": 2.0,        # gap > 2x normal cycle time is a stoppage...
        "min_gap_s": 15,            # ...and never shorter than this
        "long_stop_s": 300,         # gaps longer than this are flagged "long stop / changeover?"
    },
    "cameras": {},
}

CAUSES = [
    "empty bin",
    "short-staffed",
    "jam",
    "belt stopped",
    "changeover",
    "quality check",
    "break",
    "other",
]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def calib_path(config_path: Path) -> Path:
    config_path = Path(config_path)
    return config_path.with_name(config_path.stem + ".calib.yaml")


def load(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg["_path"] = config_path
    cfg["_name"] = config_path.stem

    calib = {}
    cp = calib_path(config_path)
    if cp.exists():
        with open(cp) as f:
            calib = yaml.safe_load(f) or {}

    if not cfg["cameras"]:
        raise ValueError(f"{config_path.name}: no cameras defined")
    for name, cam in cfg["cameras"].items():
        cam.setdefault("role", "end")
        footage = Path(cam["footage"])
        if not footage.is_absolute():
            footage = (config_path.parent / footage).resolve()
        cam["footage"] = footage
        cam["calib"] = calib.get(name, {})
    return cfg


def save_calibration(config_path: str | Path, camera: str, data: dict) -> Path:
    cp = calib_path(Path(config_path))
    calib = {}
    if cp.exists():
        with open(cp) as f:
            calib = yaml.safe_load(f) or {}
    calib[camera] = data
    with open(cp, "w") as f:
        f.write("# Written by `run.py calibrate`. Coordinates are 0..1 of frame width/height.\n")
        yaml.safe_dump(calib, f, sort_keys=False, default_flow_style=None)
    return cp


def count_camera(cfg: dict) -> str:
    """The camera whose count is the line's throughput: role 'end', else the first one."""
    for name, cam in cfg["cameras"].items():
        if cam.get("role") == "end":
            return name
    return next(iter(cfg["cameras"]))


def run_dir(cfg: dict, day: str) -> Path:
    d = cfg["_path"].parent.parent / "runs" / cfg["_name"] / day
    d.mkdir(parents=True, exist_ok=True)
    return d
