"""From tray crossings to the numbers the report needs.

  trays      one row per counted event, with `n` trays (n > 1 = touching trays)
  per_minute trays counted in every minute of footage (zeros included)
  stoppages  every gap between trays longer than max(min_gap_s, gap_multiple x normal cycle)

"Lost time" for a stoppage is the gap minus one normal cycle: the time the line
would have spent on that tray anyway is not counted as lost.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def tray_table(crossings_by_seg, merge_ratio: float) -> pd.DataFrame:
    rows = []
    for seg, crossings in crossings_by_seg:
        for c in crossings:
            rows.append({
                "time": seg.start + timedelta(seconds=c.t),
                "segment": seg.path.name,
                "offset_s": round(c.t, 2),
                "track": c.track,
                "area_frac": c.area_frac,
                "direction": c.direction,
            })
    df = pd.DataFrame(rows, columns=["time", "segment", "offset_s", "track", "area_frac", "direction"])
    if df.empty:
        df["n"] = []
        return df
    df = df.sort_values("time").reset_index(drop=True)

    # Normal tray size = median of blobs that aren't obviously merged.
    areas = df["area_frac"].to_numpy()
    med = float(np.median(areas))
    single = areas[areas <= merge_ratio * med]
    tray_area = float(np.median(single)) if len(single) else med
    n = np.where(areas > merge_ratio * tray_area, np.clip(np.rint(areas / tray_area), 2, 4), 1)
    df["n"] = n.astype(int)
    df.attrs["tray_area_frac"] = tray_area
    return df


def footage_spans(segments) -> list[tuple]:
    """Merge back-to-back segments into continuous spans of footage."""
    spans = []
    for s in sorted(segments, key=lambda s: s.start):
        if spans and s.start <= spans[-1][1] + timedelta(seconds=2):
            spans[-1][1] = max(spans[-1][1], s.end)
        else:
            spans.append([s.start, s.end])
    return [tuple(x) for x in spans]


def per_minute(trays: pd.DataFrame, spans) -> pd.DataFrame:
    idx = []
    for a, b in spans:
        idx.extend(pd.date_range(pd.Timestamp(a).floor("min"), pd.Timestamp(b).ceil("min"), freq="min",
                                 inclusive="left"))
    idx = pd.DatetimeIndex(sorted(set(idx)))
    if trays.empty:
        return pd.DataFrame({"minute": idx, "trays": 0})
    counts = trays.set_index("time")["n"].resample("min").sum()
    out = counts.reindex(idx, fill_value=0).rename("trays").reset_index().rename(columns={"index": "minute"})
    out["trays"] = out["trays"].astype(int)
    return out


def normal_cycle_s(trays: pd.DataFrame) -> float:
    """Median seconds per tray while the line is running."""
    if len(trays) < 3:
        return float("nan")
    # Seconds since the first tray (don't use astype(int64): pandas 3 may store µs, not ns).
    t = (trays["time"] - trays["time"].iloc[0]).dt.total_seconds().to_numpy()
    gaps = np.diff(t) / trays["n"].to_numpy()[1:]
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if len(gaps) else float("nan")


def find_stoppages(trays: pd.DataFrame, spans, cfg: dict) -> pd.DataFrame:
    sc = cfg["stoppage"]
    cols = ["id", "start", "end", "gap_s", "lost_s", "kind", "segment", "offset_s"]
    cycle = normal_cycle_s(trays)
    if np.isnan(cycle):
        return pd.DataFrame(columns=cols)
    threshold = max(float(sc["min_gap_s"]), float(sc["gap_multiple"]) * cycle)

    rows = []
    times = trays["time"].tolist()
    for i in range(1, len(times)):
        a, b = times[i - 1], times[i]
        gap = (b - a).total_seconds()
        if gap <= threshold:
            continue
        # A gap that runs across missing footage isn't a stoppage we saw.
        if not any(s <= a and b <= e for s, e in spans):
            continue
        prev = trays.iloc[i - 1]
        rows.append({
            "start": a,
            "end": b,
            "gap_s": round(gap, 1),
            "lost_s": round(max(0.0, gap - cycle), 1),
            "kind": "long stop / changeover?" if gap >= sc["long_stop_s"] else "stoppage",
            "segment": prev["segment"],
            "offset_s": prev["offset_s"],
        })
    df = pd.DataFrame(rows, columns=[c for c in cols if c != "id"])
    df.insert(0, "id", [f"S{i + 1}" for i in range(len(df))])
    df.attrs.update(cycle_s=cycle, threshold_s=threshold)
    return df


def money(lost_s: float, cfg: dict) -> float:
    return lost_s / 3600.0 * float(cfg["crew_size"]) * float(cfg["loaded_rate_per_hour"])
