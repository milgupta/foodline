from datetime import datetime, timedelta
from pathlib import Path

import pytest

from foodline import config as config_mod
from foodline import events
from foodline.count import Crossing
from foodline.handcount import compare
from foodline.ingest import Segment

T0 = datetime(2026, 11, 28, 14, 0, 0)
CFG = config_mod._merge(config_mod.DEFAULTS, {"crew_size": 6, "loaded_rate_per_hour": 27})


def _seg(duration=600):
    return Segment(Path("cam3_20261128_140000.mp4"), T0, duration, 15.0, 1280, 720)


def _trays(times, areas=None):
    areas = areas or [0.02] * len(times)
    cs = [Crossing(t, i, a, 1) for i, (t, a) in enumerate(zip(times, areas))]
    return events.tray_table([(_seg(), cs)], merge_ratio=1.6)


def test_cycle_time_in_seconds():
    trays = _trays([3.0 * i for i in range(1, 50)])
    assert events.normal_cycle_s(trays) == pytest.approx(3.0)


def test_stoppage_found_and_short_gap_ignored():
    times = [3.0 * i for i in range(1, 20)]          # up to 57 s
    times += [57 + 9 + 3.0 * i for i in range(0, 10)]  # 9 s gap: not a stoppage
    last = times[-1]
    times += [last + 40 + 3.0 * i for i in range(0, 10)]  # 40 s gap: stoppage
    trays = _trays(times)
    stops = events.find_stoppages(trays, events.footage_spans([_seg()]), CFG)
    assert len(stops) == 1
    assert stops.iloc[0]["gap_s"] == pytest.approx(40.0)
    assert stops.iloc[0]["lost_s"] == pytest.approx(37.0)  # gap minus one 3 s cycle


def test_touching_trays_counted_from_blob_size():
    times = [3.0 * i for i in range(1, 21)]
    areas = [0.02] * 20
    areas[5] = 0.041  # two trays touching
    trays = _trays(times, areas)
    assert trays["n"].sum() == 21
    assert trays.loc[5, "n"] == 2


def test_gap_across_missing_footage_is_not_a_stoppage():
    a = Segment(Path("a_20261128_140000.mp4"), T0, 60, 15, 1280, 720)
    b = Segment(Path("b_20261128_141000.mp4"), T0 + timedelta(minutes=10), 60, 15, 1280, 720)
    cs_a = [Crossing(3.0 * i, i, 0.02, 1) for i in range(1, 20)]
    cs_b = [Crossing(3.0 * i, 100 + i, 0.02, 1) for i in range(1, 20)]
    trays = events.tray_table([(a, cs_a), (b, cs_b)], 1.6)
    stops = events.find_stoppages(trays, events.footage_spans([a, b]), CFG)
    assert stops.empty


def test_money():
    # 10 lost minutes x 6 people x $27/hr = $27
    assert events.money(600, CFG) == pytest.approx(27.0)


def test_hand_count_matching():
    auto = [T0 + timedelta(seconds=s) for s in (3, 6, 9, 12)]
    hand = [T0 + timedelta(seconds=s) for s in (3.5, 6.8, 9.2, 20)]
    assert compare(auto, hand, tolerance_s=2.0) == 3
