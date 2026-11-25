#!/usr/bin/env python3
"""Food-Line Cameras MVP: footage in, Line Loss Report out.

  python run.py calibrate configs/mock.yaml        draw the count line + belt zone
  python run.py process   configs/mock.yaml        count trays, find stoppages
  (edit runs/<config>/<day>/stoppages.csv: fill in `cause`; notes.yaml: bottleneck/change)
  python run.py report    configs/mock.yaml        → report.html + report.pdf
  python run.py all       configs/mock.yaml        process + report
  python run.py handcount configs/mock.yaml        count by hand to check accuracy
  python run.py evaluate  configs/mock.yaml        camera vs. hand count → eval.json
"""
from __future__ import annotations

import argparse
import sys

from foodline import calibrate, handcount, pipeline
from foodline.report import render


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="draw count line and belt zone")
    c.add_argument("config")
    c.add_argument("--camera")
    c.add_argument("--at", type=float, default=5.0, help="seconds into the video to grab the frame")
    c.add_argument("--line", help="non-interactive: x1,y1,x2,y2 (0..1)")
    c.add_argument("--zone", help="non-interactive: x,y,x,y,... (0..1), 3+ points")

    for name in ("process", "all"):
        s = sub.add_parser(name, help="count trays and find stoppages" + (" + report" if name == "all" else ""))
        s.add_argument("config")
        s.add_argument("--start", help='wall-clock start of the video, e.g. "2026-11-28 10:00:00"')
        s.add_argument("--debug-video", action="store_true", help="also write an annotated video")
        if name == "all":
            s.add_argument("--no-pdf", action="store_true")

    r = sub.add_parser("report", help="build report.html / report.pdf")
    r.add_argument("config")
    r.add_argument("--run", help="run folder (default: latest)")
    r.add_argument("--no-pdf", action="store_true")

    h = sub.add_parser("handcount", help="count trays by hand")
    h.add_argument("config")
    h.add_argument("--minutes", type=float, help="stop after this many minutes")
    h.add_argument("--from", dest="start_s", type=float, default=0.0, help="start this many seconds in")

    e = sub.add_parser("evaluate", help="camera count vs. hand count")
    e.add_argument("config")
    e.add_argument("--truth", help="CSV with a `time` column (e.g. synthetic ground truth)")
    e.add_argument("--tolerance", type=float, default=2.0)
    e.add_argument("--run")

    a = p.parse_args(argv)
    if a.cmd == "calibrate":
        calibrate.run(a.config, a.camera, a.at, a.line, a.zone)
    elif a.cmd == "process":
        pipeline.run(a.config, a.start, a.debug_video)
    elif a.cmd == "all":
        out = pipeline.run(a.config, a.start, a.debug_video)
        render.build(a.config, str(out), pdf=not a.no_pdf)
    elif a.cmd == "report":
        render.build(a.config, a.run, pdf=not a.no_pdf)
    elif a.cmd == "handcount":
        handcount.play(a.config, a.minutes, a.start_s)
    elif a.cmd == "evaluate":
        handcount.evaluate(a.config, a.truth, a.tolerance, a.run)


if __name__ == "__main__":
    sys.exit(main())
