"""`run.py report`: processed run → report.html (+ report.pdf via headless Chrome)."""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import config as config_mod
from .. import events
from ..pipeline import latest_run

INK, MUTED, ACCENT, LOSS, GRID = "#1f2933", "#6b7785", "#2f6fdf", "#d64545", "#e6e9ee"
CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg"):]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def throughput_chart(pm: pd.DataFrame, stops: pd.DataFrame, target: float | None) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 2.6), dpi=100)
    minutes = len(pm)
    if minutes > 180:  # long footage: 5-minute bins, shown per minute
        s = pm.set_index("minute")["trays"].resample("5min").sum() / 5.0
        x, y, width = s.index, s.values, 4.0 / 1440
    else:
        x, y, width = pm["minute"], pm["trays"], 0.8 / 1440
    for _, r in stops.iterrows():
        ax.axvspan(r["start"], r["end"], color=LOSS, alpha=0.13, lw=0)
    ax.bar(x, y, width=width, color=ACCENT, align="edge")
    if target:
        ax.axhline(target, color=INK, lw=1, ls=(0, (4, 3)))
        ax.text(x.iloc[0] if hasattr(x, "iloc") else x[0], target, "  normal pace",
                va="bottom", ha="left", fontsize=7.5, color=INK)
    ax.set_ylabel("trays / min", fontsize=8, color=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _style(ax)
    return _svg(fig)


def timeline_chart(stops: pd.DataFrame, spans) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 0.9), dpi=100)
    for a, b in spans:
        ax.barh(0, (b - a).total_seconds() / 86400, left=a, height=0.55, color="#cfe0fa", lw=0)
    for _, r in stops.iterrows():
        ax.barh(0, (r["end"] - r["start"]).total_seconds() / 86400, left=r["start"], height=0.55,
                color=LOSS, lw=0)
        ax.text(r["start"] + (r["end"] - r["start"]) / 2, 0.42, r["id"], ha="center", va="bottom",
                fontsize=7, color=LOSS)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    return _svg(fig)


def _img64(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def _fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def build(config_path, run: str | None = None, pdf: bool = True) -> Path:
    cfg = config_mod.load(config_path)
    rdir = Path(run) if run else latest_run(cfg)
    meta = json.loads((rdir / "meta.json").read_text())
    trays = pd.read_csv(rdir / "trays.csv", parse_dates=["time"])
    pm = pd.read_csv(rdir / "per_minute.csv", parse_dates=["minute"])
    stops = pd.read_csv(rdir / "stoppages.csv", parse_dates=["start", "end"], keep_default_na=False)
    spans = [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in meta["spans"]]
    notes_path = rdir / "notes.yaml"
    if not notes_path.exists():
        notes_path.write_text(
            "# Hand-written parts of the report. Leave blank to hide a section.\n"
            "bottleneck: \"\"          # e.g. Station 2 (protein) - always 3-4 trays queued in front of it\n"
            "recommended_change: \"\"  # e.g. Move one person from station 4 to station 2 for the lunch run\n"
            "observations: []        # extra bullet points\n")
    notes = yaml.safe_load(notes_path.read_text()) or {}

    cur = cfg["currency"]
    footage_s = float(meta["footage_s"])
    running_s = max(1.0, footage_s - (stops["gap_s"].sum() if len(stops) else 0))
    total_trays = int(meta["trays"])
    cycle = meta.get("cycle_s") or float("nan")
    normal_tpm = 60.0 / cycle if cycle and cycle == cycle else None
    lost_s = float(stops["lost_s"].sum()) if len(stops) else 0.0
    lost_cost = events.money(lost_s, cfg)
    shift_s = float(cfg["shift_hours"]) * 3600
    per_shift_lost = lost_s / footage_s * shift_s if footage_s else 0
    per_year_cost = events.money(per_shift_lost, cfg) * float(cfg["work_days_per_year"])
    trays_missed = lost_s / cycle if normal_tpm else 0

    # Top losses: confirmed causes are grouped; unreviewed stoppages stand alone.
    groups = []
    if len(stops):
        reviewed = stops[stops["cause"].astype(str).str.strip() != ""]
        for cause, g in reviewed.groupby(reviewed["cause"].str.strip().str.lower()):
            groups.append({"label": cause.capitalize(), "count": len(g), "lost_s": g["lost_s"].sum(),
                           "ids": ", ".join(g["id"])})
        for _, r in stops[stops["cause"].astype(str).str.strip() == ""].iterrows():
            groups.append({"label": f"Unreviewed stop at {r['start']:%H:%M:%S}", "count": 1,
                           "lost_s": r["lost_s"], "ids": r["id"]})
    groups.sort(key=lambda g: -g["lost_s"])
    for g in groups:
        g["lost"] = _fmt_dur(g["lost_s"])
        g["cost"] = f"{cur}{events.money(g['lost_s'], cfg):,.0f}"
        g["share"] = round(100 * g["lost_s"] / lost_s) if lost_s else 0

    hourly = []
    if len(pm):
        h = pm.set_index("minute")["trays"].resample("h").agg(["sum", "count"])
        for ts, r in h.iterrows():
            if r["count"]:
                hourly.append({"hour": f"{ts:%H:00}", "trays": int(r["sum"]), "minutes": int(r["count"]),
                               "tpm": f"{r['sum'] / r['count']:.1f}"})

    stop_rows = []
    for _, r in stops.iterrows():
        stop_rows.append({
            "id": r["id"], "start": f"{r['start']:%H:%M:%S}", "gap": _fmt_dur(r["gap_s"]),
            "lost": _fmt_dur(r["lost_s"]), "cost": f"{cur}{events.money(r['lost_s'], cfg):,.0f}",
            "kind": r["kind"], "cause": str(r["cause"]).strip(), "note": str(r["note"]).strip(),
            "thumb": _img64(rdir / "thumbs" / f"{r['id']}.jpg"),
        })
    n_unreviewed = sum(1 for s in stop_rows if not s["cause"])

    accuracy = None
    ev = rdir / "eval.json"
    if ev.exists():
        accuracy = json.loads(ev.read_text())

    start = spans[0][0] if spans else trays["time"].min()
    end = spans[-1][1] if spans else trays["time"].max()
    ctx = {
        "cfg": cfg, "meta": meta, "cur": cur,
        "date": f"{start:%A, %B %-d, %Y}", "window": f"{start:%H:%M}–{end:%H:%M}",
        "footage": _fmt_dur(footage_s),
        "kpi": {
            "trays": f"{total_trays:,}",
            "avg_tpm": f"{total_trays / (footage_s / 60):.1f}" if footage_s else "–",
            "running_tpm": f"{total_trays / (running_s / 60):.1f}",
            "normal_tpm": f"{normal_tpm:.1f}" if normal_tpm else "–",
            "stops": len(stops),
            "lost": _fmt_dur(lost_s),
            "lost_pct": f"{100 * lost_s / footage_s:.0f}%" if footage_s else "–",
            "cost": f"{cur}{lost_cost:,.0f}",
            "trays_missed": f"{trays_missed:,.0f}",
            "per_shift": _fmt_dur(per_shift_lost),
            "per_year": f"{cur}{per_year_cost:,.0f}",
        },
        "threshold": _fmt_dur(meta.get("threshold_s") or cfg["stoppage"]["min_gap_s"]),
        "cycle": f"{cycle:.1f}s" if cycle == cycle else "–",
        "groups": groups[:3], "more_groups": max(0, len(groups) - 3),
        "hourly": hourly, "stops": stop_rows, "n_unreviewed": n_unreviewed,
        "notes": notes, "accuracy": accuracy,
        "chart_throughput": throughput_chart(pm, stops, normal_tpm),
        "chart_timeline": timeline_chart(stops, spans) if spans else "",
        "generated": f"{datetime.now():%b %-d, %Y %H:%M}",
        "short_footage": footage_s < 3600,
    }
    env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html"]))
    html = env.get_template("template.html.j2").render(**ctx)
    out = rdir / "report.html"
    out.write_text(html)
    print(f"Report: {out}")

    if pdf:
        chrome = next((c for c in CHROME if Path(c).exists()), None) or shutil.which("chromium")
        if chrome:
            pdf_path = rdir / "report.pdf"
            pdf_path.unlink(missing_ok=True)
            # Own throwaway profile, so it never waits on the user's open Chrome; no pipes,
            # because Chrome's helper processes keep them open past the timeout.
            with tempfile.TemporaryDirectory() as profile:
                try:
                    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-first-run",
                                    "--no-default-browser-check", f"--user-data-dir={profile}",
                                    "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", out.as_uri()],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
                except subprocess.TimeoutExpired:
                    pass
            if pdf_path.exists():
                print(f"PDF:    {pdf_path}")
            else:
                print("(PDF export failed. Open report.html and print to PDF.)")
        else:
            print("(No Chrome found for PDF. Open report.html and print to PDF.)")
    return out
