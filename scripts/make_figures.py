"""Render throughput, stable-step, accuracy and polar-filter comparisons.

Read figure-pack/ and the saved stability records; write SVG, PNG and
provenance JSON to figures/. Requires the figures extra. Standard output
lists output hashes and sizes as JSON.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "figure-pack"
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT))

W = 1000
PNG_SCALE = 2
FONTS = ["Arial", "DejaVu Sans"]
MIN_PX = 16
AXIS_PX = 18
TICK_PX = 16
LABEL_PX = 17

TORUS = "#0072B2"
LATLON = "#D55E00"
CHARCOAL = "#333333"
INK = "#1A1A1A"
SOFT = "#3D3D3D"
AXIS = "#8C8C8C"
GRID = "#E6E6E6"
OUTLINE = "#14222E"
BLUES = {144: "#A9D2EE", 288: "#3E8FC9", 576: "#0B3F66"}
SHAPES = {144: "o", 288: "s", 576: "^"}
FORBIDDEN = "\u2013\u2014"
NUMBER = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?x?$")


def px(value):
    """Convert design pixels to Matplotlib points at the reference resolution."""
    return value * 72 / 96


def rc():
    """Use the shared figure style, outlined SVG text and stable SVG identifiers."""
    return {
        "font.family": FONTS,
        "font.size": px(TICK_PX),
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": px(1.2),
        "axes.labelcolor": INK,
        "axes.labelsize": px(AXIS_PX),
        "axes.unicode_minus": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": SOFT,
        "ytick.labelcolor": SOFT,
        "xtick.labelsize": px(TICK_PX),
        "ytick.labelsize": px(TICK_PX),
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": px(1.2),
        "ytick.major.width": px(1.2),
        "xtick.major.pad": 5,
        "ytick.major.pad": 6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
        "svg.hashsalt": "sphere-on-a-torus",
        "path.simplify": False,
        "legend.frameon": False,
    }


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def read_csv(name):
    with (PACK / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b, rel=1e-12):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0)


class Canvas:
    """Place axes in design pixels on a fixed-width white canvas."""
    def __init__(self, height):
        self.height = height
        self.fig = plt.figure(figsize=(W / 96, height / 96), dpi=96)

    def axes(self, left, top, width, height):
        return self.fig.add_axes([left / W, 1 - (top + height) / self.height, width / W, height / self.height])


def comma(value, _pos=None):
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def plain(value, _pos=None):
    return f"{value:g}"


def times(value, decimals=0):
    return f"{value:,.{decimals}f}x"


def guides(ax, axis="x"):
    ax.grid(True, axis=axis, which="major", color=GRID, lw=px(1.0), zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", length=0)


def drawn_text(fig):
    """Measure every visible text label after drawing the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        size_px = artist.get_fontsize() * 96 / 72
        items.append({"text": artist.get_text(), "size_px": round(size_px, 2), "box": [box.x0, box.y0, box.x1, box.y1]})
    return items


def audit(fig, allowed):
    """Reject unlisted text, small labels, clipping and sampled line overlaps.

    These geometric checks support visual inspection; they do not replace it.
    """
    items = drawn_text(fig)
    width, height = fig.canvas.get_width_height()
    report = {"labels": [], "ticks": [], "problems": []}
    for item in items:
        text = item["text"]
        if any(ch in text for ch in FORBIDDEN):
            report["problems"].append(f"dash in {text!r}")
        if text in allowed:
            report["labels"].append(text)
        elif NUMBER.match(text) or text == r"$\times10^{-4}$":
            report["ticks"].append(text)
        else:
            report["problems"].append(f"text not in allowed labels or numeric ticks: {text!r}")
        if item["size_px"] < MIN_PX - 1e-6:
            report["problems"].append(f"text below {MIN_PX} px: {text!r}")
        x0, y0, x1, y1 = item["box"]
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            report["problems"].append(f"text outside canvas: {text!r}")
    renderer = fig.canvas.get_renderer()
    for line in fig.findobj(Line2D):
        if not line.get_visible() or line.get_linestyle() in ("None", "none", " ", "") or line.get_color() == GRID:
            continue
        path = line.get_transform().transform(np.asarray(line.get_xydata(), float))
        if len(path) < 2:
            continue
        for item in items:
            x0, y0, x1, y1 = item["box"]
            for (ax0, ay0), (ax1, ay1) in zip(path[:-1], path[1:]):
                t = np.linspace(0, 1, 200)
                xs, ys = ax0 + (ax1 - ax0) * t, ay0 + (ay1 - ay0) * t
                if ((xs > x0 + 0.5) & (xs < x1 - 0.5) & (ys > y0 + 0.5) & (ys < y1 - 0.5)).any():
                    report["problems"].append(f"line crosses {item['text']!r}")
                    break
    for spine_ax in fig.axes:
        for spine in spine_ax.spines.values():
            if not spine.get_visible():
                continue
            box = spine.get_window_extent(renderer)
            for item in items:
                x0, y0, x1, y1 = item["box"]
                if box.x0 < x1 - 0.5 and x0 + 0.5 < box.x1 and box.y0 < y1 - 0.5 and y0 + 0.5 < box.y1:
                    report["problems"].append(f"axis line crosses {item['text']!r}")
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            p, q = items[a]["box"], items[b]["box"]
            if p[0] < q[2] - 0.5 and q[0] < p[2] - 0.5 and p[1] < q[3] - 0.5 and q[1] < p[3] - 0.5:
                report["problems"].append(f"overlap: {items[a]['text']!r} and {items[b]['text']!r}")
    missing = sorted(set(allowed) - set(report["labels"]))
    if missing:
        report["problems"].append(f"missing labels: {missing}")
    report["min_text_px"] = min(item["size_px"] for item in items)
    if report["problems"]:
        raise SystemExit("\n".join(report["problems"]))
    return report


def inside(ax, xs, ys):
    """Check that every plotted point lies within the stated axis limits."""
    xlo, xhi = sorted(ax.get_xlim())
    ylo, yhi = sorted(ax.get_ylim())
    for x, y in zip(xs, ys):
        if not (xlo <= x <= xhi and ylo <= y <= yhi):
            raise SystemExit(f"point ({x}, {y}) outside the axes")
    return True


def throughput():
    """Plot modelled simulated-time throughput against the polar-limited baseline."""
    rows = read_csv("throughput.csv")
    model = json.loads((PACK / "model.json").read_text())
    names = {
        "Polar-limited lat-lon": "Polar-limited lat-lon",
        "Torus, patch work counted": "Torus, patch work counted",
        "Torus, shared patch work": "Torus, shared work estimate",
    }
    if [r["layout"] for r in rows] != list(names):
        raise SystemExit("throughput.csv rows are not in the expected order")
    base = float(rows[0]["steps_per_s"])
    checks = []
    for r in rows[1:]:
        expected = float(r["steps_per_s"]) / base * model["step_ratio"]
        if not close(expected, r["relative"], 1e-9):
            raise SystemExit(f"relative for {r['layout']} does not match the model")
        checks.append({"layout": r["layout"], "relative": float(r["relative"]), "recomputed": expected})
    if float(rows[0]["relative"]) != 1.0:
        raise SystemExit("the baseline is not 1")
    pm = model["patch_model"]
    if not (close(rows[1]["steps_per_s"], pm["steps_per_s_no_redistribution"]) and close(rows[2]["steps_per_s"], pm["steps_per_s_redistributed"])):
        raise SystemExit("step rates do not match model.json")
    values = [float(r["relative"]) for r in rows]
    labels = ["1x", "856x", "1,263x"]
    if [times(v) for v in values] != labels:
        raise SystemExit(f"value labels {[times(v) for v in values]} differ from the expected values")

    canvas = Canvas(300)
    ax = canvas.axes(300, 22, 600, 192)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 2000)
    ax.set_ylim(2.55, -0.55)
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(comma))
    ax.set_yticks(range(3))
    ax.set_yticklabels(list(names.values()), fontsize=px(LABEL_PX), color=INK)
    ax.tick_params(axis="y", length=0, pad=14)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    guides(ax)
    styles = [
        dict(marker="s", ms=px(15), mfc=CHARCOAL, mec=CHARCOAL, mew=px(1.5)),
        dict(marker="o", ms=px(17), mfc=TORUS, mec=TORUS, mew=px(1.5)),
        dict(marker="D", ms=px(14), mfc="white", mec=TORUS, mew=px(2.6)),
    ]
    for k, (value, style, label) in enumerate(zip(values, styles, labels)):
        ax.plot([value], [k], ls="none", zorder=4, clip_on=False, **style)
        ax.annotate(label, (value, k), xytext=(14, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=px(LABEL_PX), color=INK)
    ax.set_xlabel("Modelled simulated time / wall time (relative to baseline)", labelpad=10)
    inside(ax, values, range(3))
    allowed = list(names.values()) + labels + ["Modelled simulated time / wall time (relative to baseline)"]
    report = audit(canvas.fig, allowed)
    meta = {
        "source_files": ["figure-pack/throughput.csv", "figure-pack/model.json"],
        "plotted": [{"row": names[r["layout"]], "layout": r["layout"], "relative": float(r["relative"]), "label": l}
                    for r, l in zip(rows, labels)],
        "checks": {"relative_equals_step_rate_ratio_times_step_ratio": checks, "step_ratio": model["step_ratio"]},
        "axes": {"x": {"scale": "log", "range": [0.8, 2000], "ticks": [1, 10, 100, 1000]}},
        "marks": {"baseline": "charcoal square", "patch_work_counted": "filled blue circle", "shared_work_estimate": "hollow blue diamond"},
    }
    return "throughput", canvas, allowed, meta, report


def stable_step():
    """Plot both measured stability series against computed-point spacing."""
    from polarswe import configs

    rows = read_csv("stable_step_vs_spacing.csv")
    source = [json.loads(line) for line in (ROOT / "data" / "timestep.jsonl").read_text().splitlines() if line.strip()]
    matched = []
    for r in rows:
        kind = "torus" if r["grid"] == "torus" else "dfs"
        hits = [s for s in source if s.get("grid", [None])[0] == kind and s["grid"][1] == int(r["width"])
                and close(s["dt_max_s"], r["dt_max_s"])]
        if kind == "torus":
            hits = [s for s in hits if s.get("scheme_str") == configs.MAIN]
        else:
            hits = [s for s in hits if s.get("scheme", {}).get("order") == 2]
        if len(hits) != 1:
            raise SystemExit(f"no unique source record for {r['grid']} {r['width']}")
        s = hits[0]
        if not (close(s["r_max_computed"], r["r_max_computed"]) and close(s["dx_eq_computed_m"], r["dx_eq_computed_m"])):
            raise SystemExit(f"source record differs for {r['grid']} {r['width']}")
        matched.append({"grid": r["grid"], "width": int(r["width"]), "dt_max_s": float(r["dt_max_s"]),
                        "source_dt_max_s": s["dt_max_s"], "days": s["days"], "case": s.get("case"),
                        "scheme": s.get("scheme_str")})
    series = {}
    for r in rows:
        series.setdefault(r["grid"], []).append((float(r["dx_eq_computed_m"]) / 1000, float(r["r_max_computed"])))
    expected = {"torus": ["2.45", "2.32", "2.25"], "lat-lon": ["0.164", "0.082", "0.041"]}
    labels = {}
    for grid, pts in series.items():
        pts.sort(key=lambda p: -p[0])
        digits = 2 if grid == "torus" else 3
        labels[grid] = [f"{y:.{digits}f}" for _, y in pts]
        if labels[grid] != expected[grid]:
            raise SystemExit(f"labels {labels[grid]} differ from the expected values")

    canvas = Canvas(590)
    ax = canvas.axes(118, 30, 820, 470)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(500, 60)
    ax.set_ylim(0.03, 3.2)
    ax.set_xticks([400, 200, 100, 80])
    ax.set_yticks([0.04, 0.1, 0.3, 1, 3])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(plain))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(plain))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    guides(ax, "both")
    style = {
        "torus": dict(color=TORUS, marker="o", ms=px(15), mfc=TORUS, mec="white", mew=px(1.8), lw=px(2.4)),
        "lat-lon": dict(color=LATLON, marker="s", ms=px(13.5), mfc=LATLON, mec="white", mew=px(1.8), lw=px(2.4)),
    }
    names = {"torus": "Torus", "lat-lon": "Unfiltered lat-lon"}
    handles = []
    for grid in ("torus", "lat-lon"):
        xs, ys = zip(*series[grid])
        (line,) = ax.plot(xs, ys, zorder=4, clip_on=False, label=names[grid], **style[grid])
        handles.append(line)
        for (x, y), text in zip(series[grid], labels[grid]):
            if grid == "torus":
                offset, ha = (0, 12), "center"
            else:
                offset, ha = (7, 7), "left"
            ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va="bottom",
                        fontsize=px(LABEL_PX), color=INK)
        inside(ax, xs, ys)
    legend = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.03, 0.47), fontsize=px(LABEL_PX),
                       handlelength=2.2, handletextpad=0.7, labelspacing=0.8, borderaxespad=0)
    for text in legend.get_texts():
        text.set_color(INK)
    ax.set_xlabel("Equatorial spacing of computed points (km)", labelpad=10)
    ax.set_ylabel("Stable step / equatorial wave-crossing time", labelpad=12)
    allowed = ["Torus", "Unfiltered lat-lon", "Equatorial spacing of computed points (km)",
               "Stable step / equatorial wave-crossing time"] + labels["torus"] + labels["lat-lon"]
    report = audit(canvas.fig, allowed)
    meta = {
        "source_files": ["figure-pack/stable_step_vs_spacing.csv", "data/timestep.jsonl"],
        "plotted": [{"grid": r["grid"], "width": int(r["width"]), "x_km": float(r["dx_eq_computed_m"]) / 1000,
                     "y": float(r["r_max_computed"])} for r in rows],
        "value_labels": labels,
        "checks": {"dt_max_s_matches_source": matched},
        "axes": {"x": {"scale": "log", "range": [500, 60], "ticks": [400, 200, 100, 80]},
                 "y": {"scale": "log", "range": [0.03, 3.2], "ticks": [0.04, 0.1, 0.3, 1, 3]}},
    }
    return "stable-step", canvas, allowed, meta, report


def accuracy():
    """Plot all paired TRSK-to-torus height-error ratios without mixing norms."""
    rows = read_csv("accuracy.csv")
    order = ["TC2, day 5", "TC5, day 15", "TC6, day 5", "TC6, day 14", "Steady jet, day 6", "Perturbed jet, day 6"]
    levels = {144: 5, 288: 6, 576: 7}
    checks = []
    data = {}
    for r in rows:
        m = int(r["m"])
        ratio = float(r["trsk_l2"]) / float(r["torus_l2"])
        if not close(ratio, r["ratio"], 1e-12):
            raise SystemExit(f"ratio mismatch for {r['case']} {m}")
        if int(r["torus_unique"]) != m * m // 2 or int(r["trsk_cells"]) != 10 * 4 ** levels[m] + 2:
            raise SystemExit(f"pair identity mismatch for {r['case']} {m}")
        data[(r["case"], m)] = float(r["ratio"])
        checks.append({"case": r["case"], "m": m, "ratio": float(r["ratio"]), "recomputed": ratio,
                       "torus_unique": int(r["torus_unique"]), "trsk_cells": int(r["trsk_cells"])})
    if sorted({c for c, _ in data}) != sorted(order) or len(data) != 18:
        raise SystemExit("accuracy.csv does not hold 18 pairs in six cases")
    values = list(data.values())
    if min(values) <= 1:
        raise SystemExit("a ratio is not above one")

    canvas = Canvas(540)
    ax = canvas.axes(226, 64, 700, 400)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 5000)
    ax.set_ylim(5.55, -0.55)
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(comma))
    ax.set_yticks(range(6))
    ax.set_yticklabels(order, fontsize=px(LABEL_PX), color=INK)
    ax.tick_params(axis="y", length=0, pad=14)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    guides(ax)
    for k in range(6):
        ax.axhline(k, color=GRID, lw=px(1.0), zorder=0)
    ax.axvline(1, color=CHARCOAL, lw=px(1.6), zorder=2)
    ax.annotate("Equal error", (1, -0.55), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                fontsize=px(LABEL_PX), color=INK, annotation_clip=False)
    offsets = {144: -0.22, 288: 0.0, 576: 0.22}
    sizes = {144: 13, 288: 12, 576: 14}
    handles = []
    for m in (144, 288, 576):
        xs = [data[(c, m)] for c in order]
        ys = [k + offsets[m] for k in range(6)]
        (h,) = ax.plot(xs, ys, ls="none", marker=SHAPES[m], ms=px(sizes[m]), mfc=BLUES[m], mec=OUTLINE, mew=px(1.3),
                       zorder=4, clip_on=False, label=f"M = {m}")
        handles.append(h)
        inside(ax, xs, ys)
    legend = canvas.fig.legend(handles=handles, loc="lower right", bbox_to_anchor=((226 + 700) / W, 1 - 30 / 560),
                               ncol=3, fontsize=px(LABEL_PX), handletextpad=0.4, columnspacing=1.6, borderaxespad=0)
    for text in legend.get_texts():
        text.set_color(INK)
    ax.set_xlabel("TRSK height l2 / torus height l2", labelpad=10)
    allowed = order + ["M = 144", "M = 288", "M = 576", "Equal error", "TRSK height l2 / torus height l2"]
    report = audit(canvas.fig, allowed)
    tc6 = {m: data[("TC6, day 14", m)] for m in (144, 288, 576)}
    meta = {
        "source_files": ["figure-pack/accuracy.csv"],
        "plotted": [{"row": c, "m": m, "ratio": data[(c, m)]} for c in order for m in (144, 288, 576)],
        "checks": {
            "ratio_equals_trsk_l2_over_torus_l2": checks,
            "torus_unique_is_m_squared_over_2": True,
            "trsk_cells_is_10_times_4_to_level_plus_2": {str(m): levels[m] for m in levels},
            "min_ratio": min(values),
            "max_ratio": max(values),
            "all_above_one": True,
            "tc6_day14_ratios": {str(m): v for m, v in tc6.items()},
            "linf_plotted": False,
        },
        "axes": {"x": {"scale": "log", "range": [0.8, 5000], "ticks": [1, 10, 100, 1000]}},
        "marks": {str(m): {"shape": {"o": "circle", "s": "square", "^": "triangle"}[SHAPES[m]], "fill": BLUES[m],
                           "outline": OUTLINE} for m in SHAPES},
    }
    return "accuracy", canvas, allowed, meta, report


def filter_trade():
    """Plot the separate same-kernel model of filtered lat-lon versus torus."""
    rows = read_csv("filter_trade.csv")
    checks = []
    for r in rows:
        ratio = float(r["filter_steps_per_s"]) / float(r["torus_steps_per_s"])
        if not close(ratio, r["filter_over_torus"], 1e-12):
            raise SystemExit(f"ratio mismatch at {r['arcsec']}")
        checks.append({"arcsec": int(r["arcsec"]), "filter_over_torus": float(r["filter_over_torus"]), "recomputed": ratio,
                       "filter_limit": r["filter_limit"]})
    xs = [int(r["arcsec"]) for r in rows]
    ys = [float(r["filter_over_torus"]) for r in rows]
    if xs != [15, 30, 60, 120, 240, 480, 960]:
        raise SystemExit("unexpected spacings")
    wanted = {15: "1.41x", 480: "0.72x", 960: "0.62x"}
    for x, y in zip(xs, ys):
        if x in wanted and times(y, 2) != wanted[x]:
            raise SystemExit(f"label at {x} is {times(y, 2)}")

    canvas = Canvas(560)
    ax = canvas.axes(118, 24, 820, 450)
    ax.set_xscale("log", base=2)
    pad = 2 ** 0.14
    # Leave enough room for the full endpoint markers on the logarithmic axis.
    ax.set_xlim(15 / pad, 960 * pad)
    ax.set_ylim(0, 1.6)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(plain))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(plain))
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    guides(ax, "y")
    ax.axhline(1, color=CHARCOAL, lw=px(1.6), zorder=2)
    ax.annotate("Equal throughput", (960 * pad, 1), xytext=(0, 7), textcoords="offset points", ha="right", va="bottom",
                fontsize=px(LABEL_PX), color=INK)
    ax.plot(xs, ys, color=LATLON, lw=px(2.4), marker="s", ms=px(13.5), mfc=LATLON, mec="white", mew=px(1.8), zorder=4,
            clip_on=False)
    place = {15: ((-5, 11), "left", "bottom"), 480: ((-14, -6), "right", "top"), 960: ((0, -15), "center", "top")}
    for x, y in zip(xs, ys):
        if x in wanted:
            offset, ha, va = place[x]
            ax.annotate(wanted[x], (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va,
                        fontsize=px(LABEL_PX), color=INK)
    inside(ax, xs, ys)
    ax.set_xlabel("Grid spacing (arcseconds)", labelpad=10)
    ax.set_ylabel("Modelled filtered lat-lon / torus throughput", labelpad=12)
    allowed = ["Grid spacing (arcseconds)", "Modelled filtered lat-lon / torus throughput", "Equal throughput"] + list(wanted.values())
    report = audit(canvas.fig, allowed)
    meta = {
        "source_files": ["figure-pack/filter_trade.csv"],
        "plotted": [{"arcsec": x, "filter_over_torus": y} for x, y in zip(xs, ys)],
        "value_labels": {str(k): v for k, v in wanted.items()},
        "checks": {"ratio_equals_filter_over_torus_steps_per_s": checks, "equal_time_steps_assumed": True},
        "filter_limit": {str(int(r["arcsec"])): r["filter_limit"] for r in rows},
        "axes": {"x": {"scale": "log2", "range": [15, 960], "padding_octaves": 0.14, "ticks": xs},
                 "y": {"scale": "linear", "range": [0, 1.6], "ticks": [0, 0.5, 1, 1.5]}},
    }
    return "filter-trade", canvas, allowed, meta, report


def atomic_write(path, data):
    """Replace an output only after its full contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export(name, canvas, allowed, meta, report):
    """Save both image formats and their input, rendering and output records."""
    fig = canvas.fig
    svg = io.BytesIO()
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None})
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=96 * PNG_SCALE, metadata={"Software": None})
    image = Image.open(io.BytesIO(png.getvalue())).convert("RGB")
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    outputs = {}
    for suffix, data in (("svg", svg.getvalue()), ("png", out.getvalue())):
        path = OUT / f"{name}.{suffix}"
        atomic_write(path, data)
        outputs[str(path.relative_to(ROOT))] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    font = font_manager.findfont(font_manager.FontProperties(family=FONTS))
    provenance = {
        "figure": name,
        "labels": allowed,
        "source_data": {p: {"sha256": sha256(ROOT / p)} for p in meta.pop("source_files")},
        **meta,
        "text_audit": {"labels": report["labels"], "numeric": report["ticks"], "min_text_px": report["min_text_px"]},
        "render": {
            "canvas_px": [W, canvas.height],
            "png_px": [image.width, image.height],
            "svg_text": "outlined paths",
            "font": Path(font).name,
            "colours": {"torus": TORUS, "lat_lon": LATLON, "baseline": CHARCOAL},
            "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                         "numpy": np.__version__, "pillow": Image.__version__},
        },
        "outputs": outputs,
    }
    atomic_write(OUT / f"{name}.provenance.json", (json.dumps(provenance, indent=2) + "\n").encode())
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    built = {}
    with plt.style.context("default"), matplotlib.rc_context(rc()):
        for make in (throughput, stable_step, accuracy, filter_trade):
            name, canvas, allowed, meta, report = make()
            built.update(export(name, canvas, allowed, meta, report))
    print(json.dumps(built, indent=2))


if __name__ == "__main__":
    sys.exit(main())
