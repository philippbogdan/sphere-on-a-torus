"""Draw the four README figures from the clean data in figure-pack/.

Run from anywhere: python scripts/make_figures.py

Every figure is designed for a GitHub README column about 830 CSS pixels
wide. The SVG is exactly 830 CSS pixels wide (622.5 pt) and the PNG is the
same canvas at twice the pixel density.

The script checks its own output before writing anything: every text must
be approved wording or a number, sit inside the canvas, avoid every other
text, every data marker and every data line, and be at least 14 CSS pixels
tall. Cell labels in the accuracy matrix must reach 4.5:1 contrast against
their cell. Any failure stops the build.

The same inputs and package versions give byte-identical files.
"""
from __future__ import annotations

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

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, PngImagePlugin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "figure-pack"
OUT = ROOT / "figures"

# Canvas geometry in CSS pixels. Matplotlib works in inches, so one CSS
# pixel is 1/96 inch and one point is 4/3 CSS pixels.
WIDTH = 830
CSS_DPI = 96
PNG_SCALE = 2
MIN_TEXT_PX = 14

# Vertical layout, in CSS pixels from the top: the plot starts one gap below
# the title and the canvas ends one margin below the lowest label.
THROUGHPUT_TOP, THROUGHPUT_H = 88, 415
LINE_TOP = 88
LOCALITY_H = 647
FILTER_H = 659
MATRIX_TOP, ACCURACY_H = 143, 682


def pt(px):
    """Convert CSS pixels to points."""
    return px * 72 / CSS_DPI


# Type scale, in CSS pixels at 830 px display width.
TITLE_PX = 25
LABEL_PX = 18
TICK_PX = 17
VALUE_PX = 17
REFERENCE_PX = 17
CELL_PX = 19

# Shared meanings across all four figures.
TORUS = "#0072B2"  # Okabe-Ito blue: torus layouts, filled circle
LATLON = "#D55E00"  # Okabe-Ito vermillion: latitude-longitude layouts, square
INK = "#1A1A1A"  # titles, value labels
SOFT_INK = "#3D3D3D"  # ticks, axis lines, reference-line label
REFERENCE = "#595959"  # reference line
GRID = "#E3E3E3"  # non-essential guides
MATRIX_CMAP = "cividis_r"

# The only wording a figure may carry besides numbers and row or column names.
# Subtitles, notes, caveats and source lines belong in the README, not here.
FIGURE_TEXT_ROLES = {"title", "x_axis", "y_axis", "colourbar", "reference"}

FORBIDDEN = {
    "\u2013": "en dash",
    "\u2014": "em dash",
    "\u2018": "curly quote",
    "\u2019": "curly quote",
    "\u201c": "curly quote",
    "\u201d": "curly quote",
    "\u2212": "unicode minus",
}
NUMBER = re.compile(r"^\d{1,3}(,\d{3})*(\.\d+)?x?$")

BASE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": pt(TICK_PX),
    "axes.edgecolor": SOFT_INK,
    "axes.linewidth": 1.0,
    "axes.labelcolor": INK,
    "xtick.color": SOFT_INK,
    "ytick.color": SOFT_INK,
    "xtick.labelcolor": SOFT_INK,
    "ytick.labelcolor": SOFT_INK,
    "xtick.labelsize": pt(TICK_PX),
    "ytick.labelsize": pt(TICK_PX),
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3.5,
    "xtick.major.pad": 6,
    "ytick.major.pad": 8,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "savefig.bbox": None,
    "svg.fonttype": "path",
    "svg.hashsalt": "sphere-on-a-torus-figures",
    "path.simplify": False,
}


# ---------------------------------------------------------------- data


def read_csv(name):
    with (PACK / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, what):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is missing or not finite: {value!r}")
    return number


# ---------------------------------------------------------------- layout


class Canvas:
    """A figure laid out in CSS pixels measured from the top-left corner."""

    def __init__(self, height):
        self.height = height
        self.fig = plt.figure(
            figsize=(WIDTH / CSS_DPI, height / CSS_DPI), dpi=CSS_DPI
        )

    def axes(self, left, top, width, height):
        return self.fig.add_axes(
            [
                left / WIDTH,
                (self.height - top - height) / self.height,
                width / WIDTH,
                height / self.height,
            ]
        )

    def text(self, x, top, s, size, **kwargs):
        kwargs.setdefault("va", "top")
        kwargs.setdefault("ha", "left")
        kwargs.setdefault("color", SOFT_INK)
        return self.fig.text(
            x / WIDTH, 1 - top / self.height, s, fontsize=pt(size), **kwargs
        )


def heading(canvas, title, left=24):
    canvas.text(left, 22, title, TITLE_PX, color=INK, fontweight="bold")


def tidy(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def comma(value, _pos=None):
    return f"{value:,.0f}"


def times(value, decimals=None):
    if decimals is None:
        return f"{value:,.0f}x"
    return f"{value:,.{decimals}f}x"


def three_figures(value):
    """Three significant figures; values of 1,000 or more to the nearest unit."""
    if value >= 1000:
        return f"{value:,.0f}x"
    digits = 2 - int(math.floor(math.log10(abs(value))))
    return f"{round(value, digits):,.{max(digits, 0)}f}x"


# ---------------------------------------------------------------- checks


def contrast(a, b):
    def luminance(colour):
        rgb = mcolors.to_rgb(colour)
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def lstar(colour):
    rgb = mcolors.to_rgb(colour)
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return 116 * (y ** (1 / 3) if y > 216 / 24389 else (24389 / 27 * y + 16) / 116) - 16


def drawn_texts(fig):
    """Every text the renderer will actually draw, ticks outside the view excluded."""
    texts = []
    for ax in fig.axes:
        for axis, interval in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(interval)
            span = high - low
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not (low - 1e-9 * span <= loc <= high + 1e-9 * span):
                    continue
                for label in (tick.label1, tick.label2):
                    if label.get_visible() and label.get_text():
                        texts.append(label)
            for label in (axis.label,):
                if label.get_visible() and label.get_text():
                    texts.append(label)
        texts.extend(t for t in ax.texts if t.get_visible() and t.get_text())
    texts.extend(t for t in fig.texts if t.get_visible() and t.get_text())
    return texts


def marker_boxes(fig, renderer):
    boxes = []
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_gid() != "data" or line.get_marker() in (None, "None", ""):
                continue
            radius = line.get_markersize() * fig.dpi / 72 / 2 + line.get_markeredgewidth() * fig.dpi / 72 / 2
            for x, y in ax.transData.transform(np.column_stack(line.get_data())):
                boxes.append((x - radius, y - radius, x + radius, y + radius))
    return boxes


def line_samples(fig):
    samples = []
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_gid() not in ("data", "reference") or line.get_linestyle() in ("None", ""):
                continue
            points = ax.transData.transform(np.column_stack(line.get_data()))
            if line.get_gid() == "reference":
                # A reference line spans the axes; sample its drawn extent.
                x0, x1 = ax.get_window_extent().x0, ax.get_window_extent().x1
                points = np.array([[x0, points[0][1]], [x1, points[0][1]]])
            for (xa, ya), (xb, yb) in zip(points[:-1], points[1:]):
                steps = max(2, int(math.hypot(xb - xa, yb - ya)))
                for f in np.linspace(0, 1, steps):
                    samples.append((xa + f * (xb - xa), ya + f * (yb - ya)))
    return samples


def overlaps(a, b, pad=0.0):
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def audit_layout(canvas, words, approved, inside_ok=()):
    """Raise if any text is a note, unapproved, too small, clipped or colliding."""
    fig = canvas.fig
    extra = sorted(set(words) - FIGURE_TEXT_ROLES)
    if extra:
        raise ValueError(f"figure wording is not a title, axis or reference label: {extra}")
    free = [t.get_text() for t in fig.texts if t.get_visible() and t.get_text()]
    if free != [words["title"]]:
        raise ValueError(f"free-standing figure text other than the title: {free}")
    for ax in fig.axes:
        for t in ax.texts:
            s = t.get_text()
            if t.get_visible() and s and not NUMBER.match(s) and s != words.get("reference"):
                raise ValueError(f"text inside the axes is not a value or reference label: {s!r}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = CSS_DPI / fig.dpi
    texts = drawn_texts(fig)
    boxes = []
    report = {"texts": [], "min_text_px": None, "min_edge_gap_px": None}
    min_px = math.inf
    min_gap = math.inf
    for t in texts:
        s = t.get_text()
        for char, name in FORBIDDEN.items():
            if char in s:
                raise ValueError(f"{name} in figure text {s!r}")
        if s not in approved and not NUMBER.match(s):
            raise ValueError(f"unapproved figure text {s!r}")
        px = t.get_fontsize() * CSS_DPI / 72
        if px < MIN_TEXT_PX - 1e-9:
            raise ValueError(f"text {s!r} is {px:.1f} px, below {MIN_TEXT_PX} px")
        min_px = min(min_px, px)
        ext = t.get_window_extent(renderer)
        box = (ext.x0 * scale, ext.y0 * scale, ext.x1 * scale, ext.y1 * scale)
        gap = min(box[0], box[1], WIDTH - box[2], canvas.height - box[3])
        if gap < 6:
            raise ValueError(f"text {s!r} is {gap:.1f} px from the canvas edge")
        min_gap = min(min_gap, gap)
        boxes.append((t, box))
        report["texts"].append({"text": s, "px": round(px, 2), "box_css_px": [round(v, 1) for v in box]})
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1 :]:
            if overlaps(a, b, pad=2):
                raise ValueError(f"texts collide: {ta.get_text()!r} and {tb.get_text()!r}")
    markers = [tuple(v * scale for v in m) for m in marker_boxes(fig, renderer)]
    samples = [(x * scale, y * scale) for x, y in line_samples(fig)]
    for t, box in boxes:
        if t.get_text() in inside_ok:
            continue
        for m in markers:
            if overlaps(box, m, pad=2):
                raise ValueError(f"text {t.get_text()!r} covers a data marker")
        for x, y in samples:
            if box[0] - 2 < x < box[2] + 2 and box[1] - 2 < y < box[3] + 2:
                raise ValueError(f"text {t.get_text()!r} crosses a data line")
    report["min_text_px"] = round(min_px, 2)
    report["min_edge_gap_px"] = round(min_gap, 1)
    report["text_count"] = len(boxes)
    report["checks"] = [
        "no notes: only the title, axis labels, tick labels, row and column names, value labels and the reference-line label",
        "only approved wording or numbers",
        "no em dash, en dash, curly quote or unicode minus",
        f"every text at least {MIN_TEXT_PX} CSS px",
        "every text at least 6 CSS px inside the canvas",
        "no two texts overlap (2 px clearance)",
        "no text over a data marker or data line (2 px clearance)",
    ]
    return report


def check_data_inside(ax, xs, ys):
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    for x, y in zip(xs, ys):
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise ValueError(f"data point ({x}, {y}) lies outside the axes")


# ---------------------------------------------------------------- figures


def throughput():
    rows = read_csv("throughput.csv")
    labels = [r["layout"] for r in rows]
    values = [finite(r["relative"], "relative throughput") for r in rows]
    if labels != ["Polar-limited lat-lon", "Torus, patch work counted", "Torus, shared patch work"]:
        raise ValueError(f"unexpected throughput rows {labels}")
    words = {
        "title": "Whole-sphere throughput without a polar filter",
        "x_axis": "Simulated time per wall time / polar-limited lat-lon",
    }
    styles = [
        dict(marker="s", color=LATLON, mfc=LATLON, mec=LATLON),
        dict(marker="o", color=TORUS, mfc=TORUS, mec=TORUS),
        dict(marker="D", color=TORUS, mfc="white", mec=TORUS),
    ]
    value_text = [times(v) for v in values]

    canvas = Canvas(height=THROUGHPUT_H)
    heading(canvas, words["title"])
    left, top, width, height = 262, THROUGHPUT_TOP, 512, 236
    ax = canvas.axes(left, top, width, height)
    tidy(ax, left=False)
    ax.set_xscale("log", base=10)
    ax.set_xlim(1, 2000)
    ax.set_ylim(len(rows) - 0.45, -0.75)
    ax.xaxis.set_major_locator(mticker.FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(comma))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks(range(len(rows)), labels, color=INK)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.grid(axis="x", which="major", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for i, (value, style, label) in enumerate(zip(values, styles, value_text)):
        ax.axhline(i, color=GRID, linewidth=1.0, zorder=0)
        ax.plot([value], [i], linestyle="None", markersize=15, markeredgewidth=2.6,
                clip_on=False, zorder=3, gid="data", **style)
        ax.annotate(label, (value, i), xytext=(0, 15), textcoords="offset points",
                    ha="center", va="bottom", fontsize=pt(VALUE_PX), fontweight="bold",
                    color=INK)
    ax.set_xlabel(words["x_axis"], fontsize=pt(LABEL_PX), labelpad=10)
    check_data_inside(ax, values, range(len(rows)))

    approved = set(words.values()) | set(labels)
    layout = audit_layout(canvas, words, approved)
    meta = {
        "source_files": ["figure-pack/throughput.csv"],
        "columns_used": {"layout": "row label, verbatim", "relative": "x position"},
        "plotted_values": [
            {"layout": l, "relative": v, "label": t} for l, v, t in zip(labels, values, value_text)
        ],
        "transformations": [
            "relative is plotted unchanged on a log10 x axis that starts at 1 and ends at 2,000",
            "value labels are relative rounded to whole numbers with a thousands separator and an x suffix",
            "steps_per_s is not drawn; it is kept in the source file",
        ],
        "encoding": {
            "Polar-limited lat-lon": "vermillion filled square, the latitude-longitude reference",
            "Torus, patch work counted": "blue filled circle, counted-work model",
            "Torus, shared patch work": "blue open diamond, additional layout estimate",
            "row guides": "light grey horizontal lines for reading only; they are not bars and encode nothing",
        },
        "status": "every value is a model output: the paper's performance model applied to 64 CS-3 systems at 15 arcseconds. No hardware run.",
        "alt_text": (
            "Dot plot on a log scale of modelled whole-sphere throughput relative to a polar-limited "
            "latitude-longitude grid on 64 CS-3 systems at 15 arcseconds. Polar-limited lat-lon is 1x, "
            "the torus with patch work counted is 856x, and the torus with shared patch work, an "
            "additional layout estimate shown as an open diamond, is 1,263x."
        ),
    }
    return "throughput", canvas, words, meta, layout


def locality():
    rows = read_csv("stable_step.csv")
    ms = [int(r["m"]) for r in rows]
    rs = [finite(r["r"], "stable step ratio") for r in rows]
    if ms != [64, 128, 256]:
        raise ValueError(f"unexpected torus widths {ms}")
    words = {
        "title": "The poles no longer collapse the stable step",
        "x_axis": "Torus width M (computed points = M squared)",
        "y_axis": "Stable step / equatorial wave-crossing time",
    }
    value_text = [f"{r:.2f}" for r in rs]

    canvas = Canvas(height=LOCALITY_H)
    heading(canvas, words["title"])
    left, top, width, height = 112, LINE_TOP, 680, 468
    ax = canvas.axes(left, top, width, height)
    tidy(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlim(64 / 1.45, 256 * 1.45)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ms))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(comma))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(0, 3)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.plot(ms, rs, color=TORUS, linewidth=2.6, marker="o", markersize=15,
            mfc=TORUS, mec=TORUS, zorder=3, gid="data")
    for m, r, label in zip(ms, rs, value_text):
        ax.annotate(label, (m, r), xytext=(0, 15), textcoords="offset points",
                    ha="center", va="bottom", fontsize=pt(VALUE_PX), fontweight="bold",
                    color=INK)
    ax.set_xlabel(words["x_axis"], fontsize=pt(LABEL_PX), labelpad=10)
    ax.set_ylabel(words["y_axis"], fontsize=pt(LABEL_PX), labelpad=12)
    check_data_inside(ax, ms, rs)

    approved = set(words.values())
    layout = audit_layout(canvas, words, approved)
    meta = {
        "source_files": ["figure-pack/stable_step.csv"],
        "columns_used": {"m": "x position", "r": "y position and value label"},
        "plotted_values": [
            {"M": m, "r": r, "label": t} for m, r, t in zip(ms, rs, value_text)
        ],
        "r_definition": "r is the largest stable step from the TC5 bisection divided by the equatorial wave-crossing time.",
        "r_denominator": (
            "Equatorial spacing is from a lat-lon grid with the same computed-point count; "
            "wave speed is the initial maximum sqrt(gD) + |v|. Bisection tolerance is 2%."
        ),
        "transformations": [
            "r is plotted unchanged; value labels are r rounded to two decimals",
            "x is log2 with ticks at the three tested widths; y is linear from 0 to 3",
            "the connecting line only joins the three measured widths in order; it is not a fit or an interpolation claim",
        ],
        "encoding": {"torus": "blue filled circles joined by a blue line"},
        "status": "deterministic stability bisection results; one value per width, no replicates, no interval",
        "alt_text": (
            "Line and point plot of the largest stable step divided by the equatorial wave-crossing time "
            "for torus widths M of 64, 128 and 256 in the TC5 case over 50 simulated days. The ratio is "
            "2.45, 2.32 and 2.25: it stays bounded near 2.3 instead of collapsing as the grid refines, "
            "though it is not constant."
        ),
    }
    return "locality", canvas, words, meta, layout


def accuracy():
    rows = read_csv("accuracy.csv")
    order = ["TC2, day 5", "TC5, day 15", "TC6, day 5", "TC6, day 14",
             "Steady jet, day 6", "Perturbed jet, day 6"]
    widths = [144, 288, 576]
    columns = ["10k cells", "41k cells", "165k cells"]
    table = {}
    for r in rows:
        ratio = finite(r["ratio"], f"ratio for {r['case']}")
        recomputed = finite(r["trsk_l2"], "trsk_l2") / finite(r["torus_l2"], "torus_l2")
        if not math.isclose(ratio, recomputed, rel_tol=1e-12):
            raise ValueError(f"ratio column disagrees with trsk_l2 / torus_l2 for {r['case']}")
        table[(r["case"], int(r["m"]))] = r
    if sorted(table) != sorted((c, m) for c in order for m in widths):
        raise ValueError("accuracy.csv does not hold exactly the 18 expected pairs")
    values = np.array([[float(table[(c, m)]["ratio"]) for m in widths] for c in order])
    cells = [[int(table[(c, m)]["trsk_cells"]) for m in widths] for c in order]
    unique = [[int(table[(c, m)]["torus_unique"]) for m in widths] for c in order]
    for c in order:
        for m in widths:
            if int(table[(c, m)]["torus_unique"]) * 2 != m * m:
                raise ValueError("torus_unique is not M squared / 2")

    words = {
        "title": "Lower height error in all 18 TRSK comparisons",
        "x_axis": "Approximate unique-cell count",
        "colourbar": "TRSK l2 / torus l2 (log scale)",
    }
    norm = mcolors.LogNorm(vmin=1.0, vmax=float(values.max()))
    cmap = matplotlib.colormaps[MATRIX_CMAP]
    if values.min() < norm.vmin or values.max() > norm.vmax:
        raise ValueError("colour limits would saturate a cell")

    canvas = Canvas(height=ACCURACY_H)
    heading(canvas, words["title"])
    left, top, width, height = 232, MATRIX_TOP, 552, 384
    ax = canvas.axes(left, top, width, height)
    for side in ax.spines.values():
        side.set_visible(False)
    mesh = ax.pcolormesh(np.arange(4) - 0.5, np.arange(7) - 0.5, values, cmap=cmap,
                         norm=norm, edgecolors="white", linewidth=3)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(5.5, -0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(3), columns, color=INK)
    ax.set_yticks(range(6), order, color=INK)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=12)
    ax.set_xlabel(words["x_axis"], fontsize=pt(LABEL_PX), labelpad=12)
    cell_report = []
    for i, case in enumerate(order):
        for j in range(3):
            fill = cmap(norm(values[i, j]))
            options = {"#111111": contrast("#111111", fill), "#FFFFFF": contrast("#FFFFFF", fill)}
            text_colour = max(options, key=options.get)
            label = three_figures(values[i, j])
            ax.text(j, i, label, ha="center", va="center", fontsize=pt(CELL_PX),
                    fontweight="bold", color=text_colour)
            cell_report.append({
                "case": case, "column": columns[j], "ratio": float(values[i, j]), "label": label,
                "fill": mcolors.to_hex(fill), "text": text_colour,
                "contrast": round(options[text_colour], 2),
            })
    worst = min(c["contrast"] for c in cell_report)
    if worst < 4.5:
        raise ValueError(f"a cell label reaches only {worst}:1 contrast")

    cax = canvas.axes(left, MATRIX_TOP + height + 40, width, 24)
    bar = canvas.fig.colorbar(mesh, cax=cax, orientation="horizontal")
    bar.outline.set_edgecolor(SOFT_INK)
    bar.outline.set_linewidth(1.0)
    cax.xaxis.set_major_locator(mticker.FixedLocator([1, 10, 100, 1000]))
    cax.xaxis.set_major_formatter(mticker.FuncFormatter(comma))
    cax.xaxis.set_minor_formatter(mticker.NullFormatter())
    cax.tick_params(axis="x", which="major", length=6, color=SOFT_INK)
    cax.tick_params(axis="x", which="minor", length=3.5, color=SOFT_INK)
    bar.set_label(words["colourbar"], fontsize=pt(LABEL_PX), color=INK, labelpad=10)

    approved = set(words.values()) | set(order) | set(columns)
    layout = audit_layout(canvas, words, approved)
    layout["cell_label_contrast_min"] = worst
    meta = {
        "source_files": ["figure-pack/accuracy.csv"],
        "columns_used": {
            "case": "row label, verbatim",
            "m": "column (144, 288, 576)",
            "ratio": "cell colour and label; checked equal to trsk_l2 / torus_l2",
            "trsk_cells, torus_unique": "column labels",
        },
        "row_order": order,
        "columns": [
            {"label": columns[j], "M": widths[j], "trsk_cells": cells[0][j], "torus_unique": unique[0][j]}
            for j in range(3)
        ],
        "cells": cell_report,
        "colour": {
            "map": MATRIX_CMAP,
            "norm": "LogNorm",
            "vmin": 1.0,
            "vmax": float(values.max()),
            "reason": "vmin is parity, vmax is the largest ratio, so no cell is clipped or saturated",
        },
        "label_rounding": "three significant figures; values of 1,000 or more to the nearest whole number",
        "missing_values": (
            "torus_linf and trsk_linf are blank for the three TC6, day 5 rows; the figure uses only l2, "
            "which is present for all 18 pairs"
        ),
        "status": "deterministic l2 height errors from single runs; no replicates, no interval",
        "scope": "TRSK as implemented in this repository on generated icosahedral meshes; MPAS itself was not run",
        "alt_text": (
            "Colour matrix of TRSK l2 height error divided by torus l2 height error, six test cases by "
            "three resolutions of about 10k, 41k and 165k unique cells. All 18 ratios are above 1, so the "
            "torus has lower error everywhere. Smallest are TC6 at day 14 (1.16x, 1.39x, 1.83x); largest "
            "is the steady jet at 165k cells (3,182x). TC2 runs 99.7x to 659x."
        ),
    }
    return "accuracy", canvas, words, meta, layout


def filter_trade():
    rows = read_csv("filter_trade.csv")
    arcsec = [int(r["arcsec"]) for r in rows]
    ratio = [finite(r["filter_over_torus"], "filter ratio") for r in rows]
    for r in rows:
        recomputed = finite(r["filter_steps_per_s"], "filter rate") / finite(r["torus_steps_per_s"], "torus rate")
        if not math.isclose(recomputed, float(r["filter_over_torus"]), rel_tol=1e-9):
            raise ValueError(f"filter_over_torus disagrees with the step rates at {r['arcsec']}")
    if arcsec != [15, 30, 60, 120, 240, 480, 960]:
        raise ValueError(f"unexpected resolutions {arcsec}")
    words = {
        "title": "A polar filter can win while bandwidth dominates",
        "x_axis": "Lat-lon grid spacing (arcseconds)",
        "y_axis": "Filtered lat-lon throughput / torus throughput",
        "reference": "Equal throughput",
    }
    value_text = [times(v, 2) for v in ratio]

    canvas = Canvas(height=FILTER_H)
    heading(canvas, words["title"])
    left, top, width, height = 112, LINE_TOP, 680, 480
    ax = canvas.axes(left, top, width, height)
    tidy(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlim(15 / 1.4, 960 * 1.4)
    ax.xaxis.set_major_locator(mticker.FixedLocator(arcsec))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(comma))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(0, 1.6)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=REFERENCE, linewidth=1.8, linestyle=(0, (6, 4)), zorder=2, gid="reference")
    ax.plot(arcsec, ratio, color=LATLON, linewidth=2.6, marker="s", markersize=13,
            mfc=LATLON, mec=LATLON, zorder=3, gid="data")
    for a, r, label in zip(arcsec, ratio, value_text):
        below = r < 1.0
        ax.annotate(label, (a, r), xytext=(0, -15 if below else 14), textcoords="offset points",
                    ha="center", va="top" if below else "bottom", fontsize=pt(VALUE_PX),
                    fontweight="bold", color=INK)
    ax.annotate(words["reference"], (0.985, 1.0), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 7), textcoords="offset points", ha="right", va="bottom",
                fontsize=pt(REFERENCE_PX), color=SOFT_INK)
    ax.set_xlabel(words["x_axis"], fontsize=pt(LABEL_PX), labelpad=10)
    ax.set_ylabel(words["y_axis"], fontsize=pt(LABEL_PX), labelpad=12)
    check_data_inside(ax, arcsec, ratio)

    approved = set(words.values())
    layout = audit_layout(canvas, words, approved)
    meta = {
        "source_files": ["figure-pack/filter_trade.csv"],
        "columns_used": {"arcsec": "x position", "filter_over_torus": "y position and value label"},
        "plotted_values": [
            {"arcsec": a, "filter_over_torus": r, "label": t, "filter_limit": row["filter_limit"]}
            for a, r, t, row in zip(arcsec, ratio, value_text, rows)
        ],
        "transformations": [
            "filter_over_torus is plotted unchanged and checked equal to filter_steps_per_s / torus_steps_per_s",
            "x is log2 with every tested spacing labelled; y is linear from 0 to 1.6",
            "value labels are the ratio rounded to two decimals with an x suffix",
            "the dashed horizontal line marks a ratio of exactly 1",
            "the connecting line joins the seven modelled spacings in order; the model's switch from an io limit to a latency limit between 120 and 240 arcseconds is recorded in filter_limit, not drawn",
        ],
        "encoding": {
            "filtered lat-lon over torus": "vermillion filled squares joined by a vermillion line (latitude-longitude family)",
            "equal throughput": "grey dashed line at 1, labelled directly",
        },
        "status": "every value is a model output from an optimistic performance model; no hardware run, no interval",
        "alt_text": (
            "Line plot of modelled throughput of a latitude-longitude grid with a polar filter divided by "
            "torus throughput, for grid spacings from 15 to 960 arcseconds. The filtered grid is ahead by "
            "1.41x at 15 arcseconds, about 1.40x to 1.33x up to 240 arcseconds, then falls below the equal "
            "throughput line to 0.72x at 480 and 0.62x at 960 arcseconds."
        ),
    }
    return "filter-trade", canvas, words, meta, layout


# ---------------------------------------------------------------- export


def atomic_write(path, data):
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


def render_png(fig, dpi, info):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    image = Image.open(buffer).convert("RGB")
    meta = PngImagePlugin.PngInfo()
    for key, value in info.items():
        meta.add_text(key, value)
    out = io.BytesIO()
    image.save(out, format="PNG", pnginfo=meta, optimize=False, dpi=(dpi, dpi))
    return out.getvalue(), image


def export(name, canvas, words, meta, layout):
    fig = canvas.fig
    alt = meta["alt_text"]
    info = {"Title": words["title"], "Description": alt, "Software": f"scripts/make_figures.py, Matplotlib {matplotlib.__version__}"}
    svg = io.BytesIO()
    # Matplotlib rasterises a colour-bar gradient inside the SVG; 4x CSS
    # density keeps it sharp on high-density screens.
    fig.savefig(svg, format="svg", facecolor="white", dpi=CSS_DPI * 4,
                metadata={"Title": words["title"], "Description": alt, "Date": None,
                          "Creator": info["Software"]})
    files = {}
    targets = {
        OUT / f"{name}.svg": svg.getvalue(),
    }
    png, _ = render_png(fig, CSS_DPI * PNG_SCALE, info)
    targets[OUT / f"{name}.png"] = png
    for path, data in targets.items():
        atomic_write(path, data)
        files[str(path.relative_to(ROOT))] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    provenance = {
        "figure": name,
        "wording": words,
        "source_data": {
            p: {"sha256": sha256(ROOT / p)} for p in meta.pop("source_files")
        },
        "source_origin": "figure-pack/ is written by scripts/prepare_figures.py from the saved measurements in data/ (checked against data/manifest.json); no earlier plot was read",
        "uncertainty": "none shown and none exists: values are deterministic measured errors or model outputs, not replicate estimates",
        **meta,
        "render": {
            "script": "scripts/make_figures.py",
            "canvas_css_px": [WIDTH, canvas.height],
            "svg": "622.5 pt wide, which is 830 CSS px; text converted to paths so it renders identically without DejaVu Sans installed; a colour-bar gradient, where present, is embedded as a raster at 4x CSS density",
            "png": f"{WIDTH * PNG_SCALE} x {canvas.height * PNG_SCALE} px, RGB, no alpha, for display at 830 CSS px",
            "font": "DejaVu Sans (bundled with Matplotlib)",
            "background": "opaque white",
            "versions": {
                "python": platform.python_version(),
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "pillow": Image.__version__,
            },
        },
        "layout_audit": layout,
        "outputs": files,
    }
    atomic_write(OUT / f"{name}.provenance.json",
                 (json.dumps(provenance, indent=2, ensure_ascii=True) + "\n").encode())
    plt.close(fig)
    return provenance


def main():
    built = []
    with plt.style.context("default"), matplotlib.rc_context(BASE_RC):
        for make in (throughput, locality, accuracy, filter_trade):
            name, canvas, words, meta, layout = make()
            export(name, canvas, words, meta, layout)
            built.append(name)
    print("Wrote", ", ".join(built), "to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
