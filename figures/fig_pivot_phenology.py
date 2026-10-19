#!/usr/bin/env python3

from __future__ import annotations

import sys

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator
from matplotlib.transforms import Bbox

IN = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "."

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "axes.labelpad": 2,
        "pdf.fonttype": 42,
    }
)
_fp = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
assert "dejavu" not in _fp.lower(), f"DejaVu fallback: Nimbus Sans missing ({_fp})"


STYLE = {
    "pivot": ("Irrigated center pivots", "#009E73", "#00684C", "o", 2.0, 4),
    "rainfed": ("Rainfed cereal", "#E69F00", "#9A6A00", "s", 1.2, 3),
    "fallow": ("Fallow", "#D55E00", "#8F3F00", "^", 1.2, 2),
}
ORDER = ("pivot", "rainfed", "fallow")
SEASONS = (
    ("early", "Early", "2025-04-01", "2025-06-01", "0.90"),
    ("peak", "Peak", "2025-06-01", "2025-09-01", "0.84"),
    ("late", "Late", "2025-09-01", "2025-11-01", "0.90"),
)
YLIM = (-0.04, 1.10)
MIN_OBS = 15


def style_axis(ax: Axes, xminor: bool = True) -> None:
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    if xminor and not isinstance(ax.xaxis.get_minor_locator(), FixedLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True, pad=2)
    ax.tick_params(which="major", length=4.0)
    ax.tick_params(which="minor", length=2.2)
    ax.grid(axis="y", which="major", lw=0.5, color="0.86")
    ax.set_axisbelow(True)
    ax.set_ylim(*YLIM)


def covers_data(bb: Bbox, ax: Axes) -> int:

    def hits(xy: npt.ArrayLike) -> int:
        p = ax.transData.transform(np.asarray(xy, float))
        inside = (p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1)
        inside &= (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)
        return int(inside.sum())

    n = 0
    for ln in ax.get_lines():
        xy = np.asarray(ln.get_xydata(), float)
        if xy.shape[0] > 2:
            n += hits(xy)
    for col in ax.collections:
        off = col.get_offsets()
        if len(off):
            n += hits(off)
        for pth in col.get_paths():
            if len(pth.vertices):
                n += hits(pth.vertices)
    for pch in ax.patches:
        if pch.get_gid() == "season":
            continue
        vs = pch.get_path().transformed(pch.get_patch_transform()).vertices
        n += hits(vs)
    return n


def main() -> None:
    dek = pd.read_csv(f"{IN}/pheno_dekad.csv", parse_dates=["dekad_start"])
    win = pd.read_csv(f"{IN}/pheno_windows.csv")
    nfield = win[win["season"] == "peak"].groupby("class").size()

    fig = plt.figure(figsize=(6.9, 3.25), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0])
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1], sharey=ax)

    for _key, lab, a, b, grey in SEASONS:
        ax.axvspan(
            np.datetime64(a),
            np.datetime64(b),
            color=grey,
            alpha=0.35,
            lw=0,
            gid="season",
        )
        mid = np.datetime64(a) + (np.datetime64(b) - np.datetime64(a)) / 2
        ax.text(mid, 1.045, lab, ha="center", va="center", fontsize=8.5, color="0.25")
    handles = []
    for cls in ORDER[::-1]:
        lab, col, dark, mk, lw, z = STYLE[cls]
        s = dek[(dek["class"] == cls) & (dek["n_obs"] >= MIN_OBS)].sort_values(
            "dekad_start"
        )
        x = s["dekad_start"] + pd.Timedelta(days=5)
        ax.fill_between(x, s["q25"], s["q75"], color=col, alpha=0.22, lw=0, zorder=z)
        (h,) = ax.plot(
            x,
            s["median"],
            color=dark,
            lw=lw,
            marker=mk,
            ms=3.6,
            mfc=col,
            mec=dark,
            mew=0.5,
            zorder=z + 0.5,
            label=f"{lab} (n = {nfield[cls]})",
        )
        handles.append(h)
    ax.set_xlim(np.datetime64("2025-03-01"), np.datetime64("2025-12-01"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[11, 21]))
    style_axis(ax, xminor=False)
    ax.set_xlabel("Date (2025)")
    ax.set_ylabel("NDVI")

    rng = np.random.default_rng(1)
    width, pos_major = 0.26, []
    minor_pos = []
    for g, (key, lab, *_rest) in enumerate(SEASONS):
        pos_major.append(g)
        for k, cls in enumerate(ORDER):
            _l, col, dark, mk, _lw, z = STYLE[cls]
            v = win[(win["season"] == key) & (win["class"] == cls)]["ndvi"].to_numpy()
            p = g + (k - 1) * width
            minor_pos.append(p)
            jit = rng.uniform(-width * 0.30, width * 0.30, v.size)
            bx.scatter(
                p + jit,
                v,
                s=5,
                marker=mk,
                facecolor=col,
                edgecolor="none",
                alpha=0.45,
                zorder=z,
            )
            bp = bx.boxplot(
                v,
                positions=[p],
                widths=width * 0.86,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                medianprops={"color": dark, "lw": 1.2},
                boxprops={"facecolor": "none", "edgecolor": dark, "lw": 0.8},
                whiskerprops={"color": dark, "lw": 0.8},
                capprops={"color": dark, "lw": 0.8},
            )
            for art in (*bp["boxes"], *bp["medians"], *bp["whiskers"], *bp["caps"]):
                art.set_zorder(z + 5)
    bx.set_xlim(-0.5, len(SEASONS) - 0.5)
    bx.xaxis.set_major_locator(FixedLocator(pos_major))
    bx.xaxis.set_minor_locator(FixedLocator(minor_pos))
    bx.set_xticklabels([s[1] for s in SEASONS])
    style_axis(bx, xminor=False)
    bx.tick_params(axis="y", labelleft=False)
    bx.set_xlabel("Compositing window")

    tags = [
        a.text(
            0.018,
            0.975,
            t,
            transform=a.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )
        for a, t in ((ax, "(a)"), (bx, "(b)"))
    ]
    leg = fig.legend(
        handles=handles[::-1],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for a, t in zip((ax, bx), tags):
        print(f"tag {t.get_text()} hits:", covers_data(Text.get_window_extent(t, r), a))
    for t in ax.texts[:3]:
        print(
            f"season label {t.get_text()} hits:",
            covers_data(Text.get_window_extent(t, r), ax),
        )
    lb = leg.get_window_extent(r)
    print("legend hits:", covers_data(lb, ax) + covers_data(lb, bx))
    print(f"fields: {dict(nfield)}")

    for ext in ("pdf", "png"):
        fig.savefig(
            f"{OUT}/fig_pivot_phenology.{ext}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("written", OUT)


if __name__ == "__main__":
    main()
