#!/usr/bin/env python3

from __future__ import annotations

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter

INV = sys.argv[1] if len(sys.argv) > 1 else "pivots_2025.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
_fp = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
assert "dejavu" not in _fp.lower(), f"DejaVu fallback: Nimbus Sans missing ({_fp})"


SPLITS = [
    ("train", "Training", "#0072B2"),
    ("val", "Validation", "#E69F00"),
    ("test", "Test", "#009E73"),
]
MARK = {
    "candidate": ("o", "Hough candidate, accepted"),
    "added": ("^", "Added by hand"),
}


CHAIN = [
    ("Hough candidates", 1454, "#7F7F7F"),
    ("flagged for review (C < 0.70)", 982, "#7F7F7F"),
    ("accepted on review", 17, "#0072B2"),
    ("added by hand", 15, "#0072B2"),
    ("manual labels (released)", 32, "#000000"),
    ("automatic labels", 370, "#D55E00"),
    ("automatic, no manual match", 263, "#D55E00"),
]


def panel_a(ax: Axes, df: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    med_all = float(df["diam_m"].median())
    ax.axvline(med_all, color="#333333", lw=2.0, ls=(0, (4, 2)), zorder=1)
    ax.text(
        med_all + 12,
        2.62,
        f"median {med_all:.0f} m",
        ha="left",
        va="center",
        fontsize=7.2,
        color="#333333",
    )
    for k, (key, name, col) in enumerate(SPLITS):
        sub = df[df["split"] == key]
        y0 = 2 - k
        for src, (mk, _) in MARK.items():
            s = sub[sub["source"] == src]
            jit = rng.uniform(-0.2, 0.2, len(s))
            ax.scatter(
                s["diam_m"],
                y0 + jit,
                s=30,
                marker=mk,
                facecolor=col,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )
        med = float(sub["diam_m"].median())
        ax.plot(
            [med, med],
            [y0 - 0.34, y0 + 0.34],
            color=col,
            lw=2.0,
            zorder=2,
            solid_capstyle="butt",
        )
    ax.set_yticks(
        [2, 1, 0],
        [f"{nm}\n(n = {(df['split'] == k).sum()})" for k, nm, _ in SPLITS],
    )
    ax.set_ylim(-0.6, 2.85)
    ax.set_xlim(250, 1100)
    ax.set_xlabel("Equivalent diameter $D_{eq}$ (m)")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(FixedLocator([0.5, 1.5]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color="#D9D9D9", lw=0.5)
    ax.set_axisbelow(True)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=m,
            mfc="white",
            mec="black",
            mew=0.6,
            ms=5.5,
            label=lab,
        )
        for m, lab in MARK.values()
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
        borderaxespad=0.2,
    )


def panel_b(ax: Axes) -> None:
    ys = np.arange(len(CHAIN))[::-1]
    for y, (lab, n, col) in zip(ys, CHAIN, strict=True):
        emph = lab.startswith("manual")
        ax.plot([1, n], [y, y], color=col, lw=2.0 if emph else 1.2, zorder=2)
        ax.scatter(
            [n],
            [y],
            s=42 if emph else 30,
            marker="D" if emph else "o",
            color=col,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            n * 1.25,
            y,
            f"{n:,}",
            ha="left",
            va="center",
            fontsize=7.4,
            color=col,
            fontweight="bold" if emph else "normal",
        )
    ax.set_yticks(ys, [c[0] for c in CHAIN])
    ax.set_xscale("log")
    ax.set_xlim(1, 5000)
    ax.set_ylim(-0.6, len(CHAIN) + 0.05)
    ax.set_xlabel("Number of objects (log scale)")
    ax.yaxis.set_minor_locator(FixedLocator(ys[:-1] - 0.5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", which="major", color="#D9D9D9", lw=0.5)
    ax.set_axisbelow(True)

    ax.axhline(1.5, color="#BFBFBF", lw=0.7, ls=":", zorder=1)
    ax.text(
        4600,
        0.5,
        "automatic (for comparison)",
        ha="right",
        va="center",
        fontsize=6.8,
        style="italic",
        color="#D55E00",
    )
    ax.text(
        4600,
        len(CHAIN) - 0.12,
        "manual labelling chain",
        ha="right",
        va="top",
        fontsize=6.8,
        style="italic",
        color="#333333",
    )


def main() -> None:
    df = pd.read_csv(INV)
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(7.4, 2.75),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.08]},
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.06)
    panel_a(a, df)
    panel_b(b)
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.text(
            0.012,
            0.975,
            tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
    for ext in ("pdf", "png"):
        fig.savefig(
            f"{OUT}/fig_label_statistics.{ext}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print(f"{len(df)} pivots; saved {OUT}/fig_label_statistics.pdf / .png")


if __name__ == "__main__":
    main()
