#!/usr/bin/env python3

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.textpath import TextPath

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.cal": "cmsy10",
        "mathtext.fallback": "cm",
        "pdf.fonttype": 42,
    }
)
_fp = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
assert "dejavu" not in _fp.lower(), f"DejaVu fallback: Nimbus Sans missing ({_fp})"

FS = 8.0
KIND = {
    "io": ("#EDEDED", "#555555"),
    "cnn": ("#D6E6F4", "#0072B2"),
    "vit": ("#F4DDEB", "#A3437E"),
    "neck": ("#FBE8C8", "#B07800"),
    "head": ("#D2EFE4", "#007A5A"),
    "inst": ("#F8DACB", "#B34700"),
}
LEGEND = [
    ("io", "Input / output"),
    ("cnn", "CNN encoder"),
    ("vit", "Transformer encoder"),
    ("neck", "Neck (ASPP, FPN, MLP)"),
    ("head", "Decoder / head"),
    ("inst", "Instance branch"),
]
W_FIG, PAD_X, GAP, TITLE = 6.9, 0.05, 0.15, 0.22


def text_w(s: str, size: float = FS, bold: bool = False) -> float:
    prop = FontProperties(family="Nimbus Sans", weight="bold" if bold else "normal")
    return float(TextPath((0, 0), s, size=size, prop=prop).get_extents().width) / 72


def sq(c: int, r: int) -> str:
    return rf"{c}$\times${r}$^2$"


@dataclass
class Box:
    name: str
    shape: str
    kind: str
    res: int
    x: float = 0.0
    w: float = 0.0
    h: float = 0.0
    extra: list[str] = field(default_factory=list)


@dataclass
class Row:
    title: str
    boxes: list[Box]
    skips: list[tuple[int, int, str]] = field(default_factory=list)


def height(res: int) -> float:
    return 0.34 + 0.03 * np.log2(max(res, 4) / 4)


ROWS = [
    Row(
        "(a) U-Net, ResNet-34 encoder",
        [
            Box("Input", sq(24, 256), "io", 256),
            Box("Stem", sq(64, 128), "cnn", 128),
            Box("Layer 1", sq(64, 64), "cnn", 64),
            Box("Layer 2", sq(128, 32), "cnn", 32),
            Box("Layer 3", sq(256, 16), "cnn", 16),
            Box("Layer 4", sq(512, 8), "cnn", 8),
            Box("Up 1", sq(256, 16), "head", 16),
            Box("Up 2", sq(128, 32), "head", 32),
            Box("Up 3", sq(64, 64), "head", 64),
            Box("Up 4", sq(32, 128), "head", 128),
            Box("Up 5", sq(16, 256), "head", 256),
            Box("Mask", sq(1, 256), "io", 256),
        ],
        [(4, 6, ""), (3, 7, ""), (2, 8, ""), (1, 9, "skip (concat)")],
    ),
    Row(
        "(b) DeepLabV3+, ResNet-50 encoder (output stride 16)",
        [
            Box("Input", sq(24, 256), "io", 256),
            Box("Stem", sq(64, 128), "cnn", 128),
            Box("Layer 1", sq(256, 64), "cnn", 64),
            Box("Layer 2", sq(512, 32), "cnn", 32),
            Box("Layer 3", sq(1024, 16), "cnn", 16),
            Box("Layer 4, dil. 2", sq(2048, 16), "cnn", 16),
            Box("ASPP 6/12/18", sq(256, 16), "neck", 16),
            Box(r"Up $\times$4", sq(256, 64), "head", 64),
            Box(r"Concat + 3$\times$3", sq(256, 64), "head", 64),
            Box(r"Head, up $\times$4", sq(1, 256), "io", 256),
        ],
        [(2, 8, r"low-level 48$\times$64$^2$")],
    ),
    Row(
        "(c) SegFormer, Mix Transformer MiT-B2 encoder",
        [
            Box("Input", sq(24, 256), "io", 256),
            Box("Stage 1", sq(64, 64), "vit", 64),
            Box("Stage 2", sq(128, 32), "vit", 32),
            Box("Stage 3", sq(320, 16), "vit", 16),
            Box("Stage 4", sq(512, 8), "vit", 8),
            Box(r"MLP, up to 64$^2$", sq(768, 64) + r" $\times$4", "neck", 64),
            Box("Fuse MLP", sq(768, 64), "head", 64),
            Box("Classifier", sq(1, 64), "head", 64),
            Box(r"Up $\times$4", sq(1, 256), "io", 256),
        ],
        [(1, 5, "all stages"), (2, 5, ""), (3, 5, "")],
    ),
    Row(
        "(d) Mask R-CNN, ResNet-50 feature pyramid network",
        [
            Box("Input", sq(24, 256), "io", 256),
            Box("C2", sq(256, 64), "cnn", 64),
            Box("C3", sq(512, 32), "cnn", 32),
            Box("C4", sq(1024, 16), "cnn", 16),
            Box("C5", sq(2048, 8), "cnn", 8),
            Box("FPN P2-P6", r"256$\times$64$^2$...4$^2$", "neck", 64),
            Box("RPN", "RoIs + scores", "inst", 16),
            Box("RoIAlign", sq(256, 14), "inst", 14),
            Box("Mask head", sq(256, 28), "inst", 28),
            Box("Masks + boxes", r"$K\times$28$^2$", "io", 28),
        ],
        [(1, 5, "lateral"), (2, 5, ""), (3, 5, "")],
    ),
]


def layout(row: Row) -> None:
    widths = [max(text_w(b.name, bold=True), text_w(b.shape)) + 0.08 for b in row.boxes]
    gap = GAP
    avail = W_FIG - 2 * PAD_X
    if sum(widths) + gap * (len(widths) - 1) > avail:
        gap = max(0.07, (avail - sum(widths)) / (len(widths) - 1))
    total = sum(widths) + gap * (len(widths) - 1)
    assert total <= avail + 1e-6, f"{row.title}: {total:.2f} > {avail:.2f} in"
    x = PAD_X + (avail - total) / 2
    for b, w in zip(row.boxes, widths):
        b.x, b.w, b.h = x, w, height(b.res)
        x += w + gap


def skip_band(row: Row) -> float:
    return 0.05 + 0.07 * len(row.skips) + 0.13 if row.skips else 0.0


def box_max(row: Row) -> float:
    return max(height(b.res) for b in row.boxes)


def draw_row(ax: Axes, row: Row, yc: float) -> None:
    for b in row.boxes:
        face, edge = KIND[b.kind]
        ax.add_patch(
            FancyBboxPatch(
                (b.x, yc - b.h / 2),
                b.w,
                b.h,
                boxstyle="round,pad=0,rounding_size=0.04",
                fc=face,
                ec=edge,
                lw=0.8,
                zorder=3,
            )
        )
        ax.text(
            b.x + b.w / 2,
            yc + 0.075,
            b.name,
            ha="center",
            va="center",
            fontsize=FS,
            fontweight="bold",
            color=edge,
            zorder=4,
        )
        ax.text(
            b.x + b.w / 2,
            yc - 0.085,
            b.shape,
            ha="center",
            va="center",
            fontsize=FS,
            color="#222222",
            zorder=4,
        )
    for a, b in zip(row.boxes[:-1], row.boxes[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (a.x + a.w, yc),
                (b.x, yc),
                arrowstyle="-|>",
                mutation_scale=7,
                lw=0.8,
                color="#333333",
                zorder=2,
                shrinkA=0,
                shrinkB=0,
            )
        )
    top = yc + max(b.h for b in row.boxes) / 2
    for k, (i, j, lab) in enumerate(sorted(row.skips, key=lambda s: s[1] - s[0])):
        s, d = row.boxes[i], row.boxes[j]
        ys, yd = yc + s.h / 2, yc + d.h / 2
        yl = top + 0.05 + 0.07 * k
        xs, xd = s.x + s.w / 2, d.x + d.w / 2
        ax.plot([xs, xs, xd], [ys, yl, yl], color="#7A7A7A", lw=0.7, zorder=1)
        ax.add_patch(
            FancyArrowPatch(
                (xd, yl),
                (xd, yd),
                arrowstyle="-|>",
                mutation_scale=6,
                lw=0.7,
                color="#7A7A7A",
                zorder=1,
                shrinkA=0,
                shrinkB=0,
            )
        )
        if lab:
            ax.text(
                (xs + xd) / 2,
                yl + 0.012,
                lab,
                ha="center",
                va="bottom",
                fontsize=FS,
                color="#555555",
                zorder=4,
                bbox={"fc": "white", "ec": "none", "pad": 0.4},
            )


def main() -> None:
    for r in ROWS:
        layout(r)
    row_h = [TITLE + skip_band(r) + box_max(r) + 0.10 for r in ROWS]
    h_fig = sum(row_h) + 0.42
    fig = plt.figure(figsize=(W_FIG, h_fig))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, h_fig)
    ax.axis("off")
    y = h_fig
    for row, rh in zip(ROWS, row_h):
        ax.text(
            PAD_X,
            y - 0.05,
            row.title,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#222222",
        )
        yc = y - TITLE - skip_band(row) - box_max(row) / 2
        draw_row(ax, row, yc)
        y -= rh
    handles = [
        Patch(fc=KIND[k][0], ec=KIND[k][1], lw=0.8, label=lab) for k, lab in LEGEND
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0.1,
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            f"{OUT}/fig_architectures.{ext}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("written", OUT, "height", round(h_fig, 2))


if __name__ == "__main__":
    main()
