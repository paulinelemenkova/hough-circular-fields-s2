#!/usr/bin/env python3

from __future__ import annotations

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 6.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.sf": "Nimbus Sans",
        "mathtext.cal": "cmsy10",
        "mathtext.fallback": "cm",
    }
)
_fp = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
assert "dejavu" not in _fp.lower(), f"DejaVu fallback: Nimbus Sans missing ({_fp})"
print("font:", _fp)


STAGES = {
    "data": {"edge": "#4D4D4D", "fill": "#F2F2F2", "head": "#7F7F7F"},
    "gee": {"edge": "#0B6E4F", "fill": "#E3F4EC", "head": "#1A9E72"},
    "chips": {"edge": "#005A9C", "fill": "#E1EEF8", "head": "#0072B2"},
    "label": {"edge": "#A35200", "fill": "#FDEEDC", "head": "#E69F00"},
    "torch": {"edge": "#8E2C6E", "fill": "#F7E6F1", "head": "#CC79A7"},
    "infer": {"edge": "#A33A1F", "fill": "#FBE5DE", "head": "#D55E00"},
    "out": {"edge": "#3E4A8C", "fill": "#E7E9F6", "head": "#56B4E9"},
}
MODEL_COLOURS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]

W, H = 100.0, 82.0
fig = plt.figure(figsize=(7.4, 6.07))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def panel(x, y, w, h, key, number, title):
    st = STAGES[key]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.6",
            fc=st["fill"],
            ec=st["edge"],
            lw=1.0,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 4.6),
            w,
            4.6,
            boxstyle="round,pad=0,rounding_size=1.6",
            fc=st["head"],
            ec=st["edge"],
            lw=1.0,
            zorder=2,
        )
    )

    ax.add_patch(
        plt.Rectangle(
            (x + 0.05, y + h - 4.6), w - 0.1, 2.0, fc=st["head"], ec="none", zorder=2
        )
    )
    if number:
        ax.add_patch(
            plt.Circle(
                (x + 3.0, y + h - 2.3),
                1.75,
                fc="white",
                ec=st["edge"],
                lw=0.9,
                zorder=3,
            )
        )
        ax.text(
            x + 3.0,
            y + h - 2.3,
            number,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=st["edge"],
            zorder=4,
        )
        tx = x + 5.6
    else:
        tx = x + 1.6
    ax.text(
        tx,
        y + h - 2.3,
        title,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="white",
        zorder=4,
    )


def box(x, y, w, h, key, head, body="", fs=6.0, face="white", bold_col=None):
    st = STAGES[key]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.8",
            fc=face,
            ec=bold_col or st["edge"],
            lw=0.7,
            zorder=5,
        )
    )
    if body:
        ax.text(
            x + w / 2,
            y + h - 1.15,
            head,
            ha="center",
            va="top",
            fontsize=fs + 0.3,
            fontweight="bold",
            color=bold_col or st["edge"],
            zorder=6,
        )
        ax.text(
            x + w / 2,
            y + h - 3.55,
            body,
            ha="center",
            va="top",
            fontsize=fs - 0.6,
            color="#222222",
            linespacing=1.18,
            zorder=6,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            head,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=bold_col or st["edge"],
            zorder=6,
        )


def arrow(
    p0,
    p1,
    colour="#333333",
    lw=1.0,
    rad=0.0,
    label=None,
    lpos=0.5,
    loff=(0.0, 1.1),
    style="-|>",
    ls="-",
    fs=5.6,
):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=7,
            color=colour,
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=7,
        )
    )
    if label:
        lx = p0[0] + (p1[0] - p0[0]) * lpos + loff[0]
        ly = p0[1] + (p1[1] - p0[1]) * lpos + loff[1]
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            style="italic",
            color=colour,
            zorder=8,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": 0.9,
            },
        )


y1, h1 = 55.0, 25.5

panel(1.0, y1, 17.0, h1, "data", "", "Open-access data")
for i, (name, sub) in enumerate(
    [
        ("Sentinel-2 MSI", "Level-2A, 2025"),
        ("Cloud Score+", "cs_cdf, 10 m"),
        ("SRTM | WorldCover", "auxiliary layers"),
        ("HydroBASINS", "basin boundary"),
    ]
):
    yy = y1 + 15.6 - i * 5.0
    box(2.0, yy, 15.0, 4.5, "data", "", "", fs=5.9)
    ax.text(
        9.5,
        yy + 3.05,
        name,
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=STAGES["data"]["edge"],
        zorder=7,
    )
    ax.text(
        9.5,
        yy + 1.25,
        sub,
        ha="center",
        va="center",
        fontsize=5.0,
        color="#222222",
        zorder=7,
    )


panel(21.5, y1, 39.0, h1, "gee", "1", "Google Earth Engine: compositing")
box(
    23.0,
    y1 + 11.6,
    11.3,
    8.6,
    "gee",
    "Filter",
    "AOI, date window\nscene cloud < 60%\nlink Cloud Score+",
)
box(
    35.5,
    y1 + 11.6,
    11.3,
    8.6,
    "gee",
    "Cloud mask",
    r"$M_k=\mathbb{1}[\mathrm{cs\_cdf}\geq0.60]$" + "\nper-pixel, 10 m",
)
box(
    48.0,
    y1 + 11.6,
    11.3,
    8.6,
    "gee",
    "Median",
    "per window\n6 bands B2-B4,\nB8A, B11, B12",
)
arrow((34.3, y1 + 15.9), (35.5, y1 + 15.9), STAGES["gee"]["edge"])
arrow((46.8, y1 + 15.9), (48.0, y1 + 15.9), STAGES["gee"]["edge"])

season = [
    ("early", "Apr-May", "#B5D99C"),
    ("peak", "Jun-Aug", "#3C9A4F"),
    ("late", "Sep-Oct", "#C9A66B"),
]
for i, (nm, mo, col) in enumerate(season):
    xx = 23.0 + i * 8.0
    ax.add_patch(
        FancyBboxPatch(
            (xx, y1 + 4.6),
            7.2,
            5.3,
            boxstyle="round,pad=0,rounding_size=0.6",
            fc=col,
            ec=STAGES["gee"]["edge"],
            lw=0.6,
            zorder=5,
        )
    )
    ax.text(
        xx + 3.6,
        y1 + 8.1,
        nm,
        ha="center",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color="white" if nm == "peak" else "#1F3D2B",
        zorder=6,
    )
    ax.text(
        xx + 3.6,
        y1 + 6.0,
        mo,
        ha="center",
        va="center",
        fontsize=5.2,
        color="white" if nm == "peak" else "#1F3D2B",
        zorder=6,
    )
box(48.0, y1 + 4.6, 11.3, 5.3, "gee", "+ NDVI, NDWI", "", fs=5.9)
ax.text(
    40.75,
    y1 + 1.9,
    r"3 windows $\times$ (6 bands + 2 indices) = 24-band annual stack, 2025",
    ha="center",
    va="center",
    fontsize=5.6,
    color=STAGES["gee"]["edge"],
    zorder=6,
)


panel(64.0, y1, 35.0, h1, "chips", "2", "Chip export and indexing")

gx, gy, cs = 66.0, y1 + 4.0, 3.6
for r in range(4):
    for c in range(4):
        ax.add_patch(
            plt.Rectangle(
                (gx + c * cs, gy + r * cs),
                cs,
                cs,
                fc=["#CFE3F3", "#9CC5E6"][(r + c) % 2],
                ec=STAGES["chips"]["edge"],
                lw=0.5,
                zorder=5,
            )
        )
for cx_, cy_, rr in [
    (0.9, 1.2, 0.85),
    (2.3, 2.9, 1.05),
    (3.2, 0.9, 0.6),
    (1.4, 3.2, 0.55),
]:
    ax.add_patch(
        plt.Circle(
            (gx + cx_ * cs, gy + cy_ * cs),
            rr * cs * 0.55,
            fc="#2E8B57",
            ec="#145A32",
            lw=0.5,
            alpha=0.9,
            zorder=6,
        )
    )
ax.text(
    gx + 2 * cs,
    gy - 1.4,
    r"256 $\times$ 256 px, 32 px margin",
    ha="center",
    va="center",
    fontsize=5.4,
    color=STAGES["chips"]["edge"],
)
box(
    82.0,
    y1 + 12.4,
    15.6,
    7.8,
    "chips",
    "GeoTIFF (COG)",
    "24 x int16, UTM\n4 quarters merged locally",
)
box(
    82.0,
    y1 + 3.0,
    15.6,
    7.8,
    "chips",
    "Parquet index",
    "chip ID, MGRS tile, centre,\nsplit, pivot count",
)

arrow((18.0, y1 + h1 / 2), (21.5, y1 + h1 / 2), "#4D4D4D", lw=1.2)
arrow(
    (60.5, y1 + h1 / 2),
    (64.0, y1 + h1 / 2),
    STAGES["gee"]["edge"],
    lw=1.3,
    label="Drive export",
    loff=(0.0, 2.2),
)


y2, h2 = 27.5, 24.5
panel(64.0, y2, 35.0, h2, "label", "3", "Hough candidates and manual labelling")
lab = [
    ("Gaussian + Canny", r"$\|\nabla I_\sigma\|$ on peak NDVI"),
    ("Circular Hough", r"$\mathcal{A}(a,b,r)$, $r$ = 15-45 px"),
    ("Screening + NMS", r"$C=4\pi A/P^2\geq0.70$"),
    ("Manual labelling", "accept | reject | add missed"),
]
for i, (hd, bd) in enumerate(lab):
    yy = y2 + 14.6 - i * 4.75
    box(66.0, yy, 31.0, 4.1, "label", "", "", fs=5.9)
    ax.text(
        67.2,
        yy + 2.05,
        hd,
        ha="left",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=STAGES["label"]["edge"],
        zorder=7,
    )
    ax.text(
        95.8,
        yy + 2.05,
        bd,
        ha="right",
        va="center",
        fontsize=5.6,
        color="#222222",
        zorder=7,
    )
    if i < 3:
        arrow((81.5, yy), (81.5, yy - 0.65), STAGES["label"]["edge"], lw=0.7)
ax.text(
    81.5,
    y2 + 0.85,
    "semantic mask (binary) + instance mask (uint16)",
    ha="center",
    va="bottom",
    fontsize=5.5,
    style="italic",
    color=STAGES["label"]["edge"],
)

arrow(
    (81.5, y1),
    (81.5, y2 + h2),
    STAGES["chips"]["edge"],
    lw=1.3,
    label="peak-season NDVI chips",
    lpos=0.5,
    loff=(9.0, 0.0),
)


panel(1.0, y2, 59.5, h2, "torch", "4", "PyTorch: data loader and spatial-block splits")
box(
    2.5,
    y2 + 11.0,
    14.2,
    8.4,
    "torch",
    "Spatial blocks",
    "4 x 4 chips = 10.24 km\n1271 / 296 / 364 chips\none-chip buffer",
)
box(
    2.5,
    y2 + 1.4,
    14.2,
    8.4,
    "torch",
    "PivotChips Dataset",
    "z-score per band\nD4 dihedral augment\nsemantic + instances",
)
arrow((9.6, y2 + 11.0), (9.6, y2 + 9.8), STAGES["torch"]["edge"], lw=0.7)

models = [
    ("U-Net", "ResNet-34", "CNN"),
    ("DeepLabV3+", "ResNet-50, ASPP", "CNN"),
    ("SegFormer", "MiT-B2", "Transformer"),
    ("Mask R-CNN", "ResNet-50-FPN", "Instance"),
]
mx0, mw = 18.6, 21.0
for i, (nm, enc, fam) in enumerate(models):
    yy = y2 + 15.0 - i * 4.25
    ax.add_patch(
        FancyBboxPatch(
            (mx0, yy),
            mw,
            3.0,
            boxstyle="round,pad=0,rounding_size=0.6",
            fc="white",
            ec=MODEL_COLOURS[i],
            lw=0.9,
            zorder=5,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (mx0, yy),
            1.4,
            3.0,
            boxstyle="round,pad=0,rounding_size=0.5",
            fc=MODEL_COLOURS[i],
            ec="none",
            zorder=6,
        )
    )
    ax.text(
        mx0 + 2.0,
        yy + 1.5,
        nm,
        ha="left",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color=MODEL_COLOURS[i],
        zorder=7,
    )
    ax.text(
        mx0 + mw - 0.6,
        yy + 1.5,
        f"{enc}\n{fam}",
        ha="right",
        va="center",
        fontsize=4.5,
        color="#333333",
        linespacing=1.0,
        zorder=7,
    )
arrow((16.7, y2 + 5.6), (19.0, y2 + 9.0), STAGES["torch"]["edge"], lw=0.9)
ax.text(
    mx0 + mw / 2,
    y2 + 1.15,
    "intended use (not trained here)",
    ha="center",
    va="center",
    fontsize=5.0,
    style="italic",
    color=STAGES["torch"]["edge"],
    zorder=7,
)

box(
    41.5,
    y2 + 11.0,
    17.5,
    8.4,
    "torch",
    "Model reuse",
    "train on spatial blocks\ntest false positives on\n1915 pivot-free chips",
)
box(
    41.5,
    y2 + 1.4,
    17.5,
    8.4,
    "torch",
    "Label agreement",
    "automatic vs manual labels\nobject P, R, F1\n(Hungarian, IoU 0.5)",
)
arrow((39.0, y2 + 15.2), (41.5, y2 + 15.2), STAGES["torch"]["edge"], lw=0.9)
arrow((50.25, y2 + 11.0), (50.25, y2 + 9.8), STAGES["torch"]["edge"], lw=0.7)

arrow(
    (64.0, y2 + h2 / 2),
    (60.5, y2 + h2 / 2),
    STAGES["label"]["edge"],
    lw=1.3,
    label="masks",
    loff=(0.0, 1.9),
)

arrow(
    (70.0, y1),
    (52.0, y2 + h2),
    STAGES["chips"]["edge"],
    lw=1.0,
    rad=0.0,
    ls="--",
    label="24-band chips + index",
    lpos=0.5,
    loff=(-12.5, 0.0),
)


y3, h3 = 1.0, 23.5
panel(1.0, y3, 59.5, h3, "infer", "5", "Inventory from the manual labels")
steps5 = [
    ("Instances", "one mask per\npivot, cut to\neach chip"),
    ("Pivot table", "centre lon, lat\narea (ha)\nsplit, chip ID"),
    (
        "Diameter",
        r"$D_{eq}=2\sqrt{A/\pi}$" + "\n302-1014 m\nmedian 662 m",
    ),
    ("Hexagon bins", "5 km over AOI\n1.5 km cluster\nGMT maps"),
]
for i, (hd, bd) in enumerate(steps5):
    xx = 2.5 + i * 14.4
    box(xx, y3 + 6.0, 12.6, 10.0, "infer", hd, bd, fs=6.0)
    if i < 3:
        arrow(
            (xx + 12.6, y3 + 11.0),
            (xx + 14.4, y3 + 11.0),
            STAGES["infer"]["edge"],
            lw=0.9,
        )
ax.text(
    30.75,
    y3 + 2.4,
    "2025: 32 pivots, 10.5 km$^2$, in 16 of 1931 chips",
    ha="center",
    va="center",
    fontsize=5.6,
    style="italic",
    color=STAGES["infer"]["edge"],
)

arrow(
    (30.75, y2),
    (30.75, y3 + h3),
    STAGES["torch"]["edge"],
    lw=1.3,
    label="manual instance masks",
    lpos=0.5,
    loff=(9.5, 0.0),
)


panel(64.0, y3, 35.0, h3, "out", "", "Released outputs")
outs = [
    ("Dataset", "chips, masks, splits | Zenodo"),
    ("Code", "GEE, Python, HTML | GitHub"),
    ("Pivot inventory", "CSV, 2025"),
    ("Maps and figures", "GMT, Matplotlib"),
]
for i, (hd, bd) in enumerate(outs):
    yy = y3 + 14.0 - i * 4.4
    box(65.5, yy, 32.0, 3.8, "out", "", "", fs=5.9)
    ax.text(
        66.7,
        yy + 1.9,
        hd,
        ha="left",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=STAGES["out"]["edge"],
        zorder=7,
    )
    ax.text(
        96.4,
        yy + 1.9,
        bd,
        ha="right",
        va="center",
        fontsize=5.3,
        color="#222222",
        zorder=7,
    )
arrow((60.5, y3 + h3 / 2), (64.0, y3 + h3 / 2), STAGES["infer"]["edge"], lw=1.3)


ax.text(
    50.0,
    81.4,
    r"cloud (GEE) $\rightarrow$ local (labelling, inventory) $\rightarrow$ PyTorch (data loader)",
    ha="center",
    va="center",
    fontsize=5.8,
    color="#555555",
)

for ext in ("pdf", "png"):
    fig.savefig(
        f"{OUT}/fig_workflow.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02
    )
print("saved fig_workflow.pdf / fig_workflow.png")
