#!/usr/bin/env python3

from __future__ import annotations

import sys
from typing import Any, TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import patheffects as pe
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy import ndimage
from skimage.draw import disk
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import perimeter
from skimage.transform import hough_circle, hough_circle_peaks

from s2_common import _scene, search
from site_config import EPSG, ZOOM_CENTRE as CENTRE

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

N, RES = 256, 10.0
PERIOD = ("2025-06-01", "2025-08-31")
SIGMA_G = 3.0
R_MIN, R_MAX = 15, 60
TAU_H, TAU_NDVI, TAU_C = 0.30, 0.50, 0.70
FILL_MIN, OVL_NMS = 0.50, 0.30
F64: TypeAlias = npt.NDArray[np.float64]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.sf": "Nimbus Sans",
        "mathtext.cal": "cmsy10",
        "mathtext.fallback": "cm",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    }
)
_fp = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
assert "dejavu" not in _fp.lower(), f"DejaVu fallback: Nimbus Sans missing ({_fp})"
HALO = [pe.withStroke(linewidth=2.0, foreground="black")]


def chip_composite() -> dict[str, F64]:
    fwd = Transformer.from_crs(4326, EPSG, always_xy=True)
    inv = Transformer.from_crs(EPSG, 4326, always_xy=True)
    xc, yc = fwd.transform(*CENTRE)
    x0 = np.floor((xc - N * RES / 2) / RES) * RES
    y1 = np.ceil((yc + N * RES / 2) / RES) * RES
    tr = from_origin(x0, y1, RES, RES)
    w, s = inv.transform(x0, y1 - N * RES)
    e, n = inv.transform(x0 + N * RES, y1)
    items = search((w, s, e, n), *PERIOD, max_cloud=5.0)
    tile = max(items, key=lambda t: len(items[t]))
    feats = items[tile][:8]
    print(f"tile {tile}:", ", ".join(f["properties"]["datetime"][:10] for f in feats))
    scenes = [_scene(f, tr, N, N, ("B8A", "B04", "B03"), RES) for f in feats]
    with np.errstate(all="ignore"):
        return {
            b: np.nanmedian(np.stack([sc[b] for sc in scenes]), axis=0).astype(float)
            for b in ("B8A", "B04", "B03")
        }


def false_colour(c: dict[str, F64]) -> F64:
    out = []
    for b in ("B8A", "B04", "B03"):
        lo, hi = np.nanpercentile(c[b], [2, 98])
        out.append(np.clip((c[b] - lo) / (hi - lo), 0, 1))
    return np.nan_to_num(np.dstack(out))


def weak_labels(ndvi: F64) -> dict[str, Any]:
    i_sigma = gaussian(np.nan_to_num(ndvi), sigma=SIGMA_G)
    edges = canny(i_sigma, sigma=1.0, low_threshold=0.03, high_threshold=0.08)
    radii = np.arange(R_MIN, R_MAX + 1)
    acc = hough_circle(edges, radii, normalize=True)
    score, cx, cy, rr = hough_circle_peaks(
        acc,
        radii,
        min_xdistance=12,
        min_ydistance=12,
        threshold=TAU_H,
        total_num_peaks=60,
    )
    order = np.argsort(score)[::-1]
    cand = [(float(score[i]), int(cx[i]), int(cy[i]), int(rr[i])) for i in order]
    green = np.nan_to_num(ndvi) > TAU_NDVI
    inst = np.zeros(ndvi.shape, np.int32)
    kept: list[tuple[float, int, int, int, float, bool]] = []
    for sc, x, y, r in cand:
        disc_m = np.zeros(ndvi.shape, bool)
        disc_m[disk((y, x), r, shape=ndvi.shape)] = True
        d = disc_m & green
        if d.sum() < FILL_MIN * disc_m.sum():
            continue
        if (d & (inst > 0)).sum() > OVL_NMS * d.sum():
            continue
        d = ndimage.binary_fill_holes(ndimage.binary_opening(d, iterations=1))
        d &= inst == 0
        if d.sum() < 50:
            continue
        circ = float(4 * np.pi * d.sum() / perimeter(d) ** 2)
        inst[d] = len(kept) + 1
        kept.append((sc, x, y, r, circ, circ < TAU_C))
    print(
        f"{len(cand)} Hough candidates, {len(kept)} retained, "
        f"{sum(k[5] for k in kept)} flagged for review"
    )
    return {"edges": edges, "cand": cand, "kept": kept, "inst": inst}


def frame(ax: Axes, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_title(title, loc="left", pad=3)


def main() -> None:
    comp = chip_composite()
    with np.errstate(all="ignore"):
        ndvi = (comp["B8A"] - comp["B04"]) / (comp["B8A"] + comp["B04"])
    wl = weak_labels(ndvi)

    fig = plt.figure(figsize=(6.9, 5.55))
    gs = fig.add_gridspec(
        5,
        3,
        height_ratios=[1, 0.05, 0.17, 1, 0.05],
        hspace=0.04,
        wspace=0.05,
        left=0.01,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    ax = [fig.add_subplot(gs[r, c]) for r in (0, 3) for c in range(3)]
    cb = [fig.add_subplot(gs[r, c]) for r in (1, 4) for c in range(3)]

    ax[0].imshow(false_colour(comp), interpolation="nearest")
    frame(ax[0], "(a) Sentinel-2 B8A-B4-B3")
    ax[0].add_patch(Rectangle((N - 70, N - 20), 50, 5, fc="white", ec="black", lw=0.5))
    ax[0].text(
        N - 45,
        N - 24,
        "500 m",
        ha="center",
        va="bottom",
        color="white",
        fontsize=8.5,
        path_effects=HALO,
    )

    im = ax[1].imshow(ndvi, cmap="turbo", vmin=0, vmax=0.9, interpolation="nearest")
    frame(ax[1], "(b) NDVI")
    c1 = fig.colorbar(im, cax=cb[1], orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9])
    c1.ax.tick_params(labelsize=8.5, length=2.5, pad=1)

    ax[2].imshow(
        wl["edges"], cmap=ListedColormap(["black", "white"]), interpolation="nearest"
    )
    frame(ax[2], rf"(c) Canny edges, $\sigma_G$ = {SIGMA_G:g} px")

    ax[3].imshow(ndvi, cmap="gray", vmin=0, vmax=0.9, interpolation="nearest")
    norm = Normalize(TAU_H, max(c[0] for c in wl["cand"]))
    cmap = plt.get_cmap("turbo")
    for sc, x, y, r in wl["cand"][::-1]:
        ax[3].add_patch(Circle((x, y), r, fill=False, ec=cmap(norm(sc)), lw=1.0))
    frame(ax[3], f"(d) Hough candidates (n = {len(wl['cand'])})")
    c3 = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb[3], orientation="horizontal"
    )
    c3.ax.tick_params(labelsize=8.5, length=2.5, pad=1)
    c3.set_label(r"Normalised score $\mathcal{A}/2\pi r$", fontsize=8.5, labelpad=1)

    ax[4].imshow(ndvi, cmap="gray", vmin=0, vmax=0.9, interpolation="nearest")
    for sc, x, y, r, circ, review in wl["kept"]:
        ax[4].add_patch(
            Circle(
                (x, y),
                r,
                fill=False,
                lw=1.4,
                ec="#E69F00" if review else "#56B4E9",
                ls="--" if review else "-",
            )
        )
    frame(ax[4], f"(e) Retained (n = {len(wl['kept'])})")
    cb[4].axis("off")
    cb[4].legend(
        handles=[
            Line2D([], [], color="#56B4E9", lw=1.4, label=rf"$C \geq {TAU_C:.2f}$"),
            Line2D(
                [],
                [],
                color="#E69F00",
                lw=1.4,
                ls="--",
                label=rf"$C < {TAU_C:.2f}$, review",
            ),
        ],
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        borderpad=0,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    inst = wl["inst"]
    n_inst = int(inst.max())
    pal = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#999999",
    ]
    cols = ["black"] + [pal[i % len(pal)] for i in range(n_inst)]
    ax[5].imshow(
        inst,
        cmap=ListedColormap(cols),
        vmin=-0.5,
        vmax=n_inst + 0.5,
        interpolation="nearest",
    )
    ax[5].contour(inst > 0, levels=[0.5], colors="white", linewidths=0.8)
    frame(ax[5], f"(f) Semantic + instance masks ({n_inst})")
    for c in (cb[0], cb[2], cb[5]):
        c.axis("off")

    for ext in ("pdf", "png"):
        fig.savefig(
            f"{OUT}/fig_weak_labels.{ext}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("written", OUT)


if __name__ == "__main__":
    main()
