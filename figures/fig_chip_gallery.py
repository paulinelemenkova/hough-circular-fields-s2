#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy import ndimage
from skimage.draw import disk
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import hough_circle, hough_circle_peaks

from s2_common import _scene, search
from site_config import EPSG, GALLERY_SITES as CHIPS

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N, RES = 256, 10.0
PERIOD = ("2025-06-01", "2025-08-31")
BANDS = ("B8A", "B04", "B03", "B02")

SIGMA_G, R_MIN, R_MAX = 3.0, 15, 60
TAU_H, TAU_NDVI, FILL_MIN, OVL_NMS = 0.30, 0.50, 0.50, 0.30
F64: TypeAlias = npt.NDArray[np.float64]
I32: TypeAlias = npt.NDArray[np.int32]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.fallback": "cm",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)
_fp = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
assert "dejavu" not in _fp.lower(), f"DejaVu fallback: Nimbus Sans missing ({_fp})"
HALO = [pe.withStroke(linewidth=1.8, foreground="black")]
fwd = Transformer.from_crs(4326, EPSG, always_xy=True)
inv = Transformer.from_crs(EPSG, 4326, always_xy=True)
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def retry(fn: Callable[..., Any], *args: Any) -> Any:
    for k in range(5):
        try:
            return fn(*args)
        except Exception as ex:
            print(f"retry {k + 1}: {type(ex).__name__}", flush=True)
            time.sleep(2 + 2 * k)
    return fn(*args)


def composite(lon: float, lat: float) -> dict[str, F64]:
    xc, yc = fwd.transform(lon, lat)
    x0 = np.floor((xc - N * RES / 2) / RES) * RES
    y1 = np.ceil((yc + N * RES / 2) / RES) * RES
    tr = from_origin(x0, y1, RES, RES)
    w, s = inv.transform(x0, y1 - N * RES)
    e, n = inv.transform(x0 + N * RES, y1)
    items = retry(search, (w, s, e, n), *PERIOD, 5.0)
    tile = max(items, key=lambda t: len(items[t]))
    feats = items[tile][:8]
    with ThreadPoolExecutor(8) as ex:
        scenes = list(ex.map(lambda f: retry(_scene, f, tr, N, N, BANDS, RES), feats))
    with np.errstate(all="ignore"):
        return {
            b: np.nanmedian(np.stack([sc[b] for sc in scenes]), axis=0).astype(float)
            for b in BANDS
        }


def stretch(bands: list[F64]) -> F64:
    stack = np.dstack(bands)
    lo, hi = np.nanpercentile(stack, [2, 98])
    return np.nan_to_num(np.clip((stack - lo) / (hi - lo), 0, 1))


def each_stretch(bands: list[F64]) -> F64:
    out = []
    for b in bands:
        lo, hi = np.nanpercentile(b, [2, 98])
        out.append(np.clip((b - lo) / (hi - lo), 0, 1))
    return np.nan_to_num(np.dstack(out))


def instances(ndvi: F64) -> I32:
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
    green = np.nan_to_num(ndvi) > TAU_NDVI
    inst = np.zeros(ndvi.shape, np.int32)
    k = 0
    for i in np.argsort(score)[::-1]:
        disc_m = np.zeros(ndvi.shape, bool)
        disc_m[disk((int(cy[i]), int(cx[i])), int(rr[i]), shape=ndvi.shape)] = True
        d = disc_m & green
        if d.sum() < FILL_MIN * disc_m.sum():
            continue
        if (d & (inst > 0)).sum() > OVL_NMS * d.sum():
            continue
        d = ndimage.binary_fill_holes(ndimage.binary_opening(d, iterations=1))
        d &= inst == 0
        if d.sum() < 50:
            continue
        k += 1
        inst[d] = k
    return inst


def main() -> None:
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
    cols = (
        "True colour (B4-B3-B2)",
        "False colour (B8A-B4-B3)",
        "NDVI",
        "Candidate semantic",
        "Candidate instances",
    )
    fig = plt.figure(figsize=(6.9, 8.75))
    gs = fig.add_gridspec(
        len(CHIPS) + 1,
        5,
        height_ratios=[1] * len(CHIPS) + [0.06],
        hspace=0.05,
        wspace=0.04,
        left=0.045,
        right=0.995,
        top=0.975,
        bottom=0.035,
    )
    im_ndvi = None
    for r, (label, lon, lat) in enumerate(CHIPS):
        c = composite(lon, lat)
        with np.errstate(all="ignore"):
            ndvi = (c["B8A"] - c["B04"]) / (c["B8A"] + c["B04"])
        inst = instances(ndvi)
        n_inst = int(inst.max())
        print(f"{label}: {n_inst} candidate instances")
        panels = [
            stretch([c["B04"], c["B03"], c["B02"]]),
            each_stretch([c["B8A"], c["B04"], c["B03"]]),
        ]
        for k in range(5):
            ax = fig.add_subplot(gs[r, k])
            ax.set_xticks([])
            ax.set_yticks([])
            if k < 2:
                ax.imshow(panels[k], interpolation="nearest")
            elif k == 2:
                im_ndvi = ax.imshow(
                    ndvi, cmap="turbo", vmin=0, vmax=0.9, interpolation="nearest"
                )
            elif k == 3:
                ax.imshow(
                    inst > 0,
                    cmap=ListedColormap(["black", "white"]),
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
            else:
                cmap = ListedColormap(
                    ["black"] + [pal[i % 8] for i in range(max(n_inst, 1))]
                )
                ax.imshow(
                    inst,
                    cmap=cmap,
                    vmin=-0.5,
                    vmax=max(n_inst, 1) + 0.5,
                    interpolation="nearest",
                )
                ax.text(
                    0.96,
                    0.05,
                    f"n = {n_inst}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    color="white",
                    fontsize=8.5,
                    path_effects=HALO,
                )
            if r == 0:
                ax.set_title(cols[k], fontsize=8.5, pad=3)
            if k == 0:
                ax.set_ylabel(f"({chr(97 + r)}) {label}", fontsize=8.5, labelpad=3)
                if r == 0:
                    ax.add_patch(
                        Rectangle(
                            (N - 120, N - 22), 100, 6, fc="white", ec="black", lw=0.5
                        )
                    )
                    ax.text(
                        N - 70,
                        N - 27,
                        "1 km",
                        ha="center",
                        va="bottom",
                        color="white",
                        fontsize=8.5,
                        path_effects=HALO,
                    )
    cax = fig.add_subplot(gs[len(CHIPS), 2])
    cb = fig.colorbar(
        im_ndvi, cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9]
    )
    cb.ax.tick_params(labelsize=8.5, length=2.5, pad=1)
    cb.set_label("NDVI", fontsize=8.5, labelpad=1)
    for ext in ("pdf", "png"):
        fig.savefig(
            f"{OUT}/fig_chip_gallery.{ext}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("written", OUT)


if __name__ == "__main__":
    main()
