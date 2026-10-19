#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image
from site_config import EPSG

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
_fp = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
assert "dejavu" not in _fp.lower(), f"DejaVu fallback: Nimbus Sans missing ({_fp})"

HALF = 64
PIX = 10.0
SHEET_TILE, SHEET_HEAD, SHEET_COLS = 192, 18, 10
BANDS = ("B8A_peak", "B4_peak", "B3_peak")


ROWS: list[tuple[str, list[tuple[str, int, str]]]] = [
    (
        "(a) Hough candidates (source A): discs inside rectangular parcels",
        [
            ("A", 12, "disc in a square parcel"),
            ("A", 19, "disc in a square parcel"),
            ("A", 32, "disc across two parcels"),
            ("A", 41, "village edge"),
        ],
    ),
    (
        "(b) Multi-season detector (source B): parcel groups and non-field features",
        [
            ("B", 13, "group of parcels"),
            ("B", 6, "village"),
            ("B", 20, "bare hill slope"),
            ("B", 44, "reservoir shore, hill"),
        ],
    ),
    (
        "(c) Possible pivots missed during the review (c1, c2) and uncertain cases (c3, c4)",
        [
            ("B", 23, "fallow sector of a pivot"),
            ("B", 33, "full disc"),
            ("B", 21, "uncertain: parcels, faint rim"),
            ("B", 50, "uncertain: partly cut disc"),
        ],
    ),
]

Img: TypeAlias = npt.NDArray[np.float64]


def stretch(a: Img) -> Img:
    v = a[a > 0]
    lo, hi = np.percentile(v, [2, 98]) if v.size else (0.0, 1.0)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def from_stack(path: str, rec: pd.Series) -> tuple[Img, float, bool]:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        names = list(src.descriptions)
        idx = [names.index(b) + 1 for b in BANDS]
        win = Window(int(rec["col"]) - HALF, int(rec["row"]) - HALF, 2 * HALF, 2 * HALF)
        a = src.read(idx, window=win, boundless=True, fill_value=0).astype(float)
    return stretch(a).transpose(1, 2, 0), PIX, True


def from_earthsearch(rec: pd.Series) -> tuple[Img, float, bool]:
    from pyproj import Transformer

    from s2_common import compose

    fwd = Transformer.from_crs(4326, EPSG, always_xy=True)
    inv = Transformer.from_crs(EPSG, 4326, always_xy=True)
    x, y = fwd.transform(float(rec["lon"]), float(rec["lat"]))
    h = HALF * PIX
    w, s = inv.transform(x - h, y - h)
    e, n = inv.transform(x + h, y + h)
    comp, _, _, _ = compose(
        (w, s, e, n), PIX, "2025-06-01", "2025-08-31", n_per_tile=3, log=lambda *_: None
    )
    a = np.stack([comp[b] for b in ("B8A", "B04", "B03")]).astype(float)
    a = np.nan_to_num(a, nan=0.0)
    cy, cx = a.shape[1] // 2, a.shape[2] // 2
    a = a[:, cy - HALF : cy + HALF, cx - HALF : cx + HALF]
    return stretch(a).transpose(1, 2, 0), PIX, True


def from_sheet(sheet: Image.Image, tile: int) -> tuple[Img, float, bool]:
    k = tile - 1
    x0 = (k % SHEET_COLS) * SHEET_TILE
    y0 = (k // SHEET_COLS) * (SHEET_TILE + SHEET_HEAD) + SHEET_HEAD
    im = sheet.crop((x0, y0, x0 + SHEET_TILE, y0 + SHEET_TILE)).convert("RGB")
    return np.asarray(im, dtype=float) / 255.0, 2 * HALF * PIX / SHEET_TILE, False


def draw_tile(ax: Axes, rgb: Img, circle_r: float | None, tag: str, info: str) -> None:
    n = rgb.shape[0]
    ax.imshow(rgb, interpolation="nearest", extent=(0, n, n, 0))
    if circle_r is not None:
        ax.add_patch(Circle((n / 2, n / 2), circle_r, fill=False, ec="yellow", lw=1.2))
    ax.text(
        n * 0.03,
        n * 0.03,
        tag,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9},
    )
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_xlabel(info, fontsize=6.6, labelpad=2.0, linespacing=1.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--stack")
    ap.add_argument("--sheets", nargs=2, metavar=("SHEET_A", "SHEET_B"))
    ap.add_argument(
        "--earthsearch", action="store_true", help="fetch the windows from Earth Search"
    )
    ap.add_argument("--out", default=".")
    arg = ap.parse_args()
    if not (arg.stack or arg.sheets or arg.earthsearch):
        ap.error("give --stack, --sheets or --earthsearch")
    df = pd.read_csv(arg.csv)
    sheets = (
        {"A": Image.open(arg.sheets[0]), "B": Image.open(arg.sheets[1])}
        if arg.sheets
        else {}
    )

    fig, axs = plt.subplots(
        3, 4, figsize=(7.4, 6.2), gridspec_kw={"hspace": 0.40, "wspace": 0.06}
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    for r, (title, items) in enumerate(ROWS):
        for c, (src, tile, why) in enumerate(items):
            rec = df[(df["source"] == src) & (df["tile"] == tile)].iloc[0]
            if arg.stack:
                rgb, _, need = from_stack(arg.stack, rec)
            elif arg.earthsearch:
                rgb, _, need = from_earthsearch(rec)
            else:
                rgb, _, need = from_sheet(sheets[src], tile)
            info = f"{why}\n{rec['split']}, {rec['lon']:.3f}°E {rec['lat']:.3f}°N, r = {int(rec['r_px']) * 10} m"
            draw_tile(
                axs[r, c],
                rgb,
                float(rec["r_px"]) if need else None,
                f"{'abc'[r]}{c + 1}",
                info,
            )
        top = axs[r, 0].get_position().y1
        fig.text(
            0.012,
            top + 0.008,
            title,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )

    p = axs[0, 3].get_position()
    bar = p.width * 250.0 / (2 * HALF * PIX)
    yb = p.y1 + 0.018
    fig.add_artist(
        Line2D(
            [p.x1 - bar, p.x1],
            [yb, yb],
            transform=fig.transFigure,
            color="black",
            lw=2.0,
        )
    )
    fig.text(p.x1 - bar - 0.008, yb, "250 m", ha="right", va="center", fontsize=7.0)
    for ext in ("pdf", "png"):
        fig.savefig(
            f"{arg.out}/fig_false_circles.{ext}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print(f"saved {arg.out}/fig_false_circles.pdf / .png")


if __name__ == "__main__":
    main()
