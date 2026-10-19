#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours

PX, M = 256, 32
SEASONS = ("early", "peak", "late")
OUT = "labelling"


def stretch_limits(src: rasterio.DatasetReader) -> dict[str, tuple[float, float]]:
    lim = {}
    names = list(src.descriptions)
    h, w = src.height // 20, src.width // 20
    for s in SEASONS:
        for b in ("B8A", "B4", "B3"):
            a = src.read(names.index(f"{b}_{s}") + 1, out_shape=(h, w)).astype(float)
            a = a[a > 0]
            lim[f"{b}_{s}"] = (float(np.percentile(a, 2)), float(np.percentile(a, 98)))
    return lim


def cand_mask(nd: np.ndarray, gr: int, gc: int, r: int, r0: int, c0: int) -> np.ndarray:
    yy, xx = np.mgrid[0 : nd.shape[0], 0 : nd.shape[1]]
    d = (np.hypot(yy + r0 - gr, xx + c0 - gc) <= r) & (nd > 0.5)
    return ndimage.binary_fill_holes(ndimage.binary_opening(d))


def global_masks(
    src: rasterio.DatasetReader, cd: pd.DataFrame, ix: pd.DataFrame
) -> dict[int, tuple[int, int, np.ndarray]]:
    off = pd.read_csv("chips/chips_index.csv").set_index("chip_id")
    b = list(src.descriptions).index("NDVI_peak") + 1
    out = {}
    for c in cd.itertuples():
        gr = int(off.loc[c.chip_id, "row_off"]) + int(c.row)
        gc = int(off.loc[c.chip_id, "col_off"]) + int(c.col)
        h = int(c.r_px) + 2
        r0, c0 = gr - h, gc - h
        nd = (
            src.read(
                b,
                window=Window(c0, r0, 2 * h + 1, 2 * h + 1),
                boundless=True,
                fill_value=0,
            )
            / 1e4
        )
        m = cand_mask(nd, gr, gc, int(c.r_px), r0, c0)
        if m.sum() < 50:
            continue
        ys, xs = np.nonzero(m)
        out[int(c.gid)] = (
            r0 + int(ys.min()),
            c0 + int(xs.min()),
            m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1],
        )
    return out


def outline(m: np.ndarray, dy: int, dx: int) -> list[list[float]]:
    c = max(find_contours(np.pad(m, 1).astype(float), 0.5), key=len)
    c = approximate_polygon(c, tolerance=0.6)
    return [[round(float(x) - 1 + dx, 1), round(float(y) - 1 + dy, 1)] for y, x in c]


def main() -> None:
    stack = sys.argv[1]
    os.makedirs(f"{OUT}/img", exist_ok=True)
    ix = pd.read_csv("chips/chips_index.csv")
    ix = ix[~ix["buffer"]].reset_index(drop=True)
    cd = pd.read_csv("chips/candidates.csv")
    cd["fill"] = cd["area_px"] / (np.pi * cd["r_px"] ** 2)
    cd["rank"] = cd["score"] * cd["circularity"] * cd["fill"]
    info = cd.set_index("gid")

    chips: list[dict[str, Any]] = []
    with rasterio.open(stack) as src:
        lim = stretch_limits(src)
        names = list(src.descriptions)
        print("building candidate outlines ...", flush=True)
        gm = global_masks(src, cd, ix)
        for k, r in enumerate(ix.itertuples(), 1):
            ro, co = int(r.row_off), int(r.col_off)
            win = Window(co - M, ro - M, PX + 2 * M, PX + 2 * M)
            for s in SEASONS:
                rgb = []
                for b in ("B8A", "B4", "B3"):
                    a = src.read(
                        names.index(f"{b}_{s}") + 1,
                        window=win,
                        boundless=True,
                        fill_value=0,
                    ).astype(float)
                    lo, hi = lim[f"{b}_{s}"]
                    rgb.append(np.clip((a - lo) / (hi - lo), 0, 1) * 255)
                Image.fromarray(np.dstack(rgb).astype(np.uint8)).save(
                    f"{OUT}/img/{r.chip_id}_{s}.jpg", quality=88
                )
            cands = []
            for g, (gr, gc, m) in gm.items():
                h, w = m.shape
                if (
                    gr + h < ro - M
                    or gr > ro + PX + M
                    or gc + w < co - M
                    or gc > co + PX + M
                ):
                    continue
                c = info.loc[g]
                cands.append(
                    {
                        "gid": int(g),
                        "r": int(c["r_px"]),
                        "score": float(c["score"]),
                        "circ": float(c["circularity"]),
                        "fill": round(float(c["fill"]), 3),
                        "rank": round(float(c["rank"]), 4),
                        "review": bool(c["review"]),
                        "home": c["chip_id"] == r.chip_id,
                        "poly": outline(m, gr - ro + M, gc - co + M),
                    }
                )
            cands.sort(key=lambda c: -c["rank"])
            chips.append(
                {
                    "id": r.chip_id,
                    "split": r.split,
                    "lon": round(r.lon, 4),
                    "lat": round(r.lat, 4),
                    "valid": round(float(r.valid_frac), 3),
                    "cands": cands,
                }
            )
            if k % 200 == 0 or k == len(ix):
                print(f"  {k}/{len(ix)} chips", flush=True)

    chips.sort(key=lambda c: -(max([x["rank"] for x in c["cands"]], default=0.0)))
    with open(f"{OUT}/data.js", "w") as f:
        f.write("const CHIPS = ")
        json.dump(chips, f, separators=(",", ":"))
        f.write(";\n")
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(here, "label.html"), f"{OUT}/label.html")
    n = sum(len([c for c in ch["cands"] if c["home"]]) for ch in chips)
    print(
        f"done: {len(chips)} chips, {n} candidates -> open {OUT}/label.html in a browser"
    )


if __name__ == "__main__":
    main()
