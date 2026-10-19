#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy import ndimage
from skimage.draw import disk
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import perimeter
from skimage.transform import hough_circle, hough_circle_peaks
from site_config import AOI, EPSG


CHIP, NB, PX = 2560.0, 4, 256
SPLITS = ("train", "val", "test")
SIGMA_G, R_MIN, R_MAX = 3.0, 15, 60
TAU_H, TAU_NDVI, TAU_C = 0.30, 0.50, 0.70
FILL_MIN, OVL_NMS, MIN_PX = 0.50, 0.30, 50
R_KEEP, RING_MAX = 18, 0.50
OUT = "chips"

fwd = Transformer.from_crs(4326, EPSG, always_xy=True)
inv = Transformer.from_crs(EPSG, 4326, always_xy=True)


def chip_grid(blocks: pd.DataFrame) -> pd.DataFrame:
    w, s, e, n = AOI
    xs, ys = fwd.transform([w, w, e, e], [s, n, s, n])
    x0 = np.floor(min(xs) / CHIP) * CHIP
    y0 = np.floor(min(ys) / CHIP) * CHIP
    nx = int(np.ceil((max(xs) - x0) / CHIP))
    ny = int(np.ceil((max(ys) - y0) / CHIP))
    ci, cj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = x0 + (ci + 0.5) * CHIP, y0 + (cj + 0.5) * CHIP
    lon, lat = inv.transform(cx, cy)
    inside = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    ch = pd.DataFrame({"i": ci[inside], "j": cj[inside]})
    ch["bi"], ch["bj"] = ch["i"] // NB, ch["j"] // NB
    ch["x_min"] = x0 + ch["i"] * CHIP
    ch["y_max"] = y0 + (ch["j"] + 1) * CHIP
    ch["lon"], ch["lat"] = lon[inside], lat[inside]
    split_of = {
        (int(a), int(b)): int(q)
        for a, b, q in zip(
            blocks["bi"].to_numpy(), blocks["bj"].to_numpy(), blocks["split"].to_numpy()
        )
    }
    ch["split_id"] = [split_of[(a, b)] for a, b in zip(ch["bi"], ch["bj"])]
    ch["buffer"] = False
    for (bi, bj), q in split_of.items():
        for dbi, dbj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q2 = split_of.get((bi + dbi, bj + dbj))
            if q2 is None or q2 == q or q > q2:
                continue
            m = (ch["bi"] == bi) & (ch["bj"] == bj)
            if dbi == 1:
                m &= ch["i"] == bi * NB + NB - 1
            elif dbi == -1:
                m &= ch["i"] == bi * NB
            elif dbj == 1:
                m &= ch["j"] == bj * NB + NB - 1
            else:
                m &= ch["j"] == bj * NB
            ch.loc[m, "buffer"] = True
    ch["split"] = [SPLITS[q] for q in ch["split_id"]]
    ch["chip_id"] = [f"i{a:03d}_j{b:03d}" for a, b in zip(ch["i"], ch["j"])]
    return ch


def hough(ndvi: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
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
    inst = np.zeros(ndvi.shape, np.uint16)
    rows: list[dict[str, Any]] = []
    for k in np.argsort(score)[::-1]:
        disc_m = np.zeros(ndvi.shape, bool)
        disc_m[disk((int(cy[k]), int(cx[k])), int(rr[k]), shape=ndvi.shape)] = True
        d = disc_m & green
        if d.sum() < FILL_MIN * disc_m.sum():
            continue
        dist = np.hypot(
            *np.ogrid[
                -int(cy[k]) : ndvi.shape[0] - int(cy[k]),
                -int(cx[k]) : ndvi.shape[1] - int(cx[k]),
            ]
        )
        ring = (dist > 1.08 * rr[k]) & (dist <= 1.35 * rr[k])
        ring_green = float(green[ring].mean()) if ring.any() else 1.0
        if rr[k] < R_KEEP or ring_green >= RING_MAX:
            continue
        if (d & (inst > 0)).sum() > OVL_NMS * d.sum():
            continue
        d = ndimage.binary_fill_holes(ndimage.binary_opening(d)) & (inst == 0)
        if d.sum() < MIN_PX:
            continue
        circ = float(4 * np.pi * d.sum() / max(perimeter(d), 1.0) ** 2)
        inst[d] = len(rows) + 1
        ys, xs = np.nonzero(d)
        rows.append(
            {
                "cand": len(rows) + 1,
                "col": int(cx[k]),
                "row": int(cy[k]),
                "bbox": (int(ys.min()), int(xs.min())),
                "mask": d[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy(),
                "r_px": int(rr[k]),
                "score": round(float(score[k]), 3),
                "area_px": int(d.sum()),
                "circularity": round(min(circ, 1.0), 3),
                "ring_green": round(ring_green, 3),
                "review": circ < TAU_C,
            }
        )
    return inst, rows


MARGIN = 32


def process(args: tuple[str, str, int, int]) -> tuple[str, float, list[dict]]:
    stack, chip_id, col_off, row_off = args
    m, size = MARGIN, PX + 2 * MARGIN
    with rasterio.open(stack) as src:
        b = src.descriptions.index("NDVI_peak") + 1
        win = Window(col_off - m, row_off - m, size, size)
        a = src.read(b, window=win, boundless=True, fill_value=0).astype(np.float32)
        core = Window(col_off, row_off, PX, PX)
        valid = float(
            (src.read(1, window=core, boundless=True, fill_value=0) != 0).mean()
        )
    _, rows = hough(a / 1e4)
    keep = []
    for r in rows:
        if m <= r["row"] < m + PX and m <= r["col"] < m + PX:
            r["g_row"], r["g_col"] = row_off - m + r["row"], col_off - m + r["col"]
            r["g_bbox"] = (row_off - m + r["bbox"][0], col_off - m + r["bbox"][1])
            r["col"], r["row"] = r["col"] - m, r["row"] - m
            keep.append(r)
    return chip_id, valid, keep


def dedupe(cands: list[dict]) -> list[dict]:
    cands = sorted(cands, key=lambda c: -c["score"])
    out: list[dict] = []
    for c in cands:
        if all(
            np.hypot(c["g_row"] - o["g_row"], c["g_col"] - o["g_col"])
            > 0.5 * max(c["r_px"], o["r_px"])
            for o in out
        ):
            out.append(c)
    return out


def write_masks(use: pd.DataFrame, cands: list[dict], prof: dict, transform) -> None:
    r_mod, c_mod = (
        int(use["row_off"].to_numpy()[0]) % PX,
        int(use["col_off"].to_numpy()[0]) % PX,
    )
    by_chip: dict[tuple[int, int], list[dict]] = {}
    for c in cands:
        h, w = c["mask"].shape
        y, x = c["g_bbox"][0] - r_mod, c["g_bbox"][1] - c_mod
        for r0 in range(y // PX, (y + h - 1) // PX + 1):
            for c0 in range(x // PX, (x + w - 1) // PX + 1):
                by_chip.setdefault((r0, c0), []).append(c)
    for chip_id, row_off, col_off in zip(
        use["chip_id"], use["row_off"].to_numpy(), use["col_off"].to_numpy()
    ):
        row_off, col_off = int(row_off), int(col_off)
        inst = np.zeros((PX, PX), np.uint16)
        key = ((row_off - r_mod) // PX, (col_off - c_mod) // PX)
        for c in by_chip.get(key, []):
            y0, x0 = c["g_bbox"][0] - row_off, c["g_bbox"][1] - col_off
            h, w = c["mask"].shape
            ys, xs = (
                slice(max(y0, 0), min(y0 + h, PX)),
                slice(max(x0, 0), min(x0 + w, PX)),
            )
            if ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            piece = c["mask"][
                ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
            ]
            sub = inst[ys, xs]
            sub[piece & (sub == 0)] = c["gid"]
        t = transform * rasterio.Affine.translation(col_off, row_off)
        with rasterio.open(
            f"{OUT}/cand_masks/{chip_id}.tif", "w", **dict(prof, transform=t)
        ) as dst:
            dst.write(inst, 1)


def main() -> None:
    stack, blocks_csv = sys.argv[1], sys.argv[2]
    os.makedirs(f"{OUT}/cand_masks", exist_ok=True)
    ch = chip_grid(pd.read_csv(blocks_csv))
    with rasterio.open(stack) as src:
        left, top, res = src.transform.c, src.transform.f, src.res[0]
        transform = src.transform
        prof = {
            "driver": "GTiff",
            "width": PX,
            "height": PX,
            "count": 1,
            "dtype": "uint16",
            "crs": src.crs,
            "compress": "deflate",
        }
    ch["col_off"] = ((ch["x_min"] - left) / res).round().astype(int)
    ch["row_off"] = ((top - ch["y_max"]) / res).round().astype(int)
    ch["stack"] = os.path.basename(stack)
    use = ch[~ch["buffer"]]
    print(
        f"{len(ch)} chips in the AOI: {len(use)} in the dataset, "
        f"{int(ch['buffer'].sum())} buffer;",
        use["split"].value_counts().to_dict(),
    )

    jobs = [(stack, r.chip_id, r.col_off, r.row_off) for r in ch.itertuples()]
    valid, cands = {}, []
    with ProcessPoolExecutor() as ex:
        for k, (cid, v, rows) in enumerate(ex.map(process, jobs, chunksize=8), 1):
            valid[cid] = v
            cands += [dict(chip_id=cid, **r) for r in rows]
            if k % 200 == 0 or k == len(jobs):
                print(
                    f"  {k}/{len(jobs)} chips, {len(cands)} candidates so far",
                    flush=True,
                )
    cands = dedupe(cands)
    for g, c in enumerate(cands, 1):
        c["gid"] = g
    write_masks(use, cands, prof, transform)
    cands = [
        {k: v for k, v in c.items() if k not in ("mask", "bbox", "g_bbox")}
        for c in cands
    ]

    ch["valid_frac"] = ch["chip_id"].map(valid)
    ch["n_cand"] = ch["chip_id"].map(
        pd.Series([c["chip_id"] for c in cands]).value_counts()
    )
    ch["n_cand"] = ch["n_cand"].fillna(0).astype(int)
    cols = [
        "chip_id",
        "i",
        "j",
        "bi",
        "bj",
        "split",
        "buffer",
        "lon",
        "lat",
        "x_min",
        "y_max",
        "col_off",
        "row_off",
        "stack",
        "valid_frac",
        "n_cand",
    ]
    ch[cols].to_csv(f"{OUT}/chips_index.csv", index=False)
    try:
        ch[cols].to_parquet(f"{OUT}/chips_index.parquet", index=False)
    except ImportError:
        print("  (pyarrow not installed: CSV index only)")

    cd = pd.DataFrame(cands)
    if len(cd):
        meta = ch.set_index("chip_id")
        cd["x"] = meta.loc[cd["chip_id"], "x_min"].to_numpy() + (cd["col"] + 0.5) * 10
        cd["y"] = meta.loc[cd["chip_id"], "y_max"].to_numpy() - (cd["row"] + 0.5) * 10
        cd["lon"], cd["lat"] = inv.transform(cd["x"].to_numpy(), cd["y"].to_numpy())
        cd["diameter_m"] = 2 * np.sqrt(cd["area_px"] / np.pi) * 10
        cd["split"] = meta.loc[cd["chip_id"], "split"].to_numpy()
        cd["buffer"] = meta.loc[cd["chip_id"], "buffer"].to_numpy()
        cd = cd.drop(columns=["g_row", "g_col"])
    cd.to_csv(f"{OUT}/candidates.csv", index=False)

    partial = int((ch.loc[~ch["buffer"], "valid_frac"] < 0.99).sum())
    nb = int(cd["buffer"].sum()) if len(cd) else 0
    if len(cd):
        cd = cd[~cd["buffer"]]
    print(
        f"\ndone: {len(cd)} candidates in {int((ch.loc[~ch['buffer'], 'n_cand'] > 0).sum())} "
        f"dataset chips ({nb} more in buffer chips), "
        f"{int(cd['review'].sum()) if len(cd) else 0} flagged for review"
    )
    if len(cd):
        print("candidates per split:", cd["split"].value_counts().to_dict())
        print(
            f"diameter: median {cd['diameter_m'].median():.0f} m, "
            f"range {cd['diameter_m'].min():.0f}-{cd['diameter_m'].max():.0f} m"
        )
    print(f"chips partly outside the stack (valid < 99 %): {partial}")
    print(f"written: {OUT}/chips_index.csv, {OUT}/candidates.csv, {OUT}/cand_masks/")


if __name__ == "__main__":
    main()
