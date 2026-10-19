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
from scipy.ndimage import map_coordinates
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from site_config import EPSG

PX, M, OUT = 256, 32, "labels_auto"
FEAT = ("NDVI_early", "NDVI_peak", "NDVI_late", "B11_peak", "B11_early")
RADII = np.arange(18, 61)
T_CON = (0.08, 0.08, 0.08, 0.02, 0.02)
ANG = np.linspace(0, 2 * np.pi, 32, endpoint=False)
F_IN, F_OUT = np.array([0.72, 0.82, 0.92]), np.array([1.10, 1.20, 1.30])
A_CIRC, A_FILL = 0.80, 0.85
B_TH, B_RBIG = 0.30, 28
B_SCORE_BIG, B_FRAC_BIG = 0.62, 0.85
B_SCORE_SMALL, B_FRAC_SMALL = 0.72, 0.94
B_GREEN = 0.50
inv = Transformer.from_crs(EPSG, 4326, always_xy=True)


def ang_contrast(
    a: np.ndarray, cy: float, cx: float, r: float, t: float
) -> tuple[float, float]:
    def ring(f: np.ndarray) -> np.ndarray:
        y = cy + np.outer(f * r, np.sin(ANG))
        x = cx + np.outer(f * r, np.cos(ANG))
        return map_coordinates(a, [y, x], order=1, mode="nearest").mean(0)

    d = np.abs(ring(F_IN) - ring(F_OUT))
    return float((d > t).mean()), float(d.mean() / t)


def detect_b(args: tuple[str, str, int, int]) -> tuple[str, list[dict[str, Any]]]:
    stack, cid, ro, co = args
    size = PX + 2 * M
    with rasterio.open(stack) as src:
        names = list(src.descriptions)
        win = Window(co - M, ro - M, size, size)
        imgs = [
            src.read(
                names.index(f) + 1, window=win, boundless=True, fill_value=0
            ).astype(float)
            / 1e4
            for f in FEAT
        ]
        ndwi = (
            src.read(
                names.index("NDWI_peak") + 1, window=win, boundless=True, fill_value=0
            )
            / 1e4
        )
    if (imgs[1] == 0).mean() > 0.9:
        return cid, []

    edges = [
        canny(a / t * 0.06, sigma=2.0, low_threshold=0.03, high_threshold=0.08)
        for a, t in zip(imgs, T_CON)
    ]
    union = np.logical_or.reduce(edges)
    acc = hough_circle(union, RADII, normalize=True)
    sc, cx, cy, rr = hough_circle_peaks(
        acc,
        RADII,
        min_xdistance=8,
        min_ydistance=8,
        threshold=B_TH,
        total_num_peaks=150,
    )
    yy, xx = np.ogrid[:size, :size]
    out: list[dict[str, Any]] = []
    for s, x, y, r in sorted(zip(sc, cx, cy, rr), reverse=True):
        big = r >= B_RBIG
        if s < (B_SCORE_BIG if big else B_SCORE_SMALL):
            continue
        k = int(
            np.argmax([ang_contrast(a, y, x, r, t)[0] for a, t in zip(imgs, T_CON)])
        )
        frac = ang_contrast(imgs[k], y, x, r, T_CON[k])[0]
        if frac < (B_FRAC_BIG if big else B_FRAC_SMALL):
            continue

        rs = np.arange(max(int(0.85 * r), 15), int(1.2 * r) + 1)
        r = int(
            rs[np.argmax([ang_contrast(imgs[k], y, x, q, T_CON[k])[1] for q in rs])]
        )
        disc = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        if (ndwi[disc] > 0).mean() > 0.05:
            continue
        if max(float(imgs[i][disc].mean()) for i in range(3)) < B_GREEN:
            continue
        if any(np.hypot(y - o["y"], x - o["x"]) < 0.8 * max(r, o["r"]) for o in out):
            continue
        out.append({"y": int(y), "x": int(x), "r": r, "score": float(s), "cov": frac})
    keep = []
    for o in out:
        if M <= o["y"] < M + PX and M <= o["x"] < M + PX:
            keep.append(
                {
                    "source": "B",
                    "gy": ro - M + o["y"],
                    "gx": co - M + o["x"],
                    "r": o["r"],
                    "score": round(o["score"], 3),
                    "cov": round(o["cov"], 3),
                }
            )
    return cid, keep


def source_a(src: rasterio.DatasetReader, ix: pd.DataFrame) -> list[dict[str, Any]]:
    cd = pd.read_csv("chips/candidates.csv")
    cd["fill"] = cd["area_px"] / (np.pi * cd["r_px"] ** 2)
    cd = cd[(cd["circularity"] >= A_CIRC) & (cd["fill"] >= A_FILL)]
    b = list(src.descriptions).index("NDVI_peak") + 1
    out = []
    for c in cd.itertuples():
        gy = int(ix.loc[c.chip_id, "row_off"]) + int(c.row)
        gx = int(ix.loc[c.chip_id, "col_off"]) + int(c.col)
        h = int(c.r_px) + 2
        nd = (
            src.read(
                b,
                window=Window(gx - h, gy - h, 2 * h + 1, 2 * h + 1),
                boundless=True,
                fill_value=0,
            )
            / 1e4
        )
        yy, xx = np.mgrid[0 : 2 * h + 1, 0 : 2 * h + 1]
        m = (np.hypot(yy - h, xx - h) <= c.r_px) & (nd > 0.5)
        m = ndimage.binary_fill_holes(ndimage.binary_opening(m))
        if m.sum() >= 50:
            out.append(
                {
                    "source": "A",
                    "gy": gy,
                    "gx": gx,
                    "r": int(c.r_px),
                    "score": float(c.score),
                    "cov": np.nan,
                    "r0": gy - h,
                    "c0": gx - h,
                    "mask": m,
                }
            )
    return out


def disc_mask(o: dict[str, Any]) -> None:
    h = o["r"] + 1
    yy, xx = np.mgrid[0 : 2 * h + 1, 0 : 2 * h + 1]
    o["mask"] = np.hypot(yy - h, xx - h) <= o["r"]
    o["r0"], o["c0"] = o["gy"] - h, o["gx"] - h


def overlap(a: dict[str, Any], b: dict[str, Any]) -> int:
    ha, wa = a["mask"].shape
    hb, wb = b["mask"].shape
    y0, y1 = max(a["r0"], b["r0"]), min(a["r0"] + ha, b["r0"] + hb)
    x0, x1 = max(a["c0"], b["c0"]), min(a["c0"] + wa, b["c0"] + wb)
    if y0 >= y1 or x0 >= x1:
        return 0
    ma = a["mask"][y0 - a["r0"] : y1 - a["r0"], x0 - a["c0"] : x1 - a["c0"]]
    mb = b["mask"][y0 - b["r0"] : y1 - b["r0"], x0 - b["c0"] : x1 - b["c0"]]
    return int((ma & mb).sum())


def rasterise(objs: list[dict[str, Any]], ro: int, co: int) -> np.ndarray:
    inst = np.zeros((PX, PX), np.uint16)
    for o in objs:
        h, w = o["mask"].shape
        y0, x0 = o["r0"] - ro, o["c0"] - co
        ys, xs = slice(max(y0, 0), min(y0 + h, PX)), slice(max(x0, 0), min(x0 + w, PX))
        if ys.start >= ys.stop or xs.start >= xs.stop:
            continue
        piece = o["mask"][ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
        sub = inst[ys, xs]
        sub[piece & (sub == 0)] = o["id"]
    return inst


def match(ref: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    pairs = []
    for p in np.unique(pred[pred > 0]):
        pm = pred == p
        for r in np.unique(ref[pm & (ref > 0)]):
            rm = ref == r
            iou = (pm & rm).sum() / (pm | rm).sum()
            if iou >= 0.5:
                pairs.append((iou, p, r))
    tp_p: set = set()
    tp_r: set = set()
    for _, p, r in sorted(pairs, reverse=True):
        if p not in tp_p and r not in tp_r:
            tp_p.add(p)
            tp_r.add(r)
    n_p = len(np.unique(pred[pred > 0]))
    n_r = len(np.unique(ref[ref > 0]))
    return len(tp_p), n_p - len(tp_p), n_r - len(tp_r)


def main() -> None:
    stack = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    ix = pd.read_csv("chips/chips_index.csv").set_index("chip_id")
    jobs = [(stack, cid, int(r.row_off), int(r.col_off)) for cid, r in ix.iterrows()]
    print(f"source B: multi-season circle search on {len(jobs)} chips ...", flush=True)
    objs_b: list[dict[str, Any]] = []
    with ProcessPoolExecutor() as ex:
        for k, (_, rows) in enumerate(ex.map(detect_b, jobs, chunksize=4), 1):
            objs_b += rows
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} chips, {len(objs_b)} circles", flush=True)
    with rasterio.open(stack) as src:
        transform = src.transform
        crs = src.crs
        objs_a = source_a(src, ix)
    print(
        f"source A: {len(objs_a)} Hough candidates pass C >= {A_CIRC}, fill >= {A_FILL}"
    )

    objs: list[dict[str, Any]] = []
    for o in sorted(objs_a, key=lambda o: -o["score"]):
        if all(
            np.hypot(o["gy"] - p["gy"], o["gx"] - p["gx"]) > 0.5 * max(o["r"], p["r"])
            for p in objs
        ):
            objs.append(o)
    for o in sorted(objs_b, key=lambda o: -o["score"] * o["cov"]):
        disc_mask(o)
        if all(
            np.hypot(o["gy"] - p["gy"], o["gx"] - p["gx"]) > 0.5 * max(o["r"], p["r"])
            and overlap(o, p) < 0.3 * o["mask"].sum()
            for p in objs
        ):
            objs.append(o)
    for k, o in enumerate(objs, 1):
        o["id"] = k

    rows = []
    for o in objs:
        x, y = transform * (o["gx"] + 0.5, o["gy"] + 0.5)
        lon, lat = inv.transform(x, y)
        rows.append(
            {
                "id": o["id"],
                "source": o["source"],
                "row": o["gy"],
                "col": o["gx"],
                "r_px": o["r"],
                "score": o["score"],
                "coverage": o["cov"],
                "lon": round(lon, 5),
                "lat": round(lat, 5),
                "area_ha": round(float(o["mask"].sum()) / 100, 2),
            }
        )
    pv = pd.DataFrame(rows)
    pv.to_csv(f"{OUT}/pivots.csv", index=False)

    prof = {
        "driver": "GTiff",
        "width": PX,
        "height": PX,
        "count": 1,
        "dtype": "uint16",
        "crs": crs,
        "compress": "deflate",
    }
    use = ix[~ix["buffer"]]
    evals, has = [], {}
    for cid, r in use.iterrows():
        ro, co = int(r.row_off), int(r.col_off)
        inst = rasterise(objs, ro, co)
        t = transform * rasterio.Affine.translation(co, ro)
        with rasterio.open(f"{OUT}/{cid}.tif", "w", **dict(prof, transform=t)) as dst:
            dst.write(inst, 1)
        has[cid] = bool((inst > 0).any())
        hand = f"labels/{cid}.tif"
        if r.split == "test" and os.path.exists(hand):
            with rasterio.open(hand) as f:
                ref = f.read(1)
            tp, fp, fn = match(ref, inst)
            evals.append({"chip_id": cid, "tp": tp, "fp": fp, "fn": fn})
    print(
        f"\ndone: {len(objs)} pivots ({int((pv.source == 'A').sum())} from A, "
        f"{int((pv.source == 'B').sum())} from B), masks for {len(use)} chips in {OUT}/"
    )
    print(
        "chips with >= 1 pivot per split:",
        {
            s: f"{sum(has[c] for c in g.index)}/{len(g)}"
            for s, g in use.groupby("split")
        },
    )
    if evals:
        ev = pd.DataFrame(evals)
        ev.to_csv(f"{OUT}/eval_test.csv", index=False)
        tp, fp, fn = ev.tp.sum(), ev.fp.sum(), ev.fn.sum()
        p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(
            f"agreement with hand labels on {len(ev)} test chips: precision {p:.2f}, "
            f"recall {rc:.2f}, F1 {2 * p * rc / max(p + rc, 1e-9):.2f}"
        )


if __name__ == "__main__":
    main()
