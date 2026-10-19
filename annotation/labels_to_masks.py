#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy import ndimage
from site_config import EPSG

PX, M, OUT = 256, 32, "labels"
inv = Transformer.from_crs(EPSG, 4326, always_xy=True)


def main() -> None:
    with open(sys.argv[1]) as fh:
        lab = json.load(fh)
    stack = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    ix = pd.read_csv("chips/chips_index.csv")
    ix = ix[~ix["buffer"]].set_index("chip_id")
    done = set(lab.get("done", {}))
    print(f"{len(done)} of {len(ix)} chips finished")

    acc = {int(g): d for g, d in lab.get("dec", {}).items() if d["s"] == "a"}
    cd = pd.read_csv("chips/candidates.csv").set_index("gid")
    off = pd.read_csv("chips/chips_index.csv").set_index("chip_id")
    objs: list[dict] = []
    with rasterio.open(stack) as src:
        b = list(src.descriptions).index("NDVI_peak") + 1
        for g, d in acc.items():
            c = cd.loc[g]
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
            yy, xx = np.mgrid[0 : 2 * h + 1, 0 : 2 * h + 1]
            m = (np.hypot(yy + r0 - gr, xx + c0 - gc) <= c.r_px) & (nd > 0.5)
            m = ndimage.binary_fill_holes(ndimage.binary_opening(m))
            objs.append(
                {
                    "source": "candidate",
                    "gid": g,
                    "sector": d["sector"],
                    "r0": r0,
                    "c0": c0,
                    "mask": m,
                }
            )

    with rasterio.open(stack) as src:
        names = list(src.descriptions)
        bands = [names.index(f"NDVI_{s}") + 1 for s in ("early", "peak", "late")]
        transform = src.transform
        crs = src.crs
        for cid, lst in lab.get("added", {}).items():
            r = ix.loc[cid]
            for a in lst:
                gy = int(r.row_off) - M + a["y"]
                gx = int(r.col_off) - M + a["x"]
                rad = a["r"]
                h = int(np.ceil(rad)) + 2
                r0, c0 = round(gy) - h, round(gx) - h
                nd = (
                    src.read(
                        bands,
                        window=Window(c0, r0, 2 * h + 1, 2 * h + 1),
                        boundless=True,
                        fill_value=0,
                    ).max(axis=0)
                    / 1e4
                )
                yy, xx = np.mgrid[0 : 2 * h + 1, 0 : 2 * h + 1]
                disc = np.hypot(yy + r0 - gy, xx + c0 - gx) <= rad
                veg = disc & (nd > 0.5)
                if veg.sum() >= 0.5 * disc.sum():
                    veg = ndimage.binary_fill_holes(ndimage.binary_opening(veg)) & disc
                    m = veg if veg.sum() >= 50 else disc
                else:
                    m = disc
                objs.append(
                    {
                        "source": "added",
                        "gid": -1,
                        "sector": False,
                        "r0": r0,
                        "c0": c0,
                        "mask": m,
                    }
                )

    rows = []
    for k, o in enumerate(objs, 1):
        o["id"] = k
        ys, xs = np.nonzero(o["mask"])
        cy, cx = o["r0"] + ys.mean(), o["c0"] + xs.mean()
        x, y = transform * (cx + 0.5, cy + 0.5)
        lon, lat = inv.transform(x, y)
        area = float(o["mask"].sum()) * 100.0
        rows.append(
            {
                "id": k,
                "source": o["source"],
                "gid": o["gid"],
                "sector": o["sector"],
                "row": round(cy, 1),
                "col": round(cx, 1),
                "lon": round(lon, 5),
                "lat": round(lat, 5),
                "area_ha": round(area / 1e4, 2),
                "diameter_m": round(2 * np.sqrt(area / np.pi), 0),
            }
        )
    prof = {
        "driver": "GTiff",
        "width": PX,
        "height": PX,
        "count": 1,
        "dtype": "uint16",
        "crs": crs,
        "compress": "deflate",
    }
    stats = []
    for cid, r in ix.iterrows():
        if cid not in done:
            stats.append(
                {"chip_id": cid, "split": r.split, "finished": False, "n_pivots": 0}
            )
            continue
        ro, co = int(r.row_off), int(r.col_off)
        inst = np.zeros((PX, PX), np.uint16)
        for o in objs:
            h, w = o["mask"].shape
            y0, x0 = o["r0"] - ro, o["c0"] - co
            ys = slice(max(y0, 0), min(y0 + h, PX))
            xs = slice(max(x0, 0), min(x0 + w, PX))
            if ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            piece = o["mask"][
                ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
            ]
            sub = inst[ys, xs]
            sub[piece & (sub == 0)] = o["id"]
        t = transform * rasterio.Affine.translation(co, ro)
        with rasterio.open(f"{OUT}/{cid}.tif", "w", **dict(prof, transform=t)) as dst:
            dst.write(inst, 1)
        stats.append(
            {
                "chip_id": cid,
                "split": r.split,
                "finished": True,
                "n_pivots": int(len(np.unique(inst)) - 1),
            }
        )
    pd.DataFrame(rows).to_csv(f"{OUT}/pivots.csv", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(f"{OUT}/chips.csv", index=False)
    f = st[st.finished]
    print(
        f"{len(objs)} pivots ({sum(o['source'] == 'added' for o in objs)} added by hand, "
        f"{sum(o['sector'] for o in objs)} sector); {len(f)} chip masks written"
    )
    print(
        "pivots per split (chips with >= 1 pivot / chips):",
        {s: f"{int((g.n_pivots > 0).sum())}/{len(g)}" for s, g in f.groupby("split")},
    )


if __name__ == "__main__":
    main()
