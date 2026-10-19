#!/usr/bin/env python3

import glob
import os
import sys

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.windows import Window, from_bounds

year = sys.argv[1]
parts = sorted(glob.glob(f"s2stack_{year}_[NS][WE]*.tif"))
if not parts:
    sys.exit(f"no files s2stack_{year}_NW*.tif ... found in {os.getcwd()}")
print(len(parts), "input files:", *parts, sep="\n  ")

srcs = [rasterio.open(p) for p in parts]
res = srcs[0].res[0]
left = min(s.bounds.left for s in srcs)
bottom = min(s.bounds.bottom for s in srcs)
right = max(s.bounds.right for s in srcs)
top = max(s.bounds.top for s in srcs)
width = round((right - left) / res)
height = round((top - bottom) / res)
transform = rasterio.transform.from_origin(left, top, res, res)
names = srcs[0].descriptions
print(f"mosaic: {width} x {height} px, {srcs[0].count} bands, {srcs[0].dtypes[0]}")

tmp = f"s2stack_{year}_tmp.tif"
profile = srcs[0].profile.copy()
profile.update(
    driver="GTiff",
    width=width,
    height=height,
    transform=transform,
    tiled=True,
    blockxsize=512,
    blockysize=512,
    compress="deflate",
    predictor=2,
    bigtiff="yes",
    nodata=None,
)
with rasterio.open(tmp, "w+", **profile) as dst:
    for b in range(1, dst.count + 1):
        dst.set_band_description(b, names[b - 1])
    for s in srcs:
        win = (
            from_bounds(*s.bounds, transform=transform).round_offsets().round_lengths()
        )
        for _, w in s.block_windows(1):
            data = s.read(window=w)
            out = Window(
                win.col_off + w.col_off, win.row_off + w.row_off, w.width, w.height
            )
            cur = dst.read(window=out)
            valid = (data != 0).any(axis=0)
            cur[:, valid] = data[:, valid]
            dst.write(cur, window=out)
        print("  written", s.name)
for s in srcs:
    s.close()

out = f"s2stack_{year}.tif"
print("converting to COG ...")
rasterio.shutil.copy(
    tmp,
    out,
    driver="COG",
    compress="DEFLATE",
    predictor=2,
    bigtiff="YES",
    overview_resampling="average",
)
os.remove(tmp)
with rasterio.open(out) as f:
    a = f.read(
        f.descriptions.index("NDVI_peak") + 1, out_shape=(f.height // 16, f.width // 16)
    )
    print(
        f"done: {out}  {os.path.getsize(out) / 1e9:.2f} GB, "
        f"NDVI_peak median {np.median(a):.0f}, zero pixels {100 * (a == 0).mean():.1f} %"
    )
