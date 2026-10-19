#!/usr/bin/env python3

import sys

import numpy as np
import rasterio
from PIL import Image

path = sys.argv[1]
with rasterio.open(path) as src:
    print("size (px)     :", src.width, "x", src.height, "| bands:", src.count)
    print("data type     :", src.dtypes[0], "| nodata:", src.nodata)
    print("CRS / pixel   :", src.crs, "|", src.res)
    print("compression   :", src.compression, "| tiled:", src.is_tiled)
    print("overviews     :", src.overviews(1))
    names = list(src.descriptions)
    print("band names    :", names)
    for name in ("B8A_peak", "B4_peak", "NDVI_early", "NDVI_peak", "NDVI_late"):
        if name in names:
            a = src.read(names.index(name) + 1, masked=True)
            print(
                f"{name:13s} : min {a.min()}  median {np.ma.median(a):.0f}  "
                f"max {a.max()}  masked {100 * np.ma.getmaskarray(a).mean():.1f} %"
            )
    idx = [names.index(b) + 1 for b in ("B8A_peak", "B4_peak", "B3_peak")]
    rgb = src.read(idx, out_shape=(3, src.height // 4, src.width // 4)).astype(float)
lo, hi = np.nanpercentile(rgb, [2, 98])
img = (np.clip((rgb - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
out = path.replace(".tif", "_quicklook.png")
Image.fromarray(img).save(out)
print("quicklook     :", out)
