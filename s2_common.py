from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from site_config import EPSG

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

STAC = "https://earth-search.aws.element84.com/v1/search"
CRS = f"EPSG:{EPSG}"
BANDS = {"B8A": "nir08", "B04": "red", "B03": "green", "B02": "blue"}
BAD_SCL = (3, 8, 9, 10, 11)


def search(bbox, start, end, max_cloud=1.0):
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 300,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    req = urllib.request.Request(
        STAC,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    feats = json.load(urllib.request.urlopen(req, timeout=90))["features"]
    out = {}
    for f in feats:
        p = f["properties"]
        if p.get("s2:nodata_pixel_percentage", 100) > 1.0:
            continue
        tile = f"{p['mgrs:utm_zone']}{p['mgrs:latitude_band']}{p['mgrs:grid_square']}"
        out.setdefault(tile, []).append((p["eo:cloud_cover"], p["datetime"][:10], f))
    return {t: [x[2] for x in sorted(v, key=lambda x: x[0])] for t, v in out.items()}


def grid(bounds_lonlat, res):
    w, s, e, n = bounds_lonlat
    fwd = Transformer.from_crs(4326, CRS, always_xy=True)
    xs, ys = zip(*[fwd.transform(x, y) for x in (w, e) for y in (s, n)])
    x0, y1 = np.floor(min(xs) / res) * res, np.ceil(max(ys) / res) * res
    width = int(np.ceil((max(xs) - x0) / res))
    height = int(np.ceil((y1 - min(ys)) / res))
    return from_origin(x0, y1, res, res), width, height


def _read_into(href, dst, transform, resampling, res):
    native = 20.0 if href.endswith(("B8A.tif", "SCL.tif")) else 10.0
    level = int(np.floor(np.log2(res / native))) - 1 if res >= 2 * native else -1
    opts = {"OVERVIEW_LEVEL": level} if level >= 0 else {}
    h, w = dst.shape
    left, top = transform.c, transform.f
    bounds = (left, top - h * transform.e * -1, left + w * transform.a, top)
    with rasterio.open(href, **opts) as src:
        win = from_bounds(*bounds, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        win = win.round_offsets().round_lengths()
        arr = src.read(1, window=win)
        reproject(
            arr,
            dst,
            src_transform=src.window_transform(win),
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=CRS,
            resampling=resampling,
            src_nodata=0,
            dst_nodata=0,
        )


def _scene(f, transform, width, height, bands, res):
    scl = np.zeros((height, width), np.uint8)
    _read_into(f["assets"]["scl"]["href"], scl, transform, Resampling.nearest, res)
    ok = (scl > 0) & ~np.isin(scl, BAD_SCL)
    off = f["properties"].get("earthsearch:boa_offset_applied", False)
    out = {}
    for b in bands:
        arr = np.zeros((height, width), np.float32)
        _read_into(
            f["assets"][BANDS[b]]["href"],
            arr,
            transform,
            Resampling.average if res > 20 else Resampling.bilinear,
            res,
        )

        rho = (arr - (0 if off else 1000)) / 10000.0
        out[b] = np.where(ok & (arr > 0), rho, np.nan)
    return out


def compose(
    bounds_lonlat,
    res,
    start,
    end,
    n_per_tile=4,
    bands=("B8A", "B04", "B03"),
    log=print,
    workers=8,
):
    transform, width, height = grid(bounds_lonlat, res)
    items = search(bounds_lonlat, start, end)
    feats = []
    for tile, fs in sorted(items.items()):
        for f in fs[:n_per_tile]:
            log(
                f"  {tile} {f['properties']['datetime'][:10]} "
                f"cloud={f['properties']['eo:cloud_cover']:.3f}%"
            )
            feats.append(f)
    with ThreadPoolExecutor(workers) as ex:
        scenes = list(
            ex.map(lambda f: _scene(f, transform, width, height, bands, res), feats)
        )
    with np.errstate(all="ignore"):
        comp = {
            b: np.nanmedian(np.stack([sc[b] for sc in scenes]), axis=0).astype(
                np.float32
            )
            for b in bands
        }
    return comp, transform, width, height


def stretch(a, lo=2, hi=98):
    v = a[np.isfinite(a)]
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / (p2 - p1), 0, 1)
