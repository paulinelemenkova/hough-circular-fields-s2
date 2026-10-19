#!/usr/bin/env python3

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.errors
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely.geometry import box, shape
from site_config import EPSG

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

STAC = "https://earth-search.aws.element84.com/v1/search"
WC = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N36E0{lon:02d}_Map.tif"
)
CRS = f"EPSG:{EPSG}"
RES, HALF = 20.0, 6000.0
BAD_SCL = (0, 1, 3, 8, 9, 10, 11)
F32: TypeAlias = npt.NDArray[np.float32]


def stac_items(bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "limit": 100,
        "datetime": "2025-03-01T00:00:00Z/2025-11-30T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 70}},
    }
    url, feats = STAC, []
    while True:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        page = json.load(urllib.request.urlopen(req, timeout=90))
        feats.extend(page["features"])
        nxt = [lk for lk in page.get("links", []) if lk.get("rel") == "next"]
        if not nxt or not page["features"]:
            return feats
        url = nxt[0]["href"]
        body = {**body, **nxt[0].get("body", {})}


def read_into(
    href: str, dst: npt.NDArray[Any], tr: Affine, rs: Resampling, level: int = -1
) -> None:
    h, w = dst.shape
    bounds = (tr.c, tr.f + h * tr.e, tr.c + w * tr.a, tr.f)
    opts = {"OVERVIEW_LEVEL": level} if level >= 0 else {}
    with rasterio.open(href, **opts) as src:
        if src.crs.to_string() != CRS:
            t = Transformer.from_crs(CRS, src.crs, always_xy=True)
            xs, ys = t.transform([bounds[0], bounds[2]], [bounds[1], bounds[3]])
            sb = (min(xs), min(ys), max(xs), max(ys))
        else:
            sb = bounds
        win = from_bounds(*sb, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        win = win.round_offsets().round_lengths()
        arr = src.read(1, window=win)
        reproject(
            arr,
            dst,
            src_transform=src.window_transform(win),
            src_crs=src.crs,
            dst_transform=tr,
            dst_crs=CRS,
            resampling=rs,
            src_nodata=0,
            dst_nodata=0,
        )


def scene_ndvi(f: dict[str, Any], tr: Affine, n: int) -> F32:
    a = f["assets"]
    scl = np.zeros((n, n), np.uint8)
    red = np.zeros((n, n), np.float32)
    nir = np.zeros((n, n), np.float32)
    read_into(a["scl"]["href"], scl, tr, Resampling.nearest)
    read_into(a["red"]["href"], red, tr, Resampling.average, level=0)
    read_into(a["nir08"]["href"], nir, tr, Resampling.bilinear)
    off = (
        0.0 if f["properties"].get("earthsearch:boa_offset_applied", False) else 1000.0
    )
    r, q = (red - off) / 1e4, (nir - off) / 1e4
    ok = (~np.isin(scl, BAD_SCL)) & (red > 0) & (nir > 0) & (r + q > 0.02)
    with np.errstate(all="ignore"):
        nd = ((q - r) / (q + r)).astype(np.float32)
    return np.where(ok, np.clip(nd, -1, 1), np.nan).astype(np.float32)


def disc_ok(m: npt.NDArray[np.bool_]) -> bool:
    area = float(m.sum())
    if not 100 <= area <= 2200:
        return False
    ys, xs = np.nonzero(m)
    extent = area / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
    edge = m & ~ndimage.binary_erosion(m)
    perim = float(edge.sum()) * 1.12
    circ = 4 * np.pi * area / perim**2
    cov = np.cov(np.vstack([xs, ys]))
    ev = np.sort(np.linalg.eigvalsh(cov))
    ecc = float(np.sqrt(1 - ev[0] / ev[1])) if ev[1] > 0 else 1.0
    return circ > 0.75 and ecc < 0.55 and 0.70 <= extent <= 0.85


def main() -> None:
    tag, lon, lat = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    fwd = Transformer.from_crs(4326, CRS, always_xy=True)
    inv = Transformer.from_crs(CRS, 4326, always_xy=True)
    xc, yc = fwd.transform(lon, lat)
    x0, y1 = np.floor((xc - HALF) / RES) * RES, np.ceil((yc + HALF) / RES) * RES
    n = int(2 * HALF / RES)
    tr = from_origin(x0, y1, RES, RES)
    w, s = inv.transform(x0, y1 - n * RES)
    e, nn = inv.transform(x0 + n * RES, y1)
    win_ll = box(w, s, e, nn)

    feats = [
        f for f in stac_items((w, s, e, nn)) if shape(f["geometry"]).contains(win_ll)
    ]
    tiles: dict[str, int] = {}
    for f in feats:
        t = f["id"].split("_")[1]
        tiles[t] = tiles.get(t, 0) + 1
    tile = max(tiles, key=lambda k: tiles[k])
    feats = [f for f in feats if f["id"].split("_")[1] == tile]
    seen: dict[str, dict[str, Any]] = {}
    for f in feats:
        d = f["properties"]["datetime"][:10]
        if (
            d not in seen
            or f["properties"]["eo:cloud_cover"]
            < seen[d]["properties"]["eo:cloud_cover"]
        ):
            seen[d] = f
    dates = sorted(seen)
    print(f"# {tag}: tile {tile}, {len(dates)} dates", file=sys.stderr)
    with ThreadPoolExecutor(12) as ex:
        stack = np.stack(list(ex.map(lambda d: scene_ndvi(seen[d], tr, n), dates)))
    doy = np.array([np.datetime64(d) for d in dates])

    def win_med(a: str, b: str) -> F32:
        sel = (doy >= np.datetime64(a)) & (doy <= np.datetime64(b))
        with np.errstate(all="ignore"):
            out: F32 = np.nanmedian(stack[sel], axis=0)
        return out

    peak, spring = (
        win_med("2025-07-01", "2025-08-31"),
        win_med("2025-04-15", "2025-05-31"),
    )
    with np.errstate(all="ignore"):
        p90 = np.nanpercentile(stack, 90, axis=0)

    wc = np.zeros((n, n), np.uint8)
    read_into(WC.format(lon=30 if lon < 33 else 33), wc, tr, Resampling.mode, level=0)
    crop = wc == 40

    core = ndimage.binary_erosion(np.nan_to_num(peak) > 0.5, iterations=3)
    lab, _nl = ndimage.label(core)
    piv = np.zeros((n, n), np.int32)
    k = 0
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        if disc_ok(m):
            k += 1
            sub = piv[sl]
            sub[m] = k
    classes = {"pivot": piv}
    for name, mask in (
        ("rainfed", crop & (spring >= 0.45) & (peak <= 0.22)),
        ("fallow", crop & (p90 <= 0.20)),
    ):
        mask = ndimage.binary_opening(mask) & (piv == 0)
        lab2, _ = ndimage.label(mask)
        sizes = np.bincount(lab2.ravel())
        keep = np.nonzero(sizes >= 25)[0]
        keep = keep[keep > 0]
        rng = np.random.default_rng(0)
        keep = (
            rng.choice(keep, size=min(60, keep.size), replace=False)
            if keep.size
            else keep
        )
        out = np.zeros_like(lab2)
        for j, lb in enumerate(sorted(keep), start=1):
            out[lab2 == lb] = j
        classes[name] = out

    rows = ["window,class,field,date,ndvi"]
    for name, fl in classes.items():
        ids = np.unique(fl)
        ids = ids[ids > 0]
        print(f"# {tag}: {name} fields {ids.size}", file=sys.stderr)
        for fid in ids:
            px = stack[:, fl == fid]
            valid = np.isfinite(px).mean(axis=1)
            with np.errstate(all="ignore"):
                med = np.nanmedian(px, axis=1)
            for d, v, q in zip(dates, med, valid):
                if q >= 0.6 and np.isfinite(v):
                    rows.append(f"{tag},{name},{fid},{d},{v:.4f}")
    blob = base64.b64encode(gzip.compress("\n".join(rows).encode())).decode()
    print(f"# {tag}: {len(rows) - 1} rows", file=sys.stderr)
    print(f"CSV{tag}START{blob}CSV{tag}END")


if __name__ == "__main__":
    main()
