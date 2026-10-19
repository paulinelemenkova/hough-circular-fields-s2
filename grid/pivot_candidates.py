#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pyproj import Transformer
from scipy import ndimage

from s2_common import compose
from site_config import AOI, EPSG


RES = 40.0


def disc_ok(m: npt.NDArray[np.bool_]) -> bool:
    area = float(m.sum())
    if area < 12:
        return False
    ys, xs = np.nonzero(m)
    extent = area / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
    edge = m & ~ndimage.binary_erosion(m)
    circ = 4 * np.pi * area / (float(edge.sum()) * 1.12) ** 2
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))))
    ecc = float(np.sqrt(1 - ev[0] / ev[1])) if ev[1] > 0 else 1.0
    return circ > 0.70 and ecc < 0.60 and 0.65 <= extent <= 0.88


def main() -> None:
    comp, tr, _w, _h = compose(
        AOI, RES, "2025-07-01", "2025-08-31", n_per_tile=2, bands=("B8A", "B04")
    )
    with np.errstate(all="ignore"):
        ndvi = (comp["B8A"] - comp["B04"]) / (comp["B8A"] + comp["B04"])
    core = ndimage.binary_erosion(np.nan_to_num(ndvi) > 0.5, iterations=1)
    lab, _n = ndimage.label(core)
    inv = Transformer.from_crs(EPSG, 4326, always_xy=True)
    rows = ["lon,lat,diameter_m"]
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        if not disc_ok(m):
            continue
        diam = 2 * np.sqrt(m.sum() / np.pi) * RES + 2 * RES
        if not 300 <= diam <= 1200:
            continue
        cy, cx = ndimage.center_of_mass(m)
        x = tr.c + (sl[1].start + cx + 0.5) * RES
        y = tr.f - (sl[0].start + cy + 0.5) * RES
        lon, lat = inv.transform(x, y)
        rows.append(f"{lon:.5f},{lat:.5f},{diam:.0f}")
    print(f"# {len(rows) - 1} pivot candidates")
    print("CANDSTART\n" + "\n".join(rows) + "\nCANDEND")


if __name__ == "__main__":
    main()
