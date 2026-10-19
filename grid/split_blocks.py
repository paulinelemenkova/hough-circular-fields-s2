#!/usr/bin/env python3

from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt
import pandas as pd
from pyproj import Transformer
from site_config import AOI, EPSG

IN, OUT = sys.argv[1], sys.argv[2]

CHIP, NB = 2560.0, 4
SPLITS = ("train", "val", "test")
PI = np.array([0.70, 0.15, 0.15])
SEED, TRIALS = 42, 1000

fwd = Transformer.from_crs(4326, EPSG, always_xy=True)
inv = Transformer.from_crs(EPSG, 4326, always_xy=True)


def polygon(x0: float, y0: float, size: float, n: int = 8) -> str:
    t = np.linspace(0, 1, n + 1)[:-1]
    xs = np.concatenate(
        [x0 + t * size, np.full(n, x0 + size), x0 + size - t * size, np.full(n, x0)]
    )
    ys = np.concatenate(
        [np.full(n, y0), y0 + t * size, np.full(n, y0 + size), y0 + size - t * size]
    )
    lon, lat = inv.transform(xs, ys)
    pts = [f"{a:.5f} {b:.5f}" for a, b in zip(lon, lat)]
    return "\n".join(pts + pts[:1])


def objective(phi: npt.NDArray[np.int_], strata: npt.NDArray[np.int_]) -> float:
    j = 0.0
    for d in np.unique(strata):
        sel = strata == d
        n_d = sel.sum()
        for q in range(3):
            j += ((phi[sel] == q).sum() / n_d - PI[q]) ** 2
    return j


def main() -> None:
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
    chips = pd.DataFrame(
        {
            "i": ci[inside],
            "j": cj[inside],
            "bi": ci[inside] // NB,
            "bj": cj[inside] // NB,
        }
    )

    piv = pd.read_csv(f"{IN}/pivot_candidates_2025.csv")
    px, py = fwd.transform(piv["lon"].to_numpy(), piv["lat"].to_numpy())
    piv["i"] = ((px - x0) // CHIP).astype(int)
    piv["j"] = ((py - y0) // CHIP).astype(int)
    piv["bi"], piv["bj"] = piv["i"] // NB, piv["j"] // NB

    blocks = chips.groupby(["bi", "bj"]).size().rename("n_chips").reset_index()
    cnt = piv.groupby(["bi", "bj"]).size().rename("n_piv")
    blocks = blocks.merge(cnt, on=["bi", "bj"], how="left").fillna({"n_piv": 0})
    nz = blocks.loc[blocks["n_piv"] > 0, "n_piv"]
    edges = np.quantile(nz, [1 / 3, 2 / 3])
    blocks["stratum"] = np.where(
        blocks["n_piv"] == 0,
        0,
        1 + np.searchsorted(edges, blocks["n_piv"], side="left"),
    )
    strata = blocks["stratum"].to_numpy()

    rng = np.random.default_rng(SEED)
    nb = len(blocks)
    n_tr, n_va = round(PI[0] * nb), round(PI[1] * nb)
    best_j, best_phi = np.inf, np.zeros(nb, int)
    for _ in range(TRIALS):
        phi = np.full(nb, 2)
        perm = rng.permutation(nb)
        phi[perm[:n_tr]] = 0
        phi[perm[n_tr : n_tr + n_va]] = 1
        jv = objective(phi, strata)
        if jv < best_j:
            best_j, best_phi = jv, phi
    blocks["split"] = best_phi
    print(f"blocks {nb}, strata edges {edges}, best J = {best_j:.5f}")

    split_of = {(r.bi, r.bj): r.split for r in blocks.itertuples()}
    chips["split"] = [split_of[(a, b)] for a, b in zip(chips["bi"], chips["bj"])]
    chips["buffer"] = False
    for (bi, bj), q in split_of.items():
        for dbi, dbj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q2 = split_of.get((bi + dbi, bj + dbj))
            if q2 is None or q2 == q or q > q2:
                continue
            m = (chips["bi"] == bi) & (chips["bj"] == bj)
            if dbi == 1:
                m &= chips["i"] == bi * NB + NB - 1
            elif dbi == -1:
                m &= chips["i"] == bi * NB
            elif dbj == 1:
                m &= chips["j"] == bj * NB + NB - 1
            else:
                m &= chips["j"] == bj * NB
            chips.loc[m, "buffer"] = True

    for q, name in enumerate(SPLITS):
        sel = chips[(chips["split"] == q) & ~chips["buffer"]]
        with open(f"{OUT}/chips_{name}.gmt", "w") as fh:
            fh.writelines(
                ">\n" + polygon(x0 + r.i * CHIP, y0 + r.j * CHIP, CHIP, 2) + "\n"
                for r in sel.itertuples()
            )
    with open(f"{OUT}/chips_buffer.gmt", "w") as fh:
        fh.writelines(
            ">\n" + polygon(x0 + r.i * CHIP, y0 + r.j * CHIP, CHIP, 2) + "\n"
            for r in chips[chips["buffer"]].itertuples()
        )
    with open(f"{OUT}/blocks.gmt", "w") as fh:
        fh.writelines(
            ">\n"
            + polygon(x0 + r.bi * NB * CHIP, y0 + r.bj * NB * CHIP, NB * CHIP)
            + "\n"
            for r in blocks.itertuples()
        )
    piv[["lon", "lat"]].to_csv(f"{OUT}/pivots.txt", sep=" ", header=False, index=False)

    key = chips.set_index(["i", "j"])
    piv["split"] = [
        key.loc[(a, b), "split"] if (a, b) in key.index else -1
        for a, b in zip(piv["i"], piv["j"])
    ]
    piv["buffer"] = [
        bool(key.loc[(a, b), "buffer"]) if (a, b) in key.index else True
        for a, b in zip(piv["i"], piv["j"])
    ]
    rows = []
    for q, name in enumerate(SPLITS):
        rows.append(
            {
                "split": name,
                "blocks": int((blocks["split"] == q).sum()),
                "chips": int(((chips["split"] == q) & ~chips["buffer"]).sum()),
                "buffer_chips": int(((chips["split"] == q) & chips["buffer"]).sum()),
                "pivot_candidates": int(((piv["split"] == q) & ~piv["buffer"]).sum()),
            }
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT}/split_summary.csv", index=False)
    blocks.to_csv(f"{OUT}/blocks.csv", index=False)
    print(summ.to_string(index=False))
    for d in range(4):
        sel = strata == d
        print(
            f"stratum {d}: {sel.sum()} blocks ->",
            [int((best_phi[sel] == q).sum()) for q in range(3)],
        )


if __name__ == "__main__":
    main()
