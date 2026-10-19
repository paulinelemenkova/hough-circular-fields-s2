#!/usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

PX, HALF, TILE, NMAX = 256, 64, 192, 60


def main() -> None:
    stack = sys.argv[1]
    pv = pd.read_csv("labels_auto/pivots.csv")
    ix = pd.read_csv("chips/chips_index.csv")
    ix = ix[~ix["buffer"]]
    rows = []
    with rasterio.open(stack) as src:
        names = list(src.descriptions)
        bands = [names.index(b) + 1 for b in ("B8A_peak", "B4_peak", "B3_peak")]
        for p in pv.itertuples():
            c = ix[
                (ix.row_off <= p.row)
                & (p.row < ix.row_off + PX)
                & (ix.col_off <= p.col)
                & (p.col < ix.col_off + PX)
            ]
            if c.empty:
                continue
            c = c.iloc[0]
            if not os.path.exists(f"labels/{c.chip_id}.tif"):
                continue
            with rasterio.open(f"labels/{c.chip_id}.tif") as f:
                lab = f.read(1)
            y, x = int(p.row - c.row_off), int(p.col - c.col_off)
            ys, xs = np.ogrid[:PX, :PX]
            disc = (ys - y) ** 2 + (xs - x) ** 2 <= (0.7 * p.r_px) ** 2
            if (lab[disc] > 0).mean() > 0.3:
                continue
            rows.append(
                {
                    "id": p.id,
                    "source": p.source,
                    "split": c.split,
                    "chip_id": c.chip_id,
                    "row": p.row,
                    "col": p.col,
                    "r_px": p.r_px,
                    "lon": p.lon,
                    "lat": p.lat,
                }
            )
        df = pd.DataFrame(rows)
        print(
            f"{len(df)} automatic pivots are not in the hand labels:",
            df.groupby(["source", "split"]).size().to_dict(),
        )
        df["tile"] = 0
        for src_name in ("A", "B"):
            sub = df[df.source == src_name]

            sub = pd.concat(
                [
                    sub[sub.split == "test"],
                    sub[sub.split != "test"].sample(frac=1, random_state=0),
                ]
            ).head(NMAX)
            ncol = 10
            sheet = Image.new(
                "RGB",
                (ncol * TILE, ((len(sub) - 1) // ncol + 1) * (TILE + 18)),
                "white",
            )
            dr = ImageDraw.Draw(sheet)
            for k, p in enumerate(sub.itertuples()):
                a = src.read(
                    bands,
                    window=Window(p.col - HALF, p.row - HALF, 2 * HALF, 2 * HALF),
                    boundless=True,
                    fill_value=0,
                ).astype(float)
                lo, hi = np.percentile(a[a > 0], [2, 98]) if (a > 0).any() else (0, 1)
                im = Image.fromarray(
                    (np.clip((a - lo) / (hi - lo), 0, 1) * 255)
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                ).resize((TILE, TILE))
                d = ImageDraw.Draw(im)
                r = p.r_px * TILE / (2 * HALF)
                d.ellipse(
                    [TILE / 2 - r, TILE / 2 - r, TILE / 2 + r, TILE / 2 + r],
                    outline="yellow",
                    width=2,
                )
                x0, y0 = (k % ncol) * TILE, (k // ncol) * (TILE + 18)
                sheet.paste(im, (x0, y0 + 18))
                dr.text((x0 + 3, y0 + 3), f"{k + 1} {p.split} r{p.r_px}", fill="black")
                df.loc[p.Index, "tile"] = k + 1
            sheet.save(f"disagreements_{src_name}.jpg", quality=88)
            print(f"written: disagreements_{src_name}.jpg ({len(sub)} tiles)")
    df.to_csv("disagreements.csv", index=False)


if __name__ == "__main__":
    main()
