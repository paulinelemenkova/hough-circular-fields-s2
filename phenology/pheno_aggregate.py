#!/usr/bin/env python3

from __future__ import annotations

import sys

import pandas as pd

SEASONS = {
    "early": ("2025-04-01", "2025-05-31"),
    "peak": ("2025-06-01", "2025-08-31"),
    "late": ("2025-09-01", "2025-10-31"),
}


def main() -> None:
    d = pd.read_csv(sys.argv[1], parse_dates=["date"])
    d = d[["window", "class", "field", "date", "ndvi"]]
    d["uid"] = d["window"] + "-" + d["field"].astype(str)
    day = d["date"].dt.day
    offset = (day.clip(upper=30) - 1) // 10 * 10
    d["dekad_start"] = d["date"].dt.to_period("M").dt.to_timestamp() + pd.to_timedelta(
        offset, unit="D"
    )
    g = d.groupby(["class", "dekad_start"])["ndvi"]
    dek = pd.DataFrame(
        {
            "n_obs": g.size(),
            "q25": g.quantile(0.25),
            "median": g.median(),
            "q75": g.quantile(0.75),
        }
    ).reset_index()
    dek.round({"q25": 3, "median": 3, "q75": 3}).to_csv("pheno_dekad.csv", index=False)

    rows = []
    for s, (a, b) in SEASONS.items():
        sel = d[(d["date"] >= a) & (d["date"] <= b)]
        m = sel.groupby(["class", "uid"])["ndvi"].median().reset_index()
        m["season"] = s
        rows.append(m)
    win = pd.concat(rows)[["class", "uid", "season", "ndvi"]]
    win.round(3).to_csv("pheno_windows.csv", index=False)
    print(d.groupby("class")["uid"].nunique().to_string())
    print(len(dek), "dekad rows;", len(win), "window rows")


if __name__ == "__main__":
    main()
