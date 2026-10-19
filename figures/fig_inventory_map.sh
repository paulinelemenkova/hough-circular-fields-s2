#!/usr/bin/env bash
set -euo pipefail

INV=${1:-pivots_2025.csv}
OUT=${OUT:-.}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

R=${R:?set R=W/E/S/N of panel a}
RZ=${RZ:?set RZ=W/E/S/N of panel b}
EPSG=${EPSG:?set EPSG, e.g. 326NN}
AOI=${AOI:?set AOI=W/E/S/N outline}
J=M16c

python3 - "$WORK" "$INV" "$EPSG" <<'EOF'
import sys
import numpy as np
import pandas as pd
from pyproj import Transformer

work, inv, epsg = sys.argv[1], sys.argv[2], int(sys.argv[3])
fwd = Transformer.from_crs(4326, epsg, always_xy=True)
back = Transformer.from_crs(epsg, 4326, always_xy=True)
df = pd.read_csv(inv)
x, y = fwd.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
df["d_km"] = 2 * np.sqrt(df["area_ha"] * 1e4 / np.pi) / 1000
df[["lon", "lat", "d_km"]].to_csv(f"{work}/pivots.txt", sep=" ", header=False, index=False)

def hexbin(hexd: float, tag: str) -> int:
    s = hexd / np.sqrt(3)
    dx, dy = 1.5 * s, hexd
    col = np.round(x / dx).astype(int)
    cand = []
    for dc in (-1, 0, 1):
        c = col + dc
        off = (c % 2) * dy / 2
        r = np.round((y - off) / dy).astype(int)
        cand.append((np.hypot(x - c * dx, y - (r * dy + off)), c, r))
    k = np.stack([q[0] for q in cand]).argmin(axis=0)
    cc = np.choose(k, [q[1] for q in cand])
    rr = np.choose(k, [q[2] for q in cand])
    g = pd.Series(1, index=pd.MultiIndex.from_arrays([cc, rr])).groupby(level=[0, 1]).sum()
    ang = np.radians(np.arange(0, 361, 60))
    with open(f"{work}/hex_{tag}.gmt", "w") as f:
        for (c, r), n in g.items():
            hx, hy = c * dx, r * dy + (c % 2) * dy / 2
            lo, la = back.transform(hx + s * np.cos(ang), hy + s * np.sin(ang))
            f.write(f"> -Z{n}\n" + "".join(f"{a:.5f} {b:.5f}\n" for a, b in zip(lo, la)))
            clo, cla = back.transform(hx, hy)
            with open(f"{work}/lab_{tag}.txt", "a") as fl:
                fl.write(f"{clo:.5f} {cla:.5f} {n}\n")
    return int(g.max())

na, nb = hexbin(5000.0, "a"), hexbin(1500.0, "b")
with open(f"{work}/stats.txt", "w") as f:
    f.write(f"{len(df)} {df['area_ha'].sum() / 100:.1f} {np.median(df['d_km']) * 1000:.0f} {na} {nb}\n")
print(f"{len(df)} pivots, {df['area_ha'].sum() / 100:.1f} km2, median diameter "
      f"{np.median(df['d_km']) * 1000:.0f} m; max per bin: 5 km {na}, 1.5 km {nb}")
EOF
read -r NPIV AKM2 DMED NA NB <"$WORK/stats.txt"

HA=$(gmt mapproject -R$R -J$J -Wh)
HB=$(gmt mapproject -R$RZ -J$J -Wh)
gmt begin "$OUT/fig_inventory_map" pdf,png E600
  gmt set FONT_ANNOT_PRIMARY 8.5p,Helvetica FONT_LABEL 9p,Helvetica \
          MAP_FRAME_TYPE fancy MAP_FRAME_WIDTH 2.5p FORMAT_GEO_MAP ddd.xxF \
          MAP_TICK_LENGTH_PRIMARY 3p PS_CHAR_ENCODING ISOLatin1+
  gmt grdcut @earth_relief_03s -R$R -G"$WORK/relief.nc"
  gmt grdgradient "$WORK/relief.nc" -A315 -Ne0.6 -G"$WORK/shade.nc"
  gmt makecpt -Cgray -T700/2400 -H >"$WORK/relief.cpt"
  gmt makecpt -Cturbo -T0.5/$((NA + 1))/1 -H >"$WORK/a.cpt"
  gmt makecpt -Cturbo -T0.5/$((NB + 1))/1 -H >"$WORK/b.cpt"

  gmt grdimage "$WORK/relief.nc" -R$R -J$J -C"$WORK/relief.cpt" -I"$WORK/shade.nc" -t30 \
      -Y"$(awk -v h="$HB" 'BEGIN{print h+1.3}')c"
  gmt coast -Di -A5/2/2 -C#7FB2DE -W2/0.3p,#2B5C8A
  gmt plot "$WORK/hex_a.gmt" -C"$WORK/a.cpt" -L -W0.5p,gray20 -t8
  gmt text "$WORK/lab_a.txt" -F+f7p,Helvetica-Bold,black=1.5p,white+jMC
  gmt text "$WORK/lab_a.txt" -F+f7p,Helvetica-Bold,black+jMC
  echo "$AOI" | awk -F/ '{printf "%s %s\n%s %s\n%s %s\n%s %s\n%s %s\n",$1,$3,$2,$3,$2,$4,$1,$4,$1,$3}' | gmt plot -A -W0.8p,black,.
  echo "$RZ" | awk -F/ '{print $1, $3, $2, $4}' | awk '{printf "%s %s\n%s %s\n%s %s\n%s %s\n%s %s\n",$1,$2,$3,$2,$3,$4,$1,$4,$1,$2}' \
      | gmt plot -A -W1.4p,black
  echo "$RZ" | awk -F/ '{print $2, $4, "(b)"}' | gmt text -F+f9p,Helvetica-Bold+jBL -D4p/4p -Gwhite@15 -C0.05c/0.03c
  gmt basemap -Bxa0.25f0.05 -Bya0.2f0.05 -BWsNe -LjBL+o0.3c/0.35c+c37.75N+w20k+f+lkm \
      -F+gwhite@10+p0.4p,gray40+c0.08c
  echo "(a) Pivots per 5 km hexagon" | gmt text -F+cTL+f9.5p,Helvetica-Bold+jTL -Dj0.2c/0.2c \
      -Gwhite@10 -C0.1c/0.07c -W0.4p,gray40
  echo "2025: $NPIV pivots, $AKM2 km\262, median diameter $DMED m" | gmt text -F+cBR+f8.5p,Helvetica+jBR \
      -Dj0.2c/0.25c -Gwhite@10 -C0.1c/0.07c -W0.4p,gray40
  gmt colorbar -C"$WORK/a.cpt" -DJMR+o0.3c/0c+w${HA}c/0.32c -Bxa1+l"Pivots per 5 km hexagon"

  gmt basemap -R$RZ -J$J -B+g#F3F1EC -Y-"$(awk -v h="$HB" 'BEGIN{print h+1.3}')c"
  gmt plot "$WORK/hex_b.gmt" -C"$WORK/b.cpt" -L -W0.8p,gray25 -t40
  gmt plot "$WORK/pivots.txt" -SE- -W2.2p,white
  gmt plot "$WORK/pivots.txt" -SE- -W1.0p,black
  gmt text "$WORK/lab_b.txt" -F+f8p,Helvetica-Bold,black=1.6p,white+jMC
  gmt text "$WORK/lab_b.txt" -F+f8p,Helvetica-Bold,black+jMC
  gmt basemap -Bxa0.02f0.005 -Bya0.01f0.0025 -BWSne -LjBR+o0.3c/0.35c+c37.53N+w2k+f+lkm \
      -F+gwhite@10+p0.4p,gray40+c0.08c
  echo "(b) Densest cluster: pivots per 1.5 km hexagon" | gmt text -F+cTL+f9.5p,Helvetica-Bold+jTL \
      -Dj0.2c/0.2c -Gwhite@10 -C0.1c/0.07c -W0.4p,gray40
  gmt colorbar -C"$WORK/b.cpt" -DJMR+o0.3c/0c+w${HB}c/0.32c -Bxa1+l"Pivots per 1.5 km hexagon"
gmt end
echo "written: $OUT/fig_inventory_map.pdf / .png"
