# Hough-Space Priors for Circular Field Delineation in Multi-Season Sentinel-2 Composites

*In a landscape of rectangles, every square conceals a circle.*

Three Cloud Score+ seasonal medians fold spectral and phenological evidence into one 24-band tensor. A gradient-space circular Hough transform proposes discs; a rim-contrast test across seasons decides which survive; a human eye settles the rest. What remains is written as semantic and instance rasters on a spatially blocked chip grid, ready for convolutional, transformer and region-based segmentation, and counted on hexagons.

**Keywords:** Sentinel-2 MSI · Google Earth Engine · Cloud Score+ · multi-temporal compositing · NDVI phenology · circular Hough transform · Canny edge detection · circularity and rim-contrast screening · weak supervision · human-in-the-loop labelling · instance segmentation · spatial-block cross-validation · Hungarian matching · object-level F1 · hexagonal binning · GMT

```
gee/          stack/        grid/        annotation/
loader/       phenology/    figures/     s2_common.py
```

## Setup

```bash
pip install -r requirements.txt
cp site_config.example.py site_config.py
export PYTHONPATH=$PWD
```

Scene geometry lives in `site_config.py`, in the first lines of `gee/seasonal_stack_export.js` and in the environment of `figures/fig_inventory_map.sh` (`R`, `RZ`, `AOI`, `EPSG`). GMT ≥ 6.5 for the maps.

## Licence

MIT © 2026 Polina Lemenkova
