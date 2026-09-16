#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build the San Gabriel Mountains operational base grid.

Input
-----
static/dem/SanGabriels_USGS10m.tif

Source CRS:
    EPSG:4269 (NAD83 geographic)

Output
------
static/grid/SanGabriels_DEM_200m_UTM11.tif
static/grid/HeroDirt_grid_200m.npz

Operational grid:
    EPSG:26911
    NAD83 / UTM Zone 11N
    200 m cells

The original ~10 m DEM is NOT modified.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import (
    reproject,
    Resampling,
    transform_bounds,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt_CT"

SRC_DEM = (
    ROOT
    / "static"
    / "dem"
    / "Connecticut_USGS10m.tif"
)

GRID_DIR = ROOT / "static" / "grid"
GRID_DIR.mkdir(parents=True, exist_ok=True)

OUT_DEM = GRID_DIR / "Connecticut_DEM_200m_EPSG26956.tif"

OUT_GRID = GRID_DIR / "HeroDirt_grid_200m.npz"


# ============================================================
# GRID SETTINGS
# ============================================================

TARGET_CRS = CRS.from_epsg(26956)

RES = 200.0  # meters

NODATA = -9999.0


print()
print("============================================")
print(" HERO DIRT 200 m BASE GRID")
print("============================================")
print()


# ============================================================
# OPEN SOURCE DEM
# ============================================================

with rasterio.open(SRC_DEM) as src:

    print("Source DEM:")
    print(f"  {SRC_DEM}")
    print()

    print(f"Source CRS:        {src.crs}")
    print(f"Source dimensions: {src.width} x {src.height}")
    print(f"Source resolution: {src.res}")
    print()

    # --------------------------------------------------------
    # PROJECT SOURCE BOUNDS TO UTM 11N
    # --------------------------------------------------------

    left, bottom, right, top = transform_bounds(
        src.crs,
        TARGET_CRS,
        *src.bounds,
        densify_pts=21,
    )

    print("Raw projected bounds [m]:")
    print(f"  west:  {left:,.1f}")
    print(f"  east:  {right:,.1f}")
    print(f"  south: {bottom:,.1f}")
    print(f"  north: {top:,.1f}")
    print()

    # --------------------------------------------------------
    # ALIGN DOMAIN TO EXACT 200 m GRID
    #
    # Floor west/south outward.
    # Ceil east/north outward.
    #
    # This makes grid coordinates clean integer multiples
    # of 200 m.
    # --------------------------------------------------------

    west = np.floor(left / RES) * RES
    east = np.ceil(right / RES) * RES

    south = np.floor(bottom / RES) * RES
    north = np.ceil(top / RES) * RES

    width = int(round((east - west) / RES))
    height = int(round((north - south) / RES))

    transform = from_origin(
        west,
        north,
        RES,
        RES,
    )

    print("Aligned operational bounds [m]:")
    print(f"  west:  {west:,.1f}")
    print(f"  east:  {east:,.1f}")
    print(f"  south: {south:,.1f}")
    print(f"  north: {north:,.1f}")
    print()

    print("Operational grid:")
    print(f"  CRS:        {TARGET_CRS}")
    print(f"  resolution: {RES:.0f} m")
    print(f"  width:      {width}")
    print(f"  height:     {height}")
    print(
        f"  cells:      "
        f"{width * height:,}"
    )
    print()

    # --------------------------------------------------------
    # CREATE OUTPUT ARRAY
    # --------------------------------------------------------

    dem200 = np.full(
        (height, width),
        NODATA,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # REPROJECT + AGGREGATE
    #
    # Average resampling means elevations from the fine DEM
    # contribute to the 200 m mean elevation.
    # --------------------------------------------------------

    print("Reprojecting and averaging DEM to 200 m...")
    print("This may take a minute.")
    print()

    reproject(
        source=rasterio.band(src, 1),
        destination=dem200,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform,
        dst_crs=TARGET_CRS,
        dst_nodata=NODATA,
        resampling=Resampling.average,
        num_threads=4,
    )


# ============================================================
# VALID DATA
# ============================================================

valid = dem200 != NODATA

n_valid = int(valid.sum())

print("Reprojection complete.")
print()

print(
    f"Valid cells: "
    f"{n_valid:,} / {dem200.size:,} "
    f"({100 * n_valid / dem200.size:.1f}%)"
)

if n_valid > 0:

    print(
        f"Elevation min:  "
        f"{dem200[valid].min():.1f} m"
    )

    print(
        f"Elevation max:  "
        f"{dem200[valid].max():.1f} m"
    )

    print(
        f"Elevation mean: "
        f"{dem200[valid].mean():.1f} m"
    )

print()


# ============================================================
# WRITE GEOTIFF
# ============================================================

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float32",
    "crs": TARGET_CRS,
    "transform": transform,
    "nodata": NODATA,
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}

with rasterio.open(
    OUT_DEM,
    "w",
    **profile,
) as dst:

    dst.write(
        dem200,
        1,
    )

    dst.set_band_description(
        1,
        "Mean elevation [m]"
    )


# ============================================================
# GRID-CENTER COORDINATES
# ============================================================

# UTM x coordinate of cell centers

x = (
    west
    + RES / 2
    + np.arange(width) * RES
)

# UTM y coordinate of cell centers.
#
# Raster row 0 = northernmost row.

y = (
    north
    - RES / 2
    - np.arange(height) * RES
)


# ============================================================
# SAVE GRID METADATA
# ============================================================

np.savez_compressed(
    OUT_GRID,

    x=x.astype(np.float64),
    y=y.astype(np.float64),

    resolution=np.float64(RES),

    west=np.float64(west),
    east=np.float64(east),
    south=np.float64(south),
    north=np.float64(north),

    width=np.int32(width),
    height=np.int32(height),

    epsg=np.int32(26956),
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("Saved DEM:")
print(f"  {OUT_DEM}")
print()

print("Saved grid definition:")
print(f"  {OUT_GRID}")
print()

print("============================================")
print(" HERO DIRT BASE GRID COMPLETE")
print("============================================")
print()
