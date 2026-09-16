#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build LANDFIRE Existing Vegetation Cover fields
on the 200 m San Gabriel operational grid.

Input
-----
static/landfire/LF2024_EVC_CONUS.tif

Outputs
-------
static/vegetation/
    tree_cover_mean_200m.tif
    shrub_cover_mean_200m.tif
    herb_cover_mean_200m.tif
    vegetation_cover_max_200m.tif
    HeroDirt_vegetation_200m.npz

LANDFIRE EVC coding
-------------------
110-199 : Tree cover
210-299 : Shrub cover
310-399 : Herbaceous cover

For those vegetation classes:
    percent cover = VALUE modulo 100

Examples:
    145 -> 45% tree cover
    260 -> 60% shrub cover
    325 -> 25% herbaceous cover

Other special EVC classes are treated as zero cover for the
corresponding vegetation life-form rather than interpreted
as percentages.

The source raster is read only for the San Gabriel domain.
"""

from pathlib import Path

import numpy as np
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from rasterio.warp import (
    reproject,
    Resampling,
    transform_bounds,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt_CT"

GRID_FILE = (
    ROOT
    / "static"
    / "grid"
    / "HeroDirt_grid_200m.npz"
)

EVC_FILE = (
    ROOT
    / "static"
    / "landfire"
    / "LF2024_EVC_CONUS.tif"
)

OUT_DIR = (
    ROOT
    / "static"
    / "vegetation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_NPZ = (
    OUT_DIR
    / "HeroDirt_vegetation_200m.npz"
)


# ============================================================
# GRID
# ============================================================

G = np.load(
    GRID_FILE
)

west = float(G["west"])
east = float(G["east"])
south = float(G["south"])
north = float(G["north"])

width = int(G["width"])
height = int(G["height"])

x = G["x"]
y = G["y"]

res = float(G["resolution"])

TARGET_CRS = CRS.from_epsg(26956)

transform200 = from_origin(
    west,
    north,
    res,
    res,
)

shape200 = (
    height,
    width,
)

NODATA_FLOAT = -9999.0


print()
print("============================================")
print(" HERO DIRT LANDFIRE EVC BUILD")
print("============================================")
print()

print("Operational grid:")
print(f"  size:       {width} x {height}")
print(f"  resolution: {res:.0f} m")
print("  CRS:        EPSG:26956")
print()


# ============================================================
# CHECK INPUT
# ============================================================

if not EVC_FILE.exists():

    raise FileNotFoundError(
        f"Missing LANDFIRE EVC:\n{EVC_FILE}"
    )


# ============================================================
# READ ONLY SAN GABRIEL WINDOW
# ============================================================

with rasterio.open(
    EVC_FILE
) as src:

    print("LANDFIRE EVC:")
    print(f"  file:       {EVC_FILE.name}")
    print(f"  CRS:        {src.crs}")
    print(f"  resolution: {src.res}")
    print(f"  full size:  {src.width} x {src.height}")
    print(f"  nodata:     {src.nodata}")
    print()

    # Convert Hero Dirt bounds into source CRS.
    src_left, src_bottom, src_right, src_top = (
        transform_bounds(
            TARGET_CRS,
            src.crs,
            west,
            south,
            east,
            north,
            densify_pts=21,
        )
    )

    window = from_bounds(
        src_left,
        src_bottom,
        src_right,
        src_top,
        transform=src.transform,
    )

    window = (
        window
        .round_offsets()
        .round_lengths()
    )

    pad = 4

    col_off = max(
        0,
        int(window.col_off) - pad,
    )

    row_off = max(
        0,
        int(window.row_off) - pad,
    )

    col_end = min(
        src.width,
        int(window.col_off + window.width)
        + pad,
    )

    row_end = min(
        src.height,
        int(window.row_off + window.height)
        + pad,
    )

    window = rasterio.windows.Window(
        col_off,
        row_off,
        col_end - col_off,
        row_end - row_off,
    )

    evc = src.read(
        1,
        window=window,
        masked=True,
    )

    evc_transform = (
        src.window_transform(
            window
        )
    )

    source_crs = src.crs

    print(
        f"Subset size: "
        f"{evc.shape[1]} x {evc.shape[0]}"
    )

    print(
        f"Subset cells: "
        f"{evc.size:,}"
    )

    print()


# ============================================================
# DECODE EVC
# ============================================================

data = evc.filled(
    0
).astype(
    np.int16
)

valid = (
    ~np.ma.getmaskarray(
        evc
    )
)


tree_source = np.zeros(
    data.shape,
    dtype=np.float32,
)

shrub_source = np.zeros(
    data.shape,
    dtype=np.float32,
)

herb_source = np.zeros(
    data.shape,
    dtype=np.float32,
)


# Tree: 110-199
tree_mask = (
    valid
    &
    (data >= 110)
    &
    (data <= 199)
)

tree_source[
    tree_mask
] = (
    data[
        tree_mask
    ]
    %
    100
)


# Shrub: 210-299
shrub_mask = (
    valid
    &
    (data >= 210)
    &
    (data <= 299)
)

shrub_source[
    shrub_mask
] = (
    data[
        shrub_mask
    ]
    %
    100
)


# Herbaceous: 310-399
herb_mask = (
    valid
    &
    (data >= 310)
    &
    (data <= 399)
)

herb_source[
    herb_mask
] = (
    data[
        herb_mask
    ]
    %
    100
)


# Mark invalid source pixels as nodata.
for A in [
    tree_source,
    shrub_source,
    herb_source,
]:

    A[
        ~valid
    ] = NODATA_FLOAT


# ============================================================
# REPROJECT / AVERAGE TO 200 m
# ============================================================

def aggregate_cover(
    source,
):

    A = np.full(
        shape200,
        NODATA_FLOAT,
        dtype=np.float32,
    )

    reproject(
        source=source,
        destination=A,

        src_transform=evc_transform,
        src_crs=source_crs,
        src_nodata=NODATA_FLOAT,

        dst_transform=transform200,
        dst_crs=TARGET_CRS,
        dst_nodata=NODATA_FLOAT,

        resampling=Resampling.average,

        num_threads=4,
    )

    A[
        A == NODATA_FLOAT
    ] = np.nan

    A = np.clip(
        A,
        0.0,
        100.0,
    )

    return A


print(
    "Aggregating tree cover..."
)

tree = aggregate_cover(
    tree_source
)


print(
    "Aggregating shrub cover..."
)

shrub = aggregate_cover(
    shrub_source
)


print(
    "Aggregating herbaceous cover..."
)

herb = aggregate_cover(
    herb_source
)


# ============================================================
# MAXIMUM LIFEFORM COVER
#
# Useful diagnostic only.
# ============================================================

stack = np.stack(
    [
        tree,
        shrub,
        herb,
    ],
    axis=0,
)

all_nan = np.all(
    ~np.isfinite(stack),
    axis=0,
)

stack_safe = np.where(
    np.isfinite(stack),
    stack,
    -np.inf,
)

veg_max = np.max(
    stack_safe,
    axis=0,
).astype(
    np.float32
)

veg_max[
    all_nan
] = np.nan


# ============================================================
# OUTPUT PROFILE
# ============================================================

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float32",
    "crs": TARGET_CRS,
    "transform": transform200,
    "nodata": NODATA_FLOAT,
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}


def write_tif(
    filename,
    A,
    description,
):

    out = np.where(
        np.isfinite(A),
        A,
        NODATA_FLOAT,
    ).astype(
        np.float32
    )

    path = (
        OUT_DIR
        /
        filename
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            out,
            1,
        )

        dst.set_band_description(
            1,
            description,
        )

    print(
        f"Saved: {filename}"
    )


# ============================================================
# WRITE OUTPUTS
# ============================================================

print()

write_tif(
    "tree_cover_mean_200m.tif",
    tree,
    "Mean LANDFIRE tree cover [%]",
)

write_tif(
    "shrub_cover_mean_200m.tif",
    shrub,
    "Mean LANDFIRE shrub cover [%]",
)

write_tif(
    "herb_cover_mean_200m.tif",
    herb,
    "Mean LANDFIRE herbaceous cover [%]",
)

write_tif(
    "vegetation_cover_max_200m.tif",
    veg_max,
    "Maximum mean lifeform cover [%]",
)


# ============================================================
# SAVE NPZ
# ============================================================

np.savez_compressed(

    OUT_NPZ,

    tree_cover=tree,
    shrub_cover=shrub,
    herb_cover=herb,

    vegetation_cover_max=veg_max,

    x=x,
    y=y,

    resolution=np.float64(
        res
    ),

    epsg=np.int32(
        26956
    ),
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("============================================")
print(" LANDFIRE EVC SUMMARY")
print("============================================")
print()


for name, A in [
    ("tree_cover", tree),
    ("shrub_cover", shrub),
    ("herb_cover", herb),
]:

    good = np.isfinite(
        A
    )

    values = A[
        good
    ]

    if values.size:

        print(
            f"{name:18s}: "
            f"mean={np.mean(values):5.2f}%, "
            f"P50={np.percentile(values,50):5.2f}%, "
            f"P95={np.percentile(values,95):5.2f}%, "
            f"max={np.max(values):5.2f}%"
        )


print()

print(
    "Source EVC lifeform pixels:"
)

print(
    f"  tree:  "
    f"{tree_mask.sum():,}"
)

print(
    f"  shrub: "
    f"{shrub_mask.sum():,}"
)

print(
    f"  herb:  "
    f"{herb_mask.sum():,}"
)


print()
print(
    "Saved vegetation stack:"
)

print(
    f"  {OUT_NPZ}"
)

print()
print("============================================")
print(" HERO DIRT LANDFIRE BUILD COMPLETE")
print("============================================")
print()
