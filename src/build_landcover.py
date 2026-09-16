#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build NLCD land-cover / impervious fields
on the 200 m San Gabriel operational grid.

This version reads ONLY the source window overlapping
the San Gabriel domain. It does not load the full CONUS
NLCD raster into memory.

Inputs
------
static/nlcd/Annual_NLCD_LndCov_2024_CU_C1V2.tif
static/nlcd/Annual_NLCD_FctImp_2025_CU_C1V2.tif

Outputs
-------
static/landcover/
    landcover_mode_200m.tif
    water_fraction_200m.tif
    developed_fraction_200m.tif
    high_developed_fraction_200m.tif
    barren_fraction_200m.tif
    forest_fraction_200m.tif
    shrub_fraction_200m.tif
    grass_fraction_200m.tif
    impervious_mean_200m.tif
    HeroDirt_landcover_200m.npz
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

NLCD_DIR = (
    ROOT
    / "static"
    / "nlcd"
)

LANDCOVER_FILE = (
    NLCD_DIR
    / "Annual_NLCD_LndCov_2024_CU_C1V2.tif"
)

IMPERVIOUS_FILE = (
    NLCD_DIR
    / "Annual_NLCD_FctImp_2025_CU_C1V2.tif"
)

OUT_DIR = (
    ROOT
    / "static"
    / "landcover"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_NPZ = (
    OUT_DIR
    / "HeroDirt_landcover_200m.npz"
)


# ============================================================
# MASTER GRID
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
NODATA_BYTE = 0


print()
print("============================================")
print(" HERO DIRT NLCD BUILD")
print("============================================")
print()

print("Operational grid:")
print(f"  size:       {width} x {height}")
print(f"  resolution: {res:.0f} m")
print("  CRS:        EPSG:26956")
print()


# ============================================================
# CHECK INPUTS
# ============================================================

for f in [
    LANDCOVER_FILE,
    IMPERVIOUS_FILE,
]:

    if not f.exists():

        raise FileNotFoundError(
            f"Missing input:\n{f}"
        )


# ============================================================
# CLASS GROUPS
# ============================================================

CLASS_GROUPS = {

    "water_fraction": [
        11
    ],

    "developed_fraction": [
        21,
        22,
        23,
        24,
    ],

    "high_developed_fraction": [
        23,
        24,
    ],

    "barren_fraction": [
        31
    ],

    "forest_fraction": [
        41,
        42,
        43,
    ],

    "shrub_fraction": [
        52
    ],

    "grass_fraction": [
        71
    ],
}


# ============================================================
# HELPER:
# READ ONLY SOURCE WINDOW FOR HERO DIRT DOMAIN
# ============================================================

def read_domain_window(
    src,
    pad_pixels=4,
):

    """
    Transform Hero Dirt UTM bounds into the source CRS,
    determine a source raster window, pad slightly,
    and return only that subset.
    """

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

    # Expand by a few source pixels to protect
    # reprojection edges.
    window = window.round_offsets().round_lengths()

    col_off = max(
        0,
        int(window.col_off) - pad_pixels,
    )

    row_off = max(
        0,
        int(window.row_off) - pad_pixels,
    )

    col_end = min(
        src.width,
        int(window.col_off + window.width)
        + pad_pixels,
    )

    row_end = min(
        src.height,
        int(window.row_off + window.height)
        + pad_pixels,
    )

    window = rasterio.windows.Window(
        col_off,
        row_off,
        col_end - col_off,
        row_end - row_off,
    )

    A = src.read(
        1,
        window=window,
        masked=True,
    )

    transform = src.window_transform(
        window
    )

    return A, transform, window


# ============================================================
# LAND COVER
# ============================================================

print("Reading NLCD land cover:")
print(f"  {LANDCOVER_FILE.name}")


with rasterio.open(
    LANDCOVER_FILE
) as src:

    print(f"  CRS:        {src.crs}")
    print(f"  resolution: {src.res}")
    print(f"  full size:  {src.width} x {src.height}")
    print(f"  nodata:     {src.nodata}")

    lc, lc_transform, lc_window = (
        read_domain_window(
            src
        )
    )

    print(
        f"  subset size: "
        f"{lc.shape[1]} x {lc.shape[0]}"
    )

    print(
        f"  subset cells: "
        f"{lc.size:,}"
    )

    print()

    # --------------------------------------------------------
    # Dominant/modal class on 200 m grid
    # --------------------------------------------------------

    landcover_mode = np.zeros(
        shape200,
        dtype=np.uint8,
    )

    reproject(
        source=lc.filled(
            int(src.nodata)
            if src.nodata is not None
            else 0
        ),
        destination=landcover_mode,

        src_transform=lc_transform,
        src_crs=src.crs,
        src_nodata=src.nodata,

        dst_transform=transform200,
        dst_crs=TARGET_CRS,
        dst_nodata=NODATA_BYTE,

        resampling=Resampling.mode,

        num_threads=4,
    )


    # --------------------------------------------------------
    # Fractional class fields
    # --------------------------------------------------------

    lc_data = lc.filled(
        0
    )

    valid_source = (
        ~np.ma.getmaskarray(
            lc
        )
    )

    fraction_grids = {}


    for name, classes in (
        CLASS_GROUPS.items()
    ):

        print(
            f"Building {name}..."
        )

        binary = np.full(
            lc_data.shape,
            NODATA_FLOAT,
            dtype=np.float32,
        )

        binary[
            valid_source
        ] = 0.0

        binary[
            valid_source
            &
            np.isin(
                lc_data,
                classes,
            )
        ] = 1.0


        A = np.full(
            shape200,
            NODATA_FLOAT,
            dtype=np.float32,
        )


        reproject(
            source=binary,
            destination=A,

            src_transform=lc_transform,
            src_crs=src.crs,
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
            1.0,
        )


        fraction_grids[
            name
        ] = A


# ============================================================
# IMPERVIOUS SURFACE
# ============================================================

print()
print("Reading fractional impervious surface:")
print(f"  {IMPERVIOUS_FILE.name}")


with rasterio.open(
    IMPERVIOUS_FILE
) as src:

    print(f"  CRS:        {src.crs}")
    print(f"  resolution: {src.res}")
    print(f"  full size:  {src.width} x {src.height}")
    print(f"  nodata:     {src.nodata}")

    imp, imp_transform, imp_window = (
        read_domain_window(
            src
        )
    )

    print(
        f"  subset size: "
        f"{imp.shape[1]} x {imp.shape[0]}"
    )

    print(
        f"  subset cells: "
        f"{imp.size:,}"
    )

    print()


    imp_data = imp.astype(
        np.float32
    ).filled(
        NODATA_FLOAT
    )


    impervious = np.full(
        shape200,
        NODATA_FLOAT,
        dtype=np.float32,
    )


    reproject(
        source=imp_data,
        destination=impervious,

        src_transform=imp_transform,
        src_crs=src.crs,
        src_nodata=NODATA_FLOAT,

        dst_transform=transform200,
        dst_crs=TARGET_CRS,
        dst_nodata=NODATA_FLOAT,

        resampling=Resampling.average,

        num_threads=4,
    )


impervious[
    impervious == NODATA_FLOAT
] = np.nan


impervious[
    (impervious < 0)
    |
    (impervious > 100)
] = np.nan


# ============================================================
# OUTPUT PROFILES
# ============================================================

profile_float = {
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


profile_byte = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "crs": TARGET_CRS,
    "transform": transform200,
    "nodata": NODATA_BYTE,
    "compress": "deflate",
    "predictor": 2,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}


# ============================================================
# WRITER
# ============================================================

def write_float(
    filename,
    A,
    description,
):

    path = (
        OUT_DIR
        /
        filename
    )

    out = np.where(
        np.isfinite(A),
        A,
        NODATA_FLOAT,
    ).astype(
        np.float32
    )

    with rasterio.open(
        path,
        "w",
        **profile_float,
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
# WRITE MODAL LAND COVER
# ============================================================

mode_file = (
    OUT_DIR
    /
    "landcover_mode_200m.tif"
)


with rasterio.open(
    mode_file,
    "w",
    **profile_byte,
) as dst:

    dst.write(
        landcover_mode,
        1,
    )

    dst.set_band_description(
        1,
        "Modal NLCD land-cover class",
    )


print(
    "Saved: landcover_mode_200m.tif"
)


# ============================================================
# WRITE FRACTIONS
# ============================================================

for name, A in (
    fraction_grids.items()
):

    write_float(
        f"{name}_200m.tif",
        A,
        f"Fraction of 200 m cell: {name}",
    )


write_float(
    "impervious_mean_200m.tif",
    impervious,
    "Mean fractional impervious surface [%]",
)


# ============================================================
# SAVE NUMPY STACK
# ============================================================

np.savez_compressed(

    OUT_NPZ,

    landcover_mode=landcover_mode,

    water_fraction=(
        fraction_grids[
            "water_fraction"
        ]
    ),

    developed_fraction=(
        fraction_grids[
            "developed_fraction"
        ]
    ),

    high_developed_fraction=(
        fraction_grids[
            "high_developed_fraction"
        ]
    ),

    barren_fraction=(
        fraction_grids[
            "barren_fraction"
        ]
    ),

    forest_fraction=(
        fraction_grids[
            "forest_fraction"
        ]
    ),

    shrub_fraction=(
        fraction_grids[
            "shrub_fraction"
        ]
    ),

    grass_fraction=(
        fraction_grids[
            "grass_fraction"
        ]
    ),

    impervious_mean=impervious,

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
print(" NLCD SUMMARY")
print("============================================")
print()


for name, A in (
    fraction_grids.items()
):

    good = np.isfinite(
        A
    )

    if good.any():

        print(
            f"{name:26s}: "
            f"mean={np.nanmean(A):.3f}, "
            f"P95={np.nanpercentile(A,95):.3f}, "
            f"max={np.nanmax(A):.3f}"
        )


good_imp = np.isfinite(
    impervious
)


if good_imp.any():

    print()

    print(
        f"{'impervious_mean':26s}: "
        f"mean={np.nanmean(impervious):.2f}%, "
        f"P95={np.nanpercentile(impervious,95):.2f}%, "
        f"max={np.nanmax(impervious):.2f}%"
    )


print()
print(
    "Saved land-cover stack:"
)

print(
    f"  {OUT_NPZ}"
)

print()
print("============================================")
print(" HERO DIRT NLCD BUILD COMPLETE")
print("============================================")
print()
