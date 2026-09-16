#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build terrain statistics on the 200 m operational grid.

Workflow
--------
1. Reproject original geographic DEM to a 10 m UTM Zone 11N grid.
2. Compute slope and terrain orientation at 10 m.
3. Aggregate 10 m terrain metrics to the exact 200 m Hero Dirt grid.

Outputs
-------
static/terrain/SanGabriels_DEM_10m_UTM11.tif

static/terrain/
    elevation_mean_200m.tif
    elevation_std_200m.tif
    slope_mean_200m.tif
    slope_std_200m.tif
    northness_mean_200m.tif
    southness_mean_200m.tif
    eastness_mean_200m.tif
    terrain_valid_fraction_200m.tif

static/terrain/HeroDirt_terrain_200m.npz

Definitions
-----------
Slope:
    local surface slope from 10 m DEM gradients.

Orientation is calculated from the DOWNHILL direction.

northness:
    +1 = downhill toward north
    -1 = downhill toward south

southness:
    +1 = downhill toward south
    -1 = downhill toward north

eastness:
    +1 = downhill toward east
    -1 = downhill toward west

Flat terrain:
    orientation components are assigned 0.

Important
---------
Slope is NOT calculated from the 200 m mean DEM.
It is calculated at 10 m and then averaged to 200 m.
"""

from pathlib import Path

import numpy as np
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt_CT"

SOURCE_DEM = (
    ROOT
    / "static"
    / "dem"
    / "Connecticut_USGS10m.tif"
)

GRID_FILE = (
    ROOT
    / "static"
    / "grid"
    / "HeroDirt_grid_200m.npz"
)

TERRAIN_DIR = ROOT / "static" / "terrain"
TERRAIN_DIR.mkdir(parents=True, exist_ok=True)

DEM10_FILE = (
    TERRAIN_DIR
    / "Connecticut_DEM_10m_EPSG26956.tif"
)

TERRAIN_NPZ = (
    TERRAIN_DIR
    / "HeroDirt_terrain_200m.npz"
)


# ============================================================
# CONSTANTS
# ============================================================

TARGET_CRS = CRS.from_epsg(26956)

RES_FINE = 10.0
RES_COARSE = 200.0

FACTOR = int(
    round(
        RES_COARSE / RES_FINE
    )
)

NODATA_FINE = -9999.0
NODATA_OUT = -9999.0


print()
print("============================================")
print(" HERO DIRT TERRAIN STACK")
print("============================================")
print()


# ============================================================
# LOAD MASTER GRID
# ============================================================

G = np.load(GRID_FILE)


west = float(G["west"])
east = float(G["east"])
south = float(G["south"])
north = float(G["north"])

width200 = int(G["width"])
height200 = int(G["height"])

x200 = G["x"]
y200 = G["y"]


print("Operational grid:")
print(f"  width:      {width200}")
print(f"  height:     {height200}")
print(f"  resolution: {RES_COARSE:.0f} m")
print()


# ============================================================
# BUILD EXACT 10 m GRID
# ============================================================

width10 = width200 * FACTOR
height10 = height200 * FACTOR


transform10 = from_origin(
    west,
    north,
    RES_FINE,
    RES_FINE,
)


print("Fine terrain grid:")
print(f"  resolution: {RES_FINE:.0f} m")
print(f"  width:      {width10}")
print(f"  height:     {height10}")
print(
    f"  cells:      "
    f"{width10 * height10:,}"
)
print()


# ============================================================
# REPROJECT ORIGINAL DEM TO UTM 10 m
# ============================================================

if not DEM10_FILE.exists():

    print("Creating 10 m UTM DEM...")
    print(
        "This is the largest static terrain step "
        "and may take a few minutes."
    )
    print()

    profile10 = {
        "driver": "GTiff",
        "height": height10,
        "width": width10,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform10,
        "nodata": NODATA_FINE,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        SOURCE_DEM
    ) as src:

        with rasterio.open(
            DEM10_FILE,
            "w",
            **profile10,
        ) as dst:

            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform10,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA_FINE,

                # Bilinear is appropriate for continuous elevation.
                resampling=Resampling.bilinear,

                num_threads=4,
            )

            dst.set_band_description(
                1,
                "Elevation [m]"
            )

    print("10 m UTM DEM complete.")
    print()

else:

    print("Using existing 10 m UTM DEM:")
    print(f"  {DEM10_FILE}")
    print()


# ============================================================
# OUTPUT ARRAYS
# ============================================================

shape200 = (
    height200,
    width200,
)


elev_mean = np.full(
    shape200,
    np.nan,
    dtype=np.float32,
)

elev_std = np.full(
    shape200,
    np.nan,
    dtype=np.float32,
)

slope_mean = np.full(
    shape200,
    np.nan,
    dtype=np.float32,
)

slope_std = np.full(
    shape200,
    np.nan,
    dtype=np.float32,
)

northness_mean = np.full(
    shape200,
    np.nan,
    dtype=np.float32,
)

southness_mean = np.full(
    shape200,
    np.nan,
    dtype=np.float32,
)

eastness_mean = np.full(
    shape200,
    np.nan,
    dtype=np.float32,
)

valid_fraction = np.zeros(
    shape200,
    dtype=np.float32,
)


# ============================================================
# HELPERS
# ============================================================

def aggregate_block_mean(A, valid):
    """
    Aggregate a fine stripe of shape
        (20, width10)
    to
        (width200,)
    """

    B = A.reshape(
        FACTOR,
        width200,
        FACTOR,
    )

    V = valid.reshape(
        FACTOR,
        width200,
        FACTOR,
    )

    numerator = np.sum(
        np.where(
            V,
            B,
            0.0,
        ),
        axis=(0, 2),
    )

    denominator = np.sum(
        V,
        axis=(0, 2),
    )

    result = np.full(
        width200,
        np.nan,
        dtype=np.float64,
    )

    good = denominator > 0

    result[good] = (
        numerator[good]
        / denominator[good]
    )

    return result


def aggregate_block_std(A, valid):
    """
    Nan-aware standard deviation over each
    20 x 20 operational cell.
    """

    B = A.reshape(
        FACTOR,
        width200,
        FACTOR,
    )

    V = valid.reshape(
        FACTOR,
        width200,
        FACTOR,
    )

    count = np.sum(
        V,
        axis=(0, 2),
    )

    total = np.sum(
        np.where(
            V,
            B,
            0.0,
        ),
        axis=(0, 2),
    )

    mean = np.zeros(
        width200,
        dtype=np.float64,
    )

    good = count > 0

    mean[good] = (
        total[good]
        / count[good]
    )

    mean2d = mean[
        None,
        :,
        None,
    ]

    var_num = np.sum(
        np.where(
            V,
            (B - mean2d) ** 2,
            0.0,
        ),
        axis=(0, 2),
    )

    result = np.full(
        width200,
        np.nan,
        dtype=np.float64,
    )

    result[good] = np.sqrt(
        var_num[good]
        / count[good]
    )

    return result


# ============================================================
# PROCESS 200 m STRIPES
#
# Each operational row contains exactly 20 fine DEM rows.
#
# Read one extra fine row above and below plus one extra
# fine column on either side.
#
# This lets us calculate central differences without creating
# several full-domain 10 m arrays in memory.
# ============================================================

print("Calculating 10 m terrain and aggregating to 200 m...")
print()


with rasterio.open(
    DEM10_FILE
) as src:

    for j200 in range(height200):

        r0 = j200 * FACTOR

        # Read:
        #
        #   rows r0-1 through r0+20
        #   columns -1 through width10
        #
        # boundless=True fills outside-raster cells with nodata.

        window = Window(
            -1,
            r0 - 1,
            width10 + 2,
            FACTOR + 2,
        )

        zhalo = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=NODATA_FINE,
        ).astype(
            np.float64
        )

        zhalo[
            zhalo == NODATA_FINE
        ] = np.nan


        # ----------------------------------------------------
        # CENTRAL 20 ROWS / REAL DOMAIN COLUMNS
        # ----------------------------------------------------

        z = zhalo[
            1:FACTOR + 1,
            1:width10 + 1,
        ]


        # ----------------------------------------------------
        # CENTRAL-DIFFERENCE GRADIENTS
        #
        # x positive east.
        #
        # Raster rows increase southward.
        # Geographic UTM y increases northward.
        #
        # Therefore northward dz/dy is negative of the
        # row-direction central difference.
        # ----------------------------------------------------

        dzdx = (
            zhalo[
                1:FACTOR + 1,
                2:width10 + 2,
            ]
            -
            zhalo[
                1:FACTOR + 1,
                0:width10,
            ]
        ) / (
            2.0 * RES_FINE
        )


        dzdy = -(
            zhalo[
                2:FACTOR + 2,
                1:width10 + 1,
            ]
            -
            zhalo[
                0:FACTOR,
                1:width10 + 1,
            ]
        ) / (
            2.0 * RES_FINE
        )


        # ----------------------------------------------------
        # VALID TERRAIN
        # ----------------------------------------------------

        valid_z = np.isfinite(z)

        valid_slope = (
            valid_z
            & np.isfinite(dzdx)
            & np.isfinite(dzdy)
        )


        # ----------------------------------------------------
        # SLOPE
        #
        # rise/run magnitude
        # ----------------------------------------------------

        gradmag = np.sqrt(
            dzdx ** 2
            +
            dzdy ** 2
        )


        slope_deg = np.degrees(
            np.arctan(
                gradmag
            )
        )


        # ----------------------------------------------------
        # TERRAIN ORIENTATION
        #
        # Gradient points uphill:
        #
        #   (dz/dx, dz/dy)
        #
        # Downhill vector therefore:
        #
        #   (-dz/dx, -dz/dy)
        #
        # Normalize to get direction components.
        # ----------------------------------------------------

        eastness = np.zeros_like(
            gradmag
        )

        northness = np.zeros_like(
            gradmag
        )


        nonflat = (
            valid_slope
            &
            (gradmag > 1.0e-12)
        )


        eastness[nonflat] = (
            -dzdx[nonflat]
            /
            gradmag[nonflat]
        )


        northness[nonflat] = (
            -dzdy[nonflat]
            /
            gradmag[nonflat]
        )


        southness = (
            -northness
        )


        eastness[
            ~valid_slope
        ] = np.nan

        northness[
            ~valid_slope
        ] = np.nan

        southness[
            ~valid_slope
        ] = np.nan

        slope_deg[
            ~valid_slope
        ] = np.nan


        # ----------------------------------------------------
        # AGGREGATE ELEVATION
        # ----------------------------------------------------

        elev_mean[j200, :] = (
            aggregate_block_mean(
                z,
                valid_z,
            )
        )


        elev_std[j200, :] = (
            aggregate_block_std(
                z,
                valid_z,
            )
        )


        # ----------------------------------------------------
        # AGGREGATE SLOPE
        # ----------------------------------------------------

        slope_mean[j200, :] = (
            aggregate_block_mean(
                slope_deg,
                valid_slope,
            )
        )


        slope_std[j200, :] = (
            aggregate_block_std(
                slope_deg,
                valid_slope,
            )
        )


        # ----------------------------------------------------
        # AGGREGATE TERRAIN DIRECTION
        # ----------------------------------------------------

        northness_mean[j200, :] = (
            aggregate_block_mean(
                northness,
                valid_slope,
            )
        )


        southness_mean[j200, :] = (
            aggregate_block_mean(
                southness,
                valid_slope,
            )
        )


        eastness_mean[j200, :] = (
            aggregate_block_mean(
                eastness,
                valid_slope,
            )
        )


        # ----------------------------------------------------
        # VALID FRACTION
        # ----------------------------------------------------

        V = valid_z.reshape(
            FACTOR,
            width200,
            FACTOR,
        )


        valid_fraction[j200, :] = (
            np.mean(
                V,
                axis=(0, 2),
            )
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            j200 % 20 == 0
            or
            j200 == height200 - 1
        ):

            print(
                f"  row "
                f"{j200 + 1:3d}"
                f" / "
                f"{height200}"
            )


print()
print("Terrain calculation complete.")
print()


# ============================================================
# REMOVE CELLS WITH LITTLE VALID FINE-SCALE COVERAGE
# ============================================================

MIN_VALID_FRACTION = 0.50

bad = (
    valid_fraction
    <
    MIN_VALID_FRACTION
)


for A in [
    elev_mean,
    elev_std,
    slope_mean,
    slope_std,
    northness_mean,
    southness_mean,
    eastness_mean,
]:

    A[bad] = np.nan


# ============================================================
# MASTER 200 m GEOTIFF PROFILE
# ============================================================

transform200 = from_origin(
    west,
    north,
    RES_COARSE,
    RES_COARSE,
)


profile200 = {
    "driver": "GTiff",
    "height": height200,
    "width": width200,
    "count": 1,
    "dtype": "float32",
    "crs": TARGET_CRS,
    "transform": transform200,
    "nodata": NODATA_OUT,
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}


# ============================================================
# WRITE FUNCTION
# ============================================================

def write_raster(
    filename,
    A,
    description,
):

    outfile = (
        TERRAIN_DIR
        /
        filename
    )

    out = np.where(
        np.isfinite(A),
        A,
        NODATA_OUT,
    ).astype(
        np.float32
    )

    with rasterio.open(
        outfile,
        "w",
        **profile200,
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
        f"Saved: {outfile.name}"
    )


# ============================================================
# WRITE TERRAIN GEOTIFFS
# ============================================================

print("Writing 200 m terrain rasters...")
print()


write_raster(
    "elevation_mean_200m.tif",
    elev_mean,
    "Mean 10 m elevation [m]",
)


write_raster(
    "elevation_std_200m.tif",
    elev_std,
    "Subgrid elevation standard deviation [m]",
)


write_raster(
    "slope_mean_200m.tif",
    slope_mean,
    "Mean 10 m slope [degrees]",
)


write_raster(
    "slope_std_200m.tif",
    slope_std,
    "Subgrid slope standard deviation [degrees]",
)


write_raster(
    "northness_mean_200m.tif",
    northness_mean,
    "Mean downhill northness [-1 to 1]",
)


write_raster(
    "southness_mean_200m.tif",
    southness_mean,
    "Mean downhill southness [-1 to 1]",
)


write_raster(
    "eastness_mean_200m.tif",
    eastness_mean,
    "Mean downhill eastness [-1 to 1]",
)


write_raster(
    "terrain_valid_fraction_200m.tif",
    valid_fraction,
    "Fraction of valid 10 m DEM cells",
)


# ============================================================
# SAVE COMPACT NUMPY TERRAIN STACK
# ============================================================

np.savez_compressed(
    TERRAIN_NPZ,

    elevation_mean=elev_mean,
    elevation_std=elev_std,

    slope_mean=slope_mean,
    slope_std=slope_std,

    northness_mean=northness_mean,
    southness_mean=southness_mean,
    eastness_mean=eastness_mean,

    valid_fraction=valid_fraction,

    x=x200,
    y=y200,

    resolution=np.float64(
        RES_COARSE
    ),

    epsg=np.int32(
        26956
    ),
)


# ============================================================
# SUMMARY
# ============================================================

valid200 = (
    np.isfinite(
        elev_mean
    )
)


print()
print("============================================")
print(" TERRAIN SUMMARY")
print("============================================")
print()


print(
    f"Valid operational cells: "
    f"{valid200.sum():,}"
)


print(
    f"Mean elevation: "
    f"{np.nanmean(elev_mean):.1f} m"
)


print(
    f"Elevation range: "
    f"{np.nanmin(elev_mean):.1f} "
    f"to "
    f"{np.nanmax(elev_mean):.1f} m"
)


print(
    f"Mean slope: "
    f"{np.nanmean(slope_mean):.1f} deg"
)


print(
    f"Maximum mean 200 m slope: "
    f"{np.nanmax(slope_mean):.1f} deg"
)


print(
    f"Mean subgrid elevation std: "
    f"{np.nanmean(elev_std):.1f} m"
)


print()
print("Saved terrain stack:")
print(f"  {TERRAIN_NPZ}")


print()
print("============================================")
print(" HERO DIRT TERRAIN STACK COMPLETE")
print("============================================")
print()
