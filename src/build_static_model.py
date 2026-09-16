#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build the combined static model dataset.

Combines:
    terrain
    soils
    NLCD land cover
    LANDFIRE vegetation

Creates two masks:

physics_valid
    Terrain and required soil properties are available.

public_valid
    physics_valid plus simple land-surface exclusions
    for water and strongly urbanized cells.

Important
---------
The public mask thresholds are Hero Dirt product choices,
not established physical constants.
"""

from pathlib import Path

import numpy as np
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin


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

TERRAIN_FILE = (
    ROOT
    / "static"
    / "terrain"
    / "HeroDirt_terrain_200m.npz"
)

SOIL_FILE = (
    ROOT
    / "static"
    / "soils"
    / "HeroDirt_soils_200m.npz"
)

LANDCOVER_FILE = (
    ROOT
    / "static"
    / "landcover"
    / "HeroDirt_landcover_200m.npz"
)

VEGETATION_FILE = (
    ROOT
    / "static"
    / "vegetation"
    / "HeroDirt_vegetation_200m.npz"
)

OUT_DIR = (
    ROOT
    / "static"
    / "model"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_FILE = (
    OUT_DIR
    / "HeroDirt_static_200m.npz"
)


# ============================================================
# LOAD
# ============================================================

G = np.load(GRID_FILE)
T = np.load(TERRAIN_FILE)
S = np.load(SOIL_FILE)
L = np.load(LANDCOVER_FILE)
V = np.load(VEGETATION_FILE)


west = float(G["west"])
north = float(G["north"])

width = int(G["width"])
height = int(G["height"])

res = float(G["resolution"])

x = G["x"]
y = G["y"]

TARGET_CRS = CRS.from_epsg(26956)

transform = from_origin(
    west,
    north,
    res,
    res,
)


print()
print("============================================")
print(" HERO DIRT STATIC MODEL BUILD")
print("============================================")
print()


# ============================================================
# TERRAIN
# ============================================================

elevation = T["elevation_mean"]
elevation_std = T["elevation_std"]

slope = T["slope_mean"]
slope_std = T["slope_std"]

northness = T["northness_mean"]
southness = T["southness_mean"]
eastness = T["eastness_mean"]


# ============================================================
# SOILS
# ============================================================

sand = S["sand"]
silt = S["silt"]
clay = S["clay"]

ksat = S["ksat"]

bulk_density = S["bulk_density"]

theta_wp = S["theta_wp"]
theta_fc = S["theta_fc"]
theta_sat = S["theta_sat"]


# ============================================================
# LAND COVER
# ============================================================

landcover_mode = L["landcover_mode"]

water_fraction = L["water_fraction"]

developed_fraction = L["developed_fraction"]

high_developed_fraction = (
    L["high_developed_fraction"]
)

impervious = L["impervious_mean"]

forest_fraction = L["forest_fraction"]
shrub_fraction = L["shrub_fraction"]
grass_fraction = L["grass_fraction"]


# ============================================================
# VEGETATION COVER
# ============================================================

tree_cover = V["tree_cover"]
shrub_cover = V["shrub_cover"]
herb_cover = V["herb_cover"]


# ============================================================
# PHYSICS MASK
# ============================================================

physics_valid = (
    np.isfinite(elevation)
    &
    np.isfinite(slope)
    &
    np.isfinite(sand)
    &
    np.isfinite(silt)
    &
    np.isfinite(clay)
    &
    np.isfinite(ksat)
    &
    (ksat > 0)
    &
    np.isfinite(theta_wp)
    &
    np.isfinite(theta_fc)
    &
    np.isfinite(theta_sat)
    &
    (theta_wp < theta_fc)
    &
    (theta_fc < theta_sat)
)


# ============================================================
# PUBLIC PRODUCT MASK
#
# These are empirical product thresholds.
# We retain the raw variables so these can be changed later.
# ============================================================

mostly_water = (
    np.isfinite(water_fraction)
    &
    (water_fraction >= 0.50)
)

strongly_impervious = (
    np.isfinite(impervious)
    &
    (impervious >= 50.0)
)

strongly_developed = (
    np.isfinite(high_developed_fraction)
    &
    (high_developed_fraction >= 0.50)
)


public_valid = (
    physics_valid
    &
    (~mostly_water)
    &
    (~strongly_impervious)
    &
    (~strongly_developed)
)


# ============================================================
# VEGETATION FRACTIONS
#
# Convert percentage cover to 0-1 for later model use.
# ============================================================

tree_fraction = (
    tree_cover
    / 100.0
)

shrub_cover_fraction = (
    shrub_cover
    / 100.0
)

herb_cover_fraction = (
    herb_cover
    / 100.0
)


# ============================================================
# SAVE STATIC STACK
# ============================================================

np.savez_compressed(

    OUT_FILE,

    # grid
    x=x,
    y=y,
    resolution=np.float64(res),
    epsg=np.int32(26956),

    # terrain
    elevation=elevation,
    elevation_std=elevation_std,
    slope=slope,
    slope_std=slope_std,
    northness=northness,
    southness=southness,
    eastness=eastness,

    # soils
    sand=sand,
    silt=silt,
    clay=clay,
    ksat=ksat,
    bulk_density=bulk_density,
    theta_wp=theta_wp,
    theta_fc=theta_fc,
    theta_sat=theta_sat,

    # NLCD
    landcover_mode=landcover_mode,
    water_fraction=water_fraction,
    developed_fraction=developed_fraction,
    high_developed_fraction=high_developed_fraction,
    impervious_mean=impervious,
    forest_fraction=forest_fraction,
    shrub_fraction=shrub_fraction,
    grass_fraction=grass_fraction,

    # vegetation
    tree_cover=tree_cover,
    shrub_cover=shrub_cover,
    herb_cover=herb_cover,

    tree_fraction=tree_fraction,
    shrub_cover_fraction=shrub_cover_fraction,
    herb_cover_fraction=herb_cover_fraction,

    # masks
    physics_valid=physics_valid,
    public_valid=public_valid,
)


# ============================================================
# WRITE MASK GEOTIFFS
# ============================================================

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "crs": TARGET_CRS,
    "transform": transform,
    "nodata": 0,
    "compress": "deflate",
    "predictor": 2,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}


for filename, A, description in [

    (
        "physics_valid_200m.tif",
        physics_valid,
        "Hero Dirt physics-valid mask",
    ),

    (
        "public_valid_200m.tif",
        public_valid,
        "Hero Dirt public display mask",
    ),

]:

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
            A.astype(np.uint8),
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
# SUMMARY
# ============================================================

n_total = physics_valid.size

n_physics = int(
    physics_valid.sum()
)

n_public = int(
    public_valid.sum()
)


print()
print("============================================")
print(" STATIC MODEL SUMMARY")
print("============================================")
print()

print(
    f"Total cells:            "
    f"{n_total:,}"
)

print(
    f"Physics-valid cells:    "
    f"{n_physics:,} "
    f"({100*n_physics/n_total:.2f}%)"
)

print(
    f"Public-product cells:   "
    f"{n_public:,} "
    f"({100*n_public/n_total:.2f}%)"
)

print()

print(
    f"Mostly water removed:   "
    f"{np.sum(physics_valid & mostly_water):,}"
)

print(
    f"Impervious >=50%:       "
    f"{np.sum(physics_valid & strongly_impervious):,}"
)

print(
    f"High development >=50%: "
    f"{np.sum(physics_valid & strongly_developed):,}"
)

print()

print("Vegetation within public domain:")

if np.any(public_valid):

    print(
        f"  mean tree cover:  "
        f"{np.nanmean(tree_cover[public_valid]):.2f}%"
    )

    print(
        f"  mean shrub cover: "
        f"{np.nanmean(shrub_cover[public_valid]):.2f}%"
    )

    print(
        f"  mean herb cover:  "
        f"{np.nanmean(herb_cover[public_valid]):.2f}%"
    )

print()

print(
    "Saved static model:"
)

print(
    f"  {OUT_FILE}"
)

print()
print("============================================")
print(" HERO DIRT STATIC MODEL COMPLETE")
print("============================================")
print()
