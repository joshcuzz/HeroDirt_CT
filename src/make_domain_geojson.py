#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box


ROOT = Path.home() / "HeroDirt_CT"

GRID_FILE = (
    ROOT
    / "static"
    / "grid"
    / "HeroDirt_grid_200m.npz"
)

OUT_DIR = (
    ROOT
    / "static"
    / "domain"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_FILE = (
    OUT_DIR
    / "HeroDirt_Connecticut_domain.geojson"
)


# ============================================================
# LOAD MASTER GRID
# ============================================================

G = np.load(
    GRID_FILE
)

west = float(G["west"])
east = float(G["east"])
south = float(G["south"])
north = float(G["north"])


# ============================================================
# CREATE UTM DOMAIN
# ============================================================

domain = gpd.GeoDataFrame(
    {
        "name": [
            "Hero Dirt Connecticut"
        ]
    },
    geometry=[
        box(
            west,
            south,
            east,
            north,
        )
    ],
    crs="EPSG:26956",
)


# ============================================================
# CONVERT TO WGS84 FOR WEB SERVICES
# ============================================================

domain = domain.to_crs(
    "EPSG:4326"
)


domain.to_file(
    OUT_FILE,
    driver="GeoJSON",
)


print()
print("Hero Dirt domain written:")
print(f"  {OUT_FILE}")
print()

print("WGS84 bounds:")
print(domain.total_bounds)
print()
