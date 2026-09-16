#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build an interactive web map of current Hero Dirt conditions.

Creates:
    web/HeroDirt_map.html

Features
--------
- OpenStreetMap basemap
- OpenTopoMap basemap
- Semi-transparent Hero Dirt score overlay
- Semi-transparent Hero Dirt condition-class overlay
- Layer control
- Clickable map
- Hero Dirt raster embedded directly in HTML

The HTML file can be opened locally or shared with friends.
Internet access is required for the basemap tiles.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
import folium
from folium.raster_layers import ImageOverlay

import matplotlib.pyplot as plt
from matplotlib.colors import (
    Normalize,
    ListedColormap,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

CURRENT_FILE = (
    ROOT
    / "output/current/HeroDirt_current_state.npz"
)

GRID_TIF = (
    ROOT
    / "static/grid/SanGabriels_DEM_200m_UTM11.tif"
)

WEB_DIR = (
    ROOT
    / "web"
)

WEB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_HTML = (
    WEB_DIR
    / "HeroDirt_map.html"
)


# ============================================================
# LOAD CURRENT MODEL
# ============================================================

C = np.load(
    CURRENT_FILE
)


score = C[
    "score"
].astype(float)


condition = C[
    "condition"
].astype(float)


public = C[
    "public_valid"
].astype(bool)


model_time = str(
    C[
        "time"
    ]
)


score[
    ~public
] = np.nan


condition[
    ~public
] = np.nan


# ============================================================
# READ SOURCE GRID GEOMETRY
# ============================================================

with rasterio.open(
    GRID_TIF
) as src:

    src_crs = src.crs
    src_transform = src.transform
    src_width = src.width
    src_height = src.height
    src_bounds = src.bounds


# ============================================================
# REPROJECT MODEL ARRAYS TO WGS84
# ============================================================

dst_crs = "EPSG:4326"


dst_transform, dst_width, dst_height = (
    calculate_default_transform(
        src_crs,
        dst_crs,
        src_width,
        src_height,
        *src_bounds,
    )
)


score_ll = np.full(
    (
        dst_height,
        dst_width,
    ),
    np.nan,
    dtype=np.float32,
)


class_ll = np.full(
    (
        dst_height,
        dst_width,
    ),
    np.nan,
    dtype=np.float32,
)


reproject(

    source=score.astype(
        np.float32
    ),

    destination=score_ll,

    src_transform=src_transform,
    src_crs=src_crs,

    dst_transform=dst_transform,
    dst_crs=dst_crs,

    src_nodata=np.nan,
    dst_nodata=np.nan,

    resampling=Resampling.bilinear,
)


reproject(

    source=condition.astype(
        np.float32
    ),

    destination=class_ll,

    src_transform=src_transform,
    src_crs=src_crs,

    dst_transform=dst_transform,
    dst_crs=dst_crs,

    src_nodata=np.nan,
    dst_nodata=np.nan,

    resampling=Resampling.nearest,
)


# ============================================================
# WGS84 BOUNDS
# ============================================================

west = dst_transform.c

north = dst_transform.f

east = (
    west
    +
    dst_transform.a
    *
    dst_width
)

south = (
    north
    +
    dst_transform.e
    *
    dst_height
)


bounds = [
    [
        south,
        west,
    ],
    [
        north,
        east,
    ],
]


center_lat = (
    south + north
) / 2.0

center_lon = (
    west + east
) / 2.0


# ============================================================
# CREATE RGBA HERO SCORE IMAGE
# ============================================================

score_norm = Normalize(
    vmin=0.0,
    vmax=100.0,
)


score_cmap = plt.get_cmap(
    "viridis"
)


score_rgba = score_cmap(
    score_norm(
        np.nan_to_num(
            score_ll,
            nan=0.0,
        )
    )
)


# Transparent outside public mask.

score_alpha = np.where(
    np.isfinite(
        score_ll
    ),
    0.65,
    0.0,
)


score_rgba[
    ...,
    3
] = score_alpha


# ============================================================
# CREATE RGBA CLASS IMAGE
# ============================================================

class_colors = [
    "#8c510a",  # 1 Very Dry
    "#d8b365",  # 2 Dry
    "#c7eae5",  # 3 Moist/Good
    "#5ab4ac",  # 4 Hero
    "#4393c3",  # 5 Wet
    "#2166ac",  # 6 Too Wet
]


class_cmap = ListedColormap(
    class_colors
)


class_index = (
    class_ll - 1
)


class_rgba = class_cmap(
    np.clip(
        np.nan_to_num(
            class_index,
            nan=0.0,
        ),
        0,
        5,
    )
    /
    5.0
)


class_alpha = np.where(
    np.isfinite(
        class_ll
    ),
    0.65,
    0.0,
)


class_rgba[
    ...,
    3
] = class_alpha


# ============================================================
# CREATE MAP
# ============================================================

m = folium.Map(

    location=[
        center_lat,
        center_lon,
    ],

    zoom_start=10,

    tiles=None,

    control_scale=True,
)


# ============================================================
# BASEMAPS
# ============================================================

folium.TileLayer(

    tiles="OpenStreetMap",

    name="OpenStreetMap",

    control=True,

    show=True,

).add_to(
    m
)


folium.TileLayer(

    tiles=(
        "https://{s}.tile.opentopomap.org/"
        "{z}/{x}/{y}.png"
    ),

    attr=(
        "Map data © OpenStreetMap contributors, "
        "SRTM | Map style © OpenTopoMap"
    ),

    name="OpenTopoMap",

    overlay=False,

    control=True,

    show=False,

).add_to(
    m
)


# ============================================================
# HERO DIRT SCORE OVERLAY
# ============================================================

ImageOverlay(

    image=score_rgba,

    bounds=bounds,

    name="Hero Dirt Score",

    opacity=1.0,

    interactive=True,

    cross_origin=False,

    zindex=5,

    show=True,

).add_to(
    m
)


# ============================================================
# HERO DIRT CONDITION OVERLAY
# ============================================================

ImageOverlay(

    image=class_rgba,

    bounds=bounds,

    name="Hero Dirt Classes",

    opacity=1.0,

    interactive=True,

    cross_origin=False,

    zindex=6,

    show=False,

).add_to(
    m
)


# ============================================================
# TITLE
# ============================================================

title_html = f"""
<div style="
    position: fixed;
    top: 10px;
    left: 50px;
    z-index: 9999;
    background-color: rgba(255,255,255,0.90);
    padding: 10px 14px;
    border-radius: 6px;
    font-family: Arial;
    font-size: 16px;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.35);
">
    <b>Hero Dirt Forecast</b><br>
    <span style="font-size:12px;">
        Current conditions: {model_time} UTC
    </span>
</div>
"""


m.get_root().html.add_child(
    folium.Element(
        title_html
    )
)


# ============================================================
# SCORE LEGEND
# ============================================================

score_legend = """
<div style="
    position: fixed;
    bottom: 35px;
    left: 35px;
    width: 220px;
    z-index: 9999;
    background-color: rgba(255,255,255,0.92);
    padding: 10px;
    border-radius: 6px;
    font-family: Arial;
    font-size: 12px;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.35);
">

<b>Hero Dirt Score</b>

<div style="
    margin-top:6px;
    width:200px;
    height:14px;
    background:
    linear-gradient(
        to right,
        #440154,
        #3b528b,
        #21918c,
        #5ec962,
        #fde725
    );
"></div>

<div style="
    display:flex;
    justify-content:space-between;
    width:200px;
">
<span>0</span>
<span>25</span>
<span>50</span>
<span>75</span>
<span>100</span>
</div>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        score_legend
    )
)


# ============================================================
# CONDITION LEGEND
# ============================================================

class_legend = """
<div style="
    position: fixed;
    bottom: 35px;
    right: 35px;
    z-index: 9999;
    background-color: rgba(255,255,255,0.92);
    padding: 10px 12px;
    border-radius: 6px;
    font-family: Arial;
    font-size: 12px;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.35);
">

<b>Condition</b><br>

<span style="color:#8c510a;">■</span> Very Dry<br>
<span style="color:#d8b365;">■</span> Dry<br>
<span style="color:#c7eae5;">■</span> Moist/Good<br>
<span style="color:#5ab4ac;">■</span> Hero<br>
<span style="color:#4393c3;">■</span> Wet<br>
<span style="color:#2166ac;">■</span> Too Wet

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        class_legend
    )
)


# ============================================================
# SUNSET RIDGE VALIDATION MARKER
# ============================================================

folium.Marker(

    location=[
        34.21775,
        -118.12699,
    ],

    tooltip=(
        "Sunset Ridge Trail"
    ),

    popup=folium.Popup(
        """
        <b>Sunset Ridge Trail</b><br>
        Rider report: pretty dry, but not dusty.<br>
        Model class: Dry<br>
        Model F: 0.220<br>
        Hero score: 41
        """,
        max_width=300,
    ),

).add_to(
    m
)


# ============================================================
# CLICK LAT/LON
# ============================================================

folium.LatLngPopup().add_to(
    m
)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(
    m
)


# ============================================================
# FIT MAP TO HERO DIRT DOMAIN
# ============================================================

m.fit_bounds(
    bounds
)


# ============================================================
# SAVE
# ============================================================

m.save(
    OUTPUT_HTML
)


print()
print(
    "============================================"
)
print(
    " HERO DIRT INTERACTIVE MAP COMPLETE"
)
print(
    "============================================"
)
print()


print(
    f"Model time:"
)

print(
    f"  {model_time}"
)

print()


print(
    "Saved:"
)

print(
    f"  {OUTPUT_HTML}"
)

print()


print(
    "Open with:"
)

print(
    f"  open {OUTPUT_HTML}"
)

print()


print(
    "You can also send this HTML file to someone else."
)

print(
    "They will need internet access for the basemap tiles."
)

print()

print(
    "============================================"
)
print()
