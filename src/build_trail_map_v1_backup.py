#!/usr/bin/env python3

"""
Hero Dirt Forecast
Trail-based interactive map.

Version 2:
    - Download named trails / paths / tracks from OpenStreetMap
    - Cache OSM response locally
    - Sample Hero Dirt model along each trail
    - Color trail segments by local Hero Dirt condition
    - Show trail summaries in interactive popups

Output:
    web/HeroDirt_trails.html

Important:
    This is initially a map of OSM-mapped trails and paths.
    It does NOT imply bicycle legality or access.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import requests
import rasterio

from pyproj import Transformer

import folium


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

OSM_DIR = (
    ROOT
    / "data/osm"
)

WEB_DIR = (
    ROOT
    / "web"
)

OSM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WEB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OSM_CACHE = (
    OSM_DIR
    / "SanGabriels_named_trails_overpass.json"
)

OUTPUT_HTML = (
    WEB_DIR
    / "HeroDirt_trails.html"
)


# ============================================================
# SETTINGS
# ============================================================

# Sample trail geometry approximately every this many meters.

SAMPLE_SPACING_M = 100.0


# Redownload OSM each run?
#
# False:
#     use local cache if it exists.
#
# True:
#     query Overpass again.

REFRESH_OSM = False


# ============================================================
# HERO DIRT CONDITION COLORS
# ============================================================

CLASS_NAMES = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


CLASS_COLORS = {
    1: "#9c5a08",   # Very Dry
    2: "#d8ad58",   # Dry
    3: "#b9ded8",   # Moist / Good
    4: "#42a89e",   # Hero
    5: "#3182bd",   # Wet
    6: "#08519c",   # Too Wet
}


# ============================================================
# LOAD HERO DIRT MODEL
# ============================================================

C = np.load(
    CURRENT_FILE
)


x = C[
    "x"
].astype(float)

y = C[
    "y"
].astype(float)


F = C[
    "F"
].astype(float)

score = C[
    "score"
].astype(float)

condition = C[
    "condition"
].astype(int)

public = C[
    "public_valid"
].astype(bool)


model_time = str(
    C[
        "time"
    ]
)


# ============================================================
# PROJECTIONS
# ============================================================

ll_to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:26911",
    always_xy=True,
)

utm_to_ll = Transformer.from_crs(
    "EPSG:26911",
    "EPSG:4326",
    always_xy=True,
)


# ============================================================
# MODEL DOMAIN BOUNDS
# ============================================================

with rasterio.open(
    GRID_TIF
) as src:

    b = src.bounds


west_lon, south_lat = utm_to_ll.transform(
    b.left,
    b.bottom,
)

east_lon, north_lat = utm_to_ll.transform(
    b.right,
    b.top,
)


print()
print(
    "============================================"
)
print(
    " HERO DIRT TRAIL MAP"
)
print(
    "============================================"
)
print()

print(
    "Model domain:"
)

print(
    f"  south = {south_lat:.5f}"
)

print(
    f"  west  = {west_lon:.5f}"
)

print(
    f"  north = {north_lat:.5f}"
)

print(
    f"  east  = {east_lon:.5f}"
)

print()


# ============================================================
# OVERPASS QUERY
# ============================================================

def download_osm():

    """
    Download named OSM trails / paths / tracks.

    We intentionally start broadly.

    Later we can filter based on:
        bicycle
        mtb:scale
        access
        surface
        trail type
    """

    query = f"""
    [out:json][timeout:180];

    (
      way
        ["highway"~"path|track|footway|cycleway"]
        ["name"]
        ({south_lat},{west_lon},{north_lat},{east_lon});
    );

    out tags geom;
    """


    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]


    last_error = None


    for endpoint in endpoints:

        print(
            f"Querying:"
        )

        print(
            f"  {endpoint}"
        )

        try:

            response = requests.post(
                endpoint,
                data={
                    "data": query
                },
                timeout=240,
                headers={
                    "User-Agent":
                        "HeroDirtForecast/0.1 research prototype"
                },
            )


            response.raise_for_status()


            data = response.json()


            with open(
                OSM_CACHE,
                "w",
            ) as f:

                json.dump(
                    data,
                    f,
                )


            print(
                f"Saved OSM cache:"
            )

            print(
                f"  {OSM_CACHE}"
            )

            print()

            return data


        except Exception as exc:

            last_error = exc

            print(
                f"Failed:"
            )

            print(
                f"  {exc}"
            )

            print()


    raise RuntimeError(
        "All Overpass endpoints failed."
    ) from last_error


# ============================================================
# LOAD / DOWNLOAD OSM
# ============================================================

if (
    OSM_CACHE.exists()
    and
    not REFRESH_OSM
):

    print(
        "Using cached OSM trails:"
    )

    print(
        f"  {OSM_CACHE}"
    )

    print()


    with open(
        OSM_CACHE,
        "r",
    ) as f:

        osm = json.load(
            f
        )

else:

    osm = download_osm()


# ============================================================
# GRID SAMPLING
# ============================================================

def nearest_grid_cell(
    lon,
    lat,
):

    E, N = ll_to_utm.transform(
        lon,
        lat,
    )


    c = int(
        np.argmin(
            np.abs(
                x - E
            )
        )
    )


    r = int(
        np.argmin(
            np.abs(
                y - N
            )
        )
    )


    return (
        r,
        c,
        E,
        N,
    )


# ============================================================
# DISTANCE / INTERPOLATION
# ============================================================

def segment_length_m(
    lon1,
    lat1,
    lon2,
    lat2,
):

    E1, N1 = ll_to_utm.transform(
        lon1,
        lat1,
    )

    E2, N2 = ll_to_utm.transform(
        lon2,
        lat2,
    )


    return math.hypot(
        E2 - E1,
        N2 - N1,
    )


def densify_way(
    coords,
):

    """
    Convert OSM vertex geometry to points spaced approximately
    SAMPLE_SPACING_M apart.
    """

    dense = []


    if len(
        coords
    ) < 2:

        return dense


    dense.append(
        coords[
            0
        ]
    )


    for i in range(
        len(coords) - 1
    ):

        lon1, lat1 = coords[
            i
        ]

        lon2, lat2 = coords[
            i + 1
        ]


        length = segment_length_m(
            lon1,
            lat1,
            lon2,
            lat2,
        )


        n = max(
            1,
            int(
                math.ceil(
                    length
                    /
                    SAMPLE_SPACING_M
                )
            ),
        )


        for k in range(
            1,
            n + 1
        ):

            f = (
                k
                /
                n
            )


            lon = (
                lon1
                +
                f
                *
                (
                    lon2
                    -
                    lon1
                )
            )


            lat = (
                lat1
                +
                f
                *
                (
                    lat2
                    -
                    lat1
                )
            )


            dense.append(
                (
                    lon,
                    lat,
                )
            )


    return dense


# ============================================================
# PROCESS OSM WAYS
# ============================================================

trails = []


for element in osm.get(
    "elements",
    [],
):

    if element.get(
        "type"
    ) != "way":

        continue


    tags = element.get(
        "tags",
        {},
    )


    name = tags.get(
        "name",
        "",
    ).strip()


    geometry = element.get(
        "geometry",
        [],
    )


    if (
        not name
        or
        len(geometry) < 2
    ):

        continue


    coords = [
        (
            p[
                "lon"
            ],
            p[
                "lat"
            ],
        )
        for p in geometry
    ]


    dense = densify_way(
        coords
    )


    samples = []


    for lon, lat in dense:

        r, c, E, N = nearest_grid_cell(
            lon,
            lat,
        )


        if not public[
            r,
            c
        ]:

            continue


        samples.append(
            {
                "lon": lon,
                "lat": lat,

                "r": r,
                "c": c,

                "F":
                    float(
                        F[
                            r,
                            c
                        ]
                    ),

                "score":
                    float(
                        score[
                            r,
                            c
                        ]
                    ),

                "condition":
                    int(
                        condition[
                            r,
                            c
                        ]
                    ),
            }
        )


    if len(
        samples
    ) < 2:

        continue


    trail_conditions = np.array(
        [
            s[
                "condition"
            ]
            for s in samples
        ],
        dtype=int,
    )


    trail_F = np.array(
        [
            s[
                "F"
            ]
            for s in samples
        ],
        dtype=float,
    )


    trail_score = np.array(
        [
            s[
                "score"
            ]
            for s in samples
        ],
        dtype=float,
    )


    # Modal / dominant condition.

    valid_classes = trail_conditions[
        (
            trail_conditions >= 1
        )
        &
        (
            trail_conditions <= 6
        )
    ]


    if len(
        valid_classes
    ) == 0:

        continue


    counts = np.bincount(
        valid_classes,
        minlength=7,
    )


    dominant_class = int(
        np.argmax(
            counts[
                1:
            ]
        )
        +
        1
    )


    trails.append(
        {
            "name":
                name,

            "osm_id":
                element[
                    "id"
                ],

            "highway":
                tags.get(
                    "highway",
                    "",
                ),

            "surface":
                tags.get(
                    "surface",
                    "",
                ),

            "bicycle":
                tags.get(
                    "bicycle",
                    "",
                ),

            "mtb_scale":
                tags.get(
                    "mtb:scale",
                    "",
                ),

            "samples":
                samples,

            "median_F":
                float(
                    np.nanmedian(
                        trail_F
                    )
                ),

            "median_score":
                float(
                    np.nanmedian(
                        trail_score
                    )
                ),

            "dominant_class":
                dominant_class,
        }
    )


print(
    f"Usable named OSM ways: "
    f"{len(trails)}"
)

print()


# ============================================================
# MAP
# ============================================================

center_lat = (
    south_lat
    +
    north_lat
) / 2.0

center_lon = (
    west_lon
    +
    east_lon
) / 2.0


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
# ESRI TOPO BASEMAP
# ============================================================

folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Topo_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr=(
        "Tiles © Esri — Sources: Esri, Garmin, "
        "USGS, NGA, EPA, USDA, NPS"
    ),

    name="Esri World Topo",

    overlay=False,

    control=True,

    show=True,

).add_to(
    m
)


# ============================================================
# OPTIONAL OPENTOPOMAP
# ============================================================

folium.TileLayer(

    tiles=(
        "https://{s}.tile.opentopomap.org/"
        "{z}/{x}/{y}.png"
    ),

    attr=(
        "Map data © OpenStreetMap contributors | "
        "Map style © OpenTopoMap"
    ),

    name="OpenTopoMap",

    overlay=False,

    control=True,

    show=False,

).add_to(
    m
)


# ============================================================
# TRAIL FEATURE GROUP
# ============================================================

trail_layer = folium.FeatureGroup(
    name="Hero Dirt Trails",
    show=True,
)


# ============================================================
# DRAW TRAILS
# ============================================================

segment_count = 0


for trail in trails:

    samples = trail[
        "samples"
    ]


    dominant_name = CLASS_NAMES.get(
        trail[
            "dominant_class"
        ],
        "Unknown",
    )


    surface_text = (
        trail[
            "surface"
        ]
        if trail[
            "surface"
        ]
        else "not mapped"
    )


    bike_text = (
        trail[
            "bicycle"
        ]
        if trail[
            "bicycle"
        ]
        else "not mapped"
    )


    mtb_text = (
        trail[
            "mtb_scale"
        ]
        if trail[
            "mtb_scale"
        ]
        else "not mapped"
    )


    popup_html = f"""
    <div style="font-family:Arial; width:260px;">

    <b style="font-size:15px;">
        {trail['name']}
    </b>

    <br><br>

    <b>Hero Dirt condition:</b>
        {dominant_name}

    <br>

    <b>Median F:</b>
        {trail['median_F']:.2f}

    <br>

    <b>Median score:</b>
        {trail['median_score']:.0f}

    <br><br>

    <b>OSM type:</b>
        {trail['highway']}

    <br>

    <b>Surface:</b>
        {surface_text}

    <br>

    <b>Bicycle tag:</b>
        {bike_text}

    <br>

    <b>MTB scale:</b>
        {mtb_text}

    <br><br>

    <span style="font-size:10px; color:#555;">
        Trail access and bicycle legality must be verified
        independently.
    </span>

    </div>
    """


    popup = folium.Popup(
        popup_html,
        max_width=300,
    )


    # --------------------------------------------------------
    # Draw each short trail segment according to local
    # Hero Dirt condition.
    # --------------------------------------------------------

    for i in range(
        len(samples) - 1
    ):

        a = samples[
            i
        ]

        b = samples[
            i + 1
        ]


        # Use midpoint / average class logic.
        #
        # For now use the wetter/worse-directional local
        # sample endpoint if classes differ.

        ca = int(
            a[
                "condition"
            ]
        )

        cb = int(
            b[
                "condition"
            ]
        )


        if ca == cb:

            seg_class = ca

        else:

            # Use midpoint F to sample directly.

            mid_lon = (
                a[
                    "lon"
                ]
                +
                b[
                    "lon"
                ]
            ) / 2.0

            mid_lat = (
                a[
                    "lat"
                ]
                +
                b[
                    "lat"
                ]
            ) / 2.0


            rmid, cmid, _, _ = nearest_grid_cell(
                mid_lon,
                mid_lat,
            )


            seg_class = int(
                condition[
                    rmid,
                    cmid
                ]
            )


        color = CLASS_COLORS.get(
            seg_class,
            "#777777",
        )


        line = folium.PolyLine(

            locations=[
                [
                    a[
                        "lat"
                    ],
                    a[
                        "lon"
                    ],
                ],
                [
                    b[
                        "lat"
                    ],
                    b[
                        "lon"
                    ],
                ],
            ],

            color=color,

            weight=5,

            opacity=0.90,

            tooltip=(
                f"{trail['name']} — "
                f"{CLASS_NAMES.get(seg_class,'Unknown')}"
            ),

        )


        line.add_child(
            popup
        )


        line.add_to(
            trail_layer
        )


        segment_count += 1


trail_layer.add_to(
    m
)


print(
    f"Colored trail segments: "
    f"{segment_count}"
)

print()


# ============================================================
# SUNSET RIDGE VALIDATION MARKER
# ============================================================

folium.CircleMarker(

    location=[
        34.21775,
        -118.12699,
    ],

    radius=6,

    color="#000000",

    fill=True,

    fill_color="#ffffff",

    fill_opacity=1.0,

    weight=2,

    tooltip="Sunset Ridge rider observation",

    popup=folium.Popup(
        """
        <b>Sunset Ridge Trail</b><br><br>

        Rider report:<br>
        "Pretty dry, but not dusty."<br><br>

        Observed category: Dry<br>
        Model F: 0.220<br>
        Model score: 41<br>
        Model class: Dry
        """,
        max_width=300,
    ),

).add_to(
    m
)


# ============================================================
# TITLE
# ============================================================

title_html = f"""
<div style="
    position:fixed;
    top:12px;
    left:55px;
    z-index:9999;
    background-color:rgba(255,255,255,0.94);
    padding:10px 15px;
    border-radius:7px;
    font-family:Arial;
    box-shadow:0px 1px 5px rgba(0,0,0,0.35);
">

<b style="font-size:18px;">
Hero Dirt Forecast
</b>

<br>

<span style="font-size:12px;">
Current: {model_time} UTC
</span>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        title_html
    )
)


# ============================================================
# LEGEND
# ============================================================

legend_html = """
<div style="
    position:fixed;
    bottom:35px;
    left:35px;
    z-index:9999;
    background-color:rgba(255,255,255,0.94);
    padding:12px 15px;
    border-radius:7px;
    font-family:Arial;
    font-size:13px;
    box-shadow:0px 1px 5px rgba(0,0,0,0.35);
">

<b>Trail Condition</b><br><br>

<span style="color:#9c5a08;">●</span>
Very Dry<br>

<span style="color:#d8ad58;">●</span>
Dry<br>

<span style="color:#b9ded8;">●</span>
Moist / Good<br>

<span style="color:#42a89e;">●</span>
Hero<br>

<span style="color:#3182bd;">●</span>
Wet<br>

<span style="color:#08519c;">●</span>
Too Wet<br>

<br>

<span style="
    font-size:10px;
    color:#666;
">
OSM trails shown for condition guidance only.<br>
Check current access and bicycle regulations.
</span>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
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
# FIT DOMAIN
# ============================================================

m.fit_bounds(
    [
        [
            south_lat,
            west_lon,
        ],
        [
            north_lat,
            east_lon,
        ],
    ]
)


# ============================================================
# SAVE
# ============================================================

m.save(
    OUTPUT_HTML
)


print(
    "Saved:"
)

print(
    f"  {OUTPUT_HTML}"
)

print()

print(
    "Open:"
)

print(
    f"  open {OUTPUT_HTML}"
)

print()

print(
    "============================================"
)
print(
    " TRAIL MAP COMPLETE"
)
print(
    "============================================"
)
print()
