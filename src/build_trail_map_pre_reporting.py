#!/usr/bin/env python3

"""
Hero Dirt Forecast
Interactive trail forecast map with forecast controls.

Forecast horizons:
    Now
    +24 h
    +48 h
    +72 h
    +96 h
    +120 h

Uses:
    output/current/HeroDirt_current_state.npz
    output/forecast/HeroDirt_forecast_6hourly.npz
    cached OSM trail geometry

Key design:
    - sample model along OSM trails
    - merge consecutive same-class samples
    - build one Leaflet feature group per forecast horizon
    - only "Now" is shown initially

Output:
    web/HeroDirt_trails.html
"""

from pathlib import Path
import json
import math

import numpy as np
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

FORECAST_FILE = (
    ROOT
    / "output/forecast/HeroDirt_forecast_6hourly.npz"
)

GRID_TIF = (
    ROOT
    / "static/grid/SanGabriels_DEM_200m_UTM11.tif"
)

OSM_CACHE = (
    ROOT
    / "data/osm/SanGabriels_named_trails_overpass.json"
)

OUTPUT_HTML = (
    ROOT
    / "web/HeroDirt_trails.html"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SPACING_M = 100.0

FORECAST_HOURS = [
    0,
    24,
    48,
    72,
    96,
    120,
]


# ============================================================
# HERO DIRT CLASSES
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
    1: "#9c5a08",
    2: "#d8ad58",
    3: "#b9ded8",
    4: "#42a89e",
    5: "#3182bd",
    6: "#08519c",
}


# ============================================================
# LOAD CURRENT + FORECAST
# ============================================================

C = np.load(
    CURRENT_FILE
)

FCST = np.load(
    FORECAST_FILE
)


x = C[
    "x"
].astype(float)

y = C[
    "y"
].astype(float)


current_F = C[
    "F"
].astype(float)

current_score = C[
    "score"
].astype(float)

current_condition = C[
    "condition"
].astype(int)


public = C[
    "public_valid"
].astype(bool)


current_time = C[
    "time"
].astype(
    "datetime64[s]"
)


forecast_time = FCST[
    "time"
].astype(
    "datetime64[s]"
)


forecast_F = FCST[
    "F"
].astype(float)

forecast_score = FCST[
    "score"
].astype(float)

forecast_condition = FCST[
    "condition"
].astype(int)


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
# DOMAIN
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


# ============================================================
# LOAD OSM CACHE
# ============================================================

if not OSM_CACHE.exists():

    raise RuntimeError(
        f"Missing OSM cache:\n{OSM_CACHE}"
    )


with open(
    OSM_CACHE,
    "r",
) as f:

    osm = json.load(
        f
    )


# ============================================================
# HELPER: NEAREST GRID CELL
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


    return r, c


# ============================================================
# GEOMETRY HELPERS
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

    if len(
        coords
    ) < 2:

        return []


    dense = [
        coords[0]
    ]


    for i in range(
        len(coords) - 1
    ):

        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]


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
                    lon2 - lon1
                )
            )


            lat = (
                lat1
                +
                f
                *
                (
                    lat2 - lat1
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
# PREPROCESS TRAIL GEOMETRY ONCE
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
            p["lon"],
            p["lat"],
        )
        for p in geometry
    ]


    dense = densify_way(
        coords
    )


    samples = []


    for lon, lat in dense:

        r, c = nearest_grid_cell(
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
            }
        )


    if len(
        samples
    ) < 2:

        continue


    trails.append(
        {
            "name":
                name,

            "osm_id":
                element["id"],

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
        }
    )


print()
print(
    "============================================"
)
print(
    " HERO DIRT FORECAST TRAIL MAP"
)
print(
    "============================================"
)
print()

print(
    f"Usable named OSM ways: "
    f"{len(trails)}"
)

print()


# ============================================================
# FORECAST LOOKUP
# ============================================================

def get_state_for_hour(
    hour,
):

    if hour == 0:

        return (
            current_time,
            current_F,
            current_score,
            current_condition,
        )


    target_time = (
        current_time
        +
        np.timedelta64(
            hour,
            "h",
        )
    )


    k = int(
        np.argmin(
            np.abs(
                forecast_time
                -
                target_time
            )
        )
    )


    return (
        forecast_time[
            k
        ],
        forecast_F[
            k
        ],
        forecast_score[
            k
        ],
        forecast_condition[
            k
        ],
    )


# ============================================================
# MERGE SAME-CLASS SEGMENTS
# ============================================================

def make_condition_chunks(
    trail_samples,
    condition_array,
):

    """
    Turn a sampled trail into contiguous same-class chunks.
    """

    if len(
        trail_samples
    ) < 2:

        return []


    samples = []


    for p in trail_samples:

        class_id = int(
            condition_array[
                p["r"],
                p["c"]
            ]
        )


        if (
            class_id < 1
            or
            class_id > 6
        ):

            continue


        samples.append(
            {
                "lat":
                    p["lat"],

                "lon":
                    p["lon"],

                "class":
                    class_id,
            }
        )


    if len(
        samples
    ) < 2:

        return []


    chunks = []


    current_class = samples[
        0
    ][
        "class"
    ]


    current_coords = [
        (
            samples[
                0
            ][
                "lat"
            ],

            samples[
                0
            ][
                "lon"
            ],
        )
    ]


    for i in range(
        1,
        len(samples)
    ):

        point = samples[
            i
        ]


        coord = (
            point[
                "lat"
            ],
            point[
                "lon"
            ],
        )


        if (
            point[
                "class"
            ]
            ==
            current_class
        ):

            current_coords.append(
                coord
            )


        else:

            current_coords.append(
                coord
            )


            if len(
                current_coords
            ) >= 2:

                chunks.append(
                    {
                        "class":
                            current_class,

                        "coords":
                            current_coords,
                    }
                )


            previous = samples[
                i - 1
            ]


            current_coords = [
                (
                    previous[
                        "lat"
                    ],
                    previous[
                        "lon"
                    ],
                ),
                coord,
            ]


            current_class = (
                point[
                    "class"
                ]
            )


    if len(
        current_coords
    ) >= 2:

        chunks.append(
            {
                "class":
                    current_class,

                "coords":
                    current_coords,
            }
        )


    return chunks


# ============================================================
# CREATE MAP
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

    prefer_canvas=True,
)


# ============================================================
# BASEMAPS
# ============================================================

folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Topo_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr="Tiles © Esri",

    name="Esri World Topo",

    overlay=False,

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
# BUILD EACH FORECAST LAYER
# ============================================================

layer_stats = []


for hour in FORECAST_HOURS:

    (
        valid_time,
        F_array,
        score_array,
        condition_array,
    ) = get_state_for_hour(
        hour
    )


    if hour == 0:

        label = "Now"

    else:

        label = f"+{hour} h"


    fg = folium.FeatureGroup(

        name=label,

        show=(
            hour == 0
        ),

        overlay=True,
    )


    rendered_chunks = 0


    for trail in trails:

        samples = trail[
            "samples"
        ]


        Fs = []

        scores = []

        classes = []


        for p in samples:

            r = p[
                "r"
            ]

            c = p[
                "c"
            ]


            class_id = int(
                condition_array[
                    r,
                    c
                ]
            )


            if (
                class_id < 1
                or
                class_id > 6
            ):

                continue


            Fs.append(
                float(
                    F_array[
                        r,
                        c
                    ]
                )
            )


            scores.append(
                float(
                    score_array[
                        r,
                        c
                    ]
                )
            )


            classes.append(
                class_id
            )


        if len(
            classes
        ) < 2:

            continue


        classes_np = np.array(
            classes,
            dtype=int,
        )


        counts = np.bincount(
            classes_np,
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


        dominant_name = CLASS_NAMES[
            dominant_class
        ]


        median_F = float(
            np.nanmedian(
                Fs
            )
        )


        median_score = float(
            np.nanmedian(
                scores
            )
        )


        surface_text = (
            trail["surface"]
            if trail["surface"]
            else "not mapped"
        )


        bike_text = (
            trail["bicycle"]
            if trail["bicycle"]
            else "not mapped"
        )


        popup_html = f"""
        <div style="
            font-family:Arial;
            width:260px;
        ">

        <b style="
            font-size:15px;
        ">
            {trail['name']}
        </b>

        <br><br>

        <b>Forecast:</b>
            {label}

        <br>

        <b>Valid:</b>
            {valid_time}

        <br><br>

        <b>Overall condition:</b>
            {dominant_name}

        <br>

        <b>Median F:</b>
            {median_F:.2f}

        <br>

        <b>Median score:</b>
            {median_score:.0f}

        <br><br>

        <b>OSM type:</b>
            {trail['highway']}

        <br>

        <b>Surface:</b>
            {surface_text}

        <br>

        <b>Bicycle:</b>
            {bike_text}

        </div>
        """


        chunks = make_condition_chunks(
            samples,
            condition_array,
        )


        for chunk in chunks:

            class_id = chunk[
                "class"
            ]


            folium.PolyLine(

                locations=chunk[
                    "coords"
                ],

                color=CLASS_COLORS[
                    class_id
                ],

                weight=5,

                opacity=0.90,

                tooltip=(
                    f"{trail['name']} — "
                    f"{CLASS_NAMES[class_id]} "
                    f"({label})"
                ),

                popup=folium.Popup(
                    popup_html,
                    max_width=300,
                ),

            ).add_to(
                fg
            )


            rendered_chunks += 1


    fg.add_to(
        m
    )


    layer_stats.append(
        (
            label,
            rendered_chunks,
            valid_time,
        )
    )


# ============================================================
# SUNSET RIDGE OBSERVATION
# ============================================================

folium.CircleMarker(

    location=[
        34.21775,
        -118.12699,
    ],

    radius=6,

    color="#111111",

    weight=2,

    fill=True,

    fill_color="#ffffff",

    fill_opacity=1.0,

    tooltip="Sunset Ridge rider observation",

    popup=folium.Popup(
        """
        <b>Sunset Ridge Trail</b>
        <br><br>

        Rider report:
        <br>
        "Pretty dry, but not dusty."

        <br><br>

        Observed: Dry
        <br>
        Model F: 0.220
        <br>
        Model score: 41
        <br>
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

<b style="
    font-size:18px;
">
Hero Dirt Forecast
</b>

<br>

<span style="
    font-size:12px;
">
Analysis: {current_time} UTC
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

<b>Trail Condition</b>
<br><br>

<span style="color:#9c5a08;">●</span>
Very Dry
<br>

<span style="color:#d8ad58;">●</span>
Dry
<br>

<span style="color:#b9ded8;">●</span>
Moist / Good
<br>

<span style="color:#42a89e;">●</span>
Hero
<br>

<span style="color:#3182bd;">●</span>
Wet
<br>

<span style="color:#08519c;">●</span>
Too Wet

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
    "Forecast layers:"
)

for (
    label,
    chunks,
    valid_time,
) in layer_stats:

    print(
        f"  {label:6s} "
        f"{chunks:6d} chunks   "
        f"valid {valid_time}"
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
    " FORECAST TRAIL MAP COMPLETE"
)
print(
    "============================================"
)
print()
