#!/usr/bin/env python3

"""
Hero Dirt Forecast
Optimized trail-based interactive map.

Key improvement:
    Sample model conditions densely along trails, but merge
    consecutive samples with the same Hero Dirt condition into
    a single Leaflet polyline.

This avoids creating hundreds of thousands of separate map
objects and makes the map practical in a browser.

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
# LOAD MODEL
# ============================================================

C = np.load(
    CURRENT_FILE
)

x = C["x"].astype(float)
y = C["y"].astype(float)

F = C["F"].astype(float)
score = C["score"].astype(float)
condition = C["condition"].astype(int)

public = C[
    "public_valid"
].astype(bool)

model_time = str(
    C["time"]
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
# LOAD CACHED OSM
# ============================================================

if not OSM_CACHE.exists():

    raise RuntimeError(
        f"OSM cache not found:\n{OSM_CACHE}"
    )


with open(
    OSM_CACHE,
    "r",
) as f:

    osm = json.load(
        f
    )


print()
print(
    "============================================"
)
print(
    " HERO DIRT OPTIMIZED TRAIL MAP"
)
print(
    "============================================"
)
print()

print(
    f"Using cached OSM:"
)

print(
    f"  {OSM_CACHE}"
)

print()


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


    return r, c


# ============================================================
# GEOMETRY UTILITIES
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
    Densify an OSM way to approximately SAMPLE_SPACING_M.
    """

    if len(coords) < 2:

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
# BUILD TRAIL SAMPLES
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


        class_id = int(
            condition[
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


        samples.append(
            {
                "lon": lon,
                "lat": lat,
                "F": float(
                    F[
                        r,
                        c
                    ]
                ),
                "score": float(
                    score[
                        r,
                        c
                    ]
                ),
                "class": class_id,
            }
        )


    if len(
        samples
    ) < 2:

        continue


    classes = np.array(
        [
            p["class"]
            for p in samples
        ],
        dtype=int,
    )


    Fs = np.array(
        [
            p["F"]
            for p in samples
        ],
        dtype=float,
    )


    scores = np.array(
        [
            p["score"]
            for p in samples
        ],
        dtype=float,
    )


    counts = np.bincount(
        classes,
        minlength=7,
    )


    dominant_class = int(
        np.argmax(
            counts[1:]
        )
        +
        1
    )


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

            "median_F":
                float(
                    np.nanmedian(
                        Fs
                    )
                ),

            "median_score":
                float(
                    np.nanmedian(
                        scores
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
# MERGE SAME-CLASS SEGMENTS
# ============================================================

def make_condition_chunks(
    samples,
):

    """
    Merge contiguous trail samples with the same model class.

    Each output chunk contains:
        class
        coordinates

    Adjacent chunks share their boundary coordinate so the
    rendered trail remains continuous.
    """

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


        point_coord = (
            point[
                "lat"
            ],
            point[
                "lon"
            ],
        )


        point_class = point[
            "class"
        ]


        if (
            point_class
            ==
            current_class
        ):

            current_coords.append(
                point_coord
            )


        else:

            # Include transition point so segments join.

            current_coords.append(
                point_coord
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


            # New segment begins at the previous coordinate
            # to preserve continuity.

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
                point_coord,
            ]


            current_class = (
                point_class
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
# BASEMAP
# ============================================================

folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Topo_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr=(
        "Tiles © Esri"
    ),

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
# TRAIL LAYER
# ============================================================

trail_layer = folium.FeatureGroup(

    name="Hero Dirt Trails",

    show=True,
)


rendered_chunks = 0


for trail in trails:

    chunks = make_condition_chunks(
        trail[
            "samples"
        ]
    )


    dominant_name = (
        CLASS_NAMES[
            trail[
                "dominant_class"
            ]
        ]
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


    mtb_text = (
        trail["mtb_scale"]
        if trail["mtb_scale"]
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

    <b>Overall condition:</b>
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

    <b>Bicycle:</b>
        {bike_text}

    <br>

    <b>MTB scale:</b>
        {mtb_text}

    <br><br>

    <span style="
        font-size:10px;
        color:#666;
    ">
    Condition guidance only.
    Verify bicycle access independently.
    </span>

    </div>
    """


    for chunk in chunks:

        class_id = chunk[
            "class"
        ]


        color = CLASS_COLORS[
            class_id
        ]


        class_name = CLASS_NAMES[
            class_id
        ]


        folium.PolyLine(

            locations=chunk[
                "coords"
            ],

            color=color,

            weight=5,

            opacity=0.90,

            tooltip=(
                f"{trail['name']} — "
                f"{class_name}"
            ),

            popup=folium.Popup(
                popup_html,
                max_width=300,
            ),

        ).add_to(
            trail_layer
        )


        rendered_chunks += 1


trail_layer.add_to(
    m
)


# ============================================================
# SUNSET RIDGE FIELD OBSERVATION
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

    tooltip=(
        "Sunset Ridge rider observation"
    ),

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

<br><br>

<span style="
    font-size:10px;
    color:#666;
">
Colors are model-estimated trail conditions.
</span>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# CONTROLS
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(
    m
)


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
    f"Original 100-m pairwise segments would be very large."
)

print(
    f"Rendered merged condition chunks: "
    f"{rendered_chunks}"
)

print()

print(
    f"Saved:"
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
    " OPTIMIZED TRAIL MAP COMPLETE"
)
print(
    "============================================"
)
print()
