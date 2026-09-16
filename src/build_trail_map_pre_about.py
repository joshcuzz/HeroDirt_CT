#!/usr/bin/env python3

"""
Hero Dirt Forecast
Interactive trail forecast map + rider reporting.

Forecast horizons:
    Now
    +24 h
    +48 h
    +72 h
    +96 h
    +120 h

Reporting
---------
Users may:

    1. Click a location on the map.
    2. Click "Report Conditions".
    3. Enter trail, ride time, condition and comments.
    4. Download a small JSON report file.

The report can later be ingested with:

    python src/ingest_report.py <report.json>

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
# MODEL DOMAIN
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
# PREPROCESS TRAILS
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

            "samples":
                samples,
        }
    )


print()
print(
    "============================================"
)
print(
    " HERO DIRT FORECAST + REPORTING MAP"
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
# CONDITION CHUNKS
# ============================================================

def make_condition_chunks(
    trail_samples,
    condition_array,
):

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
    south_lat + north_lat
) / 2.0

center_lon = (
    west_lon + east_lon
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
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr="Tiles © Esri",

    name="Esri Satellite",

    overlay=False,

    control=True,

    show=False,

).add_to(
    m
)


# ============================================================
# FORECAST LAYERS
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
    )


    chunk_count = 0


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


            cid = int(
                condition_array[
                    r,
                    c
                ]
            )


            if not (
                1 <= cid <= 6
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
                cid
            )


        if len(
            classes
        ) < 2:

            continue


        counts = np.bincount(
            np.array(
                classes,
                dtype=int,
            ),
            minlength=7,
        )


        dominant_class = int(
            np.argmax(
                counts[1:]
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

        </div>
        """


        chunks = make_condition_chunks(
            samples,
            condition_array,
        )


        for chunk in chunks:

            cid = chunk[
                "class"
            ]


            folium.PolyLine(

                locations=chunk[
                    "coords"
                ],

                color=CLASS_COLORS[
                    cid
                ],

                weight=5,

                opacity=0.90,

                tooltip=(
                    f"{trail['name']} — "
                    f"{CLASS_NAMES[cid]} "
                    f"({label})"
                ),

                popup=folium.Popup(
                    popup_html,
                    max_width=300,
                ),

            ).add_to(
                fg
            )


            chunk_count += 1


    fg.add_to(
        m
    )


    layer_stats.append(
        (
            label,
            chunk_count,
            valid_time,
        )
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
# CONDITION LEGEND
# ============================================================

legend_html = """
<div style="
    position:fixed;
    bottom:35px;
    left:35px;
    z-index:9998;
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
Too Wet

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# REPORT BUTTON
# ============================================================

report_button_html = """
<button
    id="hero-report-button"
    style="
        position:fixed;
        bottom:35px;
        right:35px;
        z-index:10001;
        padding:12px 18px;
        border:none;
        border-radius:7px;
        background:#333333;
        color:white;
        font-family:Arial;
        font-size:14px;
        font-weight:bold;
        cursor:pointer;
        box-shadow:0px 1px 5px rgba(0,0,0,0.4);
    "
>
Report Trail Conditions
</button>
"""


m.get_root().html.add_child(
    folium.Element(
        report_button_html
    )
)


# ============================================================
# REPORT MODAL
# ============================================================

report_modal_html = """
<div
    id="hero-report-modal"
    style="
        display:none;
        position:fixed;
        z-index:20000;
        left:0;
        top:0;
        width:100%;
        height:100%;
        background:rgba(0,0,0,0.45);
        font-family:Arial;
    "
>

<div
    style="
        background:white;
        width:360px;
        max-width:90%;
        margin:6% auto;
        padding:20px;
        border-radius:9px;
        box-shadow:0px 3px 15px rgba(0,0,0,0.45);
    "
>

<div style="
    display:flex;
    justify-content:space-between;
    align-items:center;
">

<b style="font-size:19px;">
Report Trail Conditions
</b>

<span
    id="hero-report-close"
    style="
        cursor:pointer;
        font-size:24px;
    "
>
&times;
</span>

</div>

<p style="
    font-size:12px;
    color:#555;
">
Click the map before opening this form to set the
observation location. You may edit the coordinates below.
</p>


<label>Trail name</label><br>
<input
    id="report-trail"
    type="text"
    style="width:100%; margin-bottom:10px;"
>


<label>Latitude</label><br>
<input
    id="report-lat"
    type="number"
    step="0.000001"
    style="width:100%; margin-bottom:10px;"
>


<label>Longitude</label><br>
<input
    id="report-lon"
    type="number"
    step="0.000001"
    style="width:100%; margin-bottom:10px;"
>


<label>Ride time</label><br>
<input
    id="report-time"
    type="datetime-local"
    style="width:100%; margin-bottom:10px;"
>


<label>Trail condition</label><br>

<select
    id="report-condition"
    style="width:100%; margin-bottom:10px;"
>

<option value="Very Dry">
Dusty / loose
</option>

<option value="Dry">
Dry but firm
</option>

<option value="Moist/Good">
Good
</option>

<option value="Hero">
Hero
</option>

<option value="Wet">
Damp / wet but rideable
</option>

<option value="Too Wet">
Muddy / too wet
</option>

</select>


<label>Comments</label><br>

<textarea
    id="report-comment"
    rows="3"
    style="
        width:100%;
        margin-bottom:10px;
    "
></textarea>


<label>Name or initials (optional)</label><br>

<input
    id="report-name"
    type="text"
    style="
        width:100%;
        margin-bottom:15px;
    "
>


<button
    id="report-download"
    style="
        width:100%;
        padding:11px;
        border:none;
        border-radius:6px;
        background:#333333;
        color:white;
        font-weight:bold;
        cursor:pointer;
    "
>
Save Report
</button>

<p
    id="report-status"
    style="
        font-size:11px;
        color:#555;
        margin-top:10px;
    "
>
</p>

</div>
</div>
"""


m.get_root().html.add_child(
    folium.Element(
        report_modal_html
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
# REPORTING JAVASCRIPT
# ============================================================

map_name = m.get_name()


report_js = f"""
<script>

document.addEventListener(
    "DOMContentLoaded",
    function() {{

        var map = {map_name};

        var selectedLat = null;
        var selectedLon = null;
        var locationMarker = null;


        // ----------------------------------------------------
        // Remember last map click
        // ----------------------------------------------------

        map.on(
            "click",
            function(e) {{

                selectedLat =
                    e.latlng.lat;

                selectedLon =
                    e.latlng.lng;


                if (
                    locationMarker !== null
                ) {{

                    map.removeLayer(
                        locationMarker
                    );

                }}


                locationMarker =
                    L.circleMarker(
                        e.latlng,
                        {{
                            radius:5,
                            color:"#111",
                            weight:2,
                            fillColor:"#ffffff",
                            fillOpacity:1.0
                        }}
                    )
                    .addTo(
                        map
                    );

            }}
        );


        // ----------------------------------------------------
        // Report button
        // ----------------------------------------------------

        document
        .getElementById(
            "hero-report-button"
        )
        .onclick = function() {{

            document
            .getElementById(
                "hero-report-modal"
            )
            .style.display =
                "block";


            if (
                selectedLat !== null
            ) {{

                document
                .getElementById(
                    "report-lat"
                )
                .value =
                    selectedLat.toFixed(
                        6
                    );


                document
                .getElementById(
                    "report-lon"
                )
                .value =
                    selectedLon.toFixed(
                        6
                    );

            }}


            var now =
                new Date();


            var local =
                new Date(
                    now.getTime()
                    -
                    now.getTimezoneOffset()
                    *
                    60000
                );


            document
            .getElementById(
                "report-time"
            )
            .value =
                local
                .toISOString()
                .slice(
                    0,
                    16
                );

        }};


        // ----------------------------------------------------
        // Close modal
        // ----------------------------------------------------

        document
        .getElementById(
            "hero-report-close"
        )
        .onclick = function() {{

            document
            .getElementById(
                "hero-report-modal"
            )
            .style.display =
                "none";

        }};


        // ----------------------------------------------------
        // Download JSON
        // ----------------------------------------------------

        document
        .getElementById(
            "report-download"
        )
        .onclick = function() {{

            var trail =
                document
                .getElementById(
                    "report-trail"
                )
                .value
                .trim();


            var lat =
                parseFloat(
                    document
                    .getElementById(
                        "report-lat"
                    )
                    .value
                );


            var lon =
                parseFloat(
                    document
                    .getElementById(
                        "report-lon"
                    )
                    .value
                );


            var observationLocal =
                document
                .getElementById(
                    "report-time"
                )
                .value;


            var condition =
                document
                .getElementById(
                    "report-condition"
                )
                .value;


            var comment =
                document
                .getElementById(
                    "report-comment"
                )
                .value
                .trim();


            var reporter =
                document
                .getElementById(
                    "report-name"
                )
                .value
                .trim();


            if (
                trail === ""
                ||
                !isFinite(lat)
                ||
                !isFinite(lon)
                ||
                observationLocal === ""
            ) {{

                document
                .getElementById(
                    "report-status"
                )
                .innerText =
                    "Please provide trail, location and ride time.";

                return;

            }}


            var observationDate =
                new Date(
                    observationLocal
                );


            var payload = {{

                schema_version:
                    1,

                trail:
                    trail,

                latitude:
                    lat,

                longitude:
                    lon,

                observation_time_local:
                    observationLocal,

                observation_time_utc:
                    observationDate
                    .toISOString(),

                timezone_offset_minutes:
                    observationDate
                    .getTimezoneOffset(),

                observed_category:
                    condition,

                rider_description:
                    comment,

                source:
                    reporter !== ""
                    ?
                    reporter
                    :
                    "rider report",

                submitted_at_utc:
                    new Date()
                    .toISOString()

            }};


            var json =
                JSON.stringify(
                    payload,
                    null,
                    2
                );


            var blob =
                new Blob(
                    [json],
                    {{
                        type:
                            "application/json"
                    }}
                );


            var url =
                URL.createObjectURL(
                    blob
                );


            var stamp =
                new Date()
                .toISOString()
                .replace(
                    /[-:]/g,
                    ""
                )
                .replace(
                    /\\..+/,
                    ""
                );


            var a =
                document.createElement(
                    "a"
                );


            a.href =
                url;

            a.download =
                "herodirt_report_"
                +
                stamp
                +
                ".json";


            document.body.appendChild(
                a
            );


            a.click();


            document.body.removeChild(
                a
            );


            URL.revokeObjectURL(
                url
            );


            document
            .getElementById(
                "report-status"
            )
            .innerText =
                "Report saved. Thank you!";

        }};


    }}
);

</script>
"""


m.get_root().html.add_child(
    folium.Element(
        report_js
    )
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
    "Reporting:"
)

print(
    "  click map -> Report Trail Conditions"
)

print(
    "  form downloads a JSON observation"
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
    "============================================"
)
print(
    " TRAIL FORECAST + REPORTING MAP COMPLETE"
)
print(
    "============================================"
)
