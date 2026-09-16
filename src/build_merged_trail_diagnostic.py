#!/usr/bin/env python3

from pathlib import Path
import json

import folium
import geopandas as gpd


ROOT = Path.home() / "HeroDirt_CT"

OSM_FILE = (
    ROOT
    / "data/osm/Connecticut_named_trails_overpass.json"
)

CTDEEP_FILE = (
    ROOT
    / "data/ctdeep/CTDEEP_trails.geojson"
)

OUTFILE = (
    ROOT
    / "web/HeroDirt_trails_merged_diagnostic.html"
)


# ============================================================
# MAP
# ============================================================

m = folium.Map(
    location=[
        41.60,
        -72.70,
    ],
    zoom_start=8,
    tiles="CartoDB positron",
)


# ============================================================
# OSM
# ============================================================

with open(
    OSM_FILE,
    "r",
) as f:

    osm = json.load(
        f
    )


osm_count = 0

for el in osm.get(
    "elements",
    [],
):

    if el.get(
        "type"
    ) != "way":
        continue

    geom = el.get(
        "geometry"
    )

    if not geom:
        continue

    coords = [
        [
            p["lat"],
            p["lon"],
        ]
        for p in geom
    ]

    name = (
        el.get(
            "tags",
            {},
        ).get(
            "name",
            "Unnamed OSM trail",
        )
    )

    folium.PolyLine(
        coords,
        color="#377eb8",
        weight=2,
        opacity=0.75,
        tooltip=f"OSM: {name}",
    ).add_to(
        m
    )

    osm_count += 1


# ============================================================
# CT DEEP
# ============================================================

gdf = gpd.read_file(
    CTDEEP_FILE
)

print(
    "CT DEEP columns:"
)

print(
    list(
        gdf.columns
    )
)

print()

ctdeep_count = 0


def add_linestring(
    geom,
    name,
):

    global ctdeep_count

    coords = [
        [
            lat,
            lon,
        ]
        for lon, lat in geom.coords
    ]

    folium.PolyLine(
        coords,
        color="#e41a1c",
        weight=2,
        opacity=0.70,
        tooltip=f"CT DEEP: {name}",
    ).add_to(
        m
    )

    ctdeep_count += 1


for _, row in gdf.iterrows():

    geom = row.geometry

    if geom is None:
        continue

    name = "Unnamed CT DEEP trail"

    for field in [
        "TRAILNAME",
        "TRAIL_NAME",
        "NAME",
    ]:

        if (
            field in gdf.columns
            and row.get(
                field
            )
        ):

            name = str(
                row[
                    field
                ]
            )

            break

    if geom.geom_type == "LineString":

        add_linestring(
            geom,
            name,
        )

    elif geom.geom_type == "MultiLineString":

        for part in geom.geoms:

            add_linestring(
                part,
                name,
            )


# ============================================================
# CRANBURY MARKER
# ============================================================

folium.Marker(
    location=[
        41.1638,
        -73.4027,
    ],
    tooltip="Cranbury Park",
    popup="Cranbury Park diagnostic location",
).add_to(
    m
)


# ============================================================
# LEGEND
# ============================================================

legend = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background: white;
    padding: 10px 14px;
    border: 1px solid #999;
    border-radius: 6px;
    font-size: 13px;
">
<b>Trail source diagnostic</b><br>
<span style="color:#377eb8;">━━</span> OSM<br>
<span style="color:#e41a1c;">━━</span> CT DEEP
</div>
"""

m.get_root().html.add_child(
    folium.Element(
        legend
    )
)


m.save(
    OUTFILE
)


print(
    f"OSM ways drawn:     {osm_count}"
)

print(
    f"CT DEEP lines drawn:{ctdeep_count}"
)

print(
    f"Saved: {OUTFILE}"
)
