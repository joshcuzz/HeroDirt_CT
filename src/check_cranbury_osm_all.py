#!/usr/bin/env python3

from pathlib import Path
import json

import folium


ROOT = Path.home() / "HeroDirt_CT"

OSM_FILE = (
    ROOT
    / "data/osm/Connecticut_all_trails_overpass.json"
)

OUTFILE = (
    ROOT
    / "web/Cranbury_OSM_all_diagnostic.html"
)


# Cranbury Park vicinity
LAT0 = 41.1638
LON0 = -73.4027

PAD = 0.025


with open(OSM_FILE) as f:
    osm = json.load(f)


m = folium.Map(
    location=[LAT0, LON0],
    zoom_start=14,
    tiles="CartoDB positron",
)


count = 0


for element in osm.get("elements", []):

    if element.get("type") != "way":
        continue

    geom = element.get("geometry", [])

    if len(geom) < 2:
        continue

    # Keep ways intersecting Cranbury vicinity.
    nearby = False

    for p in geom:

        if (
            LAT0 - PAD <= p["lat"] <= LAT0 + PAD
            and
            LON0 - PAD <= p["lon"] <= LON0 + PAD
        ):
            nearby = True
            break

    if not nearby:
        continue

    tags = element.get("tags", {})

    highway = tags.get("highway", "")
    name = tags.get("name", "")
    surface = tags.get("surface", "")
    bicycle = tags.get("bicycle", "")
    mtb = tags.get("mtb:scale", "")
    access = tags.get("access", "")

    coords = [
        [p["lat"], p["lon"]]
        for p in geom
    ]

    tooltip = (
        f"{name if name else '(unnamed)'} | "
        f"highway={highway} | "
        f"surface={surface or '-'} | "
        f"bicycle={bicycle or '-'} | "
        f"mtb={mtb or '-'} | "
        f"access={access or '-'}"
    )

    folium.PolyLine(
        coords,
        weight=4,
        opacity=0.85,
        tooltip=tooltip,
    ).add_to(m)

    count += 1


folium.Marker(
    [LAT0, LON0],
    tooltip="Cranbury Park reference",
).add_to(m)


m.save(OUTFILE)

print()
print(f"Cranbury-area ways drawn: {count}")
print(f"Saved: {OUTFILE}")
print()
