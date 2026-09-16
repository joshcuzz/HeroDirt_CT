#!/usr/bin/env python3

from pathlib import Path
import json
import requests


ROOT = Path.home() / "HeroDirt_CT"

OUTDIR = ROOT / "data/ctdeep"
OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTFILE = (
    OUTDIR
    / "CTDEEP_trails.geojson"
)

URL = (
    "https://services1.arcgis.com/"
    "FjPcSmEFuDYlIdKC/arcgis/rest/services/"
    "DEEP_Trails_Set/FeatureServer/3/query"
)

params = {
    "where": "1=1",
    "outFields": (
        "OBJECTID,"
        "TRAILNAME,"
        "TRAILCLASS,"
        "TRAILSURF,"
        "TRAILSTAT,"
        "PUBACCESS,"
        "BIKE,"
        "MTNBIKE,"
        "TRAILMARK"
    ),
    "returnGeometry": "true",
    "outSR": "4326",
    "f": "geojson",
    "resultRecordCount": 1000,
}

features = []
offset = 0

while True:

    params[
        "resultOffset"
    ] = offset

    print(
        f"Requesting offset {offset}..."
    )

    r = requests.get(
        URL,
        params=params,
        timeout=60,
    )

    r.raise_for_status()

    data = r.json()

    batch = data.get(
        "features",
        [],
    )

    features.extend(
        batch
    )

    print(
        f"  received {len(batch)}"
    )

    if len(batch) < 1000:
        break

    offset += 1000


geojson = {
    "type": "FeatureCollection",
    "features": features,
}

with open(
    OUTFILE,
    "w",
) as f:

    json.dump(
        geojson,
        f,
    )


print()
print(
    f"Saved {len(features)} trails"
)

print(
    OUTFILE
)
