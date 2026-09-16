#!/usr/bin/env python3

import json
import math

from pathlib import Path

import numpy as np

from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]


CURRENT_FILE = (
    ROOT
    / "output/current/HeroDirt_current_state.npz"
)

OSM_CACHE = (
    ROOT
    / "data/osm/Connecticut_named_trails_overpass.json"
)


FORECAST_FILES = {
    "v1":
        ROOT
        / "output/forecast/HeroDirt_forecast_6hourly.npz",

    "k050":
        ROOT
        / "output/forecast_v2_penman_k050/HeroDirt_forecast_6hourly.npz",

    "k075":
        ROOT
        / "output/forecast_v2_penman_k075/HeroDirt_forecast_6hourly.npz",

    "k100":
        ROOT
        / "output/forecast_v2_penman_k100/HeroDirt_forecast_6hourly.npz",
}


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SPACING_M = 100.0

TARGET_HOUR = 48


# ============================================================
# LOAD GRID
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


public = C[
    "public_valid"
].astype(bool)


# ============================================================
# COORDINATE TRANSFORM
# ============================================================

ll_to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:26956",
    always_xy=True,
)


# ============================================================
# OSM
# ============================================================

with open(
    OSM_CACHE,
    "r",
) as f:

    osm = json.load(
        f
    )


# ============================================================
# GRID LOOKUP
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
# GEOMETRY
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

    if len(coords) < 2:

        return []


    dense = [
        coords[0]
    ]


    for i in range(
        len(coords) - 1
    ):

        lon1, lat1 = coords[i]

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
# BUILD TRAIL SAMPLE SET
# ============================================================

all_samples = []

trail_count = 0


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


    surface = tags.get(
        "surface",
        "",
    ).lower()


    paved_surfaces = {
        "asphalt",
        "paved",
        "concrete",
        "concrete:plates",
        "paving_stones",
    }


    if surface in paved_surfaces:

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


    samples_this_trail = []


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


        samples_this_trail.append(
            (
                r,
                c,
            )
        )


    if len(
        samples_this_trail
    ) < 2:

        continue


    trail_count += 1

    all_samples.extend(
        samples_this_trail
    )


# ============================================================
# UNIQUE GRID CELLS
# ============================================================

unique_cells = sorted(
    set(
        all_samples
    )
)
TRAIL_CELL_FILE = (
    ROOT
    / "static/model/HeroDirt_trail_cells_200m.npz"
)

trail_rows = np.array(
    [r for r, c in unique_cells],
    dtype=np.int32,
)

trail_cols = np.array(
    [c for r, c in unique_cells],
    dtype=np.int32,
)

np.savez_compressed(
    TRAIL_CELL_FILE,
    row=trail_rows,
    col=trail_cols,
)

print(
    f"Saved unique trail cells: "
    f"{TRAIL_CELL_FILE}"
)

print()
print(
    "============================================"
)
print(
    " HERO DIRT TRAIL-POINT FORECAST COMPARISON"
)
print(
    "============================================"
)

print()
print(
    f"Named unpaved OSM ways: "
    f"{trail_count:,}"
)

print(
    f"Densified trail samples: "
    f"{len(all_samples):,}"
)

print(
    f"Unique 200 m trail cells: "
    f"{len(unique_cells):,}"
)


# ============================================================
# HELPERS
# ============================================================

CLASS_NAMES = {
    1: "Very Dry / Dusty",
    2: "Dry / Firm",
    3: "Moist / Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


def summarize(
    name,
    F,
    condition,
    cells,
):

    vals = np.array(
        [
            F[
                r,
                c
            ]
            for r, c in cells
        ],
        dtype=float,
    )


    classes = np.array(
        [
            int(
                condition[
                    r,
                    c
                ]
            )
            for r, c in cells
        ],
        dtype=int,
    )


    good = (
        np.isfinite(
            vals
        )
        &
        (
            classes >= 1
        )
        &
        (
            classes <= 6
        )
    )


    vals = vals[
        good
    ]

    classes = classes[
        good
    ]


    p = np.percentile(
        vals,
        [
            5,
            25,
            50,
            75,
            95,
        ],
    )


    print()
    print(
        name
    )

    print(
        f"  n:       "
        f"{len(vals):,}"
    )

    print(
        f"  mean F:  "
        f"{np.mean(vals):.3f}"
    )

    print(
        f"  p05:     "
        f"{p[0]:.3f}"
    )

    print(
        f"  p25:     "
        f"{p[1]:.3f}"
    )

    print(
        f"  p50:     "
        f"{p[2]:.3f}"
    )

    print(
        f"  p75:     "
        f"{p[3]:.3f}"
    )

    print(
        f"  p95:     "
        f"{p[4]:.3f}"
    )

    print()

    for code in range(
        1,
        7
    ):

        pct = (
            100.0
            *
            np.mean(
                classes == code
            )
        )

        print(
            f"  "
            f"{CLASS_NAMES[code]:16s}: "
            f"{pct:6.2f}%"
        )


# ============================================================
# LOAD FORECASTS AND FIND +48 H INDEX
# ============================================================

datasets = {
    name:
        np.load(
            fn
        )
    for name, fn in FORECAST_FILES.items()
}


base = datasets[
    "v1"
]


times = base[
    "time"
]


target_index = (
    int(
        TARGET_HOUR
        /
        6
    )
    -
    1
)


print()
print(
    f"Target forecast hour: "
    f"+{TARGET_HOUR} h"
)

print(
    f"Target valid time: "
    f"{times[target_index]}"
)


# ============================================================
# DENSIFIED-SAMPLE SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " ALL DENSIFIED TRAIL SAMPLES"
)
print(
    "============================================"
)


for name in [
    "v1",
    "k050",
    "k075",
    "k100",
]:

    D = datasets[
        name
    ]

    summarize(
        name,
        D["F"][
            target_index
        ],
        D["condition"][
            target_index
        ],
        all_samples,
    )


# ============================================================
# UNIQUE-CELL SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " UNIQUE 200 M TRAIL CELLS"
)
print(
    "============================================"
)


for name in [
    "v1",
    "k050",
    "k075",
    "k100",
]:

    D = datasets[
        name
    ]

    summarize(
        name,
        D["F"][
            target_index
        ],
        D["condition"][
            target_index
        ],
        unique_cells,
    )


print()
print(
    "============================================"
)
print(
    " COMPARISON COMPLETE"
)
print(
    "============================================"
)
print()
