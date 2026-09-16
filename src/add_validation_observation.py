#!/usr/bin/env python3

"""
Hero Dirt Forecast
Add a rider / field observation to the validation database.

Current implementation includes the first observation:

    Sunset Ridge Trail, Altadena
    "pretty dry, but not dusty"

The script samples the current Hero Dirt model at the
observation location and appends both the observation and
model state to:

    output/validation/HeroDirt_validation.csv

Run this script whenever a new field report is received.
"""

from pathlib import Path
import csv
from datetime import datetime, timezone

import numpy as np
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

CURRENT_FILE = (
    ROOT
    / "output/current/HeroDirt_current_state.npz"
)

STATIC_FILE = (
    ROOT
    / "static/model/HeroDirt_static_200m.npz"
)

PLASTICITY_FILE = (
    ROOT
    / "static/soils/HeroDirt_plasticity_200m.npz"
)

VALIDATION_DIR = (
    ROOT
    / "output/validation"
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_FILE = (
    VALIDATION_DIR
    / "HeroDirt_validation.csv"
)


# ============================================================
# OBSERVATION
#
# Edit this block for each new report.
# ============================================================

TRAIL_NAME = "Sunset Ridge Trail"

LAT = 34.21775
LON = -118.12699


# Approximate observation time.
#
# Change this when known more precisely.
#
# Current report was received Sept 11, 2026.
# Use local PDT then convert manually here to UTC if desired.

OBSERVATION_TIME_UTC = "2026-09-11T19:30:00Z"


RIDER_DESCRIPTION = (
    "Pretty dry, but not dusty."
)


# Simple observational category.
#
# Suggested vocabulary:
#
# Very Dry
# Dry
# Moist/Good
# Hero
# Wet
# Too Wet
#
# This is the HUMAN interpretation of the report,
# not the model output.

OBSERVED_CATEGORY = "Dry"


SOURCE = "rider report"


# ============================================================
# LOAD MODEL DATA
# ============================================================

C = np.load(
    CURRENT_FILE
)

S = np.load(
    STATIC_FILE
)

P = np.load(
    PLASTICITY_FILE
)


# ============================================================
# TRANSFORM LOCATION
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:26911",
    always_xy=True,
)

E, N = transformer.transform(
    LON,
    LAT,
)


# ============================================================
# MODEL GRID
# ============================================================

x = C[
    "x"
].astype(float)

y = C[
    "y"
].astype(float)


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


grid_E = float(
    x[
        c
    ]
)

grid_N = float(
    y[
        r
    ]
)


grid_distance_m = float(
    np.sqrt(
        (
            grid_E - E
        ) ** 2
        +
        (
            grid_N - N
        ) ** 2
    )
)


# ============================================================
# MODEL STATE
# ============================================================

model_time = str(
    C[
        "time"
    ]
)


tread = C[
    "tread_theta"
].astype(float)

shallow = C[
    "shallow_theta"
].astype(float)

deep = C[
    "physical_deep_theta"
].astype(float)

theta05 = C[
    "theta_0_5cm"
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


CLASS_NAMES = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


model_class_id = int(
    condition[
        r,
        c
    ]
)

model_class_name = CLASS_NAMES.get(
    model_class_id,
    "Unknown",
)


# ============================================================
# STATIC PROPERTIES
# ============================================================

elevation = S[
    "elevation"
].astype(float)

slope = S[
    "slope"
].astype(float)

sand = S[
    "sand"
].astype(float)

clay = S[
    "clay"
].astype(float)

ksat = S[
    "ksat"
].astype(float)

wp = S[
    "theta_wp"
].astype(float)

fc = S[
    "theta_fc"
].astype(float)

sat = S[
    "theta_sat"
].astype(float)

PI = P[
    "plasticity_index"
].astype(float)


# ============================================================
# MATCH FLAG
# ============================================================

observed_clean = (
    OBSERVED_CATEGORY
    .strip()
    .lower()
)

model_clean = (
    model_class_name
    .strip()
    .lower()
)


category_match = (
    observed_clean
    ==
    model_clean
)


# ============================================================
# RECORD
# ============================================================

record = {

    "observation_time_utc":
        OBSERVATION_TIME_UTC,

    "model_time_utc":
        model_time,

    "trail":
        TRAIL_NAME,

    "latitude":
        LAT,

    "longitude":
        LON,

    "utm_e":
        E,

    "utm_n":
        N,

    "grid_e":
        grid_E,

    "grid_n":
        grid_N,

    "grid_distance_m":
        grid_distance_m,

    "source":
        SOURCE,

    "rider_description":
        RIDER_DESCRIPTION,

    "observed_category":
        OBSERVED_CATEGORY,

    "model_class":
        model_class_name,

    "category_match":
        category_match,

    "model_F":
        float(
            F[
                r,
                c
            ]
        ),

    "model_score":
        float(
            score[
                r,
                c
            ]
        ),

    "tread_theta":
        float(
            tread[
                r,
                c
            ]
        ),

    "shallow_theta":
        float(
            shallow[
                r,
                c
            ]
        ),

    "deep_theta":
        float(
            deep[
                r,
                c
            ]
        ),

    "theta_0_5cm":
        float(
            theta05[
                r,
                c
            ]
        ),

    "elevation_m":
        float(
            elevation[
                r,
                c
            ]
        ),

    "slope_deg":
        float(
            slope[
                r,
                c
            ]
        ),

    "sand_percent":
        float(
            sand[
                r,
                c
            ]
        ),

    "clay_percent":
        float(
            clay[
                r,
                c
            ]
        ),

    "ksat_um_s":
        float(
            ksat[
                r,
                c
            ]
        ),

    "theta_wp":
        float(
            wp[
                r,
                c
            ]
        ),

    "theta_fc":
        float(
            fc[
                r,
                c
            ]
        ),

    "theta_sat":
        float(
            sat[
                r,
                c
            ]
        ),

    "plasticity_index":
        float(
            PI[
                r,
                c
            ]
        ),
}


# ============================================================
# APPEND CSV
# ============================================================

fieldnames = list(
    record.keys()
)


file_exists = (
    CSV_FILE.exists()
)


with open(
    CSV_FILE,
    "a",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )


    if not file_exists:

        writer.writeheader()


    writer.writerow(
        record
    )


# ============================================================
# PRINT
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT VALIDATION OBSERVATION"
)
print(
    "============================================"
)
print()


print(
    f"Trail:       {TRAIL_NAME}"
)

print(
    f"Observation: {RIDER_DESCRIPTION}"
)

print(
    f"Observed:    {OBSERVED_CATEGORY}"
)

print()


print(
    "Model:"
)

print(
    f"  F:         "
    f"{record['model_F']:.3f}"
)

print(
    f"  score:     "
    f"{record['model_score']:.1f}"
)

print(
    f"  class:     "
    f"{model_class_name}"
)

print()


print(
    f"Category match: "
    f"{category_match}"
)

print()


print(
    f"Saved:"
)

print(
    f"  {CSV_FILE}"
)

print()


print(
    "============================================"
)
print()
