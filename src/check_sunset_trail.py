#!/usr/bin/env python3

"""
Hero Dirt Forecast
Check current modeled conditions near Sunset Ridge Trail,
Altadena.

Uses a representative trailhead-area coordinate:

    latitude  = 34.21775
    longitude = -118.12699

Transforms WGS84 -> UTM Zone 11N and samples the nearest
200-m Hero Dirt grid cell.

Reports:
    tread theta
    shallow theta
    deep theta
    combined 0-5 cm theta
    tread F
    Hero Dirt score
    condition class
    soil properties
"""

from pathlib import Path

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
# REPRESENTATIVE SUNSET RIDGE TRAIL LOCATION
# ============================================================

LAT = 34.21775
LON = -118.12699


# ============================================================
# TRANSFORM TO UTM 11N
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


# Nearest grid column / row.
#
# Works whether y increases or decreases.

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


grid_E = x[
    c
]

grid_N = y[
    r
]


distance = np.sqrt(
    (
        grid_E - E
    ) ** 2
    +
    (
        grid_N - N
    ) ** 2
)


# ============================================================
# CURRENT STATE
# ============================================================

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


# ============================================================
# STATIC PROPERTIES
# ============================================================

wp = S[
    "theta_wp"
].astype(float)

fc = S[
    "theta_fc"
].astype(float)

sat = S[
    "theta_sat"
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

slope = S[
    "slope"
].astype(float)

elevation = S[
    "elevation"
].astype(float)

PI = P[
    "plasticity_index"
].astype(float)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


class_id = int(
    condition[
        r,
        c
    ]
)


class_name = CLASS_NAMES.get(
    class_id,
    "Unknown",
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT — SUNSET RIDGE TRAIL CHECK"
)
print(
    "============================================"
)
print()


print(
    "Representative location:"
)

print(
    f"  latitude:  {LAT:.5f}"
)

print(
    f"  longitude: {LON:.5f}"
)

print()


print(
    "UTM coordinate:"
)

print(
    f"  E = {E:.1f} m"
)

print(
    f"  N = {N:.1f} m"
)

print()


print(
    "Nearest Hero Dirt grid cell:"
)

print(
    f"  E = {grid_E:.1f} m"
)

print(
    f"  N = {grid_N:.1f} m"
)

print(
    f"  center distance = "
    f"{distance:.1f} m"
)

print()


print(
    "Current modeled condition:"
)

print(
    f"  tread theta      = "
    f"{tread[r,c]:.4f}"
)

print(
    f"  shallow theta    = "
    f"{shallow[r,c]:.4f}"
)

print(
    f"  deep theta       = "
    f"{deep[r,c]:.4f}"
)

print(
    f"  combined 0-5 cm  = "
    f"{theta05[r,c]:.4f}"
)

print()


print(
    f"  F                = "
    f"{F[r,c]:.3f}"
)

print(
    f"  Hero score       = "
    f"{score[r,c]:.1f}"
)

print(
    f"  class            = "
    f"{class_id} ({class_name})"
)

print()


print(
    "Local soil / terrain:"
)

print(
    f"  elevation        = "
    f"{elevation[r,c]:.0f} m"
)

print(
    f"  slope            = "
    f"{slope[r,c]:.1f} deg"
)

print(
    f"  sand             = "
    f"{sand[r,c]:.1f}%"
)

print(
    f"  clay             = "
    f"{clay[r,c]:.1f}%"
)

print(
    f"  Ksat             = "
    f"{ksat[r,c]:.2f} um/s"
)

print(
    f"  theta_wp         = "
    f"{wp[r,c]:.3f}"
)

print(
    f"  theta_fc         = "
    f"{fc[r,c]:.3f}"
)

print(
    f"  theta_sat        = "
    f"{sat[r,c]:.3f}"
)

print(
    f"  plasticity index = "
    f"{PI[r,c]:.2f}"
)

print()


# ============================================================
# SIMPLE INTERPRETATION
# ============================================================

print(
    "Interpretation:"
)


if class_id == 1:

    print(
        "  Model says VERY DRY."
    )

elif class_id == 2:

    print(
        "  Model says DRY — consistent with a firm,"
    )
    print(
        "  dry surface that may still retain enough"
    )
    print(
        "  moisture to avoid being dusty."
    )

elif class_id == 3:

    print(
        "  Model says MOIST/GOOD."
    )

elif class_id == 4:

    print(
        "  Model says HERO."
    )

elif class_id == 5:

    print(
        "  Model says WET."
    )

elif class_id == 6:

    print(
        "  Model says TOO WET."
    )


print()
print(
    "============================================"
)
print()
