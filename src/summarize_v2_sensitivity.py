#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from pyproj import Transformer


ROOT = Path(
    __file__
).resolve().parents[1]


CASES = {
    "k050_t06": (
        ROOT / "output/current_v2_fullpenman_k050/HeroDirt_current_state.npz",
        ROOT / "output/forecast_v2_fullpenman_k050/HeroDirt_forecast_6hourly.npz",
    ),

    "k050_t12": (
        ROOT / "output/current_v2_fullpenman_k050_tau12/HeroDirt_current_state.npz",
        ROOT / "output/forecast_v2_fullpenman_k050_tau12/HeroDirt_forecast_6hourly.npz",
    ),

    "k050_t24": (
        ROOT / "output/current_v2_fullpenman_k050_tau24/HeroDirt_current_state.npz",
        ROOT / "output/forecast_v2_fullpenman_k050_tau24/HeroDirt_forecast_6hourly.npz",
    ),

    "k075_t06": (
        ROOT / "output/current_v2_fullpenman_k075/HeroDirt_current_state.npz",
        ROOT / "output/forecast_v2_fullpenman_k075/HeroDirt_forecast_6hourly.npz",
    ),

    "k075_t12": (
        ROOT / "output/current_v2_fullpenman_k075_tau12/HeroDirt_current_state.npz",
        ROOT / "output/forecast_v2_fullpenman_k075_tau12/HeroDirt_forecast_6hourly.npz",
    ),

    "k075_t24": (
        ROOT / "output/current_v2_fullpenman_k075_tau24/HeroDirt_current_state.npz",
        ROOT / "output/forecast_v2_fullpenman_k075_tau24/HeroDirt_forecast_6hourly.npz",
    ),
}


TRAIL_FILE = (
    ROOT
    / "static/model/HeroDirt_trail_cells_200m.npz"
)


CLASS_NAMES = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


# ============================================================
# CRANBURY LOCATION
# ============================================================

CRANBURY_LAT = 41.1638
CRANBURY_LON = -73.4027


first_current = np.load(
    list(CASES.values())[0][0]
)

x = first_current[
    "x"
].astype(float)

y = first_current[
    "y"
].astype(float)


ll_to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:26956",
    always_xy=True,
)


E, N = ll_to_utm.transform(
    CRANBURY_LON,
    CRANBURY_LAT,
)


cran_c = int(
    np.argmin(
        np.abs(
            x - E
        )
    )
)

cran_r = int(
    np.argmin(
        np.abs(
            y - N
        )
    )
)


# ============================================================
# TRAIL CELLS
# ============================================================

T = np.load(
    TRAIL_FILE
)

trail_r = T[
    "row"
].astype(int)

trail_c = T[
    "col"
].astype(int)


# ============================================================
# CURRENT / NOW SUMMARY
# ============================================================

print()
print(
    "============================================================"
)
print(
    " FULL-PENMAN V2 SENSITIVITY SUMMARY"
)
print(
    "============================================================"
)

print()
print(
    "NOW STATE"
)
print(
    "---------"
)

print(
    "case       "
    "state_F  "
    "cran_F   "
    "cran_class        "
    "trail_F"
)


for name, (
    current_file,
    forecast_file,
) in CASES.items():

    C = np.load(
        current_file
    )

    public = C[
        "public_valid"
    ].astype(bool)

    F = C[
        "F"
    ].astype(float)

    condition = C[
        "condition"
    ].astype(int)

    state_F = np.nanmean(
        F[
            public
        ]
    )

    cran_F = F[
        cran_r,
        cran_c,
    ]

    cran_class = CLASS_NAMES.get(
        int(
            condition[
                cran_r,
                cran_c,
            ]
        ),
        "Unknown",
    )

    trail_F = np.nanmean(
        F[
            trail_r,
            trail_c,
        ]
    )

    print(
        f"{name:10s} "
        f"{state_F:7.3f}  "
        f"{cran_F:7.3f}   "
        f"{cran_class:15s} "
        f"{trail_F:7.3f}"
    )


# ============================================================
# FORECAST TRAIL F
# ============================================================

print()
print(
    "TRAIL-NETWORK FORECAST F"
)
print(
    "------------------------"
)

print(
    "case       "
    "+24h    "
    "+48h    "
    "+72h    "
    "+96h"
)


indices = {
    24: 3,
    48: 7,
    72: 11,
    96: 15,
}


for name, (
    current_file,
    forecast_file,
) in CASES.items():

    D = np.load(
        forecast_file
    )

    vals = {}

    for hour, idx in indices.items():

        F = D[
            "F"
        ][
            idx
        ].astype(float)

        vals[
            hour
        ] = np.nanmean(
            F[
                trail_r,
                trail_c,
            ]
        )


    print(
        f"{name:10s} "
        f"{vals[24]:7.3f} "
        f"{vals[48]:7.3f} "
        f"{vals[72]:7.3f} "
        f"{vals[96]:7.3f}"
    )


# ============================================================
# +48 H TRAIL CLASSES
# ============================================================

print()
print(
    "+48 H TRAIL CLASS FRACTIONS"
)
print(
    "---------------------------"
)

print(
    "case       "
    "VDry    "
    "Dry     "
    "Moist   "
    "Hero    "
    "Wet"
)


idx = 7


for name, (
    current_file,
    forecast_file,
) in CASES.items():

    D = np.load(
        forecast_file
    )

    cls = D[
        "condition"
    ][
        idx
    ][
        trail_r,
        trail_c,
    ].astype(int)


    good = (
        cls >= 1
    ) & (
        cls <= 6
    )

    cls = cls[
        good
    ]


    pct = {
        code:
            100.0
            *
            np.mean(
                cls == code
            )
        for code in range(
            1,
            7
        )
    }


    print(
        f"{name:10s} "
        f"{pct[1]:7.1f} "
        f"{pct[2]:7.1f} "
        f"{pct[3]:7.1f} "
        f"{pct[4]:7.1f} "
        f"{pct[5]:7.1f}"
    )


# ============================================================
# FIRST-NIGHT REBOUND
#
# Sep 17 00 -> 12 UTC:
# approximately evening through morning in Connecticut.
# ============================================================

print()
print(
    "OVERNIGHT TRAIL REBOUND"
)
print(
    "-----------------------"
)

print(
    "case       "
    "F_00UTC  "
    "F_12UTC  "
    "delta_F"
)


for name, (
    current_file,
    forecast_file,
) in CASES.items():

    D = np.load(
        forecast_file
    )

    # Sep 17 00 UTC = index 4
    # Sep 17 12 UTC = index 6

    F0 = D[
        "F"
    ][
        4
    ].astype(float)

    F1 = D[
        "F"
    ][
        6
    ].astype(float)


    m0 = np.nanmean(
        F0[
            trail_r,
            trail_c,
        ]
    )

    m1 = np.nanmean(
        F1[
            trail_r,
            trail_c,
        ]
    )


    print(
        f"{name:10s} "
        f"{m0:8.3f} "
        f"{m1:8.3f} "
        f"{m1-m0:+8.3f}"
    )


print()
print(
    "============================================================"
)
print(
    " SENSITIVITY SUMMARY COMPLETE"
)
print(
    "============================================================"
)
print()
