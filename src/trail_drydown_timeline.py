#!/usr/bin/env python3

from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]


TRAIL_CELL_FILE = (
    ROOT
    / "static/model/HeroDirt_trail_cells_200m.npz"
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


CLASS_NAMES = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


# ============================================================
# LOAD TRAIL CELLS
# ============================================================

T = np.load(
    TRAIL_CELL_FILE
)

rows = T[
    "row"
].astype(int)

cols = T[
    "col"
].astype(int)


# ============================================================
# LOAD FORECASTS
# ============================================================

D = {
    name:
        np.load(
            fn
        )
    for name, fn in FORECAST_FILES.items()
}


times = D[
    "v1"
][
    "time"
]


print()
print(
    "============================================"
)
print(
    " HERO DIRT TRAIL DRYDOWN TIMELINE"
)
print(
    "============================================"
)

print(
    f"Unique trail cells: "
    f"{len(rows):,}"
)


# ============================================================
# MEAN F THROUGH TIME
# ============================================================

print()
print(
    "MEAN TRAIL F"
)
print(
    "------------"
)

print(
    "time                     "
    "v1     "
    "k050   "
    "k075   "
    "k100"
)


for k in range(
    len(times)
):

    vals = {}

    for name in [
        "v1",
        "k050",
        "k075",
        "k100",
    ]:

        F = D[
            name
        ][
            "F"
        ][
            k
        ]

        f = F[
            rows,
            cols,
        ].astype(float)

        vals[
            name
        ] = np.nanmean(
            f
        )


    print(
        f"{str(times[k]):24s} "
        f"{vals['v1']:6.3f} "
        f"{vals['k050']:6.3f} "
        f"{vals['k075']:6.3f} "
        f"{vals['k100']:6.3f}"
    )


# ============================================================
# CLASS FRACTIONS THROUGH TIME
# ============================================================

for name in [
    "v1",
    "k050",
    "k075",
    "k100",
]:

    print()
    print(
        "============================================"
    )
    print(
        f" {name.upper()} TRAIL CLASS EVOLUTION"
    )
    print(
        "============================================"
    )

    print(
        "time                     "
        "VDry    "
        "Dry     "
	"Moist   "
        "Hero    "
        "Wet    "
        "TooWet"
    )


    for k in range(
        len(times)
    ):

        cls = D[
            name
        ][
            "condition"
        ][
            k
        ][
            rows,
            cols,
        ].astype(int)


        good = (
            cls >= 1
        ) & (
            cls <= 6
        )

        cls = cls[
            good
        ]


        very_dry = (
            100.0
            *
            np.mean(
                cls == 1
            )
	)

        dry = (
            100.0
            *
            np.mean(
                cls == 2
            )
        )

        moist = (
            100.0
            *
            np.mean(
                cls == 3
            )
        )

        hero = (
            100.0
            *
            np.mean(
                cls == 4
            )
        )

        wet = (
            100.0
            *
            np.mean(
                cls == 5
            )
        )

        too_wet = (
            100.0
            *
            np.mean(
                cls == 6
            )
        )


        print(
            f"{str(times[k]):24s} "
            f"{very_dry:6.1f} "
	    f"{dry:6.1f} "
            f"{moist:6.1f} "
            f"{hero:6.1f} "
            f"{wet:6.1f} "
            f"{too_wet:6.1f}"
        )


print()
print(
    "============================================"
)
print(
    " TIMELINE COMPLETE"
)
print(
    "============================================"
)
print()
