#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt_CT"

MRMS_FILE = (
    ROOT
    / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)

CURRENT_FILE = (
    ROOT
    / "output/current_forecast_et4_real/HeroDirt_current_state.npz"
)

FORECAST_FILE = (
    ROOT
    / "output/forecast_forecast_et4_real/HeroDirt_forecast_6hourly.npz"
)

STATIC_FILE = (
    ROOT
    / "static/model/HeroDirt_static_200m.npz"
)


# ============================================================
# LOAD
# ============================================================

M = np.load(MRMS_FILE)
C = np.load(CURRENT_FILE)
FCST = np.load(FORECAST_FILE)
S = np.load(STATIC_FILE)


mrms_time = M[
    "time"
].astype(
    "datetime64[s]"
)

precip = M[
    "precip_mm"
].astype(float)


x = C[
    "x"
].astype(float)

y = C[
    "y"
].astype(float)


public = C[
    "public_valid"
].astype(bool)


current_tread = C[
    "tread_theta"
].astype(float)

current_F = C[
    "F"
].astype(float)


theta_fc = S[
    "theta_fc"
].astype(float)


forecast_time = FCST[
    "time"
].astype(
    "datetime64[s]"
)

forecast_F = FCST[
    "F"
].astype(float)


forecast_start = C[
    "time"
].astype(
    "datetime64[s]"
)


# ============================================================
# TODAY'S MRMS
# ============================================================

day = mrms_time.max().astype(
    "datetime64[D]"
)

mask_day = (
    mrms_time.astype(
        "datetime64[D]"
    )
    ==
    day
)

rain_mm = np.nansum(
    precip[
        mask_day
    ],
    axis=0,
)

rain_in = (
    rain_mm
    /
    25.4
)


# ============================================================
# CURRENT EXCESS TREAD WATER ABOVE FIELD CAPACITY
#
# W = theta * Z
# Tread thickness = 20 mm
# ============================================================

ZTREAD = 20.0

excess_mm = np.maximum(
    (
        current_tread
        -
        theta_fc
    )
    *
    ZTREAD,
    0.0,
)

excess_mm[
    ~public
] = np.nan


# ============================================================
# TIME TO THRESHOLD
# ============================================================

def first_time_below(
    threshold
):

    out = np.full(
        current_F.shape,
        np.nan,
        dtype=float,
    )

    already = (
        public
        &
        np.isfinite(
            current_F
        )
        &
        (
            current_F
            <
            threshold
        )
    )

    out[
        already
    ] = 0.0


    for k in range(
        len(
            forecast_time
        )
    ):

        hours = (
            forecast_time[
                k
            ]
            -
            forecast_start
        ).astype(
            "timedelta64[s]"
        ).astype(
            float
        ) / 3600.0


        hit = (
            public
            &
            np.isnan(
                out
            )
            &
            np.isfinite(
                forecast_F[
                    k
                ]
            )
            &
            (
                forecast_F[
                    k
                ]
                <
                threshold
            )
        )


        out[
            hit
        ] = hours


    return out


time_to_hero = first_time_below(
    0.80
)

time_to_good = first_time_below(
    0.65
)


# ============================================================
# SUMMARY
# ============================================================

valid_rain = (
    public
    &
    np.isfinite(
        rain_in
    )
)


def correlation_with_rain(
    field
):

    mask = (
        valid_rain
        &
        np.isfinite(
            field
        )
    )

    if np.count_nonzero(
        mask
    ) < 10:

        return np.nan

    return np.corrcoef(
        rain_in[
            mask
        ],
        field[
            mask
        ],
    )[0, 1]


print()
print(
    "============================================"
)
print(
    " HERO DIRT RECOVERY THRESHOLD DIAGNOSTIC"
)
print(
    "============================================"
)
print()

print(
    f"MRMS date: {day}"
)

print()

print(
    "Current excess tread water above field capacity [mm]:"
)

print(
    f"  mean:   "
    f"{np.nanmean(excess_mm):.3f}"
)

print(
    f"  median: "
    f"{np.nanmedian(excess_mm):.3f}"
)

print(
    f"  P95:    "
    f"{np.nanpercentile(excess_mm, 95):.3f}"
)

print(
    f"  max:    "
    f"{np.nanmax(excess_mm):.3f}"
)

print()

print(
    "Correlation with today's rainfall:"
)

print(
    f"  excess water: "
    f"{correlation_with_rain(excess_mm):.3f}"
)

print(
    f"  time to Hero (F<0.80): "
    f"{correlation_with_rain(time_to_hero):.3f}"
)

print(
    f"  time to Moist/Good (F<0.65): "
    f"{correlation_with_rain(time_to_good):.3f}"
)

print()

print(
    "Cells reaching Hero within forecast:"
)

print(
    f"  "
    f"{np.count_nonzero(np.isfinite(time_to_hero)):,}"
    f" / "
    f"{np.count_nonzero(public):,}"
)

print()

print(
    "Cells reaching Moist/Good within forecast:"
)

print(
    f"  "
    f"{np.count_nonzero(np.isfinite(time_to_good)):,}"
    f" / "
    f"{np.count_nonzero(public):,}"
)

print()


# ============================================================
# MAP 1
# RAIN
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    np.where(
        public,
        rain_in,
        np.nan,
    ),
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "MRMS precipitation [inches]"
)

ax.set_title(
    "Today's MRMS precipitation"
)

ax.set_xlabel(
    "Easting [m]"
)

ax.set_ylabel(
    "Northing [m]"
)

ax.set_aspect(
    "equal"
)

plt.tight_layout()
plt.show()


# ============================================================
# MAP 2
# CURRENT EXCESS WATER
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    excess_mm,
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "Tread water above field capacity [mm]"
)

ax.set_title(
    "Current excess tread water"
)

ax.set_xlabel(
    "Easting [m]"
)

ax.set_ylabel(
    "Northing [m]"
)

ax.set_aspect(
    "equal"
)

plt.tight_layout()
plt.show()


# ============================================================
# MAP 3
# TIME TO HERO
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    time_to_hero,
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "Hours until F < 0.80"
)

ax.set_title(
    "Time to Hero"
)

ax.set_xlabel(
    "Easting [m]"
)

ax.set_ylabel(
    "Northing [m]"
)

ax.set_aspect(
    "equal"
)

plt.tight_layout()
plt.show()


# ============================================================
# MAP 4
# TIME TO MOIST / GOOD
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    time_to_good,
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "Hours until F < 0.65"
)

ax.set_title(
    "Time to Moist / Good"
)

ax.set_xlabel(
    "Easting [m]"
)

ax.set_ylabel(
    "Northing [m]"
)

ax.set_aspect(
    "equal"
)

plt.tight_layout()
plt.show()
