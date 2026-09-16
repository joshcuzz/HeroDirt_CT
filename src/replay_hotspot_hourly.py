#!/usr/bin/env python3

"""
Hero Dirt Forecast
Hourly replay of the recent localized MRMS storm.

Purpose
-------
Replay the interval from the latest SMAP observation through
the start of the NWS forecast, using MRMS precipitation one
hour at a time.

This is a DIAGNOSTIC experiment.

The present operational model sums the analysis-period MRMS
precipitation and advances the soil model in one large step.
For short intense convective events, hourly integration is
more physically appropriate because infiltration, drainage,
runoff, and drying evolve during the event.

Meteorology
-----------
We do not yet have archived hourly meteorological forcing for
the analysis period. For this diagnostic, the first NWS
forecast block is used as a constant meteorological proxy.

Thus:

    rainfall timing = observed MRMS hourly
    surface drying meteorology = approximate

Outputs
-------
output/diagnostics/
    hotspot_hourly_replay.csv

output/figures/
    hotspot_hourly_replay.png
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from hero_score import (
    calculate_hero_score,
    CLASS_NAMES,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

STATIC_FILE = (
    ROOT
    / "static"
    / "model"
    / "HeroDirt_static_200m.npz"
)

PLASTICITY_FILE = (
    ROOT
    / "static"
    / "soils"
    / "HeroDirt_plasticity_200m.npz"
)

SMAP_FILE = (
    ROOT
    / "data"
    / "smap"
    / "processed"
    / "HeroDirt_SMAP_latest.npz"
)

MRMS_FILE = (
    ROOT
    / "data"
    / "mrms"
    / "processed"
    / "HeroDirt_MRMS_hourly.npz"
)

NWS_FILE = (
    ROOT
    / "data"
    / "nws"
    / "processed"
    / "HeroDirt_NWS_6hourly.npz"
)

OUT_DIR = (
    ROOT
    / "output"
    / "diagnostics"
)

FIG_DIR = (
    ROOT
    / "output"
    / "figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_FILE = (
    OUT_DIR
    / "hotspot_hourly_replay.csv"
)

FIG_FILE = (
    FIG_DIR
    / "hotspot_hourly_replay.png"
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

ZSURF = 50.0
ZDEEP = 150.0

INFIL_BASE_DAY = 8.0
DEEP_DRAIN_BASE_DAY = 7.0

ET_SURF_BASE_DAY = 1.5
ET_DEEP_BASE_DAY = 0.8


# ============================================================
# LOAD
# ============================================================

S = np.load(STATIC_FILE)
P = np.load(PLASTICITY_FILE)
SMAP = np.load(SMAP_FILE)
MRMS = np.load(MRMS_FILE)
NWS = np.load(NWS_FILE)


x = S["x"].astype(float)
y = S["y"].astype(float)

X, Y = np.meshgrid(
    x,
    y,
)


physics = S[
    "physics_valid"
].astype(bool)

public = S[
    "public_valid"
].astype(bool)


theta_wp = S[
    "theta_wp"
].astype(float)

theta_fc = S[
    "theta_fc"
].astype(float)

theta_sat = S[
    "theta_sat"
].astype(float)

sand = (
    S["sand"].astype(float)
    /
    100.0
)

clay = (
    S["clay"].astype(float)
    /
    100.0
)

ksat = S[
    "ksat"
].astype(float)

slope = S[
    "slope"
].astype(float)

southness = S[
    "southness"
].astype(float)

tree_fraction = S[
    "tree_fraction"
].astype(float)

shrub_fraction = S[
    "shrub_cover_fraction"
].astype(float)

herb_fraction = S[
    "herb_cover_fraction"
].astype(float)

PI = P[
    "plasticity_index"
].astype(float)


# ============================================================
# STATIC HYDROLOGY FACTORS
#
# Identical to run_soil_model.py
# ============================================================

logk = np.full(
    ksat.shape,
    np.nan,
)

good_k = (
    physics
    &
    np.isfinite(ksat)
    &
    (ksat > 0)
)

logk[
    good_k
] = np.log10(
    ksat[
        good_k
    ]
)


klo = np.nanpercentile(
    logk[
        good_k
    ],
    5,
)

khi = np.nanpercentile(
    logk[
        good_k
    ],
    95,
)


Knorm = np.clip(
    (
        logk - klo
    )
    /
    (
        khi - klo
    ),
    0.0,
    1.0,
)


Kfactor = (
    0.6
    +
    0.8
    *
    Knorm
)


texture_factor = (
    1.0
    +
    0.50
    *
    sand
    -
    0.80
    *
    clay
)

texture_factor = np.clip(
    texture_factor,
    0.5,
    1.5,
)


slope_factor = (
    0.8
    +
    0.5
    *
    np.minimum(
        slope,
        45.0,
    )
    /
    45.0
)


aspect_factor = (
    1.0
    +
    0.35
    *
    southness
)

aspect_factor = np.clip(
    aspect_factor,
    0.65,
    1.35,
)


vegetation_factor = (
    1.0
    -
    0.55
    *
    tree_fraction
    -
    0.30
    *
    shrub_fraction
    -
    0.15
    *
    herb_fraction
)

vegetation_factor = np.clip(
    vegetation_factor,
    0.45,
    1.0,
)


# ============================================================
# MRMS
# ============================================================

mrms_time = MRMS[
    "time"
].astype(
    "datetime64[s]"
)

mrms_precip = MRMS[
    "precip_mm"
].astype(float)


# ============================================================
# IDENTIFY RAINFALL HOTSPOT
# ============================================================

region = (
    public
    &
    (X >= 418000)
    &
    (X <= 434000)
    &
    (Y >= 3804000)
    &
    (Y <= 3815000)
)


accum = np.nansum(
    mrms_precip,
    axis=0,
)


candidate = np.where(
    region,
    accum,
    np.nan,
)


flat = np.nanargmax(
    candidate
)


r, c = np.unravel_index(
    flat,
    accum.shape,
)


print()
print(
    "============================================"
)
print(
    " HERO DIRT HOURLY STORM REPLAY"
)
print(
    "============================================"
)
print()


print(
    "Selected rainfall maximum:"
)

print(
    f"  UTM E: "
    f"{X[r,c]:.0f} m"
)

print(
    f"  UTM N: "
    f"{Y[r,c]:.0f} m"
)

print(
    f"  accumulated MRMS: "
    f"{accum[r,c]:.2f} mm"
)

print()


# ============================================================
# INITIAL STATE AT SMAP TIME
#
# Same initialization as operational model.
# ============================================================

smap_mean = float(
    SMAP[
        "mean_soil_moisture"
    ]
)

smap_time = SMAP[
    "observation_time_utc"
].astype(
    "datetime64[s]"
)


fc_mean = np.nanmean(
    theta_fc[
        physics
    ]
)


surface_init = (
    smap_mean
    +
    0.35
    *
    (
        theta_fc[r,c]
        -
        fc_mean
    )
)


surface_init = max(
    surface_init,
    theta_wp[r,c],
)

surface_init = min(
    surface_init,
    theta_sat[r,c],
)


deep_init = (
    theta_wp[r,c]
    +
    0.70
    *
    (
        theta_fc[r,c]
        -
        theta_wp[r,c]
    )
)


deep_init = max(
    deep_init,
    surface_init - 0.01,
)

deep_init = min(
    deep_init,
    theta_sat[r,c],
)


# ============================================================
# FORECAST START
# ============================================================

forecast_times = NWS[
    "time"
].astype(
    "datetime64[s]"
)


forecast_start = (
    forecast_times[0]
    -
    np.timedelta64(
        6,
        "h",
    )
)


# ============================================================
# METEOROLOGICAL PROXY AT HOTSPOT
# ============================================================

temperature = float(
    NWS[
        "temperature_c"
    ][
        0,
        r,
        c
    ]
)

rh = float(
    NWS[
        "relative_humidity"
    ][
        0,
        r,
        c
    ]
)

wind = float(
    NWS[
        "wind_speed_ms"
    ][
        0,
        r,
        c
    ]
)

cloud = float(
    NWS[
        "sky_cover_percent"
    ][
        0,
        r,
        c
    ]
)


print(
    "Meteorological proxy:"
)

print(
    f"  T:     {temperature:.1f} C"
)

print(
    f"  RH:    {rh:.1f}%"
)

print(
    f"  wind:  {wind:.1f} m/s"
)

print(
    f"  cloud: {cloud:.1f}%"
)

print()


# ============================================================
# SCALAR MODEL STEP
# ============================================================

def scalar_step(
    surf,
    deep,
    precip,
    dt_hours,
):

    dt_day = (
        dt_hours
        /
        24.0
    )


    wp = theta_wp[
        r,
        c
    ]

    fc = theta_fc[
        r,
        c
    ]

    sat = theta_sat[
        r,
        c
    ]


    Ws = (
        surf
        *
        ZSURF
    )

    Wd = (
        deep
        *
        ZDEEP
    )


    Ws_wp = wp * ZSURF
    Ws_fc = fc * ZSURF
    Ws_sat = sat * ZSURF

    Wd_wp = wp * ZDEEP
    Wd_fc = fc * ZDEEP
    Wd_sat = sat * ZDEEP


    # --------------------------------------------------------
    # Rain
    # --------------------------------------------------------

    Ws += precip


    runoff = max(
        Ws - Ws_sat,
        0.0,
    )

    Ws = min(
        Ws,
        Ws_sat,
    )


    # --------------------------------------------------------
    # Surface -> deep
    # --------------------------------------------------------

    surface_excess_ratio = (
        (
            Ws - Ws_fc
        )
        /
        (
            Ws_sat - Ws_fc
        )
    )

    surface_excess_ratio = np.clip(
        surface_excess_ratio,
        0.0,
        1.0,
    )


    transfer_capacity = (
        INFIL_BASE_DAY
        *
        dt_day
        *
        Kfactor[r,c]
        *
        texture_factor[r,c]
        *
        slope_factor[r,c]
    )


    transfer = (
        transfer_capacity
        *
        surface_excess_ratio
    )


    transfer = min(
        transfer,
        max(
            Ws - Ws_fc,
            0.0,
        ),
    )


    transfer = min(
        transfer,
        max(
            Wd_sat - Wd,
            0.0,
        ),
    )


    transfer = max(
        transfer,
        0.0,
    )


    Ws -= transfer
    Wd += transfer


    # --------------------------------------------------------
    # Deep drainage
    # --------------------------------------------------------

    deep_excess_ratio = (
        (
            Wd - Wd_fc
        )
        /
        (
            Wd_sat - Wd_fc
        )
    )


    deep_excess_ratio = np.clip(
        deep_excess_ratio,
        0.0,
        1.0,
    )


    drainage = (
        DEEP_DRAIN_BASE_DAY
        *
        dt_day
        *
        Kfactor[r,c]
        *
        texture_factor[r,c]
        *
        deep_excess_ratio
    )


    drainage = min(
        drainage,
        max(
            Wd - Wd_fc,
            0.0,
        ),
    )


    drainage = max(
        drainage,
        0.0,
    )


    Wd -= drainage


    # --------------------------------------------------------
    # Drying
    # --------------------------------------------------------

    Tfactor = np.clip(
        (
            temperature - 5.0
        )
        /
        20.0,
        0.2,
        1.5,
    )


    RHfactor = np.clip(
        1.0
        -
        rh / 100.0,
        0.10,
        1.0,
    )


    wind_factor = np.clip(
        0.7
        +
        0.15
        *
        wind,
        0.7,
        1.4,
    )


    cloud_factor = np.clip(
        1.0
        -
        0.50
        *
        cloud
        /
        100.0,
        0.5,
        1.0,
    )


    moisture_factor = np.clip(
        (
            Ws - Ws_wp
        )
        /
        (
            Ws_fc - Ws_wp
        ),
        0.0,
        1.0,
    )


    ETsurf = (
        ET_SURF_BASE_DAY
        *
        dt_day
        *
        Tfactor
        *
        RHfactor
        *
        wind_factor
        *
        cloud_factor
        *
        aspect_factor[r,c]
        *
        vegetation_factor[r,c]
        *
        moisture_factor
    )


    ETsurf = min(
        ETsurf,
        max(
            Ws - Ws_wp,
            0.0,
        ),
    )


    Ws -= ETsurf


    deep_moisture_factor = np.clip(
        (
            Wd - Wd_wp
        )
        /
        (
            Wd_fc - Wd_wp
        ),
        0.0,
        1.0,
    )


    ETdeep = (
        ET_DEEP_BASE_DAY
        *
        dt_day
        *
        Tfactor
        *
        RHfactor
        *
        wind_factor
        *
        deep_moisture_factor
    )


    ETdeep = min(
        ETdeep,
        max(
            Wd - Wd_wp,
            0.0,
        ),
    )


    Wd -= ETdeep


    surf_new = (
        Ws
        /
        ZSURF
    )


    deep_new = (
        Wd
        /
        ZDEEP
    )


    surf_new = np.clip(
        surf_new,
        wp,
        sat,
    )


    deep_new = np.clip(
        deep_new,
        wp,
        sat,
    )


    return (
        surf_new,
        deep_new,
        transfer,
        drainage,
        ETsurf,
        ETdeep,
        runoff,
    )


# ============================================================
# SCORE ONE CELL
# ============================================================

def score_cell(
    surf,
    deep,
):

    result = calculate_hero_score(

        surf=np.array(
            [[surf]]
        ),

        deep=np.array(
            [[deep]]
        ),

        theta_wp=np.array(
            [[theta_wp[r,c]]]
        ),

        theta_fc=np.array(
            [[theta_fc[r,c]]]
        ),

        theta_sat=np.array(
            [[theta_sat[r,c]]]
        ),

        plasticity_index=np.array(
            [[PI[r,c]]]
        ),

        public_valid=np.array(
            [[True]]
        ),
    )


    return (
        float(
            result[
                "score"
            ][0,0]
        ),

        int(
            result[
                "condition"
            ][0,0]
        ),

        float(
            result[
                "F"
            ][0,0]
        ),
    )


# ============================================================
# STORAGE FOR HISTORY
# ============================================================

history = []


surf = surface_init
deep = deep_init


score, condition, Fstate = score_cell(
    surf,
    deep,
)


history.append(
    {
        "time":
            smap_time,

        "precip":
            0.0,

        "surface_theta":
            surf,

        "deep_theta":
            deep,

        "F":
            Fstate,

        "score":
            score,

        "condition":
            condition,

        "infiltration":
            0.0,

        "drainage":
            0.0,

        "surface_et":
            0.0,

        "runoff":
            0.0,
    }
)


# ============================================================
# INITIAL DRY INTERVAL
#
# SMAP time -> first MRMS timestamp
# ============================================================

valid_indices = np.where(
    (
        mrms_time > smap_time
    )
    &
    (
        mrms_time <= forecast_start
    )
)[0]


if len(
    valid_indices
) == 0:

    raise RuntimeError(
        "No MRMS hours available after SMAP."
    )


first_time = mrms_time[
    valid_indices[0]
]


initial_dt = (
    first_time
    -
    smap_time
).astype(
    "timedelta64[s]"
).astype(
    float
) / 3600.0


# MRMS timestamp is treated as the END of its hourly interval.
#
# Only the final hour before first_time receives the first
# MRMS precipitation value.
#
# Therefore, dry from SMAP time to first_time - 1 h.

dry_dt = max(
    initial_dt - 1.0,
    0.0,
)


if dry_dt > 0:

    (
        surf,
        deep,
        infil,
        drainage,
        ets,
        etd,
        runoff,
    ) = scalar_step(
        surf,
        deep,
        precip=0.0,
        dt_hours=dry_dt,
    )


    score, condition, Fstate = score_cell(
        surf,
        deep,
    )


    history.append(
        {
            "time":
                first_time
                -
                np.timedelta64(
                    1,
                    "h"
                ),

            "precip":
                0.0,

            "surface_theta":
                surf,

            "deep_theta":
                deep,

            "F":
                Fstate,

            "score":
                score,

            "condition":
                condition,

            "infiltration":
                infil,

            "drainage":
                drainage,

            "surface_et":
                ets,

            "runoff":
                runoff,
        }
    )


# ============================================================
# HOURLY MRMS REPLAY
# ============================================================

for idx in valid_indices:

    t = mrms_time[
        idx
    ]


    rain = float(
        mrms_precip[
            idx,
            r,
            c
        ]
    )


    if not np.isfinite(
        rain
    ):

        rain = 0.0


    (
        surf,
        deep,
        infil,
        drainage,
        ets,
        etd,
        runoff,
    ) = scalar_step(

        surf,
        deep,

        precip=rain,

        dt_hours=1.0,
    )


    score, condition, Fstate = score_cell(
        surf,
        deep,
    )


    history.append(
        {
            "time":
                t,

            "precip":
                rain,

            "surface_theta":
                surf,

            "deep_theta":
                deep,

            "F":
                Fstate,

            "score":
                score,

            "condition":
                condition,

            "infiltration":
                infil,

            "drainage":
                drainage,

            "surface_et":
                ets,

            "runoff":
                runoff,
        }
    )


# ============================================================
# FINAL GAP TO FORECAST START
# ============================================================

last_time = history[
    -1
][
    "time"
]


final_dt = (
    forecast_start
    -
    last_time
).astype(
    "timedelta64[s]"
).astype(
    float
) / 3600.0


if final_dt > 0:

    (
        surf,
        deep,
        infil,
        drainage,
        ets,
        etd,
        runoff,
    ) = scalar_step(

        surf,
        deep,

        precip=0.0,

        dt_hours=final_dt,
    )


    score, condition, Fstate = score_cell(
        surf,
        deep,
    )


    history.append(
        {
            "time":
                forecast_start,

            "precip":
                0.0,

            "surface_theta":
                surf,

            "deep_theta":
                deep,

            "F":
                Fstate,

            "score":
                score,

            "condition":
                condition,

            "infiltration":
                infil,

            "drainage":
                drainage,

            "surface_et":
                ets,

            "runoff":
                runoff,
        }
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "time_utc",
            "precip_mm",
            "surface_theta",
            "deep_theta",
            "F",
            "hero_score",
            "hero_class",
            "hero_class_name",
            "surface_to_deep_mm",
            "deep_drainage_mm",
            "surface_et_mm",
            "runoff_mm",
        ]
    )


    for h in history:

        writer.writerow(
            [
                str(
                    h[
                        "time"
                    ]
                ),

                h[
                    "precip"
                ],

                h[
                    "surface_theta"
                ],

                h[
                    "deep_theta"
                ],

                h[
                    "F"
                ],

                h[
                    "score"
                ],

                h[
                    "condition"
                ],

                CLASS_NAMES[
                    h[
                        "condition"
                    ]
                ],

                h[
                    "infiltration"
                ],

                h[
                    "drainage"
                ],

                h[
                    "surface_et"
                ],

                h[
                    "runoff"
                ],
            ]
        )


# ============================================================
# SUMMARY
# ============================================================

print(
    "Hourly replay summary:"
)

print()


print(
    f"Initial surface theta: "
    f"{surface_init:.4f}"
)

print(
    f"Final surface theta:   "
    f"{surf:.4f}"
)

print()


print(
    f"Initial F: "
    f"{history[0]['F']:.3f}"
)

print(
    f"Maximum F: "
    f"{max(h['F'] for h in history):.3f}"
)

print(
    f"Final F:   "
    f"{history[-1]['F']:.3f}"
)

print()


print(
    f"Initial score: "
    f"{history[0]['score']:.1f}"
)

print(
    f"Minimum score: "
    f"{min(h['score'] for h in history):.1f}"
)

print(
    f"Final score:   "
    f"{history[-1]['score']:.1f}"
)

print()


print(
    f"Final class: "
    f"{CLASS_NAMES[history[-1]['condition']]}"
)

print()


print(
    f"Total rain: "
    f"{sum(h['precip'] for h in history):.2f} mm"
)

print(
    f"Total surface->deep: "
    f"{sum(h['infiltration'] for h in history):.2f} mm"
)

print(
    f"Total surface ET: "
    f"{sum(h['surface_et'] for h in history):.2f} mm"
)

print(
    f"Total runoff: "
    f"{sum(h['runoff'] for h in history):.2f} mm"
)

print()


# ============================================================
# PLOT
# ============================================================

plot_times = [
    np.datetime64(
        h[
            "time"
        ]
    )
    for h in history
]


rain = np.array(
    [
        h[
            "precip"
        ]
        for h in history
    ]
)


surf_series = np.array(
    [
        h[
            "surface_theta"
        ]
        for h in history
    ]
)


deep_series = np.array(
    [
        h[
            "deep_theta"
        ]
        for h in history
    ]
)


F_series = np.array(
    [
        h[
            "F"
        ]
        for h in history
    ]
)


score_series = np.array(
    [
        h[
            "score"
        ]
        for h in history
    ]
)


# ------------------------------------------------------------
# Figure 1: rainfall
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(
        12,
        9,
    )
)


ax1 = fig.add_axes(
    [
        0.10,
        0.72,
        0.82,
        0.20,
    ]
)


ax1.bar(
    plot_times,
    rain,
    width=0.025,
)


ax1.set_ylabel(
    "MRMS rain [mm h$^{-1}$]"
)


ax1.set_title(
    "Hero Dirt — Hourly Storm Replay at Rainfall Hotspot"
)


# ------------------------------------------------------------
# Figure 2: soil moisture
# ------------------------------------------------------------

ax2 = fig.add_axes(
    [
        0.10,
        0.43,
        0.82,
        0.20,
    ]
)


ax2.plot(
    plot_times,
    surf_series,
    marker="o",
    label="Surface 0–5 cm",
)


ax2.plot(
    plot_times,
    deep_series,
    marker="o",
    label="Deep 5–20 cm",
)


ax2.axhline(
    theta_fc[
        r,
        c
    ],
    linestyle="--",
    label="Field capacity",
)


ax2.axhline(
    theta_wp[
        r,
        c
    ],
    linestyle=":",
    label="Wilting point",
)


ax2.set_ylabel(
    "Volumetric water content"
)


ax2.legend(
    loc="best"
)


# ------------------------------------------------------------
# Figure 3: F + Hero score
# ------------------------------------------------------------

ax3 = fig.add_axes(
    [
        0.10,
        0.12,
        0.82,
        0.20,
    ]
)


ax3.plot(
    plot_times,
    F_series,
    marker="o",
    label="Hydraulic state F",
)


ax3.axhline(
    1.0,
    linestyle="--",
    label="Field capacity (F=1)",
)


ax3.set_ylabel(
    "Hydraulic state F"
)


ax3.set_xlabel(
    "UTC"
)


ax4 = ax3.twinx()


ax4.plot(
    plot_times,
    score_series,
    marker="s",
    label="Hero score",
)


ax4.set_ylabel(
    "Hero Dirt score"
)


lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()


ax3.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best",
)


fig.savefig(
    FIG_FILE,
    dpi=200,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    f"Saved CSV:"
)

print(
    f"  {CSV_FILE}"
)

print()


print(
    f"Saved figure:"
)

print(
    f"  {FIG_FILE}"
)

print()


print(
    "============================================"
)
print(
    " HOURLY REPLAY COMPLETE"
)
print(
    "============================================"
)
print()
