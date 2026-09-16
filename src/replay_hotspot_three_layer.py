#!/usr/bin/env python3

"""
Hero Dirt Forecast
Three-layer hotspot storm replay.

Experimental structure
----------------------
Tread:
    0-2 cm
    Z = 20 mm

Shallow:
    2-5 cm
    Z = 30 mm

Deep:
    5-20 cm
    Z = 150 mm

Motivation
----------
The existing 0-5 cm layer retains storm water too long for
an exposed rocky trail tread.

This experiment separates a rapidly responding tread layer
from the underlying shallow soil while retaining the same
total 0-5 cm depth represented by SMAP.

SMAP constrains the initial combined near-surface state.

Rain enters the tread first.

Fluxes:
    tread -> shallow
    shallow -> deep
    deep drainage
    tread ET
    shallow ET
    deep ET

This remains a conceptual / empirical model. Transfer-rate
parameters below are calibration parameters, not universal
soil constants.
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

S = np.load(
    ROOT / "static/model/HeroDirt_static_200m.npz"
)

P = np.load(
    ROOT / "static/soils/HeroDirt_plasticity_200m.npz"
)

SMAP = np.load(
    ROOT / "data/smap/processed/HeroDirt_SMAP_latest.npz"
)

MRMS = np.load(
    ROOT / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)

NWS = np.load(
    ROOT / "data/nws/processed/HeroDirt_NWS_6hourly.npz"
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
    / "hotspot_three_layer_replay.csv"
)

FIG_FILE = (
    FIG_DIR
    / "hotspot_three_layer_replay.png"
)


# ============================================================
# LAYER THICKNESSES [mm]
# ============================================================

ZTREAD = 20.0
ZSHALLOW = 30.0
ZDEEP = 150.0


# ============================================================
# EMPIRICAL HYDROLOGY PARAMETERS [mm/day]
# ============================================================

# Rapid drainage from exposed tread into 2-5 cm layer.
TREAD_TRANSFER_BASE_DAY = 48.0

# Slower transfer from 2-5 cm into 5-20 cm.
SHALLOW_TRANSFER_BASE_DAY = 16.0

# Drainage out of deep layer.
DEEP_DRAIN_BASE_DAY = 7.0


# ============================================================
# EMPIRICAL DRYING PARAMETERS [mm/day]
# ============================================================

ET_TREAD_BASE_DAY = 2.5

ET_SHALLOW_BASE_DAY = 0.7

ET_DEEP_BASE_DAY = 0.8


# ============================================================
# STATIC FIELDS
# ============================================================

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


wp = S[
    "theta_wp"
].astype(float)

fc = S[
    "theta_fc"
].astype(float)

sat = S[
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


tree = S[
    "tree_fraction"
].astype(float)

shrub = S[
    "shrub_cover_fraction"
].astype(float)

herb = S[
    "herb_cover_fraction"
].astype(float)


PI = P[
    "plasticity_index"
].astype(float)


# ============================================================
# STATIC EMPIRICAL FACTORS
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


aspect_factor = np.clip(
    1.0
    +
    0.35
    *
    southness,
    0.65,
    1.35,
)


vegetation_factor = np.clip(
    1.0
    -
    0.55
    *
    tree
    -
    0.30
    *
    shrub
    -
    0.15
    *
    herb,
    0.45,
    1.0,
)


# ============================================================
# LOCATE RAINFALL HOTSPOT
# ============================================================

mrms_time = MRMS[
    "time"
].astype(
    "datetime64[s]"
)

mrms_precip = MRMS[
    "precip_mm"
].astype(float)


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
print("============================================")
print(" HERO DIRT THREE-LAYER STORM REPLAY")
print("============================================")
print()

print("Hotspot:")
print(
    f"  E = {X[r,c]:.0f} m"
)
print(
    f"  N = {Y[r,c]:.0f} m"
)
print(
    f"  MRMS accumulation = "
    f"{accum[r,c]:.2f} mm"
)
print()


# ============================================================
# TIMING
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


valid_indices = np.where(
    (
        mrms_time > smap_time
    )
    &
    (
        mrms_time <= forecast_start
    )
)[0]


# ============================================================
# INITIAL STATE
#
# Start tread and shallow layer at same theta so their
# thickness-weighted 0-5 cm mean equals current SMAP-based
# initialization.
# ============================================================

fc_mean = np.nanmean(
    fc[
        physics
    ]
)


theta_05_init = (
    smap_mean
    +
    0.35
    *
    (
        fc[r,c]
        -
        fc_mean
    )
)


theta_05_init = np.clip(
    theta_05_init,
    wp[r,c],
    sat[r,c],
)


tread = theta_05_init
shallow = theta_05_init


deep = (
    wp[r,c]
    +
    0.70
    *
    (
        fc[r,c]
        -
        wp[r,c]
    )
)


deep = np.clip(
    deep,
    wp[r,c],
    sat[r,c],
)


# ============================================================
# METEOROLOGICAL PROXY
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


print("Meteorological proxy:")
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
# SCORE
#
# Rideability is based primarily on tread moisture.
# Deep-state penalty uses a combined shallow/deep proxy.
# ============================================================

def score_state(
    tread_theta,
    shallow_theta,
    deep_theta,
):

    # Subsurface state beneath tread:
    subsurface_theta = (
        (
            ZSHALLOW
            *
            shallow_theta
        )
        +
        (
            ZDEEP
            *
            deep_theta
        )
    ) / (
        ZSHALLOW
        +
        ZDEEP
    )


    result = calculate_hero_score(

        surf=np.array(
            [[tread_theta]]
        ),

        deep=np.array(
            [[subsurface_theta]]
        ),

        theta_wp=np.array(
            [[wp[r,c]]]
        ),

        theta_fc=np.array(
            [[fc[r,c]]]
        ),

        theta_sat=np.array(
            [[sat[r,c]]]
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
# MODEL STEP
# ============================================================

def step(
    tread,
    shallow,
    deep,
    rain,
    dt_hours,
):

    dt_day = (
        dt_hours
        /
        24.0
    )


    WP = wp[r,c]
    FC = fc[r,c]
    SAT = sat[r,c]


    # --------------------------------------------------------
    # Convert to storage [mm]
    # --------------------------------------------------------

    Wt = tread * ZTREAD
    Ws = shallow * ZSHALLOW
    Wd = deep * ZDEEP


    Wt_wp = WP * ZTREAD
    Wt_fc = FC * ZTREAD
    Wt_sat = SAT * ZTREAD

    Ws_wp = WP * ZSHALLOW
    Ws_fc = FC * ZSHALLOW
    Ws_sat = SAT * ZSHALLOW

    Wd_wp = WP * ZDEEP
    Wd_fc = FC * ZDEEP
    Wd_sat = SAT * ZDEEP


    # --------------------------------------------------------
    # Rain hits tread
    # --------------------------------------------------------

    Wt += rain


    runoff = max(
        Wt - Wt_sat,
        0.0,
    )


    Wt = min(
        Wt,
        Wt_sat,
    )


    # ========================================================
    # TREAD -> SHALLOW
    # ========================================================

    tread_excess = np.clip(
        (
            Wt - Wt_fc
        )
        /
        (
            Wt_sat - Wt_fc
        ),
        0.0,
        1.0,
    )


    tread_transfer_capacity = (
        TREAD_TRANSFER_BASE_DAY
        *
        dt_day
        *
        Kfactor[r,c]
        *
        texture_factor[r,c]
        *
        slope_factor[r,c]
    )


    tread_to_shallow = (
        tread_transfer_capacity
        *
        tread_excess
    )


    tread_to_shallow = min(
        tread_to_shallow,
        max(
            Wt - Wt_fc,
            0.0,
        ),
    )


    tread_to_shallow = min(
        tread_to_shallow,
        max(
            Ws_sat - Ws,
            0.0,
        ),
    )


    tread_to_shallow = max(
        tread_to_shallow,
        0.0,
    )


    Wt -= tread_to_shallow
    Ws += tread_to_shallow


    # ========================================================
    # SHALLOW -> DEEP
    # ========================================================

    shallow_excess = np.clip(
        (
            Ws - Ws_fc
        )
        /
        (
            Ws_sat - Ws_fc
        ),
        0.0,
        1.0,
    )


    shallow_transfer_capacity = (
        SHALLOW_TRANSFER_BASE_DAY
        *
        dt_day
        *
        Kfactor[r,c]
        *
        texture_factor[r,c]
    )


    shallow_to_deep = (
        shallow_transfer_capacity
        *
        shallow_excess
    )


    shallow_to_deep = min(
        shallow_to_deep,
        max(
            Ws - Ws_fc,
            0.0,
        ),
    )


    shallow_to_deep = min(
        shallow_to_deep,
        max(
            Wd_sat - Wd,
            0.0,
        ),
    )


    shallow_to_deep = max(
        shallow_to_deep,
        0.0,
    )


    Ws -= shallow_to_deep
    Wd += shallow_to_deep


    # ========================================================
    # DEEP DRAINAGE
    # ========================================================

    deep_excess = np.clip(
        (
            Wd - Wd_fc
        )
        /
        (
            Wd_sat - Wd_fc
        ),
        0.0,
        1.0,
    )


    deep_drainage = (
        DEEP_DRAIN_BASE_DAY
        *
        dt_day
        *
        Kfactor[r,c]
        *
        texture_factor[r,c]
        *
        deep_excess
    )


    deep_drainage = min(
        deep_drainage,
        max(
            Wd - Wd_fc,
            0.0,
        ),
    )


    deep_drainage = max(
        deep_drainage,
        0.0,
    )


    Wd -= deep_drainage


    # ========================================================
    # METEOROLOGICAL DRYING
    # ========================================================

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


    # --------------------------------------------------------
    # Tread ET
    # --------------------------------------------------------

    tread_moisture = np.clip(
        (
            Wt - Wt_wp
        )
        /
        (
            Wt_fc - Wt_wp
        ),
        0.0,
        1.0,
    )


    ET_tread = (
        ET_TREAD_BASE_DAY
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
        tread_moisture
    )


    ET_tread = min(
        ET_tread,
        max(
            Wt - Wt_wp,
            0.0,
        ),
    )


    Wt -= ET_tread


    # --------------------------------------------------------
    # Shallow ET
    # --------------------------------------------------------

    shallow_moisture = np.clip(
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


    ET_shallow = (
        ET_SHALLOW_BASE_DAY
        *
        dt_day
        *
        Tfactor
        *
        RHfactor
        *
        wind_factor
        *
        shallow_moisture
    )


    ET_shallow = min(
        ET_shallow,
        max(
            Ws - Ws_wp,
            0.0,
        ),
    )


    Ws -= ET_shallow


    # --------------------------------------------------------
    # Deep ET
    # --------------------------------------------------------

    deep_moisture = np.clip(
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


    ET_deep = (
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
        deep_moisture
    )


    ET_deep = min(
        ET_deep,
        max(
            Wd - Wd_wp,
            0.0,
        ),
    )


    Wd -= ET_deep


    # ========================================================
    # BACK TO THETA
    # ========================================================

    tread_new = np.clip(
        Wt / ZTREAD,
        WP,
        SAT,
    )


    shallow_new = np.clip(
        Ws / ZSHALLOW,
        WP,
        SAT,
    )


    deep_new = np.clip(
        Wd / ZDEEP,
        WP,
        SAT,
    )


    return (
        tread_new,
        shallow_new,
        deep_new,
        tread_to_shallow,
        shallow_to_deep,
        deep_drainage,
        ET_tread,
        ET_shallow,
        ET_deep,
        runoff,
    )


# ============================================================
# HISTORY
# ============================================================

history = []


def record(
    time,
    rain,
    tread,
    shallow,
    deep,
    ts,
    sd,
    drainage,
    et_t,
    et_s,
    et_d,
    runoff,
):

    score, condition, F = score_state(
        tread,
        shallow,
        deep,
    )


    theta05 = (
        ZTREAD
        *
        tread
        +
        ZSHALLOW
        *
        shallow
    ) / (
        ZTREAD
        +
        ZSHALLOW
    )


    history.append(
        {
            "time":
                time,

            "rain":
                rain,

            "tread":
                tread,

            "shallow":
                shallow,

            "deep":
                deep,

            "theta05":
                theta05,

            "F":
                F,

            "score":
                score,

            "class":
                condition,

            "tread_to_shallow":
                ts,

            "shallow_to_deep":
                sd,

            "drainage":
                drainage,

            "ET_tread":
                et_t,

            "ET_shallow":
                et_s,

            "ET_deep":
                et_d,

            "runoff":
                runoff,
        }
    )


record(
    smap_time,
    0.0,
    tread,
    shallow,
    deep,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


# ============================================================
# PRE-MRMS GAP
# ============================================================

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


dry_dt = max(
    initial_dt - 1.0,
    0.0,
)


if dry_dt > 0:

    (
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        et_t,
        et_s,
        et_d,
        runoff,
    ) = step(
        tread,
        shallow,
        deep,
        rain=0.0,
        dt_hours=dry_dt,
    )


    record(
        first_time
        -
        np.timedelta64(
            1,
            "h",
        ),
        0.0,
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        et_t,
        et_s,
        et_d,
        runoff,
    )


# ============================================================
# HOURLY MRMS
# ============================================================

for idx in valid_indices:

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
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        et_t,
        et_s,
        et_d,
        runoff,
    ) = step(
        tread,
        shallow,
        deep,
        rain=rain,
        dt_hours=1.0,
    )


    record(
        mrms_time[
            idx
        ],
        rain,
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        et_t,
        et_s,
        et_d,
        runoff,
    )


# ============================================================
# FINAL GAP
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
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        et_t,
        et_s,
        et_d,
        runoff,
    ) = step(
        tread,
        shallow,
        deep,
        rain=0.0,
        dt_hours=final_dt,
    )


    record(
        forecast_start,
        0.0,
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        et_t,
        et_s,
        et_d,
        runoff,
    )


# ============================================================
# SUMMARY
# ============================================================

peak_rain_index = int(
    np.argmax(
        [
            h["rain"]
            for h in history
        ]
    )
)


peak_rain_time = history[
    peak_rain_index
][
    "time"
]


recovery_hours = np.nan


for h in history[
    peak_rain_index:
]:

    if h[
        "class"
    ] < 5:

        recovery_hours = (
            (
                h[
                    "time"
                ]
                -
                peak_rain_time
            )
            .astype(
                "timedelta64[s]"
            )
            .astype(
                float
            )
            /
            3600.0
        )

        break


print(
    "Three-layer result:"
)

print()


print(
    f"Initial tread theta: "
    f"{history[0]['tread']:.4f}"
)

print(
    f"Peak tread theta:    "
    f"{max(h['tread'] for h in history):.4f}"
)

print(
    f"Final tread theta:   "
    f"{history[-1]['tread']:.4f}"
)

print()


print(
    f"Final shallow theta: "
    f"{history[-1]['shallow']:.4f}"
)

print(
    f"Final deep theta:    "
    f"{history[-1]['deep']:.4f}"
)

print()


print(
    f"Final combined 0-5 cm theta: "
    f"{history[-1]['theta05']:.4f}"
)

print()


print(
    f"Maximum tread F: "
    f"{max(h['F'] for h in history):.3f}"
)

print(
    f"Final tread F:   "
    f"{history[-1]['F']:.3f}"
)

print()


print(
    f"Minimum Hero score: "
    f"{min(h['score'] for h in history):.1f}"
)

print(
    f"Final Hero score:   "
    f"{history[-1]['score']:.1f}"
)

print(
    f"Final class:        "
    f"{CLASS_NAMES[history[-1]['class']]}"
)

print()


if np.isfinite(
    recovery_hours
):

    print(
        f"Hours after peak rain to leave "
        f"Wet/Too Wet: "
        f"{recovery_hours:.1f}"
    )

else:

    print(
        "Did not leave Wet/Too Wet "
        "during available replay window."
    )


print()


for name in [
    "rain",
    "tread_to_shallow",
    "shallow_to_deep",
    "drainage",
    "ET_tread",
    "ET_shallow",
    "ET_deep",
    "runoff",
]:

    total = sum(
        h[
            name
        ]
        for h in history
    )

    print(
        f"Total {name:18s}: "
        f"{total:6.2f} mm"
    )


print()


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
            "rain_mm",
            "tread_theta",
            "shallow_theta",
            "deep_theta",
            "combined_0_5cm_theta",
            "tread_F",
            "hero_score",
            "hero_class",
            "hero_class_name",
            "tread_to_shallow_mm",
            "shallow_to_deep_mm",
            "deep_drainage_mm",
            "tread_et_mm",
            "shallow_et_mm",
            "deep_et_mm",
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
                h["rain"],
                h["tread"],
                h["shallow"],
                h["deep"],
                h["theta05"],
                h["F"],
                h["score"],
                h["class"],
                CLASS_NAMES[
                    h[
                        "class"
                    ]
                ],
                h["tread_to_shallow"],
                h["shallow_to_deep"],
                h["drainage"],
                h["ET_tread"],
                h["ET_shallow"],
                h["ET_deep"],
                h["runoff"],
            ]
        )


# ============================================================
# PLOT
# ============================================================

times = np.array(
    [
        h["time"]
        for h in history
    ]
)


rain = np.array(
    [
        h["rain"]
        for h in history
    ]
)


tread_series = np.array(
    [
        h["tread"]
        for h in history
    ]
)


shallow_series = np.array(
    [
        h["shallow"]
        for h in history
    ]
)


deep_series = np.array(
    [
        h["deep"]
        for h in history
    ]
)


theta05_series = np.array(
    [
        h["theta05"]
        for h in history
    ]
)


F_series = np.array(
    [
        h["F"]
        for h in history
    ]
)


score_series = np.array(
    [
        h["score"]
        for h in history
    ]
)


fig = plt.figure(
    figsize=(
        12,
        10,
    )
)


# ------------------------------------------------------------
# Rain
# ------------------------------------------------------------

ax1 = fig.add_axes(
    [
        0.10,
        0.76,
        0.82,
        0.17,
    ]
)


ax1.bar(
    times,
    rain,
    width=0.025,
)


ax1.set_ylabel(
    "Rain [mm h$^{-1}$]"
)


ax1.set_title(
    "Hero Dirt — Three-Layer Storm Replay"
)


# ------------------------------------------------------------
# Moisture
# ------------------------------------------------------------

ax2 = fig.add_axes(
    [
        0.10,
        0.49,
        0.82,
        0.19,
    ]
)


ax2.plot(
    times,
    tread_series,
    marker="o",
    label="Tread 0–2 cm",
)


ax2.plot(
    times,
    shallow_series,
    marker="o",
    label="Shallow 2–5 cm",
)


ax2.plot(
    times,
    deep_series,
    marker="o",
    label="Deep 5–20 cm",
)


ax2.plot(
    times,
    theta05_series,
    linestyle="--",
    label="Combined 0–5 cm",
)


ax2.axhline(
    fc[r,c],
    linestyle=":",
    label="Field capacity",
)


ax2.set_ylabel(
    "Volumetric water content"
)


ax2.legend(
    fontsize=8,
    ncol=2,
)


# ------------------------------------------------------------
# F
# ------------------------------------------------------------

ax3 = fig.add_axes(
    [
        0.10,
        0.27,
        0.82,
        0.14,
    ]
)


ax3.plot(
    times,
    F_series,
    marker="o",
)


ax3.axhline(
    1.0,
    linestyle="--",
    label="Field capacity",
)


ax3.axhline(
    0.80,
    linestyle=":",
    label="Wet threshold",
)


ax3.set_ylabel(
    "Tread F"
)


ax3.legend()


# ------------------------------------------------------------
# Hero score
# ------------------------------------------------------------

ax4 = fig.add_axes(
    [
        0.10,
        0.07,
        0.82,
        0.13,
    ]
)


ax4.plot(
    times,
    score_series,
    marker="o",
)


ax4.set_ylim(
    0,
    105,
)


ax4.set_ylabel(
    "Hero score"
)


ax4.set_xlabel(
    "UTC"
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
    " THREE-LAYER REPLAY COMPLETE"
)
print(
    "============================================"
)
print()
