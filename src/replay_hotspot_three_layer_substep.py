#!/usr/bin/env python3

"""
Hero Dirt Forecast
Three-layer hotspot storm replay with sub-hourly integration.

Layers
------
Tread:
    0-2 cm
    Z = 20 mm

Shallow:
    2-5 cm
    Z = 30 mm

Deep:
    5-20 cm
    Z = 150 mm

Important numerical change
--------------------------
MRMS remains hourly, but each hourly accumulation is distributed
uniformly over 5-minute internal timesteps.

At each 5-minute step:

    rainfall enters tread
    -> tread-to-shallow transfer
    -> remaining saturation excess becomes runoff
    -> shallow-to-deep transfer
    -> deep drainage
    -> ET / drying

This avoids dumping an entire convective-hour accumulation into
the 2-cm tread reservoir instantaneously.

This is still a diagnostic experiment. The hydrology coefficients
remain empirical Hero Dirt calibration parameters.
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
    / "hotspot_three_layer_substep_replay.csv"
)

FIG_FILE = (
    FIG_DIR
    / "hotspot_three_layer_substep_replay.png"
)


# ============================================================
# INTERNAL TIMESTEP
# ============================================================

SUBSTEP_MINUTES = 5.0

SUBSTEP_HOURS = (
    SUBSTEP_MINUTES
    /
    60.0
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

TREAD_TRANSFER_BASE_DAY = 48.0

SHALLOW_TRANSFER_BASE_DAY = 16.0

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
    S[
        "sand"
    ].astype(float)
    /
    100.0
)


clay = (
    S[
        "clay"
    ].astype(float)
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
    np.isfinite(
        ksat
    )
    &
    (
        ksat > 0
    )
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
        logk
        -
        klo
    )
    /
    (
        khi
        -
        klo
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
# MRMS HOTSPOT
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
    (
        X >= 418000
    )
    &
    (
        X <= 434000
    )
    &
    (
        Y >= 3804000
    )
    &
    (
        Y <= 3815000
    )
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
    " HERO DIRT THREE-LAYER SUBSTEP REPLAY"
)
print(
    "============================================"
)
print()


print(
    "Hotspot:"
)


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


print(
    f"  internal timestep = "
    f"{SUBSTEP_MINUTES:.0f} min"
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
    forecast_times[
        0
    ]
    -
    np.timedelta64(
        6,
        "h",
    )
)


valid_indices = np.where(
    (
        mrms_time
        >
        smap_time
    )
    &
    (
        mrms_time
        <=
        forecast_start
    )
)[0]


if len(
    valid_indices
) == 0:

    raise RuntimeError(
        "No MRMS fields in analysis interval."
    )


# ============================================================
# INITIAL STATE
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


print(
    "Meteorological proxy:"
)


print(
    f"  T:     "
    f"{temperature:.1f} C"
)


print(
    f"  RH:    "
    f"{rh:.1f}%"
)


print(
    f"  wind:  "
    f"{wind:.1f} m/s"
)


print(
    f"  cloud: "
    f"{cloud:.1f}%"
)


print()


# ============================================================
# SCORE
# ============================================================

def score_state(
    tread_theta,
    shallow_theta,
    deep_theta,
):

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
            [
                [
                    tread_theta
                ]
            ]
        ),

        deep=np.array(
            [
                [
                    subsurface_theta
                ]
            ]
        ),

        theta_wp=np.array(
            [
                [
                    wp[r,c]
                ]
            ]
        ),

        theta_fc=np.array(
            [
                [
                    fc[r,c]
                ]
            ]
        ),

        theta_sat=np.array(
            [
                [
                    sat[r,c]
                ]
            ]
        ),

        plasticity_index=np.array(
            [
                [
                    PI[r,c]
                ]
            ]
        ),

        public_valid=np.array(
            [
                [
                    True
                ]
            ]
        ),
    )


    return (
        float(
            result[
                "score"
            ][
                0,
                0
            ]
        ),

        int(
            result[
                "condition"
            ][
                0,
                0
            ]
        ),

        float(
            result[
                "F"
            ][
                0,
                0
            ]
        ),
    )


# ============================================================
# ONE INTERNAL SUBSTEP
# ============================================================

def substep(
    tread,
    shallow,
    deep,
    rain_mm,
    dt_hours,
):

    dt_day = (
        dt_hours
        /
        24.0
    )


    WP = wp[
        r,
        c
    ]

    FC = fc[
        r,
        c
    ]

    SAT = sat[
        r,
        c
    ]


    # --------------------------------------------------------
    # Storage [mm]
    # --------------------------------------------------------

    Wt = (
        tread
        *
        ZTREAD
    )


    Ws = (
        shallow
        *
        ZSHALLOW
    )


    Wd = (
        deep
        *
        ZDEEP
    )


    Wt_wp = (
        WP
        *
        ZTREAD
    )

    Wt_fc = (
        FC
        *
        ZTREAD
    )

    Wt_sat = (
        SAT
        *
        ZTREAD
    )


    Ws_wp = (
        WP
        *
        ZSHALLOW
    )

    Ws_fc = (
        FC
        *
        ZSHALLOW
    )

    Ws_sat = (
        SAT
        *
        ZSHALLOW
    )


    Wd_wp = (
        WP
        *
        ZDEEP
    )

    Wd_fc = (
        FC
        *
        ZDEEP
    )

    Wd_sat = (
        SAT
        *
        ZDEEP
    )


    # ========================================================
    # 1. ADD RAINFALL TO TREAD
    #
    # Do NOT immediately cap at saturation.
    # Give water a chance to transfer downward first.
    # ========================================================

    Wt += rain_mm


    # ========================================================
    # 2. TREAD -> SHALLOW TRANSFER
    # ========================================================

    tread_excess_ratio = np.clip(
        (
            Wt
            -
            Wt_fc
        )
        /
        (
            Wt_sat
            -
            Wt_fc
        ),
        0.0,
        1.0,
    )


    tread_capacity = (
        TREAD_TRANSFER_BASE_DAY
        *
        dt_day
        *
        Kfactor[
            r,
            c
        ]
        *
        texture_factor[
            r,
            c
        ]
        *
        slope_factor[
            r,
            c
        ]
    )


    tread_to_shallow = (
        tread_capacity
        *
        tread_excess_ratio
    )


    tread_available = np.maximum(
        Wt
        -
        Wt_fc,
        0.0,
    )


    shallow_space = np.maximum(
        Ws_sat
        -
        Ws,
        0.0,
    )


    tread_to_shallow = min(
        tread_to_shallow,
        tread_available,
        shallow_space,
    )


    tread_to_shallow = max(
        tread_to_shallow,
        0.0,
    )


    Wt -= tread_to_shallow

    Ws += tread_to_shallow


    # ========================================================
    # 3. SATURATION-EXCESS RUNOFF
    #
    # Only after vertical transfer opportunity.
    # ========================================================

    runoff = max(
        Wt
        -
        Wt_sat,
        0.0,
    )


    Wt = min(
        Wt,
        Wt_sat,
    )


    # ========================================================
    # 4. SHALLOW -> DEEP TRANSFER
    # ========================================================

    shallow_excess_ratio = np.clip(
        (
            Ws
            -
            Ws_fc
        )
        /
        (
            Ws_sat
            -
            Ws_fc
        ),
        0.0,
        1.0,
    )


    shallow_capacity = (
        SHALLOW_TRANSFER_BASE_DAY
        *
        dt_day
        *
        Kfactor[
            r,
            c
        ]
        *
        texture_factor[
            r,
            c
        ]
    )


    shallow_to_deep = (
        shallow_capacity
        *
        shallow_excess_ratio
    )


    shallow_available = np.maximum(
        Ws
        -
        Ws_fc,
        0.0,
    )


    deep_space = np.maximum(
        Wd_sat
        -
        Wd,
        0.0,
    )


    shallow_to_deep = min(
        shallow_to_deep,
        shallow_available,
        deep_space,
    )


    shallow_to_deep = max(
        shallow_to_deep,
        0.0,
    )


    Ws -= shallow_to_deep

    Wd += shallow_to_deep


    # ========================================================
    # 5. DEEP DRAINAGE
    # ========================================================

    deep_excess_ratio = np.clip(
        (
            Wd
            -
            Wd_fc
        )
        /
        (
            Wd_sat
            -
            Wd_fc
        ),
        0.0,
        1.0,
    )


    deep_drainage = (
        DEEP_DRAIN_BASE_DAY
        *
        dt_day
        *
        Kfactor[
            r,
            c
        ]
        *
        texture_factor[
            r,
            c
        ]
        *
        deep_excess_ratio
    )


    deep_available = max(
        Wd
        -
        Wd_fc,
        0.0,
    )


    deep_drainage = min(
        deep_drainage,
        deep_available,
    )


    deep_drainage = max(
        deep_drainage,
        0.0,
    )


    Wd -= deep_drainage


    # ========================================================
    # 6. METEOROLOGICAL DRYING
    # ========================================================

    Tfactor = np.clip(
        (
            temperature
            -
            5.0
        )
        /
        20.0,
        0.2,
        1.5,
    )


    RHfactor = np.clip(
        1.0
        -
        rh
        /
        100.0,
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
            Wt
            -
            Wt_wp
        )
        /
        (
            Wt_fc
            -
            Wt_wp
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
        aspect_factor[
            r,
            c
        ]
        *
        vegetation_factor[
            r,
            c
        ]
        *
        tread_moisture
    )


    ET_tread = min(
        ET_tread,
        max(
            Wt
            -
            Wt_wp,
            0.0,
        ),
    )


    Wt -= ET_tread


    # --------------------------------------------------------
    # Shallow ET
    # --------------------------------------------------------

    shallow_moisture = np.clip(
        (
            Ws
            -
            Ws_wp
        )
        /
        (
            Ws_fc
            -
            Ws_wp
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
            Ws
            -
            Ws_wp,
            0.0,
        ),
    )


    Ws -= ET_shallow


    # --------------------------------------------------------
    # Deep ET
    # --------------------------------------------------------

    deep_moisture = np.clip(
        (
            Wd
            -
            Wd_wp
        )
        /
        (
            Wd_fc
            -
            Wd_wp
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
            Wd
            -
            Wd_wp,
            0.0,
        ),
    )


    Wd -= ET_deep


    # ========================================================
    # BACK TO THETA
    # ========================================================

    tread_new = np.clip(
        Wt
        /
        ZTREAD,
        WP,
        SAT,
    )


    shallow_new = np.clip(
        Ws
        /
        ZSHALLOW,
        WP,
        SAT,
    )


    deep_new = np.clip(
        Wd
        /
        ZDEEP,
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
# ADVANCE AN ARBITRARY INTERVAL
# ============================================================

def advance_interval(
    tread,
    shallow,
    deep,
    total_rain_mm,
    total_hours,
):

    """
    Integrate an interval using <=5-minute internal timesteps.

    Rain is distributed uniformly across the interval.
    """

    if total_hours <= 0:

        return (
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


    nsteps = max(
        1,
        int(
            np.ceil(
                total_hours
                /
                SUBSTEP_HOURS
            )
        ),
    )


    dt = (
        total_hours
        /
        nsteps
    )


    rain_step = (
        total_rain_mm
        /
        nsteps
    )


    total_ts = 0.0

    total_sd = 0.0

    total_drain = 0.0

    total_et_t = 0.0

    total_et_s = 0.0

    total_et_d = 0.0

    total_runoff = 0.0


    for _ in range(
        nsteps
    ):

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
        ) = substep(

            tread,
            shallow,
            deep,

            rain_mm=rain_step,

            dt_hours=dt,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et_t += et_t

        total_et_s += et_s

        total_et_d += et_d

        total_runoff += runoff


    return (
        tread,
        shallow,
        deep,
        total_ts,
        total_sd,
        total_drain,
        total_et_t,
        total_et_s,
        total_et_d,
        total_runoff,
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
        (
            ZTREAD
            *
            tread
        )
        +
        (
            ZSHALLOW
            *
            shallow
        )
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
# GAP BEFORE FIRST MRMS HOUR
# ============================================================

first_time = mrms_time[
    valid_indices[
        0
    ]
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
    initial_dt
    -
    1.0,
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
    ) = advance_interval(

        tread,
        shallow,
        deep,

        total_rain_mm=0.0,

        total_hours=dry_dt,
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
# HOURLY MRMS REPLAY
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
    ) = advance_interval(

        tread,
        shallow,
        deep,

        total_rain_mm=rain,

        total_hours=1.0,
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
    ) = advance_interval(

        tread,
        shallow,
        deep,

        total_rain_mm=0.0,

        total_hours=final_dt,
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
# RECOVERY TIME
# ============================================================

rain_values = np.array(
    [
        h[
            "rain"
        ]
        for h in history
    ]
)


peak_rain_index = int(
    np.argmax(
        rain_values
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


# ============================================================
# SUMMARY
# ============================================================

print(
    "Substep three-layer result:"
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
        "during replay window."
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
# MASS-BALANCE CHECK
# ============================================================

initial_storage = (
    history[
        0
    ][
        "tread"
    ]
    *
    ZTREAD
    +
    history[
        0
    ][
        "shallow"
    ]
    *
    ZSHALLOW
    +
    history[
        0
    ][
        "deep"
    ]
    *
    ZDEEP
)


final_storage = (
    history[
        -1
    ][
        "tread"
    ]
    *
    ZTREAD
    +
    history[
        -1
    ][
        "shallow"
    ]
    *
    ZSHALLOW
    +
    history[
        -1
    ][
        "deep"
    ]
    *
    ZDEEP
)


total_rain = sum(
    h[
        "rain"
    ]
    for h in history
)


total_runoff = sum(
    h[
        "runoff"
    ]
    for h in history
)


total_et = sum(
    h[
        "ET_tread"
    ]
    +
    h[
        "ET_shallow"
    ]
    +
    h[
        "ET_deep"
    ]
    for h in history
)


total_drain = sum(
    h[
        "drainage"
    ]
    for h in history
)


storage_change = (
    final_storage
    -
    initial_storage
)


residual = (
    total_rain
    -
    total_runoff
    -
    total_et
    -
    total_drain
    -
    storage_change
)


print(
    "Water-balance check:"
)


print(
    f"  initial storage: "
    f"{initial_storage:.3f} mm"
)


print(
    f"  final storage:   "
    f"{final_storage:.3f} mm"
)


print(
    f"  storage change:  "
    f"{storage_change:.3f} mm"
)


print(
    f"  residual:        "
    f"{residual:.6f} mm"
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

                h[
                    "rain"
                ],

                h[
                    "tread"
                ],

                h[
                    "shallow"
                ],

                h[
                    "deep"
                ],

                h[
                    "theta05"
                ],

                h[
                    "F"
                ],

                h[
                    "score"
                ],

                h[
                    "class"
                ],

                CLASS_NAMES[
                    h[
                        "class"
                    ]
                ],

                h[
                    "tread_to_shallow"
                ],

                h[
                    "shallow_to_deep"
                ],

                h[
                    "drainage"
                ],

                h[
                    "ET_tread"
                ],

                h[
                    "ET_shallow"
                ],

                h[
                    "ET_deep"
                ],

                h[
                    "runoff"
                ],
            ]
        )


# ============================================================
# PLOT
# ============================================================

times = np.array(
    [
        h[
            "time"
        ]
        for h in history
    ]
)


rain_series = np.array(
    [
        h[
            "rain"
        ]
        for h in history
    ]
)


tread_series = np.array(
    [
        h[
            "tread"
        ]
        for h in history
    ]
)


shallow_series = np.array(
    [
        h[
            "shallow"
        ]
        for h in history
    ]
)


deep_series = np.array(
    [
        h[
            "deep"
        ]
        for h in history
    ]
)


theta05_series = np.array(
    [
        h[
            "theta05"
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
    rain_series,
    width=0.025,
)


ax1.set_ylabel(
    "MRMS rain [mm h$^{-1}$]"
)


ax1.set_title(
    "Hero Dirt — Three-Layer Storm Replay "
    "(5-minute internal timestep)"
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
    fc[
        r,
        c
    ],
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
# Hydraulic state
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
    "Hero Dirt score"
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
    "Saved CSV:"
)


print(
    f"  {CSV_FILE}"
)


print()


print(
    "Saved figure:"
)


print(
    f"  {FIG_FILE}"
)


print()


print(
    "============================================"
)
print(
    " SUBSTEP REPLAY COMPLETE"
)
print(
    "============================================"
)
print()
