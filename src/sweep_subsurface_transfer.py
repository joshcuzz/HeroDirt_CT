#!/usr/bin/env python3

"""
Hero Dirt Forecast
Subsurface transfer sensitivity test.

Purpose
-------
Hold tread infiltration capacity fixed and test whether
remaining runoff is controlled by:

    1. shallow (2-5 cm) -> deep (5-20 cm) transfer
    2. drainage out of the deep layer

Fixed assumptions
-----------------
Tread depth:
    20 mm

Shallow depth:
    30 mm

Deep depth:
    150 mm

MRMS:
    hourly observations

Internal timestep:
    5 minutes

Tread infiltration:
    10% of local SSURGO Ksat

Test matrix
-----------
Shallow -> deep transfer:
    16, 32, 64, 128 mm/day

Deep drainage:
    7, 14, 28 mm/day

Outputs
-------
output/diagnostics/
    hotspot_subsurface_transfer_sweep.csv

output/figures/
    hotspot_subsurface_runoff.png
    hotspot_subsurface_recovery.png
    hotspot_subsurface_finalF.png
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
    ROOT
    / "static/model/HeroDirt_static_200m.npz"
)

P = np.load(
    ROOT
    / "static/soils/HeroDirt_plasticity_200m.npz"
)

SMAP = np.load(
    ROOT
    / "data/smap/processed/HeroDirt_SMAP_latest.npz"
)

MRMS = np.load(
    ROOT
    / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)

NWS = np.load(
    ROOT
    / "data/nws/processed/HeroDirt_NWS_6hourly.npz"
)


OUT_DIR = (
    ROOT
    / "output/diagnostics"
)

FIG_DIR = (
    ROOT
    / "output/figures"
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
    / "hotspot_subsurface_transfer_sweep.csv"
)

FIG_RUNOFF = (
    FIG_DIR
    / "hotspot_subsurface_runoff.png"
)

FIG_RECOVERY = (
    FIG_DIR
    / "hotspot_subsurface_recovery.png"
)

FIG_FINALF = (
    FIG_DIR
    / "hotspot_subsurface_finalF.png"
)


# ============================================================
# EXPERIMENT MATRIX
# ============================================================

SHALLOW_TRANSFER_VALUES = [
    16.0,
    32.0,
    64.0,
    128.0,
]

DEEP_DRAIN_VALUES = [
    7.0,
    14.0,
    28.0,
]


# ============================================================
# FIXED TREAD INFILTRATION
# ============================================================

KSAT_EFFICIENCY = 0.10

DRY_BOOST = 2.0


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
# DRYING PARAMETERS [mm/day]
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
# STATIC FACTORS
# ============================================================

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


# ============================================================
# LOCAL KSAT
# ============================================================

ksat_um_s = float(
    ksat[
        r,
        c
    ]
)


ksat_mm_h = (
    ksat_um_s
    *
    3.6
)


Keff_mm_h = (
    KSAT_EFFICIENCY
    *
    ksat_mm_h
)


print()
print(
    "============================================"
)
print(
    " HERO DIRT SUBSURFACE TRANSFER SWEEP"
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

print()


print(
    "Fixed tread infiltration:"
)

print(
    f"  SSURGO Ksat = "
    f"{ksat_mm_h:.2f} mm/h"
)

print(
    f"  efficiency = "
    f"{KSAT_EFFICIENCY:.2f}"
)

print(
    f"  Keff = "
    f"{Keff_mm_h:.2f} mm/h"
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


initial_tread = theta_05_init

initial_shallow = theta_05_init


initial_deep = (
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


initial_deep = np.clip(
    initial_deep,
    wp[r,c],
    sat[r,c],
)


# ============================================================
# MET PROXY
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


# ============================================================
# SCORE
# ============================================================

def score_state(
    tread,
    shallow,
    deep,
):

    subsurface = (
        ZSHALLOW
        *
        shallow
        +
        ZDEEP
        *
        deep
    ) / (
        ZSHALLOW
        +
        ZDEEP
    )


    result = calculate_hero_score(

        surf=np.array(
            [[tread]]
        ),

        deep=np.array(
            [[subsurface]]
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
# ONE INTERNAL STEP
# ============================================================

def substep(
    tread,
    shallow,
    deep,
    rain_mm,
    dt_hours,
    shallow_transfer_day,
    deep_drain_day,
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
    # Storage
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
    # ADD RAIN
    # ========================================================

    Wt += rain_mm


    # ========================================================
    # TREAD INFILTRATION CAPACITY
    # ========================================================

    Se = np.clip(
        (
            tread
            -
            WP
        )
        /
        (
            SAT
            -
            WP
        ),
        0.0,
        1.0,
    )


    moisture_multiplier = (
        1.0
        +
        DRY_BOOST
        *
        (
            1.0
            -
            Se
        )
    )


    infiltration_rate = (
        Keff_mm_h
        *
        moisture_multiplier
        *
        texture_factor[
            r,
            c
        ]
    )


    infiltration_capacity = (
        infiltration_rate
        *
        dt_hours
    )


    tread_available = max(
        Wt
        -
        Wt_fc,
        0.0,
    )


    shallow_space = max(
        Ws_sat
        -
        Ws,
        0.0,
    )


    tread_to_shallow = min(
        infiltration_capacity,
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
    # SURFACE RUNOFF
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
    # SHALLOW -> DEEP
    # ========================================================

    shallow_excess = np.clip(
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
        shallow_transfer_day
        *
        dt_day
        *
        texture_factor[
            r,
            c
        ]
    )


    shallow_to_deep = (
        shallow_capacity
        *
        shallow_excess
    )


    shallow_to_deep = min(
        shallow_to_deep,
        max(
            Ws
            -
            Ws_fc,
            0.0,
        ),
        max(
            Wd_sat
            -
            Wd,
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
        deep_drain_day
        *
        dt_day
        *
        texture_factor[
            r,
            c
        ]
        *
        deep_excess
    )


    deep_drainage = min(
        deep_drainage,
        max(
            Wd
            -
            Wd_fc,
            0.0,
        ),
    )


    deep_drainage = max(
        deep_drainage,
        0.0,
    )


    Wd -= deep_drainage


    # ========================================================
    # MET DRYING FACTORS
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


    # ========================================================
    # TREAD ET
    # ========================================================

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


    # ========================================================
    # SHALLOW ET
    # ========================================================

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


    # ========================================================
    # DEEP ET
    # ========================================================

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
# ADVANCE INTERVAL
# ============================================================

def advance_interval(
    tread,
    shallow,
    deep,
    rain_mm,
    hours,
    shallow_transfer_day,
    deep_drain_day,
):

    if hours <= 0:

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
                hours
                /
                SUBSTEP_HOURS
            )
        ),
    )


    dt = (
        hours
        /
        nsteps
    )


    rain_step = (
        rain_mm
        /
        nsteps
    )


    totals = np.zeros(
        7,
        dtype=float,
    )


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

            shallow_transfer_day=
                shallow_transfer_day,

            deep_drain_day=
                deep_drain_day,
        )


        totals += np.array(
            [
                ts,
                sd,
                drainage,
                et_t,
                et_s,
                et_d,
                runoff,
            ]
        )


    return (
        tread,
        shallow,
        deep,
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals[4],
        totals[5],
        totals[6],
    )


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    shallow_transfer_day,
    deep_drain_day,
):

    tread = initial_tread

    shallow = initial_shallow

    deep = initial_deep


    history = []


    def record(
        time,
        rain,
    ):

        score, condition, F = score_state(
            tread,
            shallow,
            deep,
        )


        history.append(
            {
                "time":
                    time,

                "rain":
                    rain,

                "F":
                    F,

                "score":
                    score,

                "class":
                    condition,
            }
        )


    record(
        smap_time,
        0.0,
    )


    total_ts = 0.0

    total_sd = 0.0

    total_drain = 0.0

    total_et_t = 0.0

    total_et_s = 0.0

    total_et_d = 0.0

    total_runoff = 0.0


    # --------------------------------------------------------
    # Pre-MRMS dry interval
    # --------------------------------------------------------

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

            rain_mm=0.0,

            hours=dry_dt,

            shallow_transfer_day=
                shallow_transfer_day,

            deep_drain_day=
                deep_drain_day,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et_t += et_t

        total_et_s += et_s

        total_et_d += et_d

        total_runoff += runoff


    # --------------------------------------------------------
    # MRMS hours
    # --------------------------------------------------------

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

            rain_mm=rain,

            hours=1.0,

            shallow_transfer_day=
                shallow_transfer_day,

            deep_drain_day=
                deep_drain_day,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et_t += et_t

        total_et_s += et_s

        total_et_d += et_d

        total_runoff += runoff


        record(
            mrms_time[
                idx
            ],
            rain,
        )


    # --------------------------------------------------------
    # Final dry interval
    # --------------------------------------------------------

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

            rain_mm=0.0,

            hours=final_dt,

            shallow_transfer_day=
                shallow_transfer_day,

            deep_drain_day=
                deep_drain_day,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et_t += et_t

        total_et_s += et_s

        total_et_d += et_d

        total_runoff += runoff


        record(
            forecast_start,
            0.0,
        )


    # --------------------------------------------------------
    # Recovery time
    # --------------------------------------------------------

    rain_series = np.array(
        [
            h[
                "rain"
            ]
            for h in history
        ]
    )


    peak_idx = int(
        np.argmax(
            rain_series
        )
    )


    peak_time = history[
        peak_idx
    ][
        "time"
    ]


    recovery = np.nan


    for h in history[
        peak_idx:
    ]:

        if h[
            "class"
        ] < 5:

            recovery = (
                (
                    h[
                        "time"
                    ]
                    -
                    peak_time
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


    final_score, final_class, final_F = (
        score_state(
            tread,
            shallow,
            deep,
        )
    )


    return {
        "shallow_transfer":
            shallow_transfer_day,

        "deep_drain":
            deep_drain_day,

        "runoff":
            total_runoff,

        "runoff_fraction":
            total_runoff
            /
            accum[
                r,
                c
            ],

        "tread_to_shallow":
            total_ts,

        "shallow_to_deep":
            total_sd,

        "deep_drainage":
            total_drain,

        "final_tread":
            tread,

        "final_shallow":
            shallow,

        "final_deep":
            deep,

        "final_F":
            final_F,

        "final_score":
            final_score,

        "final_class":
            final_class,

        "recovery":
            recovery,

        "ET_total":
            total_et_t
            +
            total_et_s
            +
            total_et_d,
    }


# ============================================================
# RUN MATRIX
# ============================================================

results = []


for shallow_transfer in SHALLOW_TRANSFER_VALUES:

    for deep_drain in DEEP_DRAIN_VALUES:

        results.append(
            run_experiment(
                shallow_transfer,
                deep_drain,
            )
        )


# ============================================================
# PRINT
# ============================================================

print(
    "Results:"
)

print()


print(
    " S->D   Drain   runoff runoff%   "
    "T->S   S->Dtot  deepOut  finalF finalH class       recover"
)

print(
    "--------------------------------------------------------------------------"
)


for R in results:

    if np.isfinite(
        R[
            "recovery"
        ]
    ):

        recovery_text = (
            f"{R['recovery']:.1f}h"
        )

    else:

        recovery_text = (
            ">window"
        )


    print(
        f"{R['shallow_transfer']:5.0f} "
        f"{R['deep_drain']:7.0f} "
        f"{R['runoff']:8.2f} "
        f"{100*R['runoff_fraction']:6.1f}% "
        f"{R['tread_to_shallow']:6.2f} "
        f"{R['shallow_to_deep']:8.2f} "
        f"{R['deep_drainage']:8.2f} "
        f"{R['final_F']:7.2f} "
        f"{R['final_score']:6.1f} "
        f"{CLASS_NAMES[R['final_class']]:11s} "
        f"{recovery_text:>8s}"
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
            "shallow_to_deep_rate_mm_day",
            "deep_drain_rate_mm_day",
            "runoff_mm",
            "runoff_fraction",
            "tread_to_shallow_mm",
            "shallow_to_deep_mm",
            "deep_drainage_mm",
            "ET_total_mm",
            "final_tread_theta",
            "final_shallow_theta",
            "final_deep_theta",
            "final_F",
            "final_score",
            "final_class",
            "final_class_name",
            "recovery_hours",
        ]
    )


    for R in results:

        writer.writerow(
            [
                R[
                    "shallow_transfer"
                ],

                R[
                    "deep_drain"
                ],

                R[
                    "runoff"
                ],

                R[
                    "runoff_fraction"
                ],

                R[
                    "tread_to_shallow"
                ],

                R[
                    "shallow_to_deep"
                ],

                R[
                    "deep_drainage"
                ],

                R[
                    "ET_total"
                ],

                R[
                    "final_tread"
                ],

                R[
                    "final_shallow"
                ],

                R[
                    "final_deep"
                ],

                R[
                    "final_F"
                ],

                R[
                    "final_score"
                ],

                R[
                    "final_class"
                ],

                CLASS_NAMES[
                    R[
                        "final_class"
                    ]
                ],

                R[
                    "recovery"
                ],
            ]
        )


# ============================================================
# BUILD MATRICES
# ============================================================

nS = len(
    SHALLOW_TRANSFER_VALUES
)

nD = len(
    DEEP_DRAIN_VALUES
)


runoff_matrix = np.full(
    (
        nD,
        nS,
    ),
    np.nan,
)


recovery_matrix = np.full(
    (
        nD,
        nS,
    ),
    np.nan,
)


F_matrix = np.full(
    (
        nD,
        nS,
    ),
    np.nan,
)


for R in results:

    i = DEEP_DRAIN_VALUES.index(
        R[
            "deep_drain"
        ]
    )

    j = SHALLOW_TRANSFER_VALUES.index(
        R[
            "shallow_transfer"
        ]
    )


    runoff_matrix[
        i,
        j
    ] = (
        100.0
        *
        R[
            "runoff_fraction"
        ]
    )


    recovery_matrix[
        i,
        j
    ] = R[
        "recovery"
    ]


    F_matrix[
        i,
        j
    ] = R[
        "final_F"
    ]


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_matrix(
    matrix,
    title,
    cbar_label,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )


    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
    )


    ax.set_xticks(
        np.arange(
            nS
        )
    )


    ax.set_xticklabels(
        [
            f"{v:.0f}"
            for v in SHALLOW_TRANSFER_VALUES
        ]
    )


    ax.set_yticks(
        np.arange(
            nD
        )
    )


    ax.set_yticklabels(
        [
            f"{v:.0f}"
            for v in DEEP_DRAIN_VALUES
        ]
    )


    ax.set_xlabel(
        "Shallow → deep transfer [mm/day]"
    )


    ax.set_ylabel(
        "Deep drainage [mm/day]"
    )


    ax.set_title(
        title
    )


    cbar = fig.colorbar(
        im,
        ax=ax,
    )


    cbar.set_label(
        cbar_label
    )


    for i in range(
        nD
    ):

        for j in range(
            nS
        ):

            value = matrix[
                i,
                j
            ]


            if np.isfinite(
                value
            ):

                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                )


    fig.tight_layout()


    fig.savefig(
        filename,
        dpi=200,
    )


    plt.close(
        fig
    )


# ============================================================
# PLOTS
# ============================================================

plot_matrix(

    runoff_matrix,

    "Hero Dirt — Runoff Sensitivity",

    "Runoff fraction [%]",

    FIG_RUNOFF,
)


plot_matrix(

    recovery_matrix,

    "Hero Dirt — Wet Recovery Sensitivity",

    "Hours to leave Wet/Too Wet",

    FIG_RECOVERY,
)


plot_matrix(

    F_matrix,

    "Hero Dirt — Final Tread Hydraulic State",

    "Final tread F",

    FIG_FINALF,
)


print(
    "Saved:"
)

print(
    f"  {CSV_FILE}"
)

print(
    f"  {FIG_RUNOFF}"
)

print(
    f"  {FIG_RECOVERY}"
)

print(
    f"  {FIG_FINALF}"
)

print()


print(
    "============================================"
)
print(
    " SUBSURFACE SWEEP COMPLETE"
)
print(
    "============================================"
)
print()
