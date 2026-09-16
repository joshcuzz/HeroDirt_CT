#!/usr/bin/env python3

"""
Hero Dirt Forecast
Three-layer infiltration-capacity sensitivity experiment.

Purpose
-------
Test whether the very high modeled runoff during the recent
Devil's Punchbowl-area convective storm is caused by the
current bucket-style surface routing.

Structure
---------
Tread:
    0-2 cm
    20 mm

Shallow:
    2-5 cm
    30 mm

Deep:
    5-20 cm
    150 mm

MRMS rainfall:
    hourly observations

Internal integration:
    5-minute timestep

Infiltration concept
--------------------
SSURGO ksat_r is assumed to be in micrometers / second.

Conversion:

    1 um/s = 3.6 mm/hour

Define an effective intact-trail conductivity:

    K_eff = efficiency * Ksat

where efficiency accounts for effects not represented by
SSURGO horizon Ksat:

    compaction
    crusting
    rock fragments
    macropores
    trail construction
    exposed bedrock
    etc.

Test efficiencies:

    0.05
    0.10
    0.25
    0.50

Dry soil is allowed a temporarily larger infiltration
capacity:

    fcap = K_eff * [1 + DRY_BOOST * (1 - Se)]

where Se is relative saturation.

This is a simplified diagnostic parameterization, not a full
Green-Ampt infiltration model.

Outputs
-------
output/diagnostics/
    hotspot_infiltration_capacity_sweep.csv

output/figures/
    hotspot_infiltration_capacity.png
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
    / "hotspot_infiltration_capacity_sweep.csv"
)

FIG_FILE = (
    FIG_DIR
    / "hotspot_infiltration_capacity.png"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

EFFICIENCIES = [
    0.05,
    0.10,
    0.25,
    0.50,
]


SUBSTEP_MINUTES = 5.0

SUBSTEP_HOURS = (
    SUBSTEP_MINUTES
    /
    60.0
)


# Dry-soil infiltration enhancement.
#
# At Se=1:
#     multiplier = 1
#
# At Se=0:
#     multiplier = 1 + DRY_BOOST
#
# Diagnostic approximation only.

DRY_BOOST = 2.0


# ============================================================
# LAYERS [mm]
# ============================================================

ZTREAD = 20.0

ZSHALLOW = 30.0

ZDEEP = 150.0


# ============================================================
# OTHER HYDROLOGY PARAMETERS
# ============================================================

# Keep these unchanged from previous three-layer experiment.

SHALLOW_TRANSFER_BASE_DAY = 16.0

DEEP_DRAIN_BASE_DAY = 7.0


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
# STATIC EMPIRICAL FACTORS
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


print()
print(
    "============================================"
)
print(
    " HERO DIRT INFILTRATION-CAPACITY TEST"
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
    f"  MRMS accumulation: "
    f"{accum[r,c]:.2f} mm"
)

print()


print(
    "Local SSURGO Ksat:"
)

print(
    f"  {ksat_um_s:.2f} um/s"
)

print(
    f"  {ksat_mm_h:.2f} mm/h"
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
# INTERNAL MODEL STEP
# ============================================================

def substep(
    tread,
    shallow,
    deep,
    rain_mm,
    dt_hours,
    efficiency,
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
    # RAINFALL
    # ========================================================

    Wt += rain_mm


    # ========================================================
    # EFFECTIVE SATURATION OF TREAD
    # ========================================================

    Se = (
        tread
        -
        WP
    ) / (
        SAT
        -
        WP
    )


    Se = np.clip(
        Se,
        0.0,
        1.0,
    )


    # ========================================================
    # INFILTRATION CAPACITY
    #
    # Ksat-derived scale.
    #
    # Dry soil temporarily receives an enhanced capacity.
    # As soil wets, capacity approaches effective Ksat.
    # ========================================================

    K_eff_mm_h = (
        efficiency
        *
        ksat_mm_h
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


    infiltration_capacity_rate = (
        K_eff_mm_h
        *
        moisture_multiplier
        *
        texture_factor[
            r,
            c
        ]
    )


    infiltration_capacity_mm = (
        infiltration_capacity_rate
        *
        dt_hours
    )


    # Water potentially available above field capacity.
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
        infiltration_capacity_mm,
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
    # SATURATION-EXCESS RUNOFF
    #
    # Only after infiltration opportunity.
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
        SHALLOW_TRANSFER_BASE_DAY
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
        DEEP_DRAIN_BASE_DAY
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
    # DRYING FACTORS
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
    # RETURN THETA
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
        infiltration_capacity_rate,
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
    efficiency,
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
            np.nan,
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


    total_ts = 0.0

    total_sd = 0.0

    total_drain = 0.0

    total_et_t = 0.0

    total_et_s = 0.0

    total_et_d = 0.0

    total_runoff = 0.0


    capacity_rates = []


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
            cap_rate,
        ) = substep(

            tread,
            shallow,
            deep,

            rain_mm=rain_step,

            dt_hours=dt,

            efficiency=efficiency,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et_t += et_t

        total_et_s += et_s

        total_et_d += et_d

        total_runoff += runoff


        capacity_rates.append(
            cap_rate
        )


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
        np.nanmean(
            capacity_rates
        ),
    )


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_experiment(
    efficiency,
):

    tread = initial_tread

    shallow = initial_shallow

    deep = initial_deep


    history = []


    def add_record(
        time,
        rain,
        cap_rate,
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

                "capacity":
                    cap_rate,
            }
        )


    add_record(
        smap_time,
        0.0,
        np.nan,
    )


    total_ts = 0.0

    total_sd = 0.0

    total_runoff = 0.0

    total_drain = 0.0

    total_et = 0.0


    # --------------------------------------------------------
    # Initial dry gap
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
            cap,
        ) = advance_interval(

            tread,
            shallow,
            deep,

            rain_mm=0.0,

            hours=dry_dt,

            efficiency=efficiency,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et += (
            et_t
            +
            et_s
            +
            et_d
        )

        total_runoff += runoff


    # --------------------------------------------------------
    # Hourly MRMS
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
            cap,
        ) = advance_interval(

            tread,
            shallow,
            deep,

            rain_mm=rain,

            hours=1.0,

            efficiency=efficiency,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et += (
            et_t
            +
            et_s
            +
            et_d
        )

        total_runoff += runoff


        add_record(
            mrms_time[
                idx
            ],
            rain,
            cap,
        )


    # --------------------------------------------------------
    # Final dry gap
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
            cap,
        ) = advance_interval(

            tread,
            shallow,
            deep,

            rain_mm=0.0,

            hours=final_dt,

            efficiency=efficiency,
        )


        total_ts += ts

        total_sd += sd

        total_drain += drainage

        total_et += (
            et_t
            +
            et_s
            +
            et_d
        )

        total_runoff += runoff


        add_record(
            forecast_start,
            0.0,
            cap,
        )


    # --------------------------------------------------------
    # Recovery time
    # --------------------------------------------------------

    rain_values = np.array(
        [
            h[
                "rain"
            ]
            for h in history
        ]
    )


    peak_index = int(
        np.argmax(
            rain_values
        )
    )


    peak_time = history[
        peak_index
    ][
        "time"
    ]


    recovery = np.nan


    for h in history[
        peak_index:
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


    # --------------------------------------------------------
    # Final diagnostics
    # --------------------------------------------------------

    score_final, class_final, F_final = (
        score_state(
            tread,
            shallow,
            deep,
        )
    )


    return {
        "efficiency":
            efficiency,

        "Keff":
            efficiency
            *
            ksat_mm_h,

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

        "drainage":
            total_drain,

        "et":
            total_et,

        "final_tread":
            tread,

        "final_shallow":
            shallow,

        "final_deep":
            deep,

        "final_F":
            F_final,

        "final_score":
            score_final,

        "final_class":
            class_final,

        "recovery":
            recovery,

        "history":
            history,
    }


# ============================================================
# RUN SWEEP
# ============================================================

results = []


for efficiency in EFFICIENCIES:

    results.append(
        run_experiment(
            efficiency
        )
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "Results:"
)

print()


print(
    " eff   Keff    runoff  runoff%  "
    "T->S   S->D   finalF  finalH  class       recovery"
)

print(
    "---------------------------------------------------------------"
)


for R in results:

    if np.isfinite(
        R[
            "recovery"
        ]
    ):

        recovery_text = (
            f"{R['recovery']:.1f} h"
        )

    else:

        recovery_text = (
            ">window"
        )


    print(
        f"{R['efficiency']:4.2f} "
        f"{R['Keff']:7.2f} "
        f"{R['runoff']:8.2f} "
        f"{100*R['runoff_fraction']:7.1f}% "
        f"{R['tread_to_shallow']:6.2f} "
        f"{R['shallow_to_deep']:6.2f} "
        f"{R['final_F']:7.2f} "
        f"{R['final_score']:7.1f} "
        f"{CLASS_NAMES[R['final_class']]:11s} "
        f"{recovery_text:>9s}"
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
            "ksat_efficiency",
            "effective_ksat_mm_h",
            "total_runoff_mm",
            "runoff_fraction",
            "tread_to_shallow_mm",
            "shallow_to_deep_mm",
            "deep_drainage_mm",
            "total_et_mm",
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
                    "efficiency"
                ],

                R[
                    "Keff"
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
                    "drainage"
                ],

                R[
                    "et"
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
# PLOT
# ============================================================

eff = np.array(
    [
        R[
            "efficiency"
        ]
        for R in results
    ]
)


runoff_fraction = np.array(
    [
        100
        *
        R[
            "runoff_fraction"
        ]
        for R in results
    ]
)


final_F = np.array(
    [
        R[
            "final_F"
        ]
        for R in results
    ]
)


recovery = np.array(
    [
        R[
            "recovery"
        ]
        for R in results
    ]
)


fig = plt.figure(
    figsize=(
        10,
        9,
    )
)


ax1 = fig.add_axes(
    [
        0.12,
        0.69,
        0.80,
        0.22,
    ]
)


ax1.plot(
    eff,
    runoff_fraction,
    marker="o",
)


ax1.set_ylabel(
    "Runoff fraction [%]"
)


ax1.set_title(
    "Hero Dirt — Ksat-Based Infiltration Capacity Sensitivity"
)


ax1.set_xlabel(
    "Effective Ksat fraction"
)


ax2 = fig.add_axes(
    [
        0.12,
        0.39,
        0.80,
        0.22,
    ]
)


ax2.plot(
    eff,
    final_F,
    marker="o",
)


ax2.axhline(
    0.80,
    linestyle=":",
)


ax2.axhline(
    1.00,
    linestyle="--",
)


ax2.set_ylabel(
    "Final tread F"
)


ax2.set_xlabel(
    "Effective Ksat fraction"
)


ax3 = fig.add_axes(
    [
        0.12,
        0.09,
        0.80,
        0.22,
    ]
)


ax3.plot(
    eff,
    recovery,
    marker="o",
)


ax3.set_ylabel(
    "Hours to leave Wet/Too Wet"
)


ax3.set_xlabel(
    "Effective Ksat fraction"
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
    "Saved:"
)

print(
    f"  {CSV_FILE}"
)

print(
    f"  {FIG_FILE}"
)

print()


print(
    "============================================"
)
print(
    " INFILTRATION TEST COMPLETE"
)
print(
    "============================================"
)
print()
