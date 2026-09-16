#!/usr/bin/env python3

"""
Hero Dirt Forecast
Whole-domain tread <-> shallow upward-exchange sensitivity.

Purpose
-------
The new three-layer operational model dries the 0-2 cm tread
too aggressively across the domain because moisture can move
downward but cannot return upward from the wetter 2-5 cm layer.

This experiment adds a mass-conserving upward equilibration
term when:

    theta_shallow > theta_tread

The moisture difference relaxes exponentially with timescale tau.

Test cases
----------
No upward exchange
tau = 6 h
tau = 12 h
tau = 24 h
tau = 48 h

Everything else is held identical to the current three-layer
operational analysis model.

Outputs
-------
output/diagnostics/upward_exchange_sweep.csv

output/figures/upward_exchange_domain_F.png
output/figures/upward_exchange_domain_score.png
output/figures/upward_exchange_hotspot_F.png
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from hero_score import calculate_hero_score


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


OUT_DIR = ROOT / "output/diagnostics"
FIG_DIR = ROOT / "output/figures"

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
    / "upward_exchange_sweep.csv"
)

FIG_F = (
    FIG_DIR
    / "upward_exchange_domain_F.png"
)

FIG_SCORE = (
    FIG_DIR
    / "upward_exchange_domain_score.png"
)

FIG_HOTSPOT = (
    FIG_DIR
    / "upward_exchange_hotspot_F.png"
)


# ============================================================
# TEST CASES
#
# None = no upward exchange
# ============================================================

TAU_VALUES = [
    None,
    6.0,
    12.0,
    24.0,
    48.0,
]


# ============================================================
# LAYERS [mm]
# ============================================================

ZTREAD = 20.0
ZSHALLOW = 30.0
ZDEEP = 150.0


# ============================================================
# HYDROLOGY PARAMETERS
# ============================================================

KSAT_EFFICIENCY = 0.10

DRY_BOOST = 2.0

SHALLOW_TRANSFER_BASE_DAY = 64.0

DEEP_DRAIN_BASE_DAY = 7.0


# ============================================================
# ET PARAMETERS [mm/day]
# ============================================================

ET_TREAD_BASE_DAY = 2.5

ET_SHALLOW_BASE_DAY = 0.7

ET_DEEP_BASE_DAY = 0.8


# ============================================================
# TIMESTEPS
# ============================================================

RAIN_SUBSTEP_HOURS = 5.0 / 60.0

DRY_SUBSTEP_HOURS = 1.0


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
    0.50 * sand
    -
    0.80 * clay
)

texture_factor = np.clip(
    texture_factor,
    0.5,
    1.5,
)


logk = np.full(
    ksat.shape,
    np.nan,
    dtype=float,
)

good_k = (
    physics
    &
    np.isfinite(ksat)
    &
    (ksat > 0)
)

logk[good_k] = np.log10(
    ksat[good_k]
)

klo = np.nanpercentile(
    logk[good_k],
    5,
)

khi = np.nanpercentile(
    logk[good_k],
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
    0.8 * Knorm
)


shallow_transfer_rate = (
    SHALLOW_TRANSFER_BASE_DAY
    *
    Kfactor
    *
    texture_factor
)

shallow_transfer_rate = np.clip(
    shallow_transfer_rate,
    16.0,
    128.0,
)


deep_drain_rate = (
    DEEP_DRAIN_BASE_DAY
    *
    Kfactor
    *
    texture_factor
)

deep_drain_rate = np.clip(
    deep_drain_rate,
    2.0,
    20.0,
)


aspect_factor = np.clip(
    1.0
    +
    0.35 * southness,
    0.65,
    1.35,
)


vegetation_factor = np.clip(
    1.0
    -
    0.55 * tree
    -
    0.30 * shrub
    -
    0.15 * herb,
    0.45,
    1.0,
)


ksat_mm_h = (
    3.6
    *
    ksat
)

Keff_mm_h = (
    KSAT_EFFICIENCY
    *
    ksat_mm_h
)


# ============================================================
# DYNAMIC INPUTS
# ============================================================

mrms_time = MRMS[
    "time"
].astype(
    "datetime64[s]"
)

mrms_precip = MRMS[
    "precip_mm"
].astype(float)


nws_time = NWS[
    "time"
].astype(
    "datetime64[s]"
)

nws_temp = NWS[
    "temperature_c"
].astype(float)

nws_rh = NWS[
    "relative_humidity"
].astype(float)

nws_wind = NWS[
    "wind_speed_ms"
].astype(float)

nws_cloud = NWS[
    "sky_cover_percent"
].astype(float)


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


forecast_start = (
    nws_time[0]
    -
    np.timedelta64(
        6,
        "h",
    )
)


analysis_indices = np.where(
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
# ============================================================

fc_mean = np.nanmean(
    fc[physics]
)


theta05_init = (
    smap_mean
    +
    0.35
    *
    (
        fc
        -
        fc_mean
    )
)


mean_before = np.nanmean(
    theta05_init[physics]
)


theta05_init += (
    smap_mean
    -
    mean_before
)


theta05_init = np.clip(
    theta05_init,
    wp,
    sat,
)


theta05_init[
    ~physics
] = np.nan


initial_tread = (
    theta05_init.copy()
)

initial_shallow = (
    theta05_init.copy()
)


initial_deep = (
    wp
    +
    0.70
    *
    (
        fc
        -
        wp
    )
)


initial_deep = np.maximum(
    initial_deep,
    theta05_init
    -
    0.01,
)


initial_deep = np.clip(
    initial_deep,
    wp,
    sat,
)


initial_deep[
    ~physics
] = np.nan


# ============================================================
# ANALYSIS MET PROXY
# ============================================================

temperature = nws_temp[0]
rh = nws_rh[0]
wind = nws_wind[0]
cloud = nws_cloud[0]


# ============================================================
# HOTSPOT
# ============================================================

accum = np.nansum(
    mrms_precip,
    axis=0,
)


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


candidate = np.where(
    region,
    accum,
    np.nan,
)


flat = np.nanargmax(
    candidate
)


hr, hc = np.unravel_index(
    flat,
    accum.shape,
)


print()
print(
    "============================================"
)
print(
    " HERO DIRT UPWARD-EXCHANGE SWEEP"
)
print(
    "============================================"
)
print()

print(
    f"Hotspot E/N: "
    f"{X[hr,hc]:.0f}, "
    f"{Y[hr,hc]:.0f}"
)

print(
    f"Storm accumulation: "
    f"{accum[hr,hc]:.2f} mm"
)

print()


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

        surf=tread,

        deep=subsurface,

        theta_wp=wp,

        theta_fc=fc,

        theta_sat=sat,

        plasticity_index=PI,

        public_valid=public,
    )


    return (
        result["score"],
        result["condition"],
        result["F"],
    )


# ============================================================
# UPWARD EXCHANGE
# ============================================================

def upward_exchange(
    Wt,
    Ws,
    dt_hours,
    tau_hours,
):

    if tau_hours is None:

        return (
            Wt,
            Ws,
            np.zeros(
                Wt.shape,
                dtype=float,
            ),
        )


    theta_t = (
        Wt
        /
        ZTREAD
    )

    theta_s = (
        Ws
        /
        ZSHALLOW
    )


    delta = np.maximum(
        theta_s
        -
        theta_t,
        0.0,
    )


    # Effective storage depth that gives
    # exponential decay of theta difference.

    Zeq = (
        ZTREAD
        *
        ZSHALLOW
        /
        (
            ZTREAD
            +
            ZSHALLOW
        )
    )


    fraction = (
        1.0
        -
        np.exp(
            -dt_hours
            /
            tau_hours
        )
    )


    Qup = (
        delta
        *
        Zeq
        *
        fraction
    )


    # Never pull shallow below WP.

    shallow_available = np.maximum(
        Ws
        -
        wp
        *
        ZSHALLOW,
        0.0,
    )


    Qup = np.minimum(
        Qup,
        shallow_available,
    )


    # Never exceed tread saturation.

    tread_space = np.maximum(
        sat
        *
        ZTREAD
        -
        Wt,
        0.0,
    )


    Qup = np.minimum(
        Qup,
        tread_space,
    )


    Qup = np.maximum(
        Qup,
        0.0,
    )


    Qup[
        ~physics
    ] = 0.0


    Wt += Qup

    Ws -= Qup


    return (
        Wt,
        Ws,
        Qup,
    )


# ============================================================
# MODEL SUBSTEP
# ============================================================

def model_substep(
    tread,
    shallow,
    deep,
    rain,
    dt_hours,
    tau_hours,
):

    dt_day = (
        dt_hours
        /
        24.0
    )


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
        wp
        *
        ZTREAD
    )

    Wt_fc = (
        fc
        *
        ZTREAD
    )

    Wt_sat = (
        sat
        *
        ZTREAD
    )


    Ws_wp = (
        wp
        *
        ZSHALLOW
    )

    Ws_fc = (
        fc
        *
        ZSHALLOW
    )

    Ws_sat = (
        sat
        *
        ZSHALLOW
    )


    Wd_wp = (
        wp
        *
        ZDEEP
    )

    Wd_fc = (
        fc
        *
        ZDEEP
    )

    Wd_sat = (
        sat
        *
        ZDEEP
    )


    # ========================================================
    # RAIN
    # ========================================================

    rain = np.where(
        np.isfinite(rain),
        np.maximum(
            rain,
            0.0,
        ),
        0.0,
    )


    Wt += rain


    # ========================================================
    # TREAD -> SHALLOW
    # ========================================================

    Se = np.clip(
        (
            tread
            -
            wp
        )
        /
        (
            sat
            -
            wp
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
        texture_factor
    )


    infiltration_capacity = (
        infiltration_rate
        *
        dt_hours
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


    qdown = np.minimum(
        infiltration_capacity,
        tread_available,
    )


    qdown = np.minimum(
        qdown,
        shallow_space,
    )


    qdown = np.maximum(
        qdown,
        0.0,
    )


    qdown[
        ~physics
    ] = 0.0


    Wt -= qdown

    Ws += qdown


    # ========================================================
    # RUNOFF
    # ========================================================

    runoff = np.maximum(
        Wt
        -
        Wt_sat,
        0.0,
    )


    runoff[
        ~physics
    ] = 0.0


    Wt = np.minimum(
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


    qsd = (
        shallow_transfer_rate
        *
        dt_day
        *
        shallow_excess
    )


    qsd = np.minimum(
        qsd,
        np.maximum(
            Ws
            -
            Ws_fc,
            0.0,
        ),
    )


    qsd = np.minimum(
        qsd,
        np.maximum(
            Wd_sat
            -
            Wd,
            0.0,
        ),
    )


    qsd = np.maximum(
        qsd,
        0.0,
    )


    qsd[
        ~physics
    ] = 0.0


    Ws -= qsd

    Wd += qsd


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


    drainage = (
        deep_drain_rate
        *
        dt_day
        *
        deep_excess
    )


    drainage = np.minimum(
        drainage,
        np.maximum(
            Wd
            -
            Wd_fc,
            0.0,
        ),
    )


    drainage = np.maximum(
        drainage,
        0.0,
    )


    drainage[
        ~physics
    ] = 0.0


    Wd -= drainage


    # ========================================================
    # UPWARD TREAD-SHALLOW EXCHANGE
    # ========================================================

    (
        Wt,
        Ws,
        qup,
    ) = upward_exchange(

        Wt,
        Ws,

        dt_hours,

        tau_hours,
    )


    # ========================================================
    # DRYING
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


    et_t = (
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
        aspect_factor
        *
        vegetation_factor
        *
        tread_moisture
    )


    et_t = np.minimum(
        et_t,
        np.maximum(
            Wt
            -
            Wt_wp,
            0.0,
        ),
    )


    et_t[
        ~physics
    ] = 0.0


    Wt -= et_t


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


    et_s = (
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


    et_s = np.minimum(
        et_s,
        np.maximum(
            Ws
            -
            Ws_wp,
            0.0,
        ),
    )


    et_s[
        ~physics
    ] = 0.0


    Ws -= et_s


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


    et_d = (
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


    et_d = np.minimum(
        et_d,
        np.maximum(
            Wd
            -
            Wd_wp,
            0.0,
        ),
    )


    et_d[
        ~physics
    ] = 0.0


    Wd -= et_d


    # ========================================================
    # BACK TO THETA
    # ========================================================

    tread = np.clip(
        Wt / ZTREAD,
        wp,
        sat,
    )


    shallow = np.clip(
        Ws / ZSHALLOW,
        wp,
        sat,
    )


    deep = np.clip(
        Wd / ZDEEP,
        wp,
        sat,
    )


    tread[
        ~physics
    ] = np.nan

    shallow[
        ~physics
    ] = np.nan

    deep[
        ~physics
    ] = np.nan


    return (
        tread,
        shallow,
        deep,
        runoff,
        qup,
    )


# ============================================================
# ADVANCE INTERVAL
# ============================================================

def advance_interval(
    tread,
    shallow,
    deep,
    rain,
    hours,
    tau_hours,
):

    has_rain = (
        np.nanmax(
            rain
        )
        >
        1.0e-8
    )


    if has_rain:

        max_dt = (
            RAIN_SUBSTEP_HOURS
        )

    else:

        max_dt = (
            DRY_SUBSTEP_HOURS
        )


    nsteps = max(
        1,
        int(
            np.ceil(
                hours
                /
                max_dt
            )
        ),
    )


    dt = (
        hours
        /
        nsteps
    )


    rain_step = (
        rain
        /
        nsteps
    )


    total_runoff = np.zeros(
        tread.shape,
        dtype=float,
    )


    total_qup = np.zeros(
        tread.shape,
        dtype=float,
    )


    for _ in range(
        nsteps
    ):

        (
            tread,
            shallow,
            deep,
            runoff,
            qup,
        ) = model_substep(

            tread,
            shallow,
            deep,

            rain_step,

            dt,

            tau_hours,
        )


        total_runoff += runoff

        total_qup += qup


    return (
        tread,
        shallow,
        deep,
        total_runoff,
        total_qup,
    )


# ============================================================
# RUN ONE TAU
# ============================================================

def run_case(
    tau_hours,
):

    tread = initial_tread.copy()

    shallow = initial_shallow.copy()

    deep = initial_deep.copy()


    hotspot_times = []
    hotspot_F = []


    # --------------------------------------------------------
    # Initial dry gap
    # --------------------------------------------------------

    first_time = mrms_time[
        analysis_indices[0]
    ]


    pre_hours = (
        first_time
        -
        smap_time
    ).astype(
        "timedelta64[s]"
    ).astype(
        float
    ) / 3600.0


    pre_hours = max(
        pre_hours
        -
        1.0,
        0.0,
    )


    if pre_hours > 0:

        zero = np.zeros(
            tread.shape,
            dtype=float,
        )


        (
            tread,
            shallow,
            deep,
            _,
            _,
        ) = advance_interval(

            tread,
            shallow,
            deep,

            zero,

            pre_hours,

            tau_hours,
        )


    # --------------------------------------------------------
    # MRMS replay
    # --------------------------------------------------------

    for idx in analysis_indices:

        rain = mrms_precip[
            idx
        ]


        (
            tread,
            shallow,
            deep,
            _,
            _,
        ) = advance_interval(

            tread,
            shallow,
            deep,

            rain,

            1.0,

            tau_hours,
        )


        _, _, F = score_state(
            tread,
            shallow,
            deep,
        )


        hotspot_times.append(
            mrms_time[
                idx
            ]
        )

        hotspot_F.append(
            F[
                hr,
                hc
            ]
        )


    # --------------------------------------------------------
    # Final gap
    # --------------------------------------------------------

    last_time = mrms_time[
        analysis_indices[
            -1
        ]
    ]


    final_hours = (
        forecast_start
        -
        last_time
    ).astype(
        "timedelta64[s]"
    ).astype(
        float
    ) / 3600.0


    if final_hours > 0:

        zero = np.zeros(
            tread.shape,
            dtype=float,
        )


        (
            tread,
            shallow,
            deep,
            _,
            _,
        ) = advance_interval(

            tread,
            shallow,
            deep,

            zero,

            final_hours,

            tau_hours,
        )


    score, condition, F = score_state(
        tread,
        shallow,
        deep,
    )


    return {
        "tau":
            tau_hours,

        "tread":
            tread,

        "shallow":
            shallow,

        "deep":
            deep,

        "F":
            F,

        "score":
            score,

        "condition":
            condition,

        "mean_F":
            np.nanmean(
                F[public]
            ),

        "mean_score":
            np.nanmean(
                score[public]
            ),

        "very_dry_pct":
            100.0
            *
            np.count_nonzero(
                public
                &
                (
                    condition
                    ==
                    1
                )
            )
            /
            np.count_nonzero(
                public
            ),

        "dry_pct":
            100.0
            *
            np.count_nonzero(
                public
                &
                (
                    condition
                    ==
                    2
                )
            )
            /
            np.count_nonzero(
                public
            ),

        "good_pct":
            100.0
            *
            np.count_nonzero(
                public
                &
                (
                    condition
                    ==
                    3
                )
            )
            /
            np.count_nonzero(
                public
            ),

        "hero_pct":
            100.0
            *
            np.count_nonzero(
                public
                &
                (
                    condition
                    ==
                    4
                )
            )
            /
            np.count_nonzero(
                public
            ),

        "hotspot_final_F":
            F[
                hr,
                hc
            ],

        "hotspot_times":
            np.array(
                hotspot_times
            ),

        "hotspot_F":
            np.array(
                hotspot_F
            ),
    }


# ============================================================
# RUN SWEEP
# ============================================================

results = []


for tau in TAU_VALUES:

    print(
        f"Running tau = "
        f"{'none' if tau is None else str(tau)+' h'}"
    )

    results.append(
        run_case(
            tau
        )
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print(
    "Results:"
)

print()

print(
    " tau[h]   meanF  meanH  VeryDry    Dry   Good   Hero  hotspotF"
)

print(
    "---------------------------------------------------------------"
)


for R in results:

    tau_text = (
        "none"
        if R["tau"] is None
        else f"{R['tau']:.0f}"
    )


    print(
        f"{tau_text:>6s} "
        f"{R['mean_F']:7.3f} "
        f"{R['mean_score']:6.1f} "
        f"{R['very_dry_pct']:8.1f}% "
        f"{R['dry_pct']:6.1f}% "
        f"{R['good_pct']:6.1f}% "
        f"{R['hero_pct']:6.1f}% "
        f"{R['hotspot_final_F']:9.3f}"
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
            "tau_hours",
            "mean_F",
            "mean_score",
            "very_dry_percent",
            "dry_percent",
            "moist_good_percent",
            "hero_percent",
            "hotspot_final_F",
        ]
    )


    for R in results:

        writer.writerow(
            [
                (
                    np.nan
                    if R["tau"] is None
                    else R["tau"]
                ),
                R["mean_F"],
                R["mean_score"],
                R["very_dry_pct"],
                R["dry_pct"],
                R["good_pct"],
                R["hero_pct"],
                R["hotspot_final_F"],
            ]
        )


# ============================================================
# DOMAIN PLOTS
# ============================================================

labels = [
    "none"
    if R["tau"] is None
    else f"{R['tau']:.0f} h"
    for R in results
]


meanF = [
    R["mean_F"]
    for R in results
]


meanH = [
    R["mean_score"]
    for R in results
]


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    labels,
    meanF,
    marker="o",
)

ax.set_ylabel(
    "Domain-mean tread F"
)

ax.set_xlabel(
    "Upward equilibration timescale"
)

ax.set_title(
    "Hero Dirt — Domain Moisture vs Upward Exchange"
)

ax.axhline(
    0.20,
    linestyle=":",
)

ax.axhline(
    0.40,
    linestyle="--",
)

fig.tight_layout()

fig.savefig(
    FIG_F,
    dpi=200,
)

plt.close(
    fig
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    labels,
    meanH,
    marker="o",
)

ax.set_ylabel(
    "Domain-mean Hero Dirt score"
)

ax.set_xlabel(
    "Upward equilibration timescale"
)

ax.set_title(
    "Hero Dirt — Domain Score vs Upward Exchange"
)

fig.tight_layout()

fig.savefig(
    FIG_SCORE,
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# HOTSPOT PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


for R in results:

    label = (
        "No upward exchange"
        if R["tau"] is None
        else f"tau = {R['tau']:.0f} h"
    )


    ax.plot(
        R[
            "hotspot_times"
        ],
        R[
            "hotspot_F"
        ],
        marker="o",
        label=label,
    )


ax.axhline(
    0.80,
    linestyle=":",
    label="Wet threshold",
)

ax.axhline(
    1.00,
    linestyle="--",
    label="Field capacity",
)


ax.set_ylabel(
    "Hotspot tread F"
)

ax.set_xlabel(
    "UTC"
)

ax.set_title(
    "Hero Dirt — Devil's Punchbowl Storm Recovery"
)

ax.legend(
    fontsize=8,
)

fig.tight_layout()

fig.savefig(
    FIG_HOTSPOT,
    dpi=200,
)

plt.close(
    fig
)


print()
print(
    "Saved:"
)

print(
    f"  {CSV_FILE}"
)

print(
    f"  {FIG_F}"
)

print(
    f"  {FIG_SCORE}"
)

print(
    f"  {FIG_HOTSPOT}"
)

print()

print(
    "============================================"
)
print(
    " UPWARD-EXCHANGE SWEEP COMPLETE"
)
print(
    "============================================"
)
print()
