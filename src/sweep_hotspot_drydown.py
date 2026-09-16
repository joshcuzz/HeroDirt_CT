#!/usr/bin/env python3

"""
Hero Dirt Forecast
Hotspot dry-down sensitivity sweep.

Tests sensitivity to:

    surface effective storage depth ZSURF
    surface-to-deep transfer baseline INFIL_BASE_DAY

Experiments
-----------
ZSURF:
    20, 30, 50 mm

INFIL_BASE_DAY:
    8, 16, 24 mm/day

Everything else is held fixed.

Purpose
-------
Determine whether the persistence of Wet / Too Wet conditions
after the localized convective storm is controlled primarily by:

    1. effective surface storage depth
    2. surface-to-deep transfer / drainage rate

Outputs
-------
output/diagnostics/
    hotspot_drydown_sweep.csv

output/figures/
    hotspot_drydown_F.png
    hotspot_drydown_score.png
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
    / "hotspot_drydown_sweep.csv"
)

FIG_F = (
    FIG_DIR
    / "hotspot_drydown_F.png"
)

FIG_SCORE = (
    FIG_DIR
    / "hotspot_drydown_score.png"
)


# ============================================================
# SWEEP
# ============================================================

ZSURF_VALUES = [
    20.0,
    30.0,
    50.0,
]

INFIL_VALUES = [
    8.0,
    16.0,
    24.0,
]


# Fixed deep-layer depth
ZDEEP = 150.0


# Fixed empirical rates
DEEP_DRAIN_BASE_DAY = 7.0
ET_SURF_BASE_DAY = 1.5
ET_DEEP_BASE_DAY = 0.8


# ============================================================
# LOAD DATA
# ============================================================

S = np.load(
    STATIC_FILE
)

P = np.load(
    PLASTICITY_FILE
)

SMAP = np.load(
    SMAP_FILE
)

MRMS = np.load(
    MRMS_FILE
)

NWS = np.load(
    NWS_FILE
)


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


print()
print(
    "============================================"
)
print(
    " HERO DIRT HOTSPOT DRYDOWN SWEEP"
)
print(
    "============================================"
)
print()


print(
    f"Hotspot:"
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


# ============================================================
# TIME WINDOW
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


if len(valid_indices) == 0:

    raise RuntimeError(
        "No MRMS data in analysis interval."
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
# INITIAL THETA
#
# Same as operational model.
# ============================================================

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


surface_init = np.clip(
    surface_init,
    theta_wp[r,c],
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
# ONE SCALAR STEP
# ============================================================

def scalar_step(
    surf,
    deep,
    precip,
    dt_hours,
    ZSURF,
    INFIL_BASE_DAY,
):

    dt_day = (
        dt_hours
        /
        24.0
    )


    wp = theta_wp[r,c]
    fc = theta_fc[r,c]
    sat = theta_sat[r,c]


    Ws = surf * ZSURF
    Wd = deep * ZDEEP


    Ws_wp = wp * ZSURF
    Ws_fc = fc * ZSURF
    Ws_sat = sat * ZSURF


    Wd_wp = wp * ZDEEP
    Wd_fc = fc * ZDEEP
    Wd_sat = sat * ZDEEP


    # --------------------------------------------------------
    # Add rainfall
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
    # Surface -> deep transfer
    # --------------------------------------------------------

    excess_ratio = np.clip(
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
        excess_ratio
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

    deep_excess_ratio = np.clip(
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
    # Atmospheric drying
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


    # --------------------------------------------------------
    # Deep ET
    # --------------------------------------------------------

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


    surf_new = np.clip(
        Ws / ZSURF,
        wp,
        sat,
    )


    deep_new = np.clip(
        Wd / ZDEEP,
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
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    ZSURF,
    INFIL_BASE_DAY,
):

    surf = surface_init
    deep = deep_init


    times_out = []
    F_out = []
    score_out = []
    class_out = []


    total_transfer = 0.0
    total_runoff = 0.0
    total_et = 0.0
    total_drainage = 0.0


    score0, class0, F0 = score_cell(
        surf,
        deep,
    )


    times_out.append(
        smap_time
    )

    F_out.append(
        F0
    )

    score_out.append(
        score0
    )

    class_out.append(
        class0
    )


    # --------------------------------------------------------
    # Dry period before first MRMS hour
    # --------------------------------------------------------

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
            surf,
            deep,
            transfer,
            drainage,
            ets,
            etd,
            runoff,
        ) = scalar_step(

            surf,
            deep,

            precip=0.0,

            dt_hours=dry_dt,

            ZSURF=ZSURF,

            INFIL_BASE_DAY=INFIL_BASE_DAY,
        )


        total_transfer += transfer
        total_drainage += drainage
        total_et += ets
        total_runoff += runoff


    # --------------------------------------------------------
    # Hourly rainfall
    # --------------------------------------------------------

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
            transfer,
            drainage,
            ets,
            etd,
            runoff,
        ) = scalar_step(

            surf,
            deep,

            precip=rain,

            dt_hours=1.0,

            ZSURF=ZSURF,

            INFIL_BASE_DAY=INFIL_BASE_DAY,
        )


        total_transfer += transfer
        total_drainage += drainage
        total_et += ets
        total_runoff += runoff


        score, Hclass, Fstate = score_cell(
            surf,
            deep,
        )


        times_out.append(
            t
        )

        F_out.append(
            Fstate
        )

        score_out.append(
            score
        )

        class_out.append(
            Hclass
        )


    # --------------------------------------------------------
    # Final gap
    # --------------------------------------------------------

    last_time = times_out[
        -1
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
            transfer,
            drainage,
            ets,
            etd,
            runoff,
        ) = scalar_step(

            surf,
            deep,

            precip=0.0,

            dt_hours=final_dt,

            ZSURF=ZSURF,

            INFIL_BASE_DAY=INFIL_BASE_DAY,
        )


        total_transfer += transfer
        total_drainage += drainage
        total_et += ets
        total_runoff += runoff


        score, Hclass, Fstate = score_cell(
            surf,
            deep,
        )


        times_out.append(
            forecast_start
        )

        F_out.append(
            Fstate
        )

        score_out.append(
            score
        )

        class_out.append(
            Hclass
        )


    times_out = np.array(
        times_out
    )

    F_out = np.array(
        F_out
    )

    score_out = np.array(
        score_out
    )

    class_out = np.array(
        class_out
    )


    # --------------------------------------------------------
    # Peak rain time
    # --------------------------------------------------------

    peak_rain_idx = valid_indices[
        np.nanargmax(
            mrms_precip[
                valid_indices,
                r,
                c
            ]
        )
    ]


    peak_rain_time = mrms_time[
        peak_rain_idx
    ]


    # --------------------------------------------------------
    # Time until no longer Wet / Too Wet
    #
    # Wet = class 5
    # Too Wet = class 6
    # --------------------------------------------------------

    after_peak = (
        times_out
        >=
        peak_rain_time
    )


    wet_after_peak = (
        after_peak
        &
        (
            class_out
            >=
            5
        )
    )


    recovery_hours = np.nan


    inds_after_peak = np.where(
        after_peak
    )[0]


    for ii in inds_after_peak:

        if class_out[
            ii
        ] < 5:

            recovery_hours = (
                (
                    times_out[ii]
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


    return {
        "ZSURF":
            ZSURF,

        "INFIL":
            INFIL_BASE_DAY,

        "times":
            times_out,

        "F":
            F_out,

        "score":
            score_out,

        "classes":
            class_out,

        "max_F":
            np.nanmax(
                F_out
            ),

        "final_F":
            F_out[
                -1
            ],

        "min_score":
            np.nanmin(
                score_out
            ),

        "final_score":
            score_out[
                -1
            ],

        "final_class":
            int(
                class_out[
                    -1
                ]
            ),

        "recovery_hours":
            recovery_hours,

        "transfer":
            total_transfer,

        "runoff":
            total_runoff,

        "surface_et":
            total_et,

        "drainage":
            total_drainage,
    }


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

results = []


for ZSURF in ZSURF_VALUES:

    for INFIL in INFIL_VALUES:

        result = run_experiment(
            ZSURF,
            INFIL,
        )

        results.append(
            result
        )


# ============================================================
# PRINT TABLE
# ============================================================

print(
    "Results:"
)

print()

print(
    " Zs   I0     maxF  finalF   minH  finalH  "
    "class       recover[h]  transfer  runoff"
)

print(
    "-------------------------------------------------------------"
)


for R in results:

    recovery_text = (
        " >window"
        if not np.isfinite(
            R[
                "recovery_hours"
            ]
        )
        else
        f"{R['recovery_hours']:7.1f}"
    )


    print(
        f"{R['ZSURF']:4.0f} "
        f"{R['INFIL']:5.0f} "
        f"{R['max_F']:7.2f} "
        f"{R['final_F']:7.2f} "
        f"{R['min_score']:6.1f} "
        f"{R['final_score']:7.1f} "
        f"{CLASS_NAMES[R['final_class']]:11s} "
        f"{recovery_text:>9s} "
        f"{R['transfer']:8.2f} "
        f"{R['runoff']:7.2f}"
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
            "surface_depth_mm",
            "infiltration_base_mm_day",
            "max_F",
            "final_F",
            "minimum_score",
            "final_score",
            "final_class",
            "final_class_name",
            "hours_to_below_wet_after_peak",
            "total_surface_to_deep_mm",
            "total_runoff_mm",
            "total_surface_et_mm",
            "total_deep_drainage_mm",
        ]
    )


    for R in results:

        writer.writerow(
            [
                R[
                    "ZSURF"
                ],

                R[
                    "INFIL"
                ],

                R[
                    "max_F"
                ],

                R[
                    "final_F"
                ],

                R[
                    "min_score"
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
                    "recovery_hours"
                ],

                R[
                    "transfer"
                ],

                R[
                    "runoff"
                ],

                R[
                    "surface_et"
                ],

                R[
                    "drainage"
                ],
            ]
        )


# ============================================================
# PLOT F
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6,
    )
)


for R in results:

    label = (
        f"Z={R['ZSURF']:.0f} mm, "
        f"I={R['INFIL']:.0f}"
    )


    ax.plot(
        R[
            "times"
        ],
        R[
            "F"
        ],
        marker="o",
        label=label,
    )


ax.axhline(
    1.0,
    linestyle="--",
    label="Field capacity",
)


ax.axhline(
    0.80,
    linestyle=":",
    label="Wet threshold",
)


ax.set_title(
    "Hotspot Dry-down Sensitivity — Hydraulic State"
)


ax.set_ylabel(
    "F = (theta - WP) / (FC - WP)"
)


ax.set_xlabel(
    "UTC"
)


ax.legend(
    fontsize=8,
    ncol=3,
)


fig.tight_layout()


fig.savefig(
    FIG_F,
    dpi=200,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# PLOT HERO SCORE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6,
    )
)


for R in results:

    label = (
        f"Z={R['ZSURF']:.0f} mm, "
        f"I={R['INFIL']:.0f}"
    )


    ax.plot(
        R[
            "times"
        ],
        R[
            "score"
        ],
        marker="o",
        label=label,
    )


ax.set_title(
    "Hotspot Dry-down Sensitivity — Hero Dirt Score"
)


ax.set_ylabel(
    "Hero Dirt score"
)


ax.set_xlabel(
    "UTC"
)


ax.set_ylim(
    0,
    105,
)


ax.legend(
    fontsize=8,
    ncol=3,
)


fig.tight_layout()


fig.savefig(
    FIG_SCORE,
    dpi=200,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    f"Saved:"
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

print()

print(
    "============================================"
)
print(
    " SWEEP COMPLETE"
)
print(
    "============================================"
)
print()
