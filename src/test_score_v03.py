#!/usr/bin/env python3

"""
Hero Dirt Forecast
Experimental Hero Dirt score v0.3.

Purpose
-------
Replace the earlier Gaussian "optimum near field capacity"
score with a more interpretable structure:

    1. hydraulic moisture state
    2. wetness above / near field capacity
    3. plasticity-dependent deformation sensitivity
    4. deep-layer wetness

Nothing in the hydrology model is changed.

This script is diagnostic only.
"""

from pathlib import Path
import numpy as np


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

CURRENT_FILE = (
    ROOT
    / "output"
    / "current"
    / "HeroDirt_current_state.npz"
)

FORECAST_FILE = (
    ROOT
    / "output"
    / "forecast"
    / "HeroDirt_forecast_6hourly.npz"
)


# ============================================================
# LOAD
# ============================================================

S = np.load(STATIC_FILE)

P = np.load(PLASTICITY_FILE)

C = np.load(CURRENT_FILE)

FCAST = np.load(FORECAST_FILE)


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


PI = P[
    "plasticity_index"
].astype(float)


theta_surface_now = C[
    "theta_surface"
].astype(float)

theta_deep_now = C[
    "theta_deep"
].astype(float)


theta_surface_forecast = FCAST[
    "theta_surface"
].astype(float)

theta_deep_forecast = FCAST[
    "theta_deep"
].astype(float)

forecast_time = FCAST[
    "time"
]


# ============================================================
# EMPIRICAL HERO DIRT PARAMETERS
#
# These are calibration parameters, not soil constants.
# ============================================================

# Moisture at which dry-side penalty essentially disappears.
#
# F = 0 : wilting point
# F = 1 : field capacity

F_DRY_FULL = 0.50


# Wet-condition penalty begins before full field capacity.
# This represents increasing susceptibility to deformation
# as the soil becomes wetter.

F_WET_START = 0.80


# Controls how rapidly wet penalty grows.

WET_BASE_STRENGTH = 1.5

WET_PI_STRENGTH = 3.0


# Deep soil wetness penalty.

DEEP_WET_STRENGTH = 0.50


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep(
    x,
):

    x = np.clip(
        x,
        0.0,
        1.0,
    )

    return (
        x * x
        *
        (
            3.0
            -
            2.0 * x
        )
    )


# ============================================================
# SCORE FUNCTION
# ============================================================

def score_v03(
    surf,
    deep,
):

    # --------------------------------------------------------
    # Hydraulic state relative to WP -> FC
    # --------------------------------------------------------

    denom = (
        theta_fc
        -
        theta_wp
    )


    F = (
        surf
        -
        theta_wp
    ) / denom


    Fdeep = (
        deep
        -
        theta_wp
    ) / denom


    # --------------------------------------------------------
    # DRY-SIDE SUITABILITY
    #
    # F=0 -> 0
    # F>=F_DRY_FULL -> approximately 1
    #
    # This replaces the old Gaussian optimum.
    # --------------------------------------------------------

    dry_coordinate = (
        F
        /
        F_DRY_FULL
    )


    Sdry = smoothstep(
        dry_coordinate
    )


    # --------------------------------------------------------
    # PLASTICITY SENSITIVITY
    #
    # Nonplastic soils = 0 sensitivity.
    #
    # For plastic soils, use a bounded PI response.
    #
    # PI/(PI+7) is simply a smooth empirical normalization.
    # It is NOT a soil-mechanics law.
    # --------------------------------------------------------

    PI_sensitivity = np.zeros(
        PI.shape,
        dtype=float,
    )


    valid_pi = (
        np.isfinite(PI)
        &
        (PI > 0.0)
    )


    PI_sensitivity[
        valid_pi
    ] = (
        PI[
            valid_pi
        ]
        /
        (
            PI[
                valid_pi
            ]
            +
            7.0
        )
    )


    # --------------------------------------------------------
    # WETNESS COORDINATE
    #
    # No wet penalty below F_WET_START.
    #
    # At F=1 we have reached field capacity.
    #
    # Above FC, continue scaling toward saturation.
    # --------------------------------------------------------

    wet_pre_fc = (
        F
        -
        F_WET_START
    ) / (
        1.0
        -
        F_WET_START
    )


    wet_pre_fc = np.clip(
        wet_pre_fc,
        0.0,
        1.0,
    )


    above_fc = (
        surf
        -
        theta_fc
    ) / (
        theta_sat
        -
        theta_fc
    )


    above_fc = np.clip(
        above_fc,
        0.0,
        1.0,
    )


    # Combine approach-to-FC and actual FC excess.
    wet_state = (
        0.35
        *
        wet_pre_fc
        +
        above_fc
    )


    # --------------------------------------------------------
    # PLASTICITY-DEPENDENT WET PENALTY
    # --------------------------------------------------------

    wet_strength = (
        WET_BASE_STRENGTH
        +
        WET_PI_STRENGTH
        *
        PI_sensitivity
    )


    Swet = np.exp(
        -wet_strength
        *
        wet_state
    )


    # --------------------------------------------------------
    # DEEP WETNESS
    #
    # Only penalize deep layer substantially once it approaches
    # or exceeds field capacity.
    # --------------------------------------------------------

    deep_wet = (
        Fdeep
        -
        0.90
    ) / (
        1.0
        -
        0.90
    )


    deep_wet = np.clip(
        deep_wet,
        0.0,
        1.0,
    )


    deep_above_fc = (
        deep
        -
        theta_fc
    ) / (
        theta_sat
        -
        theta_fc
    )


    deep_above_fc = np.clip(
        deep_above_fc,
        0.0,
        1.0,
    )


    Sdeep = np.exp(
        -DEEP_WET_STRENGTH
        *
        (
            0.25
            *
            deep_wet
            +
            deep_above_fc
        )
    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    score = (
        100.0
        *
        Sdry
        *
        Swet
        *
        Sdeep
    )


    score = np.clip(
        score,
        0.0,
        100.0,
    )


    score[
        ~public
    ] = np.nan


    # --------------------------------------------------------
    # DIRECTIONAL CONDITION CLASS
    #
    # These use hydraulic/mechanical state, not score alone.
    # --------------------------------------------------------

    condition = np.zeros(
        surf.shape,
        dtype=np.uint8,
    )


    # 1 = Very Dry
    condition[
        public
        &
        (F < 0.20)
    ] = 1


    # 2 = Dry
    condition[
        public
        &
        (F >= 0.20)
        &
        (F < 0.40)
    ] = 2


    # 3 = Moist / Good
    condition[
        public
        &
        (F >= 0.40)
        &
        (F < 0.65)
    ] = 3


    # 4 = Hero
    condition[
        public
        &
        (F >= 0.65)
        &
        (F < F_WET_START)
    ] = 4


    # 5 = Wet
    condition[
        public
        &
        (F >= F_WET_START)
        &
        (F < 1.0)
    ] = 5


    # 6 = Too Wet
    condition[
        public
        &
        (
            (F >= 1.0)
            |
            (
                above_fc > 0.10
            )
        )
    ] = 6


    return (
        score,
        condition,
        F,
        PI_sensitivity,
        Swet,
    )


# ============================================================
# CURRENT STATE
# ============================================================

(
    score_now,
    class_now,
    F_now,
    PI_sens,
    wet_penalty_now,
) = score_v03(
    theta_surface_now,
    theta_deep_now,
)


# ============================================================
# CLASS NAMES
# ============================================================

names = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


# ============================================================
# CURRENT SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT SCORE v0.3 TEST"
)
print(
    "============================================"
)
print()


print(
    f"Mean surface F: "
    f"{np.nanmean(F_now[public]):.3f}"
)

print(
    f"Median surface F: "
    f"{np.nanmedian(F_now[public]):.3f}"
)

print(
    f"Mean score v0.3: "
    f"{np.nanmean(score_now[public]):.1f}"
)

print(
    f"Median score v0.3: "
    f"{np.nanmedian(score_now[public]):.1f}"
)

print()


print(
    "Current condition classes:"
)


npublic = np.count_nonzero(
    public
)


for code in range(
    1,
    7,
):

    n = np.count_nonzero(
        public
        &
        (
            class_now
            ==
            code
        )
    )


    print(
        f"  {names[code]:11s}: "
        f"{100*n/npublic:5.1f}%"
    )


print()


# ============================================================
# SCORE PERCENTILES
# ============================================================

p = np.nanpercentile(
    score_now[
        public
    ],
    [
        5,
        10,
        25,
        50,
        75,
        90,
        95,
    ],
)


print(
    "Current score percentiles:"
)

print(
    "  P05  P10  P25  P50  P75  P90  P95"
)

print(
    " ",
    " ".join(
        f"{v:6.1f}"
        for v in p
    )
)

print()


# ============================================================
# PI EFFECT
# ============================================================

plastic = (
    public
    &
    np.isfinite(PI)
    &
    (PI > 0)
)

nonplastic = (
    public
    &
    np.isfinite(PI)
    &
    (PI <= 0)
)


print(
    "Mechanical sensitivity:"
)

print(
    f"  mean PI sensitivity, plastic soils: "
    f"{np.nanmean(PI_sens[plastic]):.3f}"
)

print(
    f"  current mean wet penalty, plastic: "
    f"{np.nanmean(wet_penalty_now[plastic]):.3f}"
)

print(
    f"  current mean wet penalty, nonplastic: "
    f"{np.nanmean(wet_penalty_now[nonplastic]):.3f}"
)

print()


# ============================================================
# FORECAST TEST
# ============================================================

print(
    "Forecast score evolution:"
)

print()


nt = len(
    forecast_time
)


for it in range(
    nt
):

    hour = (
        it + 1
    ) * 6


    if hour not in [
        24,
        48,
        72,
        96,
        120,
    ]:

        continue


    score_t, class_t, F_t, _, _ = (
        score_v03(
            theta_surface_forecast[
                it
            ],
            theta_deep_forecast[
                it
            ],
        )
    )


    print(
        f"+{hour:3d} h:"
    )

    print(
        f"  mean F:     "
        f"{np.nanmean(F_t[public]):.3f}"
    )

    print(
        f"  mean score: "
        f"{np.nanmean(score_t[public]):.1f}"
    )


    for code in range(
        1,
        7,
    ):

        n = np.count_nonzero(
            public
            &
            (
                class_t
                ==
                code
            )
        )

        print(
            f"    "
            f"{names[code]:11s}: "
            f"{100*n/npublic:5.1f}%"
        )


    print()


print(
    "============================================"
)
print(
    " SCORE v0.3 TEST COMPLETE"
)
print(
    "============================================"
)
print()
