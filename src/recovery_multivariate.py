#!/usr/bin/env python3

from pathlib import Path

import numpy as np


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


M = np.load(MRMS_FILE)
C = np.load(CURRENT_FILE)
F = np.load(FORECAST_FILE)
S = np.load(STATIC_FILE)


mrms_time = M["time"].astype("datetime64[s]")
precip = M["precip_mm"].astype(float)

public = C["public_valid"].astype(bool)

current_F = C["F"].astype(float)

forecast_time = F["time"].astype("datetime64[s]")
forecast_F = F["F"].astype(float)

forecast_start = C["time"].astype("datetime64[s]")


# ============================================================
# TODAY'S RAIN
# ============================================================

day = mrms_time.max().astype("datetime64[D]")

mask_day = (
    mrms_time.astype("datetime64[D]")
    ==
    day
)

rain_mm = np.nansum(
    precip[mask_day],
    axis=0,
)

rain_in = rain_mm / 25.4


# ============================================================
# STATIC VARIABLES
# ============================================================

sand = S["sand"].astype(float)

ksat = S["ksat"].astype(float)

theta_wp = S["theta_wp"].astype(float)
theta_fc = S["theta_fc"].astype(float)

storage_range = (
    theta_fc
    -
    theta_wp
)

tree = S["tree_fraction"].astype(float)

southness = S["southness"].astype(float)

slope = S["slope"].astype(float)


log_ksat = np.log10(
    np.maximum(
        ksat,
        1.0e-6,
    )
)


# ============================================================
# TIME TO HERO
# ============================================================

time_to_hero = np.full(
    current_F.shape,
    np.nan,
    dtype=float,
)

already = (
    public
    &
    np.isfinite(current_F)
    &
    (current_F < 0.80)
)

time_to_hero[already] = 0.0


for k in range(len(forecast_time)):

    hours = (
        forecast_time[k]
        -
        forecast_start
    ).astype(
        "timedelta64[s]"
    ).astype(float) / 3600.0

    hit = (
        public
        &
        np.isnan(time_to_hero)
        &
        np.isfinite(forecast_F[k])
        &
        (forecast_F[k] < 0.80)
    )

    time_to_hero[hit] = hours


# ============================================================
# VALID CELLS
# ============================================================

valid = (
    public
    &
    np.isfinite(time_to_hero)
    &
    np.isfinite(rain_in)
    &
    np.isfinite(sand)
    &
    np.isfinite(log_ksat)
    &
    np.isfinite(storage_range)
    &
    np.isfinite(tree)
    &
    np.isfinite(southness)
    &
    np.isfinite(slope)
)


y = time_to_hero[valid]


Xraw = np.column_stack(
    [
        rain_in[valid],
        sand[valid],
        log_ksat[valid],
        storage_range[valid],
        tree[valid],
        southness[valid],
        slope[valid],
    ]
)


names = [
    "Rainfall",
    "Sand",
    "log10 Ksat",
    "FC - WP",
    "Tree fraction",
    "Southness",
    "Slope",
]


# ============================================================
# STANDARDIZE PREDICTORS
#
# This lets coefficient magnitudes be compared directly.
# ============================================================

means = np.mean(
    Xraw,
    axis=0,
)

stds = np.std(
    Xraw,
    axis=0,
)

good_std = (
    stds > 0
)

if not np.all(good_std):

    raise RuntimeError(
        "One or more predictors have zero variance."
    )


Xstd = (
    Xraw
    -
    means
) / stds


# Add intercept.
X = np.column_stack(
    [
        np.ones(
            len(y)
        ),
        Xstd,
    ]
)


# ============================================================
# ORDINARY LEAST SQUARES
# ============================================================

beta, residuals, rank, singular = np.linalg.lstsq(
    X,
    y,
    rcond=None,
)


yhat = X @ beta

ss_res = np.sum(
    (
        y
        -
        yhat
    ) ** 2
)

ss_tot = np.sum(
    (
        y
        -
        np.mean(y)
    ) ** 2
)

r2 = (
    1.0
    -
    ss_res
    /
    ss_tot
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("============================================")
print(" MULTIVARIATE CONTROLS ON TIME TO HERO")
print("============================================")
print()

print(
    f"Valid cells: {len(y):,}"
)

print(
    f"R^2:         {r2:.3f}"
)

print()

print(
    "Standardized coefficients:"
)

print(
    "Positive = longer recovery"
)

print(
    "Negative = faster recovery"
)

print()

for name, b in zip(
    names,
    beta[1:],
):

    print(
        f"{name:20s}: "
        f"{b: .3f} h per 1-sigma"
    )

print()
