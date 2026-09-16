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
# STATIC CONTROLS
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
# CORRELATION FUNCTION
# ============================================================

def corr(name, field):

    mask = (
        public
        &
        np.isfinite(time_to_hero)
        &
        np.isfinite(field)
    )

    if np.count_nonzero(mask) < 10:
        print(f"{name:25s}: insufficient data")
        return

    r = np.corrcoef(
        field[mask],
        time_to_hero[mask],
    )[0, 1]

    print(
        f"{name:25s}: "
        f"{r: .3f}"
    )


print()
print("============================================")
print(" CONTROLS ON TIME TO HERO")
print("============================================")
print()

print(
    "Positive correlation = longer recovery"
)

print(
    "Negative correlation = faster recovery"
)

print()

corr(
    "Today's rainfall",
    rain_in,
)

corr(
    "Sand %",
    sand,
)

corr(
    "Ksat",
    ksat,
)

corr(
    "FC - WP",
    storage_range,
)

corr(
    "Tree fraction",
    tree,
)

corr(
    "Southness",
    southness,
)

corr(
    "Slope",
    slope,
)

print()
