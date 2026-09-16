#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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


M = np.load(MRMS_FILE)
C = np.load(CURRENT_FILE)
F = np.load(FORECAST_FILE)

time_mrms = M["time"].astype("datetime64[s]")
precip = M["precip_mm"].astype(float)

x = C["x"].astype(float)
y = C["y"].astype(float)

public = C["public_valid"].astype(bool)

current_condition = C["condition"].astype(int)

forecast_time = F["time"].astype("datetime64[s]")
forecast_condition = F["condition"].astype(int)


# ============================================================
# TODAY'S MRMS ACCUMULATION
# ============================================================

day = time_mrms.max().astype("datetime64[D]")

mask_day = (
    time_mrms.astype("datetime64[D]")
    ==
    day
)

rain_mm = np.nansum(
    precip[mask_day],
    axis=0,
)

rain_in = rain_mm / 25.4


# ============================================================
# FIRST TIME CELL LEAVES TOO WET
#
# condition 6 = Too Wet
# ============================================================

forecast_start = C["time"].astype("datetime64[s]")

recovery_hours = np.full(
    current_condition.shape,
    np.nan,
    dtype=float,
)

# Cells already not Too Wet at t=0
already_recovered = (
    public
    &
    (current_condition != 6)
)

recovery_hours[
    already_recovered
] = 0.0


for k in range(
    len(forecast_time)
):

    hours = (
        forecast_time[k]
        -
        forecast_start
    ).astype(
        "timedelta64[s]"
    ).astype(float) / 3600.0

    newly_recovered = (
        public
        &
        np.isnan(recovery_hours)
        &
        (forecast_condition[k] != 6)
    )

    recovery_hours[
        newly_recovered
    ] = hours


# Cells still Too Wet through end of forecast
still_too_wet = (
    public
    &
    np.isnan(recovery_hours)
)


# ============================================================
# SUMMARY
# ============================================================

valid = (
    public
    &
    np.isfinite(rain_in)
)

paired = (
    valid
    &
    np.isfinite(recovery_hours)
)


print()
print("============================================")
print(" SPATIAL RECOVERY DIAGNOSTIC")
print("============================================")
print()

print(
    f"MRMS date: {day}"
)

print(
    f"Recovered within forecast: "
    f"{np.count_nonzero(paired):,}"
)

print(
    f"Still Too Wet at forecast end: "
    f"{np.count_nonzero(still_too_wet):,}"
)

if np.count_nonzero(paired) > 10:

    corr = np.corrcoef(
        rain_in[paired],
        recovery_hours[paired],
    )[0, 1]

    print(
        f"Rain vs recovery-time correlation: "
        f"{corr:.3f}"
    )

print()


# ============================================================
# MAP 1: RAIN
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    np.where(
        public,
        rain_in,
        np.nan,
    ),
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "MRMS precipitation [inches]"
)

ax.set_title(
    "Today's MRMS precipitation"
)

ax.set_xlabel(
    "Easting [m]"
)

ax.set_ylabel(
    "Northing [m]"
)

ax.set_aspect(
    "equal"
)

plt.tight_layout()
plt.show()


# ============================================================
# MAP 2: HOURS TO LEAVE TOO WET
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    recovery_hours,
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "Hours until cell first leaves Too Wet"
)

ax.set_title(
    "Hero Dirt recovery time — ET=4 forecast"
)

ax.set_xlabel(
    "Easting [m]"
)

ax.set_ylabel(
    "Northing [m]"
)

ax.set_aspect(
    "equal"
)

plt.tight_layout()
plt.show()


# ============================================================
# SCATTER
# ============================================================

r = rain_in[paired]
h = recovery_hours[paired]

step = max(
    1,
    len(r) // 50000
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.scatter(
    r[::step],
    h[::step],
    s=4,
    alpha=0.20,
)

ax.set_xlabel(
    "Today's MRMS precipitation [inches]"
)

ax.set_ylabel(
    "Hours to leave Too Wet"
)

ax.set_title(
    "Storm magnitude vs recovery time"
)

ax.grid(
    alpha=0.25
)

plt.tight_layout()
plt.show()
