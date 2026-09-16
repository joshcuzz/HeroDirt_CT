#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt_CT"

MRMS_FILE = (
    ROOT
    / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)

STATE_FILE = (
    ROOT
    / "output/current/HeroDirt_current_state.npz"
)


# ============================================================
# LOAD
# ============================================================

M = np.load(MRMS_FILE)
S = np.load(STATE_FILE)

time = M["time"].astype("datetime64[s]")
precip_mm = M["precip_mm"].astype(float)

x = M["x"].astype(float)
y = M["y"].astype(float)

F = S["F"].astype(float)
public = S["public_valid"].astype(bool)


# ============================================================
# ACCUMULATE LATEST UTC DAY
# ============================================================

latest_time = time.max()

day = latest_time.astype("datetime64[D]")

mask_time = (
    time.astype("datetime64[D]")
    ==
    day
)

rain_mm = np.nansum(
    precip_mm[mask_time, :, :],
    axis=0,
)

rain_in = rain_mm / 25.4


# ============================================================
# VALID COMPARISON MASK
# ============================================================

valid = (
    public
    &
    np.isfinite(F)
    &
    np.isfinite(rain_in)
)


r = rain_in[valid]
f = F[valid]


# ============================================================
# SUMMARY
# ============================================================

print()
print("============================================")
print(" MRMS vs CURRENT TREAD F")
print("============================================")
print()

print(f"Date: {day}")
print(f"Hourly MRMS fields: {mask_time.sum():d}")
print()

print("Rainfall [in]:")
print(f"  mean:   {np.mean(r):.2f}")
print(f"  median: {np.median(r):.2f}")
print(f"  P95:    {np.percentile(r, 95):.2f}")
print(f"  max:    {np.max(r):.2f}")

print()

print("Tread F:")
print(f"  mean:   {np.mean(f):.3f}")
print(f"  median: {np.median(f):.3f}")
print(f"  P05:    {np.percentile(f, 5):.3f}")
print(f"  P95:    {np.percentile(f, 95):.3f}")
print(f"  min:    {np.min(f):.3f}")
print(f"  max:    {np.max(f):.3f}")

print()

corr = np.corrcoef(
    r,
    f,
)[0, 1]

print(
    f"Spatial correlation "
    f"(rain vs tread F): {corr:.3f}"
)

print()


# ============================================================
# MAP 1: MRMS PRECIP
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    rain_in,
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "Accumulated precipitation [inches]"
)

ax.set_title(
    f"MRMS precipitation — {str(day)} UTC"
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
# MAP 2: CURRENT TREAD F
# ============================================================

plot_F = np.where(
    public,
    F,
    np.nan,
)

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    plot_F,
    shading="auto",
)

cb = fig.colorbar(
    mesh,
    ax=ax,
)

cb.set_label(
    "Tread F"
)

ax.set_title(
    "Hero Dirt current tread moisture state"
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
# SCATTER: RAIN vs F
# ============================================================

# Subsample for plotting if needed.
step = max(
    1,
    len(r) // 50000
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.scatter(
    r[::step],
    f[::step],
    s=4,
    alpha=0.20,
)

ax.set_xlabel(
    "Today's MRMS precipitation [inches]"
)

ax.set_ylabel(
    "Current tread F"
)

ax.set_title(
    "Rainfall vs current tread moisture"
)

ax.grid(
    alpha=0.25
)

plt.tight_layout()

plt.show()
