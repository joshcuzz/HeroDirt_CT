#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path.home() / "HeroDirt_CT"

MRMS_FILE = (
    ROOT
    / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)


D = np.load(
    MRMS_FILE
)

time = D["time"].astype(
    "datetime64[s]"
)

precip_mm = D[
    "precip_mm"
].astype(float)

x = D["x"]
y = D["y"]


# ============================================================
# TODAY'S UTC DATE
# ============================================================

latest_time = time.max()

today = latest_time.astype(
    "datetime64[D]"
)

mask = (
    time.astype("datetime64[D]")
    ==
    today
)


print()
print("============================================")
print(" MRMS TODAY")
print("============================================")
print()

print(
    f"Date: {today}"
)

print(
    f"Hourly fields used: {mask.sum()}"
)

print(
    f"First time: {time[mask][0]}"
)

print(
    f"Last time:  {time[mask][-1]}"
)


# ============================================================
# ACCUMULATE
# ============================================================

accum_mm = np.nansum(
    precip_mm[mask, :, :],
    axis=0,
)

accum_in = (
    accum_mm
    /
    25.4
)


print()
print(
    f"Domain mean: "
    f"{np.nanmean(accum_in):.2f} in"
)

print(
    f"Maximum:     "
    f"{np.nanmax(accum_in):.2f} in"
)

print()


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    accum_in,
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
    f"MRMS precipitation — {str(today)} UTC"
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
