#!/usr/bin/env python3

"""
Hero Dirt Forecast
Diagnose the localized MRMS rainfall event responsible
for the current Wet / Too Wet hotspot.

Outputs
-------
output/figures/
    mrms_hotspot_timeseries.png
    mrms_vs_wet_overlay.png
    mrms_peak_hour.png

Also prints:
    peak MRMS hour
    maximum hourly precipitation
    accumulated hotspot precipitation
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

MRMS_FILE = (
    ROOT
    / "data"
    / "mrms"
    / "processed"
    / "HeroDirt_MRMS_hourly.npz"
)

CURRENT_FILE = (
    ROOT
    / "output"
    / "current"
    / "HeroDirt_current_state.npz"
)

FIG_DIR = (
    ROOT
    / "output"
    / "figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

S = np.load(
    STATIC_FILE
)

M = np.load(
    MRMS_FILE
)

C = np.load(
    CURRENT_FILE
)


x = S["x"].astype(float)
y = S["y"].astype(float)

X, Y = np.meshgrid(
    x,
    y,
)

public = S[
    "public_valid"
].astype(bool)


times = M[
    "time"
].astype(
    "datetime64[s]"
)

precip = M[
    "precip_mm"
].astype(float)


hero_class = C[
    "hero_class"
]


# ============================================================
# HOTSPOT REGION
#
# Same region used in previous diagnostic.
# ============================================================

hotspot = (
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


if not np.any(
    hotspot
):
    raise RuntimeError(
        "No valid cells in hotspot region."
    )


# ============================================================
# HOURLY HOTSPOT STATISTICS
# ============================================================

nt = len(
    times
)


mean_hourly = np.zeros(
    nt
)

max_hourly = np.zeros(
    nt
)


for it in range(
    nt
):

    vals = precip[
        it
    ][
        hotspot
    ]

    mean_hourly[
        it
    ] = np.nanmean(
        vals
    )

    max_hourly[
        it
    ] = np.nanmax(
        vals
    )


# ============================================================
# PEAK HOUR
# ============================================================

peak_index = int(
    np.nanargmax(
        max_hourly
    )
)


peak_time = times[
    peak_index
]


peak_field = precip[
    peak_index
]


peak_value = float(
    max_hourly[
        peak_index
    ]
)


# ============================================================
# ACCUMULATED PRECIP
# ============================================================

accum = np.nansum(
    precip,
    axis=0,
)


hotspot_accum = accum[
    hotspot
]


# maximum accumulated cell
candidate = np.where(
    hotspot,
    accum,
    np.nan,
)


flat = np.nanargmax(
    candidate
)


rmax, cmax = np.unravel_index(
    flat,
    accum.shape,
)


max_accum = float(
    accum[
        rmax,
        cmax
    ]
)


max_accum_x = float(
    X[
        rmax,
        cmax
    ]
)


max_accum_y = float(
    Y[
        rmax,
        cmax
    ]
)


# ============================================================
# PRINT EVENT SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT MRMS EVENT DIAGNOSTIC"
)
print(
    "============================================"
)
print()


print(
    f"MRMS hours: "
    f"{nt}"
)

print(
    f"Start: "
    f"{times[0]}"
)

print(
    f"End:   "
    f"{times[-1]}"
)

print()


print(
    "Peak hourly hotspot rainfall:"
)

print(
    f"  time: "
    f"{peak_time} UTC"
)

print(
    f"  maximum cell: "
    f"{peak_value:.2f} mm"
)

print(
    f"  hotspot mean: "
    f"{mean_hourly[peak_index]:.2f} mm"
)

print()


print(
    "Accumulated rainfall:"
)

print(
    f"  hotspot mean: "
    f"{np.nanmean(hotspot_accum):.2f} mm"
)

print(
    f"  hotspot median: "
    f"{np.nanmedian(hotspot_accum):.2f} mm"
)

print(
    f"  hotspot maximum: "
    f"{max_accum:.2f} mm"
)

print(
    f"  maximum location:"
)

print(
    f"    E = "
    f"{max_accum_x/1000:.2f} km"
)

print(
    f"    N = "
    f"{max_accum_y/1000:.2f} km"
)

print()


# ============================================================
# MAP GEOMETRY
# ============================================================

dx = abs(
    np.median(
        np.diff(x)
    )
)

dy = abs(
    np.median(
        np.diff(y)
    )
)


extent = [
    (np.min(x) - dx / 2) / 1000.0,
    (np.max(x) + dx / 2) / 1000.0,
    (np.min(y) - dy / 2) / 1000.0,
    (np.max(y) + dy / 2) / 1000.0,
]


origin = (
    "upper"
    if y[0] > y[-1]
    else "lower"
)


# ============================================================
# TIME-SERIES FIGURE
# ============================================================

hours = np.arange(
    nt
)


fig, ax = plt.subplots(
    figsize=(11, 5)
)


ax.plot(
    hours,
    mean_hourly,
    marker="o",
    label="Hotspot mean",
)


ax.plot(
    hours,
    max_hourly,
    marker="o",
    label="Hotspot maximum",
)


ax.axvline(
    peak_index,
    linestyle="--",
)


ax.set_xticks(
    hours
)


ax.set_xticklabels(
    [
        str(t)[5:16]
        .replace(
            "T",
            "\n"
        )
        for t in times
    ],
    rotation=45,
    ha="right",
)


ax.set_ylabel(
    "Hourly precipitation [mm]"
)


ax.set_xlabel(
    "UTC"
)


ax.set_title(
    "MRMS Rainfall — Hero Dirt Hotspot"
)


ax.legend()


fig.tight_layout()


outfile = (
    FIG_DIR
    / "mrms_hotspot_timeseries.png"
)


fig.savefig(
    outfile,
    dpi=200,
    bbox_inches="tight",
)


print(
    f"Saved: {outfile}"
)


plt.close(
    fig
)


# ============================================================
# PEAK-HOUR MAP
# ============================================================

peak_plot = peak_field.copy()

peak_plot[
    ~public
] = np.nan


fig, ax = plt.subplots(
    figsize=(12, 6)
)


im = ax.imshow(
    peak_plot,
    origin=origin,
    extent=extent,
    cmap="viridis",
    vmin=0,
    interpolation="nearest",
)


cb = fig.colorbar(
    im,
    ax=ax,
)


cb.set_label(
    "Hourly MRMS precipitation [mm]"
)


ax.set_title(
    f"MRMS Peak Hour — {peak_time} UTC"
)


ax.set_xlabel(
    "UTM Easting [km]"
)


ax.set_ylabel(
    "UTM Northing [km]"
)


ax.set_aspect(
    "equal"
)


fig.tight_layout()


outfile = (
    FIG_DIR
    / "mrms_peak_hour.png"
)


fig.savefig(
    outfile,
    dpi=200,
    bbox_inches="tight",
)


print(
    f"Saved: {outfile}"
)


plt.close(
    fig
)


# ============================================================
# ACCUMULATED RAIN + WET CONDITION OVERLAY
# ============================================================

accum_plot = accum.copy()

accum_plot[
    ~public
] = np.nan


wet = (
    (
        hero_class == 5
    )
    |
    (
        hero_class == 6
    )
)


# Convert wet mask to numeric contour field.
wet_numeric = wet.astype(float)


fig, ax = plt.subplots(
    figsize=(12, 6)
)


im = ax.imshow(
    accum_plot,
    origin=origin,
    extent=extent,
    cmap="viridis",
    vmin=0,
    interpolation="nearest",
)


cb = fig.colorbar(
    im,
    ax=ax,
)


cb.set_label(
    "Accumulated MRMS precipitation [mm]"
)


# Build coordinate arrays in km for contour.
Xkm = X / 1000.0
Ykm = Y / 1000.0


ax.contour(
    Xkm,
    Ykm,
    wet_numeric,
    levels=[
        0.5
    ],
    linewidths=1.5,
)


ax.set_title(
    "MRMS Accumulation with Current Wet / Too Wet Boundary"
)


ax.set_xlabel(
    "UTM Easting [km]"
)


ax.set_ylabel(
    "UTM Northing [km]"
)


ax.set_aspect(
    "equal"
)


fig.tight_layout()


outfile = (
    FIG_DIR
    / "mrms_vs_wet_overlay.png"
)


fig.savefig(
    outfile,
    dpi=200,
    bbox_inches="tight",
)


print(
    f"Saved: {outfile}"
)


plt.close(
    fig
)


print()
print(
    "============================================"
)
print(
    " MRMS EVENT DIAGNOSTIC COMPLETE"
)
print(
    "============================================"
)
print()
