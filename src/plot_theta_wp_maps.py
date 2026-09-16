#!/usr/bin/env python3

"""
Hero Dirt Forecast
Plot:

1. Current absolute surface soil moisture theta
2. SSURGO wilting point theta_wp

These help diagnose whether high hydraulic-state F values
are caused by genuinely high absolute moisture, or by a
small WP-to-FC interval.
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

CURRENT_FILE = (
    ROOT
    / "output"
    / "current"
    / "HeroDirt_current_state.npz"
)

OUT_DIR = (
    ROOT
    / "output"
    / "figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

S = np.load(
    STATIC_FILE
)

C = np.load(
    CURRENT_FILE
)


x = S["x"].astype(float)
y = S["y"].astype(float)

public = S[
    "public_valid"
].astype(bool)


theta_wp = S[
    "theta_wp"
].astype(float)


theta_surface = C[
    "theta_surface"
].astype(float)


# ============================================================
# MASK
# ============================================================

theta_wp_plot = theta_wp.copy()
theta_surface_plot = theta_surface.copy()

theta_wp_plot[
    ~public
] = np.nan

theta_surface_plot[
    ~public
] = np.nan


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


if y[0] > y[-1]:
    origin = "upper"
else:
    origin = "lower"


# ============================================================
# GENERIC MAP FUNCTION
# ============================================================

def make_map(
    A,
    title,
    cbar_label,
    filename,
    vmin=None,
    vmax=None,
):

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    im = ax.imshow(
        A,
        origin=origin,
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    cb = fig.colorbar(
        im,
        ax=ax,
    )

    cb.set_label(
        cbar_label
    )

    ax.set_title(
        title
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
        OUT_DIR
        /
        filename
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
# CURRENT SURFACE THETA
# ============================================================

make_map(
    theta_surface_plot,
    "Hero Dirt — Current Surface Soil Moisture",
    "Surface theta [m3/m3]",
    "surface_theta_now.png",
    vmin=0.03,
    vmax=0.18,
)


# ============================================================
# WILTING POINT
# ============================================================

make_map(
    theta_wp_plot,
    "Hero Dirt — SSURGO Wilting Point (0–15 cm)",
    "Wilting point [m3/m3]",
    "wilting_point_200m.png",
    vmin=0.02,
    vmax=0.16,
)


print()
print(
    "============================================"
)
print(
    " THETA / WILTING-POINT MAPS COMPLETE"
)
print(
    "============================================"
)
print()
