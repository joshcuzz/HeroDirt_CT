#!/usr/bin/env python3

"""
Hero Dirt Forecast
Plot soil diagnostic maps:

1. SSURGO field capacity
2. Current hydraulic state F
3. SSURGO plasticity index

Outputs
-------
output/figures/
    field_capacity_200m.png
    current_F_state.png
    plasticity_index_200m.png
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

P = np.load(
    PLASTICITY_FILE
)

C = np.load(
    CURRENT_FILE
)


x = S["x"].astype(float)
y = S["y"].astype(float)

public = S[
    "public_valid"
].astype(bool)


theta_fc = S[
    "theta_fc"
].astype(float)

PI = P[
    "plasticity_index"
].astype(float)

Fstate = C[
    "fc_state"
].astype(float)


# ============================================================
# MASK
# ============================================================

theta_fc_plot = theta_fc.copy()
PI_plot = PI.copy()
F_plot = Fstate.copy()

theta_fc_plot[
    ~public
] = np.nan

PI_plot[
    ~public
] = np.nan

F_plot[
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
# GENERIC PLOTTER
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
# FIELD CAPACITY
# ============================================================

make_map(
    theta_fc_plot,
    "Hero Dirt — SSURGO Field Capacity (0–15 cm)",
    "Field capacity [m3/m3]",
    "field_capacity_200m.png",
    vmin=0.05,
    vmax=0.30,
)


# ============================================================
# CURRENT F STATE
#
# F = 0 -> wilting point
# F = 1 -> field capacity
# ============================================================

make_map(
    F_plot,
    "Hero Dirt — Current Hydraulic State F",
    "F = (theta - WP) / (FC - WP)",
    "current_F_state.png",
    vmin=0.0,
    vmax=1.2,
)


# ============================================================
# PLASTICITY INDEX
# ============================================================

make_map(
    PI_plot,
    "Hero Dirt — SSURGO Plasticity Index (0–15 cm)",
    "Plasticity index [%]",
    "plasticity_index_200m.png",
    vmin=0.0,
    vmax=20.0,
)


print()
print(
    "============================================"
)
print(
    " SOIL DIAGNOSTIC MAPS COMPLETE"
)
print(
    "============================================"
)
print()

print(
    f"Figures directory:"
)

print(
    f"  {OUT_DIR}"
)

print()
