#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path.home() / "HeroDirt"

STATIC_FILE = (
    ROOT
    / "static"
    / "model"
    / "HeroDirt_static_200m.npz"
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

x = S["x"].astype(float)
y = S["y"].astype(float)

clay = S[
    "clay"
].astype(float)

public = S[
    "public_valid"
].astype(bool)


# Mask areas outside public Hero Dirt domain
clay_plot = clay.copy()

clay_plot[
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
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


im = ax.imshow(
    clay_plot,
    origin=origin,
    extent=extent,
    cmap="viridis",
    vmin=0,
    vmax=30,
    interpolation="nearest",
)


cb = fig.colorbar(
    im,
    ax=ax,
)

cb.set_label(
    "Clay content [%]"
)


ax.set_title(
    "Hero Dirt — SSURGO Clay Content (0–15 cm)"
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
    / "clay_200m.png"
)


fig.savefig(
    outfile,
    dpi=200,
    bbox_inches="tight",
)


print()
print(
    f"Saved: {outfile}"
)
print()


plt.show()
