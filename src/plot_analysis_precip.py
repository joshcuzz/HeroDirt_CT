#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path.home() / "HeroDirt"

S = np.load(
    ROOT / "static/model/HeroDirt_static_200m.npz"
)

C = np.load(
    ROOT / "output/current/HeroDirt_current_state.npz"
)


x = S["x"].astype(float)
y = S["y"].astype(float)

public = S["public_valid"].astype(bool)

P = C[
    "analysis_precipitation"
].astype(float)


P[
    ~public
] = np.nan


dx = abs(
    np.median(np.diff(x))
)

dy = abs(
    np.median(np.diff(y))
)


extent = [
    (np.min(x) - dx/2) / 1000,
    (np.max(x) + dx/2) / 1000,
    (np.min(y) - dy/2) / 1000,
    (np.max(y) + dy/2) / 1000,
]


origin = (
    "upper"
    if y[0] > y[-1]
    else "lower"
)


fig, ax = plt.subplots(
    figsize=(12, 6)
)


im = ax.imshow(
    P,
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
    "MRMS precipitation since SMAP [mm]"
)


ax.set_title(
    "Hero Dirt — MRMS Precipitation Since SMAP Observation"
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
    ROOT
    / "output/figures"
    / "mrms_since_smap.png"
)


fig.savefig(
    outfile,
    dpi=200,
    bbox_inches="tight",
)


print()
print(
    f"Domain mean: {np.nanmean(P):.3f} mm"
)

print(
    f"Maximum:     {np.nanmax(P):.3f} mm"
)

print(
    f"Saved: {outfile}"
)

print()


plt.show()
