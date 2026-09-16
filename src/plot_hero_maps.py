#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path.home() / "HeroDirt"

CURRENT_FILE = (
    ROOT
    / "output/current/HeroDirt_current_state.npz"
)

FORECAST_FILE = (
    ROOT
    / "output/forecast/HeroDirt_forecast_6hourly.npz"
)

FIG_DIR = (
    ROOT
    / "output/figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


C = np.load(
    CURRENT_FILE
)

F = np.load(
    FORECAST_FILE
)


def get_key(
    archive,
    names,
):

    for name in names:

        if name in archive.files:
            return archive[name]

    raise KeyError(
        f"None of these keys found: {names}\n"
        f"Available keys:\n{archive.files}"
    )


x = get_key(
    C,
    ["x"],
).astype(float)

y = get_key(
    C,
    ["y"],
).astype(float)


current_score = get_key(
    C,
    [
        "hero_score",
        "score",
    ],
).astype(float)


current_class = get_key(
    C,
    [
        "hero_class",
        "condition",
    ],
).astype(int)


current_F = get_key(
    C,
    [
        "F",
        "tread_F",
    ],
).astype(float)


current_tread = get_key(
    C,
    [
        "tread_theta",
        "surface_theta",
    ],
).astype(float)


current_theta05 = get_key(
    C,
    [
        "theta_0_5cm",
        "surface_theta",
    ],
).astype(float)


forecast_time = get_key(
    F,
    ["time"],
).astype("datetime64[s]")


forecast_score = get_key(
    F,
    [
        "hero_score",
        "score",
    ],
).astype(float)


forecast_class = get_key(
    F,
    [
        "hero_class",
        "condition",
    ],
).astype(int)


forecast_F = get_key(
    F,
    [
        "F",
        "tread_F",
    ],
).astype(float)


extent = [
    x.min(),
    x.max(),
    y.min(),
    y.max(),
]


# ============================================================
# CURRENT SCORE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

im = ax.imshow(
    current_score,
    origin="upper",
    extent=extent,
    vmin=0,
    vmax=100,
    cmap="viridis",
)

ax.set_title(
    "Hero Dirt — Current Score"
)

ax.set_xlabel(
    "UTM Easting [m]"
)

ax.set_ylabel(
    "UTM Northing [m]"
)

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Hero Dirt score"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "HeroDirt_current_score.png",
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# CURRENT HYDRAULIC STATE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

im = ax.imshow(
    current_F,
    origin="upper",
    extent=extent,
    vmin=0,
    vmax=1.5,
    cmap="viridis",
)

ax.set_title(
    "Hero Dirt — Current Tread Hydraulic State"
)

ax.set_xlabel(
    "UTM Easting [m]"
)

ax.set_ylabel(
    "UTM Northing [m]"
)

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "F = (theta - WP) / (FC - WP)"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "HeroDirt_current_F.png",
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# CURRENT TREAD THETA
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

im = ax.imshow(
    current_tread,
    origin="upper",
    extent=extent,
    cmap="viridis",
)

ax.set_title(
    "Hero Dirt — Current Tread Moisture (0–2 cm)"
)

ax.set_xlabel(
    "UTM Easting [m]"
)

ax.set_ylabel(
    "UTM Northing [m]"
)

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Volumetric water content"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "HeroDirt_current_tread_theta.png",
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# CURRENT COMBINED 0-5 CM THETA
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

im = ax.imshow(
    current_theta05,
    origin="upper",
    extent=extent,
    cmap="viridis",
)

ax.set_title(
    "Hero Dirt — Current Combined 0–5 cm Moisture"
)

ax.set_xlabel(
    "UTM Easting [m]"
)

ax.set_ylabel(
    "UTM Northing [m]"
)

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Volumetric water content"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "HeroDirt_current_theta_0_5cm.png",
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# FORECAST SCORE SNAPSHOTS
# ============================================================

forecast_start = forecast_time[0] - np.timedelta64(
    6,
    "h",
)


for target_hours in [
    24,
    48,
    72,
    96,
    120,
]:

    target_time = (
        forecast_start
        +
        np.timedelta64(
            target_hours,
            "h",
        )
    )


    k = int(
        np.argmin(
            np.abs(
                forecast_time
                -
                target_time
            )
        )
    )


    fig, ax = plt.subplots(
        figsize=(9, 8)
    )


    im = ax.imshow(
        forecast_score[k],
        origin="upper",
        extent=extent,
        vmin=0,
        vmax=100,
        cmap="viridis",
    )


    ax.set_title(
        f"Hero Dirt — +{target_hours} h"
    )


    ax.set_xlabel(
        "UTM Easting [m]"
    )


    ax.set_ylabel(
        "UTM Northing [m]"
    )


    cbar = fig.colorbar(
        im,
        ax=ax,
    )


    cbar.set_label(
        "Hero Dirt score"
    )


    fig.tight_layout()


    fig.savefig(
        FIG_DIR
        / f"HeroDirt_forecast_{target_hours:03d}h.png",
        dpi=200,
    )


    plt.close(
        fig
    )


# ============================================================
# CLASS MAP
# ============================================================

class_colors = [
    "#8c510a",  # 1 Very Dry
    "#d8b365",  # 2 Dry
    "#c7eae5",  # 3 Moist/Good
    "#5ab4ac",  # 4 Hero
    "#4393c3",  # 5 Wet
    "#2166ac",  # 6 Too Wet
]


from matplotlib.colors import ListedColormap

cmap_class = ListedColormap(
    class_colors
)


plot_class = np.where(
    current_class > 0,
    current_class,
    np.nan,
)


fig, ax = plt.subplots(
    figsize=(9, 8)
)


im = ax.imshow(
    plot_class,
    origin="upper",
    extent=extent,
    vmin=1,
    vmax=6,
    cmap=cmap_class,
)


ax.set_title(
    "Hero Dirt — Current Condition Class"
)


ax.set_xlabel(
    "UTM Easting [m]"
)


ax.set_ylabel(
    "UTM Northing [m]"
)


cbar = fig.colorbar(
    im,
    ax=ax,
    ticks=[
        1,
        2,
        3,
        4,
        5,
        6,
    ],
)


cbar.ax.set_yticklabels(
    [
        "Very Dry",
        "Dry",
        "Moist/Good",
        "Hero",
        "Wet",
        "Too Wet",
    ]
)


fig.tight_layout()


fig.savefig(
    FIG_DIR
    / "HeroDirt_current_class.png",
    dpi=200,
)


plt.close(
    fig
)


print()
print(
    "Saved figures to:"
)
print(
    f"  {FIG_DIR}"
)
print()
