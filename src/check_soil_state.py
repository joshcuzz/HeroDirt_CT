#!/usr/bin/env python3

from pathlib import Path
import numpy as np


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


S = np.load(STATIC_FILE)
C = np.load(CURRENT_FILE)


public = S["public_valid"].astype(bool)

sand = S["sand"].astype(float) / 100.0
clay = S["clay"].astype(float) / 100.0
ksat = S["ksat"].astype(float)

theta_wp = S["theta_wp"].astype(float)
theta_fc = S["theta_fc"].astype(float)
theta_sat = S["theta_sat"].astype(float)

theta_surface = C["theta_surface"].astype(float)
theta_deep = C["theta_deep"].astype(float)
score = C["hero_score"].astype(float)


# ------------------------------------------------------------
# Reconstruct Hero Dirt drainage index and optimum
# exactly as in run_soil_model.py
# ------------------------------------------------------------

good_k = public & np.isfinite(ksat) & (ksat > 0)

logk = np.full(ksat.shape, np.nan)

logk[good_k] = np.log10(
    ksat[good_k]
)

klo = np.nanpercentile(
    logk[good_k],
    5,
)

khi = np.nanpercentile(
    logk[good_k],
    95,
)

Knorm = np.clip(
    (logk - klo)
    /
    (khi - klo),
    0.0,
    1.0,
)


sand_valid = sand[public]
clay_valid = clay[public]

s_lo = np.nanpercentile(
    sand_valid,
    5,
)

s_hi = np.nanpercentile(
    sand_valid,
    95,
)

c_lo = np.nanpercentile(
    clay_valid,
    5,
)

c_hi = np.nanpercentile(
    clay_valid,
    95,
)


Sn = np.clip(
    (sand - s_lo)
    /
    (s_hi - s_lo),
    0.0,
    1.0,
)

Cn = np.clip(
    (clay - c_lo)
    /
    (c_hi - c_lo),
    0.0,
    1.0,
)


DH = (
    0.40 * Sn
    +
    0.40 * Knorm
    +
    0.20 * (1.0 - Cn)
)

DH = np.clip(
    DH,
    0.0,
    1.0,
)


F = (
    theta_surface
    -
    theta_wp
) / (
    theta_fc
    -
    theta_wp
)


Fdeep = (
    theta_deep
    -
    theta_wp
) / (
    theta_fc
    -
    theta_wp
)


Fopt = (
    0.88
    +
    0.15 * DH
)


sigma_dry = (
    0.38
    -
    0.08 * DH
)


z = (
    F
    -
    Fopt
) / sigma_dry


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def pct(A, mask=public):

    v = A[
        mask
        &
        np.isfinite(A)
    ]

    return np.nanpercentile(
        v,
        [5, 10, 25, 50, 75, 90, 95]
    )


def print_pct(name, A):

    p = pct(A)

    print(name)
    print(
        "  P05  P10  P25  P50  P75  P90  P95"
    )

    print(
        " ",
        " ".join(
            f"{x:6.3f}"
            for x in p
        )
    )

    print()


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("============================================")
print(" HERO DIRT SOIL-STATE QC")
print("============================================")
print()


print_pct(
    "Surface theta [m3/m3]",
    theta_surface,
)

print_pct(
    "Deep theta [m3/m3]",
    theta_deep,
)

print_pct(
    "Wilting point theta",
    theta_wp,
)

print_pct(
    "Field capacity theta",
    theta_fc,
)

print_pct(
    "Saturation theta",
    theta_sat,
)

print_pct(
    "Surface FC-state F",
    F,
)

print_pct(
    "Deep FC-state F",
    Fdeep,
)

print_pct(
    "Hero optimum Fopt",
    Fopt,
)

print_pct(
    "Normalized dry-side distance z",
    z,
)

print_pct(
    "Hero Dirt score",
    score,
)

print_pct(
    "Drainage index DH",
    DH,
)


# ------------------------------------------------------------
# Fractions relative to hydraulic reference states
# ------------------------------------------------------------

n = np.count_nonzero(public)

below_wp = (
    public
    &
    (theta_surface <= theta_wp)
)

between_wp_fc = (
    public
    &
    (theta_surface > theta_wp)
    &
    (theta_surface < theta_fc)
)

above_fc = (
    public
    &
    (theta_surface >= theta_fc)
)

print("Surface hydraulic-state fractions:")
print(
    f"  <= wilting point: "
    f"{100*np.count_nonzero(below_wp)/n:5.1f}%"
)

print(
    f"  WP to FC:         "
    f"{100*np.count_nonzero(between_wp_fc)/n:5.1f}%"
)

print(
    f"  >= field capacity:"
    f" {100*np.count_nonzero(above_fc)/n:5.1f}%"
)

print()


# ------------------------------------------------------------
# Texture bins
# ------------------------------------------------------------

sand_vals = sand[public]

sand_q1 = np.nanpercentile(
    sand_vals,
    33.333
)

sand_q2 = np.nanpercentile(
    sand_vals,
    66.667
)


texture_groups = [
    (
        "Lower-sand",
        public
        &
        (sand <= sand_q1)
    ),
    (
        "Middle-sand",
        public
        &
        (sand > sand_q1)
        &
        (sand <= sand_q2)
    ),
    (
        "Higher-sand",
        public
        &
        (sand > sand_q2)
    ),
]


print("Representative texture groups:")
print()

for name, mask in texture_groups:

    print(name)

    print(
        f"  cells:          "
        f"{np.count_nonzero(mask):,}"
    )

    print(
        f"  sand mean:      "
        f"{np.nanmean(sand[mask])*100:.1f}%"
    )

    print(
        f"  clay mean:      "
        f"{np.nanmean(clay[mask])*100:.1f}%"
    )

    print(
        f"  theta_surface:  "
        f"{np.nanmean(theta_surface[mask]):.4f}"
    )

    print(
        f"  theta_wp:       "
        f"{np.nanmean(theta_wp[mask]):.4f}"
    )

    print(
        f"  theta_fc:       "
        f"{np.nanmean(theta_fc[mask]):.4f}"
    )

    print(
        f"  F mean:         "
        f"{np.nanmean(F[mask]):.3f}"
    )

    print(
        f"  Fopt mean:      "
        f"{np.nanmean(Fopt[mask]):.3f}"
    )

    print(
        f"  score mean:     "
        f"{np.nanmean(score[mask]):.1f}"
    )

    print()


# ------------------------------------------------------------
# How far below optimum?
# ------------------------------------------------------------

print("Distance from Hero optimum:")
print(
    f"  mean F:      "
    f"{np.nanmean(F[public]):.3f}"
)

print(
    f"  mean Fopt:   "
    f"{np.nanmean(Fopt[public]):.3f}"
)

print(
    f"  mean deficit:"
    f" {np.nanmean((Fopt-F)[public]):.3f}"
)

print()


# ------------------------------------------------------------
# Check whether surface state is pinned against WP
# ------------------------------------------------------------

epsilon = 0.002

near_wp = (
    public
    &
    (
        theta_surface
        <=
        theta_wp
        +
        epsilon
    )
)

print(
    f"Cells within {epsilon:.3f} theta "
    "of wilting point:"
)

print(
    f"  "
    f"{100*np.count_nonzero(near_wp)/n:.1f}%"
)

print()


print("============================================")
print(" QC COMPLETE")
print("============================================")
print()
