#!/usr/bin/env python3

from pathlib import Path
import numpy as np


ROOT = Path.home() / "HeroDirt"

S = np.load(
    ROOT / "static/model/HeroDirt_static_200m.npz"
)

P = np.load(
    ROOT / "static/soils/HeroDirt_plasticity_200m.npz"
)

C = np.load(
    ROOT / "output/current/HeroDirt_current_state.npz"
)


x = S["x"].astype(float)
y = S["y"].astype(float)

X, Y = np.meshgrid(x, y)


public = S["public_valid"].astype(bool)

theta = C["theta_surface"].astype(float)
deep = C["theta_deep"].astype(float)

wp = S["theta_wp"].astype(float)
fc = S["theta_fc"].astype(float)
sat = S["theta_sat"].astype(float)

sand = S["sand"].astype(float)
clay = S["clay"].astype(float)
ksat = S["ksat"].astype(float)

PI = P["plasticity_index"].astype(float)

F = C["fc_state"].astype(float)
score = C["hero_score"].astype(float)
Hclass = C["hero_class"]


# ------------------------------------------------------------
# Region of suspicious Wet / Too Wet feature
# coordinates are meters in UTM 11N
# ------------------------------------------------------------

region = (
    public
    &
    (X >= 420000)
    &
    (X <= 432000)
    &
    (Y >= 3805000)
    &
    (Y <= 3814000)
)


if not np.any(region):
    raise RuntimeError("No valid cells found in hotspot region.")


# ------------------------------------------------------------
# Maximum-F cell
# ------------------------------------------------------------

candidate = np.where(
    region,
    F,
    np.nan,
)

flat = np.nanargmax(candidate)

r, c = np.unravel_index(
    flat,
    F.shape,
)


print()
print("============================================")
print(" HERO DIRT HOTSPOT DIAGNOSTIC")
print("============================================")
print()

print("Maximum-F cell in selected region:")
print()

print(f"UTM E:       {X[r,c]:.0f} m")
print(f"UTM N:       {Y[r,c]:.0f} m")
print()

print(f"surface theta: {theta[r,c]:.4f}")
print(f"deep theta:    {deep[r,c]:.4f}")
print()

print(f"theta_wp:      {wp[r,c]:.4f}")
print(f"theta_fc:      {fc[r,c]:.4f}")
print(f"theta_sat:     {sat[r,c]:.4f}")
print()

print(f"FC-WP range:   {fc[r,c]-wp[r,c]:.4f}")
print(f"F state:       {F[r,c]:.3f}")
print()

print(f"sand:          {sand[r,c]:.1f}%")
print(f"clay:          {clay[r,c]:.1f}%")
print(f"Ksat:          {ksat[r,c]:.2f}")
print(f"PI:            {PI[r,c]:.2f}")
print()

print(f"Hero score:    {score[r,c]:.1f}")
print(f"Hero class:    {int(Hclass[r,c])}")
print()


# ------------------------------------------------------------
# Regional statistics
# ------------------------------------------------------------

print("Hotspot-region means:")
print()

for name, A in [
    ("surface theta", theta),
    ("WP", wp),
    ("FC", fc),
    ("FC-WP", fc-wp),
    ("F", F),
    ("sand", sand),
    ("clay", clay),
    ("PI", PI),
    ("score", score),
]:

    vals = A[
        region
        &
        np.isfinite(A)
    ]

    print(
        f"{name:14s}: "
        f"mean={np.mean(vals):8.3f}  "
        f"median={np.median(vals):8.3f}"
    )


print()


# ------------------------------------------------------------
# Compare Wet/Too Wet cells specifically
# ------------------------------------------------------------

wet = (
    region
    &
    (Hclass >= 5)
)


print(
    f"Wet/Too Wet cells in region: "
    f"{np.count_nonzero(wet):,}"
)

print()


if np.any(wet):

    print("Wet/Too Wet cell means:")
    print()

    for name, A in [
        ("surface theta", theta),
        ("WP", wp),
        ("FC", fc),
        ("FC-WP", fc-wp),
        ("F", F),
        ("sand", sand),
        ("clay", clay),
        ("PI", PI),
        ("score", score),
    ]:

        vals = A[
            wet
            &
            np.isfinite(A)
        ]

        print(
            f"{name:14s}: "
            f"mean={np.mean(vals):8.3f}  "
            f"median={np.median(vals):8.3f}"
        )


print()
print("============================================")
print(" HOTSPOT QC COMPLETE")
print("============================================")
print()
