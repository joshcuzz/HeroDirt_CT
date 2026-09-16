#!/usr/bin/env python3

from pathlib import Path
import numpy as np


ROOT = Path.home() / "HeroDirt"

SOIL_FILE = (
    ROOT
    / "static"
    / "soils"
    / "HeroDirt_soils_200m.npz"
)

TERRAIN_FILE = (
    ROOT
    / "static"
    / "terrain"
    / "HeroDirt_terrain_200m.npz"
)


S = np.load(SOIL_FILE)
T = np.load(TERRAIN_FILE)


variables = [
    "sand",
    "silt",
    "clay",
    "ksat",
    "bulk_density",
    "theta_fc",
    "theta_wp",
    "theta_sat",
]


print()
print("============================================")
print(" HERO DIRT SOIL QC")
print("============================================")
print()


# ============================================================
# DISTRIBUTIONS
# ============================================================

percentiles = [
    0,
    1,
    5,
    25,
    50,
    75,
    95,
    99,
    100,
]


for name in variables:

    A = S[name]

    good = np.isfinite(A)

    values = A[good]

    print(name)
    print(
        f"  valid: "
        f"{good.sum():,} / {A.size:,} "
        f"({100 * good.mean():.2f}%)"
    )

    if values.size:

        p = np.percentile(
            values,
            percentiles,
        )

        for pp, vv in zip(
            percentiles,
            p,
        ):

            print(
                f"  P{pp:3d}: {vv:.5f}"
            )

    print()


# ============================================================
# KSAT QC
# ============================================================

K = S["ksat"]

finite_k = np.isfinite(K)

zero_k = (
    finite_k
    &
    (K <= 0)
)

positive_k = (
    finite_k
    &
    (K > 0)
)


print("Ksat QC")
print(
    f"  finite Ksat:   "
    f"{finite_k.sum():,}"
)

print(
    f"  Ksat <= 0:     "
    f"{zero_k.sum():,}"
)

print(
    f"  positive Ksat: "
    f"{positive_k.sum():,}"
)

if positive_k.any():

    print(
        f"  minimum positive Ksat: "
        f"{K[positive_k].min():.8f}"
    )

print()


# ============================================================
# SOIL WATER ORDERING
# ============================================================

wp = S["theta_wp"]
fc = S["theta_fc"]
sat = S["theta_sat"]


water_valid = (
    np.isfinite(wp)
    &
    np.isfinite(fc)
    &
    np.isfinite(sat)
)


ordering_good = (
    water_valid
    &
    (wp < fc)
    &
    (fc < sat)
)


ordering_bad = (
    water_valid
    &
    (~ordering_good)
)


print("Soil-water QC")
print(
    f"  complete water triplets: "
    f"{water_valid.sum():,}"
)

print(
    f"  wp < fc < sat:           "
    f"{ordering_good.sum():,}"
)

print(
    f"  invalid ordering:         "
    f"{ordering_bad.sum():,}"
)

if water_valid.any():

    print(
        "  valid-order fraction:      "
        f"{100 * ordering_good.sum() / water_valid.sum():.2f}%"
    )

print()


# ============================================================
# TEXTURE SUM
# ============================================================

sand = S["sand"]
silt = S["silt"]
clay = S["clay"]


texture_valid = (
    np.isfinite(sand)
    &
    np.isfinite(silt)
    &
    np.isfinite(clay)
)


texture_sum = (
    sand
    +
    silt
    +
    clay
)


print("Texture QC")

if texture_valid.any():

    ts = texture_sum[
        texture_valid
    ]

    print(
        f"  mean sand+silt+clay: "
        f"{ts.mean():.3f}%"
    )

    print(
        f"  range: "
        f"{ts.min():.3f} "
        f"to "
        f"{ts.max():.3f}%"
    )

print()


# ============================================================
# COMPLETE MODEL-READY SOIL MASK
# ============================================================

soil_complete = (
    np.isfinite(sand)
    &
    np.isfinite(silt)
    &
    np.isfinite(clay)
    &
    np.isfinite(K)
    &
    np.isfinite(wp)
    &
    np.isfinite(fc)
    &
    np.isfinite(sat)
    &
    ordering_good
)


terrain_valid = (
    np.isfinite(
        T["elevation_mean"]
    )
)


model_valid = (
    soil_complete
    &
    terrain_valid
)


print("Operational coverage")
print(
    f"  complete soils: "
    f"{soil_complete.sum():,} "
    f"({100 * soil_complete.mean():.2f}%)"
)

print(
    f"  valid terrain:  "
    f"{terrain_valid.sum():,} "
    f"({100 * terrain_valid.mean():.2f}%)"
)

print(
    f"  both:           "
    f"{model_valid.sum():,} "
    f"({100 * model_valid.mean():.2f}%)"
)

print()


print("============================================")
print(" SOIL QC COMPLETE")
print("============================================")
print()
