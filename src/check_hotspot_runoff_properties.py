#!/usr/bin/env python3

"""
Hero Dirt Forecast
Compare runoff-relevant properties at the MRMS hotspot
with the rest of the public San Gabriel domain.

Loads:
    combined static model for terrain / masks / vegetation
    SSURGO soils NPZ for AWC and soil properties

This is a physical diagnostic only.
"""

from pathlib import Path
import numpy as np


ROOT = Path.home() / "HeroDirt"

S = np.load(
    ROOT / "static/model/HeroDirt_static_200m.npz"
)

SOIL = np.load(
    ROOT / "static/soils/HeroDirt_soils_200m.npz"
)

MRMS = np.load(
    ROOT / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)


# ============================================================
# GRID
# ============================================================

x = S["x"].astype(float)
y = S["y"].astype(float)

X, Y = np.meshgrid(
    x,
    y,
)

public = S[
    "public_valid"
].astype(bool)


# ============================================================
# FIND MRMS HOTSPOT
# ============================================================

precip = MRMS[
    "precip_mm"
].astype(float)

accum = np.nansum(
    precip,
    axis=0,
)


region = (
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


candidate = np.where(
    region,
    accum,
    np.nan,
)


flat = np.nanargmax(
    candidate
)


r, c = np.unravel_index(
    flat,
    accum.shape,
)


# ============================================================
# VARIABLES
# ============================================================

variables = {

    "Elevation [m]":
        S["elevation"].astype(float),

    "Slope [deg]":
        S["slope"].astype(float),

    "Sand [%]":
        S["sand"].astype(float),

    "Clay [%]":
        S["clay"].astype(float),

    "Ksat":
        S["ksat"].astype(float),

    "AWC":
        SOIL["awc"].astype(float),

    "Field capacity":
        S["theta_fc"].astype(float),

    "Wilting point":
        S["theta_wp"].astype(float),

    "Tree cover":
        S["tree_fraction"].astype(float),

    "Shrub cover":
        S["shrub_cover_fraction"].astype(float),

    "Herb cover":
        S["herb_cover_fraction"].astype(float),

    "Impervious [%]":
        S["impervious_mean"].astype(float),
}


# ============================================================
# PERCENTILE RANK
# ============================================================

def percentile_rank(
    value,
    vals,
):

    vals = vals[
        np.isfinite(vals)
    ]

    if len(vals) == 0:
        return np.nan

    return (
        100.0
        *
        np.count_nonzero(
            vals <= value
        )
        /
        len(vals)
    )


# ============================================================
# PRINT HOTSPOT PROPERTIES
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT HOTSPOT RUNOFF PROPERTIES"
)
print(
    "============================================"
)
print()


print(
    "Hotspot location:"
)

print(
    f"  E = {X[r,c]:.0f} m"
)

print(
    f"  N = {Y[r,c]:.0f} m"
)

print(
    f"  MRMS accumulation = "
    f"{accum[r,c]:.2f} mm"
)

print()


print(
    "Hotspot property vs public-domain distribution:"
)

print()

print(
    f"{'Variable':24s} "
    f"{'Hotspot':>10s} "
    f"{'Median':>10s} "
    f"{'Pctl':>8s}"
)

print(
    "-" * 56
)


for name, A in variables.items():

    value = A[
        r,
        c
    ]

    vals = A[
        public
        &
        np.isfinite(A)
    ]

    median = np.nanmedian(
        vals
    )

    pct = percentile_rank(
        value,
        vals,
    )

    print(
        f"{name:24s} "
        f"{value:10.3f} "
        f"{median:10.3f} "
        f"{pct:7.1f}%"
    )


# ============================================================
# LOCAL 3x3 NEIGHBORHOOD
# ============================================================

r0 = max(
    r - 1,
    0
)

r1 = min(
    r + 2,
    public.shape[0]
)

c0 = max(
    c - 1,
    0
)

c1 = min(
    c + 2,
    public.shape[1]
)


print()
print(
    "Local 3x3-cell neighborhood means:"
)

print()


for name, A in variables.items():

    block = A[
        r0:r1,
        c0:c1
    ]

    good = np.isfinite(
        block
    )

    if np.any(
        good
    ):

        local_mean = np.nanmean(
            block
        )

        print(
            f"  {name:24s}: "
            f"{local_mean:.3f}"
        )


# ============================================================
# SIMPLE RUNOFF-PRONENESS CONTEXT
#
# Diagnostic only:
#
# high slope
# high clay
# low Ksat
# low AWC
#
# This is NOT a runoff equation.
# ============================================================

slope = S[
    "slope"
].astype(float)

ksat = S[
    "ksat"
].astype(float)

clay = S[
    "clay"
].astype(float)

awc = SOIL[
    "awc"
].astype(float)


def robust_normalize(
    A,
    mask,
):

    vals = A[
        mask
        &
        np.isfinite(A)
    ]

    p05 = np.nanpercentile(
        vals,
        5,
    )

    p95 = np.nanpercentile(
        vals,
        95,
    )

    out = (
        A - p05
    ) / (
        p95 - p05
    )

    return np.clip(
        out,
        0.0,
        1.0,
    )


Sn = robust_normalize(
    slope,
    public,
)

log_ksat = np.full(
    ksat.shape,
    np.nan,
    dtype=float,
)

good = (
    np.isfinite(ksat)
    &
    (ksat > 0)
)

log_ksat[good] = np.log10(
    ksat[good]
)


Kn = robust_normalize(
    log_ksat,
    public,
)


Cn = robust_normalize(
    clay,
    public,
)


An = robust_normalize(
    awc,
    public,
)


runoff_context = (
    0.30 * Sn
    +
    0.25 * Cn
    +
    0.30 * (1.0 - Kn)
    +
    0.15 * (1.0 - An)
)


hotspot_context = runoff_context[
    r,
    c
]


context_vals = runoff_context[
    public
    &
    np.isfinite(
        runoff_context
    )
]


context_pct = percentile_rank(
    hotspot_context,
    context_vals,
)


print()
print(
    "Runoff-proneness context index"
)

print(
    "(diagnostic only, not a runoff equation):"
)

print(
    f"  hotspot value: "
    f"{hotspot_context:.3f}"
)

print(
    f"  domain median: "
    f"{np.nanmedian(context_vals):.3f}"
)

print(
    f"  hotspot percentile: "
    f"{context_pct:.1f}%"
)

print()


print(
    "============================================"
)
print(
    " RUNOFF PROPERTY QC COMPLETE"
)
print(
    "============================================"
)
print()
