#!/usr/bin/env python3

"""
Hero Dirt Forecast
Compare modeled current soil moisture with SSURGO
Atterberg-limit / plasticity properties.

Diagnostics
-----------
Convert modeled volumetric water content theta to approximate
gravimetric water content:

    w = theta * rho_w / rho_b

With rho_w ~ 1 g/cm3 and rho_b in g/cm3:

    w_percent = 100 * theta / rho_b

For plastic soils:

    PL = plastic limit [% gravimetric]
    LL = liquid limit [% gravimetric]
    PI = plasticity index = LL - PL

    LI = (w - PL) / PI

Interpretation:
    LI < 0       below plastic limit
    0 <= LI <=1 within plastic range
    LI > 1       above liquid limit

Important
---------
Atterberg limits describe remolded fine-grained soil behavior.
They are not direct measurements of intact rocky trail tread.

Nonplastic soils are treated separately and are NOT assigned
a meaningful liquidity index.
"""

from pathlib import Path
import numpy as np


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


public = S[
    "public_valid"
].astype(
    bool
)

sand = S[
    "sand"
].astype(
    float
)

clay = S[
    "clay"
].astype(
    float
)

bulk_density = S[
    "bulk_density"
].astype(
    float
)

theta_surface = C[
    "theta_surface"
].astype(
    float
)

theta_deep = C[
    "theta_deep"
].astype(
    float
)


LL = P[
    "liquid_limit"
].astype(
    float
)

PI = P[
    "plasticity_index"
].astype(
    float
)

PL = P[
    "plastic_limit"
].astype(
    float
)


# ============================================================
# CONVERT MODELED THETA TO GRAVIMETRIC WATER CONTENT
# ============================================================

# rho_w ~= 1 g/cm3
#
# theta = Vw / Vtotal
#
# w = Mw / Mdry_soil
#
# approximately:
#
# w_fraction = theta * rho_w / rho_bulk
#
# with rho_w = 1:
#
# w_percent = 100 * theta / rho_bulk

surface_w = (
    100.0
    *
    theta_surface
    /
    bulk_density
)

deep_w = (
    100.0
    *
    theta_deep
    /
    bulk_density
)


# ============================================================
# VALID PLASTICITY MASK
# ============================================================

# Treat PI <= 0 as nonplastic / LI not applicable.

plastic = (
    public
    &
    np.isfinite(LL)
    &
    np.isfinite(PI)
    &
    np.isfinite(PL)
    &
    (PI > 0.0)
    &
    (LL > PL)
    &
    (PL > 0.0)
    &
    np.isfinite(bulk_density)
    &
    (bulk_density > 0.0)
)


nonplastic = (
    public
    &
    np.isfinite(PI)
    &
    (PI <= 0.0)
)


missing_plasticity = (
    public
    &
    ~plastic
    &
    ~nonplastic
)


# ============================================================
# LIQUIDITY INDEX
# ============================================================

LI_surface = np.full(
    theta_surface.shape,
    np.nan,
    dtype=float
)

LI_deep = np.full_like(
    LI_surface,
    np.nan,
)


LI_surface[
    plastic
] = (
    surface_w[
        plastic
    ]
    -
    PL[
        plastic
    ]
) / PI[
    plastic
]


LI_deep[
    plastic
] = (
    deep_w[
        plastic
    ]
    -
    PL[
        plastic
    ]
) / PI[
    plastic
]


# ============================================================
# DISTANCE TO PLASTIC LIMIT
# ============================================================

surface_minus_PL = np.full_like(
    LI_surface,
    np.nan,
)

surface_minus_LL = np.full_like(
    LI_surface,
    np.nan,
)


surface_minus_PL[
    plastic
] = (
    surface_w[
        plastic
    ]
    -
    PL[
        plastic
    ]
)


surface_minus_LL[
    plastic
] = (
    surface_w[
        plastic
    ]
    -
    LL[
        plastic
    ]
)


# ============================================================
# HELPERS
# ============================================================

def percentiles(
    A,
    mask,
):

    vals = A[
        mask
        &
        np.isfinite(A)
    ]

    if len(vals) == 0:

        return None


    return np.nanpercentile(
        vals,
        [
            5,
            10,
            25,
            50,
            75,
            90,
            95,
        ],
    )


def print_percentiles(
    title,
    A,
    mask,
):

    p = percentiles(
        A,
        mask,
    )

    print(title)

    if p is None:

        print(
            "  no valid data"
        )

        print()

        return


    print(
        "  P05  P10  P25  P50  P75  P90  P95"
    )

    print(
        " ",
        " ".join(
            f"{v:7.2f}"
            for v in p
        )
    )

    print()


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT PLASTICITY-STATE QC"
)
print(
    "============================================"
)
print()


npublic = np.count_nonzero(
    public
)

nplastic = np.count_nonzero(
    plastic
)

nnonplastic = np.count_nonzero(
    nonplastic
)

nmissing = np.count_nonzero(
    missing_plasticity
)


print(
    f"Public cells:     {npublic:,}"
)

print(
    f"Plastic cells:    {nplastic:,} "
    f"({100*nplastic/npublic:.1f}%)"
)

print(
    f"Nonplastic cells: {nnonplastic:,} "
    f"({100*nnonplastic/npublic:.1f}%)"
)

print(
    f"Missing/other:    {nmissing:,} "
    f"({100*nmissing/npublic:.1f}%)"
)

print()


# ============================================================
# BASIC WATER CONTENT
# ============================================================

print_percentiles(
    "Surface gravimetric water content [%]",
    surface_w,
    public,
)

print_percentiles(
    "Deep gravimetric water content [%]",
    deep_w,
    public,
)

print_percentiles(
    "Plastic limit PL [%]",
    PL,
    plastic,
)

print_percentiles(
    "Liquid limit LL [%]",
    LL,
    plastic,
)

print_percentiles(
    "Plasticity index PI [%]",
    PI,
    plastic,
)


# ============================================================
# LIQUIDITY INDEX
# ============================================================

print_percentiles(
    "Surface liquidity index LI",
    LI_surface,
    plastic,
)

print_percentiles(
    "Deep liquidity index LI",
    LI_deep,
    plastic,
)

print_percentiles(
    "Surface w - PL [% water content]",
    surface_minus_PL,
    plastic,
)

print_percentiles(
    "Surface w - LL [% water content]",
    surface_minus_LL,
    plastic,
)


# ============================================================
# MECHANICAL STATE FRACTIONS
# ============================================================

below_pl = (
    plastic
    &
    (surface_w < PL)
)

within_plastic = (
    plastic
    &
    (surface_w >= PL)
    &
    (surface_w <= LL)
)

above_ll = (
    plastic
    &
    (surface_w > LL)
)


print(
    "Current surface mechanical state "
    "for plastic soils:"
)

print(
    f"  below PL:          "
    f"{100*np.count_nonzero(below_pl)/nplastic:5.1f}%"
)

print(
    f"  PL to LL:          "
    f"{100*np.count_nonzero(within_plastic)/nplastic:5.1f}%"
)

print(
    f"  above LL:          "
    f"{100*np.count_nonzero(above_ll)/nplastic:5.1f}%"
)

print()


# ============================================================
# CLOSENESS TO PLASTIC LIMIT
# ============================================================

near_pl_dry = (
    plastic
    &
    (surface_w < PL)
    &
    (surface_w >= PL - 2.0)
)

near_pl_wet = (
    plastic
    &
    (surface_w >= PL)
    &
    (surface_w <= PL + 2.0)
)


print(
    "Cells close to plastic limit:"
)

print(
    f"  within 2 %-points below PL: "
    f"{100*np.count_nonzero(near_pl_dry)/nplastic:5.1f}%"
)

print(
    f"  within 2 %-points above PL: "
    f"{100*np.count_nonzero(near_pl_wet)/nplastic:5.1f}%"
)

print()


# ============================================================
# TEXTURE DEPENDENCE
# ============================================================

sand_public = sand[
    public
]

q1 = np.nanpercentile(
    sand_public,
    33.333,
)

q2 = np.nanpercentile(
    sand_public,
    66.667,
)


groups = [
    (
        "Lower-sand",
        public
        &
        (sand <= q1)
    ),
    (
        "Middle-sand",
        public
        &
        (sand > q1)
        &
        (sand <= q2)
    ),
    (
        "Higher-sand",
        public
        &
        (sand > q2)
    ),
]


print(
    "Texture-group mechanical states:"
)

print()


for name, mask0 in groups:

    mask = (
        mask0
        &
        plastic
    )

    n = np.count_nonzero(
        mask
    )


    print(name)

    print(
        f"  plastic cells: "
        f"{n:,}"
    )


    if n == 0:

        print()

        continue


    print(
        f"  sand mean:     "
        f"{np.nanmean(sand[mask]):.1f}%"
    )

    print(
        f"  clay mean:     "
        f"{np.nanmean(clay[mask]):.1f}%"
    )

    print(
        f"  surface w:     "
        f"{np.nanmean(surface_w[mask]):.2f}%"
    )

    print(
        f"  PL mean:       "
        f"{np.nanmean(PL[mask]):.2f}%"
    )

    print(
        f"  LL mean:       "
        f"{np.nanmean(LL[mask]):.2f}%"
    )

    print(
        f"  PI mean:       "
        f"{np.nanmean(PI[mask]):.2f}%"
    )

    print(
        f"  LI mean:       "
        f"{np.nanmean(LI_surface[mask]):.2f}"
    )


    below = np.count_nonzero(
        mask
        &
        (surface_w < PL)
    )

    middle = np.count_nonzero(
        mask
        &
        (surface_w >= PL)
        &
        (surface_w <= LL)
    )

    above = np.count_nonzero(
        mask
        &
        (surface_w > LL)
    )


    print(
        f"  below PL:      "
        f"{100*below/n:5.1f}%"
    )

    print(
        f"  PL to LL:      "
        f"{100*middle/n:5.1f}%"
    )

    print(
        f"  above LL:      "
        f"{100*above/n:5.1f}%"
    )

    print()


# ============================================================
# OPTIONAL SIMPLE MECHANICAL INTERPRETATION
#
# Diagnostic only. This is NOT yet the Hero Dirt score.
# ============================================================

very_dry_mech = (
    plastic
    &
    (LI_surface < -1.0)
)

firm_below_pl = (
    plastic
    &
    (LI_surface >= -1.0)
    &
    (LI_surface < 0.0)
)

plastic_soft = (
    plastic
    &
    (LI_surface >= 0.0)
    &
    (LI_surface <= 1.0)
)

liquid_like = (
    plastic
    &
    (LI_surface > 1.0)
)


print(
    "Diagnostic LI categories "
    "(not Hero Dirt score):"
)

print(
    f"  LI < -1:        "
    f"{100*np.count_nonzero(very_dry_mech)/nplastic:5.1f}%"
)

print(
    f"  -1 <= LI < 0:   "
    f"{100*np.count_nonzero(firm_below_pl)/nplastic:5.1f}%"
)

print(
    f"  0 <= LI <= 1:   "
    f"{100*np.count_nonzero(plastic_soft)/nplastic:5.1f}%"
)

print(
    f"  LI > 1:         "
    f"{100*np.count_nonzero(liquid_like)/nplastic:5.1f}%"
)

print()


print(
    "============================================"
)
print(
    " PLASTICITY QC COMPLETE"
)
print(
    "============================================"
)
print()
