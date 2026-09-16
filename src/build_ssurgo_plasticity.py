#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build SSURGO plasticity properties for the upper 0-15 cm.

Retrieves from SSURGO / Soil Data Access:

    ll_r  = liquid limit, representative value [% gravimetric]
    pi_r  = plasticity index, representative value [%]

Derived:

    pl_r  = plastic limit = liquid limit - plasticity index

Aggregation
-----------
Same philosophy as existing Hero Dirt SSURGO processing.

For each map unit:

1. Consider component horizons intersecting 0-15 cm.
2. Weight each horizon by:

       (component percent / 100)
       *
       horizon overlap thickness within 0-15 cm

3. Compute each property using only records for which that
   property is available.

Outputs
-------
static/soils/liquid_limit_200m.tif
static/soils/plasticity_index_200m.tif
static/soils/plastic_limit_200m.tif
static/soils/HeroDirt_plasticity_200m.npz
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import requests
import rasterio

from rasterio.features import rasterize


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt_CT"

GRID_FILE = (
    ROOT
    / "static"
    / "grid"
    / "HeroDirt_grid_200m.npz"
)

MUKEY_TIF = (
    ROOT
    / "static"
    / "soils"
    / "mukey_200m.tif"
)

OUT_DIR = (
    ROOT
    / "static"
    / "soils"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_NPZ = (
    OUT_DIR
    / "HeroDirt_plasticity_200m.npz"
)


# ============================================================
# SDA
# ============================================================

SDA_URL = (
    "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
)


# ============================================================
# LOAD MASTER GRID / MUKEY
# ============================================================

G = np.load(
    GRID_FILE
)

with rasterio.open(
    MUKEY_TIF
) as src:

    mukey_grid = src.read(1)

    profile = src.profile.copy()


valid_mukeys = np.unique(
    mukey_grid[
        mukey_grid > 0
    ]
)

valid_mukeys = valid_mukeys.astype(
    np.int64
)


print()
print("============================================")
print(" HERO DIRT SSURGO PLASTICITY")
print("============================================")
print()

print(
    f"Unique map units: "
    f"{len(valid_mukeys)}"
)

print()


# ============================================================
# QUERY HELPER
# ============================================================

def run_sda_query(
    query,
):

    payload = {
        "query": query,
        "format": "JSON+COLUMNNAME",
    }


    r = requests.post(
        SDA_URL,
        data=payload,
        timeout=180,
    )

    r.raise_for_status()

    data = r.json()


    if (
        "Table" not in data
        or
        len(data["Table"]) == 0
    ):

        return pd.DataFrame()


    rows = data[
        "Table"
    ]


    header = rows[
        0
    ]

    values = rows[
        1:
    ]


    return pd.DataFrame(
        values,
        columns=header,
    )


# ============================================================
# QUERY IN CHUNKS
#
# Avoid making one giant SQL IN clause.
# ============================================================

CHUNK_SIZE = 100

all_rows = []


for i0 in range(
    0,
    len(valid_mukeys),
    CHUNK_SIZE,
):

    subset = valid_mukeys[
        i0:i0 + CHUNK_SIZE
    ]


    mukey_list = ",".join(
        f"'{int(m)}'"
        for m in subset
    )


    query = f"""
    SELECT
        mu.mukey,
        co.cokey,
        co.comppct_r,
        ch.chkey,
        ch.hzdept_r,
        ch.hzdepb_r,
        ch.ll_r,
        ch.pi_r

    FROM mapunit AS mu

    INNER JOIN component AS co
        ON mu.mukey = co.mukey

    INNER JOIN chorizon AS ch
        ON co.cokey = ch.cokey

    WHERE
        mu.mukey IN ({mukey_list})

        AND co.comppct_r IS NOT NULL

        AND ch.hzdept_r IS NOT NULL
        AND ch.hzdepb_r IS NOT NULL

        AND ch.hzdept_r < 15
        AND ch.hzdepb_r > 0

    ORDER BY
        mu.mukey,
        co.cokey,
        ch.hzdept_r
    """


    df = run_sda_query(
        query
    )


    if len(df) > 0:

        all_rows.append(
            df
        )


    print(
        f"Queried "
        f"{min(i0 + CHUNK_SIZE, len(valid_mukeys))}"
        f" / {len(valid_mukeys)} map units"
    )


    time.sleep(
        0.15
    )


if not all_rows:

    raise RuntimeError(
        "No SSURGO plasticity data returned."
    )


D = pd.concat(
    all_rows,
    ignore_index=True,
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_cols = [
    "mukey",
    "comppct_r",
    "hzdept_r",
    "hzdepb_r",
    "ll_r",
    "pi_r",
]


for col in numeric_cols:

    D[col] = pd.to_numeric(
        D[col],
        errors="coerce",
    )


print()
print(
    f"Horizon rows returned: "
    f"{len(D):,}"
)

print()


# ============================================================
# 0-15 CM OVERLAP
# ============================================================

top = np.maximum(
    D["hzdept_r"].values,
    0.0,
)

bottom = np.minimum(
    D["hzdepb_r"].values,
    15.0,
)


overlap = np.maximum(
    bottom - top,
    0.0,
)


component_fraction = (
    D["comppct_r"].values
    /
    100.0
)


weight = (
    component_fraction
    *
    overlap
)


D["weight"] = weight


# ============================================================
# WEIGHTED MAP-UNIT AGGREGATION
# ============================================================

def weighted_property(
    group,
    prop,
):

    values = group[
        prop
    ].to_numpy(
        dtype=float
    )

    weights = group[
        "weight"
    ].to_numpy(
        dtype=float
    )


    good = (
        np.isfinite(values)
        &
        np.isfinite(weights)
        &
        (weights > 0)
    )


    if not np.any(
        good
    ):

        return np.nan


    return np.sum(
        values[good]
        *
        weights[good]
    ) / np.sum(
        weights[good]
    )


records = []


for mukey, group in D.groupby(
    "mukey"
):

    LL = weighted_property(
        group,
        "ll_r",
    )

    PI = weighted_property(
        group,
        "pi_r",
    )


    if (
        np.isfinite(LL)
        and
        np.isfinite(PI)
    ):

        PL = (
            LL
            -
            PI
        )

        # Physical sanity check.
        if (
            PL < 0
            or
            PL > LL
        ):

            PL = np.nan

    else:

        PL = np.nan


    records.append(
        {
            "mukey":
                int(mukey),

            "liquid_limit":
                LL,

            "plasticity_index":
                PI,

            "plastic_limit":
                PL,
        }
    )


M = pd.DataFrame(
    records
)


print(
    f"Map units with aggregation rows: "
    f"{len(M)}"
)

print()


# ============================================================
# LOOKUP DICTIONARIES
# ============================================================

lookup_ll = dict(
    zip(
        M["mukey"],
        M["liquid_limit"],
    )
)

lookup_pi = dict(
    zip(
        M["mukey"],
        M["plasticity_index"],
    )
)

lookup_pl = dict(
    zip(
        M["mukey"],
        M["plastic_limit"],
    )
)


# ============================================================
# MAP TO 200 M GRID
# ============================================================

liquid_limit = np.full(
    mukey_grid.shape,
    np.nan,
    dtype=np.float32,
)

plasticity_index = np.full_like(
    liquid_limit,
    np.nan,
)

plastic_limit = np.full_like(
    liquid_limit,
    np.nan,
)


for mukey in valid_mukeys:

    mask = (
        mukey_grid
        ==
        mukey
    )


    LL = lookup_ll.get(
        int(mukey),
        np.nan,
    )

    PI = lookup_pi.get(
        int(mukey),
        np.nan,
    )

    PL = lookup_pl.get(
        int(mukey),
        np.nan,
    )


    if np.isfinite(
        LL
    ):

        liquid_limit[
            mask
        ] = LL


    if np.isfinite(
        PI
    ):

        plasticity_index[
            mask
        ] = PI


    if np.isfinite(
        PL
    ):

        plastic_limit[
            mask
        ] = PL


# ============================================================
# WRITE GEOTIFF
# ============================================================

NODATA = -9999.0

profile.update(
    dtype="float32",
    nodata=NODATA,
    count=1,
    compress="deflate",
    predictor=3,
)


def write_tif(
    filename,
    A,
    description,
):

    out = np.where(
        np.isfinite(A),
        A,
        NODATA,
    ).astype(
        np.float32
    )


    path = (
        OUT_DIR
        /
        filename
    )


    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            out,
            1,
        )

        dst.set_band_description(
            1,
            description,
        )


    print(
        f"Saved: {path}"
    )


write_tif(
    "liquid_limit_200m.tif",
    liquid_limit,
    "SSURGO liquid limit, 0-15 cm [% gravimetric]",
)

write_tif(
    "plasticity_index_200m.tif",
    plasticity_index,
    "SSURGO plasticity index, 0-15 cm [%]",
)

write_tif(
    "plastic_limit_200m.tif",
    plastic_limit,
    "Derived plastic limit = LL - PI, 0-15 cm [% gravimetric]",
)


# ============================================================
# SAVE NPZ
# ============================================================

np.savez_compressed(

    OUT_NPZ,

    liquid_limit=liquid_limit,

    plasticity_index=plasticity_index,

    plastic_limit=plastic_limit,

    x=G["x"],
    y=G["y"],

    epsg=np.int32(
        26956
    ),
)


# ============================================================
# QC
# ============================================================

total_cells = (
    mukey_grid.size
)


def print_stats(
    name,
    A,
):

    good = np.isfinite(
        A
    )


    print(name)

    print(
        f"  valid cells: "
        f"{np.count_nonzero(good):,} "
        f"("
        f"{100*np.count_nonzero(good)/total_cells:.2f}%"
        f")"
    )


    if np.any(
        good
    ):

        vals = A[
            good
        ]

        p = np.percentile(
            vals,
            [
                5,
                25,
                50,
                75,
                95,
            ],
        )


        print(
            f"  mean: "
            f"{np.mean(vals):.2f}%"
        )

        print(
            f"  P05/P25/P50/P75/P95: "
            +
            " ".join(
                f"{v:.1f}"
                for v in p
            )
        )


    print()


print()
print(
    "============================================"
)
print(
    " PLASTICITY SUMMARY"
)
print(
    "============================================"
)
print()


print_stats(
    "Liquid limit",
    liquid_limit,
)

print_stats(
    "Plasticity index",
    plasticity_index,
)

print_stats(
    "Plastic limit",
    plastic_limit,
)


# ------------------------------------------------------------
# Nonplastic / low plasticity information
# ------------------------------------------------------------

pi_good = np.isfinite(
    plasticity_index
)

if np.any(
    pi_good
):

    zero_pi = (
        pi_good
        &
        (
            plasticity_index
            <=
            0
        )
    )

    low_pi = (
        pi_good
        &
        (
            plasticity_index
            > 0
        )
        &
        (
            plasticity_index
            < 7
        )
    )


    print(
        "PI diagnostic:"
    )

    print(
        f"  PI <= 0: "
        f"{100*np.count_nonzero(zero_pi)/np.count_nonzero(pi_good):.1f}%"
    )

    print(
        f"  0 < PI < 7: "
        f"{100*np.count_nonzero(low_pi)/np.count_nonzero(pi_good):.1f}%"
    )

    print()


print(
    f"Saved NPZ:"
)

print(
    f"  {OUT_NPZ}"
)

print()
print(
    "============================================"
)
print(
    " HERO DIRT PLASTICITY COMPLETE"
)
print(
    "============================================"
)
print()
