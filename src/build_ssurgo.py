#!/usr/bin/env python3

"""
Hero Dirt Forecast
Build SSURGO soil properties on the 200 m Connecticut grid.

Workflow
--------
1. Load Hero Dirt 200 m master grid.
2. Convert domain to EPSG:3857 for USDA WFS.
3. Download SSURGO MapunitPoly polygons in tiled requests.
4. Merge tiles and remove duplicate polygons.
5. Reproject polygons to EPSG:26956.
6. Query SSURGO tabular component/horizon data.
7. Compute component-percent and horizon-overlap weighted
   0-15 cm soil properties.
8. Rasterize onto exact Hero Dirt 200 m grid.
9. Save GeoTIFFs and NPZ stack.

Soil-water definitions
----------------------
theta_fc  = wthirdbar_r / 100
theta_wp  = wfifteenbar_r / 100
theta_sat = 1 - bulk_density / 2.65

Ksat is retained in SSURGO native ksat_r units.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rasterio

from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box


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

SOIL_DIR = (
    ROOT
    / "static"
    / "soils"
)

SOIL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

POLY_FILE = (
    SOIL_DIR
    / "SSURGO_mapunitpoly.gpkg"
)

OUT_NPZ = (
    SOIL_DIR
    / "HeroDirt_soils_200m.npz"
)


# ============================================================
# USDA SERVICES
# ============================================================

WFS_URL = (
    "https://sdmdataaccess.sc.egov.usda.gov/"
    "Spatial/SDMNAD83Geographic.wfs"
)

TABULAR_URL = (
    "https://sdmdataaccess.sc.egov.usda.gov/"
    "Tabular/post.rest"
)


# ============================================================
# SETTINGS
# ============================================================

DEPTH_TOP_CM = 0.0
DEPTH_BOTTOM_CM = 15.0

TARGET_CRS = CRS.from_epsg(
    26956
)

# USDA WFS query CRS.
WFS_CRS = "EPSG:3857"

# Connecticut statewide bounding box is too large for a
# single USDA WFS request. Split into 3 x 2 tiles.
WFS_NX = 3
WFS_NY = 2

NODATA_FLOAT = -9999.0
NODATA_INT = 0


print()
print("============================================")
print(" HERO DIRT SSURGO BUILD")
print("============================================")
print()


# ============================================================
# LOAD MASTER GRID
# ============================================================

G = np.load(
    GRID_FILE
)

west = float(
    G["west"]
)

east = float(
    G["east"]
)

south = float(
    G["south"]
)

north = float(
    G["north"]
)

width = int(
    G["width"]
)

height = int(
    G["height"]
)

x = G["x"]
y = G["y"]

res = float(
    G["resolution"]
)


transform200 = from_origin(
    west,
    north,
    res,
    res,
)


print("Operational grid:")
print(
    f"  size:       "
    f"{width} x {height}"
)
print(
    f"  resolution: "
    f"{res:.0f} m"
)
print(
    "  CRS:        EPSG:26956"
)
print()


# ============================================================
# HERO DIRT DOMAIN
# ============================================================

domain_utm = gpd.GeoDataFrame(
    {
        "id": [1]
    },
    geometry=[
        box(
            west,
            south,
            east,
            north,
        )
    ],
    crs=TARGET_CRS,
)


# ============================================================
# CONVERT DOMAIN TO EPSG:3857 FOR USDA WFS
# ============================================================

domain_wfs = domain_utm.to_crs(
    WFS_CRS
)

xmin, ymin, xmax, ymax = (
    domain_wfs.total_bounds
)


print(
    "USDA WFS query bounds "
    "[EPSG:3857]:"
)

print(
    f"  x: {xmin:,.1f} "
    f"to {xmax:,.1f}"
)

print(
    f"  y: {ymin:,.1f} "
    f"to {ymax:,.1f}"
)

print()


# ============================================================
# DOWNLOAD SSURGO MAPUNIT POLYGONS
# ============================================================

use_cache = False


if POLY_FILE.exists():

    try:

        test_poly = gpd.read_file(
            POLY_FILE
        )

        if len(test_poly) > 0:

            use_cache = True
            mupoly = test_poly

        else:

            print(
                "Cached polygon file is empty; "
                "re-downloading."
            )

            POLY_FILE.unlink()

    except Exception:

        print(
            "Cached polygon file could not "
            "be read; re-downloading."
        )

        try:

            POLY_FILE.unlink()

        except OSError:

            pass


if use_cache:

    print(
        "Using cached SSURGO polygons:"
    )

    print(
        f"  {POLY_FILE}"
    )

    print()


else:

    print(
        "Downloading SSURGO MapunitPoly "
        "polygons in tiled requests..."
    )

    print()


    # --------------------------------------------------------
    # SPLIT STATEWIDE WFS DOMAIN INTO TILES
    # --------------------------------------------------------

    x_edges = np.linspace(
        xmin,
        xmax,
        WFS_NX + 1,
    )

    y_edges = np.linspace(
        ymin,
        ymax,
        WFS_NY + 1,
    )

    tile_frames = []

    ntile = (
        WFS_NX
        *
        WFS_NY
    )

    itile = 0


    for iy_tile in range(
        WFS_NY
    ):

        for ix_tile in range(
            WFS_NX
        ):

            itile += 1

            txmin = (
                x_edges[
                    ix_tile
                ]
            )

            txmax = (
                x_edges[
                    ix_tile + 1
                ]
            )

            tymin = (
                y_edges[
                    iy_tile
                ]
            )

            tymax = (
                y_edges[
                    iy_tile + 1
                ]
            )

            tile_area = (
                (txmax - txmin)
                *
                (tymax - tymin)
            )


            print(
                f"Tile "
                f"{itile}/{ntile}"
            )

            print(
                f"  x: "
                f"{txmin:,.1f} "
                f"to "
                f"{txmax:,.1f}"
            )

            print(
                f"  y: "
                f"{tymin:,.1f} "
                f"to "
                f"{tymax:,.1f}"
            )

            print(
                f"  area: "
                f"{tile_area / 1.0e9:.2f} "
                f"x10^9 m2"
            )


            # ------------------------------------------------
            # OGC BBOX FILTER
            # ------------------------------------------------

            spatial_filter = f"""
<Filter>
  <BBOX>
    <PropertyName>Geometry</PropertyName>
    <Box srsName="EPSG:3857">
      <coordinates>{txmin},{tymin} {txmax},{tymax}</coordinates>
    </Box>
  </BBOX>
</Filter>
"""


            params = {
                "SERVICE":
                    "WFS",

                "VERSION":
                    "1.1.0",

                "REQUEST":
                    "GetFeature",

                "TYPENAME":
                    "MapunitPoly",

                "FILTER":
                    spatial_filter,

                "SRSNAME":
                    "EPSG:3857",

                "OUTPUTFORMAT":
                    "GML2",

                "MAXFEATURES":
                    "250000",
            }


            r = requests.get(
                WFS_URL,
                params=params,
                timeout=300,
            )


            print(
                "  HTTP status:",
                r.status_code,
            )


            if r.status_code != 200:

                print()
                print(
                    "USDA server response:"
                )

                print(
                    r.text[:5000]
                )

                print()


            r.raise_for_status()


            tmp_gml = (
                SOIL_DIR
                /
                f"SSURGO_tile_"
                f"{itile:02d}.gml"
            )


            tmp_gml.write_bytes(
                r.content
            )


            tile = gpd.read_file(
                tmp_gml
            )


            print(
                f"  features: "
                f"{len(tile):,}"
            )

            print()


            if len(tile) > 0:

                if tile.crs is None:

                    tile = tile.set_crs(
                        WFS_CRS
                    )

                tile_frames.append(
                    tile
                )


            try:

                tmp_gml.unlink()

            except OSError:

                pass


            # Avoid hammering USDA service.
            time.sleep(
                0.5
            )


    # --------------------------------------------------------
    # CHECK TILE DOWNLOAD
    # --------------------------------------------------------

    if len(tile_frames) == 0:

        raise RuntimeError(
            "USDA returned zero MapunitPoly "
            "features for all Connecticut tiles."
        )


    # --------------------------------------------------------
    # MERGE TILE RESULTS
    # --------------------------------------------------------

    mupoly = gpd.GeoDataFrame(
        pd.concat(
            tile_frames,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=WFS_CRS,
    )


    print(
        "Merged downloaded features: "
        f"{len(mupoly):,}"
    )


    # --------------------------------------------------------
    # FIND MUKEY COLUMN
    # --------------------------------------------------------

    cols_lower = {
        c.lower(): c
        for c in mupoly.columns
    }


    if "mukey" not in cols_lower:

        print()
        print(
            "Fields returned by USDA:"
        )

        for c in mupoly.columns:

            print(
                f"  {c}"
            )

        raise RuntimeError(
            "Could not find mukey field."
        )


    mukey_col = (
        cols_lower[
            "mukey"
        ]
    )


    if mukey_col != "mukey":

        mupoly = mupoly.rename(
            columns={
                mukey_col:
                    "mukey"
            }
        )


    mupoly["mukey"] = (
        mupoly["mukey"]
        .astype(str)
    )


    # --------------------------------------------------------
    # REMOVE DUPLICATES FROM TILE BOUNDARIES
    #
    # WFS can return the same full polygon in neighboring
    # BBOX queries. Compare mukey + geometry WKB.
    # --------------------------------------------------------

    before = len(
        mupoly
    )


    mupoly[
        "_geometry_wkb"
    ] = (
        mupoly.geometry
        .apply(
            lambda geom:
                geom.wkb
                if geom is not None
                else None
        )
    )


    mupoly = (
        mupoly
        .drop_duplicates(
            subset=[
                "mukey",
                "_geometry_wkb",
            ]
        )
        .drop(
            columns=[
                "_geometry_wkb"
            ]
        )
        .copy()
    )


    print(
        "Removed duplicate tile features: "
        f"{before - len(mupoly):,}"
    )


    # --------------------------------------------------------
    # CRS INFORMATION
    # --------------------------------------------------------

    print()
    print(
        f"Returned polygon CRS: "
        f"{mupoly.crs}"
    )

    print(
        "Raw polygon bounds:"
    )

    print(
        mupoly.total_bounds
    )


    # --------------------------------------------------------
    # PROJECT TO CONNECTICUT HERO DIRT GRID
    # --------------------------------------------------------

    mupoly = mupoly.to_crs(
        TARGET_CRS
    )


    print()
    print(
        "Projected polygon bounds:"
    )

    print(
        mupoly.total_bounds
    )


    # --------------------------------------------------------
    # CLIP TO OPERATIONAL DOMAIN
    # --------------------------------------------------------

    mupoly = gpd.clip(
        mupoly,
        domain_utm,
    )


    mupoly = mupoly[
        mupoly.geometry.notnull()
        &
        (~mupoly.geometry.is_empty)
    ].copy()


    print()
    print(
        f"Polygons after domain clip: "
        f"{len(mupoly):,}"
    )


    if len(mupoly) == 0:

        raise RuntimeError(
            "Downloaded polygons did not "
            "intersect the Hero Dirt domain "
            "after reprojection."
        )


    # --------------------------------------------------------
    # CACHE POLYGONS
    # --------------------------------------------------------

    mupoly.to_file(
        POLY_FILE,
        driver="GPKG",
    )


    print()
    print(
        "Cached:"
    )

    print(
        f"  {POLY_FILE}"
    )

    print()


# ============================================================
# CLEAN POLYGON SET
# ============================================================

if mupoly.crs != TARGET_CRS:

    mupoly = mupoly.to_crs(
        TARGET_CRS
    )


mupoly = mupoly[
    mupoly.geometry.notnull()
    &
    (~mupoly.geometry.is_empty)
].copy()


mupoly["mukey"] = (
    mupoly["mukey"]
    .astype(str)
)


mukeys = sorted(
    mupoly[
        "mukey"
    ]
    .dropna()
    .unique()
)


print(
    f"SSURGO polygons: "
    f"{len(mupoly):,}"
)

print(
    f"Unique map units: "
    f"{len(mukeys):,}"
)

print()


# ============================================================
# SDA QUERY HELPER
# ============================================================

def query_sda(sql):

    payload = {
        "SERVICE":
            "query",

        "REQUEST":
            "query",

        "QUERY":
            sql,

        "FORMAT":
            "JSON+COLUMNNAME",
    }


    r = requests.post(
        TABULAR_URL,
        data=payload,
        timeout=300,
    )


    if r.status_code != 200:

        print()
        print(
            "Soil Data Access error:"
        )

        print(
            r.text[:4000]
        )

        print()


    r.raise_for_status()


    data = r.json()


    table = data.get(
        "Table",
        [],
    )


    if len(table) <= 1:

        return pd.DataFrame()


    columns = (
        table[0]
    )

    rows = (
        table[1:]
    )


    return pd.DataFrame(
        rows,
        columns=columns,
    )


# ============================================================
# CHUNK HELPER
# ============================================================

def chunks(
    seq,
    n,
):

    for i in range(
        0,
        len(seq),
        n,
    ):

        yield seq[
            i:i+n
        ]


# ============================================================
# QUERY COMPONENT / HORIZON DATA
# ============================================================

frames = []

chunk_size = 250


print(
    "Querying SSURGO "
    "component/horizon data..."
)

print()


for ic, chunk in enumerate(
    chunks(
        mukeys,
        chunk_size,
    ),
    start=1,
):

    keylist = ",".join(
        f"'{k}'"
        for k in chunk
    )


    sql = f"""
SELECT
    mu.mukey,
    mu.musym,
    mu.muname,

    co.cokey,
    co.compname,
    co.comppct_r,

    ch.chkey,
    ch.hzdept_r,
    ch.hzdepb_r,

    ch.sandtotal_r,
    ch.silttotal_r,
    ch.claytotal_r,

    ch.awc_r,
    ch.ksat_r,

    ch.dbthirdbar_r,

    ch.wthirdbar_r,
    ch.wfifteenbar_r

FROM mapunit AS mu

INNER JOIN component AS co
    ON mu.mukey = co.mukey

INNER JOIN chorizon AS ch
    ON co.cokey = ch.cokey

WHERE
    mu.mukey IN ({keylist})

    AND ch.hzdept_r < {DEPTH_BOTTOM_CM}

    AND ch.hzdepb_r > {DEPTH_TOP_CM}

    AND co.comppct_r IS NOT NULL
"""


    df = query_sda(
        sql
    )


    if not df.empty:

        frames.append(
            df
        )


    print(
        f"  chunk "
        f"{ic:3d}: "
        f"{len(chunk):4d} map units, "
        f"{len(df):6d} horizon rows"
    )


    time.sleep(
        0.15
    )


if not frames:

    raise RuntimeError(
        "No SSURGO horizon data "
        "were returned."
    )


soil_raw = pd.concat(
    frames,
    ignore_index=True,
)


print()
print(
    "Total horizon/component rows: "
    f"{len(soil_raw):,}"
)

print()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_cols = [
    "comppct_r",
    "hzdept_r",
    "hzdepb_r",
    "sandtotal_r",
    "silttotal_r",
    "claytotal_r",
    "awc_r",
    "ksat_r",
    "dbthirdbar_r",
    "wthirdbar_r",
    "wfifteenbar_r",
]


for col in numeric_cols:

    soil_raw[col] = pd.to_numeric(
        soil_raw[col],
        errors="coerce",
    )


# ============================================================
# HORIZON OVERLAP WITH 0-15 CM
# ============================================================

top = np.maximum(
    soil_raw[
        "hzdept_r"
    ].to_numpy(),
    DEPTH_TOP_CM,
)


bottom = np.minimum(
    soil_raw[
        "hzdepb_r"
    ].to_numpy(),
    DEPTH_BOTTOM_CM,
)


soil_raw[
    "overlap_cm"
] = np.maximum(
    bottom - top,
    0.0,
)


# ============================================================
# COMPONENT x HORIZON THICKNESS WEIGHT
# ============================================================

soil_raw[
    "base_weight"
] = (
    soil_raw[
        "comppct_r"
    ]
    / 100.0
    *
    soil_raw[
        "overlap_cm"
    ]
)


# ============================================================
# VARIABLES
# ============================================================

properties = [
    "sandtotal_r",
    "silttotal_r",
    "claytotal_r",
    "awc_r",
    "ksat_r",
    "dbthirdbar_r",
    "wthirdbar_r",
    "wfifteenbar_r",
]


def weighted_property(
    group,
    prop,
):

    value = (
        group[prop]
        .to_numpy(
            dtype=float
        )
    )

    weight = (
        group[
            "base_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )


    good = (
        np.isfinite(
            value
        )
        &
        np.isfinite(
            weight
        )
        &
        (weight > 0)
    )


    if not np.any(
        good
    ):

        return np.nan


    return (
        np.sum(
            value[good]
            *
            weight[good]
        )
        /
        np.sum(
            weight[good]
        )
    )


# ============================================================
# AGGREGATE TO MAP UNIT
# ============================================================

print(
    "Aggregating 0-15 cm properties "
    "to map-unit level..."
)


records = []


for mukey, group in soil_raw.groupby(
    "mukey"
):

    rec = {
        "mukey":
            str(mukey)
    }


    for prop in properties:

        rec[prop] = (
            weighted_property(
                group,
                prop,
            )
        )


    records.append(
        rec
    )


soil_mu = pd.DataFrame(
    records
).set_index(
    "mukey"
)


# ============================================================
# DERIVED WATER CONTENTS
# ============================================================

soil_mu[
    "theta_fc"
] = (
    soil_mu[
        "wthirdbar_r"
    ]
    / 100.0
)


soil_mu[
    "theta_wp"
] = (
    soil_mu[
        "wfifteenbar_r"
    ]
    / 100.0
)


soil_mu[
    "theta_sat"
] = (
    1.0
    -
    soil_mu[
        "dbthirdbar_r"
    ]
    / 2.65
)


soil_mu.loc[
    ~soil_mu[
        "theta_sat"
    ].between(
        0.20,
        0.75,
    ),
    "theta_sat",
] = np.nan


# ============================================================
# PROPERTY MAPPING
# ============================================================

property_names = {

    "sand":
        "sandtotal_r",

    "silt":
        "silttotal_r",

    "clay":
        "claytotal_r",

    "awc":
        "awc_r",

    "ksat":
        "ksat_r",

    "bulk_density":
        "dbthirdbar_r",

    "theta_fc":
        "theta_fc",

    "theta_wp":
        "theta_wp",

    "theta_sat":
        "theta_sat",
}


# ============================================================
# RASTERIZE MAP UNIT
# ============================================================

print()
print(
    "Rasterizing SSURGO map units "
    "to 200 m grid..."
)


shapes_mukey = []


for _, row in mupoly.iterrows():

    try:

        value = int(
            row[
                "mukey"
            ]
        )

    except Exception:

        continue


    shapes_mukey.append(
        (
            row.geometry,
            value,
        )
    )


mukey_grid = rasterize(
    shapes=shapes_mukey,
    out_shape=(
        height,
        width,
    ),
    transform=transform200,
    fill=NODATA_INT,
    all_touched=False,
    dtype="int64",
)


# ============================================================
# RASTERIZE SOIL VARIABLES
# ============================================================

soil_grids = {}


for outname, source_col in (
    property_names.items()
):

    lookup = (
        soil_mu[
            source_col
        ]
        .to_dict()
    )


    shapes = []


    for _, row in mupoly.iterrows():

        mk = str(
            row[
                "mukey"
            ]
        )


        value = lookup.get(
            mk,
            np.nan,
        )


        if not np.isfinite(
            value
        ):

            continue


        shapes.append(
            (
                row.geometry,
                float(value),
            )
        )


    A = rasterize(
        shapes=shapes,
        out_shape=(
            height,
            width,
        ),
        transform=transform200,
        fill=NODATA_FLOAT,
        all_touched=False,
        dtype="float32",
    )


    A = A.astype(
        np.float32
    )


    A[
        A == NODATA_FLOAT
    ] = np.nan


    soil_grids[
        outname
    ] = A


    print(
        f"  {outname:15s}: "
        f"{np.isfinite(A).sum():,} cells"
    )


# ============================================================
# GEOTIFF PROFILE
# ============================================================

profile_float = {
    "driver":
        "GTiff",

    "height":
        height,

    "width":
        width,

    "count":
        1,

    "dtype":
        "float32",

    "crs":
        TARGET_CRS,

    "transform":
        transform200,

    "nodata":
        NODATA_FLOAT,

    "compress":
        "deflate",

    "predictor":
        3,

    "tiled":
        True,

    "blockxsize":
        256,

    "blockysize":
        256,
}


def write_float_tif(
    name,
    A,
    description,
):

    path = (
        SOIL_DIR
        /
        f"{name}_200m.tif"
    )


    out = np.where(
        np.isfinite(A),
        A,
        NODATA_FLOAT,
    ).astype(
        np.float32
    )


    with rasterio.open(
        path,
        "w",
        **profile_float,
    ) as dst:

        dst.write(
            out,
            1,
        )

        dst.set_band_description(
            1,
            description,
        )


# ============================================================
# WRITE PROPERTY RASTERS
# ============================================================

descriptions = {

    "sand":
        "Weighted sand content [%]",

    "silt":
        "Weighted silt content [%]",

    "clay":
        "Weighted clay content [%]",

    "awc":
        "Weighted available water capacity "
        "[SSURGO native units]",

    "ksat":
        "Weighted saturated hydraulic conductivity "
        "[SSURGO ksat_r native units]",

    "bulk_density":
        "Bulk density at 1/3 bar [g cm-3]",

    "theta_fc":
        "Field-capacity proxy from wthirdbar_r "
        "[m3 m-3]",

    "theta_wp":
        "Wilting-point proxy from wfifteenbar_r "
        "[m3 m-3]",

    "theta_sat":
        "Estimated saturation water content "
        "[m3 m-3]",
}


print()
print(
    "Writing soil GeoTIFFs..."
)


for name, A in (
    soil_grids.items()
):

    write_float_tif(
        name,
        A,
        descriptions[
            name
        ],
    )

    print(
        f"  saved "
        f"{name}_200m.tif"
    )


# ============================================================
# MUKEY GEOTIFF
# ============================================================

profile_mukey = (
    profile_float.copy()
)

profile_mukey.update(
    dtype="int64",
    nodata=NODATA_INT,
    predictor=2,
)


with rasterio.open(
    SOIL_DIR
    / "mukey_200m.tif",
    "w",
    **profile_mukey,
) as dst:

    dst.write(
        mukey_grid,
        1,
    )

    dst.set_band_description(
        1,
        "SSURGO mukey",
    )


print(
    "  saved mukey_200m.tif"
)


# ============================================================
# SAVE NPZ
# ============================================================

np.savez_compressed(

    OUT_NPZ,

    mukey=mukey_grid,

    sand=soil_grids[
        "sand"
    ],

    silt=soil_grids[
        "silt"
    ],

    clay=soil_grids[
        "clay"
    ],

    awc=soil_grids[
        "awc"
    ],

    ksat=soil_grids[
        "ksat"
    ],

    bulk_density=soil_grids[
        "bulk_density"
    ],

    theta_fc=soil_grids[
        "theta_fc"
    ],

    theta_wp=soil_grids[
        "theta_wp"
    ],

    theta_sat=soil_grids[
        "theta_sat"
    ],

    x=x,
    y=y,

    resolution=np.float64(
        res
    ),

    epsg=np.int32(
        26956
    ),
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "============================================"
)

print(
    " SSURGO SUMMARY"
)

print(
    "============================================"
)

print()


print(
    f"Map units: "
    f"{len(mukeys):,}"
)


print(
    f"Raster cells with mukey: "
    f"{np.count_nonzero(mukey_grid):,}"
)

print()


for name in [
    "sand",
    "silt",
    "clay",
    "ksat",
    "bulk_density",
    "theta_fc",
    "theta_wp",
    "theta_sat",
]:

    A = soil_grids[
        name
    ]


    good = np.isfinite(
        A
    )


    if not np.any(
        good
    ):

        print(
            f"{name:15s}: "
            "NO VALID DATA"
        )

        continue


    print(
        f"{name:15s}: "
        f"{np.nanmin(A):.4f} "
        f"to "
        f"{np.nanmax(A):.4f}, "
        f"mean "
        f"{np.nanmean(A):.4f}"
    )


print()

print(
    "Saved soil stack:"
)

print(
    f"  {OUT_NPZ}"
)


print()

print(
    "============================================"
)

print(
    " HERO DIRT SSURGO BUILD COMPLETE"
)

print(
    "============================================"
)

print()
