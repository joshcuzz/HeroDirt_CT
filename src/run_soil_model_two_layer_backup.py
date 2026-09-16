#!/usr/bin/env python3

"""
Hero Dirt Forecast
Two-layer soil-moisture model + Hero Dirt score v0.3.

Inputs
------
static/model/HeroDirt_static_200m.npz

static/soils/HeroDirt_plasticity_200m.npz

data/smap/processed/HeroDirt_SMAP_latest.npz
data/mrms/processed/HeroDirt_MRMS_hourly.npz
data/nws/processed/HeroDirt_NWS_6hourly.npz

Outputs
-------
output/current/
    HeroDirt_current_state.npz

    surface_theta_now.tif
    deep_theta_now.tif
    surface_fc_state_now.tif
    hero_score_now.tif
    hero_class_now.tif

output/forecast/
    HeroDirt_forecast_6hourly.npz

    hero_score_024h.tif
    hero_score_048h.tif
    hero_score_072h.tif
    hero_score_096h.tif
    hero_score_120h.tif


Model structure
---------------
Surface layer:
    0-5 cm

Deep layer:
    5-20 cm


Hydrology
---------
Conceptual / empirical two-layer bucket model.

Soil hydraulic reference states come from SSURGO:

    theta_wp
    theta_fc
    theta_sat

Current-state initialization is constrained by SMAP.

Observed precipitation comes from MRMS.

Forecast meteorology comes from NWS.


Hero Dirt score v0.3
--------------------
The rideability score is separated from the hydrology model.

Hydraulic moisture state:

    F = (theta - theta_wp) / (theta_fc - theta_wp)

Plasticity index is used as a wet-deformation sensitivity,
not as a literal trail-moisture threshold.

See:

    src/hero_score.py
"""

from pathlib import Path

import numpy as np
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin


from hero_score import (
    calculate_hero_score,
    CLASS_NAMES,
)


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


SMAP_FILE = (
    ROOT
    / "data"
    / "smap"
    / "processed"
    / "HeroDirt_SMAP_latest.npz"
)


MRMS_FILE = (
    ROOT
    / "data"
    / "mrms"
    / "processed"
    / "HeroDirt_MRMS_hourly.npz"
)


NWS_FILE = (
    ROOT
    / "data"
    / "nws"
    / "processed"
    / "HeroDirt_NWS_6hourly.npz"
)


CURRENT_DIR = (
    ROOT
    / "output"
    / "current"
)


FORECAST_DIR = (
    ROOT
    / "output"
    / "forecast"
)


CURRENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CURRENT_FILE = (
    CURRENT_DIR
    / "HeroDirt_current_state.npz"
)


FORECAST_FILE = (
    FORECAST_DIR
    / "HeroDirt_forecast_6hourly.npz"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

# Layer thicknesses [mm]
#
# 0-5 cm:
ZSURF = 50.0

# 5-20 cm:
ZDEEP = 150.0


# ------------------------------------------------------------
# Prototype empirical hydrology rates [mm/day]
# ------------------------------------------------------------

INFIL_BASE_DAY = 8.0

DEEP_DRAIN_BASE_DAY = 7.0

ET_SURF_BASE_DAY = 1.5

ET_DEEP_BASE_DAY = 0.8


# Forecast timestep [hours]
DT_FORECAST_H = 6.0


# ============================================================
# LOAD INPUT DATA
# ============================================================

S = np.load(
    STATIC_FILE
)


PLASTICITY = np.load(
    PLASTICITY_FILE
)


SMAP = np.load(
    SMAP_FILE
)


MRMS = np.load(
    MRMS_FILE
)


NWS = np.load(
    NWS_FILE
)


# ============================================================
# GRID
# ============================================================

x = S[
    "x"
]


y = S[
    "y"
]


elevation = S[
    "elevation"
]


ny, nx = elevation.shape


res = float(
    S[
        "resolution"
    ]
)


TARGET_CRS = CRS.from_epsg(
    26911
)


west = (
    float(
        x[0]
    )
    -
    res / 2.0
)


north = (
    float(
        y[0]
    )
    +
    res / 2.0
)


transform = from_origin(
    west,
    north,
    res,
    res,
)


physics_valid = (
    S[
        "physics_valid"
    ]
    .astype(
        bool
    )
)


public_valid = (
    S[
        "public_valid"
    ]
    .astype(
        bool
    )
)


# ============================================================
# STATIC SOILS
# ============================================================

theta_wp = S[
    "theta_wp"
].astype(
    np.float64
)


theta_fc = S[
    "theta_fc"
].astype(
    np.float64
)


theta_sat = S[
    "theta_sat"
].astype(
    np.float64
)


sand = (
    S[
        "sand"
    ].astype(
        np.float64
    )
    /
    100.0
)


clay = (
    S[
        "clay"
    ].astype(
        np.float64
    )
    /
    100.0
)


ksat = S[
    "ksat"
].astype(
    np.float64
)


bulk_density = S[
    "bulk_density"
].astype(
    np.float64
)


plasticity_index = (
    PLASTICITY[
        "plasticity_index"
    ].astype(
        np.float64
    )
)


# ============================================================
# TERRAIN
# ============================================================

slope = S[
    "slope"
].astype(
    np.float64
)


southness = S[
    "southness"
].astype(
    np.float64
)


# ============================================================
# VEGETATION
# ============================================================

tree_fraction = S[
    "tree_fraction"
].astype(
    np.float64
)


shrub_fraction = S[
    "shrub_cover_fraction"
].astype(
    np.float64
)


herb_fraction = S[
    "herb_cover_fraction"
].astype(
    np.float64
)


# ============================================================
# STATIC EMPIRICAL HYDROLOGY FACTORS
# ============================================================

# ------------------------------------------------------------
# Ksat factor
#
# Hero Dirt empirical transformation.
# ------------------------------------------------------------

logk = np.full(
    ksat.shape,
    np.nan,
    dtype=np.float64,
)


good_k = (
    physics_valid
    &
    np.isfinite(
        ksat
    )
    &
    (
        ksat > 0
    )
)


logk[
    good_k
] = np.log10(
    ksat[
        good_k
    ]
)


klo = np.nanpercentile(
    logk[
        good_k
    ],
    5,
)


khi = np.nanpercentile(
    logk[
        good_k
    ],
    95,
)


Knorm = (
    logk
    -
    klo
) / (
    khi
    -
    klo
)


Knorm = np.clip(
    Knorm,
    0.0,
    1.0,
)


Kfactor = (
    0.6
    +
    0.8
    *
    Knorm
)


# ------------------------------------------------------------
# Texture factor
#
# Hero Dirt empirical scaling.
# ------------------------------------------------------------

texture_factor = (
    1.0
    +
    0.50
    *
    sand
    -
    0.80
    *
    clay
)


texture_factor = np.clip(
    texture_factor,
    0.5,
    1.5,
)


# ------------------------------------------------------------
# Slope factor
# ------------------------------------------------------------

slope_factor = (
    0.8
    +
    0.5
    *
    np.minimum(
        slope,
        45.0,
    )
    /
    45.0
)


# ------------------------------------------------------------
# Aspect drying factor
# ------------------------------------------------------------

aspect_factor = (
    1.0
    +
    0.35
    *
    southness
)


aspect_factor = np.clip(
    aspect_factor,
    0.65,
    1.35,
)


# ------------------------------------------------------------
# Vegetation drying factor
#
# First-order empirical modifier.
# ------------------------------------------------------------

vegetation_factor = (
    1.0
    -
    0.55
    *
    tree_fraction
    -
    0.30
    *
    shrub_fraction
    -
    0.15
    *
    herb_fraction
)


vegetation_factor = np.clip(
    vegetation_factor,
    0.45,
    1.0,
)


# ============================================================
# SMAP INITIALIZATION
# ============================================================

smap_mean = float(
    SMAP[
        "mean_soil_moisture"
    ]
)


smap_time = (
    SMAP[
        "observation_time_utc"
    ]
    .astype(
        "datetime64[s]"
    )
)


print()
print(
    "============================================"
)
print(
    " HERO DIRT SOIL MODEL v0.3"
)
print(
    "============================================"
)
print()


print(
    f"SMAP observation time: "
    f"{smap_time}"
)


print(
    f"SMAP regional mean:    "
    f"{smap_mean:.4f}"
)


print()


# ============================================================
# INITIAL SURFACE STATE
#
# Keep current initialization unchanged for now.
#
# SMAP controls the regional mean.
# SSURGO FC structure gives spatial variation.
#
# We can revisit the downscaling separately later.
# ============================================================

fc_mean = np.nanmean(
    theta_fc[
        physics_valid
    ]
)


fc_anomaly = (
    theta_fc
    -
    fc_mean
)


theta_surface = (
    smap_mean
    +
    0.35
    *
    fc_anomaly
)


# Force regional mean back to SMAP.
surface_mean0 = np.nanmean(
    theta_surface[
        physics_valid
    ]
)


theta_surface += (
    smap_mean
    -
    surface_mean0
)


# Physical bounds
theta_surface = np.maximum(
    theta_surface,
    theta_wp,
)


theta_surface = np.minimum(
    theta_surface,
    theta_sat,
)


theta_surface[
    ~physics_valid
] = np.nan


# ============================================================
# INITIAL DEEP STATE
# ============================================================

theta_deep = (
    theta_wp
    +
    0.70
    *
    (
        theta_fc
        -
        theta_wp
    )
)


theta_deep = np.maximum(
    theta_deep,
    theta_surface
    -
    0.01,
)


theta_deep = np.minimum(
    theta_deep,
    theta_sat,
)


theta_deep[
    ~physics_valid
] = np.nan


# ============================================================
# HYDROLOGY STEP
# ============================================================

def model_step(
    surf,
    deep,
    precip_mm,
    temperature_c,
    rh,
    wind_ms,
    cloud_percent,
    dt_hours,
):

    """
    Advance two-layer soil model.

    Returns
    -------
    surface theta
    deep theta
    surface->deep transfer [mm]
    deep drainage [mm]
    surface ET [mm]
    deep ET [mm]
    runoff [mm]
    """

    dt_day = (
        dt_hours
        /
        24.0
    )


    # --------------------------------------------------------
    # Convert volumetric moisture to water storage [mm]
    # --------------------------------------------------------

    Wsurf = (
        surf
        *
        ZSURF
    )


    Wdeep = (
        deep
        *
        ZDEEP
    )


    Wsurf_wp = (
        theta_wp
        *
        ZSURF
    )


    Wsurf_fc = (
        theta_fc
        *
        ZSURF
    )


    Wsurf_sat = (
        theta_sat
        *
        ZSURF
    )


    Wdeep_wp = (
        theta_wp
        *
        ZDEEP
    )


    Wdeep_fc = (
        theta_fc
        *
        ZDEEP
    )


    Wdeep_sat = (
        theta_sat
        *
        ZDEEP
    )


    # --------------------------------------------------------
    # Add precipitation
    # --------------------------------------------------------

    P = np.where(
        np.isfinite(
            precip_mm
        ),
        precip_mm,
        0.0,
    )


    Wsurf += P


    # --------------------------------------------------------
    # Saturation-excess runoff
    # --------------------------------------------------------

    runoff = np.maximum(
        Wsurf
        -
        Wsurf_sat,
        0.0,
    )


    Wsurf = np.minimum(
        Wsurf,
        Wsurf_sat,
    )


    # --------------------------------------------------------
    # Surface -> deep transfer
    #
    # Active only above field capacity.
    # --------------------------------------------------------

    surface_excess_ratio = (
        Wsurf
        -
        Wsurf_fc
    ) / (
        Wsurf_sat
        -
        Wsurf_fc
    )


    surface_excess_ratio = np.clip(
        surface_excess_ratio,
        0.0,
        1.0,
    )


    transfer_capacity = (
        INFIL_BASE_DAY
        *
        dt_day
        *
        Kfactor
        *
        texture_factor
        *
        slope_factor
    )


    transfer_potential = (
        transfer_capacity
        *
        surface_excess_ratio
    )


    above_fc = np.maximum(
        Wsurf
        -
        Wsurf_fc,
        0.0,
    )


    deep_space = np.maximum(
        Wdeep_sat
        -
        Wdeep,
        0.0,
    )


    infiltration = np.minimum(
        transfer_potential,
        above_fc,
    )


    infiltration = np.minimum(
        infiltration,
        deep_space,
    )


    infiltration = np.maximum(
        infiltration,
        0.0,
    )


    Wsurf -= infiltration

    Wdeep += infiltration


    # --------------------------------------------------------
    # Deep drainage
    # --------------------------------------------------------

    deep_excess_ratio = (
        Wdeep
        -
        Wdeep_fc
    ) / (
        Wdeep_sat
        -
        Wdeep_fc
    )


    deep_excess_ratio = np.clip(
        deep_excess_ratio,
        0.0,
        1.0,
    )


    drainage_capacity = (
        DEEP_DRAIN_BASE_DAY
        *
        dt_day
        *
        Kfactor
        *
        texture_factor
    )


    deep_drainage = (
        drainage_capacity
        *
        deep_excess_ratio
    )


    deep_above_fc = np.maximum(
        Wdeep
        -
        Wdeep_fc,
        0.0,
    )


    deep_drainage = np.minimum(
        deep_drainage,
        deep_above_fc,
    )


    deep_drainage = np.maximum(
        deep_drainage,
        0.0,
    )


    Wdeep -= deep_drainage


    # --------------------------------------------------------
    # Meteorological drying factors
    #
    # Empirical; not Penman-Monteith.
    # --------------------------------------------------------

    Tfactor = np.clip(
        (
            temperature_c
            -
            5.0
        )
        /
        20.0,
        0.2,
        1.5,
    )


    RHfactor = np.clip(
        1.0
        -
        rh
        /
        100.0,
        0.10,
        1.0,
    )


    wind_factor = np.clip(
        0.7
        +
        0.15
        *
        wind_ms,
        0.7,
        1.4,
    )


    cloud_factor = np.clip(
        1.0
        -
        0.50
        *
        cloud_percent
        /
        100.0,
        0.5,
        1.0,
    )


    # --------------------------------------------------------
    # Surface moisture availability
    # --------------------------------------------------------

    moisture_factor = (
        Wsurf
        -
        Wsurf_wp
    ) / (
        Wsurf_fc
        -
        Wsurf_wp
    )


    moisture_factor = np.clip(
        moisture_factor,
        0.0,
        1.0,
    )


    # --------------------------------------------------------
    # Surface ET / drying
    # --------------------------------------------------------

    ETsurf = (
        ET_SURF_BASE_DAY
        *
        dt_day
        *
        Tfactor
        *
        RHfactor
        *
        wind_factor
        *
        cloud_factor
        *
        aspect_factor
        *
        vegetation_factor
        *
        moisture_factor
    )


    available_surface = np.maximum(
        Wsurf
        -
        Wsurf_wp,
        0.0,
    )


    ETsurf = np.minimum(
        ETsurf,
        available_surface,
    )


    ETsurf = np.maximum(
        ETsurf,
        0.0,
    )


    Wsurf -= ETsurf


    # --------------------------------------------------------
    # Deep-layer drying / transpiration proxy
    # --------------------------------------------------------

    deep_moisture_factor = (
        Wdeep
        -
        Wdeep_wp
    ) / (
        Wdeep_fc
        -
        Wdeep_wp
    )


    deep_moisture_factor = np.clip(
        deep_moisture_factor,
        0.0,
        1.0,
    )


    ETdeep = (
        ET_DEEP_BASE_DAY
        *
        dt_day
        *
        Tfactor
        *
        RHfactor
        *
        wind_factor
        *
        deep_moisture_factor
    )


    available_deep = np.maximum(
        Wdeep
        -
        Wdeep_wp,
        0.0,
    )


    ETdeep = np.minimum(
        ETdeep,
        available_deep,
    )


    ETdeep = np.maximum(
        ETdeep,
        0.0,
    )


    Wdeep -= ETdeep


    # --------------------------------------------------------
    # Convert storage back to theta
    # --------------------------------------------------------

    surf_new = (
        Wsurf
        /
        ZSURF
    )


    deep_new = (
        Wdeep
        /
        ZDEEP
    )


    surf_new = np.maximum(
        surf_new,
        theta_wp,
    )


    surf_new = np.minimum(
        surf_new,
        theta_sat,
    )


    deep_new = np.maximum(
        deep_new,
        theta_wp,
    )


    deep_new = np.minimum(
        deep_new,
        theta_sat,
    )


    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    arrays = [
        surf_new,
        deep_new,
        infiltration,
        deep_drainage,
        ETsurf,
        ETdeep,
        runoff,
    ]


    for A in arrays:

        A[
            ~physics_valid
        ] = np.nan


    return (
        surf_new,
        deep_new,
        infiltration,
        deep_drainage,
        ETsurf,
        ETdeep,
        runoff,
    )


# ============================================================
# HERO DIRT SCORE WRAPPER
# ============================================================

def hero_score(
    surf,
    deep,
):

    return calculate_hero_score(

        surf=surf,

        deep=deep,

        theta_wp=theta_wp,

        theta_fc=theta_fc,

        theta_sat=theta_sat,

        plasticity_index=plasticity_index,

        public_valid=public_valid,
    )


# ============================================================
# FORECAST TIMING
# ============================================================

forecast_times = (
    NWS[
        "time"
    ]
    .astype(
        "datetime64[s]"
    )
)


first_forecast_end = (
    forecast_times[
        0
    ]
)


forecast_start = (
    first_forecast_end
    -
    np.timedelta64(
        6,
        "h",
    )
)


print(
    f"Forecast integration start: "
    f"{forecast_start}"
)


print(
    f"Forecast final time:        "
    f"{forecast_times[-1]}"
)


print()


# ============================================================
# OBSERVED UPDATE:
# SMAP -> FORECAST START
# ============================================================

analysis_dt_seconds = (
    forecast_start
    -
    smap_time
).astype(
    "timedelta64[s]"
).astype(
    np.int64
)


analysis_dt_hours = max(
    0.0,
    analysis_dt_seconds
    /
    3600.0,
)


print(
    f"SMAP-to-forecast-start interval: "
    f"{analysis_dt_hours:.2f} h"
)


# ============================================================
# MRMS ANALYSIS PRECIPITATION
# ============================================================

mrms_time = (
    MRMS[
        "time"
    ]
    .astype(
        "datetime64[s]"
    )
)


mrms_precip = (
    MRMS[
        "precip_mm"
    ]
    .astype(
        np.float64
    )
)


use_mrms = (
    (
        mrms_time
        >
        smap_time
    )
    &
    (
        mrms_time
        <=
        forecast_start
    )
)


if np.any(
    use_mrms
):

    P_analysis = np.sum(
        mrms_precip[
            use_mrms,
            :,
            :
        ],
        axis=0,
    )

else:

    P_analysis = np.zeros(
        (
            ny,
            nx,
        ),
        dtype=np.float64,
    )


print(
    f"MRMS hourly fields used in analysis: "
    f"{np.count_nonzero(use_mrms)}"
)


print(
    f"Domain-mean analysis precipitation: "
    f"{np.nanmean(P_analysis[physics_valid]):.3f} mm"
)


print()


# ============================================================
# ANALYSIS METEOROLOGY
#
# Use first NWS block as short-period proxy.
# ============================================================

T_analysis = (
    NWS[
        "temperature_c"
    ][
        0
    ]
    .astype(
        np.float64
    )
)


RH_analysis = (
    NWS[
        "relative_humidity"
    ][
        0
    ]
    .astype(
        np.float64
    )
)


U_analysis = (
    NWS[
        "wind_speed_ms"
    ][
        0
    ]
    .astype(
        np.float64
    )
)


C_analysis = (
    NWS[
        "sky_cover_percent"
    ][
        0
    ]
    .astype(
        np.float64
    )
)


# ============================================================
# ADVANCE TO CURRENT / FORECAST START
# ============================================================

if analysis_dt_hours > 0:

    (
        theta_surface,
        theta_deep,
        infiltration_analysis,
        drainage_analysis,
        ETsurf_analysis,
        ETdeep_analysis,
        runoff_analysis,
    ) = model_step(

        theta_surface,
        theta_deep,

        P_analysis,

        T_analysis,
        RH_analysis,
        U_analysis,
        C_analysis,

        analysis_dt_hours,
    )

else:

    shape = (
        ny,
        nx,
    )


    infiltration_analysis = np.zeros(
        shape,
        dtype=float,
    )


    drainage_analysis = np.zeros(
        shape,
        dtype=float,
    )


    ETsurf_analysis = np.zeros(
        shape,
        dtype=float,
    )


    ETdeep_analysis = np.zeros(
        shape,
        dtype=float,
    )


    runoff_analysis = np.zeros(
        shape,
        dtype=float,
    )


# ============================================================
# CURRENT HERO DIRT SCORE v0.3
# ============================================================

current_score = hero_score(
    theta_surface,
    theta_deep,
)


score_now = current_score[
    "score"
]


class_now = current_score[
    "condition"
]


fc_state_now = current_score[
    "F"
]


deep_fc_state_now = current_score[
    "Fdeep"
]


pi_sensitivity_now = current_score[
    "pi_sensitivity"
]


dry_suitability_now = current_score[
    "dry_suitability"
]


wet_suitability_now = current_score[
    "wet_suitability"
]


deep_suitability_now = current_score[
    "deep_suitability"
]


print(
    "Current reconstructed state:"
)


print(
    f"  surface theta mean: "
    f"{np.nanmean(theta_surface[public_valid]):.4f}"
)


print(
    f"  deep theta mean:    "
    f"{np.nanmean(theta_deep[public_valid]):.4f}"
)


print(
    f"  FC state mean:      "
    f"{np.nanmean(fc_state_now[public_valid]):.3f}"
)


print(
    f"  Hero score mean:    "
    f"{np.nanmean(score_now[public_valid]):.1f}"
)


print()


# ============================================================
# SAVE CURRENT STATE
# ============================================================

np.savez_compressed(

    CURRENT_FILE,

    time=forecast_start,

    theta_surface=theta_surface.astype(
        np.float32
    ),

    theta_deep=theta_deep.astype(
        np.float32
    ),

    fc_state=fc_state_now.astype(
        np.float32
    ),

    deep_fc_state=deep_fc_state_now.astype(
        np.float32
    ),

    hero_score=score_now.astype(
        np.float32
    ),

    hero_class=class_now,

    pi_sensitivity=pi_sensitivity_now.astype(
        np.float32
    ),

    dry_suitability=dry_suitability_now.astype(
        np.float32
    ),

    wet_suitability=wet_suitability_now.astype(
        np.float32
    ),

    deep_suitability=deep_suitability_now.astype(
        np.float32
    ),

    smap_time=smap_time,

    smap_mean=np.float64(
        smap_mean
    ),

    analysis_precipitation=P_analysis.astype(
        np.float32
    ),
)


# ============================================================
# FORECAST ARRAYS
# ============================================================

nt = len(
    forecast_times
)


shape3 = (
    nt,
    ny,
    nx,
)


surf_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


deep_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


score_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


fc_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


deep_fc_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


class_forecast = np.zeros(
    shape3,
    dtype=np.uint8,
)


infiltration_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


drainage_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


ETsurf_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


ETdeep_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


runoff_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


wet_suitability_forecast = np.full(
    shape3,
    np.nan,
    dtype=np.float32,
)


# ============================================================
# RUN FORECAST
# ============================================================

surf = theta_surface.copy()

deep = theta_deep.copy()


print(
    "Running 120-hour forecast..."
)


print()


for it in range(
    nt
):

    P = (
        NWS[
            "precipitation_mm"
        ][
            it
        ]
        .astype(
            np.float64
        )
    )


    T = (
        NWS[
            "temperature_c"
        ][
            it
        ]
        .astype(
            np.float64
        )
    )


    RH = (
        NWS[
            "relative_humidity"
        ][
            it
        ]
        .astype(
            np.float64
        )
    )


    U = (
        NWS[
            "wind_speed_ms"
        ][
            it
        ]
        .astype(
            np.float64
        )
    )


    C = (
        NWS[
            "sky_cover_percent"
        ][
            it
        ]
        .astype(
            np.float64
        )
    )


    (
        surf,
        deep,
        infil,
        drainage,
        ETsurf,
        ETdeep,
        runoff,
    ) = model_step(

        surf,
        deep,

        P,

        T,
        RH,
        U,
        C,

        DT_FORECAST_H,
    )


    result = hero_score(
        surf,
        deep,
    )


    score = result[
        "score"
    ]


    Hclass = result[
        "condition"
    ]


    F = result[
        "F"
    ]


    Fdeep = result[
        "Fdeep"
    ]


    wet_suitability = result[
        "wet_suitability"
    ]


    surf_forecast[
        it
    ] = surf


    deep_forecast[
        it
    ] = deep


    score_forecast[
        it
    ] = score


    fc_forecast[
        it
    ] = F


    deep_fc_forecast[
        it
    ] = Fdeep


    class_forecast[
        it
    ] = Hclass


    wet_suitability_forecast[
        it
    ] = wet_suitability


    infiltration_forecast[
        it
    ] = infil


    drainage_forecast[
        it
    ] = drainage


    ETsurf_forecast[
        it
    ] = ETsurf


    ETdeep_forecast[
        it
    ] = ETdeep


    runoff_forecast[
        it
    ] = runoff


    if (
        (
            it + 1
        )
        %
        4
        ==
        0
        or
        it
        ==
        nt - 1
    ):

        hours = (
            (
                it + 1
            )
            *
            6
        )


        print(
            f"  +{hours:3d} h: "
            f"surf="
            f"{np.nanmean(surf[public_valid]):.4f}, "
            f"deep="
            f"{np.nanmean(deep[public_valid]):.4f}, "
            f"score="
            f"{np.nanmean(score[public_valid]):.1f}"
        )


# ============================================================
# SAVE FORECAST
# ============================================================

np.savez_compressed(

    FORECAST_FILE,

    time=forecast_times,

    theta_surface=surf_forecast,

    theta_deep=deep_forecast,

    fc_state=fc_forecast,

    deep_fc_state=deep_fc_forecast,

    hero_score=score_forecast,

    hero_class=class_forecast,

    wet_suitability=wet_suitability_forecast,

    infiltration_mm=infiltration_forecast,

    deep_drainage_mm=drainage_forecast,

    surface_et_mm=ETsurf_forecast,

    deep_et_mm=ETdeep_forecast,

    runoff_mm=runoff_forecast,

    x=x,

    y=y,

    epsg=np.int32(
        26911
    ),
)


# ============================================================
# GEOTIFF OUTPUT SETTINGS
# ============================================================

NODATA = -9999.0


profile_float = {

    "driver":
        "GTiff",

    "height":
        ny,

    "width":
        nx,

    "count":
        1,

    "dtype":
        "float32",

    "crs":
        TARGET_CRS,

    "transform":
        transform,

    "nodata":
        NODATA,

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


profile_byte = {

    "driver":
        "GTiff",

    "height":
        ny,

    "width":
        nx,

    "count":
        1,

    "dtype":
        "uint8",

    "crs":
        TARGET_CRS,

    "transform":
        transform,

    "nodata":
        0,

    "compress":
        "deflate",

    "predictor":
        2,

    "tiled":
        True,

    "blockxsize":
        256,

    "blockysize":
        256,
}


# ============================================================
# GEOTIFF WRITER
# ============================================================

def write_float_tif(
    path,
    A,
    description,
):

    out = np.where(
        np.isfinite(
            A
        ),
        A,
        NODATA,
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
# CURRENT GEOTIFFS
# ============================================================

write_float_tif(

    CURRENT_DIR
    /
    "surface_theta_now.tif",

    theta_surface,

    "Current surface volumetric water content",
)


write_float_tif(

    CURRENT_DIR
    /
    "deep_theta_now.tif",

    theta_deep,

    "Current deep volumetric water content",
)


write_float_tif(

    CURRENT_DIR
    /
    "surface_fc_state_now.tif",

    fc_state_now,

    "Surface hydraulic state relative to WP-FC interval",
)


write_float_tif(

    CURRENT_DIR
    /
    "hero_score_now.tif",

    score_now,

    "Hero Dirt score v0.3 [0-100]",
)


with rasterio.open(

    CURRENT_DIR
    /
    "hero_class_now.tif",

    "w",

    **profile_byte,

) as dst:

    dst.write(
        class_now,
        1,
    )


    dst.set_band_description(
        1,
        "Hero Dirt directional condition class v0.3",
    )


# ============================================================
# FORECAST SNAPSHOT GEOTIFFS
# ============================================================

snapshot_hours = [
    24,
    48,
    72,
    96,
    120,
]


for hour in snapshot_hours:

    idx = (
        hour // 6
        -
        1
    )


    if idx >= nt:

        continue


    write_float_tif(

        FORECAST_DIR
        /
        f"hero_score_{hour:03d}h.tif",

        score_forecast[
            idx
        ],

        f"Hero Dirt score v0.3 +{hour} h",
    )


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " SOIL MODEL v0.3 SUMMARY"
)
print(
    "============================================"
)
print()


print(
    f"Initialization SMAP mean: "
    f"{smap_mean:.4f}"
)


print(
    f"Current surface theta:    "
    f"{np.nanmean(theta_surface[public_valid]):.4f}"
)


print(
    f"Current deep theta:       "
    f"{np.nanmean(theta_deep[public_valid]):.4f}"
)


print(
    f"Current mean F:           "
    f"{np.nanmean(fc_state_now[public_valid]):.3f}"
)


print(
    f"Current Hero score v0.3:  "
    f"{np.nanmean(score_now[public_valid]):.1f}"
)


print()


# ============================================================
# FORECAST SNAPSHOTS
# ============================================================

for hour in snapshot_hours:

    idx = (
        hour // 6
        -
        1
    )


    if idx >= nt:

        continue


    mean_surface = np.nanmean(
        surf_forecast[
            idx
        ][
            public_valid
        ]
    )


    mean_deep = np.nanmean(
        deep_forecast[
            idx
        ][
            public_valid
        ]
    )


    mean_F = np.nanmean(
        fc_forecast[
            idx
        ][
            public_valid
        ]
    )


    mean_score = np.nanmean(
        score_forecast[
            idx
        ][
            public_valid
        ]
    )


    print(
        f"+{hour:3d} h: "
        f"surface={mean_surface:.4f}, "
        f"deep={mean_deep:.4f}, "
        f"F={mean_F:.3f}, "
        f"score={mean_score:.1f}"
    )


print()


# ============================================================
# WATER BALANCE
# ============================================================

total_forecast_precip = np.nanmean(

    np.nansum(
        NWS[
            "precipitation_mm"
        ],
        axis=0,
    )[
        public_valid
    ]
)


total_surface_et = np.nanmean(

    np.nansum(
        ETsurf_forecast,
        axis=0,
    )[
        public_valid
    ]
)


total_deep_et = np.nanmean(

    np.nansum(
        ETdeep_forecast,
        axis=0,
    )[
        public_valid
    ]
)


total_infil = np.nanmean(

    np.nansum(
        infiltration_forecast,
        axis=0,
    )[
        public_valid
    ]
)


total_drain = np.nanmean(

    np.nansum(
        drainage_forecast,
        axis=0,
    )[
        public_valid
    ]
)


total_runoff = np.nanmean(

    np.nansum(
        runoff_forecast,
        axis=0,
    )[
        public_valid
    ]
)


print(
    "Domain-mean 120-h water balance:"
)


print(
    f"  precipitation:     "
    f"{total_forecast_precip:.2f} mm"
)


print(
    f"  surface ET:        "
    f"{total_surface_et:.2f} mm"
)


print(
    f"  deep ET:           "
    f"{total_deep_et:.2f} mm"
)


print(
    f"  surface->deep:     "
    f"{total_infil:.2f} mm"
)


print(
    f"  deep drainage:     "
    f"{total_drain:.2f} mm"
)


print(
    f"  runoff:            "
    f"{total_runoff:.2f} mm"
)


print()


# ============================================================
# CURRENT CLASS DISTRIBUTION
# ============================================================

print(
    "Current public-domain classes:"
)


npublic = np.count_nonzero(
    public_valid
)


for code in range(
    1,
    7,
):

    count = np.count_nonzero(
        (
            class_now
            ==
            code
        )
        &
        public_valid
    )


    pct = (
        100.0
        *
        count
        /
        npublic
    )


    print(
        f"  "
        f"{CLASS_NAMES[code]:11s}: "
        f"{pct:5.1f}%"
    )


print()


# ============================================================
# SCORE COMPONENT DIAGNOSTICS
# ============================================================

plastic = (
    public_valid
    &
    np.isfinite(
        plasticity_index
    )
    &
    (
        plasticity_index
        >
        0.0
    )
)


nonplastic = (
    public_valid
    &
    np.isfinite(
        plasticity_index
    )
    &
    (
        plasticity_index
        <=
        0.0
    )
)


print(
    "Score-component diagnostics:"
)


print(
    f"  mean dry suitability:       "
    f"{np.nanmean(dry_suitability_now[public_valid]):.3f}"
)


print(
    f"  mean wet suitability:       "
    f"{np.nanmean(wet_suitability_now[public_valid]):.3f}"
)


print(
    f"  mean deep suitability:      "
    f"{np.nanmean(deep_suitability_now[public_valid]):.3f}"
)


if np.any(
    plastic
):

    print(
        f"  wet suitability, plastic:   "
        f"{np.nanmean(wet_suitability_now[plastic]):.3f}"
    )


if np.any(
    nonplastic
):

    print(
        f"  wet suitability, nonplastic:"
        f" {np.nanmean(wet_suitability_now[nonplastic]):.3f}"
    )


print()


# ============================================================
# OUTPUT PATHS
# ============================================================

print(
    "Saved current state:"
)


print(
    f"  {CURRENT_FILE}"
)


print()


print(
    "Saved forecast:"
)


print(
    f"  {FORECAST_FILE}"
)


print()


print(
    "============================================"
)
print(
    " HERO DIRT SOIL MODEL v0.3 COMPLETE"
)
print(
    "============================================"
)
print()
