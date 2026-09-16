#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt_CT"

STATIC_FILE = (
    ROOT
    / "static/model/HeroDirt_static_200m.npz"
)

NWS_FILE = (
    ROOT
    / "data/nws/processed/HeroDirt_NWS_6hourly.npz"
)


# ============================================================
# LOAD DATA
# ============================================================

S = np.load(STATIC_FILE)
N = np.load(NWS_FILE)


x = S["x"].astype(float)
y = S["y"].astype(float)

slope_deg = S["slope"].astype(float)

northness = S[
    "northness"
].astype(float)

eastness = S[
    "eastness"
].astype(float)


tree = S[
    "tree_fraction"
].astype(float)

shrub = S[
    "shrub_cover_fraction"
].astype(float)

herb = S[
    "herb_cover_fraction"
].astype(float)


physics = S[
    "physics_valid"
].astype(bool)


times = N["time"]

temperature = N[
    "temperature_c"
].astype(float)

rh = N[
    "relative_humidity"
].astype(float)

wind = N[
    "wind_speed_ms"
].astype(float)

cloud = N[
    "sky_cover_percent"
].astype(float)


# ============================================================
# GRID LATITUDE / LONGITUDE
# ============================================================

X, Y = np.meshgrid(
    x,
    y,
)


transformer = Transformer.from_crs(
    "EPSG:26956",
    "EPSG:4326",
    always_xy=True,
)


lon, lat = transformer.transform(
    X,
    Y,
)


lat_rad = np.deg2rad(
    lat
)


slope_rad = np.deg2rad(
    slope_deg
)


# ============================================================
# TERRAIN NORMAL
#
# eastness / northness point DOWNHILL.
#
# An upward-facing terrain normal therefore has:
#
#   horizontal magnitude = sin(slope)
#   vertical magnitude   = cos(slope)
#
# and its horizontal direction is the downhill direction.
# ============================================================

normal_east = (
    np.sin(slope_rad)
    *
    eastness
)

normal_north = (
    np.sin(slope_rad)
    *
    northness
)

normal_up = np.cos(
    slope_rad
)


# ============================================================
# CANOPY TRANSMISSION
#
# For this diagnostic, retain the existing Hero Dirt
# vegetation attenuation so we isolate the effect of replacing
# fixed aspect with real solar geometry.
#
# We can refine this later.
# ============================================================

canopy_transmission = np.clip(
    1.0
    -
    0.55 * tree
    -
    0.30 * shrub
    -
    0.15 * herb,
    0.45,
    1.0,
)


# ============================================================
# VPD
# ============================================================

def saturation_vapor_pressure(T):

    return (
        0.6108
        *
        np.exp(
            17.27 * T
            /
            (
                T
                +
                237.3
            )
        )
    )


es = saturation_vapor_pressure(
    temperature
)


ea = (
    rh
    /
    100.0
    *
    es
)


vpd = np.maximum(
    es - ea,
    0.0,
)


# ============================================================
# SOLAR POSITION
#
# Uses standard solar-geometry approximations.
#
# Returned sun vector is expressed in local:
#
#   east
#   north
#   up
#
# coordinates.
# ============================================================

def solar_vector(
    timestamp,
    lat_rad,
    lon_deg,
):

    """
    Solar unit vector for one UTC time.

    timestamp:
        numpy.datetime64

    lat_rad:
        2-D latitude array [radians]

    lon_deg:
        2-D longitude array [degrees]
    """

    # Convert to Python-ish calendar quantities using numpy.

    day = timestamp.astype(
        "datetime64[D]"
    )

    year = timestamp.astype(
        "datetime64[Y]"
    )

    doy = (
        day
        -
        year
    ).astype(int) + 1


    seconds_today = (
        timestamp
        -
        day
    ).astype(
        "timedelta64[s]"
    ).astype(float)


    hour_utc = (
        seconds_today
        /
        3600.0
    )


    gamma = (
        2.0
        *
        np.pi
        /
        365.0
        *
        (
            doy
            -
            1
            +
            (
                hour_utc
                -
                12.0
            )
            /
            24.0
        )
    )


    decl = (
        0.006918
        -
        0.399912
        *
        np.cos(gamma)
        +
        0.070257
        *
        np.sin(gamma)
        -
        0.006758
        *
        np.cos(
            2.0 * gamma
        )
        +
        0.000907
        *
        np.sin(
            2.0 * gamma
        )
        -
        0.002697
        *
        np.cos(
            3.0 * gamma
        )
        +
        0.00148
        *
        np.sin(
            3.0 * gamma
        )
    )


    eqtime = (
        229.18
        *
        (
            0.000075
            +
            0.001868
            *
            np.cos(gamma)
            -
            0.032077
            *
            np.sin(gamma)
            -
            0.014615
            *
            np.cos(
                2.0 * gamma
            )
            -
            0.040849
            *
            np.sin(
                2.0 * gamma
            )
        )
    )


    utc_minutes = (
        hour_utc
        *
        60.0
    )


    true_solar_minutes = (
        utc_minutes
        +
        eqtime
        +
        4.0
        *
        lon_deg
    )


    hour_angle = np.deg2rad(
        true_solar_minutes
        /
        4.0
        -
        180.0
    )


    cos_decl = np.cos(
        decl
    )

    sin_decl = np.sin(
        decl
    )


    # Local ENU sun vector.

    sun_east = (
        -cos_decl
        *
        np.sin(
            hour_angle
        )
    )


    sun_north = (
        sin_decl
        *
        np.cos(
            lat_rad
        )
        -
        cos_decl
        *
        np.cos(
            hour_angle
        )
        *
        np.sin(
            lat_rad
        )
    )


    sun_up = (
        sin_decl
        *
        np.sin(
            lat_rad
        )
        +
        cos_decl
        *
        np.cos(
            hour_angle
        )
        *
        np.cos(
            lat_rad
        )
    )


    return (
        sun_east,
        sun_north,
        sun_up,
    )


# ============================================================
# 6-HOUR SOLAR EXPOSURE
#
# NWS timestamps are END-OF-BLOCK times.
#
# Integrate solar geometry hourly through the preceding
# six-hour interval.
# ============================================================

nt = len(
    times
)


potential_solar = np.zeros_like(
    temperature,
    dtype=float,
)


for k in range(
    nt
):

    block_end = times[
        k
    ]


    accum = np.zeros(
        physics.shape,
        dtype=float,
    )


    nsamp = 0


    # Midpoints of six one-hour samples:
    #
    # end-5.5h, ..., end-0.5h

    for j in range(
        6
    ):

        hours_before_end = (
            5.5
            -
            j
        )


        sample_time = (
            block_end
            -
            np.timedelta64(
                int(
                    hours_before_end
                    *
                    3600
                ),
                "s",
            )
        )


        (
            sun_e,
            sun_n,
            sun_u,
        ) = solar_vector(
            sample_time,
            lat_rad,
            lon,
        )


        incidence = (
            normal_east
            *
            sun_e
            +
            normal_north
            *
            sun_n
            +
            normal_up
            *
            sun_u
        )


        # No direct solar exposure if:
        #
        # 1. sun is below horizon
        # 2. sun is behind the terrain surface

        incidence = np.where(
            (sun_u > 0.0)
            &
            (incidence > 0.0),
            incidence,
            0.0,
        )


        accum += incidence

        nsamp += 1


    potential_solar[
        k
    ] = (
        accum
        /
        nsamp
    )


# ============================================================
# CLOUD ATTENUATION
#
# For now retain a simple bounded attenuation.
# This is DIAGNOSTIC, not yet the final v2 physics.
# ============================================================

cloud_fraction = np.clip(
    cloud
    /
    100.0,
    0.0,
    1.0,
)


cloud_transmission = (
    1.0
    -
    0.50
    *
    cloud_fraction
)


cloud_transmission = np.clip(
    cloud_transmission,
    0.5,
    1.0,
)


solar_cloud = (
    potential_solar
    *
    cloud_transmission
)


solar_effective = (
    solar_cloud
    *
    canopy_transmission[
        None,
        :,
        :,
    ]
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT V2 DRYING DIAGNOSTIC"
)
print(
    "============================================"
)


print()
print(
    "Forecast blocks:"
)

for k in range(
    nt
):

    mask = (
        physics
        &
        np.isfinite(
            vpd[k]
        )
        &
        np.isfinite(
            solar_effective[k]
        )
    )


    print(
        f"{str(times[k]):20s}  "
        f"VPD={np.nanmean(vpd[k][mask]):5.3f} kPa  "
        f"SolarPot={np.nanmean(potential_solar[k][mask]):5.3f}  "
        f"SolarCloud={np.nanmean(solar_cloud[k][mask]):5.3f}  "
        f"SolarEff={np.nanmean(solar_effective[k][mask]):5.3f}"
    )


print()
print(
    "Spatial solar diagnostics:"
)


for label, A in [
    (
        "Potential",
        potential_solar,
    ),
    (
        "Cloud-adjusted",
        solar_cloud,
    ),
    (
        "Cloud+canopy",
        solar_effective,
    ),
]:

    good = (
        physics[
            None,
            :,
            :,
        ]
        &
        np.isfinite(
            A
        )
    )


    vals = A[
        good
    ]


    print()
    print(
        label
    )

    print(
        f"  mean: {np.mean(vals):.3f}"
    )

    print(
        f"  P05:  {np.percentile(vals,5):.3f}"
    )

    print(
        f"  P50:  {np.percentile(vals,50):.3f}"
    )

    print(
        f"  P95:  {np.percentile(vals,95):.3f}"
    )

    print(
        f"  max:  {np.max(vals):.3f}"
    )


print()
print(
    "============================================"
)
print(
    " DIAGNOSTIC COMPLETE"
)
print(
    "============================================"
)
