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
# CONSTANTS
# ============================================================

SOLAR_CONSTANT = 1361.0        # W m-2
BLOCK_HOURS = 6.0
SAMPLES_PER_BLOCK = 12         # 30-minute solar samples


# ============================================================
# LOAD
# ============================================================

S = np.load(STATIC_FILE)
N = np.load(NWS_FILE)

x = S["x"].astype(float)
y = S["y"].astype(float)

elevation = S["elevation"].astype(float)
slope_deg = S["slope"].astype(float)

northness = S["northness"].astype(float)
eastness = S["eastness"].astype(float)

tree = S["tree_fraction"].astype(float)
shrub = S["shrub_cover_fraction"].astype(float)
herb = S["herb_cover_fraction"].astype(float)

physics = S["physics_valid"].astype(bool)

times = N["time"]

temperature = N["temperature_c"].astype(float)
rh = N["relative_humidity"].astype(float)
wind = N["wind_speed_ms"].astype(float)
cloud = N["sky_cover_percent"].astype(float)


# ============================================================
# GRID LAT/LON
# ============================================================

X, Y = np.meshgrid(x, y)

transformer = Transformer.from_crs(
    "EPSG:26956",
    "EPSG:4326",
    always_xy=True,
)

lon, lat = transformer.transform(X, Y)

lat_rad = np.deg2rad(lat)
slope_rad = np.deg2rad(slope_deg)


# ============================================================
# TERRAIN NORMAL
#
# eastness / northness point downhill.
#
# Upward-facing surface normal:
#
#   horizontal magnitude = sin(slope)
#   vertical magnitude   = cos(slope)
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

normal_up = np.cos(slope_rad)


# ============================================================
# CANOPY TRANSMISSION
#
# Retain current Hero Dirt attenuation for now.
# This will eventually be reconsidered separately.
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
# CLOUD TRANSMISSION
#
# Retain current cloud formulation for this diagnostic.
# ============================================================

cloud_fraction = np.clip(
    cloud / 100.0,
    0.0,
    1.0,
)

cloud_transmission = np.clip(
    1.0
    -
    0.50 * cloud_fraction,
    0.50,
    1.0,
)


# ============================================================
# VPD
# ============================================================

def saturation_vapor_pressure(T):
    """
    Saturation vapor pressure [kPa].
    """
    return (
        0.6108
        *
        np.exp(
            17.27 * T
            /
            (T + 237.3)
        )
    )


es = saturation_vapor_pressure(temperature)

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
# SOLAR VECTOR + EXTRATERRESTRIAL NORMAL IRRADIANCE
# ============================================================

def solar_state(
    timestamp,
    lat_rad,
    lon_deg,
):
    """
    Return:

        sun_east
        sun_north
        sun_up
        extraterrestrial normal irradiance [W m-2]

    for one UTC timestamp.
    """

    day = timestamp.astype("datetime64[D]")
    year = timestamp.astype("datetime64[Y]")

    doy = (
        day - year
    ).astype(int) + 1

    seconds_today = (
        timestamp - day
    ).astype("timedelta64[s]").astype(float)

    hour_utc = (
        seconds_today
        /
        3600.0
    )

    # Fractional year [radians]
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

    # Solar declination [radians]
    decl = (
        0.006918
        -
        0.399912 * np.cos(gamma)
        +
        0.070257 * np.sin(gamma)
        -
        0.006758 * np.cos(2.0 * gamma)
        +
        0.000907 * np.sin(2.0 * gamma)
        -
        0.002697 * np.cos(3.0 * gamma)
        +
        0.00148 * np.sin(3.0 * gamma)
    )

    # Equation of time [minutes]
    eqtime = (
        229.18
        *
        (
            0.000075
            +
            0.001868 * np.cos(gamma)
            -
            0.032077 * np.sin(gamma)
            -
            0.014615 * np.cos(2.0 * gamma)
            -
            0.040849 * np.sin(2.0 * gamma)
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
        4.0 * lon_deg
    )

    hour_angle = np.deg2rad(
        true_solar_minutes / 4.0
        -
        180.0
    )

    cos_decl = np.cos(decl)
    sin_decl = np.sin(decl)

    # Sun vector in local East-North-Up coordinates
    sun_east = (
        -cos_decl
        *
        np.sin(hour_angle)
    )

    sun_north = (
        sin_decl * np.cos(lat_rad)
        -
        cos_decl
        *
        np.cos(hour_angle)
        *
        np.sin(lat_rad)
    )

    sun_up = (
        sin_decl * np.sin(lat_rad)
        +
        cos_decl
        *
        np.cos(hour_angle)
        *
        np.cos(lat_rad)
    )

    # Earth-Sun distance correction
    distance_factor = (
        1.0
        +
        0.033
        *
        np.cos(
            2.0
            *
            np.pi
            *
            doy
            /
            365.0
        )
    )

    extraterrestrial_normal = (
        SOLAR_CONSTANT
        *
        distance_factor
    )

    return (
        sun_east,
        sun_north,
        sun_up,
        extraterrestrial_normal,
    )


# ============================================================
# INTEGRATE SOLAR ENERGY OVER EACH NWS 6-HOUR BLOCK
#
# Stored NWS time = END of 6-h block.
#
# We use 30-minute midpoint samples.
# ============================================================

nt = len(times)

topo_irradiance = np.zeros_like(
    temperature,
    dtype=float,
)

horizontal_irradiance = np.zeros_like(
    temperature,
    dtype=float,
)


sample_dt_hours = (
    BLOCK_HOURS
    /
    SAMPLES_PER_BLOCK
)


for k in range(nt):

    block_end = times[k]

    topo_accum = np.zeros(
        physics.shape,
        dtype=float,
    )

    horiz_accum = np.zeros(
        physics.shape,
        dtype=float,
    )


    for j in range(SAMPLES_PER_BLOCK):

        # Midpoint of each 30-minute sample,
        # counting backward from block end.

        hours_before_end = (
            BLOCK_HOURS
            -
            (
                j
                +
                0.5
            )
            *
            sample_dt_hours
        )

        sample_seconds = int(
            round(
                hours_before_end
                *
                3600.0
            )
        )

        sample_time = (
            block_end
            -
            np.timedelta64(
                sample_seconds,
                "s",
            )
        )

        (
            sun_e,
            sun_n,
            sun_u,
            R_normal,
        ) = solar_state(
            sample_time,
            lat_rad,
            lon,
        )

        # Horizontal TOA irradiance
        horiz = np.where(
            sun_u > 0.0,
            R_normal * sun_u,
            0.0,
        )

        # Terrain-plane cosine of incidence
        incidence = (
            normal_east * sun_e
            +
            normal_north * sun_n
            +
            normal_up * sun_u
        )

        topo = np.where(
            (sun_u > 0.0)
            &
            (incidence > 0.0),
            R_normal * incidence,
            0.0,
        )

        horiz_accum += horiz
        topo_accum += topo


    horizontal_irradiance[k] = (
        horiz_accum
        /
        SAMPLES_PER_BLOCK
    )

    topo_irradiance[k] = (
        topo_accum
        /
        SAMPLES_PER_BLOCK
    )


# ============================================================
# CLOUD + CANOPY ATTENUATION
#
# IMPORTANT:
# These remain provisional empirical transmission terms.
#
# We are NOT yet calling this a full surface shortwave
# radiation estimate.
# ============================================================

cloud_adjusted = (
    topo_irradiance
    *
    cloud_transmission
)

effective_irradiance = (
    cloud_adjusted
    *
    canopy_transmission[
        None,
        :,
        :,
    ]
)


# ============================================================
# ENERGY PER 6-H BLOCK
#
# 1 W m-2 = 1 J s-1 m-2
#
# MJ m-2 over block:
#
#   mean_Wm2 * seconds / 1e6
# ============================================================

block_seconds = (
    BLOCK_HOURS
    *
    3600.0
)

effective_energy = (
    effective_irradiance
    *
    block_seconds
    /
    1.0e6
)


# ============================================================
# OPTIONAL DAILY ENERGY BY PAIRS OF FOUR 6-H BLOCKS
# ============================================================

print()
print("============================================")
print(" HERO DIRT V2 IRRADIANCE DIAGNOSTIC")
print("============================================")

print()
print("Forecast blocks:")
print()

for k in range(nt):

    mask = (
        physics
        &
        np.isfinite(vpd[k])
        &
        np.isfinite(effective_irradiance[k])
    )

    print(
        f"{str(times[k]):20s}  "
        f"VPD={np.nanmean(vpd[k][mask]):5.3f} kPa  "
        f"Rhor={np.nanmean(horizontal_irradiance[k][mask]):7.1f}  "
        f"Rtopo={np.nanmean(topo_irradiance[k][mask]):7.1f}  "
        f"Reff={np.nanmean(effective_irradiance[k][mask]):7.1f} W/m2  "
        f"E6h={np.nanmean(effective_energy[k][mask]):5.2f} MJ/m2"
    )


print()
print("Spatial effective irradiance:")
print()

good = (
    physics[
        None,
        :,
        :,
    ]
    &
    np.isfinite(
        effective_irradiance
    )
)

vals = effective_irradiance[good]

print(
    f"mean: {np.mean(vals):.1f} W/m2"
)
print(
    f"P05:  {np.percentile(vals,5):.1f} W/m2"
)
print(
    f"P50:  {np.percentile(vals,50):.1f} W/m2"
)
print(
    f"P95:  {np.percentile(vals,95):.1f} W/m2"
)
print(
    f"max:  {np.max(vals):.1f} W/m2"
)


# ============================================================
# DAILY BLOCK-INTEGRATED ENERGY
# ============================================================

print()
print("Approximate 24-h effective radiative energy:")
print()

for k0 in range(
    0,
    nt - 3,
    4,
):

    k1 = (
        k0
        +
        4
    )

    daily_energy = np.sum(
        effective_energy[
            k0:k1
        ],
        axis=0,
    )

    mask = (
        physics
        &
        np.isfinite(
            daily_energy
        )
    )

    print(
        f"{str(times[k0])} -> {str(times[k1-1])}: "
        f"mean={np.nanmean(daily_energy[mask]):5.2f} "
        f"MJ/m2/day  "
        f"P05={np.nanpercentile(daily_energy[mask],5):5.2f}  "
        f"P95={np.nanpercentile(daily_energy[mask],95):5.2f}"
    )


print()
print("============================================")
print(" DIAGNOSTIC COMPLETE")
print("============================================")
