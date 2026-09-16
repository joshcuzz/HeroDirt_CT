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

SOLAR_CONSTANT = 1361.0       # W m-2
ALBEDO_TRAIL = 0.18           # provisional bare-soil value
STEFAN_BOLTZMANN = 2.043e-10  # MJ K-4 m-2 h-1


# ============================================================
# LOAD DATA
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
# GRID LAT / LON
# ============================================================

X, Y = np.meshgrid(x, y)

transformer = Transformer.from_crs(
    "EPSG:26956",
    "EPSG:4326",
    always_xy=True,
)

lon, lat = transformer.transform(
    X,
    Y,
)

lat_rad = np.deg2rad(lat)
slope_rad = np.deg2rad(slope_deg)


# ============================================================
# TERRAIN NORMAL
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
# Keep existing Hero Dirt relation initially.
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
# METEOROLOGICAL HELPERS
# ============================================================

def saturation_vapor_pressure(T):
    """
    kPa
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


def slope_sat_vapor_curve(T):
    """
    Delta [kPa / C]
    """

    es = saturation_vapor_pressure(T)

    return (
        4098.0
        *
        es
        /
        (
            T + 237.3
        )**2
    )


def atmospheric_pressure(z):
    """
    Atmospheric pressure [kPa]
    from elevation [m].
    """

    return (
        101.3
        *
        (
            (
                293.0
                -
                0.0065 * z
            )
            /
            293.0
        )**5.26
    )


pressure = atmospheric_pressure(
    elevation
)


psychrometric = (
    0.000665
    *
    pressure
)


# ============================================================
# SOLAR STATE
# ============================================================

def solar_state(
    timestamp,
    lat_rad,
    lon_deg,
):

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
                hour_utc - 12.0
            )
            /
            24.0
        )
    )


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


    solar_minutes = (
        utc_minutes
        +
        eqtime
        +
        4.0 * lon_deg
    )


    hour_angle = np.deg2rad(
        solar_minutes / 4.0
        -
        180.0
    )


    cos_decl = np.cos(decl)
    sin_decl = np.sin(decl)


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
# PRECOMPUTE VPD ETC.
# ============================================================

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

delta = slope_sat_vapor_curve(
    temperature
)


# ============================================================
# 6-HOUR PENMAN CALCULATION
# ============================================================

nt = len(times)

ET_block = np.zeros_like(
    temperature,
    dtype=float,
)

Rn_block = np.zeros_like(
    temperature,
    dtype=float,
)

Rs_block = np.zeros_like(
    temperature,
    dtype=float,
)


for k in range(nt):

    block_end = times[k]

    T = temperature[k]
    VP = vpd[k]
    EA = ea[k]
    U = wind[k]

    cloud_fraction = np.clip(
        cloud[k] / 100.0,
        0.0,
        1.0,
    )

    # Retain old cloud transmissivity initially.
    cloud_transmission = np.clip(
        1.0
        -
        0.50 * cloud_fraction,
        0.50,
        1.0,
    )


    ET6 = np.zeros(
        physics.shape,
        dtype=float,
    )

    Rn6 = np.zeros(
        physics.shape,
        dtype=float,
    )

    Rs6 = np.zeros(
        physics.shape,
        dtype=float,
    )


    # Six one-hour substeps.
    for j in range(6):

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
            R_normal,
        ) = solar_state(
            sample_time,
            lat_rad,
            lon,
        )


        incidence = (
            normal_east * sun_e
            +
            normal_north * sun_n
            +
            normal_up * sun_u
        )


        incidence = np.where(
            (sun_u > 0.0)
            &
            (incidence > 0.0),
            incidence,
            0.0,
        )


        # ----------------------------------------------------
        # Top-of-atmosphere terrain-plane radiation
        # ----------------------------------------------------

        Ra_topo_W = (
            R_normal
            *
            incidence
        )


        # ----------------------------------------------------
        # Approximate clear-sky atmospheric transmission
        #
        # FAO-style elevation dependence.
        # ----------------------------------------------------

        clear_transmission = (
            0.75
            +
            2.0e-5
            *
            elevation
        )


        clear_transmission = np.clip(
            clear_transmission,
            0.65,
            0.85,
        )


        # ----------------------------------------------------
        # Incoming shortwave at trail surface
        # ----------------------------------------------------

        Rs_W = (
            Ra_topo_W
            *
            clear_transmission
            *
            cloud_transmission
            *
            canopy_transmission
        )


        # W/m2 -> MJ/m2/hour
        Rs = (
            Rs_W
            *
            3600.0
            /
            1.0e6
        )


        # ----------------------------------------------------
        # Net shortwave
        # ----------------------------------------------------

        Rns = (
            1.0
            -
            ALBEDO_TRAIL
        ) * Rs


        # ----------------------------------------------------
        # Approximate net longwave
        #
        # FAO-style emissivity dependence.
        #
        # Use cloud fraction directly rather than Rs/Rso
        # because canopy has already attenuated Rs.
        # ----------------------------------------------------

        Tk = (
            T
            +
            273.16
        )


        clear_sky_emissivity_term = np.clip(
            0.34
            -
            0.14
            *
            np.sqrt(
                np.maximum(
                    EA,
                    0.0,
                )
            ),
            0.05,
            0.34,
        )


        cloud_longwave_factor = np.clip(
            1.0
            -
            0.50
            *
            cloud_fraction,
            0.50,
            1.0,
        )


        Rnl = (
            STEFAN_BOLTZMANN
            *
            Tk**4
            *
            clear_sky_emissivity_term
            *
            cloud_longwave_factor
        )


        # Net radiation
        Rn = (
            Rns
            -
            Rnl
        )


        # ----------------------------------------------------
        # Soil heat flux
        #
        # FAO hourly approximation.
        # ----------------------------------------------------

        daytime = (
            sun_u > 0.0
        )


        G = np.where(
            daytime,
            0.10 * Rn,
            0.50 * Rn,
        )


        # ----------------------------------------------------
        # Hourly Penman-Monteith
        # ----------------------------------------------------

        numerator = (
            0.408
            *
            delta[k]
            *
            (
                Rn - G
            )
            +
            psychrometric
            *
            (
                37.0
                /
                (
                    T + 273.0
                )
            )
            *
            U
            *
            VP
        )


        denominator = (
            delta[k]
            +
            psychrometric
            *
            (
                1.0
                +
                0.34
                *
                U
            )
        )


        ET_hour = np.where(
            denominator > 0.0,
            numerator / denominator,
            0.0,
        )


        # No negative evaporation.
        ET_hour = np.maximum(
            ET_hour,
            0.0,
        )


        ET_hour[
            ~physics
        ] = np.nan


        ET6 += np.nan_to_num(
            ET_hour,
            nan=0.0,
        )

        Rn6 += np.nan_to_num(
            Rn,
            nan=0.0,
        )

        Rs6 += np.nan_to_num(
            Rs,
            nan=0.0,
        )


    ET_block[k] = ET6
    Rn_block[k] = Rn6
    Rs_block[k] = Rs6


# ============================================================
# OUTPUT SUMMARY
# ============================================================

print()
print("============================================")
print(" HERO DIRT V2 PENMAN DIAGNOSTIC")
print("============================================")

print()
print("Forecast blocks:")
print()

for k in range(nt):

    mask = (
        physics
        &
        np.isfinite(
            ET_block[k]
        )
    )

    print(
        f"{str(times[k]):20s}  "
        f"VPD={np.nanmean(vpd[k][mask]):5.3f} kPa  "
        f"ET6={np.nanmean(ET_block[k][mask]):5.3f} mm  "
        f"P05={np.nanpercentile(ET_block[k][mask],5):5.3f}  "
        f"P95={np.nanpercentile(ET_block[k][mask],95):5.3f}"
    )


print()
print("Approximate 24-h drying potential:")
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

    daily_ET = np.sum(
        ET_block[
            k0:k1
        ],
        axis=0,
    )


    mask = (
        physics
        &
        np.isfinite(
            daily_ET
        )
    )


    print(
        f"{str(times[k0])} -> {str(times[k1-1])}:  "
        f"mean={np.nanmean(daily_ET[mask]):5.2f} mm/day  "
        f"P05={np.nanpercentile(daily_ET[mask],5):5.2f}  "
        f"P50={np.nanpercentile(daily_ET[mask],50):5.2f}  "
        f"P95={np.nanpercentile(daily_ET[mask],95):5.2f}"
    )


print()
print("Overall:")
print()

good = (
    physics[
        None,
        :,
        :,
    ]
    &
    np.isfinite(
        ET_block
    )
)

vals = ET_block[good]

print(
    f"6-h ET mean: {np.mean(vals):.3f} mm"
)
print(
    f"6-h ET P05:  {np.percentile(vals,5):.3f} mm"
)
print(
    f"6-h ET P50:  {np.percentile(vals,50):.3f} mm"
)
print(
    f"6-h ET P95:  {np.percentile(vals,95):.3f} mm"
)
print(
    f"6-h ET max:  {np.max(vals):.3f} mm"
)

print()
print("============================================")
print(" DIAGNOSTIC COMPLETE")
print("============================================")
