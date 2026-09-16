#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
from pyproj import Transformer


# ============================================================
# HERO DIRT
# Match rider reports to archived model states
# ============================================================

REPORT_FILE = (
    "output/validation/trail_reports.csv"
)

OUTPUT_FILE = (
    "output/validation/"
    "HeroDirt_report_matchups.csv"
)

GRID_FILE = (
    "static/grid/"
    "HeroDirt_grid_200m.npz"
)

STATIC_FILE = (
    "static/model/"
    "HeroDirt_static_200m.npz"
)

PLASTICITY_FILE = (
    "static/soils/"
    "HeroDirt_plasticity_200m.npz"
)

ARCHIVE_ROOT = (
    "output/archive"
)


# ------------------------------------------------------------
# Archived condition names
# ------------------------------------------------------------

CONDITION_NAMES = {
    0: "Invalid",
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


# ------------------------------------------------------------
# Current rider-facing thresholds
#
# These are the current Hero Dirt empirical class boundaries.
# Old archived files may contain condition codes generated
# using earlier thresholds, so we also reclassify from F.
# ------------------------------------------------------------

def classify_current(F):

    if not np.isfinite(F):
        return "Invalid"

    if F < 0.10:
        return "Very Dry"

    elif F < 0.40:
        return "Dry"

    elif F < 0.65:
        return "Moist/Good"

    elif F < 0.80:
        return "Hero"

    elif F < 1.00:
        return "Wet"

    else:
        return "Too Wet"


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def np_datetime_to_timestamp(value):
    """
    Convert numpy datetime64 to pandas Timestamp in UTC.
    """

    return pd.Timestamp(
        value
    ).tz_localize("UTC")


def safe_value(arr, iy, ix):
    """
    Return grid value as float, preserving NaN.
    """

    value = arr[iy, ix]

    if np.issubdtype(
        np.asarray(value).dtype,
        np.number,
    ):
        return float(value)

    return value.item()


def find_pi_array(filename):
    """
    Load the plasticity file and try to identify
    the Plasticity Index array automatically.
    """

    if not os.path.exists(filename):

        print(
            "WARNING: Plasticity file not found:"
        )
        print(
            f"  {filename}"
        )

        return None, None

    d = np.load(
        filename,
        allow_pickle=True,
    )

    preferred_names = [
        "PI",
        "pi",
        "plasticity_index",
        "plasticity",
        "plasticity_idx",
    ]

    for name in preferred_names:

        if name in d.files:

            arr = d[name]

            if arr.ndim == 2:

                return arr, name

    # If none of the expected names exist,
    # look for a 2-D field with the grid shape.

    for name in d.files:

        arr = d[name]

        if arr.ndim == 2:

            if arr.shape == (264, 567):

                return arr, name

    return None, None


# ------------------------------------------------------------
# Load reports
# ------------------------------------------------------------

if not os.path.exists(REPORT_FILE):

    raise FileNotFoundError(
        f"Cannot find {REPORT_FILE}"
    )


reports = pd.read_csv(
    REPORT_FILE
)


if len(reports) == 0:

    raise RuntimeError(
        "No rider reports found."
    )


reports["observation_utc"] = (
    pd.to_datetime(
        reports["observation_utc"],
        utc=True,
    )
)


# ------------------------------------------------------------
# Load static grid
# ------------------------------------------------------------

grid = np.load(
    GRID_FILE,
    allow_pickle=True,
)

x = grid["x"].astype(float)
y = grid["y"].astype(float)

epsg = int(
    grid["epsg"]
)


# ------------------------------------------------------------
# Coordinate transformer
#
# Rider reports are lon/lat (EPSG:4326).
# Hero Dirt model grid is EPSG:26911.
# ------------------------------------------------------------

transformer = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{epsg}",
    always_xy=True,
)


# ------------------------------------------------------------
# Load static Hero Dirt properties
# ------------------------------------------------------------

static = np.load(
    STATIC_FILE,
    allow_pickle=True,
)


# ------------------------------------------------------------
# Load Plasticity Index
# ------------------------------------------------------------

pi_array, pi_name = find_pi_array(
    PLASTICITY_FILE
)

if pi_array is not None:

    print(
        f"Plasticity field: {pi_name}"
    )

else:

    print(
        "Plasticity Index could not be identified."
    )


# ------------------------------------------------------------
# Build catalog of every archived model valid time
# ------------------------------------------------------------

model_catalog = []


archive_dirs = sorted(
    glob.glob(
        os.path.join(
            ARCHIVE_ROOT,
            "*",
        )
    )
)


for archive_dir in archive_dirs:

    if not os.path.isdir(
        archive_dir
    ):
        continue


    archive_name = os.path.basename(
        archive_dir
    )


    # --------------------------------------------------------
    # Current / analysis state
    # --------------------------------------------------------

    current_file = os.path.join(
        archive_dir,
        "HeroDirt_current_state.npz",
    )

    if os.path.exists(
        current_file
    ):

        d = np.load(
            current_file,
            allow_pickle=True,
        )

        valid_time = (
            np_datetime_to_timestamp(
                d["time"].item()
            )
        )

        model_catalog.append(
            {
                "valid_time": valid_time,
                "archive": archive_name,
                "source": "analysis",
                "file": current_file,
                "time_index": None,
            }
        )


    # --------------------------------------------------------
    # Forecast states
    # --------------------------------------------------------

    forecast_file = os.path.join(
        archive_dir,
        "HeroDirt_forecast_6hourly.npz",
    )

    if os.path.exists(
        forecast_file
    ):

        d = np.load(
            forecast_file,
            allow_pickle=True,
        )

        times = d["time"]

        for k in range(
            len(times)
        ):

            valid_time = (
                np_datetime_to_timestamp(
                    times[k]
                )
            )

            model_catalog.append(
                {
                    "valid_time": valid_time,
                    "archive": archive_name,
                    "source": "forecast",
                    "file": forecast_file,
                    "time_index": k,
                }
            )


if len(model_catalog) == 0:

    raise RuntimeError(
        "No archived Hero Dirt states found."
    )


catalog_times = pd.DatetimeIndex(
    [
        item["valid_time"]
        for item in model_catalog
    ]
)


print()
print(
    "============================================"
)
print(
    " HERO DIRT RIDER-REPORT MATCHUP"
)
print(
    "============================================"
)
print()
print(
    f"Rider reports: {len(reports)}"
)
print(
    f"Archived valid states: {len(model_catalog)}"
)
print(
    f"First archive time: {catalog_times.min()}"
)
print(
    f"Last archive time:  {catalog_times.max()}"
)
print()


# ------------------------------------------------------------
# Match each report
# ------------------------------------------------------------

rows = []


for _, report in reports.iterrows():

    observation_time = (
        report["observation_utc"]
    )


    # --------------------------------------------------------
    # Find nearest archived valid time
    # --------------------------------------------------------

    time_diffs = np.abs(
        (
            catalog_times
            -
            observation_time
        ).total_seconds()
    )

    best_idx = int(
        np.argmin(
            time_diffs
        )
    )

    model_info = (
        model_catalog[
            best_idx
        ]
    )

    model_time = (
        model_info[
            "valid_time"
        ]
    )

    signed_offset_hours = (
        (
            model_time
            -
            observation_time
        ).total_seconds()
        /
        3600.0
    )

    absolute_offset_hours = abs(
        signed_offset_hours
    )


    # --------------------------------------------------------
    # Transform report lon/lat to Hero Dirt grid coordinates
    # --------------------------------------------------------

    lon = float(
        report["longitude"]
    )

    lat = float(
        report["latitude"]
    )


    report_x, report_y = transformer.transform(
        lon,
        lat,
    )


    # --------------------------------------------------------
    # Nearest grid cell
    # --------------------------------------------------------

    ix = int(
        np.argmin(
            np.abs(
                x
                -
                report_x
            )
        )
    )

    iy = int(
        np.argmin(
            np.abs(
                y
                -
                report_y
            )
        )
    )


    grid_x = float(
        x[ix]
    )

    grid_y = float(
        y[iy]
    )


    grid_distance_m = float(
        np.hypot(
            grid_x - report_x,
            grid_y - report_y,
        )
    )


    # --------------------------------------------------------
    # Load selected archived state
    # --------------------------------------------------------

    model = np.load(
        model_info["file"],
        allow_pickle=True,
    )


    if (
        model_info["time_index"]
        is None
    ):

        def model_field(name):
            return model[name][iy, ix]

    else:

        k = model_info[
            "time_index"
        ]

        def model_field(name):
            return model[name][k, iy, ix]


    # --------------------------------------------------------
    # Dynamic model quantities
    # --------------------------------------------------------

    tread_theta = float(
        model_field(
            "tread_theta"
        )
    )

    shallow_theta = float(
        model_field(
            "shallow_theta"
        )
    )

    deep_theta = float(
        model_field(
            "physical_deep_theta"
        )
    )

    theta_0_5cm = float(
        model_field(
            "theta_0_5cm"
        )
    )

    F = float(
        model_field(
            "F"
        )
    )

    hero_score = float(
        model_field(
            "hero_score"
        )
    )

    archived_condition_code = int(
        model_field(
            "condition"
        )
    )

    archived_condition = (
        CONDITION_NAMES.get(
            archived_condition_code,
            "Unknown",
        )
    )

    current_condition = (
        classify_current(
            F
        )
    )


    # --------------------------------------------------------
    # Precipitation field depends on selected state type
    # --------------------------------------------------------

    if (
        model_info["source"]
        ==
        "analysis"
    ):

        precip_mm = float(
            model[
                "analysis_precip_mm"
            ][iy, ix]
        )

    else:

        k = model_info[
            "time_index"
        ]

        precip_mm = float(
            model[
                "precip_mm"
            ][k, iy, ix]
        )


    # --------------------------------------------------------
    # Static properties
    # --------------------------------------------------------

    elevation = safe_value(
        static["elevation"],
        iy,
        ix,
    )

    slope = safe_value(
        static["slope"],
        iy,
        ix,
    )

    southness = safe_value(
        static["southness"],
        iy,
        ix,
    )

    sand = safe_value(
        static["sand"],
        iy,
        ix,
    )

    silt = safe_value(
        static["silt"],
        iy,
        ix,
    )

    clay = safe_value(
        static["clay"],
        iy,
        ix,
    )

    ksat = safe_value(
        static["ksat"],
        iy,
        ix,
    )

    bulk_density = safe_value(
        static["bulk_density"],
        iy,
        ix,
    )

    theta_wp = safe_value(
        static["theta_wp"],
        iy,
        ix,
    )

    theta_fc = safe_value(
        static["theta_fc"],
        iy,
        ix,
    )

    theta_sat = safe_value(
        static["theta_sat"],
        iy,
        ix,
    )


    if pi_array is not None:

        plasticity_index = float(
            pi_array[
                iy,
                ix,
            ]
        )

    else:

        plasticity_index = np.nan


    # --------------------------------------------------------
    # Compare rider observation to current classification
    # --------------------------------------------------------

    condition_match = (
        str(
            report["condition"]
        ).strip()
        ==
        current_condition
    )


    # --------------------------------------------------------
    # Assemble derived matchup row
    # --------------------------------------------------------

    row = {

        # Raw rider observation
        "report_id":
            report["id"],

        "submitted_utc":
            report["submitted_utc"],

        "observation_utc":
            observation_time.isoformat(),

        "trail_name":
            report["trail_name"],

        "latitude":
            lat,

        "longitude":
            lon,

        "observed_condition":
            report["condition"],

        "comments":
            report.get(
                "comments",
                "",
            ),

        "initials":
            report.get(
                "initials",
                "",
            ),


        # Model matchup
        "model_valid_time":
            model_time.isoformat(),

        "model_time_offset_hours":
            signed_offset_hours,

        "model_abs_time_offset_hours":
            absolute_offset_hours,

        "archive":
            model_info["archive"],

        "model_source":
            model_info["source"],

        "forecast_time_index":
            model_info[
                "time_index"
            ],


        # Grid location
        "grid_x":
            grid_x,

        "grid_y":
            grid_y,

        "grid_ix":
            ix,

        "grid_iy":
            iy,

        "grid_distance_m":
            grid_distance_m,


        # Dynamic Hero Dirt state
        "tread_theta":
            tread_theta,

        "shallow_theta":
            shallow_theta,

        "deep_theta":
            deep_theta,

        "theta_0_5cm":
            theta_0_5cm,

        "F":
            F,

        "hero_score":
            hero_score,

        "archived_condition_code":
            archived_condition_code,

        "archived_condition":
            archived_condition,

        "current_reclassified_condition":
            current_condition,

        "condition_match":
            condition_match,

        "precip_mm":
            precip_mm,


        # Static model properties
        "elevation_m":
            elevation,

        "slope_deg":
            slope,

        "southness":
            southness,

        "sand":
            sand,

        "silt":
            silt,

        "clay":
            clay,

        "ksat":
            ksat,

        "bulk_density":
            bulk_density,

        "theta_wp":
            theta_wp,

        "theta_fc":
            theta_fc,

        "theta_sat":
            theta_sat,

        "plasticity_index":
            plasticity_index,
    }


    rows.append(
        row
    )


# ------------------------------------------------------------
# Save derived matchup table
# ------------------------------------------------------------

out = pd.DataFrame(
    rows
)

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True,
)

out.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print(
    f"Matched {len(out)} rider reports."
)
print()
print(
    "Saved:"
)
print(
    f"  {OUTPUT_FILE}"
)
print()


display_cols = [
    "report_id",
    "trail_name",
    "observed_condition",
    "archived_condition",
    "current_reclassified_condition",
    "condition_match",
    "F",
    "hero_score",
    "model_valid_time",
    "model_abs_time_offset_hours",
    "grid_distance_m",
]


print(
    out[
        display_cols
    ].to_string(
        index=False
    )
)

print()
print(
    "============================================"
)
print(
    " MATCHUP COMPLETE"
)
print(
    "============================================"
)
