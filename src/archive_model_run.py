#!/usr/bin/env python3

"""
Hero Dirt Forecast
Archive the current operational model run.

Copies:

    output/current/HeroDirt_current_state.npz
    output/forecast/HeroDirt_forecast_6hourly.npz

into a timestamped archive directory:

    output/archive/YYYYMMDD_HHMM/

The timestamp comes from the current model valid time,
not from the computer clock.

Example
-------
output/archive/20260911_1800/
    HeroDirt_current_state.npz
    HeroDirt_forecast_6hourly.npz
    archive_info.txt
"""

from pathlib import Path
import shutil

import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

CURRENT_FILE = (
    ROOT
    / "output/current/HeroDirt_current_state.npz"
)

FORECAST_FILE = (
    ROOT
    / "output/forecast/HeroDirt_forecast_6hourly.npz"
)

ARCHIVE_ROOT = (
    ROOT
    / "output/archive"
)

ARCHIVE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK FILES
# ============================================================

if not CURRENT_FILE.exists():

    raise FileNotFoundError(
        f"Missing current state:\n{CURRENT_FILE}"
    )


if not FORECAST_FILE.exists():

    raise FileNotFoundError(
        f"Missing forecast:\n{FORECAST_FILE}"
    )


# ============================================================
# READ MODEL TIME
# ============================================================

C = np.load(
    CURRENT_FILE
)


model_time = C[
    "time"
].astype(
    "datetime64[m]"
)


model_time_string = str(
    model_time
)


# Convert:
#
# 2026-09-11T18:00
#
# ->
#
# 20260911_1800

archive_stamp = (
    model_time_string
    .replace(
        "-",
        "",
    )
    .replace(
        ":",
        "",
    )
    .replace(
        "T",
        "_",
    )
)


ARCHIVE_DIR = (
    ARCHIVE_ROOT
    / archive_stamp
)

ARCHIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DESTINATIONS
# ============================================================

CURRENT_DEST = (
    ARCHIVE_DIR
    / "HeroDirt_current_state.npz"
)

FORECAST_DEST = (
    ARCHIVE_DIR
    / "HeroDirt_forecast_6hourly.npz"
)

INFO_DEST = (
    ARCHIVE_DIR
    / "archive_info.txt"
)


# ============================================================
# COPY
# ============================================================

shutil.copy2(
    CURRENT_FILE,
    CURRENT_DEST,
)

shutil.copy2(
    FORECAST_FILE,
    FORECAST_DEST,
)


# ============================================================
# INFO FILE
# ============================================================

with open(
    INFO_DEST,
    "w",
) as f:

    f.write(
        "Hero Dirt Forecast archive\n"
    )

    f.write(
        "==========================\n\n"
    )

    f.write(
        f"Model analysis time: "
        f"{model_time_string} UTC\n"
    )

    f.write(
        f"Current state: "
        f"{CURRENT_DEST.name}\n"
    )

    f.write(
        f"Forecast: "
        f"{FORECAST_DEST.name}\n"
    )


# ============================================================
# PRINT
# ============================================================

print()
print(
    "============================================"
)
print(
    " HERO DIRT MODEL ARCHIVE"
)
print(
    "============================================"
)
print()

print(
    f"Model time:"
)

print(
    f"  {model_time_string} UTC"
)

print()

print(
    f"Archive directory:"
)

print(
    f"  {ARCHIVE_DIR}"
)

print()

print(
    "Archived:"
)

print(
    f"  {CURRENT_DEST.name}"
)

print(
    f"  {FORECAST_DEST.name}"
)

print(
    f"  {INFO_DEST.name}"
)

print()

print(
    "============================================"
)
