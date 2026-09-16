#!/usr/bin/env python3

from pathlib import Path
import h5py
import numpy as np


ROOT = Path.home() / "HeroDirt"
RAW_DIR = ROOT / "data" / "smap" / "raw"

files = sorted(
    RAW_DIR.glob("*.h5"),
    reverse=True
)

if not files:
    raise RuntimeError("No SMAP HDF5 files found.")

path = files[0]

print()
print("Inspecting:")
print(path)
print()


with h5py.File(path, "r") as h5:

    print("TOP-LEVEL GROUPS")
    print("----------------")
    for key in h5.keys():
        print(key)

    print()

    print("DATASETS")
    print("--------")

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(
                f"{name:75s} "
                f"shape={obj.shape} "
                f"dtype={obj.dtype}"
            )

    h5.visititems(visitor)

    print()

    # --------------------------------------------------------
    # Inspect likely AM group explicitly
    # --------------------------------------------------------

    group_name = "Soil_Moisture_Retrieval_Data_AM"

    if group_name not in h5:
        print(f"{group_name} not found.")
        raise SystemExit

    g = h5[group_name]

    print()
    print("AM GROUP FIELDS")
    print("---------------")

    for key in g.keys():
        print(key)

    print()

    for name in [
        "soil_moisture",
        "latitude",
        "longitude",
        "retrieval_qual_flag",
    ]:

        if name not in g:
            print(f"{name}: NOT FOUND")
            continue

        A = g[name][:]

        print(name)
        print(f"  shape: {A.shape}")

        if np.issubdtype(A.dtype, np.number):

            good = np.isfinite(A)

            if good.any():
                print(
                    f"  range: "
                    f"{np.nanmin(A):.6f} "
                    f"to "
                    f"{np.nanmax(A):.6f}"
                )

        print()


    # --------------------------------------------------------
    # Direct geographic test
    # --------------------------------------------------------

    if all(
        name in g
        for name in [
            "soil_moisture",
            "latitude",
            "longitude",
            "retrieval_qual_flag",
        ]
    ):

        sm = g["soil_moisture"][:].astype(float)
        lat = g["latitude"][:].astype(float)
        lon = g["longitude"][:].astype(float)
        qual = g["retrieval_qual_flag"][:]

        bbox = (
            (lat >= 33.95)
            &
            (lat <= 34.70)
            &
            (lon >= -118.75)
            &
            (lon <= -117.20)
        )

        print("SAN GABRIEL BBOX TEST")
        print("---------------------")

        print(
            "Cells geometrically inside bbox:",
            np.count_nonzero(bbox)
        )

        if np.any(bbox):

            print(
                "Latitude range:",
                np.nanmin(lat[bbox]),
                np.nanmax(lat[bbox])
            )

            print(
                "Longitude range:",
                np.nanmin(lon[bbox]),
                np.nanmax(lon[bbox])
            )

            print(
                "Soil moisture values:",
                sm[bbox]
            )

            print(
                "Quality flags:",
                qual[bbox]
            )

            print(
                "Unique quality flags:",
                np.unique(qual[bbox])
            )

print()
