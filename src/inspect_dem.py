import rasterio
import numpy as np
from pathlib import Path

dem_path = Path.home() / "HeroDirt" / "static" / "dem" / "SanGabriels_USGS10m.tif"

with rasterio.open(dem_path) as src:
    z = src.read(1, masked=True)

    print("File:", dem_path)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Resolution:", src.res)
    print("Transform:")
    print(src.transform)
    print("NoData:", src.nodata)

    valid = z.compressed()

    print("Valid cells:", valid.size)

    if valid.size > 0:
        print("Elevation min:", float(np.min(valid)))
        print("Elevation max:", float(np.max(valid)))
        print("Elevation mean:", float(np.mean(valid)))
