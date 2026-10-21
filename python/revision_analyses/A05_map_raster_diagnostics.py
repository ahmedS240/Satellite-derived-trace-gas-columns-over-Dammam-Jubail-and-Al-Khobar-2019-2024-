"""
A05 - Diagnostics of the annual composite rasters used for Fig. 2
(responds to postdoc comments on Fig. 2: CH4 legend saturation and valid-retrieval coverage)

Reads ../../Data/tiff result/Temporal Analysis/<GAS>_<City>_<YEAR>.tif (exported from GEE, EPSG:4326, 0.018 deg)
and reports, for each raster: band name, pixel size, number of pixels, number of valid (non-NaN) pixels,
valid share, and value distribution (min, 2nd, 50th, 98th percentile, max), plus the share of valid pixels
above the current Fig. 2 legend maximum.

Output: A05_fig2_raster_diagnostics.csv
"""
import glob
import re
import numpy as np
import pandas as pd
import rasterio
from common import ROOT, OUT

LEGEND = {"NO2": (0, 0.0002), "CO": (0.030, 0.036), "CH4": (1850, 1950)}  # as stated in the Fig. 2 caption
rows = []
for f in sorted(glob.glob(str(ROOT / "Data" / "tiff result" / "Temporal Analysis" / "*.tif"))):
    name = f.split("/")[-1].split("\\")[-1]
    gas, city, year = re.match(r"(NO2|CO|CH4)_(.+)_(\d{4})\.tif", name).groups()
    with rasterio.open(f) as r:
        a = r.read(1).astype(float)
        band = r.descriptions[0]
        res = r.res[0]
    v = a[np.isfinite(a)]
    lo, hi = LEGEND[gas]
    rows.append({"Gas": gas, "City": city, "Year": int(year), "Band": band, "Pixel_deg": round(res, 4),
                 "Pixels": a.size, "Valid": v.size, "Valid_share": v.size / a.size,
                 "Min": v.min() if v.size else np.nan,
                 "P02": np.percentile(v, 2) if v.size else np.nan,
                 "Median": np.median(v) if v.size else np.nan,
                 "P98": np.percentile(v, 98) if v.size else np.nan,
                 "Max": v.max() if v.size else np.nan,
                 "Share_above_legend_max": (v > hi).mean() if v.size else np.nan,
                 "Share_below_legend_min": (v < lo).mean() if v.size else np.nan})
df = pd.DataFrame(rows).sort_values(["Gas", "City", "Year"])
df.to_csv(OUT / "A05_fig2_raster_diagnostics.csv", index=False)
pd.set_option("display.width", 250)
print(df.drop(columns=["Band"]).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
