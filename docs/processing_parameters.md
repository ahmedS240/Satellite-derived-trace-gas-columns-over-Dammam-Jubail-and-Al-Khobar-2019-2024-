# Processing parameters

All values below are taken directly from the scripts in `gee/` and `python/`.

## Study period and areas
- Period: 2019-01-01 to 2024-12-31 (end date exclusive in GEE: 2025-01-01). Monthly series: 72 months per city.
- Jubail: GADM 4.1 `SAU.8.3_1` (≈6464 km²).
- Al-Khobar: GADM 4.1 `SAU.8.5_1` (≈197 km²).
- Dammam: OSM relation 12399708 (≈1846 km²).

## Satellite and ancillary variables

| Variable | GEE collection | Band | Unit | Used for |
|---|---|---|---|---|
| NO2 | `COPERNICUS/S5P/NRTI/L3_NO2` | `NO2_column_number_density` (total column) | mol m⁻² | monthly series, grid cells |
| NO2 (maps, `gee/01`) | `COPERNICUS/S5P/OFFL/L3_NO2` | `NO2_column_number_density` (total column) | mol m⁻² | 2019–2024 mean map composites |
| NO2 (maps, `gee/02`) | `COPERNICUS/S5P/OFFL/L3_NO2` | `tropospheric_NO2_column_number_density` | mol m⁻² | annual map composites (Fig. 2) |
| CO, CH4 (maps) | `COPERNICUS/S5P/OFFL/L3_CO`, `…/OFFL/L3_CH4` | as above | as above | map composites |
| CO | `COPERNICUS/S5P/NRTI/L3_CO` | `CO_column_number_density` | mol m⁻² | monthly series, grid cells |
| CH4 | `COPERNICUS/S5P/OFFL/L3_CH4` | `CH4_column_volume_mixing_ratio_dry_air` | ppb | monthly series, grid cells |
| NTL | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG` | `avg_rad` | nW cm⁻² sr⁻¹ | monthly series, grid cells |
| POP | `JRC/GHSL/P2023A/GHS_POP/2020` | `population_count` | persons per 100 m cell | grid cells |

The GEE L3 collections are pre-gridded by the GEE harpconvert pipeline. In the L3 NO2, CO and CH4 products, a quality filter on `qa_value` is applied during that conversion (NO2 qa_value > 0.75; CO and CH4 qa_value > 0.5; see the GEE catalogue pages for each collection). The annual-composite scripts (`gee/02_*`) additionally mask NO2 pixels with `cloud_fraction` > 0.3. The monthly-series and grid-cell scripts apply no further screening.

## Aggregation

**Monthly city means** (`gee/04`, `gee/05`):
- calendar-month mean image;
- then `reduceRegion(mean)` over the city polygon at 1000 m scale;
- months with no valid retrievals are exported as empty values.

**Grid cells** (`gee/06`):
- `coveringGrid` in EPSG:4326 at 5000 m nominal scale, giving cells of 0.0449° (≈4.5 km × 5.0 km, ≈22.3 km²);
- `reduceRegions(mean)` at 1000 m on the 2019–2024 period-mean image of each variable.

**GHSL NoData:** the value is −200. Four Jubail cells with negative POP are removed before clustering (A04b, A07).

## Trend analysis (A02)

- Annual means.
- Original Mann–Kendall test and Sen's slope on the annual means (n = 6).
- Monthly series:
  - seasonal Mann–Kendall (Hirsch);
  - Hamed–Rao and Yue–Wang variance corrections;
  - trend-free pre-whitening (`pymannkendall`).
- Effective sample size for correlations (A03): Bretherton et al. (1999).

## K-means clustering (A07, `python/clustering/`)

- Features: NO2, CO, CH4, NTL, POP. Standardised with `StandardScaler` (z-scores per city).
- `KMeans(n_init=10, random_state=42)`, k = 2…8.
- Selection rule, fixed before inspecting the profiles:
  1. choose the highest mean silhouette among admissible solutions;
  2. a solution is admissible if its median bootstrap ARI is ≥ 0.75 (200 resamples) and its smallest cluster has ≥ 2 cells.
- Calinski–Harabasz, Davies–Bouldin and gap statistic (20 uniform references) are reported but not used for selection.
- Selected k:

| City | k | Silhouette | Median bootstrap ARI |
|---|---|---|---|
| Dammam | 3 | 0.525 | 0.86 |
| Jubail | 2 | 0.664 | 0.89 |
| Al-Khobar | 4 | 0.438 | 0.78 |

## Extrapolation to 2030 (A06)

- OLS fit to the six annual means (2019–2024), with year centred.
- 95% prediction interval for 2025–2030 (`statsmodels`).
- Theil–Sen line reported for comparison.
- These are indicative extrapolations, not forecasts.
