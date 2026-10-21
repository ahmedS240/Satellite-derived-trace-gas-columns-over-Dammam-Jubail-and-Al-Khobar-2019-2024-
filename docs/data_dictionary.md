# Data dictionary

## Units used throughout

| Column | Meaning | Unit |
|---|---|---|
| `NO2` | TROPOMI NO2 total vertical column | mol m⁻² |
| `CO` | TROPOMI CO total column | mol m⁻² |
| `CH4` | TROPOMI column-averaged dry-air CH4 mixing ratio | ppb |
| `NTL` | VIIRS DNB monthly average radiance (`avg_rad`) | nW cm⁻² sr⁻¹ |
| `POP` | GHS-POP R2023A 2020, mean of 100 m cells within the grid cell | persons per 100 m pixel |

## `data/monthly_city_means/`

Each file holds one row per calendar month, 2019-01 to 2024-12. The columns are:
- `date` (YYYY-MM), `year`, `month`;
- the city-mean `NO2`, `CO` and `CH4`;
- `*_gas_NTL_monthly.csv` additionally contains `NTL`.

Notes:
- An empty cell means the month had no valid retrievals.
- `system:index` and `.geo` are GEE export artefacts. They are empty.
- The `POP` column in `*_gas_NTL_monthly.csv` is empty and is not used.

## `data/grid_cells/Hotspot_<City>_FINAL.csv`

Each file holds one row per grid cell, with the 2019–2024 period means. The columns are:
- `Grid_ID`: the GEE coveringGrid index, "x,y";
- `Longitude`, `Latitude`: the cell centroid;
- the five variables above;
- `.geo`: the cell polygon as GeoJSON (EPSG:4326).

**Jubail:** four cells have negative `POP`, which is GHSL NoData contamination. They are excluded from all clustering.

## `data/clusters/`

### `selected_k/<City>_Final_Clustered_MapData.csv`
- The grid cells with the K-means `Cluster` (raw label) for the data-selected k.
- `Cluster_Label`: a descriptive name based only on the cluster's mean NO2 column rank. It is **not** a source attribution.

### `selected_k/<City>_k_selection_table.csv` and `A07_k_selection_table.csv`
- Silhouette, Calinski–Harabasz, Davies–Bouldin and gap statistic for k = 2–8.
- Bootstrap ARI (median and 5th percentile).
- Cluster sizes, and whether the solution is admissible.

### `A07_selected_k.csv`
- The chosen k per city, with the reason, and the k that the other indices would choose.

### `A07_cluster_profiles_selected_k.csv`
- Mean of each variable per cluster.
- `n` and `pct`: the cell count and share.
- The cluster centroid location.

### `original_k4/`
- Labels from the original fixed k = 4 run, before the GHSL NoData cells were removed.
- Its `Cluster_Label` wording ("High Pollution / Very High Activity", etc.) was **superseded in the revision**.
- It is kept only to document the reproduction check (`data/diagnostics/A04_reproduction_check.csv`).

### `Jubail_k4_sensitivity_labels.csv` and `_profiles.csv`
- The Jubail k = 4 solution after masking the NoData cells (Supplementary Table S6).

## `data/trends_projections/`

### `A02_annual_trend_check.csv`
- Mann–Kendall test on the six annual means.
- Sen's slope per year, and the OLS slope per year.
- The 2030 OLS extrapolation and percentage change from 2024.

### `A02_monthly_trend_tests.csv`
- Monthly-series tests: original, Hamed–Rao and Yue–Wang Mann–Kendall, seasonal Mann–Kendall, and TFPW.
- Lag-1 autocorrelation of the deseasonalised series.

### `A03_correlation_table.csv`
- Monthly gas–NTL Pearson and Spearman correlations.
- Bretherton effective sample size, and the autocorrelation-adjusted p-value.
- The same statistics for the anomaly series.

### `A06_projection_sensitivity.csv`
- The OLS 2030 value with its 95% prediction interval (absolute and %).
- The Theil–Sen 2030 value.
- Sensitivity to the start year (2020–2024 and 2021–2024 fits).

## `data/diagnostics/`

### `A01_valid_months_by_year.csv`
- Valid months per variable and year.

### `A01_grid_cell_geometry*.csv`
- Cell size, area, and the fraction of each cell inside the city polygon.

### `A01_boundary_polygons.csv`
- Boundary source, ID and area.

### `A04_*`
- Reproduction of the original k = 4 labels.
- Seed and bootstrap stability.
- Indices for k = 2–8 on the unmasked data.

### `A05_fig2_raster_diagnostics.csv`
- Pixel statistics of the GeoTIFF composites behind the Fig. 2 maps.
