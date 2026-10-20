# Satellite-derived trace-gas columns over Dammam, Jubail and Al-Khobar (2019–2024)

Code and derived data for the manuscript:

> Alnashwan, A. S. *Spatio-temporal dynamics of satellite-derived trace-gas columns over selected Saudi cities (2019–2024) with indicative extrapolations to 2030.* (submitted)

This repository holds:
- the Google Earth Engine (GEE) scripts;
- the Python scripts;
- the processing parameters;
- the grid-cell tables, cluster labels, and trend and projection tables generated in the study.

The raw satellite and ancillary data are **not** redistributed. They are openly available from their providers (see [Source data](#source-data)).

**Author:** Ahmed Saad Alnashwan, MSc Environmental Science & Engineering, King Fahd University of Petroleum and Minerals (KFUPM). Supervisor: Dr. Syed Masiur Rahman.

## Repository layout

```
gee/                         Google Earth Engine (JavaScript) scripts, numbered in run order
python/
  original_colab/            the original Colab scripts (fixed k = 4 clustering; first trend/projection run)
  clustering/                final K-means workflow with data-driven k (script + notebook, 600-dpi figures)
  revision_analyses/         A01–A08: coverage, trend tests, correlations, k-selection, projections, figures
data/
  monthly_city_means/        monthly city-mean NO2, CO, CH4 (+ VIIRS NTL) series, 2019-01 to 2024-12
  grid_cells/                2019–2024 period-mean values per grid cell (input to clustering)
  clusters/
    original_k4/             labels from the original fixed-k run (kept for transparency)
    selected_k/              labels, profiles and k-selection tables for the data-selected k (used in the paper)
    Jubail_k4_sensitivity_*  Jubail k = 4 sensitivity solution (Supplementary Table S6)
  trends_projections/        Mann–Kendall/Sen tables, correlation table, 2030 extrapolations with 95% PIs
  diagnostics/               coverage, grid geometry, raster diagnostics, original-run reproduction check
docs/
  processing_parameters.md   every dataset, band, filter, scale, reducer and model setting
  data_dictionary.md         column definitions and units for every CSV
external_inputs/             place non-redistributed inputs here (see its README)
```

## Workflow

| Step | Script | Output in this repo |
|---|---|---|
| 1 | `gee/04_city_monthly_gas_means.js`, `gee/05_city_monthly_gas_and_NTL_means.js` | `data/monthly_city_means/` |
| 2 | `gee/06_grid_cell_table_for_clustering.js` | `data/grid_cells/` |
| 3 | `gee/01_…`, `gee/02_…`, `gee/03_…` | GeoTIFF composites for the maps (not included; regenerate in GEE) |
| 4 | `python/revision_analyses/A02_trend_reanalysis.py` | `data/trends_projections/A02_*.csv` |
| 5 | `python/revision_analyses/A03_correlation_reanalysis.py` | `data/trends_projections/A03_correlation_table.csv` |
| 6 | `python/revision_analyses/A07_optimal_k_selection.py` (or `python/clustering/…selected_k.*`) | `data/clusters/selected_k/` |
| 7 | `python/revision_analyses/A04_kmeans_sensitivity.py`, `A04b_…` | `data/diagnostics/A04_*`, `data/clusters/Jubail_k4_sensitivity_*` |
| 8 | `python/revision_analyses/A06_projection_sensitivity.py` | `data/trends_projections/A06_projection_sensitivity.csv` |
| 9 | `python/revision_analyses/A08_manuscript_figures.py` | trend and cluster figures |

`gee/07_percentile_persistence_EXPLORATORY_not_used_in_paper.js` is an exploratory script. It is kept for completeness, and its output is not used in the manuscript.

## Reproducing the Python analyses

```bash
pip install -r requirements.txt
cd python/revision_analyses
python A02_trend_reanalysis.py        # outputs go to ./results/ at the repository root
python A07_optimal_k_selection.py     # takes several minutes (bootstrap + gap statistic)
```

A02, A03, A04, A04b, A06 and A07 were re-run from this repository layout before release. Their outputs were identical to the tables in `data/`.

A01 needs the GADM shapefiles, and A05 needs the GeoTIFF composites, in `external_inputs/`. The scripts in `python/original_colab/` and `python/clustering/` were written for Google Colab, where the CSVs are uploaded interactively when prompted.

**Re-running the GEE scripts:** the city boundaries are referenced as private GEE assets (`projects/linen-epigram-474718-m9/assets/...`). Replace these paths with your own uploaded boundaries; see [Study-area boundaries](#study-area-boundaries).

## Source data

| Data | Provider / GEE ID | Licence |
|---|---|---|
| Sentinel-5P/TROPOMI L3 NO2, CO (NRTI), CH4 (OFFL) | Copernicus/ESA; `COPERNICUS/S5P/NRTI/L3_NO2`, `…/NRTI/L3_CO`, `…/OFFL/L3_CH4` | Copernicus open-data licence |
| VIIRS DNB monthly night-time lights | NOAA/EOG; `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG` | open |
| GHS-POP R2023A, epoch 2020, 100 m | European Commission JRC; `JRC/GHSL/P2023A/GHS_POP/2020` | CC BY 4.0 |
| WorldPop (city-level comparison) | WorldPop, University of Southampton | CC BY 4.0 |
| GADM 4.1 level-2 boundaries | gadm.org | free for academic use; **redistribution not allowed**, so not included |
| Dammam boundary | OpenStreetMap relation 12399708 | ODbL 1.0 |

### Study-area boundaries

- Jubail: GADM 4.1 `SAU.8.3_1`.
- Al-Khobar: GADM 4.1 `SAU.8.5_1`.
- Dammam: OSM relation `12399708`.

Download these and upload them to GEE as table assets.

## Important notes on interpretation

- NO2 and CO values are **column number densities (mol m⁻²)**. CH4 is a **column-averaged dry-air mixing ratio (ppb)**.
- None of these values are surface concentrations, emission fluxes or exposure estimates.
- Cluster labels are statistical groupings of standardised variables. They do not identify pollution sources.
- The 2030 values are linear extrapolations with 95% prediction intervals. They are not forecasts.

## Licence

- Code (`gee/`, `python/`): MIT, see `LICENSE`.
- Derived data (`data/`) and documentation (`docs/`): CC BY 4.0, see `LICENSE-DATA`.

## Citation

See `CITATION.cff`. A DOI will be minted through Zenodo on release.
