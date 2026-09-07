# Spatio-Temporal Analysis of Sentinel-5P/TROPOMI Atmospheric Gases over Jubail, Dammam and Al Khobar

**Author:** Ahmed Saad Alnashwan (202504410) — MSc Environmental Science & Engineering
**Institution:** King Fahd University of Petroleum and Minerals (KFUPM)
**Supervisor:** Dr. Syed Masiur Rahman
**Period:** 2019–2024, extrapolated to 2030

This repository contains the Python/Colab analysis code for the project, with each script placed alongside the exact CSV data it consumes so the pipeline can be reproduced end to end.

## Contents

| Folder | Script(s) | CSV input(s) | What it does |
|---|---|---|---|
| [`01-trend-analysis/`](01-trend-analysis) | `Trend Analysis 2030 Projections and Visualizations for Sentinel5PTROPOMI Gas Concentrations.py` | `Dammam_AllGases_2019_2024.csv`, `Jubail_AllGases_2019_2024.csv`, `Khobar_AllGases_2019_2024.csv` | Mann-Kendall trend test + Sen's slope on monthly NO₂/CO/CH₄ series per city; linear extrapolation to 2030 |
| [`02-hotspot-clustering/`](02-hotspot-clustering) | `Spatial Hotspot KMeans Clustering Analysis... 1.py`, `...2.py`, `...3.py` (run in order; part 3 is the final, hardened visualization step) | `Hotspot_Dammam_FINAL.csv`, `Hotspot_Jubail_FINAL.csv`, `Hotspot_Khobar_FINAL.csv` | K-Means (k=4) clustering of Z-scored [NO₂, CO, CH₄, NTL, POP] per grid cell to identify persistent pollution hotspots |
| [`03-macro-projections/`](03-macro-projections) | `City_Level Multi_Variable Temporal Analysis & 2030 Macro_Projections.py` | `Al_Jubail_Final.csv`, `Al_Khobar_Final.csv`, `Dammam_Final.csv` | City-level multi-variable temporal analysis linking gas concentrations to nighttime-lights and population proxies, with 2030 macro projections |

## Background

The underlying satellite data (Sentinel-5P/TROPOMI NO₂/CO/CH₄, VIIRS nighttime lights, gridded population) was acquired and pre-processed in Google Earth Engine, exported to the CSVs included here, then analyzed in Google Colab using the scripts in this repository. Full methodology, findings, and limitations are documented in the accompanying thesis report.

## Reproducing

Each folder is self-contained: open the script(s) in Google Colab or a local Python environment, and point them at the CSV(s) in the same folder. `random_state = 42` is used throughout for reproducibility.
