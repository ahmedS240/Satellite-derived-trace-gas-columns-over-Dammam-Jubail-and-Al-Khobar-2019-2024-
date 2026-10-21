"""
Shared paths and loaders for the revision analyses (A01-A08).

Repository layout:
    data/monthly_city_means/   monthly city-mean gas and NTL series (GEE exports)
    data/grid_cells/           2019-2024 period-mean grid-cell tables (GEE export)
    data/clusters/original_k4/ cluster labels from the original fixed k = 4 run
    external_inputs/           files NOT redistributed here (GADM shapefiles, GeoTIFFs); see README
    results/                   created on first run; all outputs are written here

Requirements: see requirements.txt
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
EXT = REPO / "external_inputs"
ROOT = EXT  # A01 reads EXT/gadm41_SAU_shp, A05 reads EXT/Data/tiff result/Temporal Analysis
OUT = REPO / "results"
OUT.mkdir(exist_ok=True)

CITIES = ["Dammam", "Jubail", "Khobar"]
CITY_LABEL = {"Dammam": "Dammam", "Jubail": "Jubail", "Khobar": "Al-Khobar"}
GASES = ["NO2", "CO", "CH4"]

M = DATA / "monthly_city_means"
MEAN_GAS = {c: M / f"{c}_AllGases_2019_2024.csv" for c in CITIES}          # gee/04_city_monthly_gas_means.js
PROXY = {c: M / f"{c}_gas_NTL_monthly.csv" for c in CITIES}                 # gee/05_city_monthly_gas_and_NTL_means.js
HOTSPOT = {c: DATA / "grid_cells" / f"Hotspot_{c}_FINAL.csv" for c in CITIES}  # gee/06_grid_cell_table_for_clustering.js
CLUSTERED = {c: DATA / "clusters" / "original_k4" / f"{c}_Final_Clustered_MapData.csv" for c in CITIES}


def load_monthly(city):
    """Monthly city means of NO2, CO, CH4 (mean-gas export) and NTL (proxy export)."""
    g = pd.read_csv(MEAN_GAS[city]); p = pd.read_csv(PROXY[city])
    g["date"] = pd.to_datetime(g["date"]); p["date"] = pd.to_datetime(p["date"])
    df = g[["date", "NO2", "CO", "CH4"]].merge(p[["date", "NTL"]], on="date", how="left")
    df["year"] = df["date"].dt.year; df["month"] = df["date"].dt.month
    return df.sort_values("date").reset_index(drop=True)


def load_proxy(city):
    p = pd.read_csv(PROXY[city]); p["date"] = pd.to_datetime(p["date"])
    p["year"] = p["date"].dt.year; p["month"] = p["date"].dt.month
    return p.sort_values("date").reset_index(drop=True)
