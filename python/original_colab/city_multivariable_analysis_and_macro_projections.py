"""
==============================================================================
City-Level Multi-Variable Temporal Analysis & 2030 Macro-Projections
==============================================================================
Author      : Ahmed Saad Alnashwan
Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
              MSc Environmental Science & Engineering
Supervisor  : Dr. Syed Rahman
Study Area  : Dammam, Jubail, Al-Khobar — Eastern Province, Saudi Arabia
Period      : 2019–2024 (Historical) | 2030 (Projected)
Variables   : NO₂, CO, CH₄ (gases) · NTL, POP (activity proxies)
Environment : Google Colab
==============================================================================

Description
-----------
This script performs a city-level multi-variable temporal analysis linking
atmospheric gas concentrations derived from Sentinel-5P / TROPOMI with urban
activity proxies (Nighttime Lights and Population Density). For each uploaded
city file the script:

    1. Computes Pearson correlations between each gas and each activity proxy,
       with two-tailed p-values and significance flags.
    2. Aggregates monthly observations into annual means and computes
       year-on-year (YoY) percentage growth rates for all variables.
    3. Generates dual-axis multi-variable trend plots for every gas–proxy pair
       available in the data (300 DPI, publication quality).
    4. Fits OLS linear regression models to annual means and projects all
       variables (gases and proxies) to 2030.
    5. Exports a consolidated Pearson correlation table, a YoY growth table,
       and a 2030 projection table as downloadable CSVs.

Dependencies
------------
    pip install pandas numpy matplotlib seaborn scipy scikit-learn

Usage (Google Colab)
--------------------
    Run all cells. When prompted, upload city-level proxy CSV files named in
    the format:
        <CityName>_Final.csv   (e.g., Dammam_Final.csv)

    Required CSV columns:
        - year  : Integer calendar year (e.g., 2019, 2020, …, 2024)
        - date  : Date string used as the x-axis label on time-series plots
        - NO2   : NO₂ column density        [mol m⁻²]
        - CO    : CO column density         [mol m⁻²]
        - CH4   : CH₄ column density        [ppb]
        - NTL   : Nighttime Lights radiance [nW cm⁻² sr⁻¹]
        - POP   : Population density index  [persons km⁻²]

Output
------
    - City_Temporal_Analysis/<City>_<Gas>_vs_<Proxy>_Trend.png  (all pairs)
    - <City>_Pearson_Correlation.csv                             (auto-downloaded)
    - <City>_YoY_Growth_Rates.csv                               (auto-downloaded)
    - <City>_2030_Macro_Projections.csv                         (auto-downloaded)
==============================================================================
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import io
import os
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── Third-Party Libraries ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from IPython.display import display

# ─── Colab-Specific Import ────────────────────────────────────────────────────
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    warnings.warn(
        "google.colab not found. File upload/download steps will be skipped. "
        "Place city CSV files in the working directory instead.",
        RuntimeWarning,
        stacklevel=2,
    )

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Constants & Configuration
# ==============================================================================

#: Atmospheric gas columns targeted by the analysis.
GASES: List[str] = ["NO2", "CO", "CH4"]

#: Urban activity proxy columns.
PROXIES: List[str] = ["NTL", "POP"]

#: All variable columns (gases + proxies) used for growth rate and projection.
ALL_VARIABLES: List[str] = GASES + PROXIES

#: Final historical year whose observed value anchors the projection baseline.
HIST_END_YEAR: int = 2024

#: Projection target year.
PROJ_END_YEAR: int = 2030

#: Minimum paired observations required to compute a Pearson correlation.
MIN_CORR_OBS: int = 3

#: Significance threshold for Pearson p-value flagging.
ALPHA: float = 0.05

#: Output directory for saved plot files.
OUTPUT_DIR: str = "City_Temporal_Analysis"

#: LaTeX-formatted axis labels for gas columns.
GAS_YLABEL: Dict[str, str] = {
    "NO2": r"NO$_2$ Column Density (mol m$^{-2}$)",
    "CO":  r"CO Column Density (mol m$^{-2}$)",
    "CH4": r"CH$_4$ Column Density (ppb)",
}

#: Axis labels for proxy columns.
PROXY_YLABEL: Dict[str, str] = {
    "NTL": r"NTL Radiance (nW cm$^{-2}$ sr$^{-1}$)",
    "POP": r"Population Density (persons km$^{-2}$)",
}

#: LaTeX-formatted gas name fragments used in plot titles.
GAS_TITLE: Dict[str, str] = {
    "NO2": r"NO$_2$",
    "CO":  "CO",
    "CH4": r"CH$_4$",
}

#: Proxy name fragments used in plot titles.
PROXY_TITLE: Dict[str, str] = {
    "NTL": "Nighttime Lights (NTL)",
    "POP": "Population Density (POP)",
}

#: Colours for primary (gas) and secondary (proxy) axes.
GAS_COLOR:   str = "#d62728"   # red family
PROXY_COLOR: str = "#1f6fb2"   # blue family

# ─── Matplotlib / Seaborn Global Style ───────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update(
    {
        "font.family":        "DejaVu Sans",
        "axes.titlepad":      14,
        "axes.labelpad":      8,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.8",
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    }
)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class CityAnalysisResult:
    """
    Container for all computed outputs of a single city temporal analysis.

    Attributes
    ----------
    city : str
        Name of the study city extracted from the filename.
    correlation_df : pd.DataFrame
        Pearson correlation coefficients and p-values for all gas–proxy pairs.
    growth_df : pd.DataFrame
        Annual means and year-on-year percentage growth rates per variable.
    projection_df : pd.DataFrame
        OLS 2030 projections with observed 2024 baseline and % change.
    """
    city:           str
    correlation_df: pd.DataFrame
    growth_df:      pd.DataFrame
    projection_df:  pd.DataFrame


# ==============================================================================
# Core Analysis Functions
# ==============================================================================

def load_city_dataframe(
    file_name: str,
    file_content: bytes,
) -> Optional[pd.DataFrame]:
    """
    Parse a city CSV file from raw bytes and perform basic validation.

    Column names are stripped of leading/trailing whitespace. The 'year'
    column is coerced to integer; rows with unparseable year values are dropped.

    Parameters
    ----------
    file_name : str
        Original filename (used for logging only).
    file_content : bytes
        Raw byte content of the uploaded CSV file.

    Returns
    -------
    pd.DataFrame or None
        Parsed and validated DataFrame, or None if the file is malformed or
        the 'year' column is absent.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as exc:
        logger.error("Failed to parse '%s': %s", file_name, exc)
        return None

    df.columns = df.columns.str.strip()

    if "year" not in df.columns:
        logger.error("'year' column missing in '%s'. Skipping.", file_name)
        return None

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.dropna(subset=["year"], inplace=True)
    df["year"] = df["year"].astype(int)
    logger.info("Loaded '%s' — %d rows, %d columns.", file_name, *df.shape)
    return df


def compute_pearson_correlations(
    df: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    """
    Compute Pearson correlations between every gas–proxy pair.

    Only pairs for which both columns are present and at least MIN_CORR_OBS
    non-NaN paired observations exist are included. A significance flag
    ('Significant') is appended based on ALPHA.

    Parameters
    ----------
    df : pd.DataFrame
        City DataFrame containing gas and proxy columns.
    city : str
        City name used in log messages.

    Returns
    -------
    pd.DataFrame
        Rows: one per gas–proxy pair.
        Columns: Gas, Proxy, Pearson r, P-Value, Significant (α=0.05).
        Empty DataFrame if no valid pairs found.
    """
    records = []
    for gas in GASES:
        for proxy in PROXIES:
            if gas not in df.columns or proxy not in df.columns:
                continue
            valid = df[[gas, proxy]].dropna()
            if len(valid) < MIN_CORR_OBS:
                logger.warning(
                    "%s: Insufficient paired obs for %s–%s (%d < %d). Skipping.",
                    city, gas, proxy, len(valid), MIN_CORR_OBS,
                )
                continue
            coeff, p_val = pearsonr(valid[gas], valid[proxy])
            records.append(
                {
                    "Gas":                    gas,
                    "Proxy":                  proxy,
                    "Pearson r":              round(coeff, 4),
                    "P-Value":                round(p_val, 6),
                    f"Significant (\u03b1={ALPHA})": "Yes" if p_val < ALPHA else "No",
                }
            )
    if not records:
        logger.warning("%s: No valid gas–proxy pairs for Pearson correlation.", city)
    return pd.DataFrame(records)


def compute_annual_growth_rates(
    df: pd.DataFrame,
    city: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate monthly data to annual means and compute YoY growth rates.

    Year-on-year (YoY) percentage change is defined as:
        (mean_year_t − mean_year_(t-1)) / |mean_year_(t-1)| × 100

    Parameters
    ----------
    df : pd.DataFrame
        City DataFrame with a 'year' column and variable columns.
    city : str
        City name used in log messages.

    Returns
    -------
    annual_df : pd.DataFrame
        Annual mean values for all available variables.
    growth_df : pd.DataFrame
        Annual means plus YoY growth rate columns (suffixed '_Growth_%').
    """
    available = [v for v in ALL_VARIABLES if v in df.columns]
    if not available:
        logger.warning("%s: No recognised variable columns found for growth rate calc.", city)
        return pd.DataFrame(), pd.DataFrame()

    annual_df = df.groupby("year")[available].mean().reset_index()
    growth_df = annual_df.copy()

    for col in available:
        growth_df[f"{col}_Growth_%"] = (
            growth_df[col].pct_change() * 100
        ).round(4)

    growth_cols = ["year"] + [c for c in growth_df.columns if "Growth" in c]
    logger.info(
        "%s: Annual growth rates computed for %d variables across %d years.",
        city, len(available), len(annual_df),
    )
    return annual_df, growth_df[growth_cols]


def compute_ols_projections(
    annual_df: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    """
    Fit OLS linear regression models and project all variables to PROJ_END_YEAR.

    For each variable the model is fit on all available annual mean data points.
    The last observed value (most recent year in the data, typically 2024) is
    used as the baseline for the percentage change calculation.

    Parameters
    ----------
    annual_df : pd.DataFrame
        Annual mean DataFrame with a 'year' column.
    city : str
        City name used in log messages.

    Returns
    -------
    pd.DataFrame
        One row per variable with columns:
        Variable, Last Observed Year, Observed Mean (HIST_END_YEAR),
        Projected Mean (PROJ_END_YEAR), OLS R², % Change.
        Empty DataFrame if no variables can be projected.
    """
    records = []
    available = [v for v in ALL_VARIABLES if v in annual_df.columns]

    for var in available:
        temp = annual_df[["year", var]].dropna()
        if len(temp) < 2:
            logger.warning(
                "%s: Insufficient data to project '%s' (%d point(s)).", city, var, len(temp)
            )
            continue

        X = temp["year"].values.reshape(-1, 1)
        y = temp[var].values
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))

        proj_val  = float(model.predict([[PROJ_END_YEAR]])[0])
        last_val  = float(y[-1])
        last_year = int(temp["year"].iloc[-1])
        pct       = ((proj_val - last_val) / abs(last_val)) * 100 if last_val else float("nan")

        records.append(
            {
                "Variable":                         var,
                "Last Observed Year":               last_year,
                f"Observed Mean ({last_year})":     round(last_val, 8),
                f"Projected Mean ({PROJ_END_YEAR})": round(proj_val, 8),
                "OLS R\u00b2":                      round(r2, 4),
                f"% Change ({last_year}\u2192{PROJ_END_YEAR})": round(pct, 2),
            }
        )

    if not records:
        logger.warning("%s: No variables could be projected.", city)
    return pd.DataFrame(records)


# ==============================================================================
# Visualization
# ==============================================================================

def build_dual_axis_figure(
    city: str,
    df: pd.DataFrame,
    gas: str,
    proxy: str,
) -> plt.Figure:
    """
    Build a dual-axis time-series figure comparing one gas with one proxy.

    The primary (left) y-axis plots the gas concentration; the secondary
    (right) y-axis plots the activity proxy. Both series are drawn against
    the 'date' column on the x-axis. A Pearson r annotation is placed in
    the upper-left corner when sufficient data is available.

    Parameters
    ----------
    city : str
        City name for the plot title.
    df : pd.DataFrame
        City DataFrame containing 'date', the gas column, and the proxy column.
    gas : str
        Gas column name (e.g., 'NO2').
    proxy : str
        Proxy column name (e.g., 'NTL').

    Returns
    -------
    matplotlib.figure.Figure
        Configured dual-axis figure (caller responsible for saving/showing).
    """
    fig, ax1 = plt.subplots(figsize=(13, 6))

    # ── Primary axis: gas ─────────────────────────────────────────────────────
    ax1.plot(
        df["date"], df[gas],
        color=GAS_COLOR, linewidth=2.2, marker="o", markersize=4,
        label=GAS_TITLE.get(gas, gas),
    )
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel(GAS_YLABEL.get(gas, f"{gas} Concentration"), fontsize=11, color=GAS_COLOR)
    ax1.tick_params(axis="y", labelcolor=GAS_COLOR)
    ax1.tick_params(axis="x", rotation=45)

    # ── Secondary axis: proxy ─────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(
        df["date"], df[proxy],
        color=PROXY_COLOR, linewidth=2.2, linestyle="--", marker="s", markersize=4,
        label=PROXY_TITLE.get(proxy, proxy),
    )
    ax2.set_ylabel(PROXY_YLABEL.get(proxy, proxy), fontsize=11, color=PROXY_COLOR)
    ax2.tick_params(axis="y", labelcolor=PROXY_COLOR)

    # ── Pearson r annotation ──────────────────────────────────────────────────
    valid = df[[gas, proxy]].dropna()
    if len(valid) >= MIN_CORR_OBS:
        coeff, p_val = pearsonr(valid[gas], valid[proxy])
        sig = "\u2713 Sig." if p_val < ALPHA else "\u2717 n.s."
        annot = f"Pearson r = {coeff:+.4f}\np = {p_val:.4f}  ({sig})"
        ax1.text(
            0.02, 0.97, annot,
            transform=ax1.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.75", alpha=0.9),
        )

    # ── Legend: merge both axes ───────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="lower right", fontsize=9,
        framealpha=0.9, edgecolor="0.8",
    )

    ax1.set_title(
        f"{city}: {GAS_TITLE.get(gas, gas)} Concentration vs "
        f"{PROXY_TITLE.get(proxy, proxy)}\n"
        f"Monthly Time-Series Comparison (2019\u20132024)",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.65)
    fig.tight_layout()
    return fig


# ==============================================================================
# Reporting
# ==============================================================================

def print_section_header(title: str, width: int = 62) -> None:
    """Print a formatted section header to stdout."""
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")


def export_csv(
    df: pd.DataFrame,
    filename: str,
    label: str,
) -> None:
    """
    Save a DataFrame to CSV and trigger a Colab download if running in Colab.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Output filename (written to the current working directory).
    label : str
        Human-readable description for the log message.
    """
    df.to_csv(filename, index=False, float_format="%.8g")
    logger.info("%s saved to '%s'.", label, filename)
    if IN_COLAB:
        logger.info("Initiating download: %s …", filename)
        colab_files.download(filename)


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def run_analysis() -> None:
    """
    Execute the complete city-level multi-variable temporal analysis pipeline.

    Workflow
    --------
    1. Prompt the user to upload city-level proxy CSV files (Colab) or scan
       the working directory (local fallback).
    2. For each uploaded city file:
        a. Validate and load the DataFrame.
        b. Compute Pearson correlations for all gas–proxy pairs.
        c. Aggregate to annual means and compute YoY growth rates.
        d. Fit OLS models and project all variables to 2030.
        e. Build and save dual-axis trend plots for every available gas–proxy
           combination.
        f. Export Pearson, growth rate, and projection CSVs.
    3. Print a final completion notice.
    """
    print_section_header("City-Level Multi-Variable Temporal Analysis & 2030 Projections")
    logger.info(
        "Gases: %s | Proxies: %s | Projection year: %d",
        ", ".join(GASES), ", ".join(PROXIES), PROJ_END_YEAR,
    )

    # ── Step 1: File Ingestion ─────────────────────────────────────────────────
    if IN_COLAB:
        print(
            "\nPlease upload your city-level proxy CSV files "
            "(e.g., Dammam_Final.csv, Jubail_Final.csv):\n"
        )
        uploaded: Dict[str, bytes] = colab_files.upload()
        if not uploaded:
            logger.error("No files were uploaded. Aborting.")
            return
    else:
        local_csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
        if not local_csvs:
            logger.error("No CSV files found in the working directory. Aborting.")
            return
        uploaded = {}
        for fname in local_csvs:
            with open(fname, "rb") as fh:
                uploaded[fname] = fh.read()
        logger.info("Local mode: found %d CSV file(s).", len(uploaded))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results: List[CityAnalysisResult] = []

    # ── Step 2: City Analysis Loop ─────────────────────────────────────────────
    for file_name, file_content in uploaded.items():
        city_name = file_name.split("_")[0] if "_" in file_name else os.path.splitext(file_name)[0]
        print_section_header(f"Macro-Analysis — {city_name}")

        df = load_city_dataframe(file_name, file_content)
        if df is None:
            continue

        # ── 2a. Pearson Correlation ────────────────────────────────────────────
        logger.info("%s: Computing Pearson correlations …", city_name)
        corr_df = compute_pearson_correlations(df, city_name)
        if not corr_df.empty:
            print(f"\n  [1/4] Pearson Correlations — {city_name}")
            display(corr_df)

        # ── 2b. Annual Growth Rates ────────────────────────────────────────────
        logger.info("%s: Computing annual growth rates …", city_name)
        annual_df, growth_df = compute_annual_growth_rates(df, city_name)
        if not growth_df.empty:
            print(f"\n  [2/4] Year-on-Year Growth Rates (%) — {city_name}")
            display(growth_df)

        # ── 2c. 2030 Projections ───────────────────────────────────────────────
        logger.info("%s: Computing 2030 OLS projections …", city_name)
        proj_df = pd.DataFrame()
        if not annual_df.empty:
            proj_df = compute_ols_projections(annual_df, city_name)
            if not proj_df.empty:
                print(f"\n  [3/4] 2030 Macro-Projections — {city_name}")
                display(proj_df)

        # ── 2d. Dual-Axis Trend Plots (all gas–proxy pairs) ────────────────────
        print(f"\n  [4/4] Generating Dual-Axis Trend Plots — {city_name}")
        if "date" not in df.columns:
            logger.warning(
                "%s: 'date' column not found — trend plots require a date column. Skipping plots.",
                city_name,
            )
        else:
            for gas in GASES:
                for proxy in PROXIES:
                    if gas not in df.columns or proxy not in df.columns:
                        logger.warning(
                            "%s: '%s' or '%s' column absent — skipping plot.",
                            city_name, gas, proxy,
                        )
                        continue
                    logger.info("  Building plot: %s vs %s …", gas, proxy)
                    fig = build_dual_axis_figure(city_name, df, gas, proxy)
                    plot_path = os.path.join(
                        OUTPUT_DIR, f"{city_name}_{gas}_vs_{proxy}_Trend.png"
                    )
                    fig.savefig(plot_path)
                    logger.info("  Saved: %s", plot_path)
                    plt.show()
                    plt.close(fig)

        # ── 2e. CSV Exports ────────────────────────────────────────────────────
        if not corr_df.empty:
            export_csv(
                corr_df,
                f"{city_name}_Pearson_Correlation.csv",
                f"{city_name} Pearson correlation table",
            )
        if not growth_df.empty:
            export_csv(
                growth_df,
                f"{city_name}_YoY_Growth_Rates.csv",
                f"{city_name} YoY growth rate table",
            )
        if not proj_df.empty:
            export_csv(
                proj_df,
                f"{city_name}_2030_Macro_Projections.csv",
                f"{city_name} 2030 projection table",
            )

        all_results.append(
            CityAnalysisResult(
                city=city_name,
                correlation_df=corr_df,
                growth_df=growth_df,
                projection_df=proj_df,
            )
        )

    # ── Step 3: Completion Notice ──────────────────────────────────────────────
    n_cities = len(all_results)
    n_plots  = n_cities * len(GASES) * len(PROXIES)

    print(
        f"\n✓  Analysis complete.\n"
        f"   Cities processed : {n_cities}\n"
        f"   Trend plots saved: up to {n_plots} (in '{OUTPUT_DIR}/')\n"
        f"   CSVs downloaded  : Pearson · Growth Rates · 2030 Projections per city\n"
        f"   Retrieve plots from the folder icon in the Colab left sidebar.\n"
    )


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    run_analysis()
