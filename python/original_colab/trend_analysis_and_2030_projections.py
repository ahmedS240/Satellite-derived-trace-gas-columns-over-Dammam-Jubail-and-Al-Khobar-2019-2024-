!pip install pandas numpy pymannkendall scikit-learn matplotlib seaborn

"""
==============================================================================
Sentinel-5P / TROPOMI Atmospheric Gas Trend Analysis and 2030 Projections
==============================================================================
Author      : Ahmed Saad Alnashwan
Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
              MSc Environmental Science & Engineering
Supervisor  : Dr. Syed Rahman
Study Area  : Dammam, Jubail, Al-Khobar — Eastern Province, Saudi Arabia
Period      : 2019–2024 (Historical) | 2025–2030 (Projected)
Gases       : NO₂, CO, CH₄
Environment : Google Colab
==============================================================================

Description
-----------
This script performs a rigorous spatro-temporal trend analysis on annual mean
atmospheric gas concentrations derived from Sentinel-5P / TROPOMI satellite
data. For each city–gas combination, the script:

    1. Ingests and validates user-uploaded CSV files.
    2. Aggregates raw observations into annual means.
    3. Applies the Mann–Kendall non-parametric monotonic trend test.
    4. Fits an Ordinary Least Squares (OLS) linear regression model.
    5. Projects annual mean concentrations to 2030.
    6. Produces publication-quality trend charts (300 DPI).
    7. Exports a consolidated summary table as a downloadable CSV.

Dependencies
------------
    pip install pandas numpy pymannkendall scikit-learn matplotlib seaborn

Usage (Google Colab)
--------------------
    Run all cells. When prompted, upload CSV files named in the format:
        <CityName>_AllGases_2019_2024.csv

    Required CSV columns:
        - year  : Integer calendar year (e.g., 2019, 2020, …, 2024)
        - NO2   : Annual mean NO₂ column density  [mol/m²]
        - CO    : Annual mean CO column density   [mol/m²]
        - CH4   : Annual mean CH₄ column density  [ppb or mol/m²]

Output
------
    - PhaseC_Trend_and_Projections_Summary.csv  (auto-downloaded)
    - Trend_Plots/<City>_<Gas>_Trend_2030.png   (saved in Colab filesystem)
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
import pymannkendall as mk
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
        "Place CSV files in the working directory instead.",
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

#: Atmospheric gases targeted by the analysis.
TARGET_GASES: List[str] = ["NO2", "CO", "CH4"]

#: Formatted gas labels for plot axis annotations and titles.
GAS_LABELS: Dict[str, str] = {
    "NO2": r"NO$_2$ Column Density (mol m$^{-2}$)",
    "CO":  r"CO Column Density (mol m$^{-2}$)",
    "CH4": r"CH$_4$ Column Density (ppb)",
}

#: LaTeX-formatted gas name fragments used in plot titles.
GAS_TITLES: Dict[str, str] = {
    "NO2": r"NO$_2$",
    "CO":  "CO",
    "CH4": r"CH$_4$",
}

#: First historical year in the dataset.
HIST_START_YEAR: int = 2019

#: Final historical year (inclusive) in the dataset.
HIST_END_YEAR: int = 2024

#: Projection horizon year.
PROJ_END_YEAR: int = 2030

#: Minimum data points required to run trend analysis.
MIN_DATA_POINTS: int = 3

#: Output directory for saved plot files.
OUTPUT_DIR: str = "Trend_Plots"

#: Output CSV filename for the consolidated summary table.
OUTPUT_CSV: str = "PhaseC_Trend_and_Projections_Summary.csv"

#: Significance threshold for Mann–Kendall p-value reporting.
ALPHA: float = 0.05

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
class TrendResult:
    """
    Container for all computed outputs of a single city–gas analysis.

    Attributes
    ----------
    city : str
        Name of the study city.
    gas : str
        Target gas identifier (e.g., 'NO2').
    trend_direction : str
        Mann–Kendall trend direction: 'increasing', 'decreasing', or 'no trend'.
    p_value : float
        Mann–Kendall p-value. Values below ALPHA indicate statistical significance.
    sens_slope : float
        Sen's slope estimate — median rate of change per year.
    tau : float
        Kendall's tau correlation coefficient.
    r_squared : float
        OLS coefficient of determination (R²) for the linear fit.
    hist_mean_final : float
        Observed annual mean concentration in the final historical year (2024).
    proj_mean_2030 : float
        OLS-projected annual mean concentration in 2030.
    pct_change_2030 : float
        Percentage change from the 2024 observed value to the 2030 projection.
    significant : bool
        True if the Mann–Kendall p-value is below the significance threshold.
    """
    city:             str
    gas:              str
    trend_direction:  str
    p_value:          float
    sens_slope:       float
    tau:              float
    r_squared:        float
    hist_mean_final:  float
    proj_mean_2030:   float
    pct_change_2030:  float
    significant:      bool


# ==============================================================================
# Core Analysis Functions
# ==============================================================================

def load_city_dataframe(file_name: str, file_content: bytes) -> Optional[pd.DataFrame]:
    """
    Parse a city CSV file from raw bytes and perform basic validation.

    Parameters
    ----------
    file_name : str
        Original filename as uploaded (used only for logging).
    file_content : bytes
        Raw byte content of the uploaded CSV file.

    Returns
    -------
    pd.DataFrame or None
        Parsed and validated DataFrame, or None if the file is malformed.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as exc:
        logger.error("Failed to parse '%s': %s", file_name, exc)
        return None

    if "year" not in df.columns:
        logger.error(
            "'year' column is missing in '%s'. File will be skipped.", file_name
        )
        return None

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.dropna(subset=["year"], inplace=True)
    df["year"] = df["year"].astype(int)
    logger.info("Loaded '%s' — %d rows, %d columns.", file_name, *df.shape)
    return df


def aggregate_annual_means(
    df: pd.DataFrame,
    gas: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute annual mean concentrations for a given gas column.

    Parameters
    ----------
    df : pd.DataFrame
        City DataFrame containing at least a 'year' column and the gas column.
    gas : str
        Name of the gas column to aggregate.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        Sorted arrays (years, means). Returns None if the gas column is absent
        or if fewer than MIN_DATA_POINTS valid observations exist.
    """
    if gas not in df.columns:
        logger.warning("Column '%s' not found in dataset. Skipping.", gas)
        return None

    annual = (
        df.dropna(subset=[gas])
        .groupby("year")[gas]
        .mean()
        .sort_index()
    )

    if len(annual) < MIN_DATA_POINTS:
        logger.warning(
            "Only %d valid year(s) for '%s' — minimum required is %d. Skipping.",
            len(annual), gas, MIN_DATA_POINTS,
        )
        return None

    return annual.index.values.astype(float), annual.values.astype(float)


def fit_linear_model(
    years: np.ndarray,
    values: np.ndarray,
) -> Tuple[LinearRegression, float, np.ndarray, np.ndarray]:
    """
    Fit an OLS linear regression model and generate projections through 2030.

    Parameters
    ----------
    years : np.ndarray
        1-D array of historical year values.
    values : np.ndarray
        1-D array of corresponding annual mean concentrations.

    Returns
    -------
    model : LinearRegression
        Fitted scikit-learn LinearRegression instance.
    r2 : float
        Coefficient of determination (R²) on the historical data.
    years_extended : np.ndarray
        Combined historical and future year array (HIST_START_YEAR to PROJ_END_YEAR).
    preds_extended : np.ndarray
        Predicted values for years_extended.
    """
    X = years.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, values)

    r2 = r2_score(values, model.predict(X))

    future_years = np.arange(HIST_END_YEAR + 1, PROJ_END_YEAR + 1, dtype=float)
    years_extended = np.concatenate([years, future_years])
    preds_extended = model.predict(years_extended.reshape(-1, 1))

    return model, r2, years_extended, preds_extended


def run_mann_kendall(values: np.ndarray):
    """
    Apply the original Mann–Kendall monotonic trend test.

    Parameters
    ----------
    values : np.ndarray
        Time-ordered series of annual mean concentrations.

    Returns
    -------
    mk.MannKendallTest
        Named tuple with fields: trend, h, p, z, Tau, s, var_s, slope, intercept.
    """
    return mk.original_test(values)


def compute_trend_result(
    city: str,
    gas: str,
    years: np.ndarray,
    values: np.ndarray,
) -> TrendResult:
    """
    Orchestrate Mann–Kendall test, OLS regression, and projection for one gas.

    Parameters
    ----------
    city : str
        City label.
    gas : str
        Gas identifier.
    years : np.ndarray
        Annual year array (historical period).
    values : np.ndarray
        Corresponding annual mean concentration array.

    Returns
    -------
    TrendResult
        Fully populated result dataclass instance.
    """
    mk_result = run_mann_kendall(values)
    model, r2, _, preds_extended = fit_linear_model(years, values)

    proj_2030 = float(model.predict([[PROJ_END_YEAR]])[0])
    hist_final = float(values[-1])
    pct_change = ((proj_2030 - hist_final) / abs(hist_final)) * 100 if hist_final != 0 else float("nan")

    return TrendResult(
        city=city,
        gas=gas,
        trend_direction=mk_result.trend,
        p_value=round(mk_result.p, 6),
        sens_slope=round(mk_result.slope, 10),
        tau=round(mk_result.Tau, 4),
        r_squared=round(r2, 4),
        hist_mean_final=round(hist_final, 8),
        proj_mean_2030=round(proj_2030, 8),
        pct_change_2030=round(pct_change, 2),
        significant=(mk_result.p < ALPHA),
    )


# ==============================================================================
# Visualization
# ==============================================================================

def build_trend_figure(
    city: str,
    gas: str,
    years: np.ndarray,
    values: np.ndarray,
    years_extended: np.ndarray,
    preds_extended: np.ndarray,
    result: TrendResult,
) -> plt.Figure:
    """
    Construct a publication-quality trend and projection figure.

    The figure displays:
        • Blue markers connected by a solid line — historical annual means.
        • Red dashed line — OLS linear trend extended to 2030.
        • Shaded region — projection zone (post-2024).
        • Annotated projection value at 2030.
        • Statistical summary box (Mann–Kendall τ, Sen's slope, p-value, R²).

    Parameters
    ----------
    city : str
        City name for the plot title.
    gas : str
        Gas identifier used for axis labels and titles.
    years : np.ndarray
        Historical year values.
    values : np.ndarray
        Historical annual mean concentration values.
    years_extended : np.ndarray
        Full year array from HIST_START_YEAR to PROJ_END_YEAR.
    preds_extended : np.ndarray
        OLS-predicted values for years_extended.
    result : TrendResult
        Computed statistical metrics for annotation.

    Returns
    -------
    matplotlib.figure.Figure
        Fully configured figure object (caller is responsible for saving/showing).
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # ── Historical observed data ───────────────────────────────────────────────
    ax.plot(
        years, values,
        marker="o", markersize=8, linewidth=2.2,
        color="#1f6fb2", zorder=4,
        label=f"Observed Annual Mean ({HIST_START_YEAR}–{HIST_END_YEAR})",
    )

    # ── OLS linear trend (full range) ─────────────────────────────────────────
    ax.plot(
        years_extended, preds_extended,
        linestyle="--", linewidth=2, color="#d62728", zorder=3,
        label="OLS Linear Trend & 2030 Projection",
    )

    # ── Projection shading ────────────────────────────────────────────────────
    ax.axvspan(
        HIST_END_YEAR, PROJ_END_YEAR,
        alpha=0.08, color="#d62728", label="Projection Period (2025–2030)",
    )
    ax.axvline(
        x=HIST_END_YEAR, color="dimgray",
        linestyle=":", linewidth=1.4, zorder=2,
    )
    ax.text(
        HIST_END_YEAR + 0.1, ax.get_ylim()[0],
        "← Observed  |  Projected →",
        fontsize=8.5, color="dimgray", va="bottom",
    )

    # ── 2030 projection annotation ────────────────────────────────────────────
    proj_val = preds_extended[-1]
    ax.annotate(
        f"2030 Proj.\n{proj_val:.4e}",
        xy=(PROJ_END_YEAR, proj_val),
        xytext=(-60, 20), textcoords="offset points",
        fontsize=9, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    # ── Statistical summary box ───────────────────────────────────────────────
    sig_flag = "✓ Significant" if result.significant else "✗ Not Significant"
    stats_text = (
        f"Mann–Kendall Test\n"
        f"  Trend: {result.trend_direction.title()}\n"
        f"  τ = {result.tau:+.4f}\n"
        f"  Sen's Slope = {result.sens_slope:.3e}\n"
        f"  p-value = {result.p_value:.4f}  ({sig_flag})\n"
        f"OLS R² = {result.r_squared:.4f}"
    )
    ax.text(
        0.02, 0.97, stats_text,
        transform=ax.transAxes, fontsize=8.8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.75", alpha=0.9),
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_title(
        f"{GAS_TITLES[gas]} Atmospheric Concentration — {city}\n"
        f"Trend Analysis and 2030 Projection",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(GAS_LABELS.get(gas, f"{gas} Concentration"), fontsize=11)
    ax.set_xlim(HIST_START_YEAR - 0.5, PROJ_END_YEAR + 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))

    ax.legend(
        loc="lower right", fontsize=9,
        framealpha=0.9, edgecolor="0.8",
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    return fig


# ==============================================================================
# Reporting
# ==============================================================================

def build_summary_dataframe(results: List[TrendResult]) -> pd.DataFrame:
    """
    Convert a list of TrendResult instances into a structured summary DataFrame.

    Parameters
    ----------
    results : list of TrendResult
        Computed results for all city–gas pairs.

    Returns
    -------
    pd.DataFrame
        Formatted summary table with human-readable column names.
    """
    records = [
        {
            "City":                     r.city,
            "Gas":                      r.gas,
            "MK Trend Direction":       r.trend_direction.title(),
            "Kendall τ":                r.tau,
            "Sen's Slope":              r.sens_slope,
            "p-value":                  r.p_value,
            "Significant (α=0.05)":    "Yes" if r.significant else "No",
            "OLS R²":                   r.r_squared,
            f"Observed Mean ({HIST_END_YEAR})": r.hist_mean_final,
            f"Projected Mean (2030)":   r.proj_mean_2030,
            "% Change (2024→2030)":     r.pct_change_2030,
        }
        for r in results
    ]
    return pd.DataFrame(records)


def print_section_header(title: str, width: int = 62) -> None:
    """Print a formatted section header to stdout."""
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def run_analysis() -> None:
    """
    Execute the complete Sentinel-5P trend analysis pipeline.

    Workflow
    --------
    1. Prompt the user to upload city CSV files (Colab) or scan the working
       directory (local fallback).
    2. For each uploaded file × each target gas:
        a. Validate and aggregate the data.
        b. Run Mann–Kendall test and OLS regression.
        c. Compute projections.
        d. Generate and save a publication-quality trend figure.
    3. Consolidate all results into a summary DataFrame.
    4. Display and export the summary CSV.
    """
    print_section_header(
        "Sentinel-5P / TROPOMI Atmospheric Trend Analysis"
    )
    logger.info(
        "Study period: %d–%d | Projection horizon: %d",
        HIST_START_YEAR, HIST_END_YEAR, PROJ_END_YEAR,
    )
    logger.info("Target gases: %s", ", ".join(TARGET_GASES))

    # ── Step 1: File ingestion ─────────────────────────────────────────────────
    if IN_COLAB:
        print(
            "\nPlease upload your city CSV files "
            f"(e.g., Dammam_AllGases_{HIST_START_YEAR}_{HIST_END_YEAR}.csv):\n"
        )
        uploaded: Dict[str, bytes] = colab_files.upload()
        if not uploaded:
            logger.error("No files were uploaded. Aborting.")
            return
    else:
        # Local fallback: read any CSV in the working directory.
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

    # ── Step 2: Analysis loop ──────────────────────────────────────────────────
    all_results: List[TrendResult] = []
    total_files = len(uploaded)

    for file_idx, (file_name, file_content) in enumerate(uploaded.items(), start=1):
        city_name = file_name.split("_")[0] if "_" in file_name else os.path.splitext(file_name)[0]
        print_section_header(
            f"[{file_idx}/{total_files}]  Processing: {city_name}"
        )

        df = load_city_dataframe(file_name, file_content)
        if df is None:
            continue

        for gas in TARGET_GASES:
            logger.info("  ─ Analysing %s | %s …", city_name, gas)

            result_arrays = aggregate_annual_means(df, gas)
            if result_arrays is None:
                continue

            years, values = result_arrays

            # Trend analysis + projection
            result = compute_trend_result(city_name, gas, years, values)

            # OLS data for plotting
            _, _, years_ext, preds_ext = fit_linear_model(years, values)

            # Build and save figure
            fig = build_trend_figure(
                city=city_name,
                gas=gas,
                years=years,
                values=values,
                years_extended=years_ext,
                preds_extended=preds_ext,
                result=result,
            )

            plot_path = os.path.join(OUTPUT_DIR, f"{city_name}_{gas}_Trend_2030.png")
            fig.savefig(plot_path)
            logger.info("    Saved: %s", plot_path)

            plt.show()
            plt.close(fig)

            # Log key statistics
            sig_label = "SIGNIFICANT" if result.significant else "not significant"
            logger.info(
                "    MK trend: %-12s | τ = %+.4f | p = %.4f (%s) | "
                "Sen's slope = %.3e | R² = %.4f | Δ2030 = %+.2f%%",
                result.trend_direction, result.tau, result.p_value, sig_label,
                result.sens_slope, result.r_squared, result.pct_change_2030,
            )

            all_results.append(result)

    # ── Step 3: Summary table ──────────────────────────────────────────────────
    if not all_results:
        logger.error("No valid results generated. Check your data format and column names.")
        return

    summary_df = build_summary_dataframe(all_results)

    print_section_header("Final Summary: Trend Analysis and 2030 Projections")
    display(summary_df)

    summary_df.to_csv(OUTPUT_CSV, index=False, float_format="%.8g")
    logger.info("Summary saved to '%s'.", OUTPUT_CSV)

    if IN_COLAB:
        logger.info("Initiating download of '%s' …", OUTPUT_CSV)
        colab_files.download(OUTPUT_CSV)

    print(
        f"\n✓  Analysis complete.\n"
        f"   Summary CSV : {OUTPUT_CSV}\n"
        f"   Plot folder : {OUTPUT_DIR}/  ({len(all_results)} figures)\n"
    )


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    run_analysis()
