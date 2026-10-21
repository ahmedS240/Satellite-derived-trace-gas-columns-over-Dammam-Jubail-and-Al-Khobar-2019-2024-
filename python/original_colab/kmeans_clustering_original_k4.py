"""
==============================================================================
Spatial Hotspot K-Means Clustering Analysis
Linking Atmospheric Gas Concentrations with Human Activity Proxies
==============================================================================
Author      : Ahmed Saad Alnashwan
Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
              MSc Environmental Science & Engineering
Supervisor  : Dr. Syed Rahman
Study Area  : Dammam, Jubail, Al-Khobar — Eastern Province, Saudi Arabia
Method      : K-Means Clustering (k=4) with StandardScaler normalisation
Proxies     : Nighttime Lights (NTL), Population Density (POP)
Gases       : NO2 (tropospheric), CO, CH4
Environment : Google Colab
==============================================================================

Description
-----------
This script performs unsupervised K-Means clustering on spatial hotspot grid
cells derived from Sentinel-5P / TROPOMI and VIIRS/GHSL data exported from
Google Earth Engine. For each city the script:

    1. Locates and ingests the city hotspot CSV file from uploaded files.
       Accepts any of these naming formats:
         Hotspot_Dammam_FINAL.csv      (GEE export format)
         Dammam_Hotspot_*.csv          (alternative format)
         *dammam*.csv                  (any file containing city name)
    2. Validates, selects, and standardises clustering features.
    3. Generates a dual-axis Elbow + Silhouette Plot (k=2-9) to validate k=4.
    4. Fits a K-Means model (k=4) and computes quality metrics.
    5. Auto-labels clusters by NO2 rank for interpretability.
    6. Produces publication-quality scatter plots (300 DPI).
    7. Exports a GIS-ready clustered CSV per city (auto-downloaded).
    8. Exports a cross-city quality summary CSV (auto-downloaded).

Academic Notes on k=4 Selection
---------------------------------
k=4 selected based on:
    1. Elbow Plot inflection point (WCSS vs k).
    2. Peak or near-peak Silhouette Score.
    3. Domain knowledge: 4 interpretable urban zones in Eastern Province:
         Cluster A: Industrial core   (high gas, high NTL)
         Cluster B: Urban residential (moderate gas, moderate NTL)
         Cluster C: Peri-urban        (low-moderate gas, low NTL)
         Cluster D: Rural/background  (low gas, low NTL)

Dependencies
------------
    pip install pandas numpy scikit-learn matplotlib seaborn

Usage (Google Colab)
--------------------
    Upload hotspot CSV files exported from GEE:
        Hotspot_Dammam_FINAL.csv
        Hotspot_Jubail_FINAL.csv
        Hotspot_Khobar_FINAL.csv  (or AlKhobar / Al_Khobar)

    Required CSV columns (from GEE reduceRegions export):
        - NO2   : NO2 tropospheric column density [mol m-2]
        - CO    : CO column density               [mol m-2]
        - CH4   : CH4 dry air mixing ratio        [ppb]
        - NTL   : Nighttime Lights radiance       [nW cm-2 sr-1]
        - POP   : Population count (GHSL 2020)
        - Longitude, Latitude : grid cell centroid coordinates

Output
------
    - <City>_Final_Clustered_MapData.csv     (auto-downloaded, GIS-ready)
    - PhaseC_Clustering_Quality_Summary.csv  (auto-downloaded)
    - Clustering_Analysis_Results/<City>_Elbow_Silhouette.png
    - Clustering_Analysis_Results/<City>_Clustering_Scatter.png
==============================================================================
"""

# Standard Library
import io, os, logging, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Third-Party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from IPython.display import display

# Colab-Specific Import
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    warnings.warn(
        "google.colab not found. Place CSV files in the working directory.",
        RuntimeWarning, stacklevel=2,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Constants & Configuration
# ==============================================================================

#: City search keywords — each entry is a list of acceptable name variants.
#: The file loader matches any filename containing ANY of the listed variants.
#: Add variants here if your GEE export uses a different naming convention.
CITY_SEARCH: Dict[str, List[str]] = {
    "Dammam": ["dammam"],
    "Jubail":  ["jubail", "al_jubail", "aljubail"],
    "Khobar":  ["khobar", "al_khobar", "alkhobar", "alkhobarf"],
}

#: Feature columns for clustering — must match GEE export column names.
CLUSTER_FEATURES: List[str] = ["NO2", "CO", "CH4", "NTL", "POP"]

#: Coordinate columns to preserve in GIS output (not used for clustering).
COORD_COLS: List[str] = ["Longitude", "Latitude", "Grid_ID"]

#: k=4 — justified by Elbow Plot, Silhouette Score, and domain knowledge.
N_CLUSTERS:    int   = 4
RANDOM_STATE:  int   = 42
N_INIT:        int   = 10
MIN_CELLS:     int   = N_CLUSTERS
ELBOW_K_RANGE: range = range(2, 10)
OUTPUT_DIR:    str   = "Clustering_Analysis_Results"
CLUSTER_PALETTE: str = "viridis"

#: Ranking feature for cluster label assignment.
#: NO2 for urban/traffic focus; CH4 for industrial/flaring focus (e.g., Jubail).
LABEL_RANK_COL: str = "NO2"

CLUSTER_LABEL_TIERS: List[str] = [
    "Low Pollution / Low Activity",
    "Moderate Pollution / Mixed Activity",
    "Elevated Pollution / High Activity",
    "High Pollution / Very High Activity",
]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.titlepad":     14,
    "axes.labelpad":     8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.8",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# ==============================================================================
# Data Class
# ==============================================================================

@dataclass
class ClusteringResult:
    """
    Container for all computed outputs of a single city clustering run.

    Attributes
    ----------
    city            : Study city name.
    n_cells         : Valid grid cells used (after NaN removal).
    features_used   : Feature columns present and used for clustering.
    silhouette      : Silhouette score [-1,1]; higher = better separation.
    inertia         : Within-cluster sum of squares (standardised space).
    cluster_profile : Mean ORIGINAL-SCALE feature values per cluster
                      plus cell count. Include full table in thesis appendix.
    clustered_df    : Full grid-cell DataFrame with Cluster and Cluster_Label.
    label_rank_col  : Feature used for cluster ranking (for transparency).
    """
    city:            str
    n_cells:         int
    features_used:   List[str]
    silhouette:      float
    inertia:         float
    cluster_profile: pd.DataFrame
    clustered_df:    pd.DataFrame
    label_rank_col:  str


# ==============================================================================
# Core Functions
# ==============================================================================

def find_city_file(
    city: str,
    file_list: List[str],
) -> Optional[str]:
    """
    Find the uploaded file for a given city using flexible name matching.

    Accepts multiple naming formats:
        Hotspot_Dammam_FINAL.csv   (GEE export)
        Dammam_Hotspot_2019.csv    (alternative)
        dammam_data.csv            (any file with city name)

    Parameters
    ----------
    city : str
        City key from CITY_SEARCH (e.g., 'Dammam').
    file_list : list of str
        All uploaded filenames.

    Returns
    -------
    str or None
        Matched filename, or None if not found.
    """
    variants = CITY_SEARCH.get(city, [city.lower()])
    matches  = [
        f for f in file_list
        if any(v in f.lower() for v in variants)
    ]
    if not matches:
        logger.warning(
            "%s: No file found. Searched for variants: %s", city, variants
        )
        return None
    if len(matches) > 1:
        logger.warning(
            "%s: Multiple files matched: %s. Using: %s", city, matches, matches[0]
        )
    return matches[0]


def load_city_file(
    city: str,
    file_list: List[str],
    uploaded: Dict[str, bytes],
) -> Optional[pd.DataFrame]:
    """
    Locate, parse, and return the hotspot DataFrame for a city.

    Strips column names of whitespace. Renames common GEE column
    name variants to the expected feature names (e.g., 'mean' prefix
    columns from reduceRegions output).

    Parameters
    ----------
    city      : City key.
    file_list : All uploaded filenames.
    uploaded  : filename -> bytes mapping.

    Returns
    -------
    pd.DataFrame or None
    """
    fname = find_city_file(city, file_list)
    if fname is None:
        return None
    try:
        df = pd.read_csv(io.BytesIO(uploaded[fname]))
    except Exception as exc:
        logger.error("Failed to parse '%s': %s", fname, exc)
        return None

    df.columns = df.columns.str.strip()

    # GEE reduceRegions sometimes prefixes band names with 'mean' -- strip it
    # e.g., 'NO2' stays 'NO2'; 'meanNO2' or 'NO2_mean' also accepted
    rename_map = {}
    for feat in CLUSTER_FEATURES:
        for col in df.columns:
            if col == feat:
                break
            if col.lower().replace("_", "").replace(" ", "") == feat.lower():
                rename_map[col] = feat
                break
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        logger.info("%s: Renamed columns: %s", city, rename_map)

    logger.info("Loaded '%s' -- %d rows, %d columns: %s",
                fname, len(df), len(df.columns), list(df.columns))
    return df


def prepare_clustering_data(
    df: pd.DataFrame, city: str
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray, StandardScaler]]:
    """
    Select, validate, clean, and standardise features for K-Means.

    Preserves coordinate columns (Longitude, Latitude, Grid_ID) in a
    separate DataFrame so they can be re-joined to the clustered output
    for GIS mapping.

    StandardScaler is critical: CH4 (~1900 ppb) vs NO2 (~0.0001 mol m-2)
    differ by ~7 orders of magnitude. Without scaling, CH4 dominates
    Euclidean distance and makes all other features irrelevant.

    Returns
    -------
    clean_df   : Feature-only DataFrame (scaled input rows).
    coords_df  : Coordinate columns aligned with clean_df rows.
    features   : List of feature column names used.
    scaled     : StandardScaler-transformed feature matrix.
    scaler     : Fitted StandardScaler instance.
    """
    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    missing   = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        logger.warning("%s: Features not found (excluded): %s", city, missing)
    if len(available) < 2:
        logger.error("%s: Only %d feature(s) -- need >= 2.", city, len(available))
        return None

    # Preserve coordinate columns for GIS output
    coord_cols_present = [c for c in COORD_COLS if c in df.columns]
    all_cols = available + coord_cols_present
    full_df  = df[all_cols].dropna(subset=available).copy()
    n_dropped = len(df) - len(full_df)
    if n_dropped:
        logger.info("%s: Dropped %d rows with NaN in feature columns.", city, n_dropped)

    if len(full_df) < MIN_CELLS:
        logger.error("%s: Only %d valid rows -- need >= %d.", city, len(full_df), MIN_CELLS)
        return None

    clean_df  = full_df[available].copy()
    coords_df = full_df[coord_cols_present].copy() if coord_cols_present else pd.DataFrame()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df)
    logger.info("%s: %d grid cells | Features: %s", city, len(clean_df), available)
    return clean_df, coords_df, available, scaled, scaler


def run_kmeans(
    scaled: np.ndarray,
    clean_df: pd.DataFrame,
    features_used: List[str],
) -> Tuple[np.ndarray, pd.DataFrame, float, float]:
    """
    Fit K-Means (k=N_CLUSTERS) and return labels, profile, and metrics.

    Profile computed in original (unscaled) space for interpretability.
    Silhouette computed on standardised space (standard practice).
    """
    kmeans    = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    labels    = kmeans.fit_predict(scaled)
    inertia   = kmeans.inertia_
    sil_score = silhouette_score(scaled, labels)

    profile_df = clean_df.copy()
    profile_df["Cluster"] = labels
    profile = profile_df.groupby("Cluster")[features_used].mean().round(6)
    profile["Cell Count"] = profile_df.groupby("Cluster").size()
    return labels, profile, sil_score, inertia


def assign_cluster_labels(profile: pd.DataFrame) -> Dict[int, str]:
    """
    Auto-label clusters by LABEL_RANK_COL rank (default: NO2).

    Labels validated against full centroid profile.
    Change LABEL_RANK_COL to 'CH4' for Jubail if industrial signal dominates.
    """
    rank_col = LABEL_RANK_COL if LABEL_RANK_COL in profile.columns else profile.columns[0]
    if rank_col != LABEL_RANK_COL:
        logger.warning("'%s' absent -- ranking by '%s'.", LABEL_RANK_COL, rank_col)
    ranked = profile[rank_col].rank(method="first").astype(int) - 1
    return {cluster: CLUSTER_LABEL_TIERS[rank] for cluster, rank in ranked.items()}


# ==============================================================================
# Visualization
# ==============================================================================

def build_elbow_silhouette_figure(city: str, scaled: np.ndarray) -> plt.Figure:
    """
    Dual-axis Elbow + Silhouette figure to justify k=4.

    Left (red)  -- WCSS inertia: look for elbow point.
    Right (blue) -- Silhouette score: look for peak.
    Dashed line marks selected k=N_CLUSTERS.
    Both metrics together provide stronger k justification than elbow alone.
    """
    k_list     = list(ELBOW_K_RANGE)
    inertias   = []
    sil_scores = []
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(scaled, km.labels_))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(k_list, inertias, marker="o", linewidth=2.2, color="#d62728",
             markersize=7, label="WCSS / Inertia (Elbow)")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=11, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax2 = ax1.twinx()
    ax2.plot(k_list, sil_scores, marker="s", linewidth=2.2, color="#1f6fb2",
             markersize=7, label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score (higher = better)", fontsize=11, color="#1f6fb2")
    ax2.tick_params(axis="y", labelcolor="#1f6fb2")

    ax1.axvline(x=N_CLUSTERS, color="dimgray", linestyle="--", linewidth=1.5,
                label=f"Selected k = {N_CLUSTERS}")

    best_k = k_list[int(np.argmax(sil_scores))]
    ax2.annotate(
        f"Best silhouette\nk={best_k} ({max(sil_scores):.4f})",
        xy=(best_k, max(sil_scores)),
        xytext=(10, -28), textcoords="offset points",
        fontsize=8.5, color="#1f6fb2",
        arrowprops=dict(arrowstyle="->", color="#1f6fb2", lw=1.1),
    )

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
               fontsize=8.5, framealpha=0.9, edgecolor="0.8")
    ax1.set_title(
        f"K-Means Evaluation: Elbow & Silhouette -- {city}\n"
        f"(Selected k={N_CLUSTERS} | Ranked by {LABEL_RANK_COL})",
        fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig


def build_clustering_figure(
    city: str,
    data: pd.DataFrame,
    result: ClusteringResult,
) -> plt.Figure:
    """
    Build 2-panel scatter plot: NO2 vs NTL and NO2 vs POP.

    Each point = one 5km grid cell, coloured by cluster assignment.
    Footer shows quality metrics for thesis reporting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Spatial K-Means Clustering (k={N_CLUSTERS}) -- {city}\n"
        f"Ranked by '{result.label_rank_col}' | "
        f"Silhouette = {result.silhouette:.4f} | "
        f"n = {result.n_cells} grid cells",
        fontsize=13, fontweight="bold", y=1.02,
    )

    common_kw = dict(
        data=data, hue="Cluster_Label", palette=CLUSTER_PALETTE,
        s=80, alpha=0.82, edgecolors="0.3", linewidth=0.3,
    )

    panel_config = [
        ("NTL", "NTL Radiance (nW cm-2 sr-1)",
         "NO2 vs Nighttime Lights (Economic Proxy)", axes[0]),
        ("POP", "Population Count (GHSL 2020)",
         "NO2 vs Population Density (Demographic Proxy)", axes[1]),
    ]

    for xcol, xlabel, title, ax in panel_config:
        if xcol in data.columns and "NO2" in data.columns:
            sns.scatterplot(ax=ax, x=xcol, y="NO2", **common_kw)
            ax.set_title(f"{city}: {title}", fontsize=11, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("NO2 Tropospheric Column (mol m-2)", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"{xcol} or NO2 data unavailable",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.7)

    axes[0].get_legend().remove()
    if axes[1].get_legend():
        axes[1].legend(title="Cluster Type", title_fontsize=9, fontsize=8.5,
                       loc="upper left", framealpha=0.9, edgecolor="0.8")

    footer = (
        f"K-Means | k={N_CLUSTERS} | "
        f"Grid Cells={result.n_cells} | "
        f"Silhouette={result.silhouette:.4f} | "
        f"Inertia={result.inertia:.2f} | "
        f"Features: {', '.join(result.features_used)}"
    )
    fig.text(0.5, -0.03, footer, ha="center", fontsize=8.5, color="dimgray",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="0.8"))
    fig.tight_layout()
    return fig


# ==============================================================================
# Reporting
# ==============================================================================

def build_quality_summary(results: List[ClusteringResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "City":             r.city,
        "Grid Cells":       r.n_cells,
        "Features Used":    ", ".join(r.features_used),
        "k":                N_CLUSTERS,
        "Label Rank Col":   r.label_rank_col,
        "Silhouette Score": r.silhouette,
        "Inertia":          r.inertia,
    } for r in results])


def print_section_header(title: str, width: int = 62) -> None:
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_analysis() -> None:
    """Execute the complete spatial hotspot K-Means clustering pipeline."""
    print_section_header("Spatial Hotspot K-Means Clustering Analysis")
    logger.info("k=%d | Label rank: %s | Features: %s",
                N_CLUSTERS, LABEL_RANK_COL, CLUSTER_FEATURES)

    print("\nAccepted filenames (any of these formats):")
    print("  Hotspot_Dammam_FINAL.csv  /  Hotspot_Jubail_FINAL.csv  /  Hotspot_Khobar_FINAL.csv")
    print("  Dammam_Hotspot.csv        /  Jubail_Hotspot.csv        /  AlKhobar_Hotspot.csv\n")

    # Step 1: File Ingestion
    if IN_COLAB:
        print("Upload your three hotspot CSV files now:\n")
        uploaded: Dict[str, bytes] = colab_files.upload()
        if not uploaded:
            logger.error("No files uploaded. Aborting.")
            return
    else:
        local_csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
        if not local_csvs:
            logger.error("No CSV files found. Aborting.")
            return
        uploaded = {fname: open(fname, "rb").read() for fname in local_csvs}

    file_list = list(uploaded.keys())
    logger.info("Received %d file(s): %s", len(file_list), ", ".join(file_list))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results: List[ClusteringResult] = []

    # Step 2: City Analysis Loop
    for city in CITY_SEARCH.keys():
        print_section_header(f"Processing: {city}")

        df = load_city_file(city, file_list, uploaded)
        if df is None:
            continue

        prep = prepare_clustering_data(df, city)
        if prep is None:
            continue
        clean_df, coords_df, features_used, scaled, scaler = prep

        # Elbow + Silhouette dual-axis figure
        logger.info("Generating Elbow + Silhouette Plot for %s ...", city)
        elbow_fig  = build_elbow_silhouette_figure(city, scaled)
        elbow_path = os.path.join(OUTPUT_DIR, f"{city}_Elbow_Silhouette.png")
        elbow_fig.savefig(elbow_path)
        plt.show(); plt.close(elbow_fig)
        logger.info("Saved: %s", elbow_path)

        # K-Means
        logger.info("Running K-Means (k=%d) for %s ...", N_CLUSTERS, city)
        labels, profile, silhouette, inertia = run_kmeans(scaled, clean_df, features_used)
        label_map = assign_cluster_labels(profile)
        clean_df["Cluster"]       = labels
        clean_df["Cluster_Label"] = clean_df["Cluster"].map(label_map)

        result = ClusteringResult(
            city=city, n_cells=len(clean_df), features_used=features_used,
            silhouette=round(silhouette, 4), inertia=round(inertia, 4),
            cluster_profile=profile, clustered_df=clean_df,
            label_rank_col=LABEL_RANK_COL,
        )
        all_results.append(result)

        print(f"\n  Cluster Centroid Profiles -- {city} (original scale):")
        display(profile)
        logger.info("Silhouette=%.4f | Inertia=%.2f | Cells=%d",
                    silhouette, inertia, len(clean_df))

        # Scatter plots
        scatter_fig  = build_clustering_figure(city, clean_df, result)
        scatter_path = os.path.join(OUTPUT_DIR, f"{city}_Clustering_Scatter.png")
        scatter_fig.savefig(scatter_path)
        plt.show(); plt.close(scatter_fig)
        logger.info("Saved: %s", scatter_path)

        # Export GIS-ready CSV (re-attach coordinates)
        if not coords_df.empty:
            out_df = pd.concat([coords_df.reset_index(drop=True),
                                clean_df.reset_index(drop=True)], axis=1)
        else:
            out_df = clean_df
        out_csv = f"{city}_Final_Clustered_MapData.csv"
        out_df.to_csv(out_csv, index=False)
        logger.info("GIS CSV saved: %s", out_csv)
        if IN_COLAB:
            colab_files.download(out_csv)

    # Step 3: Cross-City Quality Summary
    if not all_results:
        logger.error("No results. Check filenames contain city name and CSV extension.")
        return

    summary_df = build_quality_summary(all_results)
    print_section_header("Cross-City Clustering Quality Summary")
    display(summary_df)

    summary_csv = "PhaseC_Clustering_Quality_Summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Quality summary: %s", summary_csv)
    if IN_COLAB:
        colab_files.download(summary_csv)

    print(
        f"\nDone.\n"
        f"   Plots      : {OUTPUT_DIR}/\n"
        f"   GIS CSVs   : one per city (Longitude, Latitude, Cluster, Cluster_Label)\n"
        f"   Summary    : {summary_csv}\n"
    )


if __name__ == "__main__":
    run_analysis()
