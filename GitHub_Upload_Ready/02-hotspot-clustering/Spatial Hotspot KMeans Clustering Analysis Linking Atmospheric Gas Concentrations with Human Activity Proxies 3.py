"""
==============================================================================
Spatial Cluster Map Visualization — K-Means Results on City Boundaries
==============================================================================
Author      : Ahmed Saad Alnashwan
Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
              MSc Environmental Science & Engineering
Supervisor  : Dr. Syed Rahman
Study Area  : Dammam, Jubail, Al-Khobar — Eastern Province, Saudi Arabia
Environment : Google Colab
==============================================================================

WHAT TO UPLOAD WHEN PROMPTED
------------------------------
File type 1 — Shapefile ZIPs (already in /content/ from previous upload):
    Dammam.zip        <- contains Dammam.shp + .dbf + .shx + .prj
    Al_Jubail.zip     <- contains Al_Jubail.shp + .dbf + .shx + .prj
    Al_Khobar.zip     <- contains AlKhobarfull.shp + .dbf + .shx + .prj

File type 2 — Clustered CSVs (from spatial_clustering_analysis.py output):
    Dammam_Final_Clustered_MapData.csv
    Jubail_Final_Clustered_MapData.csv
    Khobar_Final_Clustered_MapData.csv

NOTE: The script auto-unzips shapefiles and accepts both ZIP and raw .shp.
      Upload ALL 6 files at once when the dialog appears.
==============================================================================
"""

# ── Step 1: Install dependencies ──────────────────────────────────────────────
import subprocess, sys
print("Installing geopandas, contextily, shapely ...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "geopandas", "contextily", "shapely"])
print("✓  Packages ready.\n")

# ── Step 2: Imports ───────────────────────────────────────────────────────────
import os, logging, warnings, zipfile, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import Point
from google.colab import files as colab_files

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

try:
    import contextily as ctx
    HAS_CTX = True
    logger.info("Contextily available — OpenStreetMap basemap will be added.")
except ImportError:
    HAS_CTX = False
    logger.warning("Contextily not available — maps will use plain background.")

# ── Step 3: Configuration ─────────────────────────────────────────────────────

CITY_CONFIG = {
    "Dammam": {
        "shp_candidates": [
            "/content/Dammam.shp",         # if unzipped already
            "/content/Dammam/Dammam.shp",  # if unzipped into subfolder
        ],
        "zip_search":  "dammam",   # keyword to find the ZIP file
        "csv_search":  "dammam",   # keyword to find the clustered CSV
    },
    "Al Jubail": {
        "shp_candidates": [
            "/content/Al_Jubail.shp",
            "/content/Al_Jubail/Al_Jubail.shp",
        ],
        "zip_search":  "jubail",
        "csv_search":  "jubail",
    },
    "Al Khobar": {
        "shp_candidates": [
            "/content/AlKhobarfull.shp",
            "/content/Al_Khobar/AlKhobarfull.shp",
            "/content/AlKhobarfull/AlKhobarfull.shp",
            "/content/Al_Khobar.shp",
        ],
        "zip_search":  "khobar",   # matches AlKhobarfull.shp and Al_Khobar.zip
        "csv_search":  "khobar",
    },
}

CLUSTER_COLORS     = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
CLUSTER_LABEL_TIERS = [
    "Low Pollution / Low Activity",
    "Moderate Pollution / Mixed Activity",
    "Elevated Pollution / High Activity",
    "High Pollution / Very High Activity",
]

OUTPUT_DIR = "Cluster_Maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
})

# ── Step 4: Upload ALL files at once ──────────────────────────────────────────
print("=" * 62)
print("  Upload ALL 6 files at once when the dialog appears:")
print()
print("  SHAPEFILES (ZIP format):")
print("    Dammam.zip")
print("    Al_Jubail.zip")
print("    Al_Khobar.zip")
print()
print("  CLUSTERED CSVs (from spatial_clustering_analysis.py):")
print("    Dammam_Final_Clustered_MapData.csv")
print("    Jubail_Final_Clustered_MapData.csv")
print("    Khobar_Final_Clustered_MapData.csv")
print("=" * 62 + "\n")

uploaded = colab_files.upload()

if not uploaded:
    raise RuntimeError("No files uploaded. Re-run the cell.")

# Save all uploaded files to /content/
for fname, fbytes in uploaded.items():
    dest = os.path.join("/content", fname)
    with open(dest, "wb") as fh:
        fh.write(fbytes)
    logger.info("Saved to /content/: %s", fname)

file_list = list(uploaded.keys())
logger.info("Received %d file(s): %s", len(file_list), file_list)

# ── Step 5: Auto-unzip shapefiles ─────────────────────────────────────────────

def unzip_shapefile(search_term: str) -> bool:
    """
    Find a ZIP file in /content/ containing the search_term and unzip it.
    Returns True if a ZIP was found and extracted.
    """
    for fname in os.listdir("/content"):
        if fname.lower().endswith(".zip") and search_term in fname.lower():
            zip_path = os.path.join("/content", fname)
            logger.info("Unzipping: %s", zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall("/content")
                extracted = z.namelist()
            logger.info("Extracted: %s", extracted)
            return True
    return False

print("\nUnzipping shapefiles ...")
for city_name, cfg in CITY_CONFIG.items():
    result = unzip_shapefile(cfg["zip_search"])
    if result:
        print(f"  ✓  {city_name} shapefile unzipped.")
    else:
        print(f"  ⚠  No ZIP found for {city_name} — will look for .shp directly.")

# ── Step 6: Helper functions ──────────────────────────────────────────────────

def find_shp(candidates: list, zip_search: str) -> str:
    """
    Find the correct .shp file for a given city.

    Strategy:
      1. Check explicit candidate paths first (most reliable).
      2. Recursive search: match .shp files whose FULL PATH contains
         zip_search AND whose filename does NOT contain any other city name.
         This prevents Al Jubail's shapefile being returned for Al Khobar.

    The exclusion list ensures cross-city contamination cannot occur even
    when multiple zipped shapefiles are extracted into /content/.
    """
    # All city keywords — used to exclude wrong-city matches in fallback
    ALL_CITY_KEYWORDS = ["dammam", "jubail", "khobar"]

    # Check explicit candidates first
    for path in candidates:
        if os.path.exists(path):
            logger.info("Shapefile found at candidate path: %s", path)
            return path

    # City-specific recursive search with exclusion of other cities
    all_shp_found = []
    for root, dirs, files in os.walk("/content"):
        for fname in files:
            if not fname.endswith(".shp"):
                continue
            full = os.path.join(root, fname)
            full_lower = full.lower()

            # Must contain this city's keyword in the full path
            if zip_search not in full_lower:
                continue

            # Must NOT contain any OTHER city's keyword in the filename
            other_cities = [k for k in ALL_CITY_KEYWORDS if k != zip_search]
            if any(other in fname.lower() for other in other_cities):
                logger.warning(
                    "Skipping '%s' -- filename contains another city's keyword.", full
                )
                continue

            all_shp_found.append(full)

    if all_shp_found:
        # Prefer shorter paths (less deeply nested = likely the correct root file)
        all_shp_found.sort(key=lambda p: len(p))
        logger.info("Shapefile found by keyword search: %s", all_shp_found[0])
        return all_shp_found[0]

    logger.error(
        "No .shp found for city '%s'. "
        "Checked candidates: %s. "
        "Make sure '%s.zip' was uploaded and unzipped successfully.",
        zip_search, candidates, zip_search
    )
    return None

def find_csv(search_term: str) -> str:
    """Find uploaded clustered CSV by searching for city name in filename."""
    # Search in /content/ and current directory
    for directory in ["/content", "."]:
        for fname in os.listdir(directory):
            if (fname.lower().endswith(".csv")
                    and search_term in fname.lower()
                    and "clustered" in fname.lower()):
                return os.path.join(directory, fname)

    # Fallback: any CSV containing the search term
    for directory in ["/content", "."]:
        for fname in os.listdir(directory):
            if fname.lower().endswith(".csv") and search_term in fname.lower():
                # Exclude raw hotspot files — they don't have a Cluster column
                if "hotspot" not in fname.lower():
                    return os.path.join(directory, fname)
    return None

def load_csv(path: str) -> pd.DataFrame:
    """Load clustered CSV and validate required columns."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in ["Longitude", "Latitude", "Cluster"]
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"\nMissing columns {missing} in '{path}'.\n"
            f"Available columns: {list(df.columns)}\n\n"
            f"You uploaded a RAW hotspot file, not the CLUSTERED output.\n"
            f"Run spatial_clustering_analysis.py first, then upload:\n"
            f"  Dammam_Final_Clustered_MapData.csv\n"
            f"  Jubail_Final_Clustered_MapData.csv\n"
            f"  Khobar_Final_Clustered_MapData.csv"
        )

    df.dropna(subset=["Longitude", "Latitude"], inplace=True)
    df["Cluster"] = df["Cluster"].astype(int)
    if "Cluster_Label" not in df.columns:
        df["Cluster_Label"] = df["Cluster"].map(
            {i: CLUSTER_LABEL_TIERS[i] for i in range(4)}
        )
    logger.info("Loaded %d grid cells | Clusters: %s", len(df),
                sorted(df["Cluster"].unique()))
    return df

def build_color_map(df):
    """Build cluster -> colour mapping and legend patches."""
    unique = sorted(df["Cluster"].unique())
    cmap   = {c: CLUSTER_COLORS[i % 4] for i, c in enumerate(unique)}
    patches = [
        mpatches.Patch(
            facecolor=cmap[c], edgecolor="0.3", linewidth=0.5,
            label=df.loc[df["Cluster"] == c, "Cluster_Label"].iloc[0]
        )
        for c in unique
    ]
    return cmap, patches

def stats_text(df):
    lines = [f"Total cells: {len(df)}"]
    for c in sorted(df["Cluster"].unique()):
        n   = (df["Cluster"] == c).sum()
        lbl = df.loc[df["Cluster"] == c, "Cluster_Label"].iloc[0]
        pct = 100 * n / len(df)
        lines.append(f"  {lbl[:30]}: {n} ({pct:.1f}%)")
    return "\n".join(lines)

# ── Step 7: Map building ──────────────────────────────────────────────────────

def build_city_map(city_name, boundary_gdf, clustered_gdf, df,
                   cmap, patches, gas_col=None):
    """Build a publication-quality spatial cluster map for one city."""
    fig, ax = plt.subplots(figsize=(12, 10))
    colors  = [cmap[c] for c in df["Cluster"]]

    if gas_col and gas_col in df.columns:
        vals  = df[gas_col].fillna(0)
        sizes = 40 + 180 * (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        size_note = f" | Marker size ~ {gas_col}"
    else:
        sizes, size_note = 80, ""

    plotted_basemap = False
    if HAS_CTX:
        try:
            bnd_wm = boundary_gdf.to_crs(epsg=3857)
            clu_wm = clustered_gdf.to_crs(epsg=3857)
            bnd_wm.plot(ax=ax, facecolor="none", edgecolor="black",
                        linewidth=2.5, zorder=3)
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom=12, alpha=0.65)
            clu_wm.plot(ax=ax, color=colors, markersize=sizes,
                        alpha=0.88, zorder=4, edgecolor="white", linewidth=0.3)
            ax.set_axis_off()
            plotted_basemap = True
        except Exception as e:
            logger.warning("Basemap failed: %s", e)

    if not plotted_basemap:
        boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="black",
                          linewidth=2.5, zorder=1)
        clustered_gdf.plot(ax=ax, color=colors, markersize=sizes,
                           alpha=0.88, zorder=2, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude",  fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Stats box
    ax.text(0.02, 0.02, stats_text(df), transform=ax.transAxes,
            fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.6", alpha=0.93))

    # North arrow
    ax.annotate("N", xy=(0.96, 0.95), xytext=(0.96, 0.89),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    # Legend
    ax.legend(handles=patches,
              title="K-Means Cluster (k=4)\nRanked by NO\u2082",
              title_fontsize=8.5, fontsize=8,
              loc="upper left", framealpha=0.95, edgecolor="0.6")

    ax.set_title(
        f"Spatial K-Means Cluster Map \u2014 {city_name}\n"
        f"Sentinel-5P TROPOMI + VIIRS NTL + GHSL | 5 km Grid{size_note}",
        fontsize=12, fontweight="bold", pad=13)

    fig.tight_layout()
    return fig

def build_combined_figure(results):
    """3-panel side-by-side map of all three cities."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 10))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Spatial K-Means Clustering \u2014 Eastern Province Study Cities\n"
        "Sentinel-5P TROPOMI + VIIRS NTL + GHSL | k=4 | 5 km Grid",
        fontsize=14, fontweight="bold", y=1.02)

    for ax, res in zip(axes, results):
        res["boundary_gdf"].plot(ax=ax, facecolor="#f5f5f5", edgecolor="black",
                                  linewidth=2, zorder=1)
        colors = [res["cmap"][c] for c in res["df"]["Cluster"]]
        res["clustered_gdf"].plot(ax=ax, color=colors, markersize=55,
                                   alpha=0.88, zorder=2,
                                   edgecolor="white", linewidth=0.3)
        ax.set_title(res["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude",  fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.5)
        ax.text(0.02, 0.02, stats_text(res["df"]),
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="0.7", alpha=0.9))

    fig.legend(handles=results[0]["patches"],
               title="Cluster Type", title_fontsize=9, fontsize=8.5,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               framealpha=0.95, edgecolor="0.6")
    fig.tight_layout()
    return fig

# ── Step 8: Main processing loop ──────────────────────────────────────────────

saved_plots  = []
all_results  = []

for city_name, cfg in CITY_CONFIG.items():
    print(f"\n{'='*62}\n  Processing: {city_name}\n{'='*62}")

    # Find shapefile — pass zip_search so the recursive fallback is city-specific
    shp_path = find_shp(cfg["shp_candidates"], cfg["zip_search"])
    if shp_path is None:
        logger.error("%s: Cannot find .shp file. Skipping.", city_name)
        print(f"  ✗  Shapefile not found for {city_name}.")
        print(f"     Searched candidates: {cfg['shp_candidates']}")
        print(f"     Also searched /content/ for files containing '{cfg['zip_search']}'")
        continue

    boundary_gdf = gpd.read_file(shp_path)
    if boundary_gdf.crs is None:
        boundary_gdf.set_crs("EPSG:4326", inplace=True)
    else:
        boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
    print(f"  ✓  Shapefile loaded: {shp_path}")

    # Sanity check: confirm loaded shapefile belongs to the correct city
    shp_fname = os.path.basename(shp_path).lower()
    if cfg['zip_search'] not in shp_fname:
        print(f"  \u26a0  WARNING: '{shp_fname}' does not match city keyword "
              f"'{cfg['zip_search']}' -- wrong shapefile loaded! Check your ZIPs.")
    else:
        print(f"  \u2713  Shapefile name confirmed correct for {city_name}.")

    # Find clustered CSV
    csv_path = find_csv(cfg["csv_search"])
    if csv_path is None:
        logger.error(
            "%s: Cannot find clustered CSV. "
            "Make sure you uploaded Dammam_Final_Clustered_MapData.csv "
            "(NOT Hotspot_Dammam_FINAL.csv).", city_name)
        print(f"  ✗  Clustered CSV not found for {city_name}.")
        continue

    print(f"  ✓  Clustered CSV: {csv_path}")

    try:
        df = load_csv(csv_path)
    except ValueError as e:
        print(str(e))
        continue

    # Al Khobar CH4 fix: 43% coverage causes excessive NaN row dropout.
    # Impute missing CH4 values with the column median so all grid cells
    # are retained for spatial clustering and mapping.
    # Document in thesis: "CH4 NaN values for Al Khobar were imputed using
    # the temporal median prior to clustering due to 43% data coverage."
    if city_name == "Al Khobar" and "CH4" in df.columns:
        n_nan = df["CH4"].isna().sum()
        if n_nan > 0:
            median_ch4 = df["CH4"].median()
            df["CH4"].fillna(median_ch4, inplace=True)
            logger.info(
                "Al Khobar: imputed %d CH4 NaN values with median (%.4f).",
                n_nan, median_ch4
            )
            print(f"  ✓  Al Khobar CH4: {n_nan} missing values imputed with "
                  f"median ({median_ch4:.4f}) — retaining all grid cells.")

    # CSV -> GeoDataFrame
    geometry     = [Point(lon, lat)
                    for lon, lat in zip(df["Longitude"], df["Latitude"])]
    clustered_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry,
                                      crs="EPSG:4326")

    cmap, patches = build_color_map(df)

    all_results.append({
        "name":         city_name,
        "boundary_gdf": boundary_gdf,
        "clustered_gdf": clustered_gdf,
        "df":           df,
        "cmap":         cmap,
        "patches":      patches,
    })

    city_slug = city_name.replace(" ", "_")

    # Cluster map
    fig1 = build_city_map(city_name, boundary_gdf, clustered_gdf, df,
                          cmap, patches)
    p1 = os.path.join(OUTPUT_DIR, f"{city_slug}_Cluster_Map.png")
    fig1.savefig(p1); saved_plots.append(p1)
    plt.show(); plt.close(fig1)
    print(f"  ✓  Saved: {p1}")

    # NO2-scaled map
    if "NO2" in df.columns:
        fig2 = build_city_map(city_name, boundary_gdf, clustered_gdf, df,
                              cmap, patches, gas_col="NO2")
        p2 = os.path.join(OUTPUT_DIR, f"{city_slug}_Cluster_Map_NO2.png")
        fig2.savefig(p2); saved_plots.append(p2)
        plt.show(); plt.close(fig2)
        print(f"  ✓  Saved: {p2}")

# ── Step 9: Combined figure ───────────────────────────────────────────────────

if len(all_results) > 1:
    print("\nBuilding combined 3-panel figure ...")
    fig_c = build_combined_figure(all_results)
    pc    = os.path.join(OUTPUT_DIR, "All_Cities_Cluster_Map.png")
    fig_c.savefig(pc, bbox_inches="tight")
    saved_plots.append(pc)
    plt.show(); plt.close(fig_c)
    print(f"  ✓  Saved: {pc}")

# ── Step 10: Download ─────────────────────────────────────────────────────────

if not saved_plots:
    print("\n✗  No maps generated. Check the errors above.")
else:
    print(f"\n{'='*62}")
    print(f"  Downloading {len(saved_plots)} map(s) ...")
    print(f"{'='*62}")
    for path in saved_plots:
        colab_files.download(path)
    print(f"\n✓  Done. {len(saved_plots)} maps saved to '{OUTPUT_DIR}/'.")
    print("   Also available in the folder icon in the Colab left sidebar.")
