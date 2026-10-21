"""
A08 - Build the manuscript figures produced by the analysis (600 dpi)

  Fig4_trend_PI.png        (a) Jubail NO2 and (b) Dammam CH4: annual means, OLS line, 95% prediction band to 2030
  FigS6_Khobar_CO_PI.png   Al-Khobar CO, same layout
  Fig5_cluster_maps.png    3-city cluster map (copied from 04_clustering_bestK/Cluster_Maps/All_Cities_Cluster_Map.png)
  Fig6_cluster_scatter.png Dammam / Jubail / Al-Khobar scatter plots stacked (from 04_clustering_bestK)
  FigS7_k_selection.png    elbow + silhouette plots of the three cities stacked (top to bottom) (from 04_clustering_bestK)

Output folder: results/figures
"""
import shutil
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from common import ROOT, load_monthly

Image.MAX_IMAGE_PIXELS = None
from common import OUT
FIG = OUT / "figures"
CL = Path(__file__).resolve().parents[1] / "clustering"  # run the clustering notebook first; its outputs are expected here
FIG.mkdir(exist_ok=True)
DPI = 600
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
LAB = {"NO2": "NO$_2$ total column (mol m$^{-2}$)", "CO": "CO column (mol m$^{-2}$)", "CH4": "CH$_4$ (ppb)"}
CITY = {"Dammam": "Dammam", "Jubail": "Jubail", "Khobar": "Al-Khobar"}


def panel(ax, city, gas, tag):
    df = load_monthly(city)
    ann = df.dropna(subset=[gas]).groupby("year")[gas].mean()
    yrs, vals = ann.index.values.astype(float), ann.values
    xm = yrs.mean()
    m = sm.OLS(vals, sm.add_constant(yrs - xm)).fit()
    yy = np.arange(2019, 2031)
    fr = m.get_prediction(sm.add_constant(yy - xm)).summary_frame(alpha=0.05)
    ax.axvspan(2024.5, 2030.5, color="0.93", zorder=0, label="Extrapolation period")
    ax.fill_between(yy, fr["obs_ci_lower"], fr["obs_ci_upper"], color="#9ecae1", alpha=0.55, label="95% prediction interval")
    ax.plot(yy, fr["mean"], "--", color="#d62728", lw=1.4, label="OLS fit / extrapolation")
    ax.plot(yrs, vals, "o-", color="#1f4e79", lw=1.6, ms=5, label="Annual mean (2019–2024)")
    ax.set_title(f"({tag}) {gas.replace('NO2','NO₂').replace('CH4','CH₄')} – {CITY[city]}", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel(LAB[gas]); ax.set_xticks(range(2019, 2031))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, ls=":", lw=0.5)


# Fig. 4
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
panel(axes[0], "Jubail", "NO2", "a")
panel(axes[1], "Dammam", "CH4", "b")
axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "Fig4_trend_PI.png", dpi=DPI); plt.close(fig)

# Fig. S6
fig, ax = plt.subplots(figsize=(5.2, 3.8))
panel(ax, "Khobar", "CO", "a")
ax.set_title("CO – Al-Khobar", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "FigS6_Khobar_CO_PI.png", dpi=DPI); plt.close(fig)

# Fig. 5
shutil.copy(CL / "Cluster_Maps" / "All_Cities_Cluster_Map.png", FIG / "Fig5_cluster_maps.png")


def stack(paths, out, vertical=True):
    ims = [Image.open(p).convert("RGB") for p in paths]
    if vertical:
        w = max(i.width for i in ims)
        canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
        y = 0
        for i in ims:
            canvas.paste(i, ((w - i.width) // 2, y)); y += i.height
    else:
        h = max(i.height for i in ims)
        canvas = Image.new("RGB", (sum(i.width for i in ims), h), "white")
        x = 0
        for i in ims:
            canvas.paste(i, (x, (h - i.height) // 2)); x += i.width
    canvas.save(out, dpi=(DPI, DPI), optimize=True)


R = CL / "Clustering_Analysis_Results"
stack([R / f"{c}_Clustering_Scatter.png" for c in ["Dammam", "Jubail", "Khobar"]], FIG / "Fig6_cluster_scatter.png")
stack([R / f"{c}_Elbow_Silhouette.png" for c in ["Dammam", "Jubail", "Khobar"]], FIG / "FigS7_k_selection.png", vertical=True)
print("figures written to", FIG)
