"""
A07 - Data-driven choice of k for each city (replaces the fixed k = 4)

Input: corrected grid-cell tables (cells with negative GHSL population removed; affects Jubail only).
Pipeline per city: StandardScaler -> KMeans(n_init=10, random_state=42) for k = 2..8.

Selection rule (stated before looking at the cluster profiles):
  1. Primary criterion: the k with the highest mean silhouette score (Rousseeuw, 1987).
  2. Admissibility: the solution must be stable, i.e. median bootstrap ARI >= 0.75
     (200 resamples; centroids fitted on the resample label all cells), and no cluster
     may contain fewer than 2 cells. If the best k fails, the next-best k is taken.
Other indices (Calinski-Harabasz, Davies-Bouldin, gap statistic) are reported for transparency.

Outputs (../02_outputs):
  A07_k_selection_table.csv          all indices for k = 2..8 and each city
  A07_selected_k.csv                 chosen k per city with reason
  A07_cluster_profiles_selected_k.csv
  <City>_Clustered_MapData_bestK.csv  GIS-ready labels for the chosen k
  A07_k_selection.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score)
from common import OUT, CITIES, CITY_LABEL, HOTSPOT

FEAT = ["NO2", "CO", "CH4", "NTL", "POP"]
KS = range(2, 9)
N_BOOT, N_GAP = 200, 20
ARI_MIN, MIN_SIZE = 0.75, 2


def fit(X, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def gap_stat(X, k, rng):
    """Tibshirani et al. (2001) gap statistic with uniform reference in the bounding box."""
    wk = np.log(fit(X, k).inertia_)
    lo, hi = X.min(0), X.max(0)
    ref = [np.log(KMeans(n_clusters=k, n_init=3, random_state=b).fit(rng.uniform(lo, hi, X.shape)).inertia_)
           for b in range(N_GAP)]
    return np.mean(ref) - wk, np.std(ref) * np.sqrt(1 + 1 / N_GAP)


rows, chosen, profiles = [], [], []
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, c in zip(axes, CITIES):
    rng = np.random.default_rng(2026)
    h = pd.read_csv(HOTSPOT[c]).dropna(subset=FEAT)
    h = h[h["POP"] >= 0].reset_index(drop=True)
    X = StandardScaler().fit_transform(h[FEAT])
    city_rows = []
    for k in KS:
        km = fit(X, k)
        lab = km.labels_
        boot = []
        for b in range(N_BOOT):
            idx = rng.integers(0, len(X), len(X))
            boot.append(adjusted_rand_score(lab, KMeans(n_clusters=k, n_init=10, random_state=b).fit(X[idx]).predict(X)))
        g, gs = gap_stat(X, k, rng)
        sizes = np.bincount(lab)
        city_rows.append({"City": c, "n_cells": len(h), "k": k, "silhouette": silhouette_score(X, lab),
                          "calinski_harabasz": calinski_harabasz_score(X, lab),
                          "davies_bouldin": davies_bouldin_score(X, lab), "gap": g, "gap_se": gs,
                          "boot_ARI_median": np.median(boot), "boot_ARI_p05": np.percentile(boot, 5),
                          "smallest_cluster": sizes.min(),
                          "cluster_sizes": "/".join(map(str, sorted(sizes, reverse=True)))})
    df = pd.DataFrame(city_rows)
    df["admissible"] = (df.boot_ARI_median >= ARI_MIN) & (df.smallest_cluster >= MIN_SIZE)
    order = df.sort_values("silhouette", ascending=False)
    best = order[order.admissible].iloc[0]
    top = order.iloc[0]
    reason = ("highest silhouette" if best.k == top.k else
              f"k={int(top.k)} had the highest silhouette but failed admissibility "
              f"(bootstrap ARI {top.boot_ARI_median:.2f}, smallest cluster {int(top.smallest_cluster)})")
    # gap-statistic choice (smallest k with gap(k) >= gap(k+1) - se(k+1)) for reference
    gk = None
    for i in range(len(df) - 1):
        if df.gap.iloc[i] >= df.gap.iloc[i + 1] - df.gap_se.iloc[i + 1]:
            gk = int(df.k.iloc[i]); break
    chosen.append({"City": c, "selected_k": int(best.k), "silhouette": best.silhouette,
                   "boot_ARI_median": best.boot_ARI_median, "reason": reason,
                   "k_by_CH_max": int(df.loc[df.calinski_harabasz.idxmax(), "k"]),
                   "k_by_DB_min": int(df.loc[df.davies_bouldin.idxmin(), "k"]), "k_by_gap": gk})
    rows.append(df)
    km = fit(X, int(best.k))
    h["Cluster"] = km.labels_
    h[["Longitude", "Latitude", "Grid_ID"] + FEAT + ["Cluster"]].to_csv(OUT / f"{c}_Clustered_MapData_bestK.csv", index=False)
    p = h.groupby("Cluster")[FEAT].mean()
    p["n"] = h.Cluster.value_counts().sort_index()
    p["pct"] = 100 * p.n / len(h)
    p["lon_mean"] = h.groupby("Cluster").Longitude.mean()
    p["lat_mean"] = h.groupby("Cluster").Latitude.mean()
    p.insert(0, "City", c); p.insert(1, "k", int(best.k))
    profiles.append(p.reset_index())
    ax.plot(df.k, df.silhouette, "o-", color="#1f6fb2", label="Silhouette")
    ax.axvline(best.k, color="#d62728", ls="--", lw=1, label=f"Selected k = {int(best.k)}")
    ax.set_title(f"{CITY_LABEL[c]} (n = {len(h)})"); ax.set_xlabel("k"); ax.set_ylabel("Silhouette score")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "A07_k_selection.png", dpi=300)

tab = pd.concat(rows); tab.to_csv(OUT / "A07_k_selection_table.csv", index=False)
sel = pd.DataFrame(chosen); sel.to_csv(OUT / "A07_selected_k.csv", index=False)
prof = pd.concat(profiles); prof.to_csv(OUT / "A07_cluster_profiles_selected_k.csv", index=False)
pd.set_option("display.width", 250)
print(tab.round(3).to_string(index=False)); print(sel.round(3).to_string(index=False))
print(prof.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
