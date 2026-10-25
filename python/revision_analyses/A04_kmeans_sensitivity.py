"""
A04 - K-means validation and sensitivity (Section 2.6 / Table 3)

Reproduces the original clustering exactly (features NO2, CO, CH4, NTL, POP; rows with NaN dropped;
StandardScaler; KMeans k=4, n_init=10, random_state=42), then:
  1. diagnostics for k = 2..8: inertia, silhouette, Calinski-Harabasz, Davies-Bouldin;
  2. seed stability of k=4: 100 random_state values, adjusted Rand index (ARI) vs the reference;
  3. bootstrap stability of k=4: 200 resamples, model refit on resample and used to label all cells, ARI vs reference;
  4. agreement between adjacent solutions (ARI k=3 vs 4, 4 vs 5);
  5. NoData check: GHSL population NoData (-200) values; Jubail rerun with negative-population cells removed.
Items 1-4 are computed on the CORRECTED input (cells with negative POP removed; affects Jubail only:
372 -> 368 cells). The reproduction check uses the ORIGINAL input to confirm the exported labels.
Population is GHSL 2020 population_count averaged over each grid cell = mean persons per 100 m pixel.

Outputs: A04_k_diagnostics.csv, A04_stability_summary.csv, A04_reproduction_check.csv,
         A04_jubail_masked_profiles.csv, A04_cluster_profiles_reproduced.csv, A04_k_diagnostics.png
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
from common import OUT, CITIES, CITY_LABEL, HOTSPOT, CLUSTERED

FEAT = ["NO2", "CO", "CH4", "NTL", "POP"]
rng = np.random.default_rng(2026)


def fit(X, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def load(city, mask_negative_pop=False):
    h = pd.read_csv(HOTSPOT[city]).dropna(subset=FEAT).reset_index(drop=True)
    if mask_negative_pop:
        h = h[h["POP"] >= 0].reset_index(drop=True)
    return h


diag, stab, repro, prof = [], [], [], []
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, c in zip(axes, CITIES):
    # reproduction check against the exported labels (original input)
    h0 = load(c)
    X0 = StandardScaler().fit_transform(h0[FEAT])
    ref0 = fit(X0, 4)
    ex = pd.read_csv(CLUSTERED[c])
    m = h0[["Grid_ID"]].assign(new=ref0.labels_).merge(ex[["Grid_ID", "Cluster"]], on="Grid_ID", how="inner")
    repro.append({"City": c, "n_cells": len(h0), "n_matched": len(m),
                  "ARI_vs_exported_labels": adjusted_rand_score(m["Cluster"], m["new"]),
                  "silhouette_k4": silhouette_score(X0, ref0.labels_), "inertia_k4": ref0.inertia_,
                  "negative_POP_cells": int((h0["POP"] < 0).sum())})

    # corrected input for all validation analyses
    h = load(c, mask_negative_pop=True)
    X = StandardScaler().fit_transform(h[FEAT])
    ref = fit(X, 4)
    lab = ref.labels_
    p = h.assign(Cluster=lab).groupby("Cluster")[FEAT].mean()
    p["n"] = pd.Series(lab).value_counts().sort_index()
    p["pct"] = 100 * p["n"] / len(h)
    p.insert(0, "City", c)
    prof.append(p.reset_index())

    # 1. k diagnostics
    ks = range(2, 9)
    sil = []
    for k in ks:
        km = fit(X, k)
        sizes = np.bincount(km.labels_)
        sil.append(silhouette_score(X, km.labels_))
        diag.append({"City": c, "k": k, "inertia": km.inertia_, "silhouette": sil[-1],
                     "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_),
                     "smallest_cluster_n": sizes.min(), "cluster_sizes": "/".join(map(str, sorted(sizes, reverse=True)))})
    ax.plot(list(ks), sil, "o-", color="#1f6fb2")
    ax.axvline(4, color="0.6", ls="--", lw=1)
    ax.set_title(f"{CITY_LABEL[c]} (n = {len(h)})")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette score")

    # 2. seed stability
    ari_seed = [adjusted_rand_score(lab, fit(X, 4, seed=s).labels_) for s in range(100)]
    # 3. bootstrap stability
    ari_boot = []
    for b in range(200):
        idx = rng.integers(0, len(X), len(X))
        km = KMeans(n_clusters=4, n_init=10, random_state=b).fit(X[idx])
        ari_boot.append(adjusted_rand_score(lab, km.predict(X)))
    # 4. adjacent k agreement
    l3, l5 = fit(X, 3).labels_, fit(X, 5).labels_
    stab.append({"City": c,
                 "n_cells": len(h), "silhouette_k4": silhouette_score(X, lab),
                 "seed_ARI_median": np.median(ari_seed), "seed_ARI_min": np.min(ari_seed),
                 "seed_share_identical(ARI=1)": np.mean(np.isclose(ari_seed, 1)),
                 "boot_ARI_median": np.median(ari_boot), "boot_ARI_p05": np.percentile(ari_boot, 5),
                 "boot_ARI_p95": np.percentile(ari_boot, 95),
                 "ARI_k3_vs_k4": adjusted_rand_score(l3, lab), "ARI_k5_vs_k4": adjusted_rand_score(l5, lab)})
fig.tight_layout()
fig.savefig(OUT / "A04_k_diagnostics.png", dpi=300)

# 5. Jubail with negative-population (GHSL NoData) cells removed
hj = load("Jubail", mask_negative_pop=True)
Xj = StandardScaler().fit_transform(hj[FEAT])
jrows = []
for k in [3, 4]:
    km = fit(Xj, k)
    pj = hj.assign(Cluster=km.labels_).groupby("Cluster")[FEAT].mean()
    pj["n"] = np.bincount(km.labels_)
    pj["pct"] = 100 * pj["n"] / len(hj)
    pj["silhouette"] = silhouette_score(Xj, km.labels_)
    pj.insert(0, "k", k)
    jrows.append(pj.reset_index())
# agreement of masked k=4 with the original labels on the common cells
h_all = load("Jubail")
lab_all = fit(StandardScaler().fit_transform(h_all[FEAT]), 4).labels_
lab_m4 = fit(Xj, 4).labels_
lab_m3 = fit(Xj, 3).labels_
mm = h_all[["Grid_ID"]].assign(orig=lab_all).merge(hj[["Grid_ID"]].assign(m4=lab_m4, m3=lab_m3), on="Grid_ID")
jm = pd.concat(jrows)
jm["ARI_vs_original_on_common_cells"] = jm["k"].map({4: adjusted_rand_score(mm.orig, mm.m4),
                                                    3: adjusted_rand_score(mm.orig, mm.m3)})

pd.DataFrame(diag).to_csv(OUT / "A04_k_diagnostics.csv", index=False)
pd.DataFrame(stab).to_csv(OUT / "A04_stability_summary.csv", index=False)
pd.DataFrame(repro).to_csv(OUT / "A04_reproduction_check.csv", index=False)
pd.concat(prof).to_csv(OUT / "A04_cluster_profiles_reproduced.csv", index=False)
jm.to_csv(OUT / "A04_jubail_masked_profiles.csv", index=False)
pd.set_option("display.width", 250)
for name, d in [("reproduction", pd.DataFrame(repro)), ("diagnostics", pd.DataFrame(diag)),
                ("stability", pd.DataFrame(stab)), ("jubail masked", jm)]:
    print("\n==", name); print(d.round(4).to_string())
