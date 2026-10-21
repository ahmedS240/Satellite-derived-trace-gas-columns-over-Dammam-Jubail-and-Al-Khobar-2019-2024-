"""
A04b - Corrected Jubail clustering with GHSL population NoData removed

Four Jubail cells in Hotspot_Jubail_FINAL.csv have negative POP (one equals the GHSL NoData value -200;
three are partial averages contaminated by NoData). The original k=4 run therefore produced a
singleton cluster (cell 1095,614) and inflated the POP standard deviation used in the Z-scores.
This script removes those 4 cells and reruns the identical pipeline (StandardScaler, KMeans k=4,
n_init=10, random_state=42). Cluster numbers are the raw K-means labels, identical to those produced
by the corrected Colab notebook (Untitled12_corrected.ipynb) and used in Table 3.

Output: Jubail_Clustered_MapData_NoDataMasked.csv (GIS-ready: Longitude, Latitude, Grid_ID, features, Cluster)
        A04b_jubail_masked_profiles.csv
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from common import OUT, HOTSPOT

FEAT = ["NO2", "CO", "CH4", "NTL", "POP"]
h = pd.read_csv(HOTSPOT["Jubail"]).dropna(subset=FEAT)
removed = h[h["POP"] < 0][["Grid_ID", "POP"]]
h = h[h["POP"] >= 0].reset_index(drop=True)
X = StandardScaler().fit_transform(h[FEAT])
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
h["Cluster"] = km.labels_
h[["Longitude", "Latitude", "Grid_ID"] + FEAT + ["Cluster"]].to_csv(OUT / "Jubail_Clustered_MapData_NoDataMasked.csv", index=False)
prof = h.groupby("Cluster")[FEAT].mean()
prof["n"] = h["Cluster"].value_counts().sort_index()
prof["pct"] = 100 * prof["n"] / len(h)
prof["silhouette"] = silhouette_score(X, km.labels_)
prof.to_csv(OUT / "A04b_jubail_masked_profiles.csv")
print("Removed cells:\n", removed.to_string(index=False))
print(prof.to_string(float_format=lambda x: f"{x:.6g}"))
