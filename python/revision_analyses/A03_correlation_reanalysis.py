"""
A03 - Gas-NTL correlation re-analysis (Section 3.4 / Table S1)

The original Table S1 was computed from the 72 MONTHLY city means (not 6 annual values).
This script:
  1. reproduces Table S1 (Pearson r, naive p, n);
  2. corrects significance for serial autocorrelation using the effective sample size
     n_eff = n (1 - r1x*r1y) / (1 + r1x*r1y)  (Bretherton et al., 1999), p from t with n_eff-2 df;
  3. repeats the test on deseasonalised anomalies (monthly climatology removed), so that a shared
     annual cycle cannot drive the correlation;
  4. reports Spearman rho for robustness.

Output: A03_correlation_table.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import OUT, CITIES, GASES, load_proxy


def r_and_neff(x, y):
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    r1x, r1y = x.autocorr(1), y.autocorr(1)
    neff = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    neff = max(min(neff, n), 3)
    tstat = r * np.sqrt((neff - 2) / (1 - r ** 2))
    p_adj = 2 * stats.t.sf(abs(tstat), neff - 2)
    return n, r, p, r1x, r1y, neff, p_adj


rows = []
for c in CITIES:
    d = load_proxy(c).set_index("date")
    for gas in GASES:
        n, r, p, r1x, r1y, neff, padj = r_and_neff(d[gas], d["NTL"])
        rho, prho = stats.spearmanr(d[gas], d["NTL"], nan_policy="omit")
        # deseasonalised anomalies
        ag = d[gas] - d[gas].groupby(d.index.month).transform("mean")
        an = d["NTL"] - d["NTL"].groupby(d.index.month).transform("mean")
        n2, r2, p2, _, _, neff2, padj2 = r_and_neff(ag, an)
        rows.append({"City": c, "Gas": gas, "n_months": n, "Pearson_r": r, "p_naive": p,
                     "lag1_gas": r1x, "lag1_NTL": r1y, "n_eff": neff, "p_autocorr_adjusted": padj,
                     "Spearman_rho": rho, "Spearman_p_naive": prho,
                     "anom_r": r2, "anom_p_naive": p2, "anom_n_eff": neff2, "anom_p_adjusted": padj2})
out = pd.DataFrame(rows)
out.to_csv(OUT / "A03_correlation_table.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string())
