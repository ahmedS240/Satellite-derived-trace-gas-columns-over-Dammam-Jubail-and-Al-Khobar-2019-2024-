"""
A06 - 2030 extrapolation uncertainty and sensitivity (Section 4.2 / Table 2 / Table S2)

For each city and variable (NO2, CO, CH4, NTL), annual means are formed from the available months and:
  1. OLS fitted to 2019-2024 -> 2030 value with 95% prediction interval (statsmodels, t with n-2 df);
  2. Theil-Sen line (median slope, median intercept) -> 2030 value;
  3. OLS fitted to 2020-2024 and 2021-2024 (excluding the 2019-2020 pre-/lockdown years) -> 2030 value.
Percentage changes are relative to the observed 2024 annual mean.
Al-Khobar CH4 is reported but flagged: 2019 has 1 valid month and 2020 none.

Outputs: A06_projection_sensitivity.csv, A06_projection_PI_plots.png
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, CITIES, CITY_LABEL, load_monthly

VARS = ["NO2", "CO", "CH4", "NTL"]


def ols_pred(years, vals, at=2030):
    X = sm.add_constant(years - years.mean())
    m = sm.OLS(vals, X).fit()
    pr = m.get_prediction(np.array([[1.0, at - years.mean()]])).summary_frame(alpha=0.05)
    return float(pr["mean"].iloc[0]), float(pr["obs_ci_lower"].iloc[0]), float(pr["obs_ci_upper"].iloc[0]), m


rows = []
fig, axes = plt.subplots(3, 4, figsize=(15, 9))
for i, c in enumerate(CITIES):
    df = load_monthly(c)
    for j, v in enumerate(VARS):
        ann = df.dropna(subset=[v]).groupby("year")[v].agg(["mean", "count"])
        yrs = ann.index.values.astype(float)
        vals = ann["mean"].values
        last = vals[-1]
        p, lo, hi, m = ols_pred(yrs, vals)
        ts = stats.theilslopes(vals, yrs)
        p_ts = ts.intercept + ts.slope * 2030
        rec = {"City": c, "Variable": v, "n_years": len(yrs), "Observed_2024": last,
               "OLS_slope": m.params[1], "OLS_slope_p": m.pvalues[1],
               "OLS_2030": p, "PI95_low": lo, "PI95_high": hi,
               "OLS_pct": 100 * (p - last) / abs(last),
               "PI95_low_pct": 100 * (lo - last) / abs(last), "PI95_high_pct": 100 * (hi - last) / abs(last),
               "TheilSen_2030": p_ts, "TheilSen_pct": 100 * (p_ts - last) / abs(last)}
        for start in (2020, 2021):
            k = yrs >= start
            if k.sum() >= 3:
                lr = stats.linregress(yrs[k], vals[k])
                q = lr.intercept + lr.slope * 2030
                rec[f"OLS_{start}_2024_pct"] = 100 * (q - last) / abs(last)
        rec["flag"] = "unreliable: 2019 has 1 valid month and 2020 none" if (c == "Khobar" and v == "CH4") else ""
        rows.append(rec)
        ax = axes[i, j]
        yy = np.arange(2019, 2031)
        X = sm.add_constant(yy - yrs.mean())
        fr = m.get_prediction(X).summary_frame(alpha=0.05)
        ax.fill_between(yy, fr["obs_ci_lower"], fr["obs_ci_upper"], color="#9ecae1", alpha=0.5, label="95% PI")
        ax.plot(yy, fr["mean"], "--", color="#d62728", lw=1.3, label="OLS")
        ax.plot(yrs, vals, "o-", color="#1f6fb2", label="Annual mean")
        ax.axvspan(2024.5, 2030.5, color="0.92", zorder=0)
        ax.set_title(f"{CITY_LABEL[c]} {v}", fontsize=10)
        ax.tick_params(labelsize=8)
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "A06_projection_PI_plots.png", dpi=300)
out = pd.DataFrame(rows)
out.to_csv(OUT / "A06_projection_sensitivity.csv", index=False)
pd.set_option("display.width", 260)
print(out.drop(columns=["OLS_slope"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
