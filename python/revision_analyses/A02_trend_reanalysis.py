"""
A02 - Trend re-analysis (responds to postdoc comment on Section 2.5 and Table 2)

1. Reproduces the original annual analysis (annual mean of available months ->
   pymannkendall.original_test, OLS, 2030 value) to check Table 2.
2. Adds the monthly tests the manuscript describes but the original code did not run:
     - original MK on the 72 monthly values (reference only, ignores autocorrelation)
     - Hamed & Rao (1998) variance-corrected modified MK
     - Yue & Wang (2004) variance-corrected modified MK
     - trend-free pre-whitening MK (Yue et al., 2002)
     - seasonal MK (Hirsch et al., 1982), period = 12
   plus lag-1 autocorrelation of the deseasonalised series.
Sen's slopes from monthly tests are converted to per-year units (x 12).

Outputs: A02_annual_trend_check.csv, A02_monthly_trend_tests.csv
"""
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
from common import OUT, CITIES, GASES, load_monthly

# ---------------- values printed in the manuscript (Table 2) for comparison
TABLE2 = {
    ("Dammam", "NO2"): (0.467, 2.05e-6, 0.260, 0.48, 0.000140, 0.000157, 12.0),
    ("Dammam", "CO"): (0.333, 3.05e-4, 0.452, 0.26, 0.03609, 0.03651, 1.2),
    ("Dammam", "CH4"): (0.867, 6.35, 0.024, 0.88, 1940.65, 1979.59, 2.0),
    ("Jubail", "NO2"): (0.467, 8.10e-7, 0.260, 0.49, 0.000104, 0.000111, 6.3),
    ("Jubail", "CO"): (0.067, 1.53e-4, 1.000, 0.03, 0.03487, 0.03442, -1.3),
    ("Jubail", "CH4"): (1.000, 6.68, 0.009, 0.93, 1941.96, 1982.81, 2.1),
    ("Khobar", "NO2"): (0.200, 9.25e-7, 0.707, 0.07, 0.000147, 0.000160, 8.6),
    ("Khobar", "CO"): (0.067, 1.88e-4, 1.000, 0.08, 0.03569, 0.03542, -0.8),
    ("Khobar", "CH4"): (0.000, -0.535, 1.000, 0.06, 1921.61, 1928.30, 0.4),
}

annual_rows, monthly_rows = [], []
for c in CITIES:
    df = load_monthly(c)
    for gas in GASES:
        # ---------- annual (as in original script: mean of available months)
        ann = df.dropna(subset=[gas]).groupby("year")[gas].agg(["mean", "count"])
        yrs = ann.index.values.astype(float)
        vals = ann["mean"].values
        r = mk.original_test(vals)
        lr = stats.linregress(yrs, vals)
        proj = lr.intercept + lr.slope * 2030
        pct = (proj - vals[-1]) / abs(vals[-1]) * 100
        t = TABLE2[(c, gas)]
        annual_rows.append({
            "City": c, "Gas": gas, "n_years": len(vals),
            "months_per_year": ";".join(f"{int(y)}:{n}" for y, n in zip(ann.index, ann["count"])),
            "MK_trend": r.trend, "tau": r.Tau, "Sen_slope_per_yr": r.slope, "p": r.p,
            "OLS_slope_per_yr": lr.slope, "OLS_R2": lr.rvalue ** 2,
            "Mean_last_year": vals[-1], "OLS_2030": proj, "Pct_change_2030": pct,
            "MS_tau": t[0], "MS_Sen": t[1], "MS_p": t[2], "MS_R2": t[3], "MS_mean2024": t[4],
            "MS_proj2030": t[5], "MS_pct": t[6],
            "Sen_sign_matches_OLS_sign": np.sign(r.slope) == np.sign(lr.slope),
        })

        # ---------- monthly
        s = df.set_index("date")[gas]
        n_valid = int(s.notna().sum())
        rec = {"City": c, "Gas": gas, "n_valid_months": n_valid}
        if n_valid < 36:
            rec["note"] = "Too few valid months (<36) for monthly trend testing"
            monthly_rows.append(rec)
            continue
        x = s.values
        clim = s.groupby(s.index.month).transform("mean")
        anom = (s - clim).dropna()
        rec["lag1_autocorr_deseasonalised"] = anom.autocorr(lag=1)
        xs = s.dropna().values  # gap-removed series for non-seasonal tests
        tests = {
            "MK_original": mk.original_test(xs),
            "MK_HamedRao": mk.hamed_rao_modification_test(xs),
            "MK_YueWang": mk.yue_wang_modification_test(xs),
            "MK_TFPW": mk.trend_free_pre_whitening_modification_test(xs),
            "MK_seasonal": mk.seasonal_test(x, period=12),  # NaNs handled per season
            # same tests on deseasonalised anomalies (removes the annual cycle)
            "MK_HamedRao_deseason": mk.hamed_rao_modification_test(anom.values),
        }
        for k, res in tests.items():
            rec[f"{k}_trend"] = res.trend
            rec[f"{k}_p"] = res.p
            rec[f"{k}_Sen_per_yr"] = res.slope * (1 if k == "MK_seasonal" else 12)
        rec["note"] = "Seasonal MK slope is already per year (period=12)"
        monthly_rows.append(rec)

ann = pd.DataFrame(annual_rows)
mon = pd.DataFrame(monthly_rows)
ann.to_csv(OUT / "A02_annual_trend_check.csv", index=False)
mon.to_csv(OUT / "A02_monthly_trend_tests.csv", index=False)
pd.set_option("display.width", 250)
print(ann[["City", "Gas", "months_per_year", "tau", "MS_tau", "Sen_slope_per_yr", "MS_Sen", "p", "MS_p",
           "OLS_R2", "MS_R2", "OLS_2030", "MS_proj2030", "Pct_change_2030", "MS_pct", "Sen_sign_matches_OLS_sign"]].to_string())
cols = ["City", "Gas", "n_valid_months", "lag1_autocorr_deseasonalised"] + \
       [f"{k}_{m}" for k in ["MK_original", "MK_HamedRao", "MK_YueWang", "MK_TFPW", "MK_seasonal", "MK_HamedRao_deseason"] for m in ["p", "Sen_per_yr"]]
print(mon.reindex(columns=cols).to_string())
