"""
Batch-forecast demand for the citywide series and the busiest zones, saving a
single tidy file the Streamlit app serves. Run this offline (locally or in
Colab) whenever you want to refresh the app's forecasts, then commit
outputs/forecasts.csv.

This is the standard production split: heavy model fitting happens in a batch
job; the serving layer (the app) just reads the results.

Output: outputs/forecasts.csv with columns
        level ('city' or 'zone'), zone_id, ds, yhat, yhat_lower, yhat_upper
"""
import logging
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from prophet import Prophet

CITY_CSV = "data/processed/citywide_hourly.csv"
ZONE_CSV = "data/processed/zone_hourly_demand.csv"
OUT = "outputs/forecasts.csv"

HORIZON = 168       # forecast 7 days ahead
COVERAGE = 0.95     # forecast busiest zones covering this share of trips


def fit_forecast(history, horizon):
    m = Prophet(daily_seasonality=20, weekly_seasonality=10,
                yearly_seasonality=False, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.1)
    m.fit(history)
    future = m.make_future_dataframe(periods=horizon, freq="h")
    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        fc[c] = fc[c].clip(lower=0).round(1)
    return fc[fc.ds > history.ds.max()].reset_index(drop=True)


def main():
    os.makedirs("outputs", exist_ok=True)
    rows = []

    # citywide
    city = pd.read_csv(CITY_CSV, parse_dates=["ds"])[["ds", "y"]]
    fc = fit_forecast(city, HORIZON)
    fc.insert(0, "zone_id", -1)
    fc.insert(0, "level", "city")
    rows.append(fc)
    print(f"[batch] citywide done ({len(fc)} rows)")

    # busiest zones
    zh = pd.read_csv(ZONE_CSV, parse_dates=["pickup_hour"])
    idx = pd.date_range(zh.pickup_hour.min(), zh.pickup_hour.max(), freq="h")
    totals = zh.groupby("zone_id")["demand"].sum().sort_values(ascending=False)
    cum = totals.cumsum() / totals.sum()
    zones = cum[cum <= COVERAGE].index.tolist()
    print(f"[batch] forecasting {len(zones)} zones "
          f"({totals[zones].sum()/totals.sum():.1%} of trips)")

    for i, zid in enumerate(zones, 1):
        s = (zh[zh.zone_id == zid].set_index("pickup_hour")["demand"]
             .reindex(idx, fill_value=0))
        hist = pd.DataFrame({"ds": idx, "y": s.values})
        fc = fit_forecast(hist, HORIZON)
        fc.insert(0, "zone_id", int(zid))
        fc.insert(0, "level", "zone")
        rows.append(fc)
        if i % 15 == 0:
            print(f"[batch] {i}/{len(zones)} zones", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"[batch] wrote {OUT} ({len(out):,} rows, "
          f"{out.zone_id.nunique()} series)")


if __name__ == "__main__":
    main()
