"""
Make forward predictions of taxi demand.

Unlike the evaluation in 03/04 (which scored models on a held-out window), this
script fits Prophet on ALL available history and forecasts the *future* — the
next N hours beyond the data — with uncertainty intervals.

Why Prophet here and not the hybrid: the LSTM corrects Prophet using the actual
previous hour, so it only does one-step-ahead. For a multi-hour forward horizon,
Prophet gives a proper forecast with confidence bands.

Examples
--------
  # citywide, next 7 days
  python src/predict.py --level city --horizon 168

  # a single zone (e.g. 161 = Midtown Center), next 48 hours
  python src/predict.py --level zone --zone 161 --horizon 48

Output: outputs/forecast_<...>.csv with columns
        ds, yhat, yhat_lower, yhat_upper
"""
import argparse
import logging
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from prophet import Prophet

CITY_CSV = "data/processed/citywide_hourly.csv"
ZONE_CSV = "data/processed/zone_hourly_demand.csv"


def fit_prophet(history: pd.DataFrame) -> Prophet:
    """history: columns ds, y on a gap-free hourly index."""
    m = Prophet(
        daily_seasonality=20,
        weekly_seasonality=10,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )
    m.fit(history)
    return m


def forecast(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    m = fit_prophet(history)
    future = m.make_future_dataframe(periods=horizon, freq="h")
    fc = m.predict(future)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        out[c] = out[c].clip(lower=0).round(1)
    # keep only the future portion
    return out[out.ds > history.ds.max()].reset_index(drop=True)


def get_city_history() -> pd.DataFrame:
    df = pd.read_csv(CITY_CSV, parse_dates=["ds"])
    return df[["ds", "y"]]


def get_zone_history(zone_id: int) -> pd.DataFrame:
    df = pd.read_csv(ZONE_CSV, parse_dates=["pickup_hour"])
    idx = pd.date_range(df.pickup_hour.min(), df.pickup_hour.max(), freq="h")
    s = (df[df.zone_id == zone_id].set_index("pickup_hour")["demand"]
         .reindex(idx, fill_value=0))
    if s.sum() == 0:
        raise ValueError(f"No demand found for zone {zone_id}")
    return pd.DataFrame({"ds": idx, "y": s.values})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["city", "zone"], default="city")
    ap.add_argument("--zone", type=int, default=None,
                    help="zone id (required when --level zone)")
    ap.add_argument("--horizon", type=int, default=168,
                    help="hours to forecast ahead (default 168 = 1 week)")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if args.level == "city":
        hist = get_city_history()
        fc = forecast(hist, args.horizon)
        path = "outputs/forecast_city.csv"
    else:
        if args.zone is None:
            ap.error("--zone is required when --level zone")
        hist = get_zone_history(args.zone)
        fc = forecast(hist, args.horizon)
        path = f"outputs/forecast_zone_{args.zone}.csv"

    fc.to_csv(path, index=False)
    print(f"History ends: {hist.ds.max()}")
    print(f"Forecasting {args.horizon}h -> {fc.ds.min()} .. {fc.ds.max()}")
    print(f"Wrote {path}\n")
    print("First 12 hours of the forecast:")
    print(fc.head(12).to_string(index=False))
    peak = fc.loc[fc.yhat.idxmax()]
    print(f"\nPredicted peak: {peak.yhat:.0f} pickups at {peak.ds}")


if __name__ == "__main__":
    main()
