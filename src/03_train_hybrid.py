"""
Stage 3 — Prophet + LSTM hybrid forecaster (citywide hourly demand).

Design
------
The hybrid decomposes demand into two parts and lets each model do what it is
good at:

  1. Prophet learns the smooth structure — trend plus daily and weekly
     seasonality.  It captures the rush-hour shape and the weekday/weekend
     pattern but misses short, irregular swings.
  2. An LSTM is trained on Prophet's *residuals* (actual - Prophet).  It learns
     the autocorrelated, short-horizon wiggles Prophet leaves behind.

  Final forecast:  y_hat = prophet(t) + lstm_residual(t)

Evaluation is one-step-ahead on a held-out final 14 days, the realistic regime
for operational dispatch. The hybrid is compared against four baselines so the
added complexity has to earn its place:
  * naive_24     — same hour yesterday
  * naive_168    — same hour last week
  * prophet_only — seasonal/trend baseline alone
  * lstm_only    — LSTM trained directly on demand
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"

from prophet import Prophet
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K

tf.random.set_seed(42)
np.random.seed(42)

CITY_CSV = "data/processed/citywide_hourly.csv"
OUT_METRICS = "outputs/metrics.json"
OUT_PRED = "outputs/test_predictions.csv"

TEST_HOURS = 24 * 14   # final 14 days held out
WINDOW = 168           # LSTM looks back one full week (captures weekly cycle)
EPOCHS = 60


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def make_sequences(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_lstm(window):
    m = models.Sequential([
        layers.Input((window, 1)),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def main():
    df = pd.read_csv(CITY_CSV, parse_dates=["ds"])
    y = df.y.values.astype(float)
    n = len(df)
    split = n - TEST_HOURS
    print(f"[model] train hours: {split} | test hours: {TEST_HOURS}")

    # ---------------------------------------------------------------- Prophet
    train_df = df.iloc[:split][["ds", "y"]]
    # Higher Fourier orders let Prophet model the sharp morning trough and
    # evening rush peak that the default (order 4) over-smooths; multiplicative
    # mode reflects that weekday/weekend swings scale with the demand level.
    prophet = Prophet(
        daily_seasonality=20,
        weekly_seasonality=10,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )
    prophet.fit(train_df)
    full_future = df[["ds"]]
    prophet_fit = prophet.predict(full_future)["yhat"].values
    prophet_fit = np.clip(prophet_fit, 0, None)
    print("[model] Prophet fitted.")

    # Residuals Prophet leaves behind (what the LSTM will learn).
    resid = y - prophet_fit
    r_mean, r_std = resid[:split].mean(), resid[:split].std()
    resid_n = (resid - r_mean) / r_std

    # ------------------------------------------------ LSTM on Prophet residuals
    Xr, yr = make_sequences(resid_n, WINDOW)
    seq_split = split - WINDOW
    Xr_tr, yr_tr = Xr[:seq_split], yr[:seq_split]
    Xr_tr = Xr_tr[..., None]

    lstm_resid = build_lstm(WINDOW)
    es = callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    lstm_resid.fit(Xr_tr, yr_tr, validation_split=0.1, epochs=EPOCHS,
                   batch_size=64, verbose=0, callbacks=[es])
    rhat_n = lstm_resid.predict(Xr[..., None], verbose=0).flatten()
    rhat = rhat_n * r_std + r_mean
    # align residual preds to absolute time index (start at WINDOW)
    hybrid_full = prophet_fit.copy()
    hybrid_full[WINDOW:] = prophet_fit[WINDOW:] + rhat
    hybrid_full = np.clip(hybrid_full, 0, None)
    print("[model] LSTM (residual) trained.", flush=True)
    K.clear_session()

    # --------------------------------------------------- LSTM-only on raw demand
    y_mean, y_std = y[:split].mean(), y[:split].std()
    y_n = (y - y_mean) / y_std
    Xy, yy = make_sequences(y_n, WINDOW)
    Xy_tr, yy_tr = Xy[:seq_split][..., None], yy[:seq_split]
    lstm_only = build_lstm(WINDOW)
    lstm_only.fit(Xy_tr, yy_tr, validation_split=0.1, epochs=EPOCHS,
                  batch_size=64, verbose=0, callbacks=[es])
    yhat_only_n = lstm_only.predict(Xy[..., None], verbose=0).flatten()
    lstm_only_full = np.empty(n)
    lstm_only_full[:WINDOW] = np.nan
    lstm_only_full[WINDOW:] = yhat_only_n * y_std + y_mean
    lstm_only_full = np.clip(lstm_only_full, 0, None)
    print("[model] LSTM-only trained.", flush=True)

    # ------------------------------------------------------------- test window
    test_slice = slice(split, n)
    actual = y[test_slice]
    preds = {
        "naive_24":     y[split - 24:n - 24],
        "naive_168":    y[split - 168:n - 168],
        "prophet_only": prophet_fit[test_slice],
        "lstm_only":    lstm_only_full[test_slice],
        "hybrid":       hybrid_full[test_slice],
    }

    metrics = {}
    for name, p in preds.items():
        metrics[name] = {"RMSE": round(rmse(actual, p), 2),
                         "MAE": round(mae(actual, p), 2)}

    best = min(metrics, key=lambda k: metrics[k]["RMSE"])
    print("\n[results] one-step-ahead, final 14 days")
    print(f"{'model':<14}{'RMSE':>9}{'MAE':>9}")
    for k, v in metrics.items():
        star = "  <-- best" if k == best else ""
        print(f"{k:<14}{v['RMSE']:>9}{v['MAE']:>9}{star}")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_METRICS, "w") as f:
        json.dump({"test_hours": TEST_HOURS, "window": WINDOW,
                   "metrics": metrics, "best": best}, f, indent=2)

    out = df.iloc[test_slice][["ds"]].copy()
    out["actual"] = actual
    for k, p in preds.items():
        out[k] = p
    out.to_csv(OUT_PRED, index=False)
    print(f"\n[results] wrote {OUT_METRICS} and {OUT_PRED}")


if __name__ == "__main__":
    main()
