"""
NYC Taxi Demand Forecasting — Streamlit dashboard.

This app is a *presentation* layer. It reads the artefacts produced by the
offline pipeline (PySpark aggregation + Prophet/LSTM training) and makes them
interactive. It deliberately does NOT run Spark or train models at request
time, so it deploys cleanly on Streamlit Community Cloud with only pandas /
numpy / plotly.

Expected files (created by src/01..04):
  data/processed/citywide_hourly.csv
  data/processed/zone_hourly_demand.csv
  data/raw/taxi_zone_lookup.csv
  outputs/metrics.json
  outputs/zone_metrics.json
  outputs/test_predictions.csv
  outputs/demand_map.html
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

st.set_page_config(page_title="NYC Taxi Demand Forecasting",
                   page_icon="🚕", layout="wide")


# ----------------------------------------------------------------- loaders
@st.cache_data
def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


@st.cache_data
def load_csv(p, **kw):
    return pd.read_csv(p, **kw) if Path(p).exists() else None


metrics = load_json(OUT / "metrics.json")
zone_metrics = load_json(OUT / "zone_metrics.json")
preds = load_csv(OUT / "test_predictions.csv", parse_dates=["ds"])
zone_hourly = load_csv(PROC / "zone_hourly_demand.csv", parse_dates=["pickup_hour"])
lookup = load_csv(RAW / "taxi_zone_lookup.csv")
forecasts = load_csv(OUT / "forecasts.csv", parse_dates=["ds"])


# ----------------------------------------------------------------- header
st.title("🚕 NYC Taxi Demand Forecasting")
st.caption(
    "Spatiotemporal forecasting of hourly Yellow-taxi pickups across NYC zones — "
    "PySpark aggregation, a Prophet + LSTM hybrid, and per-zone Prophet models."
)

# KPI row
c1, c2, c3, c4 = st.columns(4)
if zone_metrics:
    c1.metric("Zone-level RMSE", zone_metrics["zone_level_RMSE"])
    c4.metric("Zones modelled",
              f'{zone_metrics["modelled_zones"]}  ({zone_metrics["coverage"]:.0%} of trips)')
if metrics:
    h = metrics["metrics"]["hybrid"]
    c2.metric("Citywide hybrid RMSE", h["RMSE"])
    c3.metric("Citywide hybrid MAE", h["MAE"])

st.divider()

tab_perf, tab_fc, tab_map, tab_zone, tab_about = st.tabs(
    ["📊 Model performance", "🔮 Forecast", "🗺️ Demand map",
     "🔍 Zone explorer", "ℹ️ About"]
)

# ----------------------------------------------------------------- performance
with tab_perf:
    if metrics is None or preds is None:
        st.warning("Run the pipeline first (src/03_train_hybrid.py) to generate metrics.")
    else:
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Model comparison")
            st.caption("One-step-ahead, held-out final 14 days. Lower is better.")
            m = metrics["metrics"]
            mdf = (pd.DataFrame(m).T.reset_index()
                   .rename(columns={"index": "model"})
                   .sort_values("RMSE", ascending=True))
            fig = px.bar(mdf, x="RMSE", y="model", orientation="h",
                         text="RMSE", color="model",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, height=320,
                              yaxis_title="", xaxis_title="RMSE (pickups/hour)")
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "The hybrid beats Prophet-only and LSTM-only. A weekly "
                "seasonal-naive baseline (same hour last week) is a strong "
                "competitor on this regular aggregate series — reported "
                "honestly rather than hidden."
            )

        with right:
            st.subheader("Actual vs predicted")
            model_cols = [c for c in preds.columns if c not in ("ds", "actual")]
            chosen = st.multiselect("Models to overlay", model_cols,
                                    default=["hybrid", "naive_168"])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=preds.ds, y=preds.actual,
                                      name="actual", line=dict(color="#222", width=2)))
            for col in chosen:
                fig2.add_trace(go.Scatter(x=preds.ds, y=preds[col], name=col,
                                          opacity=0.8))
            fig2.update_layout(height=360, xaxis_title="",
                               yaxis_title="Pickups / hour",
                               legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------- forecast
with tab_fc:
    st.subheader("Forecast future demand")
    if forecasts is None:
        st.warning(
            "No forecasts found. Generate them offline with "
            "`python src/batch_forecast.py` and commit `outputs/forecasts.csv`."
        )
    else:
        ctrl, _ = st.columns([2, 1])
        with ctrl:
            level = st.radio("Level", ["City", "Zone"], horizontal=True)

        if level == "City":
            fdf = forecasts[forecasts.level == "city"].copy()
            title = "Citywide"
        else:
            zf = forecasts[forecasts.level == "zone"]
            zids = sorted(zf.zone_id.unique())
            names = {}
            if lookup is not None:
                lk = lookup.set_index("LocationID")["Zone"]
                names = {z: lk.get(z, "Unknown") for z in zids}
            pick = st.selectbox(
                "Zone", zids,
                format_func=lambda z: f"{names.get(z, 'Zone')} (#{z})")
            fdf = zf[zf.zone_id == pick].copy()
            title = f"{names.get(pick, 'Zone')} (#{pick})"

        fdf = fdf.sort_values("ds")
        max_h = len(fdf)
        hours = st.slider("Hours ahead", min_value=6,
                          max_value=int(max_h), value=min(72, int(max_h)), step=6)
        view = fdf.head(hours)

        # plot: prediction line with shaded uncertainty band
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(view.ds) + list(view.ds[::-1]),
            y=list(view.yhat_upper) + list(view.yhat_lower[::-1]),
            fill="toself", fillcolor="rgba(255,140,0,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="uncertainty",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=view.ds, y=view.yhat, name="forecast",
            line=dict(color="#e8590c", width=2)))
        fig.update_layout(height=420, xaxis_title="",
                          yaxis_title="Predicted pickups / hour",
                          legend=dict(orientation="h"),
                          title=f"{title} — next {hours}h")
        st.plotly_chart(fig, use_container_width=True)

        peak = view.loc[view.yhat.idxmax()]
        k1, k2, k3 = st.columns(3)
        k1.metric("Predicted peak", f"{peak.yhat:.0f} /hr")
        k2.metric("Peak time", peak.ds.strftime("%a %H:%M"))
        k3.metric("Avg over window", f"{view.yhat.mean():.0f} /hr")

        st.caption(
            "Forecasts are Prophet projections (trend + daily/weekly seasonality) "
            "with 80% uncertainty intervals, precomputed by `src/batch_forecast.py`. "
            "The shaded band is the predicted range."
        )
        st.download_button(
            "⬇️ Download this forecast (CSV)",
            view.to_csv(index=False).encode(),
            file_name=f"forecast_{title.split()[0].lower()}.csv",
            mime="text/csv")

# ----------------------------------------------------------------- map
with tab_map:
    st.subheader("Predicted mean hourly demand by zone")
    map_path = OUT / "demand_map.html"
    if map_path.exists():
        st.components.v1.html(map_path.read_text(), height=600, scrolling=False)
        st.caption("Choropleth of per-zone Prophet predictions over the held-out fortnight. "
                   "Hover a zone for its name, borough and predicted pickups/hour.")
    else:
        st.warning("demand_map.html not found. Run src/04_zone_map.py to generate it.")

# ----------------------------------------------------------------- zone explorer
with tab_zone:
    st.subheader("Demand profile by zone")
    if zone_hourly is None or lookup is None:
        st.warning("Run src/01_aggregate_spark.py to generate zone_hourly_demand.csv.")
    else:
        lk = lookup.rename(columns={"LocationID": "zone_id"})
        totals = (zone_hourly.groupby("zone_id")["demand"].sum()
                  .sort_values(ascending=False))
        opts = (lk.set_index("zone_id")["Zone"].reindex(totals.index)
                .fillna("Unknown"))
        labels = {zid: f"{opts[zid]} (#{zid})" for zid in totals.index}
        pick = st.selectbox("Zone (busiest first)", list(totals.index),
                            format_func=lambda z: labels[z])

        s = zone_hourly[zone_hourly.zone_id == pick].copy()
        s["hour"] = s.pickup_hour.dt.hour
        s["dow"] = s.pickup_hour.dt.dayofweek

        a, b = st.columns(2)
        with a:
            byhr = s.groupby("hour")["demand"].mean().reindex(range(24), fill_value=0)
            fig3 = px.line(x=byhr.index, y=byhr.values, markers=True)
            fig3.update_layout(title="Average demand by hour of day", height=300,
                               xaxis_title="Hour", yaxis_title="Avg pickups")
            st.plotly_chart(fig3, use_container_width=True)
        with b:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            bydow = s.groupby("dow")["demand"].mean().reindex(range(7), fill_value=0)
            fig4 = px.bar(x=days, y=bydow.values)
            fig4.update_layout(title="Average demand by day of week", height=300,
                               xaxis_title="", yaxis_title="Avg pickups")
            st.plotly_chart(fig4, use_container_width=True)

        fig5 = px.line(s.sort_values("pickup_hour"), x="pickup_hour", y="demand")
        fig5.update_layout(title="Hourly demand over the full period", height=300,
                           xaxis_title="", yaxis_title="Pickups")
        st.plotly_chart(fig5, use_container_width=True)

# ----------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
        ### How this works
        The heavy lifting happens **offline** in the pipeline (`src/01..04`):
        PySpark collapses ~9.5M raw trips into hourly zone demand, a Prophet +
        LSTM hybrid forecasts citywide demand, and a per-zone Prophet loop drives
        the choropleth. This dashboard simply reads the resulting artefacts, so it
        stays light enough to deploy on Streamlit Community Cloud.

        **Data:** NYC TLC Yellow-taxi trip records, Jan–Mar 2024.

        **Headline results (held-out final 14 days):**
        - Citywide hybrid — RMSE ≈ 553, MAE ≈ 397 (~9% MAPE)
        - Zone-level — RMSE ≈ 39.7, MAE ≈ 27.4
        """
    )
    if metrics:
        st.json(metrics, expanded=False)
    if zone_metrics:
        st.json(zone_metrics, expanded=False)
