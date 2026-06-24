"""
Stage 4 — Spatial forecasting and visualisation.

Fits a Prophet model per taxi zone (a scalable, embarrassingly-parallel loop)
to forecast hourly pickup demand, evaluates them pooled on the held-out final
14 days to get a genuine *zone-level* RMSE, and renders the predicted demand as
an interactive Folium choropleth of NYC.

To keep runtime sane we model the busiest zones that together account for the
large majority of trips; the long tail of near-empty zones is left out of the
modelled set (they carry almost no demand and add noise, not signal).

Outputs:
  outputs/zone_metrics.json   — pooled zone-level RMSE/MAE + coverage
  outputs/demand_map.html     — interactive choropleth of predicted demand
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet
import geopandas as gpd
import folium

DEMAND_CSV = "data/processed/zone_hourly_demand.csv"
LOOKUP_CSV = "data/raw/taxi_zone_lookup.csv"
SHP_GLOB = "data/raw/zones_unz/**/*.shp"

TEST_HOURS = 24 * 14
COVERAGE = 0.90   # model the busiest zones covering this share of all trips


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    df = pd.read_csv(DEMAND_CSV, parse_dates=["pickup_hour"])
    full_idx = pd.date_range(df.pickup_hour.min(), df.pickup_hour.max(), freq="h")

    # Rank zones by total demand; keep the busiest covering COVERAGE of trips.
    totals = df.groupby("zone_id")["demand"].sum().sort_values(ascending=False)
    cum = totals.cumsum() / totals.sum()
    modelled = cum[cum <= COVERAGE].index.tolist()
    if not modelled:
        modelled = [totals.index[0]]
    print(f"[zones] {len(modelled)} zones modelled "
          f"(of {df.zone_id.nunique()}), covering "
          f"{totals[modelled].sum() / totals.sum():.1%} of trips")

    split = len(full_idx) - TEST_HOURS
    all_actual, all_pred = [], []
    pred_mean_test = {}      # zone_id -> mean predicted demand over test window
    per_zone_rmse = {}

    for i, zid in enumerate(modelled, 1):
        s = (df[df.zone_id == zid].set_index("pickup_hour")["demand"]
             .reindex(full_idx, fill_value=0))
        z = pd.DataFrame({"ds": full_idx, "y": s.values})
        train = z.iloc[:split]

        m = Prophet(daily_seasonality=12, weekly_seasonality=8,
                    yearly_seasonality=False, seasonality_mode="multiplicative",
                    changepoint_prior_scale=0.1)
        m.fit(train)
        fc = m.predict(z[["ds"]])["yhat"].clip(lower=0).values

        actual = z.y.values[split:]
        pred = fc[split:]
        all_actual.append(actual)
        all_pred.append(pred)
        per_zone_rmse[int(zid)] = round(rmse(actual, pred), 2)
        pred_mean_test[int(zid)] = float(pred.mean())

        if i % 15 == 0:
            print(f"[zones] fitted {i}/{len(modelled)}", flush=True)

    all_actual = np.concatenate(all_actual)
    all_pred = np.concatenate(all_pred)
    zone_rmse = rmse(all_actual, all_pred)
    zone_mae = mae(all_actual, all_pred)
    mean_demand = float(all_actual.mean())
    print(f"\n[zones] pooled zone-level RMSE: {zone_rmse:.2f} | "
          f"MAE: {zone_mae:.2f} | mean zone-hour demand: {mean_demand:.1f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/zone_metrics.json", "w") as f:
        json.dump({
            "modelled_zones": len(modelled),
            "coverage": round(float(totals[modelled].sum() / totals.sum()), 4),
            "zone_level_RMSE": round(zone_rmse, 2),
            "zone_level_MAE": round(zone_mae, 2),
            "mean_zone_hour_demand": round(mean_demand, 2),
            "per_zone_rmse_sample": dict(list(per_zone_rmse.items())[:10]),
        }, f, indent=2)

    # --- choropleth -----------------------------------------------------
    shp = glob.glob(SHP_GLOB, recursive=True)[0]
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    gdf["LocationID"] = gdf["LocationID"].astype(int)
    gdf["pred_demand"] = gdf["LocationID"].map(pred_mean_test)
    plot = gdf.dropna(subset=["pred_demand"]).copy()

    centre = [40.75, -73.95]
    fmap = folium.Map(location=centre, zoom_start=11, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=plot.to_json(),
        data=plot,
        columns=["LocationID", "pred_demand"],
        key_on="feature.properties.LocationID",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        nan_fill_color="lightgray",
        legend_name="Predicted mean hourly pickups (held-out fortnight)",
    ).add_to(fmap)

    folium.GeoJson(
        plot.to_json(),
        style_function=lambda _: {"fillOpacity": 0, "color": "transparent"},
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "borough", "pred_demand"],
            aliases=["Zone:", "Borough:", "Pred. pickups/hr:"],
            localize=True,
        ),
    ).add_to(fmap)

    fmap.save("outputs/demand_map.html")
    print("[zones] wrote outputs/demand_map.html and outputs/zone_metrics.json")


if __name__ == "__main__":
    main()
