# NYC Taxi Demand Forecasting

Spatiotemporal forecasting of hourly Yellow-taxi pickup demand across New York
City taxi zones. The pipeline ingests raw NYC TLC trip records with **PySpark**,
forecasts demand with a **Prophet + LSTM hybrid**, benchmarks it against honest
baselines, and visualises predicted demand on an interactive **Folium**
choropleth.

## Results (held-out final 14 days)

Two complementary views were evaluated, both one-step-ahead on data the models
never saw during training.

**Citywide hourly demand** (mean ≈ 4,358 pickups/hour)

| Model            | RMSE   | MAE    |
|------------------|--------|--------|
| naive (t−24h)    | 1313.3 | 873.7  |
| naive (t−168h)   | **534.4** | **343.3** |
| Prophet only     | 1112.1 | 871.5  |
| LSTM only        | 557.3  | 398.5  |
| **Prophet+LSTM hybrid** | 552.7  | 397.3  |

The hybrid clearly beats Prophet-only and LSTM-only. A weekly seasonal-naive
baseline (same hour last week) is a very strong competitor on this highly
regular aggregate series and edges the hybrid — a deliberate finding: simple
baselines must be respected, and reported, not hidden.

**Zone-level demand** (38 busiest zones, ≈ 90% of all trips; mean ≈ 110
pickups/zone/hour)

| Metric | Value |
|--------|-------|
| Pooled RMSE | **39.73** |
| Pooled MAE  | 27.38 |

Per-zone Prophet models drive the choropleth and give the genuine spatial error.

## Pipeline

```
data/raw/                      raw TLC parquet + zone lookup + zone shapefile
  ├── yellow_tripdata_2024-01.parquet
  ├── yellow_tripdata_2024-02.parquet
  ├── yellow_tripdata_2024-03.parquet
  ├── taxi_zone_lookup.csv
  └── taxi_zones.zip

src/
  ├── 01_aggregate_spark.py    PySpark: 9.5M trips -> hourly zone demand
  ├── 02_build_series.py       gap-free citywide hourly series
  ├── 03_train_hybrid.py       Prophet + LSTM hybrid + baselines + metrics
  └── 04_zone_map.py           per-zone Prophet + zone RMSE + Folium map

outputs/
  ├── metrics.json             citywide model comparison
  ├── zone_metrics.json        zone-level RMSE/MAE
  ├── test_predictions.csv     actual vs each model on the test window
  └── demand_map.html          interactive choropleth
```

## Method

1. **Aggregate (PySpark).** Read the raw parquet files, drop rows outside the
   study window or with invalid zone ids, truncate each pickup to the hour, and
   count trips per (zone, hour). ~9.5M rows collapse to ~237k.
2. **Build series.** Sum to a gap-free citywide hourly series (absent hours = 0).
3. **Hybrid model.** Prophet learns trend + daily/weekly seasonality
   (multiplicative, higher Fourier orders for sharp rush-hour peaks). An LSTM
   (one-week look-back) is trained on Prophet's *residuals*; the forecast is
   `prophet(t) + lstm_residual(t)`. Compared against naive and single-model
   baselines.
4. **Spatial.** A Prophet model per busy zone (scalable loop) produces the
   zone-level forecasts and the predicted-demand choropleth.

## Getting the data

Download from the NYC TLC site (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

* `yellow_tripdata_2024-01.parquet` … `2024-03.parquet`
* `taxi_zone_lookup.csv`
* `taxi_zones.zip`

Place them in `data/raw/` (unzip `taxi_zones.zip` so the `.shp` is reachable, or
let `04_zone_map.py`'s glob find it under `data/raw/zones_unz/`). To scale up,
add more monthly files — the Spark stage handles them unchanged.

## Run

```bash
pip install -r requirements.txt
export RAW_GLOB="data/raw/yellow_tripdata_*.parquet"
python src/01_aggregate_spark.py
python src/02_build_series.py
python src/03_train_hybrid.py
python src/04_zone_map.py
```

Requires Java (8/11/17/21) for PySpark.

## Notes & honest limitations

* Evaluation is one-step-ahead; multi-step (pure) forecasting is harder and the
  gap to seasonal-naive would change.
* Three months of data is enough for daily/weekly seasonality but not for yearly
  or holiday effects — add more months to extend.
* The long tail of low-volume zones is excluded from modelling; they carry
  little demand and mostly add noise.

## Streamlit dashboard

An interactive dashboard (`app.py`) presents the results: model comparison,
actual-vs-predicted, the demand choropleth, and a per-zone profile explorer.

It reads only the **precomputed artefacts**, so it needs no Spark/TensorFlow at
runtime and deploys on Streamlit Community Cloud.

Run locally:
```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

Deploy on Streamlit Community Cloud:
1. Push this repo to GitHub, including the generated `data/processed/*.csv` and
   `outputs/*` files (they are small and the app needs them).
2. Go to share.streamlit.io → New app → pick the repo.
3. Main file: `app.py`. Requirements file: `requirements-streamlit.txt`.
4. Deploy.

The raw parquet files are NOT needed by the app and can stay out of the repo.
