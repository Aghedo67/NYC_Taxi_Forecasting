"""
Stage 2 — Build model-ready time series from the aggregated demand table.

Produces two artefacts:
  * citywide_hourly.csv  — total pickups per hour on a gap-free hourly index
                           (the series the hybrid model is trained on)
  * zone_hourly_demand   — passed through; used later for the spatial map

Missing (zone, hour) combinations mean zero recorded pickups, so when we build
the continuous index we fill absent hours with 0 rather than dropping them.
"""
import os
import pandas as pd

IN_CSV = "data/processed/zone_hourly_demand.csv"
OUT_CITY = "data/processed/citywide_hourly.csv"


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["pickup_hour"])
    print(f"[series] zone-hour rows: {len(df):,}  | zones: {df.zone_id.nunique()}")

    # Citywide hourly demand on a complete, gap-free hourly index.
    city = (
        df.groupby("pickup_hour")["demand"].sum().rename("y").reset_index()
        .rename(columns={"pickup_hour": "ds"})
    )
    full_idx = pd.date_range(city.ds.min(), city.ds.max(), freq="h")
    city = (
        city.set_index("ds").reindex(full_idx, fill_value=0)
        .rename_axis("ds").reset_index()
    )
    print(f"[series] citywide hourly points: {len(city):,} "
          f"({city.ds.min()} -> {city.ds.max()})")
    print(f"[series] mean hourly demand: {city.y.mean():.1f} | peak: {city.y.max():,}")

    os.makedirs(os.path.dirname(OUT_CITY), exist_ok=True)
    city.to_csv(OUT_CITY, index=False)
    print(f"[series] wrote {OUT_CITY}")


if __name__ == "__main__":
    main()
