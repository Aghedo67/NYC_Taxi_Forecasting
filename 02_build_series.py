"""
Stage 1 — Scalable aggregation with PySpark.

Reads the raw NYC TLC Yellow-taxi Parquet files (millions of rows each) and
collapses them into a compact hourly pickup-count table per taxi zone. Doing the
heavy lifting in Spark means the pipeline scales to many months / the full
fleet without ever loading raw trips into pandas.

Input : data/raw/yellow_tripdata_YYYY-MM.parquet  (one or more)
Output: data/processed/zone_hourly_demand.csv      (zone x hour pickup counts)
"""
import glob
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- paths ---------------------------------------------------------------
RAW_GLOB = os.environ.get("RAW_GLOB", "data/raw/yellow_tripdata_*.parquet")
OUT_CSV = "data/processed/zone_hourly_demand.csv"

# Study window. Pickup timestamps outside this range are sensor/garbage rows
# (the TLC files always contain a few stray dates from other years).
START = "2024-01-01"
END = "2024-04-01"  # exclusive upper bound (Jan–Mar 2024)


def build_spark():
    return (
        SparkSession.builder.appName("nyc_taxi_demand_aggregation")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {RAW_GLOB}")
    print(f"[spark] reading {len(files)} file(s):")
    for f in files:
        print("   -", os.path.basename(f))

    df = spark.read.parquet(*files)
    raw_count = df.count()
    print(f"[spark] raw trip rows: {raw_count:,}")

    # --- clean -----------------------------------------------------------
    # Keep only valid pickups inside the study window with a real zone id.
    # Valid TLC zone ids are 1..263 (264/265 are 'Unknown'/'N/A').
    df = (
        df.select(
            F.col("tpep_pickup_datetime").alias("pickup"),
            F.col("PULocationID").alias("zone_id"),
        )
        .where(F.col("pickup") >= F.lit(START))
        .where(F.col("pickup") < F.lit(END))
        .where((F.col("zone_id") >= 1) & (F.col("zone_id") <= 263))
    )

    # Truncate each pickup to the top of its hour, then count trips.
    df = df.withColumn("pickup_hour", F.date_trunc("hour", F.col("pickup")))
    agg = (
        df.groupBy("zone_id", "pickup_hour")
        .agg(F.count(F.lit(1)).alias("demand"))
        .orderBy("zone_id", "pickup_hour")
    )

    kept = agg.agg(F.sum("demand")).collect()[0][0]
    print(f"[spark] trips kept after cleaning: {kept:,}")
    print(f"[spark] aggregated rows (zone x hour): {agg.count():,}")

    # Small enough now for a single CSV; coalesce to one part file.
    pdf = agg.toPandas()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pdf.to_csv(OUT_CSV, index=False)
    print(f"[spark] wrote {OUT_CSV}  ({len(pdf):,} rows)")

    spark.stop()


if __name__ == "__main__":
    main()
