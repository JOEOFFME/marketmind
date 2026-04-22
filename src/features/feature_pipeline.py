"""
Phase 2 — Feature Engineering
Joins raw_pois + places_ratings + weather_observations
and computes ML-ready features for each rated location.

Usage:
    poetry run python src/features/feature_pipeline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text
from loguru import logger

from src.config import settings


def load_data(engine) -> tuple:
    logger.info("Loading data from database...")

    places = pd.read_sql("SELECT * FROM places_ratings", engine)
    pois   = pd.read_sql("SELECT * FROM raw_pois", engine)
    weather = pd.read_sql("SELECT * FROM weather_observations", engine)

    logger.info(f"  places_ratings : {len(places):,} rows")
    logger.info(f"  raw_pois       : {len(pois):,} rows")
    logger.info(f"  weather        : {len(weather):,} rows")

    return places, pois, weather


def compute_proximity_features(places: pd.DataFrame, pois: pd.DataFrame) -> pd.DataFrame:
    """
    For each rated place, compute distances to key POI types.
    Uses vectorized haversine — no PostGIS needed here.
    """
    logger.info("Computing proximity features...")

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Filter POI subsets
    pharmacies    = pois[pois["amenity"] == "pharmacy"]
    banks         = pois[pois["amenity"] == "bank"]
    supermarkets  = pois[pois["shop"]    == "supermarket"]
    cafes         = pois[pois["amenity"] == "cafe"]
    restaurants   = pois[pois["amenity"] == "restaurant"]
    fuel          = pois[pois["amenity"] == "fuel"]

    results = []
    for _, place in places.iterrows():
        lat, lon = place["latitude"], place["longitude"]

        def min_dist(subset):
            if subset.empty:
                return 9999.0
            dists = haversine(lat, lon, subset["latitude"].values, subset["longitude"].values)
            return float(dists.min())

        def count_within(subset, radius=500):
            if subset.empty:
                return 0
            dists = haversine(lat, lon, subset["latitude"].values, subset["longitude"].values)
            return int((dists <= radius).sum())

        results.append({
            "place_id":                place["place_id"],
            # Distance to nearest key amenity (meters)
            "dist_nearest_pharmacy":   min_dist(pharmacies),
            "dist_nearest_bank":       min_dist(banks),
            "dist_nearest_supermarket":min_dist(supermarkets),
            "dist_nearest_fuel":       min_dist(fuel),
            # Count within 500m radius
            "cafes_500m":              count_within(cafes),
            "restaurants_500m":        count_within(restaurants),
            "pharmacies_500m":         count_within(pharmacies),
            "banks_500m":              count_within(banks),
            "total_pois_500m":         count_within(pois),
        })

    proximity_df = pd.DataFrame(results)
    logger.info(f"  Computed proximity features for {len(proximity_df):,} places")
    return proximity_df


def compute_district_features(places: pd.DataFrame) -> pd.DataFrame:
    """District-level aggregated features."""
    logger.info("Computing district features...")

    district_stats = places.groupby("district").agg(
        district_avg_rating    =("rating", "mean"),
        district_avg_reviews   =("review_count", "mean"),
        district_place_count   =("place_id", "count"),
        district_avg_score     =("success_score", "mean"),
    ).reset_index()

    return places.merge(district_stats, on="district", how="left")


def compute_competition_features(places: pd.DataFrame) -> pd.DataFrame:
    """Competition density per type per district."""
    logger.info("Computing competition features...")

    type_counts = places.groupby(["district", "place_type"]).size().reset_index(name="type_count_in_district")
    places = places.merge(type_counts, on=["district", "place_type"], how="left")

    # Is this place above average for its type in its district?
    type_avg = places.groupby(["district", "place_type"])["success_score"].mean().reset_index(name="type_avg_score")
    places = places.merge(type_avg, on=["district", "place_type"], how="left")
    places["above_type_avg"] = (places["success_score"] > places["type_avg_score"]).astype(int)

    return places


def compute_weather_features(weather: pd.DataFrame) -> dict:
    """Aggregate weather into seasonal features."""
    logger.info("Computing weather features...")

    weather["date"] = pd.to_datetime(weather["date"])
    weather["month"] = weather["date"].dt.month

    monthly = weather.groupby("month").agg(
        avg_temp_max   =("temp_max", "mean"),
        avg_temp_min   =("temp_min", "mean"),
        avg_precip     =("precipitation", "mean"),
        avg_wind       =("wind_speed", "mean"),
    ).reset_index()

    # Peak tourist months (Apr-May, Sep-Oct) = higher foot traffic
    monthly["is_peak_season"] = monthly["month"].isin([4, 5, 9, 10]).astype(int)
    # Ramadan indicator (approximate — varies yearly)
    monthly["is_ramadan_period"] = monthly["month"].isin([3, 4]).astype(int)

    return {
        "avg_annual_temp":    weather["temp_max"].mean(),
        "rainy_days_per_year": (weather["precipitation"] > 1.0).sum(),
        "peak_season_months": 4,
    }


def build_feature_table(places, pois, weather) -> pd.DataFrame:
    logger.info("Building feature table...")

    # District features
    df = compute_district_features(places)

    # Competition features
    df = compute_competition_features(df)

    # Proximity features
    proximity = compute_proximity_features(df, pois)
    df = df.merge(proximity, on="place_id", how="left")

    # Weather summary (scalar features — same for all Rabat locations)
    weather_feats = compute_weather_features(weather)
    for k, v in weather_feats.items():
        df[k] = v

    # Price level encoding
    price_map = {
        "PRICE_LEVEL_FREE": 0,
        "PRICE_LEVEL_INEXPENSIVE": 1,
        "PRICE_LEVEL_MODERATE": 2,
        "PRICE_LEVEL_EXPENSIVE": 3,
        "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    }
    df["price_level_encoded"] = df["price_level"].map(price_map).fillna(1)

    # Place type encoding
    df = pd.get_dummies(df, columns=["place_type"], prefix="type")

    # District encoding
    df = pd.get_dummies(df, columns=["district"], prefix="district")

    # Log-transform skewed features
    df["log_review_count"] = np.log1p(df["review_count"])
    df["log_total_pois_500m"] = np.log1p(df["total_pois_500m"])

    logger.info(f"Feature table: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def save_features(df: pd.DataFrame, engine) -> None:
    # Save full feature table
    df.to_sql("features", engine, if_exists="replace", index=False)
    logger.success(f"Saved {len(df):,} rows → features table")

    # Also save as parquet for ML training
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/processed/features.parquet", index=False)
    logger.success("Saved → data/processed/features.parquet")


def main():
    engine = create_engine(settings.database_url)

    places, pois, weather = load_data(engine)
    df = build_feature_table(places, pois, weather)

    # Print feature summary
    logger.info(f"\nFeature columns ({len(df.columns)}):\n{list(df.columns)}")
    logger.info(f"\nTarget variable summary:\n{df['success_score'].describe()}")

    save_features(df, engine)
    logger.info("=== Feature engineering complete ===")


if __name__ == "__main__":
    main()
