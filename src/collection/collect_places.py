"""
Phase 2 — Data Collection
Fetches Google Places (New API) ratings for Rabat districts.
Usage: make collect-places
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
import httpx
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger

from src.config import settings

NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"

RABAT_DISTRICTS = [
    {"name": "Agdal",      "lat": 33.9989, "lng": -6.8504},
    {"name": "Hassan",     "lat": 34.0131, "lng": -6.8326},
    {"name": "Hay Riad",   "lat": 33.9650, "lng": -6.8589},
    {"name": "Medina",     "lat": 34.0242, "lng": -6.8344},
    {"name": "Souissi",    "lat": 33.9875, "lng": -6.8217},
    {"name": "L'Ocean",    "lat": 34.0089, "lng": -6.8506},
    {"name": "Akkari",     "lat": 34.0156, "lng": -6.8634},
    {"name": "Youssoufia", "lat": 33.9823, "lng": -6.8712},
]

PLACE_TYPES = [
    "restaurant", "cafe", "supermarket",
    "pharmacy", "bank", "shopping_mall",
]


def fetch_district(district: dict) -> list:
    results = []
    for place_type in PLACE_TYPES:
        payload = {
            "includedTypes": [place_type],
            "maxResultCount": 20,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude":  district["lat"],
                        "longitude": district["lng"],
                    },
                    "radius": 800.0,
                }
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": settings.google_places_api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.rating,"
                "places.userRatingCount,places.priceLevel,"
                "places.location,places.primaryType"
            ),
        }
        try:
            resp = httpx.post(NEARBY_URL, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            for place in data.get("places", []):
                results.append({
                    "place_id":     place.get("id", ""),
                    "name":         place.get("displayName", {}).get("text", ""),
                    "district":     district["name"],
                    "place_type":   place_type,
                    "rating":       place.get("rating", None),
                    "review_count": place.get("userRatingCount", 0),
                    "price_level":  place.get("priceLevel", None),
                    "latitude":     place.get("location", {}).get("latitude", None),
                    "longitude":    place.get("location", {}).get("longitude", None),
                })
            time.sleep(0.2)

        except Exception as e:
            logger.warning(f"Failed {place_type} in {district['name']}: {e}")

    return results


def compute_success_score(df: pd.DataFrame) -> pd.DataFrame:
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df["rating"]       = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"]       = df["rating"].fillna(df["rating"].median())

    df["log_reviews"]  = np.log1p(df["review_count"])
    df["rating_norm"]  = (df["rating"] - 1) / 4
    df["reviews_norm"] = df["log_reviews"] / df["log_reviews"].max()

    df["success_score"] = (
        0.4 * df["rating_norm"] +
        0.6 * df["reviews_norm"]
    ) * 100

    return df


def save_to_db(df: pd.DataFrame) -> None:
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS places_ratings (
                id            SERIAL PRIMARY KEY,
                place_id      TEXT UNIQUE,
                name          TEXT,
                district      TEXT,
                place_type    TEXT,
                rating        FLOAT,
                review_count  INTEGER,
                price_level   TEXT,
                latitude      FLOAT,
                longitude     FLOAT,
                success_score FLOAT,
                collected_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()

    df.to_sql("places_ratings", engine, if_exists="replace", index=False)
    logger.success(f"Saved {len(df):,} rows → places_ratings")


def main():
    if not settings.google_places_api_key:
        logger.error("GOOGLE_PLACES_API_KEY not set in .env")
        sys.exit(1)

    logger.info("=== Google Places (New API) collection starting ===")
    all_results = []

    for district in RABAT_DISTRICTS:
        logger.info(f"Fetching {district['name']}...")
        results = fetch_district(district)
        all_results.extend(results)
        logger.info(f"  → {len(results)} places found")
        time.sleep(0.5)

    if not all_results:
        logger.error("No results returned — check your API key and billing")
        sys.exit(1)

    df = pd.DataFrame(all_results).drop_duplicates(subset="place_id")
    logger.info(f"Total unique places: {len(df):,}")

    df = compute_success_score(df)
    logger.info(
        f"\nTop 10 by success score:\n"
        f"{df.nlargest(10, 'success_score')[['name','district','rating','review_count','success_score']].to_string()}"
    )

    save_to_db(df)
    logger.info("=== Google Places collection complete ===")


if __name__ == "__main__":
    main()
