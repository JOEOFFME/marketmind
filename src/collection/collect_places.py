"""
Phase 2 — Data Collection
Fetches Google Places (New API) ratings for Rabat districts.
Usage: make collect-places
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import httpx
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text

from src.config import settings

NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"

RABAT_DISTRICTS = [
    {"name": "Agdal", "lat": 33.9989, "lng": -6.8504},
    {"name": "Hassan", "lat": 34.0131, "lng": -6.8326},
    {"name": "Hay Riad", "lat": 33.9650, "lng": -6.8589},
    {"name": "Medina", "lat": 34.0242, "lng": -6.8344},
    {"name": "Souissi", "lat": 33.9875, "lng": -6.8217},
    {"name": "L'Ocean", "lat": 34.0089, "lng": -6.8506},
    {"name": "Akkari", "lat": 34.0156, "lng": -6.8634},
    {"name": "Youssoufia", "lat": 33.9823, "lng": -6.8712},
    {"name": "Yacoub El Mansour", "lat": 33.9920, "lng": -6.8650},
    {"name": "Takaddoum", "lat": 33.9760, "lng": -6.8240},
    {"name": "Aviation", "lat": 33.9890, "lng": -6.8360},
    {"name": "Hay El Fath", "lat": 33.9930, "lng": -6.8580},
]

PLACE_TYPES = [
    "restaurant",
    "cafe",
    "supermarket",
    "pharmacy",
    "bank",
    "shopping_mall",
    "bakery",
    "clothing_store",
    "electronics_store",
    "beauty_salon",
    "gym",
    "hotel",
    "movie_theater",
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
                        "latitude": district["lat"],
                        "longitude": district["lng"],
                    },
                    "radius": 1200.0,
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
                results.append(
                    {
                        "place_id": place.get("id", ""),
                        "name": place.get("displayName", {}).get("text", ""),
                        "district": district["name"],
                        "place_type": place_type,
                        "rating": place.get("rating", None),
                        "review_count": place.get("userRatingCount", 0),
                        "price_level": place.get("priceLevel", None),
                        "latitude": place.get("location", {}).get("latitude", None),
                        "longitude": place.get("location", {}).get("longitude", None),
                    }
                )
            time.sleep(0.2)

        except Exception as e:
            logger.warning(f"Failed {place_type} in {district['name']}: {e}")

    return results


def compute_success_score(df: pd.DataFrame) -> pd.DataFrame:
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Success target = Wilson score lower bound (95% CI) of the rating.
    # We treat the 1-5 rating as a "satisfaction proportion" p in [0, 1] and the
    # review_count as the sample size n, then take the *lower* bound of the
    # confidence interval. This is deliberately conservative: a 5-star place with
    # 2 reviews stays low (unproven), while a bad rating backed by many reviews is
    # confidently low (negative reviews ARE counted). No reviews -> no signal -> 0.
    z = 1.96  # 95% confidence
    n = df["review_count"]
    p = ((df["rating"] - 1) / 4).clip(0, 1)  # normalized rating as a proportion

    with np.errstate(divide="ignore", invalid="ignore"):
        center = p + z**2 / (2 * n)
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        lower = (center - margin) / (1 + z**2 / n)

    # n == 0 yields NaN above -> no reviews means no evidence -> score 0.
    df["success_score"] = (np.maximum(lower.fillna(0), 0) * 100).round(2)

    # Keep raw rating & volume as separate ML features so the model can still
    # learn from both dimensions independently of the (clean) target.
    df["log_reviews"] = np.log1p(df["review_count"])
    df["rating_norm"] = p
    df["reviews_norm"] = df["log_reviews"] / df["log_reviews"].max()

    return df


def save_to_db(df: pd.DataFrame) -> None:
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        conn.execute(
            text(
                """
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
        """
            )
        )
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
    top_cols = ["name", "district", "rating", "review_count", "success_score"]
    top10 = df.nlargest(10, "success_score")[top_cols].to_string()
    logger.info(f"\nTop 10 by success score:\n{top10}")

    save_to_db(df)
    logger.info("=== Google Places collection complete ===")


if __name__ == "__main__":
    main()
