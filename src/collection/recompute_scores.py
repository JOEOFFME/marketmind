"""
Recompute `success_score` in place from data already stored in the database.

Use this after changing the scoring formula in `compute_success_score`: it
reloads `places_ratings`, recomputes the score from the existing `rating` and
`review_count` columns, and writes the table back — WITHOUT calling the Google
Places API (so it costs no quota).

Usage: poetry run python src/collection/recompute_scores.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

from src.collection.collect_places import compute_success_score, save_to_db
from src.config import settings

# Columns derived by compute_success_score; dropped before recomputing so they
# are regenerated cleanly instead of being duplicated.
DERIVED_COLS = ["success_score", "log_reviews", "rating_norm", "reviews_norm"]


def main() -> None:
    engine = create_engine(settings.database_url)
    df = pd.read_sql("SELECT * FROM places_ratings", engine)
    logger.info(f"Loaded {len(df)} places from places_ratings")

    before = df["success_score"].describe() if "success_score" in df else None

    df = df.drop(columns=[c for c in DERIVED_COLS if c in df.columns])
    df = compute_success_score(df)

    save_to_db(df)
    logger.success(f"Recomputed success_score for {len(df)} places")

    if before is not None:
        logger.info(f"\nOld success_score:\n{before}")
    logger.info(f"\nNew success_score:\n{df['success_score'].describe()}")
    top = df.nlargest(10, "success_score")[
        ["name", "district", "rating", "review_count", "success_score"]
    ].to_string(index=False)
    logger.info(f"\nTop 10 by new success_score:\n{top}")


if __name__ == "__main__":
    main()

    