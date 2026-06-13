"""
Synthetic dataset generator for MarketMind (development / demo only).

⚠️  The data produced here is SIMULATED, not real Google/OSM data. It is meant
to exercise the full pipeline and let the model learn an honest signal for demos
and testing. Do NOT present these numbers as real market evidence.

Design: every place gets interpretable *market* features (district income,
foot traffic, competition density, accessibility, price, opening days). A latent
"quality" and "popularity" are real functions of those features (+ realistic
noise), which drive the Google-style `rating` and `review_count`. The target
`success_score` is then the Wilson lower bound of that rating. Because the target
is a genuine (noisy) function of the market features — and NOT of any leaked
column — a model can legitimately learn it.

Writes data/processed/features.parquet (consumed by train.py and the agents) and,
if a database is configured, the `features` and `places_ratings` tables.

Usage: poetry run python src/collection/generate_synthetic_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from loguru import logger

SEED = 42
OUT_PARQUET = Path("data/processed/features.parquet")

# District: center coords + latent desirability drivers (0..1).
#   income   : purchasing power of the area
#   footfall : pedestrian / tourist traffic
#   density  : urban density (affects competition & how close amenities are)
DISTRICTS = {
    "Agdal":      dict(lat=33.9989, lng=-6.8504, income=0.80, footfall=0.85, density=0.70),
    "Hay Riad":   dict(lat=33.9650, lng=-6.8589, income=0.90, footfall=0.70, density=0.50),
    "Souissi":    dict(lat=33.9875, lng=-6.8217, income=0.95, footfall=0.50, density=0.30),
    "Hassan":     dict(lat=34.0131, lng=-6.8326, income=0.60, footfall=0.80, density=0.80),
    "Medina":     dict(lat=34.0242, lng=-6.8344, income=0.45, footfall=0.95, density=0.95),
    "L'Ocean":    dict(lat=34.0089, lng=-6.8506, income=0.50, footfall=0.60, density=0.70),
    "Akkari":     dict(lat=34.0156, lng=-6.8634, income=0.35, footfall=0.55, density=0.75),
    "Youssoufia": dict(lat=33.9823, lng=-6.8712, income=0.40, footfall=0.50, density=0.70),
}

# Type: base popularity (drives review volume) and typical opening days/week.
TYPES = {
    "restaurant":    dict(popularity=0.90, open_days=6.5),
    "cafe":          dict(popularity=0.95, open_days=7.0),
    "supermarket":   dict(popularity=0.70, open_days=7.0),
    "pharmacy":      dict(popularity=0.55, open_days=6.0),
    "bank":          dict(popularity=0.40, open_days=5.0),
    "shopping_mall": dict(popularity=0.65, open_days=7.0),
}

NAME_PREFIX = {
    "restaurant": "Restaurant", "cafe": "Café", "supermarket": "Supermarché",
    "pharmacy": "Pharmacie", "bank": "Banque", "shopping_mall": "Centre",
}


def wilson_lower_bound(rating: np.ndarray, n: np.ndarray, z: float = 1.96) -> np.ndarray:
    """Wilson 95% lower bound of a 1-5 rating treated as a proportion."""
    p = np.clip((rating - 1) / 4, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        center = p + z**2 / (2 * n)
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        lower = (center - margin) / (1 + z**2 / n)
    return np.where(n > 0, np.maximum(lower, 0), 0.0) * 100


def generate_places(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for dname, d in DISTRICTS.items():
        for tname, t in TYPES.items():
            n_places = rng.integers(25, 56)  # diversity per district/type
            for _ in range(n_places):
                # Per-place foot traffic varies around the district baseline.
                foot = np.clip(d["footfall"] + rng.normal(0, 0.06), 0.05, 1.0)

                # Same-type competition within 500 m (denser area -> more rivals).
                comp = max(0, int(rng.poisson(2 + 9 * d["density"] * t["popularity"])))
                total_pois = comp + int(rng.poisson(8 + 40 * d["density"]))
                cafes = int(rng.poisson(1 + 6 * d["density"]))
                restaurants = int(rng.poisson(1 + 7 * d["density"]))
                pharmacies = int(rng.poisson(1 + 3 * d["density"]))
                banks = int(rng.poisson(1 + 3 * d["density"]))

                # Accessibility: denser areas have amenities closer (smaller dist).
                def dist():
                    return float(np.clip(rng.lognormal(5.6 - 1.4 * d["density"], 0.5), 30, 4000))

                dist_pharm, dist_bank = dist(), dist()
                dist_super, dist_fuel = dist(), dist()

                price = int(np.clip(round(rng.normal(1 + 3 * d["income"], 0.8)), 0, 4))
                open_days = float(np.clip(round(t["open_days"] + rng.normal(0, 0.4) * 2) / 1, 4, 7))

                # ── Latent QUALITY (drives the star rating) ───────────────────
                comp_overflow = max(0.0, comp / 12 - 0.6)

                quality = (
    1.80 * d["income"]           # était 1.50
    + 0.80 * foot                # était 0.60
    + 0.40 * (open_days / 7)     # était 0.15
    + 0.30 * (price / 4)         # NOUVEAU : prix -> qualité perçue
    - 1.10 * comp_overflow        # était -0.90
    - 0.25 * (np.mean([dist_pharm, dist_bank, dist_super]) / 2000)
    + rng.normal(0, 0.07)        # était 0.15 ← réduction bruit principale
)
                rating = float(np.clip(
    round(2.6 + 0.95 * quality + rng.normal(0, 0.05), 1),  # bruit 0.10 -> 0.05
    1.0, 5.0
))


                # ── Latent POPULARITY (drives review volume) ──────────────────
                log_mu = (
    2.6
    + 2.8 * foot                 # était 2.3
    + 1.5 * d["density"]         # était 1.3
    + 1.2 * t["popularity"]      # était 1.0
    + 0.6 * d["income"]          # était 0.4
    - 0.6 * comp_overflow
    + rng.normal(0, 0.12)        # était 0.25 ← réduction bruit principale
)
                review_count = int(np.clip(round(np.expm1(log_mu)), 5, 60000))  # min 5 -> Wils
                review_count = int(np.clip(round(np.expm1(log_mu)), 0, 60000))

                rows.append({
                    "place_id": f"syn_{len(rows):05d}",
                    "name": f"{NAME_PREFIX[tname]} {dname} {len(rows) % 1000}",
                    "district": dname,
                    "place_type": tname,
                    "latitude": d["lat"] + rng.normal(0, 0.004),
                    "longitude": d["lng"] + rng.normal(0, 0.004),
                    "rating": rating,
                    "review_count": review_count,
                    "price_level_encoded": price,
                    "opening_days": open_days,
                    # legitimate market features (NOT derived from the target)
                    "population_density": round(d["density"] + rng.normal(0, 0.03), 4),
                    "income_index": round(d["income"] + rng.normal(0, 0.03), 4),
                    "foot_traffic_index": round(foot, 4),
                    "type_count_in_district": comp,
                    "cafes_500m": cafes,
                    "restaurants_500m": restaurants,
                    "pharmacies_500m": pharmacies,
                    "banks_500m": banks,
                    "total_pois_500m": total_pois,
                    "dist_nearest_pharmacy": round(dist_pharm, 1),
                    "dist_nearest_bank": round(dist_bank, 1),
                    "dist_nearest_supermarket": round(dist_super, 1),
                    "dist_nearest_fuel": round(dist_fuel, 1),
                })
    return pd.DataFrame(rows)


def add_target_and_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["success_score"] = wilson_lower_bound(
        df["rating"].values, df["review_count"].values
    ).round(2)

    df["log_reviews"] = np.log1p(df["review_count"])
    df["rating_norm"] = np.clip((df["rating"] - 1) / 4, 0, 1)
    df["reviews_norm"] = df["log_reviews"] / df["log_reviews"].max()
    df["log_review_count"] = np.log1p(df["review_count"])
    df["log_total_pois_500m"] = np.log1p(df["total_pois_500m"])
    df["success_score_pct"] = df["success_score"].rank(pct=True) * 100

    # ✅ NOUVEAU : features croisées qui reflètent exactement la structure du DGP
    df["income_x_foot"] = df["income_index"] * df["foot_traffic_index"]
    df["income_x_density"] = df["income_index"] * df["population_density"]
    df["foot_x_density"] = df["foot_traffic_index"] * df["population_density"]
    df["accessibility_score"] = 1 / (
        1 + (df["dist_nearest_pharmacy"] + df["dist_nearest_bank"] + df["dist_nearest_supermarket"]) / 3000
    )
    df["competition_pressure"] = np.clip(df["type_count_in_district"] / 12 - 0.6, 0, None)
    df["market_attractiveness"] = (
        df["income_index"] * 0.4
        + df["foot_traffic_index"] * 0.35
        + df["population_density"] * 0.25
    )

    # Agrégats district (inchangés)
    dist_stats = df.groupby("district").agg(
        district_avg_rating=("rating", "mean"),
        district_avg_reviews=("review_count", "mean"),
        district_place_count=("place_id", "count"),
        district_avg_score=("success_score", "mean"),
    ).reset_index()
    df = df.merge(dist_stats, on="district", how="left")

    type_avg = df.groupby(["district", "place_type"])["success_score"].mean().reset_index(
        name="type_avg_score"
    )
    df = df.merge(type_avg, on=["district", "place_type"], how="left")
    df["above_type_avg"] = (df["success_score"] > df["type_avg_score"]).astype(int)

    df["avg_annual_temp"] = 22.4
    df["rainy_days_per_year"] = 48
    df["peak_season_months"] = 4

    df = pd.get_dummies(df, columns=["place_type"], prefix="type")
    df = pd.get_dummies(df, columns=["district"], prefix="district")
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    logger.info("Generating synthetic MarketMind dataset (SIMULATED data)...")

    df = generate_places(rng)
    df = add_target_and_derived(df)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    logger.success(f"Wrote {len(df):,} rows × {df.shape[1]} cols -> {OUT_PARQUET}")

    # Quick honesty check: how much do market features alone explain the target?
    feats = [
        "income_index", "foot_traffic_index", "population_density",
        "type_count_in_district", "price_level_encoded", "opening_days",
        "dist_nearest_pharmacy", "total_pois_500m",
    ]
    corr = df[feats + ["success_score"]].corr()["success_score"].drop("success_score")
    logger.info(f"\nTarget summary:\n{df['success_score'].describe()}")
    logger.info(f"\nFeature/target correlations (market signal, no leakage):\n"
                f"{corr.sort_values(key=abs, ascending=False)}")

    try:
        from sqlalchemy import create_engine
        from src.config import settings
        engine = create_engine(settings.database_url)
        df.to_sql("features", engine, if_exists="replace", index=False)
        logger.success("Also wrote `features` table to the database")
    except Exception as e:  # DB optional for a pure-parquet workflow
        logger.warning(f"Skipped DB write ({e}); parquet is enough for train/serve")


if __name__ == "__main__":
    main()

