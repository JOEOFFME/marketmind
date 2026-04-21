"""
Phase 1 — Data Collection
Fetches marketplace-relevant POIs from OpenStreetMap for Rabat, Morocco
and stores them in PostGIS.

Usage:
    poetry run python src/collection/collect_osm.py
    make collect-osm
"""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import osmnx as ox
from loguru import logger
from sqlalchemy import create_engine, text

from src.config import settings

# Tags to pull from OSM — covers traditional + modern commerce
OSM_TAGS = {
    "shop": True,
    "amenity": [
        "restaurant", "cafe", "fast_food", "bar",
        "marketplace", "market", "bank", "pharmacy",
    ],
    "landuse": ["retail", "commercial"],
    "building": ["retail", "commercial", "supermarket"],
}

COLUMNS_TO_KEEP = [
    "geometry", "name", "amenity", "shop",
    "landuse", "building", "addr:street", "opening_hours",
]


def fetch_pois() -> "gpd.GeoDataFrame":
    import geopandas as gpd  # local import so script fails fast if missing

    south, west, north, east = settings.rabat_bbox
    logger.info(f"Fetching OSM features for Rabat bbox: {settings.rabat_bbox}")

    gdf = ox.features_from_bbox(
        bbox=(north, south, east, west),
        tags=OSM_TAGS,
    )

    # Keep only relevant columns that exist in this result
    cols = [c for c in COLUMNS_TO_KEEP if c in gdf.columns]
    gdf = gdf[cols].copy()

    # Normalise geometry to points (some OSM features are polygons/multipolygons)
    gdf["geometry"] = gdf["geometry"].centroid
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    gdf["source"] = "osm"
    gdf["collected_at"] = pd.Timestamp.now(tz="UTC")

    logger.success(f"Fetched {len(gdf):,} POIs from OSM")
    return gdf


def save_to_postgis(gdf: "gpd.GeoDataFrame") -> None:
    engine = create_engine(settings.database_url)

    # Ensure PostGIS is enabled (idempotent)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.commit()

    gdf.to_postgis(
        name="raw_pois",
        con=engine,
        if_exists="replace",
        index=False,
        dtype={"geometry": "Geometry"},
    )
    logger.success(f"Saved {len(gdf):,} rows → raw_pois (PostGIS)")


def main() -> None:
    logger.info("=== OSM collection starting ===")
    gdf = fetch_pois()

    # Quick sanity print
    logger.info(f"Sample:\n{gdf[['name', 'amenity', 'shop']].head(10).to_string()}")

    save_to_postgis(gdf)
    logger.info("=== OSM collection complete ===")


if __name__ == "__main__":
    main()
