"""
Phase 1 — Data Collection
Reads Rabat POIs from a local Morocco OSM extract (data/external/morocco-latest.osm.pbf)
Much faster and more reliable than live Overpass API queries.

Usage:
    make collect-osm
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import osmium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text
from loguru import logger

from src.config import settings

PBF_PATH = Path("data/external/morocco-latest.osm.pbf")

# Rabat bounding box
SOUTH, WEST, NORTH, EAST = settings.rabat_bbox

# OSM tags we care about
RELEVANT_AMENITIES = {
    "restaurant", "cafe", "fast_food", "bar",
    "marketplace", "market", "bank", "pharmacy", "shop"
}


class RabatPOIHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.pois = []

    def _in_rabat(self, lat, lon):
        return SOUTH <= lat <= NORTH and WEST <= lon <= EAST

    def _extract(self, tags, lat, lon):
        amenity  = tags.get("amenity", "")
        shop     = tags.get("shop", "")
        landuse  = tags.get("landuse", "")
        building = tags.get("building", "")

        if not any([amenity, shop,
                    landuse in ("retail", "commercial"),
                    building in ("retail", "commercial", "supermarket")]):
            return

        self.pois.append({
            "name":           tags.get("name", ""),
            "amenity":        amenity,
            "shop":           shop,
            "landuse":        landuse,
            "building":       building,
            "opening_hours":  tags.get("opening_hours", ""),
            "addr_street":    tags.get("addr:street", ""),
            "latitude":       lat,
            "longitude":      lon,
        })

    def node(self, n):
        if n.location.valid() and self._in_rabat(n.location.lat, n.location.lon):
            self._extract(n.tags, n.location.lat, n.location.lon)

    def way(self, w):
        try:
            if w.nodes and self._in_rabat(
                w.nodes[0].location.lat, w.nodes[0].location.lon
            ):
                lat = sum(nd.location.lat for nd in w.nodes) / len(w.nodes)
                lon = sum(nd.location.lon for nd in w.nodes) / len(w.nodes)
                self._extract(w.tags, lat, lon)
        except Exception:
            pass


def fetch_pois() -> gpd.GeoDataFrame:
    logger.info(f"Reading POIs from {PBF_PATH} ...")
    handler = RabatPOIHandler()
    handler.apply_file(str(PBF_PATH), locations=True)

    df = pd.DataFrame(handler.pois)
    logger.info(f"Found {len(df):,} POIs in Rabat bbox")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(row.longitude, row.latitude) for row in df.itertuples()],
        crs="EPSG:4326",
    )
    gdf["source"]       = "osm_pbf"
    gdf["collected_at"] = pd.Timestamp.now(tz="UTC")
    return gdf


def save_to_postgis(gdf: gpd.GeoDataFrame) -> None:
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.commit()

    gdf.to_postgis(
        name="raw_pois",
        con=engine,
        if_exists="replace",
        index=False,
    )
    logger.success(f"Saved {len(gdf):,} rows → raw_pois (PostGIS)")


def main():
    if not PBF_PATH.exists():
        logger.error(f"PBF file not found at {PBF_PATH}")
        logger.error("Run: wget https://download.geofabrik.de/africa/morocco-latest.osm.pbf -P data/external/")
        sys.exit(1)

    logger.info("=== OSM collection starting ===")
    gdf = fetch_pois()
    logger.info(f"\n{gdf[['name','amenity','shop']].head(10).to_string()}")
    save_to_postgis(gdf)
    logger.info("=== OSM collection complete ===")


if __name__ == "__main__":
    main()
