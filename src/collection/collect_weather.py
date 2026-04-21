"""
Phase 1 — Data Collection
Fetches 2 years of daily weather data for Rabat from Open-Meteo (free, no key).

Usage:
    poetry run python src/collection/collect_weather.py
    make collect-weather
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

from src.config import settings

# Rabat coordinates
LAT, LON = 34.020, -6.841
START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather() -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "Africa/Casablanca",
    }
    logger.info(f"Fetching weather {START_DATE} → {END_DATE} for Rabat...")
    resp = httpx.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()["daily"]
    df = pd.DataFrame({
        "date":          pd.to_datetime(data["time"]),
        "temp_max":      data["temperature_2m_max"],
        "temp_min":      data["temperature_2m_min"],
        "precipitation": data["precipitation_sum"],
        "wind_speed":    data["wind_speed_10m_max"],
        "collected_at":  pd.Timestamp.now(tz="UTC"),
    })
    logger.success(f"Fetched {len(df)} daily records")
    return df


def save_to_db(df: pd.DataFrame) -> None:
    engine = create_engine(settings.database_url)
    df.to_sql(
        "weather_observations",
        engine,
        if_exists="replace",
        index=False,
    )
    logger.success(f"Saved {len(df)} rows → weather_observations")


def main() -> None:
    logger.info("=== Weather collection starting ===")
    df = fetch_weather()
    logger.info(f"\n{df.head()}")
    save_to_db(df)
    logger.info("=== Weather collection complete ===")


if __name__ == "__main__":
    main()
