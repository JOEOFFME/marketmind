from src.config import settings


def test_settings_loads():
    assert settings.database_url.startswith("postgresql://")


def test_rabat_bbox():
    south, west, north, east = settings.rabat_bbox
    # Rabat is roughly 33.9–34.1°N, -7.0–-6.7°E
    assert 33.5 < south < 34.5
    assert -7.5 < west  < -6.0
    assert south < north
    assert west  < east
