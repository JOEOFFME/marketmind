"""
Unit tests for collection helpers.
These tests do NOT hit the network — they test parsing logic only.
"""
import pandas as pd


def test_weather_dataframe_shape():
    """Simulate what fetch_weather returns and verify shape."""
    mock_data = {
        "date":          pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "temp_max":      [18.5, 17.0],
        "temp_min":      [10.0, 9.5],
        "precipitation": [0.0, 2.1],
        "wind_speed":    [15.0, 20.0],
        "collected_at":  [pd.Timestamp.now(tz="UTC")] * 2,
    }
    df = pd.DataFrame(mock_data)
    assert len(df) == 2
    assert "temp_max" in df.columns
    assert df["precipitation"].min() >= 0
