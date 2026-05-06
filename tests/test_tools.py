import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from src.tools import compute_threshold, detect_anomaly, get_weather_context


def make_mock_df() -> pd.DataFrame:
    """Creates a minimal mock DataFrame for testing.
    Day 1 (2026-02-01): normal conditions (AQI=30)
    Day 2 (2026-02-02): anomaly (AQI=80)
    """
    return pd.DataFrame({
        "date": pd.date_range("2026-02-01", periods=48, freq="h", tz="UTC"),
        "european_aqi": [30.0] * 24 + [80.0] * 24,
        "pm2_5":        [10.0] * 24 + [40.0] * 24,
        "no2":          [20.0] * 24 + [50.0] * 24,
        "o3":           [50.0] * 24 + [30.0] * 24,
        "dust":         [ 0.5] * 24 + [20.0] * 24,
        "temperature":  [12.0] * 48,
        "wind_speed":   [ 5.0] * 48,
        "wind_direction": [180.0] * 48,
        "precipitation": [0.0] * 48,
        "humidity":     [70.0] * 48,
    })


@pytest.fixture
def mock_df():
    with patch("src.tools.load_data", return_value=make_mock_df()):
        yield


class TestComputeThreshold:
    def test_rcompute_threshold(self):
        """daily max: [30, 80] → mean=55, std(ddof=1)=35.35 → threshold=125.7"""
        df = make_mock_df()
        expected = 55.0 + 2 * np.std([30.0, 80.0], ddof=1)
        assert compute_threshold(df) == expected


class TestDetectAnomaly:
    def test_out_of_range_returns_error(self, mock_df):
        result = detect_anomaly.invoke({"date": "2020-01-01"})
        assert "error" in result

    def test_normal_day_values(self, mock_df):
        """Day 1: AQI=30, below threshold=105 → no anomaly."""
        result = detect_anomaly.invoke({"date": "2026-02-01"})
        assert result["is_anomaly"] is False
        assert result["aqi_max"] == 30.0
        assert result["aqi_mean"] == 30.0
        assert result["aqi_threshold"] == 125.7

    def test_o3_has_max_day(self, mock_df):
        result = detect_anomaly.invoke({"date": "2026-02-01"})
        assert "max_day" in result["pollutants"]["o3"]
        assert result["pollutants"]["o3"]["max_day"] == 50.0


class TestGetWeatherContext:
    def test_out_of_range_returns_error(self, mock_df):
        result = get_weather_context.invoke({"date": "2020-01-01"})
        assert "error" in result

    def test_expected_values(self, mock_df):
        """Day 1: temperature=12, wind=5, precipitation=0, humidity=70."""
        result = get_weather_context.invoke({"date": "2026-02-01"})
        assert result["temperature_mean"] == 12.0
        assert result["temperature_max"] == 12.0
        assert result["wind_speed_mean"] == 5.0
        assert result["wind_speed_min"] == 5.0
        assert result["precipitation_total"] == 0.0
        assert result["humidity_mean"] == 70.0