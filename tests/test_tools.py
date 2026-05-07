from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.tools import compute_threshold, detect_anomaly, get_weather_context


def make_mock_df() -> pd.DataFrame:
    """Creates a minimal mock DataFrame for testing.
    - Days 1-28 (2026-02-01 to 2026-02-28): normal conditions (AQI=30, dust=0)
    - Day 29 (2026-03-01): clear anomaly (AQI=90, dust=50)
    """
    n_normal = 28 * 24
    n_anomaly = 24
    n_total = n_normal + n_anomaly

    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "date": pd.date_range("2026-02-01", periods=n_total, freq="h", tz="UTC"),
        "european_aqi":   [30.0] * n_normal + [90.0] * n_anomaly,
        "pm2_5": list(rng.normal(10, 3, n_normal)) + [10.5] * n_anomaly, 
        "no2":   list(rng.normal(20, 5, n_normal)) + [21.0] * n_anomaly,
        "dust":  [0.0] * n_normal + [50.0] * n_anomaly,                   # higher z-score
        "o3":             [50.0] * n_normal + [30.0] * n_anomaly,
        "temperature":    [12.0] * n_total,
        "wind_speed":     [ 5.0] * n_total,
        "wind_direction": [180.0] * n_total,
        "precipitation":  [ 0.0] * n_total,
        "humidity":       [70.0] * n_total,
    })

@pytest.fixture
def mock_df():
    with patch("src.tools.load_data", return_value=make_mock_df()):
        yield

class TestComputeThreshold:
    def test_not_nan(self):
        result = compute_threshold(make_mock_df())
        assert result == result  # NaN != NaN

    def test_above_mean(self):
        df = make_mock_df()
        daily_max = df.groupby(df["date"].dt.date)["european_aqi"].max()
        assert compute_threshold(df) > daily_max.mean()

    def test_expected_value(self):
        daily_max = [30.0] * 28 + [90.0]
        expected = float(np.mean(daily_max) + 2 * np.std(daily_max, ddof=1))
        assert compute_threshold(make_mock_df()) == expected


class TestDetectAnomaly:
    def test_out_of_range_returns_error(self, mock_df):
        result = detect_anomaly.invoke({"date": "2020-01-01"})
        assert "error" in result

    def test_normal_day_values(self, mock_df):
        """2026-02-01: AQI=30, below threshold → no anomaly."""
        result = detect_anomaly.invoke({"date": "2026-02-01"})
        assert result["is_anomaly"] is False
        assert result["aqi_max"] == 30.0
        assert result["aqi_mean"] == 30.0

    def test_anomaly_day_values(self, mock_df):
        """2026-03-01: AQI=90, above threshold → anomaly, dust is main pollutant."""
        result = detect_anomaly.invoke({"date": "2026-03-01"})
        assert result["is_anomaly"] is True
        assert result["aqi_max"] == 90.0
        assert result["main_pollutant"] == "dust"

    def test_o3_has_max_day(self, mock_df):
        result = detect_anomaly.invoke({"date": "2026-02-01"})
        assert "max_day" in result["pollutants"]["o3"]
        assert result["pollutants"]["o3"]["max_day"] == 50.0


class TestGetWeatherContext:
    def test_out_of_range_returns_error(self, mock_df):
        result = get_weather_context.invoke({"date": "2020-01-01"})
        assert "error" in result

    def test_expected_values(self, mock_df):
        """All days have identical weather values."""
        result = get_weather_context.invoke({"date": "2026-02-01"})
        assert result["temperature_mean"] == 12.0
        assert result["temperature_max"] == 12.0
        assert result["wind_speed_mean"] == 5.0
        assert result["wind_speed_min"] == 5.0
        assert result["precipitation_total"] == 0.0
        assert result["humidity_mean"] == 70.0