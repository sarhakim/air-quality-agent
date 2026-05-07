from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.tools import compute_pollutant_thresholds, compute_threshold, detect_anomaly, get_weather_context


def make_mock_df() -> pd.DataFrame:
    """Creates a minimal mock DataFrame for testing.
    - Days 1-28 (2026-02-01 to 2026-02-28): normal conditions (AQI=30, dust=0, o3=50)
    - Day 29 (2026-03-01): clear anomaly (AQI=90, dust=50)
    - Day 30 (2026-03-02): O3-only anomaly (AQI=30, o3=200) — tests pollutant-level detection
    """
    n_normal = 28 * 24
    n_anomaly = 24
    n_o3 = 24
    n_total = n_normal + n_anomaly + n_o3

    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "date": pd.date_range("2026-02-01", periods=n_total, freq="h", tz="UTC"),
        "european_aqi":   [30.0] * n_normal + [90.0] * n_anomaly + [30.0] * n_o3,
        "pm2_5": list(rng.normal(10, 3, n_normal)) + [10.5] * n_anomaly + [10.5] * n_o3,
        "no2":   list(rng.normal(20, 5, n_normal)) + [21.0] * n_anomaly + [21.0] * n_o3,
        "dust":  [0.0] * n_normal + [50.0] * n_anomaly + [0.0] * n_o3,
        "o3":    [50.0] * n_normal + [30.0] * n_anomaly + [200.0] * n_o3,  # pic O3 isolé
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
        daily_max = [30.0] * 28 + [90.0] + [30.0]
        expected = float(np.mean(daily_max) + 2 * np.std(daily_max, ddof=1))
        assert compute_threshold(make_mock_df()) == expected


class TestComputePollutantThresholds:
    def test_returns_all_pollutants(self):
        result = compute_pollutant_thresholds(make_mock_df())
        assert set(result.keys()) == {"pm2_5", "no2", "dust", "o3"}

    def test_no_nan(self):
        result = compute_pollutant_thresholds(make_mock_df())
        for key, val in result.items():
            assert val == val, f"{key} threshold is NaN"

    def test_o3_threshold_exceeded_by_anomaly_day(self):
        """O3=200 on 2026-03-02 should exceed its mean+2std threshold."""
        df = make_mock_df()
        thresholds = compute_pollutant_thresholds(df)
        assert 200.0 > thresholds["o3"]

    def test_dust_threshold_exceeded_by_anomaly_day(self):
        """dust=50 on 2026-03-01 should exceed its mean+2std threshold."""
        df = make_mock_df()
        thresholds = compute_pollutant_thresholds(df)
        assert 50.0 > thresholds["dust"]


class TestDetectAnomaly:
    def test_out_of_range_returns_error(self, mock_df):
        result = detect_anomaly.invoke({"date": "2020-01-01"})
        assert "error" in result

    def test_normal_day_values(self, mock_df):
        """2026-02-01: AQI=30, no pollutant anomaly → no anomaly."""
        result = detect_anomaly.invoke({"date": "2026-02-01"})
        assert result["is_anomaly"] is False
        assert result["is_anomaly_aqi"] is False
        assert result["anomalous_pollutants"] == []
        assert result["aqi_max"] == 30.0
        assert result["aqi_mean"] == 30.0

    def test_anomaly_day_aqi(self, mock_df):
        """2026-03-01: AQI=90 → anomaly via AQI, dust is main pollutant."""
        result = detect_anomaly.invoke({"date": "2026-03-01"})
        assert result["is_anomaly"] is True
        assert result["is_anomaly_aqi"] is True
        assert result["main_pollutant"] == "dust"
        assert "dust" in result["anomalous_pollutants"]

    def test_anomaly_day_o3_only(self, mock_df):
        """2026-03-02: AQI=30 (normal) but O3=200 → anomaly via pollutant threshold only."""
        result = detect_anomaly.invoke({"date": "2026-03-02"})
        assert result["is_anomaly"] is True
        assert result["is_anomaly_aqi"] is False
        assert "o3" in result["anomalous_pollutants"]

    def test_o3_has_max_day(self, mock_df):
        result = detect_anomaly.invoke({"date": "2026-02-01"})
        assert "max_day" in result["pollutants"]["o3"]
        assert result["pollutants"]["o3"]["max_day"] == 50.0

    def test_o3_max_day_anomaly(self, mock_df):
        """2026-03-02: O3 max_day should be 200."""
        result = detect_anomaly.invoke({"date": "2026-03-02"})
        assert result["pollutants"]["o3"]["max_day"] == 200.0


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