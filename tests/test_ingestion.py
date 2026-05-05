from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion import AirQualityIngestion

EXPECTED_AQI_COLUMNS = {"date", "pm2_5", "no2", "o3", "dust", "european_aqi"}
EXPECTED_WEATHER_COLUMNS = {"date", "temperature", "wind_speed", "wind_direction", "precipitation", "humidity"}
DATE_START = datetime(2026, 2, 1, tzinfo=UTC)


def make_mock_hourly(n: int = 720):
    timestamp_start = int(DATE_START.timestamp())

    pm2_5 = MagicMock(ValuesAsNumpy=MagicMock(return_value=np.full(n, 0.0)))
    no2   = MagicMock(ValuesAsNumpy=MagicMock(return_value=np.full(n, 1.0)))
    o3    = MagicMock(ValuesAsNumpy=MagicMock(return_value=np.full(n, 2.0)))
    dust  = MagicMock(ValuesAsNumpy=MagicMock(return_value=np.full(n, 3.0)))
    aqi   = MagicMock(ValuesAsNumpy=MagicMock(return_value=np.full(n, 4.0)))

    hourly = MagicMock()
    hourly.Time.return_value = timestamp_start
    hourly.TimeEnd.return_value = timestamp_start + n * 3600
    hourly.Interval.return_value = 3600
    hourly.Variables.side_effect = [pm2_5, no2, o3, dust, aqi]
    return hourly


def make_mock_response(n: int = 720):
    response = MagicMock()
    response.Hourly.return_value = make_mock_hourly(n)
    return [response]


@pytest.fixture
def ingestion_client():
    with patch("ingestion.requests_cache.CachedSession"), \
         patch("ingestion.retry"), \
         patch("ingestion.openmeteo_requests.Client"):
        return AirQualityIngestion(past_days=90)


def test_fetch_air_quality(ingestion_client):
    """Columns present and variables mapped to correct columns."""
    ingestion_client.client.weather_api.return_value = make_mock_response()
    df = ingestion_client.fetch_air_quality()

    assert EXPECTED_AQI_COLUMNS.issubset(df.columns)
    assert (df["pm2_5"] == 0.0).all()
    assert (df["no2"] == 1.0).all()
    assert (df["o3"] == 2.0).all()
    assert (df["dust"] == 3.0).all()
    assert (df["european_aqi"] == 4.0).all()


def test_fetch_weather(ingestion_client):
    """Weather returns expected columns."""
    ingestion_client.client.weather_api.return_value = make_mock_response()
    df = ingestion_client.fetch_weather()
    assert EXPECTED_WEATHER_COLUMNS.issubset(df.columns)


def test_fetch_all(ingestion_client):
    """Merged DataFrame is non-empty with no nulls on critical columns."""
    ingestion_client.client.weather_api.side_effect = [
        make_mock_response(),
        make_mock_response(),
    ]
    df = ingestion_client.fetch_all()

    assert len(df) > 0
    assert df[["temperature", "european_aqi"]].isnull().sum().sum() == 0