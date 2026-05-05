from datetime import datetime, timedelta

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

LATITUDE = 48.85
LONGITUDE = 2.35
AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


class AirQualityIngestion:
    """Fetches and merges air quality and weather data for Paris."""

    def __init__(self, past_days: int = 90):
        """
        Args:
            past_days: Number of past days to fetch data for.
        """
        self.past_days = past_days
        self.end_date = datetime.today().strftime("%Y-%m-%d")
        self.start_date = (datetime.today() - timedelta(days=past_days)).strftime("%Y-%m-%d")
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    def fetch_air_quality(self) -> pd.DataFrame:
        """Fetches hourly air quality data from Open-Meteo CAMS API."""
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ["pm2_5", "nitrogen_dioxide", "ozone", "dust", "european_aqi"],
            "past_days": self.past_days
        }
        response = self.client.weather_api(AQI_URL, params=params)[0]
        hourly = response.Hourly()

        return pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "pm2_5": hourly.Variables(0).ValuesAsNumpy(),
            "no2": hourly.Variables(1).ValuesAsNumpy(),
            "o3": hourly.Variables(2).ValuesAsNumpy(),
            "dust": hourly.Variables(3).ValuesAsNumpy(),
            "european_aqi": hourly.Variables(4).ValuesAsNumpy(),
        })

    def fetch_weather(self) -> pd.DataFrame:
        """Fetches hourly historical weather data from Open-Meteo Archive API."""
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": [
                "temperature_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "relative_humidity_2m"
            ],
            "start_date": self.start_date,
            "end_date": self.end_date
        }
        response = self.client.weather_api(WEATHER_URL, params=params)[0]
        hourly = response.Hourly()

        return pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(1).ValuesAsNumpy(),
            "wind_direction": hourly.Variables(2).ValuesAsNumpy(),
            "precipitation": hourly.Variables(3).ValuesAsNumpy(),
            "humidity": hourly.Variables(4).ValuesAsNumpy(),
        })

    def fetch_all(self) -> pd.DataFrame:
        """Fetches, merges and cleans all data sources."""
        df_aqi = self.fetch_air_quality()
        df_weather = self.fetch_weather()
        df = pd.merge(df_aqi, df_weather, on="date", how="inner")
        df = df.dropna(subset=["temperature"])
        return df