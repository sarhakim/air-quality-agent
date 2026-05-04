from datetime import datetime, timezone

import pandas as pd
from langchain_core.tools import tool

from src.ingestion import AirQualityIngestion


def _load_data() -> pd.DataFrame:
    """Loads air quality and weather data."""
    return AirQualityIngestion(past_days=90).fetch_all()


@tool
def detect_anomaly(date: str) -> dict:
    """
    Detects whether air quality in Paris is anomalous on a given date.
    Returns the AQI level, whether it is an anomaly, and the main pollutant responsible.

    Args:
        date: date in YYYY-MM-DD format.
    """
    df = _load_data()

    df["day"] = df["date"].dt.date
    target_day = pd.to_datetime(date).date()
    df_day = df[df["day"] == target_day]

    if df_day.empty:
        return {"error": f"No data available for {date}"}

    # Daily max per day for a consistent baseline
    daily_max = df.groupby("day")["european_aqi"].max()
    aqi_threshold = daily_max.mean() + 2 * daily_max.std()

    aqi_max = df_day["european_aqi"].max()
    aqi_mean = df_day["european_aqi"].mean()
    is_anomaly = bool(aqi_max > aqi_threshold)

    # Identify the main pollutant using z-scores
    pollutants = {}
    for col in ["pm2_5", "no2", "o3", "dust"]:
        day_mean = df_day[col].mean()
        z_score = (day_mean - df[col].mean()) / df[col].std()
        pollutants[col] = {
            "mean_day": round(float(day_mean), 2),
            "z_score": round(float(z_score), 2)
        }

    main_pollutant = max(pollutants, key=lambda x: pollutants[x]["z_score"])

    return {
        "date": date,
        "aqi_max": round(float(aqi_max), 1),
        "aqi_mean": round(float(aqi_mean), 1),
        "aqi_threshold": round(float(aqi_threshold), 1),
        "is_anomaly": is_anomaly,
        "pollutants": pollutants,
        "main_pollutant": main_pollutant
    }

@tool
def get_weather_context(date: str) -> dict:
    """
    Returns weather conditions in Paris for a given date.
    Useful to contextualize air quality anomalies (low wind = pollution stagnation, 
    rain = air washout, high temperature = ozone formation).

    Args:
        date: date in YYYY-MM-DD format.
    """
    df = _load_data()

    df["day"] = df["date"].dt.date
    target_day = pd.to_datetime(date).date()
    df_day = df[df["day"] == target_day]

    if df_day.empty:
        return {"error": f"No data available for {date}"}

    return {
        "date": date,
        "temperature_mean": round(float(df_day["temperature"].mean()), 1),
        "temperature_max": round(float(df_day["temperature"].max()), 1),
        "wind_speed_mean": round(float(df_day["wind_speed"].mean()), 1),
        "wind_speed_min": round(float(df_day["wind_speed"].min()), 1),
        "precipitation_total": round(float(df_day["precipitation"].sum()), 1),
        "humidity_mean": round(float(df_day["humidity"].mean()), 1),
    }

@tool
def get_current_data() -> dict:
    """
    Returns the latest available air quality and weather measurements for Paris.
    Useful to answer questions about current conditions.
    """
    df = _load_data()
    now = datetime.now(timezone.utc)
    df_past = df[df["date"] <= now].dropna()
    latest = df_past.iloc[-1]

    return {
        "datetime": str(latest["date"]),
        "european_aqi": round(float(latest["european_aqi"]), 1),
        "pm2_5": round(float(latest["pm2_5"]), 2),
        "no2": round(float(latest["no2"]), 2),
        "o3": round(float(latest["o3"]), 2),
        "dust": round(float(latest["dust"]), 2),
        "temperature": round(float(latest["temperature"]), 1),
        "wind_speed": round(float(latest["wind_speed"]), 1),
        "precipitation": round(float(latest["precipitation"]), 1),
        "humidity": round(float(latest["humidity"]), 1),
    }

@tool
def summarize_situation(days: int = 7) -> dict:
    """
    Summarizes air quality in Paris over the last N days.
    Returns anomaly count, worst days, and overall AQI trend.

    Args:
        days: number of past days to summarize (default: 7).
    """
    df = _load_data()

    df["day"] = df["date"].dt.date
    cutoff = df["day"].max() - pd.Timedelta(days=days)
    df_period = df[df["day"] > cutoff]

    if df_period.empty:
        return {"error": f"No data available for the last {days} days"}

    # Daily max AQI
    daily_max = df_period.groupby("day")["european_aqi"].max()

    # Threshold from full history
    full_daily_max = df.groupby("day")["european_aqi"].max()
    aqi_threshold = full_daily_max.mean() + 2 * full_daily_max.std()

    anomaly_days = daily_max[daily_max > aqi_threshold]
    worst_day = daily_max.idxmax()

    return {
        "period_days": days,
        "aqi_mean": round(float(daily_max.mean()), 1),
        "aqi_max": round(float(daily_max.max()), 1),
        "aqi_threshold": round(float(aqi_threshold), 1),
        "anomaly_count": int(len(anomaly_days)),
        "anomaly_days": [str(d) for d in anomaly_days.index.tolist()],
        "worst_day": str(worst_day),
    }