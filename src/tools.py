import os
from datetime import UTC, datetime

import pandas as pd
from langchain_core.tools import tool

from src.ingestion import AirQualityIngestion

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_PATH = os.path.join(BASE_DIR, "data", "air_quality_snapshot.csv")

def load_data() -> pd.DataFrame:
    """Loads air quality data from snapshot if available, otherwise fetches from API."""
    if os.path.exists(SNAPSHOT_PATH):
        return pd.read_csv(SNAPSHOT_PATH, parse_dates=["date"])
    return AirQualityIngestion(past_days=90).fetch_all()

def compute_threshold(df: pd.DataFrame) -> float:
    """Computes the AQI anomaly threshold (daily max mean + 2 std)."""
    daily_max = df.groupby(df["date"].dt.date)["european_aqi"].max().dropna()
    return float(daily_max.mean() + 2 * daily_max.std())

def compute_pollutant_thresholds(df: pd.DataFrame) -> dict:
    """Computes anomaly thresholds per pollutant (daily max mean + 2 std)."""
    thresholds = {}
    for col in ["pm2_5", "no2", "dust"]:
        daily_mean = df.groupby(df["date"].dt.date)[col].mean()
        thresholds[col] = float(daily_mean.mean() + 2 * daily_mean.std())
    daily_max_o3 = df.groupby(df["date"].dt.date)["o3"].max()
    thresholds["o3"] = float(daily_max_o3.mean() + 2 * daily_max_o3.std())
    return thresholds


@tool
def detect_anomaly(date: str) -> dict:
    """
    Detects whether air quality in Paris was anomalous on a specific past date.
    Returns AQI levels, anomaly verdict, and z-scores per pollutant.

    Use this tool for:
    - Questions about a specific past date ("was the air quality bad on March 6?")
    - Detecting pollution episodes or anomalies on a given day
    - Identifying which pollutant caused an anomaly

    Do NOT use for current conditions — use get_current_data instead.
    If is_anomaly is true, always follow up with get_weather_context for the same date.

    Note: O3 returns both mean_day and max_day — always report max_day as ozone
    peaks hourly due to photochemical cycles. If max_day > 100 µg/m³, mention it
    is approaching the EEA threshold of 120 µg/m³.

    Args:
        date: date in YYYY-MM-DD format. Must be within the data window (Feb-May 2026).
    """
    df = load_data()

    df["day"] = df["date"].dt.date
    target_day = pd.to_datetime(date).date()
    df_day = df[df["day"] == target_day]

    if df_day.empty:
        return {"error": f"No data available for {date}"}

    aqi_threshold = compute_threshold(df)
    pollutant_thresholds = compute_pollutant_thresholds(df)

    aqi_max = df_day["european_aqi"].max()
    aqi_mean = df_day["european_aqi"].mean()
    is_anomaly_aqi = bool(aqi_max > aqi_threshold)

    # Identify the main pollutant using z-scores
    pollutants = {}
    for col in ["pm2_5", "no2", "dust"]:
        day_mean = df_day[col].mean()
        z_score = (day_mean - df[col].mean()) / df[col].std()
        pollutants[col] = {
            "mean_day": round(float(day_mean), 2),
            "z_score": round(float(z_score), 2)
        }

    # O3 — use both mean and max (photochemical cycle causes high hourly peaks)
    o3_mean = df_day["o3"].mean()
    o3_max = df_day["o3"].max()
    o3_z_score = (o3_mean - df["o3"].mean()) / df["o3"].std()
    pollutants["o3"] = {
        "mean_day": round(float(o3_mean), 2),
        "max_day": round(float(o3_max), 2),
        "z_score": round(float(o3_z_score), 2),
        "warning": "O3 peaks hourly — max may exceed mean significantly" if o3_max > 80 else None,
    }

    # Anomaly per pollutant (same logic as AQI: mean + 2 std)
    pollutant_values = {
        "o3": o3_max,  # max_day pour O3, cohérent avec compute_pollutant_thresholds
        "no2": pollutants["no2"]["mean_day"],
        "pm2_5": pollutants["pm2_5"]["mean_day"],
        "dust": pollutants["dust"]["mean_day"],
    }
    anomalous_pollutants = [
        p for p in pollutant_values
        if pollutant_values[p] > pollutant_thresholds[p]
    ]

    is_anomaly = is_anomaly_aqi or bool(anomalous_pollutants)
    main_pollutant = max(pollutants, key=lambda x: pollutants[x]["z_score"])

    return {
        "date": date,
        "aqi_max": round(float(aqi_max), 1),
        "aqi_mean": round(float(aqi_mean), 1),
        "aqi_threshold": round(float(aqi_threshold), 1),
        "is_anomaly": is_anomaly,
        "is_anomaly_aqi": is_anomaly_aqi,
        "anomalous_pollutants": anomalous_pollutants,  # [] si aucun
        "pollutants": pollutants,
        "main_pollutant": main_pollutant
    }

@tool
def get_weather_context(date: str) -> dict:
    """
    Returns meteorological conditions in Paris for a given past date.
    Use to explain why a pollution anomaly occurred.

    Use this tool:
    - Always after detect_anomaly returns is_anomaly=true
    - For questions about wind, rain, temperature on a specific day
    - To explain stagnation (low wind), air washout (precipitation), or ozone formation (high temperature)

    Key interpretations:
    - wind_speed_mean < 5 km/h → stagnation, pollutants accumulate
    - precipitation_total > 0 → air washout effect
    - temperature_max > 20°C → favors ozone formation

    Note: wind_speed is in km/h.

    Args:
        date: date in YYYY-MM-DD format.
    """
    df = load_data()

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

    Use this tool for:
    - Questions about current or present conditions ("what is the air quality now?")
    - Questions using words like "currently", "right now", "today", "at the moment"

    Important: data may be up to 12 hours old due to CAMS model update frequency.
    Always mention this delay when reporting current conditions.
    The datetime field may show a timestamp from today or yesterday — this is normal.

    Do NOT use for questions about specific past dates — use detect_anomaly instead.
    """
    df = load_data()
    now = datetime.now(UTC)
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
    Returns anomaly count, worst days, and overall AQI statistics.

    Use this tool for:
    - Questions about recent trends ("how has the air quality been lately?")
    - Counting anomalies over a period ("how many bad days in the last 30 days?")
    - Getting an overview without a specific date

    Args:
        days: number of past days to summarize (default: 7, max: 90).
    """
    df = load_data()

    df["day"] = df["date"].dt.date
    cutoff = df["day"].max() - pd.Timedelta(days=days)
    df_period = df[df["day"] > cutoff]

    if df_period.empty:
        return {"error": f"No data available for the last {days} days"}

    # Daily max AQI
    daily_max = df_period.groupby("day")["european_aqi"].max()

    # Threshold from full history
    aqi_threshold = compute_threshold(df)

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