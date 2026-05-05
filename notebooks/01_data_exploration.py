#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Paramètres — Paris
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params = {
    "latitude": 48.85,
    "longitude": 2.35,
    "hourly": ["pm2_5", "nitrogen_dioxide", "ozone", "dust", "european_aqi"],
    "past_days": 90
}

responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")


# In[2]:


hourly = response.Hourly()
hourly_pm2_5 = hourly.Variables(0).ValuesAsNumpy()
hourly_no2 = hourly.Variables(1).ValuesAsNumpy()
hourly_o3 = hourly.Variables(2).ValuesAsNumpy()
hourly_dust = hourly.Variables(3).ValuesAsNumpy()
hourly_aqi = hourly.Variables(4).ValuesAsNumpy()

df = pd.DataFrame({
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "pm2_5": hourly_pm2_5,
    "no2": hourly_no2,
    "o3": hourly_o3,
    "dust": hourly_dust,
    "european_aqi": hourly_aqi,
})         # index 2 = ozone

print(df.shape)
print(df.describe())


# In[3]:


fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

for ax, col in zip(axes, ["pm2_5", "no2", "o3"]):
    ax.plot(df["date"], df[col], linewidth=0.7)
    mean, std = df[col].mean(), df[col].std()
    ax.axhline(mean + 2*std, color="red", linestyle="--", alpha=0.7, label="mean+2σ")
    ax.set_ylabel(col)
    ax.legend()

plt.tight_layout()
plt.show()


# In[4]:


fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(df["date"], df["european_aqi"], linewidth=0.7, color="purple")
axes[0].axhline(40, color="orange", linestyle="--", alpha=0.7, label="fair (40)")
axes[0].axhline(60, color="red", linestyle="--", alpha=0.7, label="moderate (60)")
axes[0].set_ylabel("European AQI")
axes[0].legend()

axes[1].plot(df["date"], df["dust"], linewidth=0.7, color="brown")
axes[1].set_ylabel("Dust")

plt.tight_layout()
plt.show()


# In[5]:


# Zoom sur février-mars pour voir la corrélation
mask = (df["date"] >= "2026-02-15") & (df["date"] <= "2026-03-20")
df_zoom = df[mask]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axes[0].plot(df_zoom["date"], df_zoom["european_aqi"], color="purple")
axes[0].axhline(60, color="red", linestyle="--", alpha=0.7)
axes[0].set_ylabel("AQI")

axes[1].plot(df_zoom["date"], df_zoom["pm2_5"], color="blue")
axes[1].set_ylabel("PM2.5")

axes[2].plot(df_zoom["date"], df_zoom["dust"], color="brown")
axes[2].set_ylabel("Dust")

plt.tight_layout()
plt.show()


# In[14]:


weather_params = {
    "latitude": 48.85,
    "longitude": 2.35,
    "hourly": [
        "temperature_2m",
        "wind_speed_10m", 
        "wind_direction_10m",
        "precipitation",
        "relative_humidity_2m"
    ],
    "start_date": "2026-02-01",
    "end_date": "2026-05-04"
}

weather_responses = openmeteo.weather_api(
    "https://archive-api.open-meteo.com/v1/archive",  # <- URL différente
    params=weather_params
)


# In[22]:


weather_response = weather_responses[0]

hourly_w = weather_response.Hourly()
df_weather = pd.DataFrame({
    "date": pd.date_range(
        start=pd.to_datetime(hourly_w.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_w.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_w.Interval()),
        inclusive="left"
    ),
    "temperature": hourly_w.Variables(0).ValuesAsNumpy(),
    "wind_speed": hourly_w.Variables(1).ValuesAsNumpy(),
    "wind_direction": hourly_w.Variables(2).ValuesAsNumpy(),
    "precipitation": hourly_w.Variables(3).ValuesAsNumpy(),
    "humidity": hourly_w.Variables(4).ValuesAsNumpy(),
})

# Merger avec df qualité de l'air
df_full = pd.merge(df, df_weather, on="date", how="inner")
print(df_full.shape)
print(df_full.describe())


# In[23]:


df.date.max()


# In[24]:


df.dropna().date.max()


# In[25]:


df_weather.date.max()


# In[26]:


df_weather


# In[20]:


df["day"] = df["date"].dt.date
daily_max = df.groupby("day")["european_aqi"].max()
aqi_threshold = daily_max.mean() + 2 * daily_max.std()

print("Seuil anomalie:", round(aqi_threshold, 1))
print("\nJours anomaux:")
print(daily_max[daily_max > aqi_threshold].sort_values(ascending=False))
print("\nJours normaux (AQI < 30):")
print(daily_max[daily_max < 30].sample(5, random_state=42))


# In[27]:


df = df_full.copy()
mask = (df["date"].dt.date >= pd.to_datetime("2026-04-07").date()) & \
       (df["date"].dt.date <= pd.to_datetime("2026-04-09").date())
df[mask].groupby("day")[["o3", "european_aqi", "temperature", "dust"]].max()


# In[31]:


df_full.to_csv("data/air_quality_snapshot.csv", index=False)
print(f"Saved {len(df_full)} rows")


# In[ ]:




