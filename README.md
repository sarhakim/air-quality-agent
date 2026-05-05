# Air Quality Agent — Paris IDF

A conversational agent that monitors air quality in Paris, detects anomalies, 
and explains their meteorological causes using real-time and historical data.

## Architecture

Open-Meteo API ──► ingestion.py ──► snapshot.csv ──► tools ──► LangGraph agent                                                

**Stack**: LangGraph · Claude Haiku · Open-Meteo · LangSmith · Python 3.12 · uv

## Tools

| Tool | Description |
|------|-------------|
| `detect_anomaly` | Detects AQI anomalies on a given date using z-scores |
| `get_weather_context` | Returns meteorological conditions to explain anomalies |
| `get_current_data` | Returns latest available air quality measurements |
| `summarize_situation` | Summarizes anomaly count over the last N days |

## Quickstart

```bash
git clone https://github.com/sarhakim/air-quality-agent
cd air-quality-agent
uv sync
cp .env.example .env
uv run main.py
```

## Evaluation

### Dataset

15 hand-built test cases covering anomaly detection (Saharan dust, PM10, ozone), 
current conditions, out-of-range dates, weather context, and multilingual questions (FR/EN).
Evaluated with Claude Haiku as LLM-as-judge, tracked via LangSmith.

### Results

| Model | Tools OK | Avg Score | Latency P50 | Cost (15 cases) |
|-------|----------|-----------|-------------|-----------------|
| Llama 3.1 8B (local) | 53%* | 2.67/5* | 7.7s | $0 |
| Claude Haiku v1 | 100% | 4.33/5 | 3.9s | $0.11 |
| Claude Haiku v2 | 100% | 4.60/5 | 4.3s | $0.12 |


Tracked via [LangSmith](https://smith.langchain.com).

### Prompt iterations

**v1 → v2 improvements:**
- Improved tool docstrings with explicit routing rules ("use this for X, not for Y")
- Added few-shot examples covering edge cases (ozone peaks, out-of-range dates, anticyclonic stagnation)
- Fixed ground truth for current conditions (CAMS 12h delay context)
- Added O3 max_day reporting rule (hourly peaks vs daily mean)

**Key score improvements:**

| Case | v1 | v2 | Issue fixed |
|------|----|----|-------------|
| weather_001 | 2/5 | 5/5 | Agent now uses "anticyclonic stagnation" terminology |
| current_001 | 2/5 | 4/5 | Added CAMS delay disclaimer |
| summary_001 | 2/5 | 5/5 | Fixed ground truth (0 anomalies is correct) |
| ozone_001 | 3/5 | 4/5 | Agent now reports O3 max_day |

## Known Limitations

- **Spatial resolution**: Open-Meteo CAMS model at ~11km resolution — no arrondissement-level granularity (AirParif API would be needed)
- **Static snapshot**: data covers Feb–May 2026; regenerate snapshot for newer data
- **Ozone detection**: daily AQI threshold misses hourly O3 peaks — max_day reported but not used for anomaly classification
- **Model dependency**: multi-step tool calling requires Claude Haiku or better — Llama 3.1 8B fails on sequential tool calls

## Data Sources

- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) — CAMS European model
- [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) — Historical weather