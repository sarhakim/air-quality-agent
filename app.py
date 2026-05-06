import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from plotly.subplots import make_subplots

from src.agent import app
from src.tools import compute_threshold, load_data

load_dotenv()


def get_agent_response(question: str, history: list[dict]) -> str:
    """Invokes the agent and returns the answer."""
    messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]
    messages.append(HumanMessage(content=question))
    result = app.invoke({"messages": messages})
    answer = result["messages"][-1].content
    if isinstance(answer, list):
        answer = " ".join(b.get("text", "") for b in answer if isinstance(b, dict))
    return answer

def get_aqi_label(aqi: float) -> tuple[str, str]:
    """Returns (label, color) for a given AQI value."""
    if aqi < 20: return "Good", "#00cc66"
    elif aqi < 40: return "Fair", "#88cc00"
    elif aqi < 60: return "Moderate", "#ffaa00"
    elif aqi < 80: return "Poor", "#ff6600"
    else: return "Very Poor", "#ff3b3b"


st.title("Air Quality Agent — Paris IDF")
tab_dashboard, tab_chat = st.tabs(["Dashboard", "Chat"])

with tab_dashboard:
    df = load_data()
    threshold = compute_threshold(df)

    # Current conditions
    now = pd.Timestamp.now(tz="UTC")
    latest = df[df["date"] <= now].dropna(subset=["temperature"]).iloc[-1]
    aqi_label, aqi_color = get_aqi_label(latest["european_aqi"])

    st.subheader("Current conditions")
    st.caption(f"Last update: {str(latest['date'])[:16]} UTC · Data may be up to 12h old (CAMS model)")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("AQI", f"{latest['european_aqi']:.0f}", aqi_label)
    c2.metric("PM2.5 (µg/m³)", f"{latest['pm2_5']:.1f}")
    c3.metric("NO₂ (µg/m³)", f"{latest['no2']:.1f}")
    c4.metric("O₃ (µg/m³)", f"{latest['o3']:.1f}")
    c5.metric("Wind (km/h)", f"{latest['wind_speed']:.1f}")
    c6.metric("Temp (°C)", f"{latest['temperature']:.1f}")

    st.divider()

    # AQI time series
    st.subheader("AQI — last 90 days")

    daily = df.groupby(df["date"].dt.date).agg(
        aqi_max=("european_aqi", "max"),
        aqi_mean=("european_aqi", "mean"),
        dust_mean=("dust", "mean"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["is_anomaly"] = daily["aqi_max"] > threshold
    anomalies = daily[daily["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["aqi_mean"],
        fill="tozeroy", fillcolor="rgba(74,158,255,0.1)",
        line=dict(color="#4a9eff", width=1.5),
        name="AQI mean",
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ff3b3b", line_width=1,
        annotation_text=f"Anomaly threshold ({threshold:.0f})",
        annotation_font_color="#ff3b3b", annotation_font_size=10,
    )
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["aqi_max"],
            mode="markers", marker=dict(color="#ff3b3b", size=9),
            name="Anomaly",
        ))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly list
    if not anomalies.empty:
        st.subheader("Detected anomalies")
        for _, row in anomalies.sort_values("date", ascending=False).iterrows():
            dust = " 🏜️ Saharan dust" if row["dust_mean"] > 5 else ""
            st.warning(f"**{row['date'].strftime('%B %d, %Y')}** — AQI max {row['aqi_max']:.0f}{dust}")
    else:
        st.success("No anomalies detected in the last 90 days.")

    st.divider()

    # Pollutant breakdown
    st.subheader("Pollutant breakdown — daily means")

    fig2 = make_subplots(rows=1, cols=4, subplot_titles=["PM2.5", "NO₂", "O₃", "Dust"])
    pollutants = [("pm2_5", 1), ("no2", 2), ("o3", 3), ("dust", 4)]
    colors = ["#4a9eff", "#aa66ff", "#ffaa00", "#ff6600"]

    for (col, idx), color in zip(pollutants, colors):
        d = df.groupby(df["date"].dt.date)[col].mean().reset_index()
        d.columns = ["date", "value"]
        d["date"] = pd.to_datetime(d["date"])
        fig2.add_trace(go.Scatter(
            x=d["date"], y=d["value"],
            line=dict(color=color, width=1.2),
            fill="tozeroy", fillcolor=f"{color}",
            showlegend=False,
        ), row=1, col=idx)

    fig2.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)


with tab_chat:
    st.subheader("Ask the agent")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask about air quality in Paris..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_agent_response(question, st.session_state.messages[:-1])
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})