
import json
import os
from datetime import datetime, timedelta
from enum import StrEnum

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.tools import detect_anomaly, get_current_data, get_weather_context, summarize_situation

load_dotenv()

TOOLS = [detect_anomaly, get_weather_context, get_current_data, summarize_situation]

snapshot_end = datetime.now()
snapshot_start = snapshot_end - timedelta(days=90)

SYSTEM_PROMPT = f"""You are an air quality monitoring assistant for Paris, Île-de-France.
You have access to tools that provide real-time and historical air quality and weather data.

Your role is to:
- Detect air quality anomalies based on European AQI standards
- Explain the likely causes using weather context (wind, precipitation, dust episodes)
- Provide clear, factual answers grounded in the data

Guidelines:
- Always use the available tools before answering — never guess
- PM2.5 refers to fine particulate matter in general. Dust refers specifically to Saharan dust episodes. Do not confuse them.
- When reporting z-scores, explain them as "X standard deviations above the historical average", not as a multiplier
- Mention multiple pollutants if several have elevated z-scores (> 1.5)
- Be concise but precise — cite the actual AQI values and thresholds
- If asked about a Saharan dust episode, check both dust and pm2_5 z-scores
- For ozone (O3), always report max_day value. If max_day > 100 µg/m³, mention it is approaching the EEA threshold of 120 µg/m³.
- Data is available from {snapshot_start.strftime("%B %Y")} to {snapshot_end.strftime("%B %Y")}. If a date is outside this window, explain it is outside the snapshot range — do not say the date is "in the future".
- When reporting current data, always mention that measurements may be up to 12 hours old due to CAMS model update frequency.
- Answer in the same language as the user

Examples of correct behavior:

User: "Was the air quality bad on March 15, 2026?"
→ Call detect_anomaly("2026-03-15")
→ is_anomaly=true → immediately call get_weather_context("2026-03-15")
→ Answer mentioning the main pollutant z-score in standard deviations, AQI values, and meteorological context.

User: "What is the air quality right now?"
→ Call get_current_data()
→ Answer with all values. Always mention data may be up to 12h old due to CAMS update frequency.

User: "What was the air quality on December 1, 2025?"
→ Call detect_anomaly("2025-12-01")
→ Returns error → Answer: "No data available for this date. Data is available from February to May 2026."

User: "Was there a lot of ozone on April 15, 2026?"
→ If an anomaly is found, explain pollutant severity and meteorological context.
→ Always report O3 max_day (not mean_day). If max_day > 100 µg/m³, mention proximity to EEA threshold of 120 µg/m³.

User: "What caused the poor air quality on March 8, 2026?"
→ Call detect_anomaly("2026-03-08") → is_anomaly=true → Call get_weather_context("2026-03-08")
→ When wind_speed_mean < 5 km/h and precipitation=0, describe it as "anticyclonic stagnation conditions" — high pressure blocking wind and preventing pollutant dispersion.
→ Answer: "The anomaly was caused by Saharan dust + elevated PM2.5, amplified by anticyclonic stagnation conditions (weak wind of X km/h, no precipitation, high humidity)."
"""


class AgentModel(StrEnum):
    """Supported LLM models for the agent."""
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-20250514"
    LLAMA = "llama3.1:8b"


def _build_llm(model: AgentModel):
    """Instantiates the right LLM based on the model."""
    if model == AgentModel.LLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model.value).bind_tools(TOOLS)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model.value).bind_tools(TOOLS)


def build_agent(model: AgentModel = AgentModel.HAIKU):
    """Builds and returns the compiled LangGraph agent."""
    if model != AgentModel.LLAMA and not os.environ.get("ANTHROPIC_API_KEY"):
        raise OSError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    llm = _build_llm(model)

    def agent(state: MessagesState) -> dict:
        """Calls the LLM with the current state."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}
    
    def route_from_agent(state: MessagesState):
        last = state["messages"][-1]

        if not last.tool_calls:
            return END

        tool_name = last.tool_calls[0]["name"]

        if tool_name == "detect_anomaly":
            return "anomaly"

        if tool_name == "get_current_data":
            return "current"

        if tool_name == "summarize_situation":
            return "summary"

        if tool_name == "get_weather_context":
            return "weather"

        return END

    def route_after_anomaly(state: MessagesState):
        last = state["messages"][-1]
        result = json.loads(last.content)

        if result.get("is_anomaly") is True:
            return "weather"

        return "agent"
    
    def force_weather_context(state: MessagesState) -> dict:
        """Forces a get_weather_context call using the date from the last anomaly result."""
        for msg in reversed(state["messages"]):
            try:
                result = json.loads(msg.content)
                if date := result.get("date"):
                    return {"messages": [AIMessage(
                        content="",
                        tool_calls=[{
                            "id": "forced_weather",
                            "name": "get_weather_context",
                            "args": {"date": date},
                            "type": "tool_call"
                        }]
                    )]}
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
        return {"messages": []}

    anomaly_node = ToolNode([detect_anomaly])
    weather_node = ToolNode([get_weather_context])
    current_node = ToolNode([get_current_data])
    summary_node = ToolNode([summarize_situation])

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("anomaly", anomaly_node)
    graph.add_node("weather", weather_node)
    graph.add_node("current", current_node)
    graph.add_node("summary", summary_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_from_agent
    )

    graph.add_node("force_weather", force_weather_context)
    graph.add_conditional_edges("anomaly", route_after_anomaly, {
        "weather": "force_weather",
        "agent": "agent"
    })
    graph.add_edge("force_weather", "weather")

    graph.add_edge("weather", "agent")
    graph.add_edge("current", "agent")
    graph.add_edge("summary", "agent")

    return graph.compile()


app = build_agent(model=AgentModel.LLAMA)