
import os
from enum import StrEnum

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.tools import detect_anomaly, get_current_data, get_weather_context, summarize_situation

TOOLS = [detect_anomaly, get_weather_context, get_current_data, summarize_situation]


SYSTEM_PROMPT = """You are an air quality monitoring assistant for Paris, Île-de-France.
You have access to tools that provide real-time and historical air quality and weather data.

Your role is to:
- Detect air quality anomalies based on European AQI standards
- Explain the likely causes using weather context (wind, precipitation, dust episodes)
- Provide clear, factual answers grounded in the data

Guidelines:
- Always use the available tools before answering — never guess
- PM2.5 refers to fine particulate matter in general. Dust refers specifically to Saharan dust episodes. Do not confuse them.
- When an anomaly is detected, you MUST immediately call get_weather_context for the same date — do not ask the user, just call it.
- When reporting z-scores, explain them as "X standard deviations above the historical average", not as a multiplier
- Mention multiple pollutants if several have elevated z-scores (> 1.5)
- Be concise but precise — cite the actual AQI values and thresholds
- If asked about a Saharan dust episode, check both dust and pm2_5 z-scores
- Answer in the same language as the user

Tool calling sequence for anomaly questions:
1. Call detect_anomaly
2. If is_anomaly is true → immediately call get_weather_context for the same date
3. Then provide your final answer
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

    def should_continue(state: MessagesState) -> str:
        """Routes to tools if the LLM called one, otherwise ends."""
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


app = build_agent()