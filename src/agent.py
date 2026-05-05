from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.tools import detect_anomaly, get_current_data, get_weather_context, summarize_situation

TOOLS = [detect_anomaly, get_weather_context, get_current_data, summarize_situation]

#llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(TOOLS)
#llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(TOOLS)
llm = ChatOllama(model="llama3.1:8b").bind_tools(TOOLS)

from langchain_core.messages import SystemMessage

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

def agent(state: MessagesState) -> dict:
    """Calls the LLM with the current state."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


def should_continue(state: MessagesState) -> str:
    """Routes to tools if the LLM called one, otherwise ends."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build graph
tool_node = ToolNode(TOOLS)

graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()