import json
from unittest import mock
from unittest.mock import MagicMock, patch

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tools import detect_anomaly, get_weather_context

import src.agent
from src.agent import TOOLS, route_after_anomaly, route_from_agent, weather_from_anomaly
from tests.test_tools import make_mock_df
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from src.agent import build_agent, AgentModel

def make_state_with_tool_call(tool_name: str) -> dict:
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"id": "x", "name": tool_name, "args": {}, "type": "tool_call"}]
    )]}


def make_state_with_tool_message(content: dict) -> dict:
    return {"messages": [ToolMessage(
        content=json.dumps(content),
        tool_call_id="x"
    )]}

def test_graph_calls_weather_after_anomaly():
    """When detect_anomaly returns is_anomaly=true, get_weather_context is called."""
    

    # Mock the LLM to return a tool call for detect_anomaly
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "detect_anomaly",
            "args": {"date": "2026-03-06"},
            "type": "tool_call"
        }]
    )


    def fake_detect_anomaly(date: str) -> dict:
        return {
        "date": "2026-03-06",
        "is_anomaly": True,
        "aqi_max": 72.5,
        "aqi_threshold": 61.9,
        "pollutants": {},
        "main_pollutant": "dust"
    }

    @mock.create_autospec
    def fake_get_weather_context(date: str) -> dict:
        return {
        "date": "2026-03-06",
        "wind_speed_mean": 3.1,
        "precipitation_total": 0.0,
        "humidity_mean": 87.6,
        "temperature_mean": 10.8,
        "temperature_max": 16.5,
        "wind_speed_min": 1.7,
    }

    with patch("src.agent._build_llm", return_value=mock_llm), \
         patch("src.tools.load_data", return_value=make_mock_df()), \
         patch("src.agent.detect_anomaly.func", fake_detect_anomaly), \
         patch("src.agent.get_weather_context.func", fake_get_weather_context) as mock_get_weather_context:
        
        # After weather is called, LLM should return final answer
        mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{
                "id": "call_1",
                "name": "detect_anomaly", 
                "args": {"date": "2026-03-06"},
                "type": "tool_call"
            }]),
            AIMessage(content="Oui, anomalie détectée.")  # final answer
        ]
            
        app = build_agent()
        result = app.invoke({
            "messages": [HumanMessage(content="Y a-t-il eu une anomalie le 6 mars ?")]
        })

    mock_get_weather_context.assert_called_once_with(date="2026-03-06")
    assert len(result) == 6
    assert result["messages"][-1] == AIMessage(content="Oui, anomalie détectée.")


class TestRouteFromAgent:
    def test_routes_to_detect_anomaly(self):
        state = make_state_with_tool_call("detect_anomaly")
        assert route_from_agent(state) == "detect_anomaly"

    def test_routes_to_get_current_data(self):
        state = make_state_with_tool_call("get_current_data")
        assert route_from_agent(state) == "get_current_data"

    def test_routes_to_summary(self):
        state = make_state_with_tool_call("summarize_situation")
        assert route_from_agent(state) == "summary"

    def test_routes_to_get_weather(self):
        state = make_state_with_tool_call("get_weather_context")
        assert route_from_agent(state) == "get_weather"

    def test_routes_to_end_when_no_tool_calls(self):
        state = {"messages": [AIMessage(content="Final answer")]}
        from langgraph.graph import END
        assert route_from_agent(state) == END


class TestRouteAfterAnomaly:
    def test_routes_to_weather_when_anomaly(self):
        state = make_state_with_tool_message({"is_anomaly": True, "date": "2026-03-06"})
        assert route_after_anomaly(state) == "weather_from_anomaly"

    def test_routes_to_agent_when_no_anomaly(self):
        state = make_state_with_tool_message({"is_anomaly": False, "date": "2026-02-01"})
        assert route_after_anomaly(state) == "agent"

    def test_routes_to_agent_on_error(self):
        state = make_state_with_tool_message({"error": "No data available"})
        assert route_after_anomaly(state) == "agent"

    def test_routes_to_agent_on_invalid_json(self):
        state = {"messages": [ToolMessage(content="not json", tool_call_id="x")]}
        assert route_after_anomaly(state) == "agent"


class TestWeatherFromAnomaly:
    def test_injects_correct_date(self):
        state = make_state_with_tool_message({"is_anomaly": True, "date": "2026-03-06"})
        result = weather_from_anomaly(state)
        tool_call = result["messages"][0].tool_calls[0]
        assert tool_call["name"] == "get_weather_context"
        assert tool_call["args"]["date"] == "2026-03-06"

    def test_returns_empty_when_no_date(self):
        state = make_state_with_tool_message({"is_anomaly": True})
        result = weather_from_anomaly(state)
        assert result["messages"] == []
