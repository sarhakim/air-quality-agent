import json

from anthropic import Anthropic
from langchain_core.messages import HumanMessage

from eval.config import DATASET_PATH
from src.agent import AgentModel, build_agent

JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's answer about air quality in Paris.

Question: {question}
Agent's answer: {answer}
Ground truth: {ground_truth}

Evaluate on: factual accuracy, completeness, clarity.
Respond with this JSON:
{{
    "score": <float between 0.0 and 1.0>,
    "factual_accuracy": <true/false>,
    "completeness": <true/false>,
    "reasoning": "<one sentence>"
}}"""


class BaseEvaluator:
    """Shared logic between local and LangSmith evaluators."""

    def __init__(self, agent_model: AgentModel = AgentModel.HAIKU):
        self.agent_app = build_agent(model=agent_model)
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> list[dict]:
        """Loads the evaluation dataset from JSON."""
        with open(DATASET_PATH) as f:
            return json.load(f)

    def _invoke_agent(self, question: str) -> dict:
        """Invokes the agent and extracts the answer and tools called."""
        result = self.agent_app.invoke({"messages": [HumanMessage(content=question)]})
        messages = result["messages"]

        tools_called = [
            tc["name"]
            for msg in messages
            if hasattr(msg, "tool_calls") and msg.tool_calls
            for tc in msg.tool_calls
        ]

        final_answer = messages[-1].content
        if isinstance(final_answer, list):
            final_answer = " ".join(b.get("text", "") for b in final_answer if isinstance(b, dict))

        return {"answer": final_answer, "tools_called": tools_called}

    def _llm_as_judge(self, question: str, answer: str, ground_truth: str) -> dict:
        """Uses llm to judge the quality of the agent's answer."""
        prompt = JUDGE_PROMPT.format(question=question, answer=answer, ground_truth=ground_truth)

        #response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
        #raw = response["message"]["content"].strip()

        response = Anthropic().messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "{"},
            ],
            temperature=0
        )
        raw = "{" + response.content[0].text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "score": 0,
                "factual_accuracy": False,
                "completeness": False,
                "reasoning": "Failed to parse judge response",
            }
