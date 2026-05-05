from dotenv import load_dotenv

load_dotenv()

import json
import os

import ollama
from langchain_core.messages import HumanMessage

from src.agent import AgentModel, build_agent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Evaluator:
    """Runs the full evaluation pipeline for the air quality agent."""

    def __init__(self, dataset_path: str | None = None, model: AgentModel = AgentModel.HAIKU):
        self.agent_app = build_agent(model=model)
        self.dataset_path = dataset_path or os.path.join(BASE_DIR, "eval", "eval_dataset.json")
        self.output_path = os.path.join(BASE_DIR, "eval", "results.json")
        self.dataset = self._load_dataset()

    def run(self) -> list[dict]:
        """Runs the full evaluation pipeline and saves results."""
        results = [self._evaluate_case(case) for case in self.dataset]

        total = len(results)
        print("\n=== EVALUATION SUMMARY ===")
        tools_ok = sum(r["tools_ok"] for r in results)
        mention_ok = sum(r["mention_ok"] for r in results)
        avg_score = sum(r["judge_score"] for r in results) / total
        print(
            f"Total: {total} | Tools OK: {tools_ok}/{total} "
            f"| Mention OK: {mention_ok}/{total} | Avg: {avg_score:.2f}/5"
        )
        self._save_results(results)
        return results

    def _evaluate_case(self, case: dict) -> dict:
        """Evaluates a single case from the dataset."""
        print(f"\n[{case['id']}] {case['question']}")

        agent_output = self._invoke_agent(case["question"])
        expected = case["expected"]

        tools_ok = all(t in agent_output["tools_called"] for t in expected.get("tools_called", []))
        must_mention = expected.get("must_mention", [])
        mention_ok = any(k.lower() in agent_output["answer"].lower() for k in must_mention) if must_mention else True
        judge = self._llm_as_judge(case["question"], agent_output["answer"], case["ground_truth"])

        print(f"tools_ok={tools_ok} | mention_ok={mention_ok} | judge_score={judge['score']}/5")
        print(f"reasoning: {judge['reasoning']}")

        return {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],          
            "agent_answer": agent_output["answer"],
            "tools_ok": tools_ok,
            "mention_ok": mention_ok,
            "judge_score": judge["score"],
            "judge_reasoning": judge["reasoning"],
            "tools_called": agent_output["tools_called"],
            "expected_tools": expected.get("tools_called", []),
        }

    def _load_dataset(self) -> list[dict]:
        """Loads the evaluation dataset from JSON."""
        with open(self.dataset_path) as f:
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
        """Uses llama to judge the quality of the agent's answer."""
        prompt = f"""You are evaluating the quality of an AI assistant's answer about air quality in Paris.
            Question: {question}
            Agent's answer: {answer}
            Ground truth: {ground_truth}

            Evaluate on: factual accuracy, completeness, clarity.
            Respond ONLY with this JSON:
            {{
                "score": <1-5>,
                "factual_accuracy": <true/false>,
                "completeness": <true/false>,
                "reasoning": "<one sentence>"
            }}"""

        response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
        raw = response["message"]["content"].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "score": 0,
                "factual_accuracy": False,
                "completeness": False,
                "reasoning": "Failed to parse judge response",
            }

    def _save_results(self, results: list[dict]) -> None:
        """Saves evaluation results to JSON and CSV."""
        import pandas as pd

        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        df = pd.DataFrame(results)
        df["tools_called"] = df["tools_called"].apply(lambda x: ", ".join(x))
        df["expected_tools"] = df["expected_tools"].apply(lambda x: ", ".join(x))
        df.to_csv(self.output_path.replace(".json", ".csv"), index=False)

if __name__ == "__main__":
    Evaluator(model=AgentModel.LLAMA).run()