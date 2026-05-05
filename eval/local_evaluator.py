import json
import os

import pandas as pd
from dotenv import load_dotenv

from eval.base_evaluator import BaseEvaluator
from eval.config import BASE_DIR
from src.agent import AgentModel

load_dotenv()


class Evaluator(BaseEvaluator):
    """Local evaluation pipeline with LLM-as-judge."""

    def __init__(
        self,
        agent_model: AgentModel = AgentModel.HAIKU,
        output_name: str = "results",
    ):
        super().__init__(agent_model=agent_model)
        runs_dir = os.path.join(BASE_DIR, "eval", "runs")
        os.makedirs(runs_dir, exist_ok=True)
        self.output_path = os.path.join(runs_dir, f"results_{output_name}.json")

    def run(self) -> list[dict]:
        """Runs the full evaluation pipeline and saves results."""
        results = [self._evaluate_case(case) for case in self.dataset]

        total = len(results)
        tools_ok = sum(r["tools_ok"] for r in results)
        mention_ok = sum(r["mention_ok"] for r in results)
        avg_score = sum(r["judge_score"] for r in results) / total

        print("\n=== EVALUATION SUMMARY ===")
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

    def _save_results(self, results: list[dict]) -> None:
        """Saves evaluation results to JSON and CSV."""
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        df = pd.DataFrame(results)
        df["tools_called"] = df["tools_called"].apply(lambda x: ", ".join(x))
        df["expected_tools"] = df["expected_tools"].apply(lambda x: ", ".join(x))
        df.to_csv(self.output_path.replace(".json", ".csv"), index=False)
