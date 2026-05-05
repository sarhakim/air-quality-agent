
from dotenv import load_dotenv
from langsmith.evaluation import EvaluationResult, evaluate

from eval.base_evaluator import BaseEvaluator
from src.agent import AgentModel

load_dotenv()

LANGSMITH_DATASET = "air-quality-agent-eval"


class LangSmithEvaluator(BaseEvaluator):
    """LangSmith evaluation pipeline with custom evaluators."""

    def __init__(self, agent_model: AgentModel = AgentModel.HAIKU):
        super().__init__(agent_model=agent_model)
        self.dataset_by_id = {case["id"]: case for case in self.dataset}

    def run(self, experiment_prefix: str = "haiku") -> None:
        """Runs evaluation via LangSmith Experiments."""
        evaluate(
            self._run_agent_on_example,
            data=LANGSMITH_DATASET,
            experiment_prefix=experiment_prefix,
            evaluators=[
                self._tools_ok_evaluator,
                self._judge_evaluator,
            ],
        )

    def _run_agent_on_example(self, example: dict) -> dict:
        """Runs the agent on a LangSmith dataset example."""
        agent_output = self._invoke_agent(example["inputs_1"])
        return {"output": agent_output["answer"], "tools_called": agent_output["tools_called"]}

    def _tools_ok_evaluator(self, run, example) -> EvaluationResult:
        """Checks if the agent called the expected tools."""
        case_id = example.outputs.get("id")
        case = self.dataset_by_id.get(case_id, {})
        expected_tools = case.get("expected", {}).get("tools_called", [])
        tools_called = run.outputs.get("tools_called", [])
        score = int(all(t in tools_called for t in expected_tools))
        return EvaluationResult(key="tools_ok", score=score)

    def _judge_evaluator(self, run, example) -> EvaluationResult:
        """LLM-as-judge evaluator."""
        question = example.inputs.get("inputs_1", "")
        answer = run.outputs.get("output", "")
        ground_truth = example.outputs.get("outputs_1", "")
        result = self._llm_as_judge(question, answer, ground_truth)
        return EvaluationResult(key="judge_score", score=result["score"])


if __name__ == "__main__":
    LangSmithEvaluator(agent_model=AgentModel.HAIKU).run(experiment_prefix="haiku-v1")