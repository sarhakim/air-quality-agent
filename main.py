from dotenv import load_dotenv

from eval.evaluator import Evaluator
from src.agent import AgentModel

load_dotenv()

if __name__ == "__main__":
    evaluator = Evaluator(agent_model=AgentModel.HAIKU)
    evaluator.run_langsmith(experiment_prefix="haiku-v1")  # via LangSmith
    # evaluator.run()  # via notre pipeline local