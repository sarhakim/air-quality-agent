# eval/compare.py

import json
import os

import pandas as pd
from dotenv import load_dotenv

from eval.config import DATASET_PATH, RUNS_DIR
from eval.evaluator import Evaluator
from src.agent import AgentModel

load_dotenv()


def run_evaluations() -> None:
    """Runs evaluation for both Haiku and Llama models."""

    os.makedirs(RUNS_DIR, exist_ok=True)

    for model, name in [(AgentModel.HAIKU, "haiku"), (AgentModel.LLAMA, "llama")]:
        print(f"\n{'='*50}")
        print(f"Running evaluation with {name.upper()}")
        print(f"{'='*50}")
        Evaluator(agent_model=model, output_name=name).run()


def load_run(name: str) -> pd.DataFrame:
    """Loads a run's results and enriches with dataset context."""
    path = os.path.join(RUNS_DIR, f"results_{name}.json")
    with open(path) as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    df["model"] = name

    # Enrich with dataset ground truth
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    df_dataset = pd.DataFrame([
        {
            "id": case["id"],
            "ground_truth": case["ground_truth"],
            "expected_is_anomaly": case["expected"].get("is_anomaly"),
        }
        for case in dataset
    ])

    return df.merge(df_dataset, on="id", how="left")


def compare() -> pd.DataFrame:
    """Merges Haiku and Llama results and prints comparison summary."""
    df_haiku = load_run("haiku")
    df_llama = load_run("llama")

    df = pd.concat([df_haiku, df_llama], ignore_index=True)
    df["tools_called"] = df["tools_called"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )
    df["expected_tools"] = df["expected_tools"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )

    # Summary by model
    print("\n=== COMPARISON SUMMARY ===")
    summary = df.groupby("model").agg(
        tools_ok=("tools_ok", "mean"),
        mention_ok=("mention_ok", "mean"),
        avg_score=("judge_score", "mean"),
        total=("id", "count"),
    ).round(2)
    print(summary)

    # Summary by model + category
    print("\n=== BY CATEGORY ===")
    by_cat = df.groupby(["model", "category"]).agg(
        avg_score=("judge_score", "mean"),
        tools_ok=("tools_ok", "mean"),
    ).round(2)
    print(by_cat)

    # Save
    output_path = os.path.join(RUNS_DIR, "comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nComparison saved to {output_path}")
    return df


if __name__ == "__main__":
    run_evaluations()
    compare()