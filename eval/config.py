import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "eval", "eval_dataset.json")
RUNS_DIR = os.path.join(BASE_DIR, "eval", "runs")
