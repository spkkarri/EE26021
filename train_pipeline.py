import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STAGES = [
    "src/RUL_prediction/pipeline/stage_01_data_ingestion.py",
    "src/RUL_prediction/pipeline/stage_02_prepare_base_model.py",
    "src/RUL_prediction/pipeline/stage_03_training.py",
    "src/RUL_prediction/pipeline/stage_04_evaluation.py",
]

EXPECTED_OUTPUTS = [
    "artifacts/data_ingestion/train_processed.csv",
    "artifacts/data_ingestion/test_processed.csv",
    "artifacts/training/model.h5",
    "artifacts/training/scaler.pkl",
    "artifacts/training/feature_columns.json",
    "scores.json",
]


def run_stage(stage_path: str):
    print(f"Running {stage_path}")
    result = subprocess.run(
        [sys.executable, stage_path],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Stage failed: {stage_path}")


def verify_outputs():
    missing = [path for path in EXPECTED_OUTPUTS if not (BASE_DIR / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected outputs: {missing}")


def main():
    for stage in STAGES:
        run_stage(stage)
    verify_outputs()
    print("Training pipeline test passed")


if __name__ == "__main__":
    main()
