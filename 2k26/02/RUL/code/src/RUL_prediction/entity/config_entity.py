from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    rul_data_path: Path
    processed_train_path: Path
    processed_test_path: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    train_data_path: Path
    model_path: Path
    feature_columns_path: Path


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    test_data_path: Path
    rul_data_path: Path
    model_path: Path
    feature_columns_path: Path
    metric_file_path: Path
