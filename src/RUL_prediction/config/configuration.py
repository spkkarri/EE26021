from pathlib import Path

from RUL_prediction.constants import CONFIG_FILE_PATH
from RUL_prediction.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
)
from RUL_prediction.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            rul_data_path=Path(config.rul_data_path),
            processed_train_path=Path(config.processed_train_path),
            processed_test_path=Path(config.processed_test_path),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([Path(config.root_dir)])
        return PrepareBaseModelConfig(root_dir=Path(config.root_dir))

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        create_directories([Path(config.root_dir)])
        return PrepareCallbacksConfig(root_dir=Path(config.root_dir))

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        create_directories([Path(config.root_dir)])

        return TrainingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            model_path=Path(config.model_path),
            feature_columns_path=Path(config.feature_columns_path),
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        create_directories([Path(config.root_dir)])

        return EvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            rul_data_path=Path(config.rul_data_path),
            model_path=Path(config.model_path),
            feature_columns_path=Path(config.feature_columns_path),
            metric_file_path=Path(config.metric_file_path),
        )
