from pathlib import Path

import pandas as pd

from RUL_prediction import logger
from RUL_prediction.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.columns = ["unit", "cycle"] + [f"f{i}" for i in range(1, 25)]

    def _read_raw(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] > len(self.columns):
            df = df.iloc[:, : len(self.columns)]
        df.columns = self.columns
        return df

    def _add_rul_target(self, train_df: pd.DataFrame) -> pd.DataFrame:
        max_cycles = train_df.groupby("unit")["cycle"].max().rename("max_cycle")
        merged = train_df.merge(max_cycles, on="unit", how="left")
        merged["RUL"] = merged["max_cycle"] - merged["cycle"]
        return merged.drop(columns=["max_cycle"])

    def run(self):
        logger.info("Starting data ingestion")
        train_raw = self._read_raw(self.config.train_data_path)
        test_raw = self._read_raw(self.config.test_data_path)

        train_df = self._standardize(train_raw)
        test_df = self._standardize(test_raw)

        train_with_rul = self._add_rul_target(train_df)

        self.config.processed_train_path.parent.mkdir(parents=True, exist_ok=True)
        train_with_rul.to_csv(self.config.processed_train_path, index=False)
        test_df.to_csv(self.config.processed_test_path, index=False)

        logger.info("Saved processed train data to %s", self.config.processed_train_path)
        logger.info("Saved processed test data to %s", self.config.processed_test_path)
