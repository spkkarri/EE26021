import sys
from pathlib import Path

# Allow running this file directly without installing the package.
SRC_ROOT = Path(_file_).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
    
from RUL_prediction import logger
from RUL_prediction.components.data_ingestion import DataIngestion
from RUL_prediction.config.configuration import ConfigurationManager


STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()


if __name__ == "__main__":
    try:
        logger.info("stage %s started", STAGE_NAME)
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info("stage %s completed", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise
