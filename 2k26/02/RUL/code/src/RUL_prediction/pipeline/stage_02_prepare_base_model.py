from RUL_prediction import logger
from RUL_prediction.components.prepare_base_model import PrepareBaseModel
from RUL_prediction.config.configuration import ConfigurationManager


STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        stage_config = config.get_prepare_base_model_config()
        PrepareBaseModel(config=stage_config).run()


if __name__ == "__main__":
    try:
        logger.info("stage %s started", STAGE_NAME)
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info("stage %s completed", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise
