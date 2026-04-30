from RUL_prediction import logger
from RUL_prediction.components.training import Training
from RUL_prediction.config.configuration import ConfigurationManager


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        trainer = Training(config=training_config)
        trainer.run()


if __name__ == "__main__":
    try:
        logger.info("stage %s started", STAGE_NAME)
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info("stage %s completed", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise
