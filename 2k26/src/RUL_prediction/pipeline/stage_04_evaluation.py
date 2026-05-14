from RUL_prediction import logger
from RUL_prediction.components.evaluation import Evaluation
from RUL_prediction.config.configuration import ConfigurationManager


STAGE_NAME = "Evaluation"


class EvaluationPipeline:
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluator = Evaluation(config=evaluation_config)
        evaluator.run()


if __name__ == "__main__":
    try:
        logger.info("stage %s started", STAGE_NAME)
        obj = EvaluationPipeline()
        obj.main()
        logger.info("stage %s completed", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise
