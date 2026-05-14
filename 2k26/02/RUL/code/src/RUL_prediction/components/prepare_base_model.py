from RUL_prediction import logger
from RUL_prediction.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def run(self):
        logger.info("Prepare base model stage completed (no-op for current CNN-LSTM workflow)")
