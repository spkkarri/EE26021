import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from RUL_prediction import logger
from RUL_prediction.entity.config_entity import EvaluationConfig
from RUL_prediction.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sequence_length = 30

    def run(self):
        logger.info("Starting CNN-LSTM evaluation")
        test_df = pd.read_csv(self.config.test_data_path)
        rul_df = pd.read_csv(self.config.rul_data_path, sep=r'\s+', header=None, names=['RUL'])
        
        # Load model & scaler
        model_dir = Path("artifacts/training")
        model = tf.keras.models.load_model(str(model_dir / "model.h5"), compile=False)
        scaler = joblib.load(model_dir / "scaler.pkl")
        
        drop_cols = ['f1', 'f5', 'f10', 'f16', 'f18', 'f19']
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        feature_columns = [col for col in test_df.columns if col not in ['unit', 'cycle']]
        
        test_df[feature_columns] = scaler.transform(test_df[feature_columns])
        
        y_pred = []
        for unit_id in test_df['unit'].unique():
            unit_data = test_df[test_df['unit'] == unit_id]
            features = unit_data[feature_columns].values
            
            if len(features) < self.sequence_length:
                padding = np.tile(features[0], (self.sequence_length - len(features), 1))
                features = np.vstack((padding, features))
                
            seq = features[-self.sequence_length:]
            seq = np.expand_dims(seq, axis=0)
            
            pred = model.predict(seq, verbose=0)
            y_pred.append(pred[0][0])
            
        y_true = rul_df['RUL'].values
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        scores = {"rmse": rmse, "mae": mae}
        metric_file = Path(self.config.metric_file_path)
        metric_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(metric_file, scores)
        logger.info(f"Evaluation completed. RMSE: {rmse:.2f}, MAE: {mae:.2f}")
