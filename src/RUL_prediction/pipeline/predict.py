import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # We look for the .h5 model and the scaler
        model_dir = Path("artifacts/training")
        self.model = tf.keras.models.load_model(str(model_dir / "model.h5"), compile=False)
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        
        import json
        with open(model_dir / "feature_columns.json", "r") as f:
            self.feature_cols = json.load(f)["feature_columns"]
            
        self.sequence_length = 30
        
    def predict(self, data):
        # Extract 'features' key if it exists (JSON wrapping)
        if isinstance(data, dict):
            if 'features' in data:
                data = data['features']
            # Wrap single dict into a list so DataFrame interprets it as 1 row with columns
            if isinstance(data, dict):
                data = [data]
                
        df = pd.DataFrame(data)
        
        # Ensure exact columns as passed during fit
        features = df[self.feature_cols]

        scaled_features = self.scaler.transform(features)

        if len(scaled_features) < self.sequence_length:
            padding = np.tile(scaled_features[-1], (self.sequence_length - len(scaled_features), 1))
            scaled_features = np.vstack((padding, scaled_features))
            
        # Take the last sequence_length cycles
        seq = scaled_features[-self.sequence_length:]
        seq = np.expand_dims(seq, axis=0) # Shape: (1, 30, num_features)
        
        prediction = self.model.predict(seq, verbose=0)
        return float(prediction[0][0])
