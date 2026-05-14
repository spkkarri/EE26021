import pandas as pd
import numpy as np
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

from RUL_prediction import logger
from RUL_prediction.entity.config_entity import TrainingConfig
from RUL_prediction.utils.common import save_json

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.sequence_length = 30

    def prepare_data(self, df):
        # Drop constant uninformative sensors for FD001
        drop_cols = ['f1', 'f5', 'f10', 'f16', 'f18', 'f19']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # Clip RUL (Piecewise Linear RUL)
        if 'RUL' in df.columns:
            df['RUL'] = df['RUL'].clip(upper=125)
            
        return df

    def generate_sequences(self, df, feature_cols):
        X, y = [], []
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id]
            features = unit_data[feature_cols].values
            
            if 'RUL' in df.columns:
                labels = unit_data['RUL'].values
            else:
                labels = None

            for i in range(len(unit_data) - self.sequence_length + 1):
                X.append(features[i:i + self.sequence_length])
                if labels is not None:
                    y.append(labels[i + self.sequence_length - 1])
                    
        return np.array(X), (np.array(y) if labels is not None else None)

    def run(self):
        logger.info("Starting CNN-LSTM model training")
        
        df = pd.read_csv(self.config.train_data_path)
        df = self.prepare_data(df)
        
        feature_columns = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL']]
        
        # Normalization
        scaler = MinMaxScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Save scaler for inference
        scaler_path = self.config.model_path.parent / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

        # Sequences
        X_train, y_train = self.generate_sequences(df, feature_columns)
        
        # Build CNN-LSTM
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.sequence_length, len(feature_columns))),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        logger.info("Fitting CNN-LSTM Model...")
        model.fit(X_train, y_train, epochs=25, batch_size=200, callbacks=[early_stop], verbose=1)
        
        # Save model
        model_path = str(self.config.model_path)
        model.save(model_path)
        
        save_json(self.config.feature_columns_path, {"feature_columns": feature_columns})
        logger.info(f"Training completed. Model saved to {model_path}")
