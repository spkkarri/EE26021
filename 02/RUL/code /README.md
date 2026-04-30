# RUL Prediction Project

End-to-end Remaining Useful Life (RUL) prediction system for NASA CMAPSS FD001 turbofan engine data.

This project includes:

- data ingestion and preprocessing
- CNN-LSTM model training
- evaluation (RMSE, MAE)
- Flask API for prediction and valuation
- pipeline testing with artifact checks

## Demo Video

- Project demo: https://drive.google.com/file/d/1M-_lYp8kUR0XsWfL9QgXQYeuBkEZS_YZ/view?usp=sharing

## Problem Statement

Given multivariate engine sensor data across operating cycles, predict each engine's Remaining Useful Life (RUL), i.e., how many cycles remain before failure.

## Dataset

This project uses NASA CMAPSS FD001 files in [Data](Data):

- [Data/train_FD001.txt](Data/train_FD001.txt): run-to-failure training trajectories
- [Data/test_FD001.txt](Data/test_FD001.txt): partial test trajectories
- [Data/RUL_FD001.txt](Data/RUL_FD001.txt): true final RUL for each test engine

## Project Structure

- [main.py](main.py): runs the 4-stage training/evaluation pipeline
- [test_pipeline.py](test_pipeline.py): stage-by-stage test runner
- [application.py](application.py): Flask app entry (UI + API)
- [app.py](app.py): small app launcher wrapper
- [config/config.yaml](config/config.yaml): central path and artifact configuration
- [dvc.yaml](dvc.yaml): DVC pipeline stage definitions
- [src/RUL_prediction/components](src/RUL_prediction/components): core stage implementations
- [src/RUL_prediction/pipeline](src/RUL_prediction/pipeline): stage wrappers + inference pipeline
- [templates/index.html](templates/index.html): frontend UI
- [artifacts](artifacts): generated model/data artifacts

## Pipeline Workflow

The pipeline runs in this order:

1. Data Ingestion
   - script: [src/RUL_prediction/pipeline/stage_01_data_ingestion.py](src/RUL_prediction/pipeline/stage_01_data_ingestion.py)
   - component: [src/RUL_prediction/components/data_ingestion.py](src/RUL_prediction/components/data_ingestion.py)
   - outputs:
     - [artifacts/data_ingestion/train_processed.csv](artifacts/data_ingestion/train_processed.csv)
     - [artifacts/data_ingestion/test_processed.csv](artifacts/data_ingestion/test_processed.csv)

2. Prepare Base Model (structural placeholder)
   - script: [src/RUL_prediction/pipeline/stage_02_prepare_base_model.py](src/RUL_prediction/pipeline/stage_02_prepare_base_model.py)

3. Training
   - script: [src/RUL_prediction/pipeline/stage_03_training.py](src/RUL_prediction/pipeline/stage_03_training.py)
   - component: [src/RUL_prediction/components/training.py](src/RUL_prediction/components/training.py)
   - model: CNN-LSTM (Conv1D + LSTM)
   - outputs:
     - [artifacts/training/model.h5](artifacts/training/model.h5)
     - [artifacts/training/scaler.pkl](artifacts/training/scaler.pkl)
     - [artifacts/training/feature_columns.json](artifacts/training/feature_columns.json)

4. Evaluation
   - script: [src/RUL_prediction/pipeline/stage_04_evaluation.py](src/RUL_prediction/pipeline/stage_04_evaluation.py)
   - component: [src/RUL_prediction/components/evaluation.py](src/RUL_prediction/components/evaluation.py)
   - output metric file: [scores.json](scores.json)

## Configuration

All important paths are controlled in [config/config.yaml](config/config.yaml):

- dataset paths
- artifact directories
- model and scaler paths
- metric output path

Configuration objects are created in [src/RUL_prediction/config/configuration.py](src/RUL_prediction/config/configuration.py).

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Run full training + evaluation:

```bash
python main.py
```

Run pipeline test:

```bash
python test_pipeline.py
```

Start Flask app:

```bash
python application.py
```

## API Endpoints

Base app file: [application.py](application.py)

- `GET /`
  - returns the web UI

- `GET /health`
  - health check
  - response: `{ "status": "ok" }`

- `GET /train`
  - triggers [main.py](main.py) in a subprocess

- `GET /testdata`
  - returns sample test rows with computed true RUL values

- `POST /predict` or `POST /predictdata`
  - runs inference and valuation
  - expects JSON containing features (or a wrapped `features` object)

Example request body:

```json
{
  "features": {
    "unit": 1,
    "cycle": 20,
    "f2": 0.5,
    "f3": 0.1
  }
}
```

Note:

- the payload must include all feature columns used during training (see [artifacts/training/feature_columns.json](artifacts/training/feature_columns.json)).

## Inference and Valuation

- Inference pipeline: [src/RUL_prediction/pipeline/predict.py](src/RUL_prediction/pipeline/predict.py)
  - loads model from [artifacts/training/model.h5](artifacts/training/model.h5)
  - loads scaler from [artifacts/training/scaler.pkl](artifacts/training/scaler.pkl)
  - applies sequence windowing (`sequence_length = 30`)

- Valuation helper: [valutation.PY](valutation.PY)
  - computes true RUL and prediction error statistics

## Testing

[test_pipeline.py](test_pipeline.py) validates the pipeline by:

- running each stage script in order
- failing fast if any stage exits with non-zero status
- checking required outputs exist

Expected checked outputs:

- [artifacts/data_ingestion/train_processed.csv](artifacts/data_ingestion/train_processed.csv)
- [artifacts/data_ingestion/test_processed.csv](artifacts/data_ingestion/test_processed.csv)
- [artifacts/training/model.h5](artifacts/training/model.h5)
- [artifacts/training/scaler.pkl](artifacts/training/scaler.pkl)
- [artifacts/training/feature_columns.json](artifacts/training/feature_columns.json)
- [scores.json](scores.json)

## DVC Pipeline

Pipeline definition is in [dvc.yaml](dvc.yaml), with stages:

- `data_ingestion`
- `prepare_base_model`
- `training`
- `evaluation`

## Deployment

### Local

```bash
python application.py
```

### Render / Gunicorn

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

## Dependencies

All dependencies are listed in [requirements.txt](requirements.txt).

## Team

Project by Team RUL Prediction.
