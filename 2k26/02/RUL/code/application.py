import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin

from RUL_prediction.pipeline.predict import PredictionPipeline
from valutation import compute_valuation

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()


client_app = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/profiles/<path:filename>", methods=["GET"])
def profile_image(filename):
    return send_from_directory(BASE_DIR / "profiles", filename)


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train_route():
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        return jsonify({"status": "failed", "error": result.stderr}), 500
    return jsonify({"status": "ok", "message": "Training done successfully"})


@app.route("/testdata", methods=["GET"])
@cross_origin()
def test_data_route():
    try:
        cols = ["unit", "cycle"] + [f"f{i}" for i in range(1, 25)]
        test_df = pd.read_csv(
            BASE_DIR / "Data" / "test_FD001.txt",
            sep=r"\s+",
            header=None,
            engine="python",
        ).iloc[:, :26]
        test_df.columns = cols

        rul_df = pd.read_csv(
            BASE_DIR / "Data" / "RUL_FD001.txt",
            sep=r"\s+",
            header=None,
            engine="python",
        )

        max_cycles = test_df.groupby("unit")["cycle"].max()

        # Keep only unique records so table does not show duplicated rows.
        unique_df = test_df.drop_duplicates().head(10)

        records = []
        for idx, row in enumerate(unique_df.itertuples(index=False), start=1):
            row_dict = {"id": idx}
            for col in cols:
                value = getattr(row, col)
                if col in ["unit", "cycle"]:
                    row_dict[col] = int(value)
                else:
                    row_dict[col] = float(value)

            unit_id = int(row_dict["unit"])
            cycle = int(row_dict["cycle"])
            rul_end = int(rul_df.iloc[unit_id - 1, 0])
            true_rul = (int(max_cycles.loc[unit_id]) - cycle) + rul_end
            row_dict["rul"] = int(true_rul)
            records.append(row_dict)

        return jsonify({"rows": records})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["GET", "POST"])
@app.route("/predictdata", methods=["GET", "POST"])
@cross_origin()
def predict_route():
    if request.method == "GET":
        return render_template("index.html")

    payload = request.get_json()
    if payload is None:
        return (
            jsonify(
                {
                    "error": "Invalid JSON payload",
                    "hint": "Send a POST request with application/json body.",
                }
            ),
            400,
        )

    try:
        result = client_app.classifier.predict(payload)
        features = payload.get("features", payload) if isinstance(payload, dict) else {}
        unit_id = int(features.get("unit"))
        cycle = int(features.get("cycle"))
        valuation_result = compute_valuation(result, unit_id, cycle)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "rul": float(result),
            "valuation": valuation_result,
        }
    )


def run_local():
    port = int(os.environ.get("PORT", "5000"))
    url = f"http://127.0.0.1:{port}/predictdata"

    print("Starting server...")
    if os.environ.get("RUNNING_IN_DOCKER") != "1":
        webbrowser.open(url)

    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    run_local()
