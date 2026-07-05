"""
Flask API that serves the trained house price model via a /predict endpoint.

Run:
    python app.py
The server starts on http://127.0.0.1:5001

Endpoints:
    GET  /         -> basic info + usage
    GET  /health   -> health check (confirms the model is loaded)
    POST /predict  -> predict price(s) from JSON

Examples:
    # Single house
    curl -X POST http://127.0.0.1:5001/predict \
        -H "Content-Type: application/json" \
        -d '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
             "mainroad": "yes", "guestroom": "no", "basement": "no",
             "hotwaterheating": "no", "airconditioning": "yes"}'

    # Multiple houses (send a JSON list)
    curl -X POST http://127.0.0.1:5001/predict \
        -H "Content-Type: application/json" \
        -d '[{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
              "mainroad": "yes", "guestroom": "no", "basement": "no",
              "hotwaterheating": "no", "airconditioning": "yes"}]'
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", "house_price_model.pkl")

FEATURE_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
]

app = Flask(__name__)

# Load the model once at startup so every request reuses it.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Run 'python train_model.py' first."
    )
model = joblib.load(MODEL_PATH)


def validate_and_frame(payload):
    """Turn the incoming JSON into a validated DataFrame.

    Accepts either a single object or a list of objects. Raises ValueError with
    a helpful message if required feature columns are missing.
    """
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise ValueError("Request body is empty.")

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Item {i} must be a JSON object, got {type(record).__name__}.")
        missing = [c for c in FEATURE_COLUMNS if c not in record]
        if missing:
            raise ValueError(f"Item {i} is missing required fields: {missing}")

    return pd.DataFrame(records, columns=FEATURE_COLUMNS)


@app.get("/")
def index():
    return jsonify(
        {
            "service": "House Price Prediction API",
            "endpoints": {
                "GET /health": "Health check",
                "POST /predict": "Predict price(s) from JSON",
            },
            "required_fields": FEATURE_COLUMNS,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "model_path": MODEL_PATH})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return (
            jsonify({"error": "Request body must be valid JSON with Content-Type: application/json."}),
            400,
        )

    try:
        X = validate_and_frame(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    preds = model.predict(X)
    predictions = [round(float(p), 2) for p in preds]

    # Return a single number for a single-object request, else a list.
    if not isinstance(payload, list):
        return jsonify({"predicted_price": predictions[0]})
    return jsonify({"predicted_prices": predictions})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
