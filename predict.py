"""
Predict house prices using the trained model saved in a .pkl file.

Usage examples:
    # Predict for a single house passed via command-line arguments
    python predict.py --area 7420 --bedrooms 4 --bathrooms 2 --stories 3 \
        --mainroad yes --guestroom no --basement no \
        --hotwaterheating no --airconditioning yes

    # Predict for every row in a CSV file (without a price column)
    python predict.py --input new_houses.csv --output predictions.csv
"""

import argparse
import os

import joblib
import pandas as pd

MODEL_PATH = "house_price_model.pkl"

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


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run 'python train_model.py' first."
        )
    return joblib.load(model_path)


def predict_single(model, args) -> float:
    """Build a one-row DataFrame from CLI args and predict its price."""
    row = {
        "area": args.area,
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "stories": args.stories,
        "mainroad": args.mainroad,
        "guestroom": args.guestroom,
        "basement": args.basement,
        "hotwaterheating": args.hotwaterheating,
        "airconditioning": args.airconditioning,
    }
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return float(model.predict(X)[0])


def predict_from_file(model, input_path: str, output_path: str) -> None:
    """Predict prices for every row in a CSV file."""
    df = pd.read_csv(input_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["predicted_price"] = model.predict(df[FEATURE_COLUMNS])
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} predictions to: {os.path.abspath(output_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict house prices from a trained model.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained .pkl model")

    # Batch prediction from a file
    parser.add_argument("--input", help="CSV file of houses to score")
    parser.add_argument("--output", default="predictions.csv", help="Where to write predictions")

    # Single prediction from CLI args
    parser.add_argument("--area", type=float, help="Lot area in square feet")
    parser.add_argument("--bedrooms", type=int, help="Number of bedrooms")
    parser.add_argument("--bathrooms", type=int, help="Number of bathrooms")
    parser.add_argument("--stories", type=int, help="Number of stories")
    parser.add_argument("--mainroad", choices=["yes", "no"], help="Connected to the main road")
    parser.add_argument("--guestroom", choices=["yes", "no"], help="Has a guest room")
    parser.add_argument("--basement", choices=["yes", "no"], help="Has a basement")
    parser.add_argument("--hotwaterheating", choices=["yes", "no"], help="Has hot water heating")
    parser.add_argument("--airconditioning", choices=["yes", "no"], help="Has air conditioning")

    args = parser.parse_args()
    model = load_model(args.model)

    if args.input:
        predict_from_file(model, args.input, args.output)
        return

    # Otherwise, run a single prediction. Fill in sensible defaults for any
    # value that was not provided so the script always produces an example.
    defaults = dict(
        area=7420, bedrooms=4, bathrooms=2, stories=3,
        mainroad="yes", guestroom="no", basement="no",
        hotwaterheating="no", airconditioning="yes",
    )
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    price = predict_single(model, args)
    print("\nInput house:")
    for col in FEATURE_COLUMNS:
        print(f"  {col:18}: {getattr(args, col)}")
    print(f"\nPredicted price: {price:,.2f}")


if __name__ == "__main__":
    main()
