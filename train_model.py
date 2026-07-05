"""
Train a house price prediction model on the Housing dataset and save it as a .pkl file.

Usage:
    python train_model.py
    python train_model.py --data Housing-selected-columns.csv --model house_price_model.pkl
"""

import argparse
import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV dataset into a DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected a '{TARGET}' column in the dataset.")
    return df


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    """Build a preprocessing + model pipeline.

    Numeric columns are standardized and categorical (yes/no) columns are
    one-hot encoded. Everything is bundled into a single pipeline so the saved
    .pkl can take raw feature values directly.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a house price prediction model.")
    parser.add_argument(
        "--data",
        default="Housing-selected-columns.csv",
        help="Path to the CSV dataset (default: Housing-selected-columns.csv)",
    )
    parser.add_argument(
        "--model",
        default="house_price_model.pkl",
        help="Output path for the trained model (default: house_price_model.pkl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing (default: 0.2)",
    )
    args = parser.parse_args()

    # 1. Load data
    df = load_data(args.data)
    print(f"Loaded dataset with shape {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2. Identify feature types automatically
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numeric features:     {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # 4. Build and train the pipeline
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)


    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("my-first-experiment")

    # 5. Evaluate
    preds = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_param("test_size", args.test_size)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)


    print("\nModel performance on the held-out test set:")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  R^2  : {r2:.4f}")

    # 6. Persist the whole pipeline (preprocessing + model) to a .pkl file
    joblib.dump(pipeline, args.model)
    print(f"\nModel saved to: {os.path.abspath(args.model)}")


if __name__ == "__main__":
    main()
