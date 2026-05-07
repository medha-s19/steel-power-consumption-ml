from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .data_preprocessing import preprocess_data
    from .feature_engineering import build_features
    from .model_training import load_model_bundle
except ImportError:
    from data_preprocessing import preprocess_data
    from feature_engineering import build_features
    from model_training import load_model_bundle


def predict(model_path: str | Path, feature_frame: pd.DataFrame) -> pd.Series:
    """Generate predictions using a saved model."""
    if not Path(model_path).exists():
        raise FileNotFoundError("Model has not been trained yet.")

    artifact = load_model_bundle(model_path)
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns", [])
    if feature_columns:
        feature_frame = feature_frame.reindex(columns=feature_columns, fill_value=0)

    predictions = model.predict(feature_frame)
    return pd.Series(predictions, index=feature_frame.index, name="prediction")


def predict_from_raw(model_path: str | Path, raw_input: dict) -> float:
    """Generate one prediction from raw user input values."""
    if not Path(model_path).exists():
        raise FileNotFoundError("Model has not been trained yet.")

    if not raw_input:
        raise ValueError("Prediction input is empty.")

    artifact = load_model_bundle(model_path)
    feature_columns = artifact.get("feature_columns", [])
    frame = preprocess_data(pd.DataFrame([raw_input]))
    features, _ = build_features(frame)

    if feature_columns:
        features = features.reindex(columns=feature_columns, fill_value=0)
        
    return float(artifact["model"].predict(features)[0])
