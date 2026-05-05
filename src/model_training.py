from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from .data_preprocessing import TARGET_COLUMN, load_data, preprocess_data
    from .feature_engineering import build_features
except ImportError:
    from data_preprocessing import TARGET_COLUMN, load_data, preprocess_data
    from feature_engineering import build_features


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "Steel_industry_data.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model.pkl"


def get_available_models(random_state: int = 42) -> Dict[str, Any]:
    """Return simple regression models available in the current environment."""
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            random_state=random_state,
            n_jobs=1,
        ),
        "Linear Regression": LinearRegression(),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=1,
        )
    except ImportError:
        pass

    return models


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Calculate common regression metrics."""
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
    }


def prepare_training_data(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean raw data and return model-ready features and target."""
    if frame.empty:
        raise ValueError("Dataset is empty.")

    cleaned_frame = preprocess_data(frame)
    if cleaned_frame.empty:
        raise ValueError("Dataset is empty after preprocessing.")

    if target_column not in cleaned_frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    features, target = build_features(cleaned_frame, target_column=target_column)
    if target is None or target.empty:
        raise ValueError(f"Target column '{target_column}' could not be prepared.")

    if features.empty:
        raise ValueError("No feature columns were available for training.")

    return features, target


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = "Random Forest",
    random_state: int = 42,
) -> Tuple[Any, Dict[str, float]]:
    """Train the selected regression model and return metrics."""
    if len(features) < 2:
        raise ValueError("At least two rows are required to train and evaluate a model.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    models = get_available_models(random_state=random_state)
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not available.")

    model = models[model_name]
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    return model, metrics


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train all available models and compare their metrics."""
    if len(features) < 2:
        raise ValueError("At least two rows are required to compare models.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    rows = []
    for model_name, model in get_available_models(random_state=random_state).items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        rows.append({"model": model_name, **metrics})

    return pd.DataFrame(rows).sort_values("rmse")


def save_model(
    model: Any,
    model_path: str | Path,
    feature_columns: list[str] | None = None,
    metrics: Dict[str, float] | None = None,
    model_name: str | None = None,
) -> Path:
    """Persist a trained model plus metadata to disk."""
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns or [],
        "metrics": metrics or {},
    }
    joblib.dump(artifact, output_path)
    return output_path


def load_model_bundle(model_path: str | Path) -> Dict[str, Any]:
    """Load a persisted model artifact."""
    artifact = joblib.load(Path(model_path))
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {"model": artifact, "feature_columns": [], "metrics": {}}


def load_model(model_path: str | Path) -> Any:
    """Load only the estimator from a persisted model artifact."""
    return load_model_bundle(model_path)["model"]


def train_from_csv(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    target_column: str = TARGET_COLUMN,
) -> Tuple[Path, Dict[str, float]]:
    """Train the model from a CSV file and save the artifact."""
    features, target = prepare_training_data(load_data(data_path), target_column=target_column)
    model, metrics = train_model(features, target)
    saved_path = save_model(
        model,
        model_path,
        feature_columns=list(features.columns),
        metrics=metrics,
        model_name="Random Forest",
    )
    return saved_path, metrics


if __name__ == "__main__":
    saved_model, training_metrics = train_from_csv()
    print(f"Model saved to: {saved_model}")
    print(f"MSE: {training_metrics['mse']:.4f}")
    print(f"RMSE: {training_metrics['rmse']:.4f}")
    print(f"R2: {training_metrics['r2']:.4f}")
