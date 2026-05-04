from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


TARGET_COLUMN = "Usage_kWh"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the steel energy consumption dataset."""
    return pd.read_csv(Path(path))


def preprocess_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw steel energy consumption records."""
    cleaned = frame.copy()
    cleaned = cleaned.dropna()

    if "date" in cleaned.columns:
        dates = pd.to_datetime(cleaned["date"], dayfirst=True, errors="coerce")
        cleaned = cleaned.assign(
            hour=dates.dt.hour,
            month=dates.dt.month,
            day_of_month=dates.dt.day,
        )
        cleaned = cleaned.drop(columns=["date"])
        cleaned = cleaned.dropna()

    return cleaned


def load_and_preprocess(
    path: str | Path,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data, clean it, and split it into feature and target columns."""
    cleaned = preprocess_data(load_data(path))
    if target_column not in cleaned.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    features = cleaned.drop(columns=[target_column])
    target = cleaned[target_column]
    return features, target