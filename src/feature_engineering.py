from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def build_features(
    frame: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Create model-ready features from a cleaned dataframe."""
    working_frame = frame.copy()
    target = None

    if target_column and target_column in working_frame.columns:
        target = working_frame.pop(target_column)

    feature_frame = pd.get_dummies(working_frame, drop_first=False)
    return feature_frame, target
