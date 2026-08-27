"""
Temporal Feature & Cyclical Encoding Module

WHAT: Extracts time-of-day and day-of-week attributes from timestamps and transforms them using
cyclical sine/cosine encodings.

WHY: Time attributes are periodic/cyclical. Treating hour 23 (11 PM) and hour 0 (12 AM midnight) as linear numbers
creates an artificial numerical gap of |23 - 0| = 23, even though they are only 1 hour apart in reality.

FIRST-PRINCIPLES INTUITION & MATHEMATICS:
By mapping a periodic variable v in [0, M-1] onto a 2D unit circle via angle theta = (2 * pi * v) / M:
    sin_component = sin(2 * pi * v / M)
    cos_component = cos(2 * pi * v / M)

At hour 23: (sin(-0.26), cos(0.96))
At hour 0:  (sin(0.00), cos(1.00))
Euclidean distance between hour 23 and hour 0 on the unit circle is small (~0.26), matching physical reality.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def cyclical_encode(value: float, max_value: float) -> Tuple[float, float]:
    """
    Encode a periodic value onto a 2D unit circle (sin, cos).

    WHAT: Converts scalar v in [0, max_value-1] to (sin(2*pi*v/max_value), cos(2*pi*v/max_value)).

    WHY: Eliminates artificial numerical boundary discontinuities at midnight and week boundaries.
    """
    radians = 2.0 * np.pi * (value / max_value)
    return float(np.sin(radians)), float(np.cos(radians))


def generate_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = "played_at"
) -> pd.DataFrame:
    """
    Extract temporal attributes and cyclical encodings from a timestamp column.

    WHAT: Adds hour, day_of_week, is_weekend, hour_sin, hour_cos, day_sin, and day_cos columns.

    WHY: Music listening habits follow circadian (time-of-day) and weekly (workday vs weekend) cycles.
    """
    df_out = df.copy()

    if timestamp_col not in df_out.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    if not pd.api.types.is_datetime64_any_dtype(df_out[timestamp_col]):
        df_out[timestamp_col] = pd.to_datetime(df_out[timestamp_col], utc=True)

    dt = df_out[timestamp_col].dt

    # Standard integer representations
    df_out["hour"] = dt.hour
    df_out["day_of_week"] = dt.dayofweek # 0 = Monday, 6 = Sunday
    df_out["is_weekend"] = (df_out["day_of_week"] >= 5).astype(int)

    # Cyclical Sine/Cosine transformations
    # Hour of day (M = 24.0)
    hour_rad = 2.0 * np.pi * df_out["hour"] / 24.0
    df_out["hour_sin"] = np.sin(hour_rad)
    df_out["hour_cos"] = np.cos(hour_rad)

    # Day of week (M = 7.0)
    day_rad = 2.0 * np.pi * df_out["day_of_week"] / 7.0
    df_out["day_sin"] = np.sin(day_rad)
    df_out["day_cos"] = np.cos(day_rad)

    return df_out
