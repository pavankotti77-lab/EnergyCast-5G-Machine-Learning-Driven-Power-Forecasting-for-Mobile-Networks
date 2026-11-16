"""Feature engineering pipeline for 5G base station energy prediction.

Computes calendar features, cyclical sine/cosine encodings, causal lag features,
strictly non-leaking rolling averages, interaction terms, and chronological splitting.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: List[str] = [
    # Core telemetry
    "data_traffic_gb",
    "active_users",
    "prb_load_pct",
    "num_active_carriers",
    "temperature_c",
    "is_weekend",
    "is_holiday",
    # Calendar & cyclical
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Interaction terms
    "traffic_per_user",
    "load_x_carriers",
    "cooling_demand",
    # Causal lag features (shifted past data only)
    "lag_1h",
    "lag_2h",
    "lag_24h",
    "lag_168h",
    # Causal rolling means
    "rolling_mean_3h",
    "rolling_mean_24h",
    # Station type one-hot encodings
    "type_Campus",
    "type_Highway",
    "type_Industrial",
    "type_Suburban",
    "type_Urban",
]

TARGET_COLUMN: str = "energy_kwh"


def build_features(df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """Generate predictive features from cleaned telemetry with zero data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing 'timestamp', 'station_id', 'station_type',
        and core network metrics.
    drop_warmup : bool, default True
        Whether to drop the initial warm-up rows where 168h lags are NaN.

    Returns
    -------
    pd.DataFrame
        Enriched feature dataframe.
    """
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)

    # 1. Calendar & Cyclical features
    hours = data["timestamp"].dt.hour
    dow = data["timestamp"].dt.day_of_week
    months = data["timestamp"].dt.month

    data["hour"] = hours
    data["day_of_week"] = dow
    data["month"] = months

    data["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # 2. Domain Interaction features
    data["traffic_per_user"] = data["data_traffic_gb"] / (data["active_users"] + 1e-4)
    data["load_x_carriers"] = (data["prb_load_pct"] / 100.0) * data["num_active_carriers"]
    # Cooling demand activated above 21°C
    data["cooling_demand"] = np.maximum(0.0, data["temperature_c"] - 21.0) * (data["prb_load_pct"] / 100.0)

    # 3. Lags and Rolling statistics per station (strictly causal using shift)
    station_dfs = []
    for station_id, group in data.groupby("station_id", sort=False):
        group = group.copy().sort_values(by="timestamp")

        # Explicit lag shifts
        group["lag_1h"] = group[TARGET_COLUMN].shift(1)
        group["lag_2h"] = group[TARGET_COLUMN].shift(2)
        group["lag_24h"] = group[TARGET_COLUMN].shift(24)
        group["lag_168h"] = group[TARGET_COLUMN].shift(168)

        # Causal rolling statistics: MUST shift(1) so current hour is excluded
        shifted_series = group[TARGET_COLUMN].shift(1)
        group["rolling_mean_3h"] = shifted_series.rolling(window=3, min_periods=1).mean()
        group["rolling_mean_24h"] = shifted_series.rolling(window=24, min_periods=1).mean()

        station_dfs.append(group)

    featured_df = pd.concat(station_dfs, ignore_index=True)

    # 4. One-hot encode station_type
    expected_types = ["Campus", "Highway", "Industrial", "Suburban", "Urban"]
    for stype in expected_types:
        featured_df[f"type_{stype}"] = (featured_df["station_type"] == stype).astype(float)

    # 5. Handle warm-up period
    if drop_warmup:
        min_hours = featured_df.groupby("station_id")["timestamp"].count().min()
        if min_hours > 180:
            # Longest lag is 168 hours; drop initial warm-up rows
            featured_df = featured_df.dropna().reset_index(drop=True)
        else:
            # For short telemetry series, backfill early lags so data is not wiped out
            lag_cols = ["lag_1h", "lag_2h", "lag_24h", "lag_168h", "rolling_mean_3h", "rolling_mean_24h"]
            for col in lag_cols:
                featured_df[col] = featured_df.groupby("station_id")[col].bfill().ffill()
            featured_df = featured_df.dropna().reset_index(drop=True)

    return featured_df


def chronological_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into Train, Validation, and Test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Featured dataset with 'timestamp' column.
    train_ratio : float
        Fraction for training set (default 0.70).
    val_ratio : float
        Fraction for validation set (default 0.15).
    test_ratio : float
        Fraction for test set (default 0.15).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"

    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    unique_timestamps = np.sort(df_sorted["timestamp"].unique())
    n_times = len(unique_timestamps)

    train_end_idx = int(n_times * train_ratio)
    val_end_idx = int(n_times * (train_ratio + val_ratio))

    train_cutoff = unique_timestamps[train_end_idx - 1]
    val_cutoff = unique_timestamps[val_end_idx - 1]

    train_df = df_sorted[df_sorted["timestamp"] <= train_cutoff].copy().reset_index(drop=True)
    val_df = (
        df_sorted[(df_sorted["timestamp"] > train_cutoff) & (df_sorted["timestamp"] <= val_cutoff)]
        .copy()
        .reset_index(drop=True)
    )
    test_df = df_sorted[df_sorted["timestamp"] > val_cutoff].copy().reset_index(drop=True)

    return train_df, val_df, test_df
