"""Data preprocessing pipeline for 5G base station energy telemetry.

Handles timestamp sorting, deduplication, time-based missing value imputation,
per-station IQR outlier detection, and physical validity bounding.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    clip_outliers: bool = True,
    numeric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Clean and preprocess raw 5G base station telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    clip_outliers : bool, default True
        If True, clips outlier values to IQR fences. Also adds `is_outlier` flag.
    numeric_cols : Optional[List[str]]
        List of numeric columns to impute and bound.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe sorted by station_id and timestamp, with no missing values.
    """
    clean_df = df.copy()

    # 1. Parse timestamps and sort per station
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])
    clean_df = clean_df.drop_duplicates(subset=["station_id", "timestamp"])
    clean_df = clean_df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)

    if numeric_cols is None:
        numeric_cols = [
            "data_traffic_gb",
            "active_users",
            "prb_load_pct",
            "num_active_carriers",
            "temperature_c",
            "is_weekend",
            "is_holiday",
            "energy_kwh",
        ]

    # Cast numeric columns
    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 2. Per-station missing value imputation
    # Time-based interpolation for numeric series, followed by ffill/bfill for boundary points
    imputed_groups = []
    for station_id, group in clean_df.groupby("station_id", sort=False):
        group = group.copy().set_index("timestamp")
        # Categorical column imputation (mode)
        if "station_type" in group.columns and group["station_type"].isna().any():
            mode_val = group["station_type"].mode()
            fill_mode = mode_val.iloc[0] if not mode_val.empty else "Urban"
            group["station_type"] = group["station_type"].fillna(fill_mode)

        # Interpolate numeric features along the time index
        for col in numeric_cols:
            if col in group.columns:
                group[col] = group[col].interpolate(method="time")
                group[col] = group[col].ffill().bfill()

        group = group.reset_index()
        imputed_groups.append(group)

    clean_df = pd.concat(imputed_groups, ignore_index=True)

    # 3. IQR Outlier Detection and Capping per station
    clean_df["is_outlier"] = 0
    outlier_target_cols = [c for c in ["energy_kwh", "prb_load_pct", "data_traffic_gb"] if c in clean_df.columns]

    for station_id, group_indices in clean_df.groupby("station_id").groups.items():
        sub_df = clean_df.loc[group_indices]
        for col in outlier_target_cols:
            q25 = sub_df[col].quantile(0.25)
            q75 = sub_df[col].quantile(0.75)
            iqr = q75 - q25
            lower_fence = q25 - 2.5 * iqr
            upper_fence = q75 + 2.5 * iqr

            outliers = (sub_df[col] < lower_fence) | (sub_df[col] > upper_fence)
            outlier_idx = sub_df[outliers].index
            clean_df.loc[outlier_idx, "is_outlier"] = 1

            if clip_outliers:
                clean_df.loc[group_indices, col] = clean_df.loc[group_indices, col].clip(
                    lower=max(0.0, lower_fence),
                    upper=upper_fence,
                )

    # 4. Enforce strict physical boundary constraints
    if "prb_load_pct" in clean_df.columns:
        clean_df["prb_load_pct"] = clean_df["prb_load_pct"].clip(lower=0.0, upper=100.0)
    if "energy_kwh" in clean_df.columns:
        clean_df["energy_kwh"] = clean_df["energy_kwh"].clip(lower=0.1)
    if "data_traffic_gb" in clean_df.columns:
        clean_df["data_traffic_gb"] = clean_df["data_traffic_gb"].clip(lower=0.0)
    if "active_users" in clean_df.columns:
        clean_df["active_users"] = clean_df["active_users"].clip(lower=0)
    if "num_active_carriers" in clean_df.columns:
        clean_df["num_active_carriers"] = clean_df["num_active_carriers"].clip(lower=1, upper=8)
    if "is_weekend" in clean_df.columns:
        clean_df["is_weekend"] = clean_df["is_weekend"].clip(lower=0, upper=1).astype(int)
    if "is_holiday" in clean_df.columns:
        clean_df["is_holiday"] = clean_df["is_holiday"].clip(lower=0, upper=1).astype(int)

    return clean_df
