"""Test suite for feature engineering and chronological dataset splitting (Gate 2)."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_synthetic_5g_data
from src.preprocessing import preprocess_data
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    chronological_train_val_test_split,
)


@pytest.fixture(scope="module")
def sample_featured_data():
    """Fixture providing clean preprocessed and featured telemetry data."""
    raw_df = generate_synthetic_5g_data(num_hours=400, save_path=None)
    clean_df = preprocess_data(raw_df)
    featured_df = build_features(clean_df, drop_warmup=True)
    return clean_df, featured_df


def test_no_nan_values_after_warmup_drop(sample_featured_data):
    """Gate 2: Assert no NaN values exist in any feature after dropping warm-up rows."""
    _, featured_df = sample_featured_data
    assert not featured_df.empty
    nan_counts = featured_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum()
    assert nan_counts.sum() == 0, f"Found NaNs after warm-up drop:\n{nan_counts[nan_counts > 0]}"


def test_no_future_leakage_in_lags_and_rollings():
    """Gate 2: Assert that no feature uses current or future target information."""
    raw_df = generate_synthetic_5g_data(num_hours=300, save_path=None)
    clean_df = preprocess_data(raw_df)

    # Pick one station to inspect step by step
    station_data = clean_df[clean_df["station_id"] == "BS_01"].sort_values(by="timestamp").reset_index(drop=True)

    featured_station = build_features(station_data, drop_warmup=False)

    # Check for row index 200:
    idx = 200
    target_actual = station_data.loc[idx, "energy_kwh"]
    past_1 = station_data.loc[idx - 1, "energy_kwh"]
    past_2 = station_data.loc[idx - 2, "energy_kwh"]
    past_3 = station_data.loc[idx - 3, "energy_kwh"]
    past_24 = station_data.loc[idx - 24, "energy_kwh"]
    past_168 = station_data.loc[idx - 168, "energy_kwh"]

    assert np.isclose(featured_station.loc[idx, "lag_1h"], past_1)
    assert np.isclose(featured_station.loc[idx, "lag_2h"], past_2)
    assert np.isclose(featured_station.loc[idx, "lag_24h"], past_24)
    assert np.isclose(featured_station.loc[idx, "lag_168h"], past_168)

    # Rolling mean 3h must strictly average past 1, past 2, past 3 — excluding target_actual!
    expected_roll_3h = (past_1 + past_2 + past_3) / 3.0
    assert np.isclose(featured_station.loc[idx, "rolling_mean_3h"], expected_roll_3h)
    assert not np.isclose(featured_station.loc[idx, "rolling_mean_3h"], (target_actual + past_1 + past_2) / 3.0)


def test_chronological_split(sample_featured_data):
    """Gate 2: Assert chronological split preserves time ordering with zero test overlap."""
    _, featured_df = sample_featured_data
    train_df, val_df, test_df = chronological_train_val_test_split(
        featured_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # Check total rows
    assert len(train_df) + len(val_df) + len(test_df) == len(featured_df)

    # Check chronological order
    max_train_ts = train_df["timestamp"].max()
    min_val_ts = val_df["timestamp"].min()
    max_val_ts = val_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()

    assert max_train_ts < min_val_ts, f"Train max ({max_train_ts}) must be < Val min ({min_val_ts})"
    assert max_val_ts < min_test_ts, f"Val max ({max_val_ts}) must be < Test min ({min_test_ts})"


def test_feature_columns_complete(sample_featured_data):
    """Check that all defined FEATURE_COLUMNS exist in output."""
    _, featured_df = sample_featured_data
    for col in FEATURE_COLUMNS:
        assert col in featured_df.columns, f"Missing feature column: {col}"
