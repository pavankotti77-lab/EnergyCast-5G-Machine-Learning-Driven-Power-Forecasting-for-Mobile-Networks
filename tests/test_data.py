"""Test suite for 5G data generation and schema validation (Gate 1)."""

import time
from pathlib import Path
import pandas as pd
import pytest

from src.config import REQUIRED_COLUMNS, RAW_DATA_PATH
from src.data_generator import generate_synthetic_5g_data, validate_dataset_schema


def test_generation_execution_time_under_10_seconds():
    """Gate 1: Dataset generation must complete in under 10 seconds."""
    t0 = time.time()
    # Test generation with full 8760 hours
    df = generate_synthetic_5g_data(num_hours=8760, save_path=None)
    t1 = time.time()
    elapsed = t1 - t0
    assert elapsed < 10.0, f"Generation took {elapsed:.2f}s, exceeding 10s limit."
    assert len(df) == 8760 * 5


def test_dataset_shape_and_columns():
    """Verify exact required columns and row count."""
    df = generate_synthetic_5g_data(num_hours=240, save_path=None)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 240 * 5
    assert set(df["station_id"].unique()) == {"BS_01", "BS_02", "BS_03", "BS_04", "BS_05"}


def test_no_all_nan_columns():
    """Assert no column is completely empty."""
    df = generate_synthetic_5g_data(num_hours=500, save_path=None)
    for col in REQUIRED_COLUMNS:
        assert not df[col].isna().all(), f"Column {col} is all-NaN."


def test_missing_values_and_outliers_injected():
    """Verify that ~1% missing values and outliers are injected for preprocessing."""
    df = generate_synthetic_5g_data(num_hours=1000, inject_missing_pct=0.01, inject_outlier_pct=0.005, save_path=None)
    # Check that at least some missing values exist
    assert df.isna().sum().sum() > 0
    # Check that timestamps and station IDs are never missing
    assert df["timestamp"].isna().sum() == 0
    assert df["station_id"].isna().sum() == 0


def test_schema_validation():
    """Test schema validation helper for uploads and custom CSVs."""
    df = generate_synthetic_5g_data(num_hours=50, save_path=None)
    is_valid, msg = validate_dataset_schema(df)
    assert is_valid
    assert msg == ""

    # Test invalid dataframe with missing column
    df_invalid = df.drop(columns=["energy_kwh"])
    is_valid, msg = validate_dataset_schema(df_invalid)
    assert not is_valid
    assert "Missing required columns" in msg

    # Test empty dataframe
    is_valid, msg = validate_dataset_schema(pd.DataFrame())
    assert not is_valid


def test_csv_persistence(tmp_path):
    """Check that saved CSV exists and loads identically."""
    test_csv_path = tmp_path / "test_5g_dataset.csv"
    generate_synthetic_5g_data(num_hours=100, save_path=test_csv_path)
    assert test_csv_path.exists()
    loaded_df = pd.read_csv(test_csv_path)
    assert len(loaded_df) == 500
    assert list(loaded_df.columns) == REQUIRED_COLUMNS
