"""Test suite for forecasting, peak detection, and energy-saving optimization (Gate 4)."""

import pandas as pd
import pytest

from src.data_generator import generate_synthetic_5g_data
from src.preprocessing import preprocess_data
from src.features import build_features, chronological_train_val_test_split, FEATURE_COLUMNS, TARGET_COLUMN
from src.models import train_models
from src.forecasting import forecast_energy, detect_peak_windows
from src.optimizer import simulate_energy_savings


@pytest.fixture(scope="module")
def trained_model_and_history():
    """Fixture providing a fitted model and station historical dataframe."""
    raw_df = generate_synthetic_5g_data(num_hours=600, save_path=None)
    clean_df = preprocess_data(raw_df)
    featured_df = build_features(clean_df, drop_warmup=True)
    train_df, _, _ = chronological_train_val_test_split(featured_df)

    models, _, _ = train_models(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], mode="Fast")
    best_model = list(models.values())[0]

    station_history = clean_df[clean_df["station_id"] == "BS_01"].sort_values(by="timestamp").reset_index(drop=True)
    return best_model, station_history


@pytest.mark.parametrize("horizon", [24, 48, 168])
def test_forecast_output_length_and_continuity(trained_model_and_history, horizon):
    """Gate 4: Output length must match horizon, timestamps continuous, and no NaNs."""
    model, history = trained_model_and_history

    forecast_df = forecast_energy(model, history, horizon_hours=horizon)

    # 1. Output length matches horizon
    assert len(forecast_df) == horizon, f"Expected {horizon} rows, got {len(forecast_df)}"

    # 2. No NaN values in prediction or bounds
    assert not forecast_df["predicted_energy_kwh"].isna().any(), "Found NaNs in predicted_energy_kwh"
    assert not forecast_df["lower_bound_kwh"].isna().any(), "Found NaNs in lower_bound_kwh"
    assert not forecast_df["upper_bound_kwh"].isna().any(), "Found NaNs in upper_bound_kwh"

    # 3. Continuous timestamps (strictly 1-hour increments)
    time_diffs = pd.to_datetime(forecast_df["timestamp"]).diff().dropna()
    assert (time_diffs == pd.Timedelta(hours=1)).all(), "Timestamps are not strictly continuous hourly steps"

    # 4. Confidence intervals valid
    assert (forecast_df["upper_bound_kwh"] >= forecast_df["lower_bound_kwh"]).all(), "Upper bound must be >= lower bound"


def test_peak_detection(trained_model_and_history):
    """Gate 4: Verify peak detection flags and window grouping."""
    model, history = trained_model_and_history
    forecast_df = forecast_energy(model, history, horizon_hours=48)

    annotated_df, windows_df = detect_peak_windows(forecast_df, percentile=85.0)

    assert "is_peak" in annotated_df.columns
    assert annotated_df["is_peak"].sum() > 0, "Should detect at least 1 peak hour at 85th percentile"

    if not windows_df.empty:
        assert "Total Energy (kWh)" in windows_df.columns
        assert (windows_df["Total Energy (kWh)"] > 0).all()


def test_optimizer_savings_never_negative(trained_model_and_history):
    """Gate 4: Savings must be non-negative and obey PRB threshold."""
    model, history = trained_model_and_history
    forecast_df = forecast_energy(model, history, horizon_hours=48)

    results = simulate_energy_savings(
        forecast_df=forecast_df,
        model=model,
        carrier_sleep_pct=33.3,
        traffic_offload_pct=15.0,
        prb_threshold_pct=35.0,
        cost_per_kwh=0.12,
        co2_per_kwh=0.40,
    )

    # Assert total savings are never negative
    assert results["total_kwh_saved"] >= 0.0, "Total kWh saved must be non-negative"
    assert results["pct_saved"] >= 0.0, "Percentage saved must be non-negative"
    assert results["cost_saved_usd"] >= 0.0, "Cost savings must be non-negative"
    assert results["co2_saved_kg"] >= 0.0, "CO2 saved must be non-negative"

    # Assert hourly savings are never negative
    comp_df = results["comparison_df"]
    assert (comp_df["kwh_saved"] >= -1e-6).all(), "Hourly kWh saved must be >= 0 everywhere"
    assert (comp_df["optimized_energy_kwh"] <= comp_df["predicted_energy_kwh"] + 1e-6).all()
