"""End-to-end smoke test suite verifying all modules, components, and workflows."""

import importlib
import pytest
import pandas as pd
from src.config import (
    AIM,
    REFERENCES,
    REQUIRED_COLUMNS,
    STATION_PROFILES,
    is_xgboost_available,
)
from src.data_generator import generate_synthetic_5g_data, validate_dataset_schema
from src.preprocessing import preprocess_data
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    chronological_train_val_test_split,
)
from src.models import train_models, get_model_candidates
from src.evaluation import (
    compute_metrics,
    compute_naive_baseline,
    compute_per_station_errors,
    evaluate_models_leaderboard,
    extract_feature_importance,
)
from src.forecasting import forecast_energy, detect_peak_windows
from src.optimizer import simulate_energy_savings


def test_metadata_and_references():
    """Verify official project aim and academic references."""
    assert len(AIM) > 20
    assert len(REFERENCES) == 4
    assert "Wireless Communications" in REFERENCES[0]
    assert "What Will 5G Be?" in REFERENCES[1]
    assert "Scikit-learn" in REFERENCES[2]
    assert "Pattern Recognition" in REFERENCES[3]


def test_end_to_end_pipeline_workflow():
    """Verify full end-to-end workflow from generation to optimization."""
    # 1. Data generation
    df_raw = generate_synthetic_5g_data(num_hours=240, save_path=None)
    assert len(df_raw) == 240 * 5

    # 2. Preprocessing
    df_clean = preprocess_data(df_raw)
    assert df_clean.isna().sum().sum() == 0

    # 3. Features
    df_feat = build_features(df_clean, drop_warmup=True)
    train_df, val_df, test_df = chronological_train_val_test_split(df_feat)

    # 4. Model Training
    models, times, fallback = train_models(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        mode="Fast",
    )
    assert len(models) >= 2

    # 5. Leaderboard
    leaderboard, best_name, _ = evaluate_models_leaderboard(
        models,
        test_df[FEATURE_COLUMNS],
        test_df[TARGET_COLUMN],
        times,
    )
    assert not leaderboard.empty

    best_model = models[best_name]

    # 6. Forecasting
    st1_history = df_clean[df_clean["station_id"] == "BS_01"]
    fc_df = forecast_energy(best_model, st1_history, horizon_hours=24)
    assert len(fc_df) == 24
    assert not fc_df["predicted_energy_kwh"].isna().any()

    # 7. Peak detection
    annotated_fc, windows_df = detect_peak_windows(fc_df, percentile=80.0)
    assert "is_peak" in annotated_fc.columns

    # 8. Optimization simulation
    sim_res = simulate_energy_savings(
        fc_df,
        best_model,
        carrier_sleep_pct=30.0,
        traffic_offload_pct=10.0,
    )
    assert sim_res["total_kwh_saved"] >= 0.0
    assert sim_res["pct_saved"] >= 0.0
