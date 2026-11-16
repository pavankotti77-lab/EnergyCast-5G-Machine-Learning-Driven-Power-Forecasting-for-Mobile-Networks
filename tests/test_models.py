"""Test suite for model training, baseline benchmarking, and evaluation (Gate 3)."""

import numpy as np
import pytest

from src.data_generator import generate_synthetic_5g_data
from src.preprocessing import preprocess_data
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    chronological_train_val_test_split,
)
from src.models import train_models
from src.evaluation import (
    compute_metrics,
    compute_naive_baseline,
    evaluate_models_leaderboard,
    load_model_artifacts,
    save_model_artifacts,
)


@pytest.fixture(scope="module")
def dataset_and_split():
    """Generate, preprocess, and feature split dataset for modeling tests."""
    raw_df = generate_synthetic_5g_data(num_hours=2000, save_path=None)
    clean_df = preprocess_data(raw_df)
    featured_df = build_features(clean_df, drop_warmup=True)
    train_df, val_df, test_df = chronological_train_val_test_split(featured_df)
    return train_df, val_df, test_df


def test_models_train_and_beat_naive_baseline(dataset_and_split):
    """Gate 3: Verify trained model beats naive 'same hour yesterday' baseline."""
    train_df, val_df, test_df = dataset_and_split

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Train in fast mode
    models, times, fallback_used = train_models(X_train, y_train, mode="Fast")
    assert len(models) >= 2, "Must train at least 2 candidate models"

    # Evaluate leaderboard
    leaderboard, best_model_name, _ = evaluate_models_leaderboard(models, X_test, y_test, times)
    best_model = models[best_model_name]

    # Naive baseline
    naive_metrics = compute_naive_baseline(test_df)
    best_pred = best_model.predict(X_test)
    best_metrics = compute_metrics(y_test.to_numpy(), best_pred)

    print(f"\nBest Model: {best_model_name}")
    print(f"Model RMSE: {best_metrics['RMSE']:.4f} vs Naive RMSE: {naive_metrics['RMSE']:.4f}")
    print(f"Model R²:   {best_metrics['R2']:.4f} vs Naive R²:   {naive_metrics['R2']:.4f}")

    # Assert model significantly beats baseline
    assert best_metrics["RMSE"] < naive_metrics["RMSE"], (
        f"Model RMSE ({best_metrics['RMSE']}) must beat Naive Baseline RMSE ({naive_metrics['RMSE']})"
    )


def test_r2_score_above_threshold(dataset_and_split):
    """Gate 3: Assert R² on test set is above 0.85."""
    train_df, val_df, test_df = dataset_and_split

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    models, _, _ = train_models(X_train, y_train, mode="Fast")
    best_r2 = max(
        compute_metrics(y_test.to_numpy(), m.predict(X_test))["R2"]
        for m in models.values()
    )

    assert best_r2 >= 0.85, f"Expected R² >= 0.85 on test set, got {best_r2:.4f}"


def test_persistence_roundtrip(tmp_path, dataset_and_split):
    """Test saving and loading artifacts with joblib."""
    train_df, _, test_df = dataset_and_split
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    models, _, _ = train_models(X_train, y_train, mode="Fast")
    model_name = list(models.keys())[0]
    model = models[model_name]

    model_p = tmp_path / "model.joblib"
    meta_p = tmp_path / "meta.joblib"

    save_model_artifacts(
        model,
        model_name,
        FEATURE_COLUMNS,
        {"R2": 0.92, "RMSE": 0.25},
        model_p,
        meta_p,
    )

    loaded_m, loaded_meta = load_model_artifacts(model_p, meta_p)
    assert loaded_m is not None
    assert loaded_meta["model_name"] == model_name
    assert loaded_meta["feature_cols"] == FEATURE_COLUMNS
