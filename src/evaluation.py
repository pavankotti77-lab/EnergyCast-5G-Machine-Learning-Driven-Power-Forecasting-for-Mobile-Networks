"""Model evaluation, metrics leaderboard, and persistence module.

Calculates MAE, RMSE (cross-version safe), R², guarded MAPE,
feature importances, per-station breakdowns, and joblib model persistence.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import BEST_MODEL_PATH, METADATA_PATH


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth energy values.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'MAE', 'RMSE', 'R2', 'MAPE'.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    # Cross scikit-learn compatible RMSE
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # Guarded MAPE to prevent division by zero
    denom = np.maximum(np.abs(y_t), 1e-3)
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def evaluate_models_leaderboard(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    training_times: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, str, Dict[str, np.ndarray]]:
    """Generate comprehensive comparison leaderboard for all trained models.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of model name -> fitted model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    training_times : Optional[Dict[str, float]]
        Training duration in seconds per model.

    Returns
    -------
    Tuple[pd.DataFrame, str, Dict[str, np.ndarray]]
        (leaderboard_df, best_model_name, predictions_dict)
    """
    rows = []
    predictions_dict = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions_dict[name] = y_pred

        metrics = compute_metrics(y_test.to_numpy(), y_pred)
        train_time = training_times.get(name, 0.0) if training_times else 0.0

        rows.append(
            {
                "Model": name,
                "R² Score": metrics["R2"],
                "RMSE (kWh)": metrics["RMSE"],
                "MAE (kWh)": metrics["MAE"],
                "MAPE (%)": metrics["MAPE"],
                "Train Time (s)": train_time,
            }
        )

    leaderboard = pd.DataFrame(rows)
    # Rank by lowest RMSE first, then highest R2
    leaderboard = leaderboard.sort_values(by=["RMSE (kWh)", "R² Score"], ascending=[True, False]).reset_index(drop=True)
    best_model_name = str(leaderboard.iloc[0]["Model"])

    return leaderboard, best_model_name, predictions_dict


def compute_naive_baseline(test_df: pd.DataFrame) -> Dict[str, float]:
    """Compute performance of naive 'same hour yesterday' (24h lag) baseline.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test set containing 'energy_kwh' and 'lag_24h'.

    Returns
    -------
    Dict[str, float]
        Metrics dictionary for the naive baseline.
    """
    y_true = test_df["energy_kwh"].to_numpy()
    y_naive = test_df["lag_24h"].to_numpy()
    return compute_metrics(y_true, y_naive)


def compute_per_station_errors(
    test_df: pd.DataFrame,
    best_model: Any,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Compute test metrics segmented by individual base station.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing 'station_id' and features.
    best_model : Any
        Trained model instance.
    feature_cols : List[str]
        List of feature column names.

    Returns
    -------
    pd.DataFrame
        Breakdown table of MAE, RMSE, R² per base station.
    """
    station_rows = []
    for station_id, group in test_df.groupby("station_id"):
        stype = group["station_type"].iloc[0] if "station_type" in group.columns else "Unknown"
        y_true = group["energy_kwh"].to_numpy()
        X_sub = group[feature_cols]
        y_pred = best_model.predict(X_sub)
        metrics = compute_metrics(y_true, y_pred)
        station_rows.append(
            {
                "Station ID": station_id,
                "Type": stype,
                "Samples": len(group),
                "MAE (kWh)": metrics["MAE"],
                "RMSE (kWh)": metrics["RMSE"],
                "R² Score": metrics["R2"],
                "MAPE (%)": metrics["MAPE"],
            }
        )

    return pd.DataFrame(station_rows).sort_values(by="Station ID").reset_index(drop=True)


def extract_feature_importance(
    model: Any,
    feature_cols: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Extract or calculate feature importance from model.

    Parameters
    ----------
    model : Any
        Trained regression model or pipeline.
    feature_cols : List[str]
        Names of input features.
    X_val : Optional[pd.DataFrame]
        Validation features (used for permutation importance in linear pipelines).
    y_val : Optional[pd.Series]
        Validation target.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'feature' and 'importance', sorted descending.
    """
    # 1. Tree models with feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # 2. Linear Regression inside Pipeline
    elif hasattr(model, "named_steps") and "regressor" in model.named_steps:
        reg = model.named_steps["regressor"]
        if hasattr(reg, "coef_"):
            importances = np.abs(reg.coef_)
        else:
            importances = np.ones(len(feature_cols)) / len(feature_cols)
    elif X_val is not None and y_val is not None:
        perm = permutation_importance(model, X_val, y_val, n_repeats=3, random_state=42)
        importances = np.maximum(0, perm.importances_mean)
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    # Normalize to 100%
    tot = np.sum(importances)
    if tot > 0:
        importances = (importances / tot) * 100.0

    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    return df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)


def save_model_artifacts(
    model: Any,
    model_name: str,
    feature_cols: List[str],
    metrics: Dict[str, float],
    model_path: Path = BEST_MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    """Save trained best model and metadata to disk via joblib.

    Parameters
    ----------
    model : Any
        Fitted best model object.
    model_name : str
        Name of model algorithm.
    feature_cols : List[str]
        Ordered list of feature column names used during training.
    metrics : Dict[str, float]
        Evaluation metrics on the test set.
    model_path : Path
        Destination path for model .joblib file.
    metadata_path : Path
        Destination path for metadata .joblib file.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(metadata, metadata_path)


def load_model_artifacts(
    model_path: Path = BEST_MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Load model and metadata from disk if available.

    Parameters
    ----------
    model_path : Path
        Path to model .joblib file.
    metadata_path : Path
        Path to metadata .joblib file.

    Returns
    -------
    Tuple[Optional[Any], Optional[Dict[str, Any]]]
        (model, metadata) or (None, None) if not present.
    """
    if not model_path.exists() or not metadata_path.exists():
        return None, None

    try:
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        return model, metadata
    except Exception:
        return None, None
