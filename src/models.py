"""Model training pipeline for 5G base station energy prediction.

Supports Linear Regression (with StandardScaler Pipeline), Random Forest,
and XGBoost (with graceful automatic fallback to sklearn GradientBoostingRegressor).
Offers Fast and Thorough (TimeSeriesSplit RandomizedSearchCV) training modes.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, is_xgboost_available

# Try importing XGBoost
_HAS_XGB = is_xgboost_available()
if _HAS_XGB:
    try:
        import xgboost as xgb
    except Exception:
        _HAS_XGB = False


def get_model_candidates(fast_mode: bool = True) -> Dict[str, Dict[str, Any]]:
    """Return dictionary of models and candidate search grids.

    Parameters
    ----------
    fast_mode : bool, default True
        If True, provides fast tuned defaults. If False, provides hyperparameter search spaces.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Model definitions with instances and optional parameter grids.
    """
    candidates = {}

    # 1. Linear Model (Ridge with standard scaling)
    candidates["Linear Regression"] = {
        "estimator": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
            ]
        ),
        "param_dist": {
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "is_fallback": False,
    }

    # 2. Random Forest Regressor
    candidates["Random Forest"] = {
        "estimator": RandomForestRegressor(
            n_estimators=100 if fast_mode else 80,
            max_depth=16 if fast_mode else 14,
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "param_dist": {
            "n_estimators": [60, 100, 140],
            "max_depth": [10, 14, 18, None],
            "min_samples_split": [2, 4, 8],
        },
        "is_fallback": False,
    }

    # 3. XGBoost or Sklearn GradientBoosting Regressor Fallback
    if _HAS_XGB:
        candidates["XGBoost"] = {
            "estimator": xgb.XGBRegressor(
                n_estimators=120 if fast_mode else 100,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "param_dist": {
                "n_estimators": [80, 120, 160],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.08, 0.15],
            },
            "is_fallback": False,
        }
    else:
        # Graceful fallback to Scikit-Learn GradientBoostingRegressor
        candidates["XGBoost (Fallback: Gradient Boosting)"] = {
            "estimator": GradientBoostingRegressor(
                n_estimators=100 if fast_mode else 80,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                random_state=RANDOM_SEED,
            ),
            "param_dist": {
                "n_estimators": [60, 100],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.04, 0.08, 0.12],
            },
            "is_fallback": True,
        }

    return candidates


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mode: str = "Fast",
    progress_callback: Optional[Any] = None,
) -> Tuple[Dict[str, RegressorMixin], Dict[str, float], bool]:
    """Train all candidate models and record execution times.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    mode : str, default "Fast"
        "Fast" for direct training, "Thorough" for RandomizedSearchCV.
    progress_callback : Optional[Callable], default None
        Optional callback function reporting progress percentage and step string.

    Returns
    -------
    Tuple[Dict[str, RegressorMixin], Dict[str, float], bool]
        (trained_models, training_durations_seconds, used_fallback_flag)
    """
    fast_mode = (mode == "Fast")
    candidates = get_model_candidates(fast_mode=fast_mode)
    trained_models: Dict[str, RegressorMixin] = {}
    training_times: Dict[str, float] = {}
    used_fallback: bool = False

    total_candidates = len(candidates)
    for idx, (name, config) in enumerate(candidates.items()):
        if config["is_fallback"]:
            used_fallback = True

        if progress_callback:
            progress_callback(
                (idx / total_candidates),
                f"Training {name} ({mode} mode)...",
            )

        start_time = time.time()
        estimator = config["estimator"]

        if fast_mode:
            estimator.fit(X_train, y_train)
            best_est = estimator
        else:
            # Thorough mode: RandomizedSearchCV with TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            search = RandomizedSearchCV(
                estimator,
                param_distributions=config["param_dist"],
                n_iter=6,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_SEED,
                n_jobs=-1 if not config["is_fallback"] else 1,
            )
            search.fit(X_train, y_train)
            best_est = search.best_estimator_

        duration = time.time() - start_time
        trained_models[name] = best_est
        training_times[name] = round(duration, 2)

    if progress_callback:
        progress_callback(1.0, "Model training complete!")

    return trained_models, training_times, used_fallback
