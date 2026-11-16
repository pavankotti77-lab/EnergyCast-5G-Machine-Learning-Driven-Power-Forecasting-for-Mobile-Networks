"""Recursive multi-step forecasting and high-consumption peak detection for 5G base stations.

Projects future energy demand 24, 48, or 168 hours ahead with causal lag propagation,
dynamic calendar alignment, and calibrated residual confidence intervals.
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, TARGET_COLUMN


def project_future_covariates(
    history_station_df: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
    station_id: str,
    station_type: str,
) -> pd.DataFrame:
    """Project future external telemetry (traffic, PRB load, temperature) for forecasting.

    Uses seasonal profile matching based on day-of-week and hour from history,
    plus smooth calendar progression.

    Parameters
    ----------
    history_station_df : pd.DataFrame
        Historical telemetry for this station.
    future_timestamps : pd.DatetimeIndex
        Future timestamp grid.
    station_id : str
        Base station ID (e.g., 'BS_01').
    station_type : str
        Profile type (e.g., 'Urban').

    Returns
    -------
    pd.DataFrame
        DataFrame of projected future external features.
    """
    hist = history_station_df.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist["hour"] = hist["timestamp"].dt.hour
    hist["day_of_week"] = hist["timestamp"].dt.day_of_week

    # Average profile by (day_of_week, hour)
    profile = hist.groupby(["day_of_week", "hour"])[
        ["data_traffic_gb", "active_users", "prb_load_pct", "num_active_carriers", "temperature_c"]
    ].mean().reset_index()

    future_rows = []
    # Holiday calendar check
    holiday_dates = {(1, 1), (1, 16), (2, 20), (5, 29), (7, 4), (9, 4), (10, 9), (11, 23), (11, 24), (12, 25)}

    for ts in future_timestamps:
        h = ts.hour
        dow = ts.day_of_week
        month = ts.month
        is_wknd = 1 if dow >= 5 else 0
        is_hol = 1 if (month, ts.day) in holiday_dates else 0

        # Match typical profile
        match = profile[(profile["day_of_week"] == dow) & (profile["hour"] == h)]
        if not match.empty:
            traffic = float(match["data_traffic_gb"].iloc[0])
            users = int(match["active_users"].iloc[0])
            prb = float(match["prb_load_pct"].iloc[0])
            carriers = int(match["num_active_carriers"].iloc[0])
            temp = float(match["temperature_c"].iloc[0])
        else:
            traffic = 25.0
            users = 300
            prb = 45.0
            carriers = 3
            temp = 22.0

        future_rows.append(
            {
                "timestamp": ts,
                "station_id": station_id,
                "station_type": station_type,
                "data_traffic_gb": traffic,
                "active_users": users,
                "prb_load_pct": prb,
                "num_active_carriers": carriers,
                "temperature_c": temp,
                "is_weekend": is_wknd,
                "is_holiday": is_hol,
                "hour": h,
                "day_of_week": dow,
                "month": month,
                "hour_sin": np.sin(2 * np.pi * h / 24.0),
                "hour_cos": np.cos(2 * np.pi * h / 24.0),
                "dow_sin": np.sin(2 * np.pi * dow / 7.0),
                "dow_cos": np.cos(2 * np.pi * dow / 7.0),
                "traffic_per_user": traffic / (users + 1e-4),
                "load_x_carriers": (prb / 100.0) * carriers,
                "cooling_demand": max(0.0, temp - 21.0) * (prb / 100.0),
                "type_Campus": 1.0 if station_type == "Campus" else 0.0,
                "type_Highway": 1.0 if station_type == "Highway" else 0.0,
                "type_Industrial": 1.0 if station_type == "Industrial" else 0.0,
                "type_Suburban": 1.0 if station_type == "Suburban" else 0.0,
                "type_Urban": 1.0 if station_type == "Urban" else 0.0,
            }
        )

    return pd.DataFrame(future_rows)


def forecast_energy(
    model: Any,
    history_station_df: pd.DataFrame,
    horizon_hours: int = 24,
    residual_std: float = 0.16,
) -> pd.DataFrame:
    """Perform recursive multi-step forecasting for a single base station.

    Parameters
    ----------
    model : Any
        Trained regression model.
    history_station_df : pd.DataFrame
        Historical telemetry for the selected station sorted by timestamp.
    horizon_hours : int, default 24
        Forecast horizon (24, 48, or 168 hours).
    residual_std : float, default 0.16
        Standard deviation of residuals on the validation set for confidence interval bounds.

    Returns
    -------
    pd.DataFrame
        Forecast dataframe with predictions, confidence bounds, and peak flags.
    """
    hist = history_station_df.sort_values(by="timestamp").copy().reset_index(drop=True)
    station_id = str(hist["station_id"].iloc[0])
    station_type = str(hist["station_type"].iloc[0])

    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])
    future_timestamps = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")

    # Generate baseline future covariates
    future_df = project_future_covariates(hist, future_timestamps, station_id, station_type)

    # Buffer of past energy values (need at least 168 past points)
    past_energy = list(hist[TARGET_COLUMN].values[-200:])

    predictions: List[float] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    lag_1_list: List[float] = []
    lag_2_list: List[float] = []
    lag_24_list: List[float] = []
    lag_168_list: List[float] = []
    rolling_3_list: List[float] = []
    rolling_24_list: List[float] = []

    for step_idx in range(horizon_hours):
        # Build causal lag features from the updated buffer
        lag_1 = past_energy[-1]
        lag_2 = past_energy[-2]
        lag_24 = past_energy[-24]
        lag_168 = past_energy[-168] if len(past_energy) >= 168 else past_energy[-24]

        # Shifted rolling means of past values
        rolling_3 = float(np.mean(past_energy[-3:]))
        rolling_24 = float(np.mean(past_energy[-24:]))

        lag_1_list.append(lag_1)
        lag_2_list.append(lag_2)
        lag_24_list.append(lag_24)
        lag_168_list.append(lag_168)
        rolling_3_list.append(rolling_3)
        rolling_24_list.append(rolling_24)

        row_dict = future_df.iloc[step_idx].to_dict()
        row_dict["lag_1h"] = lag_1
        row_dict["lag_2h"] = lag_2
        row_dict["lag_24h"] = lag_24
        row_dict["lag_168h"] = lag_168
        row_dict["rolling_mean_3h"] = rolling_3
        row_dict["rolling_mean_24h"] = rolling_24

        # Assemble feature vector in exact order
        feat_vector = pd.DataFrame([row_dict])[FEATURE_COLUMNS]

        # Predict next hour
        pred_kwh = float(model.predict(feat_vector)[0])
        pred_kwh = max(0.1, pred_kwh)
        predictions.append(round(pred_kwh, 3))

        # Confidence interval: 95% band widening slightly over horizon
        uncertainty = 1.96 * residual_std * np.sqrt(1.0 + 0.015 * step_idx)
        lower_bounds.append(round(max(0.05, pred_kwh - uncertainty), 3))
        upper_bounds.append(round(pred_kwh + uncertainty, 3))

        # Append prediction to buffer for next recursive step
        past_energy.append(pred_kwh)

    future_df["predicted_energy_kwh"] = predictions
    future_df["lower_bound_kwh"] = lower_bounds
    future_df["upper_bound_kwh"] = upper_bounds
    future_df["lag_1h"] = lag_1_list
    future_df["lag_2h"] = lag_2_list
    future_df["lag_24h"] = lag_24_list
    future_df["lag_168h"] = lag_168_list
    future_df["rolling_mean_3h"] = rolling_3_list
    future_df["rolling_mean_24h"] = rolling_24_list

    return future_df


def detect_peak_windows(
    forecast_df: pd.DataFrame,
    percentile: float = 90.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify high-consumption hours and group contiguous hours into peak windows.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame with 'predicted_energy_kwh' and 'timestamp'.
    percentile : float, default 90.0
        Percentile threshold for peak classification (e.g. 90.0).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (annotated_forecast_df, peak_windows_summary_df)
    """
    df = forecast_df.copy()
    threshold = float(np.percentile(df["predicted_energy_kwh"], percentile))
    df["is_peak"] = (df["predicted_energy_kwh"] >= threshold).astype(int)
    df["peak_threshold_kwh"] = round(threshold, 3)

    # Group contiguous peak hours
    windows = []
    in_peak = False
    window_id = 1
    w_start = None
    w_hours = []

    for _, row in df.iterrows():
        if row["is_peak"] == 1:
            if not in_peak:
                in_peak = True
                w_start = row["timestamp"]
                w_hours = [row["predicted_energy_kwh"]]
            else:
                w_hours.append(row["predicted_energy_kwh"])
        else:
            if in_peak:
                in_peak = False
                w_end = row["timestamp"]
                windows.append(
                    {
                        "Window": f"Peak #{window_id}",
                        "Start Time": pd.to_datetime(w_start).strftime("%Y-%m-%d %H:%M"),
                        "End Time": pd.to_datetime(w_end).strftime("%Y-%m-%d %H:%M"),
                        "Duration (Hours)": len(w_hours),
                        "Avg Power (kW)": round(float(np.mean(w_hours)), 3),
                        "Peak Power (kW)": round(float(np.max(w_hours)), 3),
                        "Total Energy (kWh)": round(float(np.sum(w_hours)), 2),
                    }
                )
                window_id += 1

    # If ending on peak
    if in_peak:
        w_end = df.iloc[-1]["timestamp"]
        windows.append(
            {
                "Window": f"Peak #{window_id}",
                "Start Time": pd.to_datetime(w_start).strftime("%Y-%m-%d %H:%M"),
                "End Time": pd.to_datetime(w_end).strftime("%Y-%m-%d %H:%M"),
                "Duration (Hours)": len(w_hours),
                "Avg Power (kW)": round(float(np.mean(w_hours)), 3),
                "Peak Power (kW)": round(float(np.max(w_hours)), 3),
                "Total Energy (kWh)": round(float(np.sum(w_hours)), 2),
            }
        )

    windows_df = pd.DataFrame(windows) if windows else pd.DataFrame(
        columns=["Window", "Start Time", "End Time", "Duration (Hours)", "Avg Power (kW)", "Peak Power (kW)", "Total Energy (kWh)"]
    )

    return df, windows_df
