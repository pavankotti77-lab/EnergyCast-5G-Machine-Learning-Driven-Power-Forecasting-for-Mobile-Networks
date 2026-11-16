"""Realistic synthetic 5G Base Station Energy and Network Data Generator.

Simulates 12 months of hourly telemetry across 5 diverse base stations
following standard 3GPP network deployment and thermodynamic power profiles.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    STATION_PROFILES,
)


def generate_synthetic_5g_data(
    start_date: str = "2023-01-01 00:00:00",
    num_hours: int = 8760,
    seed: int = RANDOM_SEED,
    inject_missing_pct: float = 0.01,
    inject_outlier_pct: float = 0.005,
    save_path: Optional[Path] = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Generate physically plausible hourly 5G base-station telemetry data.

    Parameters
    ----------
    start_date : str
        Start timestamp in format 'YYYY-MM-DD HH:MM:SS'.
    num_hours : int
        Number of consecutive hours to simulate (8760 = 365 days).
    seed : int
        Random seed for reproducibility.
    inject_missing_pct : float
        Fraction of values to inject as NaNs (default 0.01 = 1%).
    inject_outlier_pct : float
        Fraction of rows to inject as anomalous spikes/glitches (default 0.005 = 0.5%).
    save_path : Optional[Path]
        If provided, saves generated dataset to this CSV path.

    Returns
    -------
    pd.DataFrame
        Synthetic 5G base station dataset matching required columns.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate hourly timestamps
    timestamps = pd.date_range(start=start_date, periods=num_hours, freq="h")
    hours = timestamps.hour.to_numpy()
    day_of_week = timestamps.day_of_week.to_numpy()  # 0=Mon, 6=Sun
    day_of_year = timestamps.day_of_year.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)

    # US Federal / Common Holidays (approx 10 days)
    holiday_dates = {
        (1, 1),    # New Year
        (1, 16),   # MLK Day
        (2, 20),   # Presidents Day
        (5, 29),   # Memorial Day
        (7, 4),    # Independence Day
        (9, 4),    # Labor Day
        (10, 9),   # Columbus Day
        (11, 23),  # Thanksgiving
        (11, 24),  # Black Friday
        (12, 25),  # Christmas
    }
    is_holiday = np.array(
        [1 if (ts.month, ts.day) in holiday_dates else 0 for ts in timestamps],
        dtype=int,
    )

    # 2. Ambient Temperature Simulation (°C)
    # Seasonal wave (warmest in July/Aug, coldest in Jan/Feb) + Diurnal cycle (warmest at 15:00)
    seasonal_temp = 15.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.0)
    diurnal_temp = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    temp_noise = rng.normal(0, 1.8, size=num_hours)
    base_temperature = seasonal_temp + diurnal_temp + temp_noise

    all_dfs = []

    # 3. Simulate per-station telemetry
    for station_id, meta in STATION_PROFILES.items():
        stype = str(meta["type"])
        base_kw = float(meta["base_power_kw"])
        max_carriers = int(meta["max_carriers"])
        traffic_mult = float(meta["traffic_mult"])

        # Base station micro-climate variation
        station_temp = np.clip(
            base_temperature + rng.normal(0, 0.8, size=num_hours),
            -15.0,
            44.0,
        )

        # Diurnal and weekly patterns tuned per station profile
        diurnal_pattern = np.zeros(num_hours)
        for h in range(24):
            mask = (hours == h)
            if stype == "Urban":
                # High day and evening traffic, nightlife dip after 3 AM
                diurnal_pattern[mask] = 0.25 + 0.65 / (1.0 + np.exp(-(h - 8) / 2.0)) - 0.2 / (1.0 + np.exp(-(h - 23) / 1.5))
            elif stype == "Suburban":
                # Twin commuter peaks: morning (7-9am) & evening (18-22pm)
                diurnal_pattern[mask] = (
                    0.20
                    + 0.45 * np.exp(-((h - 8) ** 2) / 8.0)
                    + 0.60 * np.exp(-((h - 20) ** 2) / 10.0)
                )
            elif stype == "Industrial":
                # Sharp workday peak 08:00 - 18:00, very low night
                diurnal_pattern[mask] = (
                    0.15
                    + 0.75 * (1.0 / (1.0 + np.exp(-(h - 7) / 0.8)))
                    * (1.0 - 1.0 / (1.0 + np.exp(-(h - 18) / 0.8)))
                )
            elif stype == "Campus":
                # Academic day peak + late-night study activity (20:00 - 02:00)
                diurnal_pattern[mask] = (
                    0.22
                    + 0.50 * np.exp(-((h - 14) ** 2) / 16.0)
                    + 0.45 * np.exp(-((h - 22) ** 2) / 12.0)
                )
            elif stype == "Highway":
                # Commuter flow peaks (07-09, 16-19)
                diurnal_pattern[mask] = (
                    0.18
                    + 0.55 * np.exp(-((h - 8) ** 2) / 6.0)
                    + 0.60 * np.exp(-((h - 17) ** 2) / 8.0)
                )

        # Weekend / holiday modifiers per station
        activity_factor = np.ones(num_hours)
        if stype == "Industrial":
            # Weekend & holiday traffic plummets
            activity_factor[is_weekend == 1] *= 0.35
            activity_factor[is_holiday == 1] *= 0.20
        elif stype == "Suburban":
            # Weekends have sustained daytime home traffic
            activity_factor[is_weekend == 1] *= 1.15
        elif stype == "Urban":
            # Weekend evening entertainment peaks
            activity_factor[is_weekend == 1] *= 1.20
        elif stype == "Campus":
            # Summer break dip (June-August: days 152 to 243)
            summer_mask = (day_of_year >= 152) & (day_of_year <= 243)
            activity_factor[summer_mask] *= 0.55
            activity_factor[is_weekend == 1] *= 0.80
        elif stype == "Highway":
            # Long weekend getaways boost
            activity_factor[is_weekend == 1] *= 1.25
            activity_factor[is_holiday == 1] *= 1.40

        # Inject random special events (stadium game, street festival, concert)
        num_events = rng.integers(6, 12)
        event_hours = rng.choice(num_hours - 6, size=num_events, replace=False)
        event_boost = np.zeros(num_hours)
        for ev in event_hours:
            duration = rng.integers(3, 6)
            surge = rng.uniform(0.6, 1.4)
            event_boost[ev : ev + duration] += surge

        # Combine into normalized network load demand [0.05, 0.98]
        norm_load = np.clip(
            (diurnal_pattern * activity_factor + event_boost)
            * traffic_mult
            + rng.normal(0, 0.04, size=num_hours),
            0.05,
            0.98,
        )

        # 4. Generate Core Telemetry Metrics
        # Active users (scales with load)
        base_users = 350 if stype == "Urban" else (220 if stype in ["Campus", "Suburban"] else 150)
        active_users = np.clip(
            (norm_load * base_users * rng.uniform(0.92, 1.08, size=num_hours)).round(),
            10,
            1200,
        ).astype(int)

        # Data traffic GB (Gbps aggregate over 1 hour)
        traffic_factor = 28.0 if stype == "Urban" else 18.0
        data_traffic_gb = np.clip(
            norm_load * traffic_factor * (active_users / base_users)
            + rng.normal(0, 1.0, size=num_hours),
            0.5,
            180.0,
        ).round(2)

        # PRB load percentage (0 - 100%)
        prb_load_pct = np.clip(
            norm_load * 100.0 + rng.normal(0, 2.5, size=num_hours),
            2.0,
            98.5,
        ).round(1)

        # Number of active carriers (sleep mode during very low load)
        # Dynamically configured: 1 carrier sleep when load < 25%, full carriers when load >= 25%
        num_active_carriers = np.where(
            prb_load_pct < 22.0,
            max(1, max_carriers - 1),
            max_carriers,
        )
        # Occasional 2-carrier sleep in deep night (02:00 - 05:00) if load < 15%
        deep_sleep_mask = (prb_load_pct < 15.0) & (hours >= 2) & (hours <= 5)
        num_active_carriers = np.where(
            deep_sleep_mask,
            max(1, max_carriers - 2),
            num_active_carriers,
        )

        # 5. Realistic Physical Power Equation (kWh consumed in 1 hour)
        # Power components:
        # a) Static hardware power (standby, baseband, power supplies, backhaul interface)
        # Scales slightly with active carrier count
        p_static = base_kw * (0.65 + 0.35 * (num_active_carriers / max_carriers))

        # b) Load-dependent RF transmission power: linear + carrier scaling
        p_load = (
            1.25
            * (prb_load_pct / 100.0)
            * (num_active_carriers / max_carriers)
            + 0.008 * data_traffic_gb
        )

        # c) Non-linear power amplifier distortion / back-off term at high saturation
        p_nonlinear = 0.28 * np.power(prb_load_pct / 100.0, 2.2)

        # d) HVAC Cooling Power (Equipment heat dissipation + ambient heat ingress)
        # Heat dissipation increases when load is high; active cooling triggers above 22°C
        heat_load_factor = 0.7 + 0.3 * (prb_load_pct / 100.0)
        temp_delta = np.maximum(0.0, station_temp - 21.0)
        p_cooling = 0.045 * np.power(temp_delta, 1.25) * heat_load_factor

        # e) Ambient noise and small sensor variance
        p_noise = rng.normal(0, 0.06, size=num_hours)

        # Total energy in kWh
        energy_kwh = np.clip(
            p_static + p_load + p_nonlinear + p_cooling + p_noise,
            0.5,
            12.0,
        ).round(3)

        df_station = pd.DataFrame(
            {
                "timestamp": timestamps,
                "station_id": station_id,
                "station_type": stype,
                "data_traffic_gb": data_traffic_gb,
                "active_users": active_users,
                "prb_load_pct": prb_load_pct,
                "num_active_carriers": num_active_carriers,
                "temperature_c": station_temp.round(1),
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "energy_kwh": energy_kwh,
            }
        )
        all_dfs.append(df_station)

    # Concatenate all base stations
    full_df = pd.concat(all_dfs, ignore_index=True)

    # 6. Inject Outliers (~0.5%)
    if inject_outlier_pct > 0.0:
        n_outliers = int(len(full_df) * inject_outlier_pct)
        outlier_indices = rng.choice(full_df.index, size=n_outliers, replace=False)
        for idx in outlier_indices:
            anomaly_type = rng.choice(["energy_spike", "energy_drop", "sensor_glitch"])
            if anomaly_type == "energy_spike":
                # Surge e.g. cooling failure or power spike
                full_df.loc[idx, "energy_kwh"] = round(float(full_df.loc[idx, "energy_kwh"]) * rng.uniform(2.2, 3.5), 3)
            elif anomaly_type == "energy_drop":
                # Meter error or temporary grid brownout
                full_df.loc[idx, "energy_kwh"] = round(float(full_df.loc[idx, "energy_kwh"]) * rng.uniform(0.1, 0.3), 3)
            elif anomaly_type == "sensor_glitch":
                full_df.loc[idx, "prb_load_pct"] = round(rng.uniform(115.0, 160.0), 1)

    # 7. Inject Missing Values (~1%)
    if inject_missing_pct > 0.0:
        impute_cols = [
            "data_traffic_gb",
            "active_users",
            "prb_load_pct",
            "temperature_c",
            "energy_kwh",
        ]
        total_cells = len(full_df) * len(impute_cols)
        n_missing = int(total_cells * inject_missing_pct)
        for _ in range(n_missing):
            r_idx = rng.integers(0, len(full_df))
            c_name = rng.choice(impute_cols)
            full_df.loc[r_idx, c_name] = np.nan

    # Format timestamp as string for CSV compatibility
    full_df["timestamp"] = full_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Reorder columns explicitly
    full_df = full_df[REQUIRED_COLUMNS]

    # Save to disk if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(save_path, index=False)

    return full_df


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that a dataframe conforms to the expected 5G energy schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message). If is_valid is True, error_message is empty.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"

    if len(df) == 0:
        return False, "The dataset contains 0 rows."

    # Check for all-NaN columns
    all_nan_cols = [col for col in REQUIRED_COLUMNS if df[col].isna().all()]
    if all_nan_cols:
        return False, f"Columns are completely empty (all NaN): {', '.join(all_nan_cols)}"

    return True, ""


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = generate_synthetic_5g_data()
    t1 = time.time()
    print(f"Generated {len(df)} rows across {df['station_id'].nunique()} stations in {t1 - t0:.2f} seconds.")
    print("Columns:", list(df.columns))
    print("Null count:\n", df.isna().sum())
