"""Energy-saving what-if simulation and network resource optimization engine.

Simulates carrier sleep modes and dynamic traffic offloading during low-load intervals,
evaluates savings with trained machine learning models, and computes financial and CO2 impacts.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import DEFAULT_CO2_KG_PER_KWH, DEFAULT_ELECTRICITY_COST_USD, DEFAULT_PRB_THRESHOLD_PCT
from src.features import FEATURE_COLUMNS


def simulate_energy_savings(
    forecast_df: pd.DataFrame,
    model: Any,
    carrier_sleep_pct: float = 33.3,
    traffic_offload_pct: float = 15.0,
    prb_threshold_pct: float = DEFAULT_PRB_THRESHOLD_PCT,
    cost_per_kwh: float = DEFAULT_ELECTRICITY_COST_USD,
    co2_per_kwh: float = DEFAULT_CO2_KG_PER_KWH,
) -> Dict[str, Any]:
    """Simulate what-if energy efficiency scenarios on forecasted network demand.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecasted dataframe output from `forecast_energy`.
    model : Any
        Trained regression model.
    carrier_sleep_pct : float, default 33.3
        Percentage reduction in active carriers during off-peak eligible hours.
    traffic_offload_pct : float, default 15.0
        Percentage of traffic offloaded to small cells / C-RAN during eligible hours.
    prb_threshold_pct : float, default 30.0
        Maximum PRB load percentage eligible for power-saving sleep mode.
    cost_per_kwh : float, default 0.12
        Electricity cost in USD per kWh.
    co2_per_kwh : float, default 0.40
        Carbon emissions factor in kg CO2 per kWh.

    Returns
    -------
    Dict[str, Any]
        Dictionary with simulation results, hourly comparison table, and recommendations.
    """
    sim_df = forecast_df.copy()

    # Mask for eligible off-peak hours
    eligible_mask = sim_df["prb_load_pct"] <= prb_threshold_pct
    eligible_hours_count = int(eligible_mask.sum())

    optimized_kwh: List[float] = []

    # Prepare features for re-prediction
    for idx, row in sim_df.iterrows():
        base_kwh = float(row["predicted_energy_kwh"])

        if eligible_mask.loc[idx]:
            # Apply carrier reduction (at least 1 carrier must remain online)
            orig_carriers = float(row["num_active_carriers"])
            reduced_carriers = max(1.0, np.round(orig_carriers * (1.0 - carrier_sleep_pct / 100.0)))

            # Apply traffic offload
            orig_traffic = float(row["data_traffic_gb"])
            reduced_traffic = max(0.2, orig_traffic * (1.0 - traffic_offload_pct / 100.0))

            orig_prb = float(row["prb_load_pct"])
            reduced_prb = max(1.0, orig_prb * (1.0 - traffic_offload_pct / 100.0))

            # Reconstruct row feature dictionary
            opt_row = row.to_dict()
            opt_row["num_active_carriers"] = reduced_carriers
            opt_row["data_traffic_gb"] = reduced_traffic
            opt_row["prb_load_pct"] = reduced_prb
            opt_row["load_x_carriers"] = (reduced_prb / 100.0) * reduced_carriers
            opt_row["traffic_per_user"] = reduced_traffic / (float(row["active_users"]) + 1e-4)
            opt_row["cooling_demand"] = max(0.0, float(row["temperature_c"]) - 21.0) * (reduced_prb / 100.0)

            # Ensure all feature columns exist
            for c in FEATURE_COLUMNS:
                if c not in opt_row:
                    opt_row[c] = 0.0

            # Re-predict with ML model
            feat_vec = pd.DataFrame([opt_row])[FEATURE_COLUMNS]
            pred_opt = float(model.predict(feat_vec)[0])

            # Safety constraint: savings are never negative (optimized energy <= baseline energy)
            pred_opt = min(base_kwh, max(0.1, pred_opt))
            optimized_kwh.append(round(pred_opt, 3))
        else:
            # High-load hours operate at baseline
            optimized_kwh.append(base_kwh)

    sim_df["optimized_energy_kwh"] = optimized_kwh
    sim_df["kwh_saved"] = (sim_df["predicted_energy_kwh"] - sim_df["optimized_energy_kwh"]).round(3)
    sim_df["is_sleep_active"] = eligible_mask.astype(int)

    # Aggregate key performance indicators
    total_baseline_kwh = float(sim_df["predicted_energy_kwh"].sum())
    total_optimized_kwh = float(sim_df["optimized_energy_kwh"].sum())
    total_kwh_saved = max(0.0, total_baseline_kwh - total_optimized_kwh)
    pct_saved = (total_kwh_saved / total_baseline_kwh * 100.0) if total_baseline_kwh > 0 else 0.0
    cost_saved = total_kwh_saved * cost_per_kwh
    co2_saved = total_kwh_saved * co2_per_kwh

    # Plain language operator recommendation
    rec_text = (
        f"Sleep mode eligible for {eligible_hours_count} of {len(sim_df)} hours (PRB load ≤ {prb_threshold_pct:.0f}%). "
        f"Putting carrier capacity to sleep saves approximately {total_kwh_saved:.2f} kWh ({pct_saved:.1f}% reduction), "
        f"yielding ${cost_saved:.2f} USD operational savings and avoiding {co2_saved:.2f} kg CO₂ emissions."
    )

    return {
        "baseline_kwh": round(total_baseline_kwh, 2),
        "optimized_kwh": round(total_optimized_kwh, 2),
        "total_kwh_saved": round(total_kwh_saved, 2),
        "pct_saved": round(pct_saved, 1),
        "cost_saved_usd": round(cost_saved, 2),
        "co2_saved_kg": round(co2_saved, 2),
        "eligible_hours": eligible_hours_count,
        "total_hours": len(sim_df),
        "recommendation": rec_text,
        "comparison_df": sim_df[
            [
                "timestamp",
                "station_id",
                "predicted_energy_kwh",
                "optimized_energy_kwh",
                "kwh_saved",
                "prb_load_pct",
                "num_active_carriers",
                "is_sleep_active",
            ]
        ],
    }
