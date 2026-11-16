"""Configuration module for 5G Energy Predictor.

Contains paths, color palettes, physical parameters, model configuration,
and project metadata.
"""

from pathlib import Path
from typing import Dict, List

# Global Random Seed
RANDOM_SEED: int = 42

# Project Paths (Dynamic & Relative)
SRC_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SRC_DIR.parent
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RAW_DATA_PATH: Path = DATA_DIR / "5g_energy_dataset.csv"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_energy_model.joblib"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.joblib"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Project Metadata
PROJECT_TITLE: str = "Machine Learning-Based Energy Consumption Prediction in 5G Mobile Communication Networks"

AIM: str = (
    "To develop a machine learning-based system that predicts energy consumption "
    "in 5G mobile communication networks using historical network and power usage data, "
    "helping operators identify high-energy periods and improve energy efficiency."
)

PROBLEM_STATEMENT: str = (
    "5G mobile networks support high data rates, large numbers of connected devices, "
    "and data-intensive applications. These capabilities can increase the energy required "
    "by base stations and other network components. Energy demand also changes with traffic volume, "
    "active users, time, and network load. Without accurate estimation, network operators may "
    "find it difficult to plan power usage efficiently, which can increase operating costs and "
    "environmental impact. Therefore, a predictive approach is needed to estimate future energy "
    "consumption and support efficient network operation."
)

SOLUTION: str = (
    "The proposed system uses historical network data containing parameters such as timestamp, "
    "data traffic, number of active users, base-station load, and recorded energy consumption. "
    "The data is cleaned and preprocessed, and relevant features are created for machine learning. "
    "Models such as Linear Regression, Random Forest, or XGBoost can be trained to predict future "
    "energy consumption. The predicted values are compared with actual readings using evaluation "
    "measures such as MAE, RMSE, and R². A simple Python-based dashboard can display actual and "
    "predicted energy usage, highlight high-consumption periods, and help operators plan energy-saving "
    "measures such as adjusting network resources during low-traffic periods. The project is "
    "software-based and can be demonstrated using a suitable public or sample dataset; it does not "
    "require physical 5G hardware."
)

CONCLUSION: str = (
    "The proposed machine learning approach can provide early estimates of energy consumption in 5G "
    "mobile communication networks and support better energy management. By comparing different "
    "prediction models, the most suitable model can be selected based on prediction accuracy. The "
    "project demonstrates how machine learning can contribute to efficient network operation, reduced "
    "energy waste, and more sustainable wireless communication infrastructure."
)

REFERENCES: List[str] = [
    "T. S. Rappaport, Wireless Communications: Principles and Practice, 2nd ed., Prentice Hall, 2002.",
    "J. G. Andrews et al., “What Will 5G Be?”, IEEE Journal on Selected Areas in Communications, vol. 32, no. 6, pp. 1065–1082, 2014.",
    "F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "C. M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006.",
]

# Color Palette (Dark Theme / Telecom Control Room)
COLORS: Dict[str, str] = {
    "bg_dark": "#0B1020",
    "bg_card": "rgba(19, 27, 46, 0.75)",
    "bg_card_hover": "rgba(25, 36, 62, 0.85)",
    "border_subtle": "rgba(56, 189, 248, 0.15)",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "cyan": "#22D3EE",
    "violet": "#8B5CF6",
    "emerald": "#34D399",
    "amber": "#F59E0B",
    "rose": "#FB7185",
    "blue": "#38BDF8",
    "grid_line": "rgba(148, 163, 184, 0.1)",
}

# Base Station Profiles
STATION_PROFILES: Dict[str, Dict[str, object]] = {
    "BS_01": {
        "type": "Urban",
        "description": "High-density commercial and entertainment district with nightlife and heavy evening traffic.",
        "base_power_kw": 2.2,
        "max_carriers": 4,
        "traffic_mult": 1.4,
    },
    "BS_02": {
        "type": "Suburban",
        "description": "Residential zone with commuter peaks (morning & evening) and low midday usage.",
        "base_power_kw": 1.6,
        "max_carriers": 3,
        "traffic_mult": 0.9,
    },
    "BS_03": {
        "type": "Industrial",
        "description": "Smart manufacturing and logistics park with high weekday 8 AM - 6 PM usage and low weekend load.",
        "base_power_kw": 1.8,
        "max_carriers": 3,
        "traffic_mult": 1.0,
    },
    "BS_04": {
        "type": "Campus",
        "description": "University research campus with intense daytime academic use and late-night student study traffic.",
        "base_power_kw": 1.9,
        "max_carriers": 4,
        "traffic_mult": 1.2,
    },
    "BS_05": {
        "type": "Highway",
        "description": "Transport corridor with morning/evening vehicular peaks and holiday weekend travel surges.",
        "base_power_kw": 1.4,
        "max_carriers": 3,
        "traffic_mult": 0.8,
    },
}

# Required Columns for Datasets
REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "station_id",
    "station_type",
    "data_traffic_gb",
    "active_users",
    "prb_load_pct",
    "num_active_carriers",
    "temperature_c",
    "is_weekend",
    "is_holiday",
    "energy_kwh",
]

# Physical and Optimizer Constants
DEFAULT_ELECTRICITY_COST_USD: float = 0.12  # USD per kWh
DEFAULT_CO2_KG_PER_KWH: float = 0.40  # kg CO2 per kWh
DEFAULT_PRB_THRESHOLD_PCT: float = 30.0  # Safe threshold for sleep mode carrier deactivation
DEFAULT_PEAK_PERCENTILE: float = 90.0  # 90th percentile threshold for high energy detection

# Safe XGBoost availability check
def is_xgboost_available() -> bool:
    """Check if XGBoost native library is installed and importable."""
    try:
        import xgboost  # noqa: F401
        return True
    except (ImportError, Exception):
        return False
