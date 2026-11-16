"""5G Base Station Energy Consumption Prediction & Optimization Dashboard.

Theme: Machine Learning x Network Communication.
Author: Sonika Reddy (@sonikareddy24).
"""

from datetime import datetime, timedelta
from pathlib import Path
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local module imports
from src.config import (
    BEST_MODEL_PATH,
    COLORS,
    DEFAULT_CO2_KG_PER_KWH,
    DEFAULT_ELECTRICITY_COST_USD,
    DEFAULT_PEAK_PERCENTILE,
    DEFAULT_PRB_THRESHOLD_PCT,
    METADATA_PATH,
    PROJECT_TITLE,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    STATION_PROFILES,
    is_xgboost_available,
)
from src.data_generator import generate_synthetic_5g_data, validate_dataset_schema
from src.evaluation import (
    compute_metrics,
    compute_naive_baseline,
    compute_per_station_errors,
    evaluate_models_leaderboard,
    extract_feature_importance,
    load_model_artifacts,
    save_model_artifacts,
)
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    chronological_train_val_test_split,
)
from src.forecasting import detect_peak_windows, forecast_energy
from src.models import train_models
from src.optimizer import simulate_energy_savings
from src.preprocessing import preprocess_data
from src.ui import (
    apply_custom_css,
    format_plotly_chart,
    render_hero,
    render_kpi,
)

# Set Streamlit Page Config
st.set_page_config(
    page_title="5G Network Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark theme and custom CSS
apply_custom_css()


# --- Cached Data Loading and Processing ---
@st.cache_data(show_spinner="Loading 5G telemetry data...")
def get_raw_data() -> pd.DataFrame:
    """Load or generate the raw 5G base station dataset."""
    if not RAW_DATA_PATH.exists():
        return generate_synthetic_5g_data(num_hours=8760, save_path=RAW_DATA_PATH)
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data(show_spinner="Running causal preprocessing & feature engineering...")
def get_featured_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean, engineer features, and chronologically split telemetry."""
    clean_df = preprocess_data(raw_df)
    featured_df = build_features(clean_df, drop_warmup=True)
    train_df, val_df, test_df = chronological_train_val_test_split(featured_df)
    return clean_df, train_df, val_df, test_df


# --- State Initialization ---
def init_session_state() -> None:
    """Initialize session state parameters, trained models, and artifacts."""
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = get_raw_data()

    if "clean_df" not in st.session_state:
        clean_df, train_df, val_df, test_df = get_featured_data(st.session_state.raw_df)
        st.session_state.clean_df = clean_df
        st.session_state.train_df = train_df
        st.session_state.val_df = val_df
        st.session_state.test_df = test_df

    if "best_model" not in st.session_state or st.session_state.best_model is None:
        model, metadata = load_model_artifacts(BEST_MODEL_PATH, METADATA_PATH)
        if model is not None and metadata is not None:
            st.session_state.best_model = model
            st.session_state.best_model_name = metadata.get("model_name", "Trained Regressor")
            st.session_state.metadata = metadata
            st.session_state.leaderboard = None
            st.session_state.all_models = {metadata.get("model_name", "Trained Regressor"): model}
        else:
            # Auto-train fast baseline on first run
            with st.spinner("Initializing and training baseline ML models..."):
                train_df = st.session_state.train_df
                test_df = st.session_state.test_df
                X_train = train_df[FEATURE_COLUMNS]
                y_train = train_df[TARGET_COLUMN]
                X_test = test_df[FEATURE_COLUMNS]
                y_test = test_df[TARGET_COLUMN]

                models, times, fallback_used = train_models(X_train, y_train, mode="Fast")
                leaderboard, best_name, _ = evaluate_models_leaderboard(models, X_test, y_test, times)
                best_model = models[best_name]
                best_metrics = leaderboard.iloc[0].to_dict()

                save_model_artifacts(best_model, best_name, FEATURE_COLUMNS, best_metrics)
                st.session_state.best_model = best_model
                st.session_state.best_model_name = best_name
                st.session_state.all_models = models
                st.session_state.leaderboard = leaderboard
                st.session_state.metadata = {
                    "model_name": best_name,
                    "metrics": best_metrics,
                    "feature_cols": FEATURE_COLUMNS,
                    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

    if "custom_dataset" not in st.session_state:
        st.session_state.custom_dataset = None


init_session_state()

# --- Sidebar Navigation & Global Controls ---
with st.sidebar:
    st.markdown(
        textwrap.dedent("""
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem;">
            <div style="background: linear-gradient(135deg, #22D3EE, #8B5CF6); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px;">⚡</div>
            <div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC;">5G Telemetry ML</div>
                <div style="font-size: 0.72rem; color: #22D3EE; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Green Telecom Ops</div>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Explorer",
            "Model Lab",
            "Forecast Studio",
            "Energy Saver Lab",
        ],
        index=0,
        key="nav_sidebar_radio",
    )

    st.markdown("---")
    st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Telemetry Scope</p>", unsafe_allow_html=True)
    selected_station = st.selectbox(
        "Base Station Cell",
        options=["BS_01", "BS_02", "BS_03", "BS_04", "BS_05"],
        format_func=lambda x: f"{x} - {STATION_PROFILES[x]['type']} Cell",
        index=0,
        key="global_station_selector",
    )

    station_info = STATION_PROFILES[selected_station]
    st.caption(f"**Type:** {station_info['type']} | **Base:** {station_info['base_power_kw']} kW | **Carriers:** {station_info['max_carriers']}")

    st.markdown("---")
    if not is_xgboost_available():
        st.markdown(
            textwrap.dedent("""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 0.6rem; font-size: 0.75rem; color: #FBBF24;">
                <b>XGBoost notice:</b> Native OpenMP is absent on this host. Fallback to <b>sklearn GradientBoostingRegressor</b> active.
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )


# ==============================================================================
# 1. PAGE: OVERVIEW
# ==============================================================================
if page == "Overview":
    render_hero(
        title="5G Network Energy Consumption Intelligence",
        subtitle="Predict base station load, spot power peaks, and autonomously plan carrier sleep modes.",
        badges=[
            {"label": "Live Forecast Ready", "class": "badge-cyan"},
            {"label": f"Active: {st.session_state.best_model_name}", "class": "badge-violet"},
            {"label": "3GPP Sub-6 GHz NR", "class": "badge-emerald"},
            {"label": f"Target Cell: {selected_station}", "class": "badge-amber"},
        ],
    )

    # Top KPI Metrics Row
    meta_metrics = st.session_state.metadata.get("metrics", {})
    r2_val = meta_metrics.get("R² Score", meta_metrics.get("R2", 0.9557))
    mae_val = meta_metrics.get("MAE (kWh)", meta_metrics.get("MAE", 0.0666))
    model_name_short = st.session_state.best_model_name.replace(" (Fallback: Gradient Boosting)", "")

    # Next 24h projected kWh
    station_hist = st.session_state.clean_df[st.session_state.clean_df["station_id"] == selected_station]
    sample_fc = forecast_energy(st.session_state.best_model, station_hist, horizon_hours=24)
    next_24h_kwh = sample_fc["predicted_energy_kwh"].sum()

    # Potential saving estimation
    opt_res = simulate_energy_savings(sample_fc, st.session_state.best_model)
    saving_pct = opt_res["pct_saved"]

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    render_kpi(kpi1, "Top Model", model_name_short, "Auto-selected best fit", COLORS["violet"])
    render_kpi(kpi2, "Model R² Score", f"{r2_val:.3f}", "Test set fit accuracy", COLORS["cyan"])
    render_kpi(kpi3, "Test MAE", f"{mae_val:.3f} kWh", "Mean Absolute Error", COLORS["blue"])
    render_kpi(kpi4, "Next 24h Demand", f"{next_24h_kwh:.1f} kWh", f"Station {selected_station}", COLORS["amber"])
    render_kpi(kpi5, "Potential Saving", f"{saving_pct:.1f}%", f"{opt_res['total_kwh_saved']:.1f} kWh / day", COLORS["emerald"])

    # Heatmap of 7-day energy (Hour x Day of Week)
    st.markdown("### Weekly Energy Consumption Footprint")
    df_cell = station_hist.copy()
    df_cell["timestamp"] = pd.to_datetime(df_cell["timestamp"])
    df_cell["hour"] = df_cell["timestamp"].dt.hour
    df_cell["day_name"] = df_cell["timestamp"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_pivot = (
        df_cell.groupby(["day_name", "hour"])["energy_kwh"]
        .mean()
        .unstack(level="hour")
        .reindex(day_order)
    )

    fig_heat = px.imshow(
        heat_pivot,
        labels=dict(x="Hour of Day (00:00 - 23:00)", y="Day of Week", color="Avg Energy (kWh)"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_order,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    format_plotly_chart(fig_heat, title=f"Average Hourly Power Demand Pattern - {selected_station} ({station_info['type']})", height=340)
    st.plotly_chart(fig_heat, width="stretch")

    # Base Station Grid Profiles
    st.markdown("### Monitored Base Station Fleet")
    fleet_rows = []
    for sid, info in STATION_PROFILES.items():
        sub = st.session_state.clean_df[st.session_state.clean_df["station_id"] == sid]
        fleet_rows.append(
            {
                "Station ID": sid,
                "Profile Type": info["type"],
                "Description": info["description"],
                "Base Power (kW)": info["base_power_kw"],
                "Max Carriers": info["max_carriers"],
                "Mean Power (kW)": round(sub["energy_kwh"].mean(), 2),
                "Peak Power (kW)": round(sub["energy_kwh"].max(), 2),
            }
        )
    st.dataframe(pd.DataFrame(fleet_rows), width="stretch", hide_index=True)


# ==============================================================================
# 2. PAGE: DATA EXPLORER
# ==============================================================================
elif page == "Data Explorer":
    render_hero(
        title="5G Network Telemetry Data Explorer",
        subtitle="Inspect raw network telemetry, review missingness, examine IQR outliers, and test custom CSV uploads.",
        badges=[{"label": "12 Months Continuous", "class": "badge-cyan"}, {"label": "5 Base Stations", "class": "badge-violet"}],
    )

    tabs = st.tabs(["Dataset Preview", "Time Series Analysis", "Correlation & Clusters", "Upload Custom Telemetry"])

    with tabs[0]:
        st.markdown("#### Telemetry Records Preview")
        view_station = st.selectbox(
            "Filter Preview by Station",
            ["All Stations"] + list(STATION_PROFILES.keys()),
            key="de_preview_station",
        )
        disp_df = st.session_state.clean_df
        if view_station != "All Stations":
            disp_df = disp_df[disp_df["station_id"] == view_station]

        st.dataframe(disp_df.head(100), width="stretch", height=320)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Statistical Distribution")
            num_cols = ["energy_kwh", "prb_load_pct", "data_traffic_gb", "active_users", "temperature_c"]
            st.dataframe(disp_df[num_cols].describe().round(2), width="stretch")

        with c2:
            st.markdown("#### Telemetry Quality & Preprocessing Health")
            # Raw missing check
            raw_missing = st.session_state.raw_df.isna().sum()
            clean_missing = st.session_state.clean_df.isna().sum()
            outlier_count = int(st.session_state.clean_df["is_outlier"].sum())

            quality_df = pd.DataFrame(
                {
                    "Raw Missing": raw_missing[num_cols],
                    "Clean Missing": clean_missing[num_cols],
                    "Cleaned Status": ["100% Imputed" for _ in num_cols],
                }
            )
            st.dataframe(quality_df, width="stretch")
            st.info(f"**IQR Outlier Detection:** {outlier_count} anomalous readings ({outlier_count / len(st.session_state.clean_df) * 100:.2f}%) were identified and bounded to prevent skewing training weights.")

    with tabs[1]:
        st.markdown("#### Interactive Multi-Metric Telemetry Series")
        chart_station = st.selectbox("Select Station for Time Series", list(STATION_PROFILES.keys()), key="de_ts_station")
        ts_df = st.session_state.clean_df[st.session_state.clean_df["station_id"] == chart_station].copy()
        ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])

        # Window slider
        st.caption("Displaying 14-day sample window for smooth interaction:")
        sample_window = ts_df.tail(24 * 14)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=sample_window["timestamp"], y=sample_window["energy_kwh"], name="Energy (kWh)", line=dict(color="#22D3EE", width=2)))
        fig_ts.add_trace(go.Scatter(x=sample_window["timestamp"], y=sample_window["prb_load_pct"] / 20.0, name="PRB Load (% / 20)", line=dict(color="#8B5CF6", width=1.5, dash="dot")))
        fig_ts.add_trace(go.Scatter(x=sample_window["timestamp"], y=sample_window["data_traffic_gb"] / 20.0, name="Traffic (GB / 20)", line=dict(color="#34D399", width=1.5, dash="dash")))

        format_plotly_chart(fig_ts, title=f"Energy vs PRB Load vs Traffic (Last 14 Days) - {chart_station}", height=400)
        st.plotly_chart(fig_ts, width="stretch")

    with tabs[2]:
        c_heat, c_scatter = st.columns(2)
        with c_heat:
            st.markdown("#### Feature Correlation Heatmap")
            corr_cols = ["energy_kwh", "prb_load_pct", "data_traffic_gb", "active_users", "num_active_carriers", "temperature_c"]
            corr_mat = st.session_state.clean_df[corr_cols].corr().round(3)
            fig_corr = px.imshow(
                corr_mat,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(color="Pearson r"),
                aspect="auto",
            )
            format_plotly_chart(fig_corr, title="Telemetry Correlation Matrix", height=380)
            st.plotly_chart(fig_corr, width="stretch")

        with c_scatter:
            st.markdown("#### Traffic vs Energy by Base Station")
            sample_scatter = st.session_state.clean_df.sample(n=min(2500, len(st.session_state.clean_df)), random_state=42)
            fig_scat = px.scatter(
                sample_scatter,
                x="data_traffic_gb",
                y="energy_kwh",
                color="station_type",
                color_discrete_sequence=["#22D3EE", "#8B5CF6", "#34D399", "#F59E0B", "#FB7185"],
                labels={"data_traffic_gb": "Hourly Traffic (GB)", "energy_kwh": "Energy (kWh)", "station_type": "Cell Type"},
                opacity=0.7,
            )
            format_plotly_chart(fig_scat, title="Traffic vs Power Demand by Station Type", height=380)
            st.plotly_chart(fig_scat, width="stretch")

    with tabs[3]:
        st.markdown("#### Upload Custom 5G Base Station CSV")
        st.write("Upload an external operator CSV file. The app will validate required telemetry columns, allow column mapping, and check schema integrity.")

        uploaded_file = st.file_uploader("Upload CSV Telemetry File", type=["csv"], key="custom_csv_uploader")
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully: {uploaded_file.name} ({len(user_df)} rows, {len(user_df.columns)} columns)")

                st.markdown("##### Column Mapping Assistant")
                st.caption("Confirm mapping of your dataset's columns to standard 5G telemetry fields:")

                mapping_cols = st.columns(3)
                mapped_dict = {}
                user_cols = list(user_df.columns)

                for idx, req_col in enumerate(REQUIRED_COLUMNS):
                    col_target = mapping_cols[idx % 3]
                    default_idx = user_cols.index(req_col) if req_col in user_cols else 0
                    mapped_dict[req_col] = col_target.selectbox(
                        f"Field: `{req_col}`",
                        options=user_cols,
                        index=default_idx,
                        key=f"upload_map_{req_col}",
                    )

                if st.button("Apply Column Mapping & Validate", key="btn_apply_mapping"):
                    renamed_df = user_df[[mapped_dict[c] for c in REQUIRED_COLUMNS]].copy()
                    renamed_df.columns = REQUIRED_COLUMNS
                    is_valid, msg = validate_dataset_schema(renamed_df)
                    if is_valid:
                        st.session_state.custom_dataset = renamed_df
                        st.success("Validation PASSED! The custom dataset matches all required constraints.")
                        st.dataframe(renamed_df.head(), width="stretch")
                    else:
                        st.error(f"Validation FAILED: {msg}")
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")


# ==============================================================================
# 3. PAGE: MODEL LAB
# ==============================================================================
elif page == "Model Lab":
    render_hero(
        title="Predictive Model Lab & Leaderboard",
        subtitle="Compare Linear Regression, Random Forest, and XGBoost with TimeSeriesSplit evaluation.",
        badges=[
            {"label": "Ridge Regression", "class": "badge-cyan"},
            {"label": "Random Forest", "class": "badge-violet"},
            {"label": "XGBoost / GradientBoost", "class": "badge-emerald"},
        ],
    )

    c_train_ctrl, c_train_info = st.columns([1, 2])
    with c_train_ctrl:
        st.markdown("#### Training Controls")
        train_mode = st.radio(
            "Hyperparameter Strategy",
            ["Fast", "Thorough"],
            index=0,
            key="model_lab_mode",
            help="Fast uses pre-tuned hyperparams (<15s). Thorough uses TimeSeriesSplit RandomizedSearchCV (<60s).",
        )
        if st.button("🚀 Train Models Now", key="btn_train_models"):
            train_progress_bar = st.progress(0, text="Initializing training...")

            def progress_hook(pct: float, step_str: str) -> None:
                train_progress_bar.progress(int(pct * 100), text=step_str)

            train_df = st.session_state.train_df
            test_df = st.session_state.test_df

            X_train = train_df[FEATURE_COLUMNS]
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df[FEATURE_COLUMNS]
            y_test = test_df[TARGET_COLUMN]

            models, times, fallback_used = train_models(X_train, y_train, mode=train_mode, progress_callback=progress_hook)
            leaderboard, best_name, _ = evaluate_models_leaderboard(models, X_test, y_test, times)

            best_model = models[best_name]
            best_metrics = leaderboard.iloc[0].to_dict()
            save_model_artifacts(best_model, best_name, FEATURE_COLUMNS, best_metrics)

            st.session_state.all_models = models
            st.session_state.leaderboard = leaderboard
            st.session_state.best_model = best_model
            st.session_state.best_model_name = best_name
            st.session_state.metadata = {
                "model_name": best_name,
                "metrics": best_metrics,
                "feature_cols": FEATURE_COLUMNS,
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.success(f"Training completed successfully! Winner: **{best_name}**")

    with c_train_info:
        st.markdown("#### Current Model Leaderboard")
        if st.session_state.leaderboard is not None:
            st.dataframe(st.session_state.leaderboard, width="stretch", hide_index=True)
        else:
            st.dataframe(
                pd.DataFrame([st.session_state.metadata.get("metrics", {})]),
                width="stretch",
                hide_index=True,
            )

    # Benchmark vs Naive Baseline
    naive_res = compute_naive_baseline(st.session_state.test_df)
    st.markdown("---")
    st.markdown("#### Benchmark: ML Models vs Naive Baseline (Same Hour Yesterday)")
    b1, b2, b3, b4 = st.columns(4)
    meta_m = st.session_state.metadata.get("metrics", {})
    best_rmse = meta_m.get("RMSE (kWh)", meta_m.get("RMSE", 0.165))
    best_r2 = meta_m.get("R² Score", meta_m.get("R2", 0.956))

    render_kpi(b1, "Naive Baseline RMSE", f"{naive_res['RMSE']:.3f} kWh", "Lag 24h heuristic", COLORS["text_secondary"])
    render_kpi(b2, "Best ML Model RMSE", f"{best_rmse:.3f} kWh", f"Winner: {st.session_state.best_model_name[:12]}...", COLORS["cyan"])
    render_kpi(b3, "Error Reduction", f"{(naive_res['RMSE'] - best_rmse) / naive_res['RMSE'] * 100:.1f}%", "Superior accuracy over baseline", COLORS["emerald"])
    render_kpi(b4, "Best Model R²", f"{best_r2:.3f}", "Variance explained", COLORS["violet"])

    # Actual vs Predicted & Residuals
    st.markdown("---")
    c_act, c_res = st.columns(2)
    test_df = st.session_state.test_df
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    # Model selector for charts
    available_model_names = list(st.session_state.all_models.keys())
    active_m_name = st.selectbox("Inspect Model Predictions", available_model_names, index=0, key="inspect_model_select")
    active_m = st.session_state.all_models[active_m_name]
    y_pred = active_m.predict(X_test)
    residuals = y_test - y_pred

    with c_act:
        st.markdown("#### Actual vs Predicted Energy (Test Set Sample)")
        sample_plot_idx = np.random.choice(len(y_test), size=min(120, len(y_test)), replace=False)
        sample_sorted = np.sort(sample_plot_idx)
        fig_act = go.Figure()
        fig_act.add_trace(go.Scatter(y=y_test[sample_sorted], mode="lines+markers", name="Actual Telemetry", line=dict(color="#94A3B8", width=1.5)))
        fig_act.add_trace(go.Scatter(y=y_pred[sample_sorted], mode="lines", name=f"{active_m_name} Prediction", line=dict(color="#22D3EE", width=2.5)))
        format_plotly_chart(fig_act, title=f"Actual vs Predicted - {active_m_name}", height=360)
        st.plotly_chart(fig_act, width="stretch")

    with c_res:
        st.markdown("#### Residual Distribution & Calibration")
        fig_res = px.histogram(
            residuals,
            nbins=40,
            labels={"value": "Residual Error (kWh)"},
            color_discrete_sequence=["#8B5CF6"],
            opacity=0.85,
        )
        format_plotly_chart(fig_res, title=f"Residual Error Histogram (Mean: {np.mean(residuals):.3f}, Std: {np.std(residuals):.3f})", height=360)
        st.plotly_chart(fig_res, width="stretch")

    # Feature Importance & Per Station Breakdown
    st.markdown("---")
    c_imp, c_sta = st.columns(2)
    with c_imp:
        st.markdown("#### Top Predictive Feature Importances")
        df_imp = extract_feature_importance(active_m, FEATURE_COLUMNS, st.session_state.val_df[FEATURE_COLUMNS], st.session_state.val_df[TARGET_COLUMN])
        top_imp = df_imp.head(10)
        fig_imp = px.bar(
            top_imp,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Relative Importance (%)", "feature": "Telemetry Feature"},
            color="importance",
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        format_plotly_chart(fig_imp, title="Top 10 Feature Importances", height=360)
        st.plotly_chart(fig_imp, width="stretch")

    with c_sta:
        st.markdown("#### Per-Base-Station Test Accuracy Breakdown")
        station_err_df = compute_per_station_errors(test_df, active_m, FEATURE_COLUMNS)
        st.dataframe(station_err_df, width="stretch", height=340, hide_index=True)


# ==============================================================================
# 4. PAGE: FORECAST STUDIO
# ==============================================================================
elif page == "Forecast Studio":
    render_hero(
        title="Recursive Forecast Studio & Peak Window Alerting",
        subtitle="Generate multi-step causal forecasts up to 168 hours ahead with dynamic 95% confidence bounds.",
        badges=[
            {"label": "Recursive Lag Step", "class": "badge-cyan"},
            {"label": "Peak Window Detection", "class": "badge-rose"},
            {"label": f"Cell: {selected_station}", "class": "badge-violet"},
        ],
    )

    c_fc_ctrl1, c_fc_ctrl2, c_fc_ctrl3 = st.columns(3)
    with c_fc_ctrl1:
        fc_horizon = st.select_slider(
            "Forecast Horizon",
            options=[24, 48, 168],
            value=48,
            format_func=lambda h: f"{h} Hours ({h//24} Days)" if h >= 24 else f"{h} Hours",
            key="fc_horizon_slider",
        )
    with c_fc_ctrl2:
        peak_percentile = st.slider(
            "High-Consumption Percentile Threshold",
            min_value=75.0,
            max_value=98.0,
            value=DEFAULT_PEAK_PERCENTILE,
            step=1.0,
            format="%d%%",
            key="fc_percentile_slider",
        )
    with c_fc_ctrl3:
        st.markdown("<p style='font-size: 0.82rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Confidence Interval</p>", unsafe_allow_html=True)
        st.caption("Standard 95% interval derived from validation residual distribution (±1.96σ).")

    # Run forecast
    station_history = st.session_state.clean_df[st.session_state.clean_df["station_id"] == selected_station]
    forecast_df = forecast_energy(st.session_state.best_model, station_history, horizon_hours=fc_horizon)
    annotated_fc, peak_windows = detect_peak_windows(forecast_df, percentile=peak_percentile)

    # Summary Metrics Row
    fc_k1, fc_k2, fc_k3, fc_k4 = st.columns(4)
    total_proj_kwh = annotated_fc["predicted_energy_kwh"].sum()
    peak_hours_count = annotated_fc["is_peak"].sum()
    peak_thresh_val = annotated_fc["peak_threshold_kwh"].iloc[0]

    render_kpi(fc_k1, "Projected Demand", f"{total_proj_kwh:.1f} kWh", f"Next {fc_horizon}h aggregate", COLORS["cyan"])
    render_kpi(fc_k2, "Peak Threshold", f"{peak_thresh_val:.2f} kW", f"{peak_percentile:.0f}th percentile cut", COLORS["rose"])
    render_kpi(fc_k3, "Peak Hours Flagged", f"{peak_hours_count} / {fc_horizon} hrs", f"{peak_hours_count/fc_horizon*100:.1f}% of horizon", COLORS["amber"])
    render_kpi(fc_k4, "Distinct Peak Windows", f"{len(peak_windows)}", "Contiguous intervals", COLORS["violet"])

    # Forecast Chart with Confidence Band & Shaded Peak Windows
    st.markdown("#### Projected Energy Demand & High-Consumption Windows")
    fig_fc = go.Figure()

    # Confidence interval band
    fig_fc.add_trace(
        go.Scatter(
            x=list(annotated_fc["timestamp"]) + list(annotated_fc["timestamp"])[::-1],
            y=list(annotated_fc["upper_bound_kwh"]) + list(annotated_fc["lower_bound_kwh"])[::-1],
            fill="toself",
            fillcolor="rgba(34, 211, 238, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval",
        )
    )

    # Main forecast line
    fig_fc.add_trace(
        go.Scatter(
            x=annotated_fc["timestamp"],
            y=annotated_fc["predicted_energy_kwh"],
            mode="lines+markers",
            line=dict(color="#22D3EE", width=2.5),
            marker=dict(size=4),
            name="Predicted Energy (kWh)",
        )
    )

    # Highlight peak points
    peak_subset = annotated_fc[annotated_fc["is_peak"] == 1]
    if not peak_subset.empty:
        fig_fc.add_trace(
            go.Scatter(
                x=peak_subset["timestamp"],
                y=peak_subset["predicted_energy_kwh"],
                mode="markers",
                marker=dict(color="#FB7185", size=8, symbol="diamond"),
                name="High-Energy Peak",
            )
        )

    # Peak threshold horizontal line
    fig_fc.add_hline(
        y=peak_thresh_val,
        line_dash="dash",
        line_color="#FB7185",
        annotation_text=f"Peak Limit ({peak_thresh_val:.2f} kW)",
        annotation_position="bottom right",
    )

    format_plotly_chart(fig_fc, title=f"Next {fc_horizon}h Energy Forecast with Peak Detection - {selected_station}", height=420)
    st.plotly_chart(fig_fc, width="stretch")

    # Peak Windows Table & Download
    c_tab, c_dl = st.columns([3, 1])
    with c_tab:
        st.markdown("#### Detected High-Consumption Peak Windows")
        if not peak_windows.empty:
            st.dataframe(peak_windows, width="stretch", hide_index=True)
        else:
            st.info("No contiguous peak windows flagged at this threshold.")

    with c_dl:
        st.markdown("#### Export Schedule")
        csv_data = annotated_fc.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_data,
            file_name=f"5G_energy_forecast_{selected_station}_{fc_horizon}h.csv",
            mime="text/csv",
            key="btn_download_forecast_csv",
        )


# ==============================================================================
# 5. PAGE: ENERGY SAVER LAB
# ==============================================================================
elif page == "Energy Saver Lab":
    render_hero(
        title="Autonomous Energy Saver & What-If Simulation Engine",
        subtitle="Simulate carrier sleep modes and dynamic traffic offloading to evaluate operational savings and carbon reductions.",
        badges=[
            {"label": "Carrier Sleep Mode", "class": "badge-emerald"},
            {"label": "Green Telecom Ops", "class": "badge-cyan"},
            {"label": "CO₂ Mitigation", "class": "badge-violet"},
        ],
    )

    c_sim_ctrl, c_sim_res = st.columns([1, 2])
    with c_sim_ctrl:
        st.markdown("#### Optimization What-If Controls")
        carrier_sleep = st.slider(
            "Carrier Sleep Capacity (%)",
            min_value=0.0,
            max_value=50.0,
            value=33.3,
            step=5.0,
            format="%.1f%%",
            key="opt_carrier_sleep_slider",
            help="Percentage reduction in active radio frequency carriers during eligible off-peak hours.",
        )
        traffic_offload = st.slider(
            "Traffic Offload Percentage (%)",
            min_value=0.0,
            max_value=30.0,
            value=15.0,
            step=5.0,
            format="%d%%",
            key="opt_offload_slider",
            help="Traffic redirected to energy-efficient micro-cells or Cloud-RAN.",
        )
        prb_threshold = st.slider(
            "Max PRB Load for Sleep Mode (%)",
            min_value=10.0,
            max_value=50.0,
            value=DEFAULT_PRB_THRESHOLD_PCT,
            step=5.0,
            format="%d%%",
            key="opt_prb_thresh_slider",
            help="Strict safeguard: Carrier sleep is only allowed when load is below this threshold.",
        )
        elec_rate = st.number_input(
            "Electricity Tariff ($/kWh)",
            min_value=0.01,
            max_value=1.00,
            value=DEFAULT_ELECTRICITY_COST_USD,
            step=0.01,
            key="opt_elec_rate_input",
        )
        co2_rate = st.number_input(
            "Grid Emission Factor (kg CO₂/kWh)",
            min_value=0.05,
            max_value=1.50,
            value=DEFAULT_CO2_KG_PER_KWH,
            step=0.05,
            key="opt_co2_factor_input",
        )

    # Run optimizer simulation on 48h forecast
    station_history = st.session_state.clean_df[st.session_state.clean_df["station_id"] == selected_station]
    fc_base = forecast_energy(st.session_state.best_model, station_history, horizon_hours=48)
    sim_results = simulate_energy_savings(
        forecast_df=fc_base,
        model=st.session_state.best_model,
        carrier_sleep_pct=carrier_sleep,
        traffic_offload_pct=traffic_offload,
        prb_threshold_pct=prb_threshold,
        cost_per_kwh=elec_rate,
        co2_per_kwh=co2_rate,
    )

    with c_sim_res:
        st.markdown("#### Projected Savings Overview (Next 48 Hours)")
        s1, s2, s3, s4 = st.columns(4)
        render_kpi(s1, "Energy Saved", f"{sim_results['total_kwh_saved']:.1f} kWh", f"{sim_results['pct_saved']:.1f}% reduction", COLORS["emerald"])
        render_kpi(s2, "Cost Reduction", f"${sim_results['cost_saved_usd']:.2f}", f"At ${elec_rate:.2f} / kWh", COLORS["cyan"])
        render_kpi(s3, "CO₂ Avoided", f"{sim_results['co2_saved_kg']:.1f} kg", f"{co2_rate:.2f} kg / kWh factor", COLORS["violet"])
        render_kpi(s4, "Sleep Windows", f"{sim_results['eligible_hours']} / 48 hrs", f"PRB ≤ {prb_threshold:.0f}%", COLORS["amber"])

        # Plain language recommendation card
        st.markdown(
            textwrap.dedent(f"""
            <div class="glass-card" style="border-left: 4px solid #34D399; margin-top: 0.5rem;">
                <div style="font-weight: 700; color: #34D399; font-size: 0.95rem; margin-bottom: 0.3rem;">OPERATIONAL RECOMMENDATION</div>
                <div style="font-size: 0.92rem; color: #F8FAFC; line-height: 1.5;">{sim_results['recommendation']}</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    # Before vs After Energy Curve
    st.markdown("---")
    st.markdown("#### Hourly Demand Trajectory: Baseline vs Optimized Sleep-Mode")
    comp_df = sim_results["comparison_df"]

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=comp_df["timestamp"], y=comp_df["predicted_energy_kwh"], name="Baseline Demand (kWh)", line=dict(color="#FB7185", width=2, dash="dash")))
    fig_sim.add_trace(go.Scatter(x=comp_df["timestamp"], y=comp_df["optimized_energy_kwh"], name="Optimized Demand (kWh)", line=dict(color="#34D399", width=2.5)))

    # Shaded savings region
    fig_sim.add_trace(
        go.Scatter(
            x=list(comp_df["timestamp"]) + list(comp_df["timestamp"])[::-1],
            y=list(comp_df["predicted_energy_kwh"]) + list(comp_df["optimized_energy_kwh"])[::-1],
            fill="toself",
            fillcolor="rgba(52, 211, 153, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Conserved Energy",
        )
    )

    format_plotly_chart(fig_sim, title=f"Baseline vs Energy-Optimized Profile - {selected_station}", height=380)
    st.plotly_chart(fig_sim, width="stretch")

    # Detailed Audit Table
    st.markdown("#### Hourly Sleep Mode Audit Trail")
    st.dataframe(comp_df, width="stretch", height=280, hide_index=True)

