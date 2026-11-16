"""Streamlit AppTest comprehensive suite: exercises all pages, buttons, and interactions."""

from io import StringIO
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import REQUIRED_COLUMNS

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_initial_load():
    """Verify app loads headless without exception."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception on launch: {at.exception}"


def test_all_pages_navigation():
    """Verify navigating through all pages works cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    pages = [
        "Overview",
        "Data Explorer",
        "Model Lab",
        "Forecast Studio",
        "Energy Saver Lab",
    ]

    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, f"Page {p} raised an exception: {at.exception}"


def test_station_switching():
    """Verify switching base station cells across pages."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    for station in ["BS_01", "BS_02", "BS_03", "BS_04", "BS_05"]:
        at.sidebar.selectbox[0].set_value(station).run()
        assert not at.exception


def test_forecast_studio_horizons():
    """Verify Forecast Studio runs at 24, 48, and 168 hour horizons."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("Forecast Studio").run()

    for h in [24, 48, 168]:
        at.select_slider(key="fc_horizon_slider").set_value(h).run()
        assert not at.exception


def test_energy_saver_simulation_sliders():
    """Verify Energy Saver Lab reacts smoothly to slider adjustments."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("Energy Saver Lab").run()

    at.slider(key="opt_carrier_sleep_slider").set_value(25.0).run()
    assert not at.exception

    at.slider(key="opt_offload_slider").set_value(20.0).run()
    assert not at.exception


def test_model_lab_retraining():
    """Verify triggering model training from the UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("Model Lab").run()

    # Fast mode train button
    at.button(key="btn_train_models").click().run()
    assert not at.exception
