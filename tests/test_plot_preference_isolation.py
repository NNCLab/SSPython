import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog

from core.app_settings import SettingsStore
from ui.pages.real_time_page import RealTimePage
from ui.widgets.ica_widget import ICAPlottingDialog
from ui.widgets.psd_plot import PSDPlotSettingsDialog
from ui.widgets.real_time_widget import RealTimeERP, RealTimeMainWidget


@pytest.fixture
def settings_store(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    store = SettingsStore(QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat,
    ))
    for module in ("ica_widget", "psd_plot", "real_time_widget"):
        monkeypatch.setattr(f"ui.widgets.{module}.get_settings_store", lambda: store)
    yield store
    app.processEvents()


def snapshot(store):
    return {key: store.get(key) for key in store.settings.allKeys()}


def test_ica_plot_accept_uses_current_values_without_replacing_suggestions(settings_store):
    key = "pipelines/standard/ica/plotting"
    settings_store.set(key, {"number_of_columns": 3, "decimate": 2})
    before = snapshot(settings_store)
    dialog = ICAPlottingDialog()
    dialog.settings_widget.number_of_columns_input.setValue(7)
    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.get_settings()["number_of_columns"] == 7
    assert snapshot(settings_store) == before
    reopened = ICAPlottingDialog()
    assert reopened.get_settings()["number_of_columns"] == 3
    dialog.close()
    reopened.close()


def test_psd_plot_accept_uses_current_values_without_replacing_suggestions(settings_store):
    settings_store.set("appearance/plots/psd", {"fmax": 40.0})
    before = snapshot(settings_store)
    epochs = mne.EpochsArray(
        np.zeros((2, 2, 101)), mne.create_info(["C3", "C4"], 100, "eeg"),
        tmin=-0.2, verbose=False,
    )
    dialog = PSDPlotSettingsDialog(epochs)
    dialog.settings_widget.frequency_range_input.setValue((5.0, 25.0))
    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.get_plot_params()["fmax"] == 25.0
    assert snapshot(settings_store) == before
    reopened = PSDPlotSettingsDialog(epochs)
    assert reopened.get_plot_params()["fmax"] == 40.0
    dialog.close()
    reopened.close()


def test_live_settings_accept_applies_current_values_without_saving_defaults():
    previous = {"art_rem": (-0.005, 0.005)}
    edited = {"art_rem": (-0.003, 0.009)}
    dialog = Mock()
    dialog.exec.return_value = QDialog.DialogCode.Accepted
    dialog.get_settings.return_value = edited
    visualizer = SimpleNamespace(
        _current_live_plot_settings=Mock(return_value=previous),
        apply_live_settings=Mock(),
    )

    with patch("ui.widgets.real_time_widget.RealTimeSettings", return_value=dialog):
        RealTimeERP.update_settings(visualizer)

    dialog.settings_widget.save_settings.assert_not_called()
    visualizer.apply_live_settings.assert_called_once_with(edited)


def test_live_launch_uses_current_values_without_saving_defaults():
    connection = Mock()
    connection.get_settings.return_value = {
        "info": object(), "event_channels": ["STI"], "stream_name": "run-specific",
    }
    plotting = Mock()
    plotting.get_settings.return_value = {"art_rem": (-0.003, 0.009)}
    launcher = SimpleNamespace(
        connection_widget=connection, plot_settings_widget=plotting,
        rt_erp_widget=None, _on_visualizer_closed=Mock(),
    )
    params = {**connection.get_settings.return_value, **plotting.get_settings.return_value}

    with patch("ui.widgets.real_time_widget.RealTimeERP") as visualizer:
        RealTimeMainWidget.launch_visualizer(launcher)

    connection.save_settings.assert_not_called()
    plotting.save_settings.assert_not_called()
    visualizer.assert_called_once_with(params, launcher)
    visualizer.return_value.start_visualization.assert_called_once_with()


def test_updated_preferences_refresh_live_setup_without_changing_active_run(settings_store):
    page = RealTimePage()
    active_visualizer = Mock()
    page.widget.rt_erp_widget = active_visualizer
    settings_store.set("real_time/connection", {"stream_name": "expert-suggestion"})
    settings_store.set("real_time/plotting", {"art_rem": (-0.003, 0.009)})
    before = snapshot(settings_store)

    page.on_settings_updated()

    assert page.widget.connection_widget.stream_name_input.text() == "expert-suggestion"
    assert page.widget.plot_settings_widget.art_rem_input.value() == pytest.approx((-0.003, 0.009))
    active_visualizer.apply_live_settings.assert_not_called()
    assert snapshot(settings_store) == before
    page.close()
