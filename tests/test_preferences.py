import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QPushButton

from core.app_settings import SettingsStore
from ui.pages.preferences_page import PreferencesPage
from ui.widgets.preprocessing_widgets import ArtifactRemovalDialog
from ui.widgets.psd_plot import PSDPlotSettingsDialog


class _ImmediateWorker:
    def __init__(self, operation, **_kwargs):
        self.operation = operation
        self._error = None

    def exec_with_dialog(self, *_args, **_kwargs):
        return self.operation()


class TestPreferenceSuggestions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SettingsStore(QSettings(
            str(Path(directory.name) / "settings.ini"), QSettings.Format.IniFormat,
        ))
        for module in (
            "ui.widgets.preprocessing_widgets",
            "ui.widgets.tools.global_preferences",
            "ui.widgets.run_ica_widget",
            "ui.widgets.ica_widget",
            "ui.widgets.auto_label",
            "ui.widgets.real_time_widget",
            "ui.widgets.psd_plot",
            "utils",
        ):
            patcher = patch(f"{module}.get_settings_store", return_value=self.store)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target in (
            "ui.pages.preferences_page.apply_theme",
            "ui.pages.preferences_page.QMessageBox.information",
        ):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.page = PreferencesPage(self.store)
        self.addCleanup(self.page.close)

    def snapshot(self):
        return {key: self.store.get(key) for key in self.store.settings.allKeys()}

    def artifact_dialog(self):
        processor = SimpleNamespace(
            has=lambda _stage: False,
            paths={},
            run_artifact_removal=Mock(),
        )
        dialog = ArtifactRemovalDialog(processor)
        self.addCleanup(dialog.close)
        return dialog, processor

    def test_saved_pulse_suggestions_survive_dialog_edits_apply_and_reopen(self):
        suggestions = (-0.004, 0.012)
        self.page.artifact_widget.window_edit.setValue(suggestions)
        self.page.save_all_settings()
        saved = self.snapshot()
        self.assertEqual(
            self.store.get(self.page.artifact_widget.get_settings_path())["window"],
            suggestions,
        )

        dialog, processor = self.artifact_dialog()
        self.assertEqual(dialog.settings_widget.window_edit.value(), suggestions)
        run_window = (-0.007, 0.020)
        dialog.settings_widget.window_edit.setValue(run_window)
        with patch(
            "ui.widgets.preprocessing_widgets.confirm_processing_update", return_value=True,
        ), patch("ui.widgets.preprocessing_widgets.Worker", _ImmediateWorker):
            dialog.run_process()

        self.assertEqual(processor.run_artifact_removal.call_args.kwargs["window"], run_window)
        self.assertEqual(self.snapshot(), saved)
        reopened, _ = self.artifact_dialog()
        self.assertEqual(reopened.settings_widget.window_edit.value(), suggestions)
        self.assertNotIn("Save as defaults", [
            button.text() for button in reopened.findChildren(QPushButton)
        ])

    def test_cancel_preserves_saved_pulse_preferences(self):
        suggestions = (-0.003, 0.009)
        self.page.artifact_widget.window_edit.setValue(suggestions)
        self.page.save_all_settings()
        saved = self.snapshot()
        dialog, processor = self.artifact_dialog()
        self.assertFalse(hasattr(dialog, "reset_button"))
        dialog.settings_widget.window_edit.setValue((-0.005, 0.015))
        with patch(
            "ui.widgets.preprocessing_widgets.confirm_processing_update", return_value=False,
        ), patch("ui.widgets.preprocessing_widgets.Worker") as worker:
            dialog.run_process()
        worker.assert_not_called()
        processor.run_artifact_removal.assert_not_called()
        dialog.reject()
        self.assertEqual(self.snapshot(), saved)

    def test_invalid_later_form_preserves_all_saved_preferences(self):
        saved = self.snapshot()
        self.page.artifact_widget.window_edit.setValue((-0.004, 0.012))
        self.page.global_settings_widget.output_dir_input.setText("new-derivatives")
        self.page.global_settings_widget.plot_settings_widget.cmap_input.setCurrentText("viridis")
        self.page.continuous_filter_widget.resample_edit.setText("0")
        emitted = Mock()
        self.page.settings_saved.connect(emitted)
        with patch("ui.pages.preferences_page.QMessageBox.critical") as error:
            self.page.save_all_settings()
        error.assert_called_once()
        emitted.assert_not_called()
        self.assertEqual(self.snapshot(), saved)

    def test_invalid_jobs_does_not_save_plot_preferences(self):
        saved = self.snapshot()
        general = self.page.global_settings_widget
        general.plot_settings_widget.cmap_input.setCurrentText("viridis")
        general.n_jobs_input.setText("invalid")
        with self.assertRaises(ValueError):
            general.save_settings()
        self.assertEqual(self.snapshot(), saved)

    def test_psd_suggestions_are_saved_without_recording_bounds(self):
        self.page.psd_plot_widget.frequency_range_input.setValue((60.0, 120.0))
        self.page.psd_plot_widget.time_range_input.setValue((-0.3, 0.6))
        self.page.save_all_settings()
        self.page.load_all_settings()

        params = self.page.psd_plot_widget.get_settings()
        self.assertEqual((params["fmin"], params["fmax"]), (60.0, 120.0))
        self.assertEqual((params["tmin"], params["tmax"]), (-0.3, 0.6))
        self.assertEqual(self.store.get("appearance/plots/psd"), params)

    def test_saved_psd_suggestions_survive_plot_dialog_edits_and_reopen(self):
        self.page.psd_plot_widget.frequency_range_input.setValue((60.0, 120.0))
        self.page.psd_plot_widget.time_range_input.setValue((-0.3, 0.6))
        self.page.save_all_settings()
        saved = self.snapshot()
        epochs = mne.EpochsArray(
            np.zeros((1, 2, 601)), mne.create_info(["C3", "C4"], 500, "eeg"),
            tmin=-0.5, verbose=False,
        )
        dialog = PSDPlotSettingsDialog(epochs)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.get_plot_params()["fmax"], 120.0)
        self.assertEqual(dialog.get_plot_params()["tmin"], -0.3)
        self.assertEqual(dialog.get_plot_params()["tmax"], 0.6)
        dialog.settings_widget.frequency_range_input.setValue((5.0, 30.0))
        dialog.accept()

        self.assertEqual(dialog.get_plot_params()["fmax"], 30.0)
        self.assertEqual(self.snapshot(), saved)
        reopened = PSDPlotSettingsDialog(epochs)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_plot_params()["fmax"], 120.0)

    def test_invalid_psd_range_prevents_all_preference_writes(self):
        saved = self.snapshot()
        self.page.global_settings_widget.output_dir_input.setText("new-derivatives")
        self.page.psd_plot_widget.frequency_range_input.setValue((120.0, 60.0))
        with patch("ui.pages.preferences_page.QMessageBox.critical") as error:
            self.page.save_all_settings()
        error.assert_called_once()
        self.assertEqual(self.snapshot(), saved)

    def test_ica_label_suggestions_can_be_saved_from_preferences(self):
        self.page.ica_labeling_widget.label_checkboxes["other"].setChecked(False)
        self.page.save_all_settings()
        self.page.load_all_settings()

        self.assertFalse(self.store.get("ica_label/exclude/other"))
        self.assertFalse(self.page.ica_labeling_widget.get_settings()["other"])


if __name__ == "__main__":
    unittest.main()
