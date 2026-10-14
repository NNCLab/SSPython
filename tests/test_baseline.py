import json
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
from PySide6.QtWidgets import QApplication, QDialog

from core.app_settings import SettingsStore
from core.baseline import event_baseline_default, validate_baseline, voltage_baseline_default
from core.processing import Preprocessor
from ui.pages.erp_analysis_page import ErpAnalysisPage, TopoplotTimeWindowDialog
from ui.widgets.evoked_plot import EvokedPlotWidget
from ui.widgets.preprocessing_widgets import PreprocessingSettingsWidget
from ui.widgets.time_frequency_widget import (
    POWER_DB_MODE, RAW_POWER_MODE, ZSCORE_MODE, TimeFrequencyWidget,
    apply_time_frequency_transform,
)


def make_epochs(mode="event", *, tmin=-0.2, final_baseline="unspecified"):
    info = mne.create_info(["Cz", "Pz"], 100, "eeg")
    data = np.broadcast_to(np.linspace(2e-6, 5e-6, 41), (2, 2, 41)).copy()
    history = [] if mode is None else [{"epoching": {"mode": mode}}]
    if final_baseline != "unspecified":
        history.append({"filter_epochs": {"baseline": final_baseline}})
    info["description"] = json.dumps(history)
    return mne.EpochsArray(data, info, tmin=tmin, baseline=None, verbose=False)


class TestBaselineSemantics(unittest.TestCase):
    def test_none_and_entire_epoch_are_distinct(self):
        epochs = make_epochs()
        self.assertIsNone(validate_baseline(epochs.times, None))
        self.assertEqual(validate_baseline(epochs.times, (None, None)), (-0.2, 0.2))

    def test_defaults_require_event_metadata_and_a_pre_event_interval(self):
        self.assertEqual(event_baseline_default(make_epochs()), (-0.2, 0.0))
        for epochs in (make_epochs("fixed"), make_epochs(None), make_epochs(tmin=0.0)):
            with self.subTest(description=epochs.info["description"], tmin=epochs.tmin):
                self.assertIsNone(event_baseline_default(epochs))
        self.assertIsNone(voltage_baseline_default(make_epochs(final_baseline=None)))
        corrected = make_epochs().apply_baseline((-0.2, 0), verbose=False)
        self.assertIsNone(voltage_baseline_default(corrected))

    def test_invalid_intervals_never_fall_back(self):
        times = make_epochs().times
        for baseline in ((0.1, -0.1), (0, 0), (-1, -0.3), (-0.2, 1),
                         (float("nan"), 0), (0.001, 0.002)):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                validate_baseline(times, baseline)

    def test_power_without_normalization_preserves_absolute_differences(self):
        times = np.array([-0.1, 0.0, 0.1])
        power = np.array([[1, 2, 4], [10, 20, 40]]) * 1e-12
        original = power.copy()
        linear, units = apply_time_frequency_transform(power, times)
        np.testing.assert_allclose(linear, power * 1e12)
        self.assertEqual(units, "Power (uV^2)")
        logarithmic, units = apply_time_frequency_transform(
            power, times, mode=POWER_DB_MODE, apply_baseline=False,
            baseline_s=(10, 20),
        )
        np.testing.assert_allclose(logarithmic, 10 * np.log10(power * 1e12))
        np.testing.assert_allclose(logarithmic[1] - logarithmic[0], 10)
        self.assertEqual(units, "Power (dB re 1 uV^2)")
        np.testing.assert_array_equal(power, original)
        with self.assertRaisesRegex(ValueError, "explicit reference"):
            apply_time_frequency_transform(power, times, mode=ZSCORE_MODE)

    def test_power_normalization_uses_only_the_requested_reference(self):
        times = np.array([-0.1, 0.0, 0.1])
        power = np.array([[1, 3, 6]]) * 1e-12
        result, units = apply_time_frequency_transform(
            power, times, mode=POWER_DB_MODE, apply_baseline=True,
            baseline_s=(-0.1, 0),
        )
        np.testing.assert_allclose(result, 10 * np.log10([[0.5, 1.5, 3]]))
        self.assertEqual(units, "Power change (dB)")
        for mode in (RAW_POWER_MODE, POWER_DB_MODE, ZSCORE_MODE):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                apply_time_frequency_transform(
                    power, times, mode=mode, apply_baseline=True, baseline_s=(1, 2),
                )


class TestBaselineProcessing(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def preprocessor(self, epochs):
        preprocessor = Preprocessor.__new__(Preprocessor)
        preprocessor._epochs = epochs
        preprocessor.has = lambda stage: stage == "epochs"
        preprocessor._clear_downstream_files = Mock()
        preprocessor.paths = {
            "preprocessed": Path(self.directory.name) / "final-epo.fif",
            "epochs_ica": Path(self.directory.name) / "epochs-ica.fif",
        }
        return preprocessor

    def filter(self, epochs, **params):
        preprocessor = self.preprocessor(epochs)
        preprocessor.filter_epochs(
            bandpass=(None, None), interpolate_bad_channels=False, verbose=False,
            **params,
        )
        saved = mne.read_epochs(preprocessor.paths["preprocessed"], preload=True, verbose=False)
        return saved, json.loads(saved.info["description"])[-1]["filter_epochs"]

    def test_disabled_correction_preserves_offset_and_logs_none(self):
        for mode in ("event", "fixed", None):
            with self.subTest(mode=mode):
                epochs = make_epochs(mode)
                saved, log = self.filter(epochs)
                np.testing.assert_allclose(saved.get_data(), epochs.get_data(), rtol=1e-6)
                self.assertIsNone(saved.baseline)
                self.assertIsNone(log["baseline"])
                self.assertIsNone(log["baseline_requested"])

    def test_explicit_correction_works_for_event_and_resting_data(self):
        for mode, baseline in (("event", (-0.2, 0)), ("fixed", (None, None))):
            with self.subTest(mode=mode):
                epochs = make_epochs(mode)
                original = epochs.get_data().copy()
                saved, log = self.filter(epochs, baseline=baseline)
                expected = epochs.copy().apply_baseline(baseline, verbose=False)
                np.testing.assert_allclose(saved.get_data(), expected.get_data(), atol=1e-12)
                np.testing.assert_array_equal(epochs.get_data(), original)
                self.assertEqual(log["baseline_requested"], list(baseline))
                np.testing.assert_allclose(log["baseline"], saved.baseline)

    def test_invalid_baseline_does_not_write_or_invalidate_output(self):
        preprocessor = self.preprocessor(make_epochs())
        with self.assertRaises(ValueError):
            preprocessor.filter_epochs(baseline=(0, 2), verbose=False)
        self.assertFalse(preprocessor.paths["preprocessed"].exists())
        preprocessor._clear_downstream_files.assert_not_called()

    def test_ica_fitting_keeps_unbaselined_data_with_applied_projection(self):
        epochs = make_epochs()
        epochs.set_eeg_reference("average", projection=True, verbose=False).apply_proj(verbose=False)
        # Different channel offsets survive referencing, so unintended baselining is visible.
        epochs._data[:, 0] += 2e-6
        epochs._data[:, 1] -= 2e-6
        original = epochs.get_data().copy()
        preprocessor = self.preprocessor(epochs)
        with patch("core.processing.mne.preprocessing.ICA") as ica_class:
            ica_class.return_value.info = {}
            preprocessor.run_epochs_ica(verbose=False)
        fitted = ica_class.return_value.fit.call_args.args[0]
        self.assertIsNone(fitted.baseline)
        np.testing.assert_array_equal(fitted.get_data(), original)
        np.testing.assert_array_equal(epochs.get_data(), original)


class TestBaselineUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SettingsStore(QSettings(
            str(Path(directory.name) / "settings.ini"), QSettings.Format.IniFormat,
        ))
        for module in ("ui.widgets.evoked_plot", "ui.widgets.preprocessing_widgets"):
            patcher = patch(f"{module}.get_settings_store", return_value=self.store)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.app.processEvents)

    def close_later(self, widget):
        self.addCleanup(widget.close)
        return widget

    def preprocessing_widget(self, mode, **kwargs):
        preprocessor = SimpleNamespace(epochs=make_epochs(mode, **kwargs), epoching_mode=mode or "unknown")
        return self.close_later(PreprocessingSettingsWidget(preprocessor))

    def test_preprocessing_modes_keep_independent_baseline_preferences(self):
        event = self.preprocessing_widget("event")
        self.assertEqual(event.get_params()["baseline"], (-0.2, 0))
        event.save_settings()
        resting = self.preprocessing_widget("fixed", tmin=0)
        self.assertIsNone(resting.get_params()["baseline"])
        self.assertFalse(resting.baseline_input.interval.isEnabled())
        resting.baseline_input.checkbox.setChecked(True)
        self.assertEqual(resting.get_params()["baseline"], (0, 0.4))
        resting.save_settings()
        self.assertEqual(self.preprocessing_widget("fixed", tmin=0).get_params()["baseline"], (0, 0.4))
        event.baseline_input.checkbox.setChecked(False)
        event.save_settings()
        self.assertIsNone(self.preprocessing_widget("event").get_params()["baseline"])

    def test_legacy_unchecked_endpoints_migrate_to_none(self):
        self.store.set("pipelines/standard/preprocessing/apply", {"baseline": (None, None)})
        self.assertIsNone(self.preprocessing_widget("event").get_params()["baseline"])

    def test_saved_pre_event_baseline_is_disabled_when_new_epochs_have_no_pre_event_data(self):
        self.preprocessing_widget("event").save_settings()
        widget = self.preprocessing_widget("event", tmin=0)
        self.assertIsNone(widget.get_params()["baseline"])
        self.assertFalse(widget.baseline_input.checkbox.isChecked())

    def test_erp_none_preserves_offsets_and_manual_changes_are_reversible(self):
        epochs = make_epochs(final_baseline=None)
        source = epochs.get_data().copy()
        widget = self.close_later(EvokedPlotWidget(params={"baseline": None}))
        widget.update_plot(epochs)
        np.testing.assert_allclose(widget.display_evoked.data, epochs.average().data)
        self.assertIn("none", widget.baseline_status_label.text())
        widget.baseline_control.interval.setValue((-0.2, 0))
        widget.baseline_control.checkbox.setChecked(True)
        expected = epochs.average().apply_baseline((-0.2, 0), verbose=False)
        np.testing.assert_allclose(widget.display_evoked.data, expected.data, atol=1e-12)
        widget.baseline_control.checkbox.setChecked(False)
        np.testing.assert_allclose(widget.display_evoked.data, epochs.average().data)
        np.testing.assert_array_equal(epochs.get_data(), source)

    def test_erp_preserves_preprocessing_choice_and_identifies_applied_baseline(self):
        widget = self.close_later(EvokedPlotWidget())
        epochs = make_epochs(final_baseline=None)
        widget.update_plot(epochs)
        self.assertFalse(widget.baseline_control.checkbox.isChecked())
        corrected = make_epochs().apply_baseline((-0.2, 0), verbose=False)
        widget.update_plot(corrected)
        self.assertFalse(widget.baseline_control.checkbox.isChecked())
        self.assertIn("already applied", widget.baseline_status_label.text())
        np.testing.assert_allclose(widget.display_evoked.data, corrected.average().data)

    def test_topoplot_respects_none_and_selected_erp_display_baseline(self):
        page = self.close_later(ErpAnalysisPage())
        epochs = make_epochs("fixed", tmin=0)
        page.tep = SimpleNamespace(epochs=epochs, label="Rest")
        page.evoked_plot_widget.update_plot(epochs)
        dialog = self.close_later(TopoplotTimeWindowDialog(epochs))
        dialog.start_input.setValue(0.1)
        dialog.end_input.setValue(0.3)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                page.evoked_plot_widget.baseline_control.setValue((0, 0.1) if enabled else None)
                with (
                    patch("ui.pages.erp_analysis_page.TopoplotTimeWindowDialog", return_value=dialog),
                    patch.object(dialog, "exec", return_value=QDialog.DialogCode.Accepted),
                    patch.object(mne.Evoked, "plot_topo", autospec=True) as plot,
                ):
                    plot.return_value.axes = []
                    page.topoplot()
                expected = epochs.average()
                if enabled:
                    expected.apply_baseline((0, 0.1), verbose=False)
                expected.crop(0.1, 0.3)
                np.testing.assert_allclose(plot.call_args.args[0].data, expected.data, atol=1e-12)

    def test_tfr_defaults_and_zscore_toggle_are_explicit(self):
        for mode, tmin, enabled in (("event", -0.2, True), ("fixed", 0, False), (None, -0.2, False), ("event", 0, False)):
            with self.subTest(mode=mode, tmin=tmin):
                epochs = make_epochs(mode, tmin=tmin)
                values = np.broadcast_to(np.linspace(1e-12, 4e-12, 41), (2, 2, 41)).copy()
                tfr = mne.time_frequency.AverageTFRArray(epochs.info, values, epochs.times, [8, 12])
                widget = self.close_later(TimeFrequencyWidget(epochs, tfr=tfr))
                self.assertEqual(widget.baseline_checkbox.isChecked(), enabled)
                self.assertEqual(widget.baseline_start_input.isEnabled(), enabled)
                if not enabled:
                    np.testing.assert_allclose(widget.display_data.values, values.mean(axis=0) * 1e12)
                    self.assertIn("Reference: none", widget.context_label.text())
                widget.baseline_checkbox.setChecked(True)
                widget.transform_combo.setCurrentText(ZSCORE_MODE)
                self.assertEqual(widget.display_data.units, "Power (z-score)")
                widget.baseline_checkbox.setChecked(False)
                self.assertEqual(widget.transform_combo.currentText(), RAW_POWER_MODE)
                self.assertFalse(widget.baseline_stop_input.isEnabled())
                np.testing.assert_allclose(widget.display_data.values, values.mean(axis=0) * 1e12)


if __name__ == "__main__":
    unittest.main()
