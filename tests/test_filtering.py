import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.app_settings import SettingsStore
from core.pipelines import DatasetRecord, get_pipeline
from core.filtering import FilterConfig, NotchConfig, apply_bandpass, apply_notch
from core.processing import Preprocessor
from ui.widgets.ica_widget import ICAPlottingSettingsWidget
from ui.widgets.preprocessing_widgets import (
    ArtifactRemovalSettingsWidget, EpochingSettingsWidget, FilterContinuousDialog,
    FilterContinuousWidget, PreprocessingSettingsWidget,
)
from ui.widgets.run_ica_widget import ICASettingsWidget
from ui.widgets.dataset_files import DatasetFileController


def signal_raw():
    sfreq = 200
    times = np.arange(4000) / sfreq
    signal = np.sin(2 * np.pi * 10 * times) + np.sin(2 * np.pi * 60 * times)
    return mne.io.RawArray(np.array([signal, signal / 2]) * 1e-6,
                           mne.create_info(["Cz", "Pz"], sfreq, "eeg"), verbose=False)


class TestFilterDesigns(unittest.TestCase):
    def test_defaults_preserve_existing_bandpass_and_notch_results(self):
        source = signal_raw()
        expected = source.copy().notch_filter([50], verbose=False).filter(
            1, 40, method="iir", iir_params={"order": 3, "ftype": "butter", "phase": "zero-double"},
            verbose=False,
        )
        actual = source.copy()
        apply_notch(actual, 50)
        apply_bandpass(actual, (1, 40))
        np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-15)

    def test_method_phase_and_family_reach_mne(self):
        for config in (
            FilterConfig(iir_order=5, iir_ftype="cheby1", iir_rp=0.5, phase="forward"),
            FilterConfig(method="fir", filter_length="2s", fir_window="hann",
                         fir_design="firwin2", l_trans_bandwidth=2, h_trans_bandwidth=5),
        ):
            with self.subTest(config=config):
                raw = signal_raw()
                before = raw.get_data().copy()
                log = apply_bandpass(raw, (5, 40), config)
                self.assertEqual(log, config.to_dict())
                self.assertFalse(np.allclose(raw.get_data(), before, atol=1e-9))
                self.assertLess(np.std(raw.get_data()), np.std(before))

    def test_iir_notch_handles_harmonics(self):
        raw = signal_raw()
        config = NotchConfig(FilterConfig(iir_order=2), width=1, trans_bandwidth=1)
        log = apply_notch(raw, 30, config)
        self.assertEqual(log["frequencies"], [30, 60, 90])
        self.assertLess(np.std(raw.get_data()), np.std(signal_raw().get_data()))

    def test_invalid_frequency_and_method_combinations_are_rejected(self):
        raw = signal_raw()
        for bandpass, config in (
            ((40, 5), None), ((1, 100), None), ((float("nan"), 40), None),
            ((1, 40), FilterConfig(method="fir", phase="forward")),
            ((1, 40), FilterConfig(iir_order=0)),
        ):
            with self.subTest(bandpass=bandpass, config=config), self.assertRaises(ValueError):
                apply_bandpass(raw.copy(), bandpass, config)
        for frequency in (0, -50, 100, float("nan")):
            with self.subTest(notch=frequency), self.assertRaises(ValueError):
                apply_notch(raw.copy(), frequency)


class TestRepeatableFiltering(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "subject_raw.fif"
        signal_raw().save(self.source, overwrite=True, verbose=False)
        self.processor = self.load_processor()

    def load_processor(self):
        processor = Preprocessor(self.source, self.root / "derivatives")
        self.addCleanup(processor.release_cached_stages, processor.processing_order, False)
        return processor

    def test_continuous_rerun_after_reload_restarts_before_filtering_and_resampling(self):
        self.processor.filter_continuous(bandpass=(5, 40), resample=100)
        processor = self.load_processor()
        config = FilterConfig(iir_order=5, phase="forward")
        processor.filter_continuous(bandpass=(3, 70), filter_config=config)
        expected = signal_raw()
        apply_bandpass(expected, (3, 70), config)
        self.assertEqual(processor.filtered_raw.info["sfreq"], 200)
        np.testing.assert_allclose(processor.filtered_raw.get_data(), expected.get_data(), atol=1e-12)
        np.testing.assert_allclose(processor.raw.get_data(), signal_raw().get_data(), atol=1e-12)
        log = json.loads(processor.filtered_raw.info["description"])[-1]["filter_continuous"]
        self.assertEqual(log["filter_config"]["iir_order"], 5)

    def test_failed_design_preserves_existing_outputs_and_checkpoint(self):
        self.processor.filter_continuous(bandpass=(5, 40))
        epochs = self.processor.paths["epochs"]
        epochs.write_bytes(b"existing epochs")
        original = self.processor.paths["filtered_raw"].read_bytes()
        checkpoint = self.processor.paths["filter_input_raw"].read_bytes()
        for bandpass, config in (
            ((5, 100), None),
            ((5, 40), FilterConfig(method="fir", filter_length="1ms")),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.processor.filter_continuous(bandpass=bandpass, filter_config=config)
            self.assertEqual(epochs.read_bytes(), b"existing epochs")
            self.assertEqual(self.processor.paths["filtered_raw"].read_bytes(), original)
            self.assertEqual(self.processor.paths["filter_input_raw"].read_bytes(), checkpoint)
        # The placeholder is deliberately not an Epochs file.
        epochs.unlink()

    def test_rerun_preserves_artifact_cleanup_and_subsequent_review(self):
        raw = self.processor.raw.copy().load_data()
        raw.set_annotations(mne.Annotations([1], [0], ["Pulse"]))
        self.processor.update_raw_review(raw)
        self.processor.run_artifact_removal(window=(-0.01, 0.02), smoothing=None, verbose=False)
        self.processor.filter_continuous(bandpass=(5, 40), resample=100)
        reviewed = self.processor.filtered_raw.copy()
        reviewed.info["bads"] = ["Pz"]
        reviewed.annotations.append(2, 0.1, "BAD_review")
        self.processor.update_raw_review(reviewed)
        self.processor.filter_continuous(bandpass=(3, 60))
        self.assertEqual(self.processor.filtered_raw.info["bads"], ["Pz"])
        self.assertIn("BAD_review", self.processor.filtered_raw.annotations.description)
        history = self.processor._processing_log(self.processor.filtered_raw)
        self.assertEqual(sum("artifact_removal" in entry for entry in history), 1)
        self.assertEqual(sum("filter_continuous" in entry for entry in history), 1)
        self.assertEqual(self.processor.filtered_raw.info["sfreq"], 200)

    def test_epoching_is_repeatable_and_final_filter_records_design(self):
        config = FilterConfig(iir_order=5, phase="forward")
        self.processor.run_epoching(mode="fixed", fixed_duration=2,
                                    bandpass=(5, 40), filter_config=config, verbose=False)
        first = self.processor.epochs.get_data().copy()
        self.processor.run_epoching(mode="fixed", fixed_duration=2,
                                    bandpass=(5, 40), filter_config=config, verbose=False)
        np.testing.assert_allclose(self.processor.epochs.get_data(), first, atol=1e-15)
        np.testing.assert_allclose(self.processor.raw.get_data(), signal_raw().get_data(), atol=1e-12)
        self.processor.filter_epochs(bandpass=(8, 35), filter_config=config,
                                      interpolate_bad_channels=False, verbose=False)
        log = self.processor._processing_log(self.processor.preprocessed)[-1]["filter_epochs"]
        self.assertEqual(log["filter_config"], config.to_dict())

    def test_legacy_derivative_rebuilds_interpolation_from_original_samples(self):
        raw = self.processor.raw.copy().load_data()
        raw.set_annotations(mne.Annotations([1], [0], ["Pulse"]))
        self.processor.update_raw_review(raw)
        self.processor.run_artifact_removal(window=(-0.01, 0.02), smoothing=None, verbose=False)
        self.processor.filter_continuous(bandpass=(5, 40))
        expected = self.processor.filtered_raw.get_data().copy()
        self.processor.paths["filter_input_raw"].unlink()
        self.processor.filter_continuous(bandpass=(5, 40))
        np.testing.assert_allclose(self.processor.filtered_raw.get_data(), expected, atol=1e-12)

    def test_invalid_epoch_and_final_filters_preserve_saved_outputs(self):
        self.processor.run_epoching(mode="fixed", fixed_duration=2, verbose=False)
        self.processor.filter_epochs(bandpass=(8, 35), interpolate_bad_channels=False,
                                      verbose=False)
        saved = {stage: self.processor.paths[stage].read_bytes()
                 for stage in ("epochs", "preprocessed")}
        with self.assertRaises(ValueError):
            self.processor.run_epoching(mode="fixed", fixed_duration=2, bandpass=(1, 150),
                                        resample=50, verbose=False)
        with self.assertRaises(ValueError):
            self.processor.filter_epochs(bandpass=(1, 150), interpolate_bad_channels=False,
                                          verbose=False)
        for stage, content in saved.items():
            self.assertEqual(self.processor.paths[stage].read_bytes(), content)


class TestAdvancedSettingsUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SettingsStore(QSettings(str(Path(self.directory.name) / "settings.ini"),
                                             QSettings.Format.IniFormat))
        for module in ("preprocessing_widgets", "run_ica_widget", "ica_widget"):
            patcher = patch(f"ui.widgets.{module}.get_settings_store", return_value=self.store)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_filter_defaults_are_independent_and_old_settings_load(self):
        self.store.set("pipelines/standard/preprocessing/continuous_filter",
                       {"bandpass": (2, 40), "notch": 50})
        continuous = FilterContinuousWidget()
        epoching = EpochingSettingsWidget()
        final = PreprocessingSettingsWidget()
        self.assertEqual(continuous.get_params()["filter_config"], FilterConfig().to_dict())
        continuous.advanced_settings.bandpass_design.order_input.setValue(5)
        continuous.save_settings()
        reloaded = FilterContinuousWidget()
        self.assertEqual(reloaded.get_params()["filter_config"]["iir_order"], 5)
        self.assertEqual(epoching.get_params()["filter_config"]["iir_order"], 3)
        self.assertEqual(final.get_params()["filter_config"]["iir_order"], 3)
        reloaded.restore_defaults()
        self.assertEqual(reloaded.get_params()["bandpass"], (0.5, None))
        self.assertEqual(reloaded.get_params()["filter_config"]["iir_order"], 3)
        self.assertEqual(FilterContinuousWidget().get_params()["filter_config"]["iir_order"], 5)

    def test_advanced_controls_hide_without_changing_their_values(self):
        artifact = ArtifactRemovalSettingsWidget()
        ica = ICASettingsWidget("continuous", max_components=8)
        inspection = ICAPlottingSettingsWidget()
        for widget, control, read in (
            (artifact, artifact.span_spinbox, artifact.get_params),
            (ica, ica.max_iter_input, ica.get_params),
            (inspection, inspection.decimate_input, inspection.get_settings),
        ):
            with self.subTest(widget=type(widget).__name__):
                before = read()
                self.assertTrue(widget.advanced_settings.content.isHidden())
                self.assertTrue(widget.advanced_settings.content.isAncestorOf(control))
                widget.advanced_settings.toggle_button.click()
                self.assertFalse(widget.advanced_settings.content.isHidden())
                widget.advanced_settings.toggle_button.click()
                self.assertEqual(read(), before)

    def test_fir_design_and_notch_settings_roundtrip(self):
        widget = FilterContinuousWidget()
        design = FilterConfig(method="fir", phase="minimum", filter_length="3s",
                              fir_window="hann", fir_design="firwin2",
                              l_trans_bandwidth=1.5, h_trans_bandwidth=5)
        notch = NotchConfig(FilterConfig(iir_order=6, phase="forward"), width=2,
                            trans_bandwidth=0.5)
        widget.set_params(widget.get_defaults() | {
            "filter_config": design.to_dict(), "notch_config": notch.to_dict(),
        })
        widget.save_settings()
        reloaded = FilterContinuousWidget()
        self.assertEqual(reloaded.get_params()["filter_config"], design.to_dict())
        self.assertEqual(reloaded.get_params()["notch_config"], notch.to_dict())
        reloaded.set_pipeline("another")
        self.assertEqual(reloaded.get_params()["filter_config"], FilterConfig().to_dict())

    def test_resetting_continuous_derivatives_includes_filter_input(self):
        root = Path(self.directory.name)
        pipeline = get_pipeline("standard")
        dataset = DatasetRecord(root / "source_raw.fif", pipeline, root, {
            "raw": root / "source_raw.fif",
            "filtered_raw": root / "source_desc-filtered_raw.fif",
            "filter_input_raw": root / "source_desc-filterinput_raw.fif",
        })
        dataset.paths["filtered_raw"].touch()
        dataset.paths["filter_input_raw"].touch()
        panel = DatasetFileController()
        panel.current_dataset = dataset
        self.assertEqual(panel._existing_derivative_paths_from_stage("filtered_raw"), [
            dataset.paths["filtered_raw"], dataset.paths["filter_input_raw"],
        ])
        self.assertEqual(panel._existing_derivative_paths_from_stage("epochs"), [])
        panel.deleteLater()

    def test_applying_filter_does_not_silently_save_defaults(self):
        processor = Mock()
        processor.has.return_value = False
        dialog = FilterContinuousDialog(processor)
        original_defaults = self.store.get(dialog.settings_widget.get_settings_path())
        dialog.settings_widget.advanced_settings.bandpass_design.order_input.setValue(6)
        with patch("ui.widgets.preprocessing_widgets.Worker") as worker:
            worker.return_value._error = None
            dialog.run_process()
        self.assertEqual(self.store.get(dialog.settings_widget.get_settings_path()), original_defaults)
        dialog._save_defaults()
        self.assertEqual(self.store.get(dialog.settings_widget.get_settings_path())["filter_config"]["iir_order"], 6)
        dialog.close()


if __name__ == "__main__":
    unittest.main()
