import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog

from core.analysis_events import (
    compute_event_tfrs,
    event_options,
    select_event_epochs,
    tfr_event_metadata,
)
from core.processing import TMSEEGAnalysis
from ui.pages.erp_analysis_page import ErpAnalysisPage, TopoplotTimeWindowDialog
from ui.widgets.event_selection import EventSelector
from ui.widgets.evoked_plot import EvokedPlotWidget
from ui.widgets.psd_plot import PSDPlotSettingsDialog
from ui.widgets.source_estimate_widget import ComputeSTCSettingsDialog, compute_stc
from ui.widgets.tep_analysis.excitability import AnalysisMode, ExcitabilityApp
from ui.widgets.tep_analysis.natural_frequency import NaturalFrequencyApp
from ui.widgets.time_frequency_widget import (
    ComputeTFRSettingsDialog,
    RAW_POWER_MODE,
    TimeFrequencyWidget,
    build_time_frequency_display,
)


class _ImmediateWorker:
    def __init__(self, function, **kwargs):
        self.function = function

    def exec_with_dialog(self, *args):
        return self.function()


class TestAnalysisEventSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        info = mne.create_info(["C3", "C4", "Cz", "Pz"], 1000, "eeg")
        info.set_montage("standard_1020")
        events = np.array([[0, 0, 0], [600, 0, 2], [1200, 0, 0], [1800, 0, 2]])
        signal = np.sin(np.arange(501) * 2 * np.pi * 10 / 1000) * 1e-6
        data = np.tile(signal, (4, 4, 1))
        data[events[:, 2] == 2] *= 3
        self.epochs = mne.EpochsArray(
            data, info, events=events,
            event_id={"Pulse": 0, "Pulse/Sham": 2, "Alias": 0, "Absent": 8},
            tmin=-0.2, on_missing="ignore", verbose=False,
        )
        self.addCleanup(self.app.processEvents)

    def select(self, selector, code):
        selector.setCurrentIndex(selector.findData(code))

    def test_exact_selection_handles_zero_aliases_hierarchies_and_absent_events(self):
        selector = EventSelector(self.epochs)
        self.addCleanup(selector.close)
        self.assertEqual(selector.count(), 3)
        self.assertEqual(event_options(self.epochs), [(0, "Pulse / Alias", 2), (2, "Pulse/Sham", 2)])
        self.select(selector, 0)
        np.testing.assert_array_equal(selector.selected_epochs().events[:, 2], [0, 0])
        self.assertIn("event_id=0", selector.selected_label())
        np.testing.assert_array_equal(select_event_epochs(self.epochs, "Pulse").events[:, 2], [0, 0])
        with self.assertRaisesRegex(ValueError, "No epochs match"):
            select_event_epochs(self.epochs, 8)
        self.select(selector, None)
        self.assertEqual(len(selector.selected_epochs()), 4)

    def test_evoked_plot_uses_shared_selector_for_exact_ids_and_good_channels(self):
        self.epochs.info["bads"] = ["C4"]
        original = self.epochs.get_data().copy()
        widget = EvokedPlotWidget(params={"baseline": None})
        self.addCleanup(widget.close)
        widget.update_plot(self.epochs, label="Test")
        self.assertIsInstance(widget.event_selector, EventSelector)
        for code in (2, 0, None):
            self.select(widget.event_selector, code)
            expected = select_event_epochs(self.epochs, code).average().pick(["C3", "Cz", "Pz"])
            np.testing.assert_allclose(widget.display_evoked.data, expected.data)
            self.assertNotIn("C4", widget.display_evoked.ch_names)
            self.assertEqual(widget.evoked.nave, 4 if code is None else 2)
            self.assertIn(widget.event_selector.selected_label(), widget.title_label.text())
        np.testing.assert_array_equal(self.epochs.get_data(), original)

    def test_evoked_selector_retains_valid_selection_and_clears_stale_epochs(self):
        widget = EvokedPlotWidget(params={"baseline": None})
        self.addCleanup(widget.close)
        widget.update_plot(self.epochs)
        self.select(widget.event_selector, 2)
        widget.update_plot(self.epochs.copy())
        self.assertEqual(widget.event_selector.currentData(), 2)
        self.assertEqual(widget.evoked.nave, 2)

        remaining = select_event_epochs(self.epochs, 0)
        widget.update_plot(remaining)
        self.assertIsNone(widget.event_selector.currentData())
        self.assertEqual(widget.event_selector.count(), 2)
        np.testing.assert_allclose(widget.evoked.data, remaining.average().data)
        self.select(widget.event_selector, 0)
        self.assertEqual(widget.selected_event_code, 0)

        widget.update_plot(remaining.average())
        self.assertTrue(widget.event_selector.isHidden())
        self.assertFalse(widget.event_selector.isEnabled())
        self.assertIsNone(widget.event_selector.epochs)
        self.assertIsNone(widget.selected_event_code)
        self.assertIsNone(widget.selected_event_label)
        widget.update_plot(None)
        self.assertIsNone(widget.display_evoked)

    def test_response_amplitude_updates_traces_metrics_and_topographies(self):
        dialog = ExcitabilityApp(self.epochs)
        self.addCleanup(dialog.close)
        dialog.channels_input.setText("C3, C4")
        for code in (2, 0, None):
            self.select(dialog.event_selector, code)
            expected = select_event_epochs(self.epochs, code).average().data
            np.testing.assert_allclose(dialog.ax.lines[0].get_ydata(), expected[0] * 1e6)
            with patch("mne.viz.plot_topomap", wraps=mne.viz.plot_topomap) as topomap:
                dialog._update_view()
            indices = self.epochs.time_as_index(dialog.span * 1e-3)
            window = expected[:, indices[0]:indices[1] + 1]
            np.testing.assert_allclose(topomap.call_args_list[0].args[0], window.mean(axis=1))
            np.testing.assert_allclose(topomap.call_args_list[1].args[0], np.sqrt((window ** 2).mean(axis=1)))
            for mode in AnalysisMode:
                dialog._change_mode(mode)
                trace = expected[0] * 1e6
                if mode in (AnalysisMode.GMFP, AnalysisMode.LFP):
                    trace = np.abs(trace)
                np.testing.assert_allclose(dialog.outcome_ax.lines[0].get_ydata(), trace, atol=1e-12)
            self.assertIn(dialog.event_selector.selected_label(), dialog.figure._suptitle.get_text())
        self.assertEqual(len(self.epochs), 4)

    def test_natural_frequency_uses_selected_trials_and_clears_old_result(self):
        dialog = NaturalFrequencyApp(self.epochs)
        self.addCleanup(dialog.close)
        dialog.channels_input.setText("C3 C4")

        def calculate(epochs, roi, freqs, *args):
            evoked = epochs.average().data
            return {
                "nat_freq": 10.0, "times": epochs.times, "freqs": freqs, "roi": roi,
                "avg_tfr": np.ones((2, len(freqs), len(epochs.times))),
                "stat_mask": np.ones((2, len(freqs), len(epochs.times))),
                "agg_freq": np.ones(len(freqs)), "evoked_all": evoked,
                "evoked_roi": evoked[:2],
            }

        for code in (2, 0):
            self.select(dialog.event_selector, code)
            self.assertEqual(len(dialog.figure.axes), 0)
            with (
                patch("ui.widgets.tep_analysis.natural_frequency.Worker", _ImmediateWorker),
                patch("ui.widgets.tep_analysis.natural_frequency.calculate_nf_data", side_effect=calculate) as calculator,
            ):
                dialog.run_analysis()
            np.testing.assert_array_equal(calculator.call_args.args[0].events[:, 2], [code, code])
            self.assertIn(f"event_id={code}", dialog.figure._suptitle.get_text())
            self.assertEqual(len(dialog.figure.axes), 4)

    def test_psd_page_plots_only_the_selected_event(self):
        page = ErpAnalysisPage()
        self.addCleanup(page.close)
        page.tep = SimpleNamespace(epochs=self.epochs, label="Test")
        dialog = PSDPlotSettingsDialog(self.epochs)
        self.addCleanup(dialog.close)
        self.select(dialog.event_selector, 0)
        with (
            patch("ui.pages.erp_analysis_page.PSDPlotSettingsDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=QDialog.DialogCode.Accepted),
            patch.object(mne.BaseEpochs, "plot_psd", autospec=True) as plot,
        ):
            page.plot_psd()
        np.testing.assert_array_equal(plot.call_args.args[0].events[:, 2], [0, 0])
        self.assertIn("event_id=0", plot.return_value.suptitle.call_args.args[0])

    def test_topoplot_uses_selected_event_and_time_window(self):
        page = ErpAnalysisPage()
        self.addCleanup(page.close)
        page.tep = SimpleNamespace(epochs=self.epochs, label="Test")
        dialog = TopoplotTimeWindowDialog(self.epochs)
        self.addCleanup(dialog.close)
        self.select(dialog.event_selector, 0)
        dialog.start_input.setValue(0.02)
        dialog.end_input.setValue(0.12)
        with (
            patch("ui.pages.erp_analysis_page.TopoplotTimeWindowDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=QDialog.DialogCode.Accepted),
            patch.object(mne.Evoked, "plot_topo", autospec=True) as plot,
        ):
            plot.return_value.axes = []
            page.topoplot()
        expected = select_event_epochs(self.epochs, 0).average().crop(0.02, 0.12)
        np.testing.assert_allclose(plot.call_args.args[0].data, expected.data)
        self.assertEqual(plot.call_args.args[0].nave, 2)
        self.assertIn("event_id=0", plot.call_args.kwargs["title"])

    def test_stc_uses_exact_event_for_covariance_and_evoked(self):
        dialog = ComputeSTCSettingsDialog(self.epochs)
        self.addCleanup(dialog.close)
        self.select(dialog.condition_combo, 0)
        params = dialog.get_compute_params()
        params["set_eeg_reference"] = False
        with (
            patch("ui.widgets.source_estimate_widget.prepare_stc_source_model"),
            patch("ui.widgets.source_estimate_widget.mne.compute_covariance") as covariance,
            patch("ui.widgets.source_estimate_widget.mne.make_forward_solution"),
            patch("ui.widgets.source_estimate_widget.make_inverse_operator"),
            patch("ui.widgets.source_estimate_widget.apply_inverse") as inverse,
        ):
            compute_stc(self.epochs, **params)
        np.testing.assert_array_equal(covariance.call_args.args[0].events[:, 2], [0, 0])
        np.testing.assert_allclose(inverse.call_args.args[0].data, select_event_epochs(self.epochs, 0).average().data)
        page = ErpAnalysisPage()
        self.addCleanup(page.close)
        page.tep = SimpleNamespace(epochs=self.epochs, label="Test", derivatives={"stc_metadata": {"compute": {"condition": 0}}})
        self.assertIn("event_id=0", page._stc_plot_title())

    def tfr_params(self, **kwargs):
        return dict(method="morlet", freqs=[10.0, 20.0], n_cycles=1.0, verbose=False, **kwargs)

    def test_averaged_tfr_and_itc_keep_event_selection_after_save_reload(self):
        power, itc = compute_event_tfrs(self.epochs, **self.tfr_params(average=True, return_itc=True))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "power-tfr.h5"
            itc_path = Path(folder) / "itc-tfr.h5"
            mne.time_frequency.write_tfrs(path, power, overwrite=True, verbose=False)
            mne.time_frequency.write_tfrs(itc_path, itc, overwrite=True, verbose=False)
            power = mne.time_frequency.read_tfrs(path, verbose=False)
            itc = mne.time_frequency.read_tfrs(itc_path, verbose=False)
        widget = TimeFrequencyWidget(self.epochs, tfr_derivatives={"Power (TFR)": power, "ITC": itc})
        self.addCleanup(widget.close)
        widget.transform_combo.setCurrentText(RAW_POWER_MODE)
        widget.baseline_checkbox.setChecked(False)
        self.assertEqual(widget.event_combo.count(), 3)
        for code in (2, 0, None):
            self.select(widget.event_combo, code)
            expected_power, expected_itc = select_event_epochs(self.epochs, code).compute_tfr(**self.tfr_params(average=True, return_itc=True))
            np.testing.assert_allclose(widget.display_data.values, expected_power.data.mean(axis=0) * 1e12)
            np.testing.assert_allclose(widget.display_data.erp_channel_uV, select_event_epochs(self.epochs, code).average().data * 1e6)
            self.assertEqual(widget.display_data.n_epochs, 4 if code is None else 2)
            widget.derivative_combo.setCurrentIndex(1)
            np.testing.assert_allclose(widget.display_data.values, expected_itc.data.mean(axis=0))
            widget.derivative_combo.setCurrentIndex(0)

    def test_specific_average_cannot_be_mislabeled_as_all_events(self):
        dialog = ComputeTFRSettingsDialog(self.epochs)
        self.addCleanup(dialog.close)
        self.select(dialog.event_selector, 0)
        power = compute_event_tfrs(self.epochs, event_code=dialog.event_selector.currentData(), **self.tfr_params(average=True))
        self.assertEqual(tfr_event_metadata(power), {"event_id": 0})
        widget = TimeFrequencyWidget(self.epochs, tfr=power)
        self.addCleanup(widget.close)
        self.assertEqual(widget.event_combo.count(), 1)
        self.assertEqual(widget.event_combo.currentData(), 0)
        self.assertEqual(widget.display_data.event_label, "Pulse / Alias")
        np.testing.assert_allclose(widget.display_data.erp_channel_uV, select_event_epochs(self.epochs, 0).average().data * 1e6)
        with self.assertRaisesRegex(ValueError, "pooled TFR"):
            build_time_frequency_display(epochs=self.epochs, tfr=power, event_code=2)

    def test_trial_tfr_selection_also_works_when_wrapped_in_a_list(self):
        power = compute_event_tfrs(self.epochs, **self.tfr_params(average=False))
        display = build_time_frequency_display(
            epochs=self.epochs, tfr=[power], event_code=2,
            transform_mode=RAW_POWER_MODE, apply_baseline=False,
        )
        expected = select_event_epochs(self.epochs, 2).compute_tfr(**self.tfr_params(average=False))
        np.testing.assert_allclose(display.values, expected.data.mean(axis=(0, 1)) * 1e12)
        self.assertEqual(display.n_epochs, 2)

    def test_compute_page_saves_selected_event_and_removes_incompatible_old_itc(self):
        with tempfile.TemporaryDirectory() as folder:
            epoch_path = Path(folder) / "test_preprocessed-epo.fif"
            self.epochs.save(epoch_path, overwrite=True, verbose=False)
            analysis = TMSEEGAnalysis(epoch_path, output_dir=folder)
            _, old_itc = self.epochs.compute_tfr(**self.tfr_params(average=True, return_itc=True))
            old_itc.save(analysis.paths["itc"], overwrite=True, verbose=False)
            analysis.derivatives["itc"] = old_itc
            page = ErpAnalysisPage()
            self.addCleanup(page.close)
            page.tep = analysis
            dialog = ComputeTFRSettingsDialog(analysis.epochs)
            self.addCleanup(dialog.close)
            self.select(dialog.event_selector, 0)
            dialog.freqs_input.setText("10,20")
            dialog.n_cycles_input.setText("1")
            dialog.average_checkbox.setChecked(True)
            with (
                patch("ui.pages.erp_analysis_page.ComputeTFRSettingsDialog", return_value=dialog),
                patch.object(dialog, "exec", return_value=QDialog.DialogCode.Accepted),
                patch("ui.pages.erp_analysis_page.Worker", _ImmediateWorker),
                patch.object(page, "update_ui_state"),
                patch.object(page, "open_time_frequency_widget"),
            ):
                page.run_time_frequency()
            saved = mne.time_frequency.read_tfrs(analysis.paths["tfr"], verbose=False)
            self.assertEqual(tfr_event_metadata(saved), {"event_id": 0})
            self.assertEqual(saved.nave, 2)
            self.assertFalse(analysis.paths["itc"].exists())
            self.assertIsNone(analysis.derivatives["itc"])
            self.assertEqual(len(analysis.epochs), 4)

    def test_compute_from_legacy_average_allows_selecting_any_epoch_event(self):
        legacy = self.epochs.compute_tfr(**self.tfr_params(average=True))
        widget = TimeFrequencyWidget(self.epochs, tfr=legacy)
        self.addCleanup(widget.close)
        dialog = ComputeTFRSettingsDialog(self.epochs)
        self.addCleanup(dialog.close)

        def configure_and_accept():
            self.select(dialog.event_selector, 2)
            dialog.freqs_input.setText("10,20")
            dialog.n_cycles_input.setText("1")
            dialog.average_checkbox.setChecked(True)
            return QDialog.DialogCode.Accepted

        with (
            patch("ui.widgets.time_frequency_widget.ComputeTFRSettingsDialog", return_value=dialog),
            patch.object(dialog, "exec", side_effect=configure_and_accept),
            patch("ui.widgets.time_frequency_widget.Worker", _ImmediateWorker),
        ):
            widget.compute_time_frequency()
        self.assertEqual(widget.event_combo.currentData(), 2)
        self.assertEqual(widget.display_data.n_epochs, 2)
        self.assertEqual(widget.display_data.event_label, "Pulse/Sham")
        np.testing.assert_allclose(widget.display_data.erp_channel_uV, select_event_epochs(self.epochs, 2).average().data * 1e6)

    def test_derivative_loader_preserves_pooled_api_and_event_viewer_results(self):
        with tempfile.TemporaryDirectory() as folder:
            epoch_path = Path(folder) / "test_preprocessed-epo.fif"
            self.epochs.save(epoch_path, overwrite=True, verbose=False)
            analysis = TMSEEGAnalysis(epoch_path, output_dir=folder)
            power = compute_event_tfrs(self.epochs, **self.tfr_params(average=True))
            mne.time_frequency.write_tfrs(analysis.paths["tfr"], power, overwrite=True, verbose=False)
            analysis.derivatives = analysis._get_derivatives()
            self.assertIsInstance(analysis.derivatives["tfr"], mne.time_frequency.AverageTFR)
            self.assertEqual(len(analysis.derivatives["tfr_events"]), 3)
            page = ErpAnalysisPage()
            self.addCleanup(page.close)
            page.tep = analysis
            self.assertIs(page._available_time_frequency_derivatives()["Power (TFR)"], analysis.derivatives["tfr_events"])


if __name__ == "__main__":
    unittest.main()
