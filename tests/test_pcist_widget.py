"""Interactive PCIst inspection uses cached results and exact event selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
from PySide6.QtWidgets import QApplication

from core.external.pci_st import distance2transition
from core.pcist import calculate_pcist
from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.widgets.tep_analysis.pcist import PCIstDialog


class _ImmediateWorker:
    def __init__(self, function, *args, **kwargs):
        kwargs.pop("parent", None)
        kwargs.pop("add_loggers", None)
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def exec_with_dialog(self, *args, **kwargs):
        return self.function(*self.args, **self.kwargs)


class TestPCIstWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        sfreq = 250.0
        times = np.arange(251) / sfreq - 0.5
        response = (times >= 0) & (times < 0.3)
        traces = []
        for frequency, phase in ((8, 0.1), (13, 0.7), (21, 1.2)):
            carrier = np.sin(2 * np.pi * frequency * times + phase)
            traces.append(carrier * np.where(response, 8.0, 0.2) * 1e-6)
        signal = np.array(traces)
        data = np.stack([signal, signal * 1.5, signal, signal * 1.5])
        cls.epochs = mne.EpochsArray(
            data, mne.create_info(["C3", "C4", "Cz"], sfreq, "eeg"),
            events=np.array([[0, 0, 0], [300, 0, 2], [600, 0, 0], [900, 0, 2]]),
            event_id={"Pulse": 0, "Pulse/Sham": 2}, tmin=-0.5,
            verbose=False,
        )
        cls.params = dict(
            baseline_window=(-400.0, -50.0), response_window=(0.0, 300.0),
            k=1.2, min_snr=0.0, max_var=100.0, n_components=None,
            n_steps=9, max_thr_p=1.0, embed=False, L=2, tau=1,
            baseline_corr=False, avgref=False, resample=None,
        )
        cls.result = calculate_pcist(cls.epochs, **cls.params)
        if cls.result["n_dims"] < 2:
            raise AssertionError("Synthetic data must retain at least two components.")

    def setUp(self):
        self.dialog = PCIstDialog(self.epochs)
        self.addCleanup(self.dialog.close)
        self.addCleanup(self.app.processEvents)

    def test_all_calculation_parameters_are_exposed(self):
        self.assertIsNone(self.dialog.result)
        self.assertTrue(set(self.params).issubset(self.dialog.get_parameters()))

    def test_rounded_window_controls_stay_within_exact_epoch_bounds(self):
        # At 512 Hz both endpoints round outward in the three-decimal controls.
        epochs = mne.EpochsArray(
            np.ones((2, 2, 254)) * 1e-6,
            mne.create_info(["C3", "C4"], 512.0, "eeg"),
            tmin=-0.2, verbose=False,
        )
        dialog = PCIstDialog(epochs)
        self.addCleanup(dialog.close)
        dialog.parameter_inputs["n_steps"].setValue(3)
        params = dialog.get_parameters()
        lower, upper = epochs.times[[0, -1]] * 1000
        self.assertEqual(lower, -199.21875)
        self.assertGreaterEqual(params["baseline_window"][0], lower)
        self.assertLessEqual(params["response_window"][1], upper)
        result = calculate_pcist(epochs, **params)
        self.assertTrue(np.isfinite(result["PCI"]))

    def test_slider_recomputes_transition_matrices_without_recalculating_pcist(self):
        self.dialog.set_result(self.result)
        optimum = int(np.argmax(self.result["NST_diff"][:, 0]))
        self.assertEqual(self.dialog.threshold_slider.value(), optimum)
        index = (optimum + 1) % self.params["n_steps"]
        with patch("ui.widgets.tep_analysis.pcist.calculate_pcist") as calculate:
            self.dialog.threshold_slider.setValue(index)
            self.app.processEvents()
        calculate.assert_not_called()
        threshold = self.result["thresholds"][index, 0]
        for axis, distance in zip(
            self.dialog.transition_axes,
            (self.result["D_base"][0], self.result["D_resp"][0]),
        ):
            np.testing.assert_array_equal(
                np.asarray(axis.images[0].get_array()),
                distance2transition(distance, threshold),
            )
        self.assertIs(self.dialog.result, self.result)

    def test_component_and_optimal_button_select_that_components_optimum(self):
        self.dialog.set_result(self.result)
        self.dialog.component_selector.setCurrentIndex(1)
        optimum = int(np.argmax(self.result["NST_diff"][:, 1]))
        self.assertEqual(self.dialog.threshold_slider.value(), optimum)
        self.dialog.threshold_slider.setValue((optimum + 1) % self.params["n_steps"])
        self.dialog.optimal_button.click()
        self.assertEqual(self.dialog.threshold_slider.value(), optimum)
        for axis, distance in zip(
            self.dialog.distance_axes,
            (self.result["D_base"][1], self.result["D_resp"][1]),
        ):
            np.testing.assert_array_equal(np.asarray(axis.images[0].get_array()), distance)

    def test_curve_click_selects_nearest_threshold(self):
        self.dialog.set_result(self.result)
        index = 2
        event = SimpleNamespace(
            inaxes=self.dialog.curve_ax,
            xdata=float(self.result["thresholds"][index, 0]), button=1,
        )
        self.dialog._on_curve_click(event)
        self.assertEqual(self.dialog.threshold_slider.value(), index)

    def test_curve_scaling_matches_component_contribution(self):
        self.dialog.set_result(self.result)
        curves = {line.get_label(): line for line in self.dialog.curve_ax.lines}
        response = np.asarray(curves["NST response"].get_ydata())
        baseline = np.asarray(curves["k × NST baseline (k=1.2)"].get_ydata())
        delta = np.asarray(curves["ΔNST (signed)"].get_ydata())
        np.testing.assert_allclose(response - baseline, delta)
        self.assertAlmostEqual(max(0, float(delta.max())), self.result["dNST"][0])
        np.testing.assert_allclose(
            response, self.result["NST_resp"][:, 0] * len(self.result["response_times"]),
        )

    def test_curve_click_respects_toolbar_zoom(self):
        self.dialog.set_result(self.result)
        before = self.dialog.threshold_slider.value()
        self.dialog.toolbar.zoom()
        event = SimpleNamespace(
            inaxes=self.dialog.curve_ax,
            xdata=float(self.result["thresholds"][(before + 1) % self.params["n_steps"], 0]),
            button=1,
        )
        self.dialog._on_curve_click(event)
        self.assertEqual(self.dialog.threshold_slider.value(), before)

    def test_parameter_change_marks_results_stale(self):
        self.dialog.set_result(self.result)
        self.dialog.channels_input.setText("C3, Cz")
        self.assertTrue(self.dialog.stale_label.text().strip())

    def test_zero_components_disables_matrix_inspection(self):
        result = calculate_pcist(self.epochs, **{**self.params, "min_snr": 1e12})
        self.assertEqual(result["PCI"], 0)
        self.dialog.set_result(result)
        self.assertFalse(self.dialog.component_selector.isEnabled())
        self.assertFalse(self.dialog.threshold_slider.isEnabled())
        self.assertFalse(self.dialog.optimal_button.isEnabled())

    def test_run_analysis_preserves_exact_event_zero_and_selected_channels(self):
        self.dialog.event_selector.setCurrentIndex(self.dialog.event_selector.findData(0))
        self.dialog.channels_input.setText("C3, Cz")
        with (
            patch("ui.widgets.tep_analysis.pcist.Worker", _ImmediateWorker),
            patch("ui.widgets.tep_analysis.pcist.calculate_pcist", wraps=calculate_pcist) as calculate,
        ):
            self.dialog.run_analysis()
        self.assertIsNotNone(self.dialog.result)
        self.assertEqual(calculate.call_args.kwargs["event_code"], 0)
        self.assertEqual(calculate.call_args.kwargs["picks"], ["C3", "Cz"])
        self.assertEqual(self.dialog.result["n_epochs"], 2)
        self.assertEqual(self.dialog.result["ch_names"], ["C3", "Cz"])

    def test_analysis_page_button_opens_pcist_for_loaded_epochs(self):
        page = ErpAnalysisPage()
        self.addCleanup(page.close)
        self.assertFalse(page.pcist_button.isEnabled())
        with patch("ui.pages.erp_analysis_page.PCIstDialog") as dialog_class:
            page.run_pcist()
            dialog_class.assert_not_called()
            page.tep = SimpleNamespace(epochs=self.epochs)
            page.pcist_button.setEnabled(True)
            page.pcist_button.click()
            dialog_class.assert_called_once_with(self.epochs, parent=page)
            dialog_class.return_value.exec.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
