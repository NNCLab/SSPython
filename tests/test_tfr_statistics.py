import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog
from statsmodels.stats.multitest import fdrcorrection

from core.analysis_events import compute_event_tfrs, tfr_event_metadata
from core.processing import TMSEEGAnalysis
from core.tfr_statistics import (
    bootstrap_power, compute_tfr_statistics, load_tfr_statistics,
    matching_tfr_mask, save_tfr_statistics,
)
from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.widgets.tep_analysis.natural_frequency import TFstat, calculate_nf_data, NaturalFrequencyApp
from ui.widgets.time_frequency_widget import (
    ComputeTFRSettingsDialog, RAW_POWER_MODE, TimeFrequencyWidget,
    build_time_frequency_display,
)


def historical_bootstrap(data, times, baseline, alpha, n_permutations, apply_fdr, seed):
    """Reference from the old script, kept independent of helpers."""
    rng = np.random.RandomState(seed)
    samples = np.where((times >= baseline[0]) & (times <= baseline[1]))[0]
    n_trials, n_channels, n_freqs, n_times = data.shape
    mask = np.zeros((n_channels, n_freqs, n_times), dtype=bool)
    for ch in range(n_channels):
        ch_data = data[:, ch]
        average = ch_data.mean(axis=0)
        boots = np.zeros((n_permutations, n_freqs, 2))
        for permutation in range(n_permutations):
            resampled = np.zeros((n_trials, n_freqs, len(samples)))
            for trial in range(n_trials):
                indices = rng.choice(samples, len(samples), replace=True)
                resampled[trial] = ch_data[trial, :, indices].T
            boot_mean = resampled.mean(axis=0)
            boots[permutation, :, 0] = boot_mean.max(axis=1)
            boots[permutation, :, 1] = boot_mean.min(axis=1)
        pvals = np.zeros_like(average)
        for freq in range(n_freqs):
            for time in range(n_times):
                p1 = np.sum(boots[:, freq, 0] >= average[freq, time]) / n_permutations
                p2 = np.sum(boots[:, freq, 1] <= average[freq, time]) / n_permutations
                pvals[freq, time] = min(p1, p2)
        if apply_fdr:
            _, corrected = fdrcorrection(pvals.ravel(), alpha=alpha / 2)
            pvals = corrected.reshape(pvals.shape)
        mask[ch] = pvals < alpha
    return mask


class TestBootstrapCompatibility(unittest.TestCase):
    def test_masks_match_historical_script_with_and_without_fdr(self):
        data = np.random.RandomState(5).uniform(1, 5, (4, 2, 3, 11))
        data[..., -1] = 15
        data[..., -2] = 0.1
        times = np.linspace(-0.5, 0.5, 11)
        for fdr in (False, True):
            expected = historical_bootstrap(data, times, (-0.5, -0.1), 0.1, 50, fdr, 42)
            actual = bootstrap_power(
                data, times, baseline=(-0.5, -0.1), alpha=0.1,
                n_permutations=50, fdr_correction=fdr, random_state=42, verbose=False,
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(actual[..., -2:].all())
            self.assertEqual(actual.dtype, bool)

    def test_zero_endpoint_and_singleton_axes_are_preserved(self):
        data = np.array([[[[1.0, 1.0, 100.0, 100.0]]]])
        times = np.array([-0.2, -0.1, 0.0, 0.1])
        mask = bootstrap_power(data, times, baseline=(None, 0), n_permutations=20,
                               random_state=3, verbose=False)
        self.assertEqual(mask.shape, (1, 1, 4))
        np.testing.assert_array_equal(mask, historical_bootstrap(data, times, (-0.2, 0), 0.05, 20, True, 3))

    def test_natural_frequency_and_core_use_the_same_test(self):
        data = np.random.RandomState(2).uniform(1, 4, (3, 1, 2, 10))
        times = np.linspace(-0.5, 0.4, 10)
        kwargs = dict(baseline=(-0.5, -0.1), n_permutations=15, verbose=False)
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(10)
        nf = TFstat(data[:, 0], times, **kwargs)
        np.random.seed(10)
        core = TMSEEGAnalysis.bootstrap_tfr(SimpleNamespace(data=data, times=times), **kwargs)
        np.testing.assert_array_equal(nf, core)

    def test_invalid_inputs_are_rejected_before_bootstrapping(self):
        times = np.linspace(-0.2, 0.2, 5)
        data = np.ones((2, 1, 1, 5))
        for kwargs in ({"baseline": None}, {"baseline": (1, 2)}, {"baseline": (0, 0)},
                       {"n_permutations": 0}, {"alpha": 0}, {"alpha": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_power(data, times, verbose=False, **kwargs)

    def test_natural_frequency_uses_all_mne_baseline_modes(self):
        times = np.array([-0.2, -0.1, 0.0, 0.1])
        values = np.array([[[[1., 3., 10., 10.], [9., 11., 20., 20.]]]])
        class Epochs:
            ch_names = ["Cz"]
            def copy(self): return self
            def pick(self, *args, **kwargs): return self
            def compute_tfr(self, **kwargs): return SimpleNamespace(data=values.copy(), ch_names=self.ch_names)
            def average(self): return SimpleNamespace(data=np.zeros((1, 4)))
        epochs = Epochs()
        epochs.times = times
        modes = ["mean", "ratio", "logratio", "percent", "zscore", "zlogratio", None]
        with patch("ui.widgets.tep_analysis.natural_frequency.TFstat", return_value=np.ones((2, 4))):
            for mode in modes:
                result = calculate_nf_data(epochs, ["Cz"], np.array([10, 20]), 1,
                                           (-0.2, -0.1), (0, 0.1), 10, mode)
                expected = values.mean(axis=0)
                if mode is not None:
                    expected = mne.baseline.rescale(expected, times, (-0.2, -0.1), mode=mode, verbose=False)
                np.testing.assert_allclose(result["avg_tfr"], expected)
                if mode in ("mean", "ratio"):
                    self.assertEqual(result["nat_freq"], 20 if mode == "mean" else 10)


class TestEventTFRStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        info = mne.create_info(["C3", "C4"], 200, "eeg")
        info.set_montage("standard_1020")
        events = np.array([[0, 0, 0], [100, 0, 2], [200, 0, 0], [300, 0, 2]])
        data = np.random.RandomState(5).normal(0, 1e-6, (4, 2, 81))
        data[1::2, :, 40:] *= 5
        self.epochs = mne.EpochsArray(data, info, events=events, event_id={"Pulse": 0, "Sham": 2},
                                      tmin=-0.2, baseline=None, verbose=False)
        self.params = dict(method="morlet", freqs=[10., 20.], n_cycles=1., average=True, verbose=False)
        self.settings = dict(baseline=(-0.2, -0.1), alpha=0.05, n_permutations=10,
                             fdr_correction=True, random_state=7, verbose=False)

    def close_later(self, widget):
        self.addCleanup(widget.close)
        return widget

    def compute(self, event_code=None, **overrides):
        params = dict(self.params, **overrides)
        power = compute_event_tfrs(self.epochs, event_code=event_code, **params)
        statistics = compute_tfr_statistics(self.epochs, power, event_code=event_code,
                                             compute_params=params, settings=self.settings)
        return power, statistics

    def test_event_masks_survive_save_reload_and_match_averaged_and_trial_outputs(self):
        powers, statistics = self.compute()
        trials, trial_statistics = self.compute(average=False)
        self.assertEqual([r["event_id"] for r in statistics["records"]], [None, 0, 2])
        for record in statistics["records"]:
            code = record["event_id"]
            target = next(p for p in powers if tfr_event_metadata(p)["event_id"] == code)
            np.testing.assert_array_equal(matching_tfr_mask(statistics, target, code),
                                          matching_tfr_mask(trial_statistics, trials, code))
        with tempfile.TemporaryDirectory() as folder:
            path, masks = Path(folder) / "power-tfr.h5", Path(folder) / "masks.npz"
            mne.time_frequency.write_tfrs(path, powers, overwrite=True, verbose=False)
            save_tfr_statistics(masks, statistics)
            reloaded = mne.time_frequency.read_tfrs(path, verbose=False)
            loaded_statistics = load_tfr_statistics(masks)
            self.assertEqual(loaded_statistics["settings"]["random_state"], 7)
            for power in reloaded:
                code = tfr_event_metadata(power)["event_id"]
                np.testing.assert_array_equal(matching_tfr_mask(loaded_statistics, power, code),
                                              matching_tfr_mask(statistics, power, code))
            self.assertIsNone(matching_tfr_mask(statistics, powers[0].copy().crop(-0.1, 0.1)))
            replacement = compute_event_tfrs(self.epochs, **self.params)
            self.assertIsNone(matching_tfr_mask(statistics, replacement[0]))
            reordered = powers[0].copy().reorder_channels(["C4", "C3"])
            np.testing.assert_array_equal(matching_tfr_mask(statistics, reordered),
                                          matching_tfr_mask(statistics, powers[0])[::-1])
            save_tfr_statistics(masks, None)
            self.assertFalse(masks.exists())

    def test_selected_event_zero_has_its_own_mask(self):
        power, statistics = self.compute(event_code=0)
        self.assertEqual([r["event_id"] for r in statistics["records"]], [0])
        self.assertEqual(statistics["records"][0]["n_epochs"], 2)
        self.assertIsNotNone(matching_tfr_mask(statistics, power, 0))
        self.assertIsNone(matching_tfr_mask(statistics, power, 2))

    def test_stockwell_and_multitaper_use_trial_statistics_on_matching_axes(self):
        for method in ("stockwell", "multitaper"):
            params = dict(self.params, method=method, freqs=[10, 30], decim=2)
            if method == "stockwell":
                params.pop("n_cycles")
                params["average"] = False  # MNE still returns condition averages.
            powers = compute_event_tfrs(self.epochs, **params)
            statistics = compute_tfr_statistics(self.epochs, powers, compute_params=params, settings=self.settings)
            self.assertEqual(len(powers), 3)
            for power in powers:
                self.assertEqual(matching_tfr_mask(statistics, power).shape, power.data.shape)
            if method == "stockwell":
                trial_powers = np.stack([self.epochs[i:i + 1].compute_tfr(**params).data for i in range(4)])
                np.testing.assert_allclose(trial_powers.mean(axis=0), powers[0].data)
                expected = bootstrap_power(trial_powers, powers[0].times, **self.settings)
                np.testing.assert_array_equal(matching_tfr_mask(statistics, powers[0]), expected)

    def test_compute_controls_validate_statistics_independently_of_display_baseline(self):
        dialog = self.close_later(ComputeTFRSettingsDialog(self.epochs))
        self.assertIsNone(dialog.get_statistics_params())
        dialog.output_combo.setCurrentText("complex")
        dialog.statistics_group.setChecked(True)
        self.assertEqual(dialog.output_combo.currentText(), "power")
        self.assertFalse(dialog.output_combo.isEnabled())
        self.assertIsNone(dialog._validation_error())
        dialog.seed_checkbox.setChecked(True)
        dialog.seed_input.setValue(12)
        self.assertEqual(dialog.get_statistics_params()["random_state"], 12)
        dialog.time_range_input.setValue((0, 0.2))
        self.assertIn("statistics baseline", dialog._validation_error())
        dialog.statistics_group.setChecked(False)
        self.assertIsNone(dialog._validation_error())
        nf = self.close_later(NaturalFrequencyApp(self.epochs))
        self.assertEqual(nf.baseline_mode_combo.currentText(), "mean")
        self.assertEqual([nf.baseline_mode_combo.itemText(i) for i in range(6)],
                         ["mean", "ratio", "logratio", "percent", "zscore", "zlogratio"])

    def test_alpha_fades_heatmap_and_masks_summaries_without_changing_erp_or_itc(self):
        powers, statistics = self.compute()
        mask = statistics["records"][0]["mask"]
        mask[:] = False
        mask[0, :, 40:] = True
        itc = self.epochs.compute_tfr(**dict(self.params, return_itc=True))[1]
        widget = self.close_later(TimeFrequencyWidget(
            self.epochs, tfr_derivatives={"Power": powers, "ITC": itc}, tfr_statistics={"Power": statistics},
        ))
        self.assertTrue(widget.mask_checkbox.isEnabled())
        self.assertFalse(widget.mask_checkbox.isChecked())
        original = widget.display_data.values.copy()
        erp = widget.display_data.erp_uV.copy()
        widget.mask_checkbox.setChecked(True)
        np.testing.assert_allclose(widget.display_data.values, original)
        np.testing.assert_allclose(widget.image_artist.get_alpha(), 0.25 + 0.75 * mask.mean(axis=0))
        expected = np.where(mask, powers[0].data * 1e12, 0).mean(axis=0)
        np.testing.assert_allclose(widget.display_data.summary_values, expected)
        tm, fm = widget._roi_masks()
        np.testing.assert_allclose(widget.spectrum_line.get_xdata(), expected[:, tm].mean(axis=1))
        np.testing.assert_allclose(widget.time_power_line.get_ydata(), expected[fm].mean(axis=0))
        self.assertIn(f"{expected[np.ix_(fm, tm)].mean():.3f}", widget.roi_mean_label.text())
        np.testing.assert_allclose(widget.display_data.erp_uV, erp)
        widget.channels_input.setText("C4")
        widget._on_display_setting_changed()
        np.testing.assert_allclose(widget.display_data.summary_values, 0)
        np.testing.assert_allclose(widget.image_artist.get_alpha(), 0.25)
        widget.derivative_combo.setCurrentIndex(1)
        self.assertFalse(widget.mask_checkbox.isEnabled())
        self.assertFalse(widget.mask_checkbox.isChecked())
        self.assertIsNone(widget.display_data.image_alpha)
        np.testing.assert_allclose(widget.display_data.values, itc.data[1])
        widget.derivative_combo.setCurrentIndex(0)
        self.assertTrue(widget.mask_checkbox.isChecked())
        widget.mask_checkbox.setChecked(False)
        np.testing.assert_allclose(widget.display_data.summary_values, widget.display_data.values)
        widget.event_combo.setCurrentIndex(widget.event_combo.findData(2))
        self.assertTrue(widget.mask_checkbox.isEnabled())
        statistics["records"] = [r for r in statistics["records"] if r["event_id"] != 2]
        widget._render()
        self.assertFalse(widget.mask_checkbox.isEnabled())

    def test_itc_ignores_masks_even_when_called_directly(self):
        itc = self.epochs.compute_tfr(**dict(self.params, return_itc=True))[1]
        display = build_time_frequency_display(epochs=self.epochs, tfr=itc, is_itc=True,
                                                stat_mask=np.zeros_like(itc.data, dtype=bool))
        np.testing.assert_allclose(display.summary_values, itc.data.mean(axis=0))
        self.assertIsNone(display.image_alpha)

    def test_masked_summaries_normalize_using_unmasked_baseline_per_channel(self):
        power, statistics = self.compute(event_code=0)
        mask = matching_tfr_mask(statistics, power)
        mask[:] = False
        mask[0, :, 40:] = True
        display = build_time_frequency_display(
            epochs=self.epochs, tfr=power, transform_mode=RAW_POWER_MODE,
            apply_baseline=True, baseline_s=(-0.2, -0.1), stat_mask=mask,
        )
        corrected = mne.baseline.rescale(power.data * 1e12, power.times, (-0.2, -0.1), verbose=False)
        np.testing.assert_allclose(display.summary_values, np.where(mask, corrected, 0).mean(axis=0))
        unmasked = build_time_frequency_display(
            epochs=self.epochs, tfr=power, transform_mode=RAW_POWER_MODE,
            apply_baseline=True, baseline_s=(-0.2, -0.1),
        )
        np.testing.assert_allclose(display.values, unmasked.values)

    def test_plot_window_recomputation_replaces_its_statistics(self):
        class ImmediateWorker:
            def __init__(self, function, **kwargs): self.function = function
            def exec_with_dialog(self, *args): return self.function()
        power, statistics = self.compute(event_code=0)
        widget = self.close_later(TimeFrequencyWidget(
            self.epochs, tfr_derivatives={"Power": power}, tfr_statistics={"Power": statistics},
        ))
        dialog = self.close_later(ComputeTFRSettingsDialog(self.epochs))
        dialog.freqs_input.setText("10,20")
        dialog.n_cycles_input.setText("1")
        dialog.statistics_group.setChecked(True)
        dialog.permutations_input.setValue(10)
        with (
            patch("ui.widgets.time_frequency_widget.ComputeTFRSettingsDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=QDialog.DialogCode.Accepted),
            patch("ui.widgets.time_frequency_widget.Worker", ImmediateWorker),
        ):
            widget.compute_time_frequency()
            self.assertTrue(widget.mask_checkbox.isEnabled())
            widget.mask_checkbox.setChecked(True)
            dialog.statistics_group.setChecked(False)
            widget.compute_time_frequency()
            self.assertFalse(widget.mask_checkbox.isEnabled())
            self.assertFalse(widget.mask_checkbox.isChecked())
            self.assertIsNone(widget.display_data.image_alpha)

    def test_compute_page_persists_statistics_and_clears_them_when_recomputed_without_statistics(self):
        class ImmediateWorker:
            def __init__(self, function, **kwargs): self.function = function
            def exec_with_dialog(self, *args): return self.function()
        with tempfile.TemporaryDirectory() as folder:
            epoch_path = Path(folder) / "test_preprocessed-epo.fif"
            self.epochs.save(epoch_path, overwrite=True, verbose=False)
            analysis = TMSEEGAnalysis(epoch_path, folder)
            page = self.close_later(ErpAnalysisPage())
            page.tep = analysis
            dialog = self.close_later(ComputeTFRSettingsDialog(analysis.epochs))
            dialog.freqs_input.setText("10,20")
            dialog.n_cycles_input.setText("1")
            dialog.average_checkbox.setChecked(True)
            dialog.statistics_group.setChecked(True)
            dialog.permutations_input.setValue(10)
            with (
                patch("ui.pages.erp_analysis_page.ComputeTFRSettingsDialog", return_value=dialog),
                patch.object(dialog, "exec", return_value=QDialog.DialogCode.Accepted),
                patch("ui.pages.erp_analysis_page.Worker", ImmediateWorker),
                patch.object(page, "update_ui_state"),
                patch.object(page, "open_time_frequency_widget"),
            ):
                page.run_time_frequency()
                saved = analysis._get_derivatives()
                self.assertIsNotNone(matching_tfr_mask(saved["tfr_statistics"], saved["tfr"], None))
                with patch("ui.pages.erp_analysis_page.TimeFrequencyDialog") as viewer:
                    page._show_time_frequency_dialog()
                    self.assertIs(viewer.call_args.kwargs["tfr_statistics"]["Power (TFR)"],
                                  analysis.derivatives["tfr_statistics"])
                for key in ("tfr_mask", "itc_mask"):
                    np.save(analysis.paths[key], np.ones((2, 2, 81), dtype=bool))
                saved_power = analysis.paths["tfr"].read_bytes()
                saved_masks = analysis.paths["tfr_statistics"].read_bytes()
                with (
                    patch("ui.pages.erp_analysis_page.compute_tfr_statistics", side_effect=ValueError("invalid baseline")),
                    self.assertRaisesRegex(ValueError, "invalid baseline"),
                ):
                    page.run_time_frequency()
                self.assertEqual(analysis.paths["tfr"].read_bytes(), saved_power)
                self.assertEqual(analysis.paths["tfr_statistics"].read_bytes(), saved_masks)
                dialog.statistics_group.setChecked(False)
                page.run_time_frequency()
                self.assertFalse(analysis.paths["tfr_statistics"].exists())
                self.assertFalse(analysis.paths["tfr_mask"].exists())
                self.assertFalse(analysis.paths["itc_mask"].exists())
                self.assertIsNone(analysis.derivatives["tfr_statistics"])


if __name__ == "__main__":
    unittest.main()
