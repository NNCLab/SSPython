import unittest

import mne
import numpy as np

from core.external import pci_st
from core.pcist import calculate_pcist


def make_epochs():
    rng = np.random.default_rng(8)
    times = np.arange(251) / 250 - 0.5
    data = rng.normal(0, 0.1e-6, (4, 4, 251))
    response = times >= 0
    for channel, frequency in enumerate((9, 17, 31)):
        data[:, channel, response] += 3e-6 * np.sin(2 * np.pi * frequency * times[response])
    data[2:] *= 2
    epochs = mne.EpochsArray(
        data, mne.create_info(["Cz", "Pz", "Fz", "EOG"], 250,
                             ["eeg", "eeg", "eeg", "eog"]),
        events=np.array([[0, 0, 0], [251, 0, 0], [502, 0, 1], [753, 0, 1]]),
        event_id={"stim/A": 0, "stim/B": 1}, tmin=-0.5,
        baseline=None, verbose=False,
    )
    epochs.info["bads"] = ["Fz"]
    return epochs


def dense_reference(signal, times, baseline_window, response_window, *, k,
                    n_steps, max_thr_p=1, embed=False, L=2, tau=1):
    """Small independent reference retaining all thresholds' matrices."""
    if embed:
        cut = (L - 1) * tau
        coordinates = np.stack([
            signal[:, cut - lag * tau:signal.shape[1] - lag * tau]
            for lag in range(L)
        ], axis=1)
        times = times[cut:]
    else:
        coordinates = signal[:, None, :]
    parts = [coordinates[:, :, (times >= a) & (times < b)]
             for a, b in (baseline_window, response_window)]
    distances = [np.linalg.norm(part[:, :, :, None] - part[:, :, None, :], axis=1)
                 for part in parts]
    thresholds = np.array([
        np.linspace(np.median(base), np.max(response) * max_thr_p, n_steps)
        for base, response in zip(*distances)
    ]).T
    matrices, curves = [], []
    for distance in distances:
        recurrence = distance[None, :, :, :] <= thresholds[:, :, None, None]
        transition = np.pad(np.abs(np.diff(recurrence.astype(int), axis=3)),
                            ((0, 0), (0, 0), (0, 0), (0, 1)))
        matrices.append(transition)
        curves.append(transition.mean(axis=(2, 3)))
    difference = curves[1] - k * curves[0]
    best = difference.argmax(axis=0)
    indices = np.arange(signal.shape[0])
    return {
        "D_base": distances[0], "D_resp": distances[1],
        "NST_base": curves[0], "NST_resp": curves[1],
        "NST_diff": difference, "thresholds": thresholds,
        "max_thresholds": thresholds[best, indices],
        "T_base": matrices[0][best, indices],
        "T_resp": matrices[1][best, indices],
        "dNST": np.maximum(difference[best, indices] * parts[1].shape[-1], 0),
    }


class TestPCIstAlgorithm(unittest.TestCase):
    def test_streamed_threshold_scan_matches_dense_reference(self):
        times = np.arange(-20, 21, dtype=float)
        signal = np.random.default_rng(41).normal(size=(3, len(times)))
        signal[:, times >= 0] *= 3
        for embed in (False, True):
            with self.subTest(embed=embed):
                params = dict(baseline_window=(-20, -3), response_window=(0, 20),
                              k=1.2, n_steps=11, max_thr_p=0.9,
                              embed=embed, L=3, tau=2)
                expected = dense_reference(signal, times, **params)
                actual = pci_st.state_transition_quantification(signal, times, **params)
                for key, value in expected.items():
                    np.testing.assert_allclose(actual[key], value, err_msg=key)

    def test_component_variance_tracks_snr_filtered_original_rank(self):
        times = np.arange(-3, 4)
        signal = np.array([[10, 10, 10, 9, 0, 0, 0],
                           [.1, .1, .1, 0, 3, 0, 0]])
        result = pci_st.calc_PCIst(
            signal, times, full_return=True, baseline_window=(-3, 0),
            response_window=(0, 3), k=1.2, max_var=100, min_snr=1,
        )
        np.testing.assert_array_equal(result["component_indices"], [1])
        np.testing.assert_allclose(result["var_exp"], [10])
        self.assertEqual(result["signal_svd"].shape[0], 1)

    def test_zero_signal_returns_complete_empty_diagnostics(self):
        epochs = make_epochs()
        epochs._data[:] = 0
        with np.errstate(all="raise"):
            result = calculate_pcist(epochs)
        self.assertEqual(result["PCI"], 0)
        self.assertEqual(result["n_dims"], 0)
        self.assertEqual(result["thresholds"].shape, (100, 0))
        self.assertEqual(result["D_base"].shape, (0, len(result["baseline_times"]),
                                                  len(result["baseline_times"])))
        self.assertEqual(result["T_resp"].shape, (0, len(result["response_times"]),
                                                  len(result["response_times"])))

    def test_zero_baseline_with_response_retains_infinite_snr(self):
        epochs = make_epochs()
        epochs._data[:, :, epochs.times < 0] = 0
        result = calculate_pcist(epochs, min_snr=0)
        self.assertGreater(result["n_dims"], 0)
        self.assertTrue(np.all(np.isinf(result["snrs"])))
        self.assertTrue(np.isfinite(result["PCI"]))


class TestPCIstEpochAnalysis(unittest.TestCase):
    def setUp(self):
        self.epochs = make_epochs()

    def test_exact_event_zero_good_eeg_units_and_unchanged_source(self):
        original = self.epochs.get_data().copy()
        result = calculate_pcist(self.epochs, event_code=0, min_snr=0)
        self.assertEqual(result["n_epochs"], 2)
        self.assertEqual(result["ch_names"], ["Cz", "Pz"])
        self.assertIn("event_id=0", result["event_label"])
        expected = pci_st.calc_PCIst(
            original[:2, :2].mean(axis=0) * 1e6, self.epochs.times * 1000,
            full_return=True, **result["params"],
        )
        for key in ("PCI", "signal_evk", "signal_svd", "D_base", "NST_diff"):
            np.testing.assert_allclose(result[key], expected[key], err_msg=key)
        np.testing.assert_array_equal(self.epochs.get_data(), original)

    def test_matrix_times_account_for_resampling_and_embedding(self):
        result = calculate_pcist(self.epochs, min_snr=0, embed=True, L=3,
                                 tau=2, resample=125)
        effective = result["times"][4:]
        for kind, key in (("base", "baseline"), ("resp", "response")):
            window = result["params"][f"{key}_window"]
            expected = effective[(effective >= window[0]) & (effective < window[1])]
            np.testing.assert_array_equal(result[f"{key}_times"], expected)
            self.assertEqual(result[f"D_{kind}"].shape[1:], (len(expected), len(expected)))

    def test_named_event_is_exact_and_unknown_event_is_reported(self):
        named = calculate_pcist(self.epochs, event_code="stim/A")
        coded = calculate_pcist(self.epochs, event_code=0)
        self.assertEqual(named["PCI"], coded["PCI"])
        self.assertIn("event_id=0", named["event_label"])
        with self.assertRaisesRegex(ValueError, "Unknown PCIst event"):
            calculate_pcist(self.epochs, event_code="stim")

    def test_invalid_channel_selections(self):
        for picks in ([], ["EOG"], ["Fz"], ["missing"], ["Cz", "Cz"], [0]):
            with self.subTest(picks=picks), self.assertRaises(ValueError):
                calculate_pcist(self.epochs, picks=picks)

    def test_invalid_parameters_and_empty_effective_windows(self):
        invalid = [
            {"baseline_window": (-600, -50)}, {"response_window": (0, 600)},
            {"baseline_window": (0, -50)}, {"baseline_window": (-400, 50)},
            {"response_window": (1, 2)}, {"k": 1}, {"min_snr": -1},
            {"max_var": 0}, {"max_var": 101}, {"max_var": float("nan")},
            {"n_steps": 1}, {"n_steps": 2.5}, {"n_components": 3},
            {"max_thr_p": 0}, {"embed": "False"}, {"L": 0}, {"tau": 0},
            {"resample": 0}, {"resample": 1}, {"resample": 5},
            {"embed": True, "L": 100, "tau": 3}, {"unknown": 1},
        ]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                calculate_pcist(self.epochs, **parameters)

    def test_nonfinite_samples_are_reported_instead_of_pci_zero(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "NaN or infinite"):
                self.epochs._data[0, 0, 0] = value
                calculate_pcist(self.epochs)

    def test_no_accepted_components_retains_valid_zero_result(self):
        result = calculate_pcist(self.epochs, min_snr=1e12)
        self.assertEqual(result["PCI"], 0)
        self.assertEqual(result["n_dims"], 0)
        self.assertEqual(len(result["dNST"]), 0)


if __name__ == "__main__":
    unittest.main()
