import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ui.widgets.tep_analysis.natural_frequency import calculate_nf_data


class _FakeEpochs:
    def __init__(self, tfr_data, times, ch_names):
        self._tfr_data = tfr_data
        self.times = times
        self.ch_names = ch_names

    def copy(self):
        return self

    def pick(self, roi, verbose=False):
        del verbose
        assert roi == self.ch_names
        return self

    def compute_tfr(self, **kwargs):
        del kwargs
        return SimpleNamespace(data=self._tfr_data, ch_names=self.ch_names)

    def average(self):
        return SimpleNamespace(data=self._tfr_data.mean(axis=(0, 2)))


class TestNaturalFrequency(unittest.TestCase):
    def test_unmasked_fallback_aggregates_only_the_response_window(self):
        times = np.array([-0.1, 0.05, 0.2])
        freqs = np.array([10.0, 20.0])

        # Within the response window, 10 Hz has the most power. Across the full
        # epoch, the late 20 Hz power would incorrectly make 20 Hz the winner.
        channel_power = np.array(
            [
                [0.0, 10.0, 0.0],
                [0.0, 1.0, 100.0],
            ]
        )
        tfr_data = np.tile(channel_power, (1, 2, 1, 1))
        epochs = _FakeEpochs(tfr_data, times, ["C3", "C4"])

        with patch(
            "ui.widgets.tep_analysis.natural_frequency.TFstat",
            return_value=np.zeros((len(freqs), len(times))),
        ):
            result = calculate_nf_data(
                epochs=epochs,
                roi=epochs.ch_names,
                freqs=freqs,
                n_cycles=3.5,
                baseline=[-0.1, -0.1],
                response=[0.0, 0.1],
                n_permutations=10,
            )

        np.testing.assert_allclose(result["agg_freq"], [10.0, 1.0])
        self.assertEqual(result["nat_freq"], 10.0)


if __name__ == "__main__":
    unittest.main()
