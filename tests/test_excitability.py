import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from ui.widgets.tep_analysis.excitability import AnalysisMode, ExcitabilityApp


class _FakeEpochs:
    def __init__(self):
        self.times = np.array([-0.002, 0.0, 0.002])
        self._data = np.full((2, 3), 1e-6)

    def average(self):
        return SimpleNamespace(data=self._data)

    def copy(self):
        return self

    def pick(self, channels):
        assert channels == ["C3", "C4"]
        return self


class TestResponseAmplitudeAUC(unittest.TestCase):
    def test_auc_uses_time_in_milliseconds_for_gmfp_and_lfp(self):
        for mode in (AnalysisMode.GMFP, AnalysisMode.LFP):
            with self.subTest(mode=mode):
                app = SimpleNamespace(
                    epochs=_FakeEpochs(),
                    span=np.array([-2.0, 2.0]),
                    mode=mode,
                    selected_channels=["C3", "C4"],
                    outcome_label=Mock(),
                    outcome_ax=Mock(),
                )

                ExcitabilityApp._plot_outcome_and_update_label(app)

                metric = "GMFP" if mode == AnalysisMode.GMFP else "LFP"
                app.outcome_label.setText.assert_called_once_with(
                    f"AUC ({metric}): 4.00 µV·ms"
                )


if __name__ == "__main__":
    unittest.main()
