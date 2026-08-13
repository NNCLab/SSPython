import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import mne
import numpy as np

from core.processing import Preprocessor


class TestInitialEpochsMetadata(unittest.TestCase):
    def test_epoching_logs_applied_continuous_ica_details(self):
        info = mne.create_info(["EEG 001", "EEG 002"], 100, "eeg")
        raw = mne.io.RawArray(np.zeros((2, 500)), info, verbose=False)
        raw.set_annotations(
            mne.Annotations(onset=[1.0], duration=[0.0], description=["TMS"])
        )

        ica = Mock()
        ica.method = "fastica"
        ica.n_components_ = np.int64(2)
        ica.exclude = [np.int32(1)]
        ica.apply.side_effect = lambda instance: instance

        with tempfile.TemporaryDirectory() as directory:
            preprocessor = Preprocessor.__new__(Preprocessor)
            preprocessor.processing_order = [
                "raw",
                "filtered_raw",
                "continuous_ica",
                "epochs",
                "epochs_ica",
                "preprocessed",
            ]
            preprocessor.paths = {
                "epochs": Path(directory) / "initial-epo.fif",
            }
            preprocessor.has = lambda stage: stage == "continuous_ica"
            preprocessor._raw = raw
            preprocessor._continuous_ica = ica

            preprocessor.run_epoching(tlim=(-0.1, 0.1), verbose=False)

            saved_epochs = mne.read_epochs(
                preprocessor.paths["epochs"],
                preload=True,
                verbose=False,
            )
            epoching_log = json.loads(saved_epochs.info["description"])[-1]["epoching"]

        ica.apply.assert_called_once_with(raw)
        self.assertTrue(epoching_log["continuous_ica"])
        self.assertEqual(
            epoching_log["ica"],
            {
                "applied": True,
                "source_stage": "continuous_ica",
                "method": "fastica",
                "n_components": 2,
                "excluded_components": [1],
                "excluded_count": 1,
            },
        )


class TestFinalEpochsMetadata(unittest.TestCase):
    @staticmethod
    def _preprocessor(*, ica=None):
        source_epochs = Mock()
        source_epochs.info = {
            "description": json.dumps([{"epoching": {"mode": "event"}}])
        }

        final_epochs = Mock()
        final_epochs.info = {"description": source_epochs.info["description"], "bads": []}
        final_epochs.times = np.array([-0.1, 0.0, 0.1])
        source_epochs.load_data.return_value.copy.return_value = final_epochs

        preprocessor = Preprocessor.__new__(Preprocessor)
        preprocessor._epochs = source_epochs
        preprocessor.paths = {"preprocessed": Path("final-epo.fif")}
        preprocessor.has = lambda stage: stage == "epochs" or (
            stage == "epochs_ica" and ica is not None
        )
        if ica is not None:
            preprocessor._epochs_ica = ica
        return preprocessor, final_epochs

    @staticmethod
    def _filter_log(epochs):
        descriptions = json.loads(epochs.info["description"])
        return descriptions[-1]["filter_epochs"]

    def test_filter_epochs_logs_applied_ica_details(self):
        ica = Mock()
        ica.method = "infomax"
        ica.n_components_ = np.int64(3)
        ica.exclude = [np.int32(0), np.int64(2)]
        preprocessor, final_epochs = self._preprocessor(ica=ica)

        preprocessor.filter_epochs(
            bandpass=(None, None),
            interpolate_bad_channels=False,
            reference=None,
            baseline=None,
            verbose=False,
        )

        ica.apply.assert_called_once_with(final_epochs, verbose=False)
        self.assertEqual(
            self._filter_log(final_epochs)["ica"],
            {
                "applied": True,
                "source_stage": "epochs_ica",
                "method": "infomax",
                "n_components": 3,
                "excluded_components": [0, 2],
                "excluded_count": 2,
            },
        )

    def test_filter_epochs_logs_when_ica_was_not_applied(self):
        preprocessor, final_epochs = self._preprocessor()

        preprocessor.filter_epochs(
            bandpass=(None, None),
            interpolate_bad_channels=False,
            reference=None,
            baseline=None,
            verbose=False,
        )

        self.assertEqual(
            self._filter_log(final_epochs)["ica"],
            {"applied": False, "source_stage": "epochs_ica"},
        )


if __name__ == "__main__":
    unittest.main()
