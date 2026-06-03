import unittest
from pathlib import Path
from core.processing import Preprocessor
import mne
import os
import numpy as np
import shutil

import shutil

class TestCore(unittest.TestCase):

    def setUp(self):
        # Create a dummy raw file for testing
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        self.raw_filepath = self.test_dir / "test_raw.fif"
        
        # Create a simple MNE Raw object
        ch_names = ['EEG 001', 'EEG 002', 'EEG 003']
        sfreq = 2048
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        data = np.random.randn(3, 4096)
        raw = mne.io.RawArray(data, info)
        raw.save(self.raw_filepath, overwrite=True)

    def tearDown(self):
        # Clean up the dummy files and directories
        shutil.rmtree(self.test_dir)

    def test_preprocessor_initialization(self):
        """Test if the Preprocessor class can be initialized correctly."""
        output_dir = self.test_dir / "derivatives"
        preprocessor = Preprocessor(self.raw_filepath, output_dir)
        self.assertIsInstance(preprocessor, Preprocessor)
        self.assertEqual(preprocessor.label, "test")

    def test_artifact_removal(self):
        """Test the artifact removal method."""
        output_dir = self.test_dir / "derivatives"
        preprocessor = Preprocessor(self.raw_filepath, output_dir)
        
        # Add a dummy annotation
        sfreq = 2048
        annot = mne.Annotations(onset=[512/sfreq], duration=[0.01], description=['TMS'])
        preprocessor.raw.set_annotations(annot)
        
        preprocessor.run_artifact_removal()
        self.assertTrue(preprocessor.has("filtered_raw"))

    def test_raw_review_saves_to_filtered_derivative_without_touching_source(self):
        """Test raw inspection edits are persisted only to the derivative raw."""
        output_dir = self.test_dir / "derivatives"
        preprocessor = Preprocessor(self.raw_filepath, output_dir)

        reviewed_raw = preprocessor.raw.copy()
        reviewed_raw.info["bads"] = ["EEG 002"]
        reviewed_raw.set_annotations(
            mne.Annotations(
                onset=[0.5],
                duration=[0.25],
                description=["BAD_manual"],
            )
        )
        preprocessor.paths["epochs"].touch()

        preprocessor.update_raw_review(reviewed_raw)

        self.assertTrue(preprocessor.has("filtered_raw"))
        self.assertFalse(preprocessor.paths["epochs"].exists())

        filtered = mne.io.read_raw_fif(
            preprocessor.paths["filtered_raw"],
            preload=False,
            verbose="error",
        )
        self.assertEqual(filtered.info["bads"], ["EEG 002"])
        self.assertEqual(len(filtered.annotations), 1)
        self.assertEqual(filtered.annotations.description[0], "BAD_manual")
        if hasattr(filtered, "close"):
            filtered.close()

        source = mne.io.read_raw_fif(self.raw_filepath, preload=False, verbose="error")
        self.assertEqual(source.info["bads"], [])
        self.assertEqual(len(source.annotations), 0)
        if hasattr(source, "close"):
            source.close()

    def test_raw_review_can_update_existing_filtered_derivative(self):
        """Test raw inspection edits can overwrite an existing filtered raw."""
        output_dir = self.test_dir / "derivatives"
        preprocessor = Preprocessor(self.raw_filepath, output_dir)

        first_review = preprocessor.raw.copy()
        first_review.info["bads"] = ["EEG 001"]
        preprocessor.update_raw_review(first_review)

        reloaded_preprocessor = Preprocessor(self.raw_filepath, output_dir)
        second_review = reloaded_preprocessor.filtered_raw.copy()
        second_review.info["bads"] = ["EEG 003"]
        second_review.set_annotations(
            mne.Annotations(
                onset=[1.0],
                duration=[0.1],
                description=["BAD_review"],
            )
        )

        reloaded_preprocessor.update_raw_review(second_review)

        updated = mne.io.read_raw_fif(
            reloaded_preprocessor.paths["filtered_raw"],
            preload=False,
            verbose="error",
        )
        self.assertEqual(updated.info["bads"], ["EEG 003"])
        self.assertEqual(len(updated.annotations), 1)
        self.assertEqual(updated.annotations.description[0], "BAD_review")
        if hasattr(updated, "close"):
            updated.close()

    def test_epoching(self):
        """Test the epoching method."""
        output_dir = self.test_dir / "derivatives"
        preprocessor = Preprocessor(self.raw_filepath, output_dir)
        
        # Add a dummy annotation
        annot = mne.Annotations(onset=[1], duration=[0.01], description=['TMS'])
        preprocessor.raw.set_annotations(annot)
        
        preprocessor.run_epoching(tlim=(-0.2, 0.2))
        self.assertTrue(preprocessor.has("epochs"))

if __name__ == '__main__':
    unittest.main()
