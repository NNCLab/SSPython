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
