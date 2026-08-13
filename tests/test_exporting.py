import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

from core.exporting import (
    evoked_filename,
    export_evoked_files,
    export_preprocessed_files,
)


class TestExporting(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.source_directory = self.root / "source"
        self.source_directory.mkdir()
        self.export_directory = self.root / "export"

        info = mne.create_info(["Cz", "Pz"], sfreq=100.0, ch_types="eeg")
        data = np.arange(3 * 2 * 20, dtype=float).reshape(3, 2, 20) * 1e-6
        self.epochs = mne.EpochsArray(data, info, tmin=-0.1, verbose=False)
        self.source_path = (
            self.source_directory / "sub-01_task-tms_desc-preprocessed_epo.fif"
        )
        self.epochs.save(self.source_path, overwrite=True, verbose="error")

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_export_preprocessed_files_copies_selected_file(self):
        exported = export_preprocessed_files([self.source_path], self.export_directory)

        self.assertEqual(exported, [self.export_directory / self.source_path.name])
        self.assertEqual(exported[0].read_bytes(), self.source_path.read_bytes())

    def test_export_evoked_files_saves_epoch_average(self):
        exported = export_evoked_files([self.source_path], self.export_directory)

        expected_path = self.export_directory / "sub-01_task-tms_desc-preprocessed_ave.fif"
        self.assertEqual(exported, [expected_path])
        saved_evoked = mne.read_evokeds(expected_path, condition=0, verbose="error")
        np.testing.assert_allclose(saved_evoked.data, self.epochs.average().data)
        self.assertEqual(saved_evoked.nave, len(self.epochs))

    def test_existing_export_requires_explicit_overwrite(self):
        export_preprocessed_files([self.source_path], self.export_directory)

        with self.assertRaises(FileExistsError):
            export_preprocessed_files([self.source_path], self.export_directory)

        exported = export_preprocessed_files(
            [self.source_path], self.export_directory, overwrite=True
        )
        self.assertEqual(len(exported), 1)

    def test_evoked_filename_uses_mne_ave_suffix(self):
        self.assertEqual(
            evoked_filename(Path("example_desc-preprocessed_epo.fif")),
            "example_desc-preprocessed_ave.fif",
        )
