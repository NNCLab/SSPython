import json
import shutil
import unittest
from pathlib import Path

import mne
import numpy as np
from PySide6.QtWidgets import QApplication
from scipy.io import savemat

from core.conversion import (
    ConversionEntry,
    convert_files,
    discover_convertible_files,
    load_raw_from_mat,
    normalize_output_filename,
)
from ui.widgets.tools.conversion_tool import ConversionToolDialog


class TestConversion(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_conversion_data")
        self.test_dir.mkdir(exist_ok=True)

        self.raw_info = mne.create_info(["Fz", "Cz"], sfreq=200, ch_types="eeg")
        self.raw_a_path = self.test_dir / "recording_a.fif"
        self.raw_b_path = self.test_dir / "recording_b.fif"
        self.existing_raw_path = self.test_dir / "already_raw.fif"

        raw_a = mne.io.RawArray(np.random.randn(2, 100), self.raw_info, verbose=False)
        raw_b = mne.io.RawArray(np.random.randn(2, 120), self.raw_info, verbose=False)
        existing_raw = mne.io.RawArray(
            np.random.randn(2, 80), self.raw_info, verbose=False
        )
        raw_a.save(self.raw_a_path, overwrite=True, verbose="error")
        raw_b.save(self.raw_b_path, overwrite=True, verbose="error")
        existing_raw.save(self.existing_raw_path, overwrite=True, verbose="error")

        self.mat_path = self.test_dir / "gtec_export.mat"
        self.channel_info_path = self.test_dir / "channels.json"
        mat_data = np.zeros((3, 60), dtype=float)
        mat_data[0, :] = 20.0
        mat_data[1, :] = 10.0
        mat_data[2, 10] = 1.0
        savemat(self.mat_path, {"y": mat_data, "SR": np.array([[200.0]])})
        with self.channel_info_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "ch_names": ["Fz", "Cz", "STI 014"],
                    "ch_types": ["eeg", "eeg", "stim"],
                },
                stream,
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_normalize_output_filename_enforces_raw_suffix(self):
        self.assertEqual(normalize_output_filename("subject01"), "subject01_raw.fif")
        self.assertEqual(
            normalize_output_filename("subject01_raw.fif"), "subject01_raw.fif"
        )
        self.assertEqual(
            normalize_output_filename("subject01.vhdr"), "subject01_raw.fif"
        )

    def test_discover_convertible_files_finds_supported_sources(self):
        supported = discover_convertible_files(self.test_dir)
        supported_names = {path.name for path in supported}
        self.assertIn("recording_a.fif", supported_names)
        self.assertIn("recording_b.fif", supported_names)
        self.assertIn("gtec_export.mat", supported_names)
        self.assertNotIn("already_raw.fif", supported_names)

    def test_load_raw_from_mat_builds_raw_and_annotations(self):
        raw = load_raw_from_mat(
            self.mat_path,
            channel_info_path=self.channel_info_path,
        )

        self.assertEqual(raw.ch_names, ["Fz", "Cz", "STI 014"])
        self.assertAlmostEqual(raw.info["sfreq"], 200.0)
        self.assertEqual(len(raw.annotations), 1)
        self.assertEqual(raw.annotations.description[0], "STIM 1")

    def test_convert_files_can_merge_selected_outputs(self):
        output_a = self.test_dir / "recording_a_raw.fif"
        output_b = self.test_dir / "recording_b_raw.fif"
        merged_output = self.test_dir / "merged_raw.fif"

        result = convert_files(
            [
                ConversionEntry(self.raw_a_path, output_a, merge=True),
                ConversionEntry(self.raw_b_path, output_b, merge=True),
            ],
            builtin_montage="standard_1020",
            merged_output_path=merged_output,
        )

        self.assertEqual(result["converted"], [output_a, output_b])
        self.assertEqual(result["merged"], merged_output)
        merged_raw = mne.io.read_raw_fif(merged_output, preload=False, verbose="error")
        self.assertEqual(merged_raw.n_times, 220)

    def test_conversion_dialog_requires_mat_info_for_selected_mat_files(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        dialog = ConversionToolDialog()
        dialog.source_folder = self.test_dir
        dialog.populate_sources([self.mat_path])
        dialog.montage_combo.setCurrentText("standard_1020")

        with self.assertRaisesRegex(ValueError, "MAT channel info JSON"):
            dialog._collect_entries()

        dialog.close()


if __name__ == "__main__":
    unittest.main()
