import json
import shutil
import unittest
from pathlib import Path

import mne
import numpy as np
from PySide6.QtWidgets import QApplication
from scipy.io import savemat

from core.channel_info import load_channel_info
from core.conversion import (
    ConversionEntry,
    convert_files,
    discover_convertible_files,
    load_raw_from_mat,
    load_raw_from_source,
    normalize_output_filename,
)
from core.merge import MergeEntry, merge_fif_files, validate_merge_entries
from ui.widgets.tools.conversion_tool import ConversionToolDialog


class TestConversion(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_conversion_data")
        self.test_dir.mkdir(exist_ok=True)

        raw_info = mne.create_info(["Fz", "Cz"], sfreq=200, ch_types="eeg")
        self.raw_a_path = self.test_dir / "recording_a.fif"
        self.raw_b_path = self.test_dir / "recording_b.fif"
        self.existing_raw_path = self.test_dir / "already_raw.fif"

        raw_a = mne.io.RawArray(np.random.randn(2, 100), raw_info, verbose=False)
        raw_a.set_annotations(
            mne.Annotations(
                onset=[0.1],
                duration=[0.0],
                description=["TMS"],
            )
        )
        raw_b = mne.io.RawArray(np.random.randn(2, 120), raw_info, verbose=False)
        raw_b.set_annotations(
            mne.Annotations(
                onset=[0.15],
                duration=[0.0],
                description=["TMS"],
            )
        )
        existing_raw = mne.io.RawArray(
            np.random.randn(2, 80), raw_info, verbose=False
        )
        raw_a.save(self.raw_a_path, overwrite=True, verbose="error")
        raw_b.save(self.raw_b_path, overwrite=True, verbose="error")
        existing_raw.save(self.existing_raw_path, overwrite=True, verbose="error")

        generic_info = mne.create_info(["EEG1", "EEG2"], sfreq=200, ch_types="eeg")
        self.generic_raw_path = self.test_dir / "generic_recording.fif"
        generic_raw = mne.io.RawArray(
            np.random.randn(2, 90), generic_info, verbose=False
        )
        generic_raw.save(self.generic_raw_path, overwrite=True, verbose="error")

        self.mat_path = self.test_dir / "gtec_export.mat"
        self.channel_info_path = self.test_dir / "channels.json"
        self.generic_channel_info_path = self.test_dir / "generic_channels.json"
        self.list_channel_info_path = self.test_dir / "list_channels.json"

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

        with self.generic_channel_info_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "ch_names": ["Fz", "Cz"],
                    "ch_types": ["eeg", "eeg"],
                },
                stream,
            )

        with self.list_channel_info_path.open("w", encoding="utf-8") as stream:
            json.dump(
                [
                    {"name": "Fz", "type": "eeg"},
                    {"name": "Cz", "type": "eeg"},
                ],
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
        self.assertIn("generic_recording.fif", supported_names)
        self.assertNotIn("already_raw.fif", supported_names)

    def test_load_channel_info_accepts_list_format(self):
        channel_info = load_channel_info(self.list_channel_info_path)
        self.assertEqual(channel_info.ch_names, ["Fz", "Cz"])
        self.assertEqual(channel_info.ch_types, ["eeg", "eeg"])

    def test_load_raw_from_mat_builds_raw_and_annotations(self):
        raw = load_raw_from_mat(
            self.mat_path,
            channel_info=self.channel_info_path,
        )

        self.assertEqual(raw.ch_names, ["Fz", "Cz", "STI 014"])
        self.assertAlmostEqual(raw.info["sfreq"], 200.0)
        self.assertEqual(len(raw.annotations), 1)
        self.assertEqual(raw.annotations.description[0], "STIM 1")

    def test_load_raw_from_source_applies_optional_channel_info_to_other_formats(self):
        raw = load_raw_from_source(
            self.generic_raw_path,
            channel_info=self.generic_channel_info_path,
            preload=False,
        )

        self.assertEqual(raw.ch_names, ["Fz", "Cz"])
        self.assertEqual(raw.get_channel_types(), ["eeg", "eeg"])
        raw.close()

    def test_convert_files_converts_selected_outputs(self):
        output_a = self.test_dir / "recording_a_raw.fif"
        output_b = self.test_dir / "recording_b_raw.fif"

        result = convert_files(
            [
                ConversionEntry(self.raw_a_path, output_a),
                ConversionEntry(self.raw_b_path, output_b),
            ],
            builtin_montage="standard_1020",
        )

        self.assertEqual(result["converted"], [output_a, output_b])
        self.assertTrue(output_a.exists())
        self.assertTrue(output_b.exists())

    def test_convert_files_accepts_optional_channel_info_for_non_mat_sources(self):
        output_path = self.test_dir / "generic_recording_raw.fif"

        result = convert_files(
            [ConversionEntry(self.generic_raw_path, output_path)],
            builtin_montage="standard_1020",
            channel_info=self.generic_channel_info_path,
        )

        self.assertEqual(result["converted"], [output_path])
        converted_raw = mne.io.read_raw_fif(output_path, preload=False, verbose="error")
        self.assertEqual(converted_raw.ch_names, ["Fz", "Cz"])
        converted_raw.close()

    def test_validate_merge_entries_rejects_annotation_name_collisions(self):
        with self.assertRaisesRegex(ValueError, "Rename 'TMS' before merging"):
            validate_merge_entries(
                [
                    MergeEntry(self.raw_a_path),
                    MergeEntry(self.raw_b_path),
                ]
            )

    def test_merge_fif_files_renames_annotations_before_merge(self):
        merged_output = self.test_dir / "merged_annotations.fif"

        result = merge_fif_files(
            [
                MergeEntry(self.raw_a_path, {"TMS": "TMS A"}),
                MergeEntry(self.raw_b_path, {"TMS": "TMS B"}),
            ],
            merged_output,
        )

        merged_raw = mne.io.read_raw_fif(result, preload=False, verbose="error")
        self.assertIn("TMS A", merged_raw.annotations.description.tolist())
        self.assertIn("TMS B", merged_raw.annotations.description.tolist())
        merged_raw.close()

    def test_conversion_dialog_requires_mat_info_for_selected_mat_files(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        dialog = ConversionToolDialog()
        dialog.conversion_widget.source_folder = self.test_dir
        dialog.conversion_widget.populate_sources([self.mat_path])
        dialog.conversion_widget.montage_combo.setCurrentText("standard_1020")

        with self.assertRaisesRegex(ValueError, "channel info JSON"):
            dialog.conversion_widget._collect_entries()

        dialog.close()


if __name__ == "__main__":
    unittest.main()
