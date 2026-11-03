import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtWidgets import QMessageBox

from ui.pages.preprocessing_page import ProcessingPage


class TestReviewConfirmation(unittest.TestCase):
    def test_cancelled_epoch_review_names_downstream_files_and_preserves_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            epochs_path = root / "subject_desc-epoched_epo.fif"
            output = root / "subject_desc-preprocessed_epo.fif"
            epochs_path.write_bytes(b"saved epochs")
            output.write_bytes(b"saved output")
            processor = Mock()
            processor.paths = {"epochs": epochs_path}
            processor.has.return_value = True
            processor.existing_downstream_files.return_value = [("preprocessed", output)]
            page = SimpleNamespace(
                preprocessor=processor,
                viz_epochs=SimpleNamespace(info={"bads": ["C3"]}, ch_names=["C3"], selection=[0]),
                original_bads=[], original_selection=[0],
                isVisible=lambda: False,
                _refresh_after_step=Mock(),
            )
            with patch.object(
                QMessageBox, "question", return_value=QMessageBox.StandardButton.No
            ) as question, patch.object(QMessageBox, "critical") as error:
                ProcessingPage._after_epochs_inspected(page)

            question.assert_called_once()
            self.assertIn(str(output), question.call_args.args[2])
            self.assertIn(str(epochs_path), question.call_args.args[2])
            self.assertEqual(question.call_args.args[4], QMessageBox.StandardButton.No)
            processor.update_epochs.assert_not_called()
            error.assert_not_called()
            self.assertEqual(epochs_path.read_bytes(), b"saved epochs")
            self.assertEqual(output.read_bytes(), b"saved output")

    def test_cancelled_raw_review_discloses_checkpoint_deletion_or_replacement(self):
        for checkpoint_channels, section in (
            (["C3"], "These dependent files will be deleted:"),
            (["C3", "C4"], "These saved files will be replaced:"),
        ):
            with self.subTest(channels=checkpoint_channels), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "subject_desc-filterinput_raw.fif"
                filtered = root / "subject_desc-filtered_raw.fif"
                output = root / "subject_desc-preprocessed_epo.fif"
                mne.io.RawArray(
                    np.zeros((len(checkpoint_channels), 100)),
                    mne.create_info(checkpoint_channels, 100, "eeg"),
                    verbose=False,
                ).save(checkpoint, verbose=False)
                checkpoint_bytes = checkpoint.read_bytes()
                filtered.write_bytes(b"saved filtered recording")
                output.write_bytes(b"saved result")
                reviewed = mne.io.RawArray(
                    np.zeros((2, 100)), mne.create_info(["C3", "C4"], 100, "eeg"),
                    verbose=False,
                )
                processor = Mock()
                processor.paths = {"filtered_raw": filtered, "filter_input_raw": checkpoint}
                processor.has.return_value = True
                processor.existing_downstream_files.return_value = [("preprocessed", output)]
                page = SimpleNamespace(
                    preprocessor=processor,
                    _raw_review_pending=True, _raw_review_figure=Mock(),
                    viz_raw=reviewed, original_raw_bads=["C4"],
                    original_raw_annotations=reviewed.annotations.copy(),
                    isVisible=lambda: False, _refresh_after_step=Mock(),
                )
                with patch.object(
                    QMessageBox, "question", return_value=QMessageBox.StandardButton.No
                ) as question, patch.object(QMessageBox, "critical") as error:
                    ProcessingPage._after_raw_inspected(page)

                question.assert_called_once()
                warning = question.call_args.args[2]
                affected_section = warning.split(section)[1].split("\n\n")[0]
                self.assertIn(str(checkpoint), affected_section)
                self.assertIn(str(output), warning)
                self.assertEqual(question.call_args.args[4], QMessageBox.StandardButton.No)
                processor.update_raw_review.assert_not_called()
                error.assert_not_called()
                self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
                self.assertEqual(filtered.read_bytes(), b"saved filtered recording")
                self.assertEqual(output.read_bytes(), b"saved result")


if __name__ == "__main__":
    unittest.main()
