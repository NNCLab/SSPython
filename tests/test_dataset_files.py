import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QWidget

from core.pipelines import DatasetRecord, get_pipeline
from ui.widgets.dataset_files import DatasetFileController
from ui.widgets.tools.object_info_widget import ObjectDetailsDialog


class ImmediateWorker:
    def __init__(self, task, **kwargs):
        self.task = task
        self._result = None

    def exec_with_dialog(self, *args):
        self._result = self.task()
        return self._result

    def deleteLater(self):
        pass


class TestDatasetFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.parent = QWidget()
        self.addCleanup(self.parent.deleteLater)
        self.controller = DatasetFileController(self.parent)
        self.raw_path = self.root / "subject.session.v2_raw.fif"
        self.epochs_path = self.root / "subject.session.v2_desc-epoched_epo.fif"
        info = mne.create_info(["Cz", "Pz"], 100, "eeg")
        mne.io.RawArray(np.zeros((2, 100)), info).save(self.raw_path, overwrite=True)
        mne.EpochsArray(np.zeros((2, 2, 51)), info, tmin=-0.1).save(self.epochs_path, overwrite=True)
        self.dataset = DatasetRecord(self.raw_path, get_pipeline(None), self.root, {
            "raw": self.raw_path, "epochs": self.epochs_path,
        })
        self.controller.set_dataset(self.dataset)
        worker = patch("ui.widgets.dataset_files.Worker", ImmediateWorker)
        worker.start()
        self.addCleanup(worker.stop)

    def test_information_identifies_the_selected_file_and_releases_its_reader(self):
        dialogs = []

        def inspect(dialog):
            dialogs.append(dialog)
            self.assertEqual(dialog.windowTitle(), self.epochs_path.stem)
            self.assertIn(str(self.epochs_path), [label.text() for label in dialog.findChildren(QLabel)])
            self.assertTrue(hasattr(dialog, "delete_button"))
            dialog.open_location_button.click()
            return QDialog.DialogCode.Accepted

        with patch.object(ObjectDetailsDialog, "exec", new=inspect), \
                patch.object(self.controller, "_open_file_location") as open_location:
            self.controller.inspect_stage("epochs")
        open_location.assert_called_once_with(self.epochs_path)
        self.assertIsNone(dialogs[0].summary_widget.mne_object)
        # A lingering lazy Epochs reader would prevent this on Windows.
        self.epochs_path.unlink()
        self.assertTrue(self.raw_path.exists())

    def test_delete_request_releases_inspection_before_confirmation_and_respects_cancel(self):
        dialogs = []
        confirmations = []

        def request_delete(dialog):
            dialogs.append(dialog)
            dialog.delete_button.click()
            return QDialog.DialogCode.Accepted

        def cancel_confirmation(message_box):
            confirmations.append(message_box)
            self.assertIsNone(dialogs[0].summary_widget.mne_object)
            self.assertIn(self.epochs_path.name, message_box.informativeText())
            self.assertEqual(message_box.defaultButton().text(), "Cancel")
            return QDialog.DialogCode.Rejected

        with patch.object(ObjectDetailsDialog, "exec", new=request_delete), \
                patch("ui.widgets.dataset_files.QMessageBox.exec", new=cancel_confirmation):
            self.controller.inspect_stage("epochs")
        self.assertEqual(len(confirmations), 1)
        self.assertTrue(self.epochs_path.exists())
        self.assertTrue(self.raw_path.exists())

    def test_source_and_analysis_only_input_offer_information_without_deletion(self):
        analysis_only = DatasetRecord(None, get_pipeline(None), self.root, {"preprocessed": self.epochs_path})
        dialogs = []

        def inspect(dialog):
            dialogs.append(dialog)
            self.assertTrue(hasattr(dialog, "open_location_button"))
            self.assertFalse(hasattr(dialog, "delete_button"))
            return QDialog.DialogCode.Accepted

        with patch.object(ObjectDetailsDialog, "exec", new=inspect):
            self.controller.inspect_stage("raw")
            self.controller.set_dataset(analysis_only)
            self.controller.inspect_stage("preprocessed")
        self.assertEqual(len(dialogs), 2)
        self.assertTrue(self.raw_path.exists())
        self.assertTrue(self.epochs_path.exists())

    def test_pending_skipped_and_failed_files_do_not_open_information(self):
        with patch("ui.widgets.dataset_files.load_stage_object", side_effect=OSError("Unreadable file")) as load, \
                patch("ui.widgets.dataset_files.QMessageBox.critical") as error, \
                patch.object(ObjectDetailsDialog, "exec") as dialog_exec:
            self.controller.inspect_stage("preprocessed")  # Pending
            self.controller.inspect_stage("continuous_ica")  # Skipped after saved epochs
            load.assert_not_called()
            self.controller.inspect_stage("epochs")
            load.assert_called_once_with("epochs", self.epochs_path, "epochs")
            error.assert_called_once()
            dialog_exec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
