import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QWidget

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
        self.derivative_root = self.root / "derivatives" / "standard"
        folder = self.derivative_root / "sub-01" / "eeg"
        folder.mkdir(parents=True)
        self.epochs_path = folder / "subject.session.v2_desc-epoched_epo.fif"
        info = mne.create_info(["Cz", "Pz"], 100, "eeg")
        mne.io.RawArray(np.zeros((2, 100)), info, verbose=False).save(
            self.raw_path, overwrite=True, verbose=False
        )
        mne.EpochsArray(np.zeros((2, 2, 51)), info, tmin=-0.1, verbose=False).save(
            self.epochs_path, overwrite=True, verbose=False
        )
        self.dataset = DatasetRecord(self.raw_path, get_pipeline(None), self.derivative_root, {
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
            self.assertEqual(dialog.windowTitle(), self.dataset.display_file_name("epochs"))
            self.assertEqual(dialog.path_field.text(), str(self.epochs_path))
            self.assertTrue(dialog.path_field.isReadOnly())
            self.assertTrue(hasattr(dialog, "delete_button"))
            dialog.open_location_button.click()
            return QDialog.DialogCode.Accepted

        with patch.object(ObjectDetailsDialog, "exec", new=inspect), \
                patch.object(self.controller, "_open_file_location") as open_location:
            self.controller.inspect_stage("epochs")
        open_location.assert_called_once_with(self.epochs_path.parent)
        self.assertEqual(dialogs[0].summary_widget.metric_labels, [])
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
            self.assertEqual(dialogs[0].summary_widget.metric_labels, [])
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

    def test_confirmed_deletion_releases_then_removes_dependent_files(self):
        final_path = self.epochs_path.with_name("subject_desc-preprocessed_epo.fif")
        final_path.touch()
        self.dataset.paths["preprocessed"] = final_path
        changed = []
        self.controller.derivatives_changed.connect(lambda: changed.append(True))
        released = []

        def release(paths):
            self.assertTrue(all(path.is_file() for path in paths))
            released.extend(paths)

        self.controller.derivatives_about_to_be_deleted.connect(release)

        def confirm(message_box):
            for button in message_box.buttons():
                if message_box.buttonRole(button) == QMessageBox.ButtonRole.DestructiveRole:
                    button.click()
                    return QDialog.DialogCode.Accepted
            self.fail("No delete confirmation button")

        with patch("ui.widgets.dataset_files.QMessageBox.exec", new=confirm):
            self.controller.confirm_and_delete_derivatives("epochs")

        self.assertEqual(released, [self.epochs_path, final_path])
        self.assertEqual(changed, [True])
        self.assertFalse(self.epochs_path.exists())
        self.assertFalse(final_path.exists())
        self.assertTrue(self.raw_path.is_file())

    def test_open_derivatives_folder_uses_dataset_directory_and_not_raw_directory(self):
        self.dataset.paths["preprocessed"] = self.epochs_path.with_name(
            "subject_desc-preprocessed_epo.fif"
        )
        with patch("ui.widgets.dataset_files.QDesktopServices.openUrl", return_value=True) as opened:
            self.controller.open_derivatives_folder()

        opened.assert_called_once()
        url = opened.call_args.args[0]
        self.assertEqual(Path(url.toLocalFile()).resolve(), self.epochs_path.parent.resolve())
        self.assertNotEqual(Path(url.toLocalFile()).resolve(), self.raw_path.parent.resolve())

    def test_public_actions_do_not_delete_source_or_analysis_only_input(self):
        self.assertFalse(self.controller.can_delete_stage("raw"))
        self.controller.confirm_and_delete_derivatives("raw")
        self.controller.set_dataset(
            DatasetRecord(None, get_pipeline(None), self.derivative_root, {
                "preprocessed": self.epochs_path,
            })
        )
        self.assertFalse(self.controller.can_delete_stage("preprocessed"))
        self.controller.confirm_and_delete_derivatives("preprocessed")
        self.assertTrue(self.raw_path.exists())
        self.assertTrue(self.epochs_path.exists())

    def test_analysis_only_folder_uses_generated_outputs_instead_of_import_location(self):
        self.controller.set_dataset(
            DatasetRecord(None, get_pipeline(None), self.derivative_root, {
                "preprocessed": self.raw_path,
            })
        )
        self.assertEqual(self.controller.derivatives_folder(), self.derivative_root)

    def test_missing_derivative_folder_is_unavailable(self):
        self.controller.set_dataset(
            DatasetRecord(self.raw_path, get_pipeline(None), self.root / "missing", {
                "raw": self.raw_path,
            })
        )
        self.assertIsNone(self.controller.derivatives_folder())

        def inspect(dialog):
            self.assertFalse(dialog.open_location_button.isEnabled())
            return QDialog.DialogCode.Accepted

        with patch.object(ObjectDetailsDialog, "exec", new=inspect):
            self.controller.inspect_stage("raw")


if __name__ == "__main__":
    unittest.main()
