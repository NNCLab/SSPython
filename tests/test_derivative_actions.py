import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QToolButton

from core.pipelines import build_processing_paths, get_pipeline
from main import MainWindow
from ui.widgets.tools.object_info_widget import ObjectDetailsDialog


class TestDerivativeActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        raw_path = root / "sub-01_task-tms_raw.fif"
        info = mne.create_info(["C3", "C4"], 100, "eeg")
        raw = mne.io.RawArray(np.zeros((2, 1000)), info, verbose=False)
        raw.save(raw_path, verbose=False)
        pipeline = get_pipeline("standard")
        paths = build_processing_paths(raw_path, pipeline.derivative_root(root, "derivatives"))
        paths["filtered_raw"].parent.mkdir(parents=True, exist_ok=True)
        raw.save(paths["filtered_raw"], verbose=False)
        epochs = mne.EpochsArray(np.zeros((2, 2, 21)), info, tmin=-0.1, verbose=False)
        epochs.save(paths["preprocessed"], verbose=False)

        self.window = MainWindow()
        self.addCleanup(self.window.close)
        self.window.set_current_folder(str(root))
        self.window.show()
        self.window.navigate_to_page("Preprocessing")
        self.addCleanup(self.window._release_derivative_resources, tuple(paths.values()))

    def test_saved_stage_header_keeps_folder_and_downstream_deletion_accessible(self):
        page = self.window.page_lookup["Preprocessing"]
        stage = page.stage_sections["filtered_raw"]
        self.assertIsNone(page.findChild(QToolButton, "fileActionsButton"))
        folder = self.window.dataset_files.derivatives_folder()

        def inspect_details(dialog):
            self.assertTrue(dialog.open_location_button.isEnabled())
            self.assertTrue(dialog.delete_button.isEnabled())
            dialog.open_location_button.click()
            dialog.delete_button.click()
            return QDialog.DialogCode.Accepted

        with (
            patch.object(ObjectDetailsDialog, "exec", new=inspect_details),
            patch.object(self.window.dataset_files, "_open_file_location") as open_folder,
            patch.object(self.window.dataset_files, "confirm_and_delete_derivatives") as delete,
        ):
            self.app.processEvents()
            QTest.mouseClick(stage.title, Qt.MouseButton.LeftButton)
            open_folder.assert_called_once_with(folder)
            delete.assert_called_once_with("filtered_raw")

    def test_analysis_exposes_folder_and_file_information_actions(self):
        self.window.navigate_to_page("Analysis")
        page = self.window.page_lookup["Analysis"]
        self.assertTrue(page.open_derivatives_button.isEnabled())
        self.assertTrue(page.file_info_button.isEnabled())

        with patch.object(self.window.dataset_files, "open_derivatives_folder") as open_folder:
            page.open_derivatives_button.click()
            open_folder.assert_called_once_with()
        with patch.object(self.window.dataset_files, "inspect_stage") as inspect:
            page.file_info_button.click()
            inspect.assert_called_once_with("preprocessed")

        self.window.set_current_dataset(None)
        self.assertFalse(page.open_derivatives_button.isEnabled())
        self.assertFalse(page.file_info_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
