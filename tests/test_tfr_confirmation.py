import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QWidget

from ui.pages.erp_analysis_page import ErpAnalysisPage


class _ImmediateWorker:
    def __init__(self, operation, **_kwargs):
        self.operation = operation
        self._error = None

    def exec_with_dialog(self, *_args, **_kwargs):
        return self.operation()


class _AnalysisPageHarness(QWidget):
    run_time_frequency = ErpAnalysisPage.run_time_frequency

    def update_ui_state(self):
        pass

    def open_time_frequency_widget(self):
        pass


class TestTFRDeletionConfirmation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.paths = {
            key: root / f"subject_{key}.h5"
            for key in ("tfr", "itc", "tfr_statistics", "tfr_mask", "itc_mask")
        }
        self.analysis = SimpleNamespace(
            epochs=object(), paths=self.paths,
            derivatives={"tfr": "saved power", "itc": "saved coherence"},
        )
        self.page = _AnalysisPageHarness()
        self.addCleanup(self.page.close)
        self.page.tep = self.analysis
        self.dialog = Mock()
        self.dialog.exec.return_value = QDialog.DialogCode.Accepted
        self.dialog.get_compute_params.return_value = {"return_itc": False}
        self.dialog.get_statistics_params.return_value = None
        self.dialog.event_selector.currentData.return_value = 2
        dialog_patch = patch(
            "ui.pages.erp_analysis_page.ComputeTFRSettingsDialog", return_value=self.dialog
        )
        dialog_patch.start()
        self.addCleanup(dialog_patch.stop)

    def write_existing(self, keys):
        for key in keys:
            self.paths[key].write_bytes(f"saved {key}".encode())

    def test_declining_recompute_preserves_all_files_and_does_not_start_worker(self):
        self.write_existing(self.paths)
        before = {key: path.read_bytes() for key, path in self.paths.items()}

        with (
            patch(
                "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question,
            patch("ui.pages.erp_analysis_page.Worker") as worker,
        ):
            self.page.run_time_frequency()

        question.assert_called_once()
        warning = question.call_args.args[2]
        for path in self.paths.values():
            self.assertIn(str(path), warning)
        self.assertEqual(question.call_args.args[4], QMessageBox.StandardButton.No)
        worker.assert_not_called()
        self.assertEqual(before, {key: path.read_bytes() for key, path in self.paths.items()})
        self.assertEqual(self.analysis.derivatives, {"tfr": "saved power", "itc": "saved coherence"})

    def test_old_masks_require_confirmation_even_when_power_file_is_missing(self):
        self.write_existing(("tfr_mask",))

        with (
            patch(
                "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question,
            patch("ui.pages.erp_analysis_page.Worker") as worker,
        ):
            self.page.run_time_frequency()

        question.assert_called_once()
        self.assertIn(str(self.paths["tfr_mask"]), question.call_args.args[2])
        worker.assert_not_called()
        self.assertEqual(self.paths["tfr_mask"].read_bytes(), b"saved tfr_mask")

    def test_accepted_recompute_deletes_stale_outputs_only_after_confirmation(self):
        self.write_existing(self.paths)
        result = object()

        def compute(*_args, **_kwargs):
            question.assert_called_once()
            return result

        def save_power(path, _result, **_kwargs):
            Path(path).write_bytes(b"new power")

        with (
            patch(
                "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ) as question,
            patch("ui.pages.erp_analysis_page.Worker", _ImmediateWorker),
            patch("ui.pages.erp_analysis_page.compute_event_tfrs", side_effect=compute),
            patch("ui.pages.erp_analysis_page.mne.time_frequency.write_tfrs", side_effect=save_power),
            patch.object(self.page, "update_ui_state"),
            patch.object(self.page, "open_time_frequency_widget"),
        ):
            self.page.run_time_frequency()

        self.assertEqual(self.paths["tfr"].read_bytes(), b"new power")
        for key in ("itc", "tfr_statistics", "tfr_mask", "itc_mask"):
            self.assertFalse(self.paths[key].exists())
        self.assertIs(self.analysis.derivatives["tfr"], result)
        self.assertIsNone(self.analysis.derivatives["itc"])

    def test_requested_statistics_and_itc_are_listed_as_replacements(self):
        self.write_existing(self.paths)
        self.dialog.get_compute_params.return_value = {"return_itc": True}
        self.dialog.get_statistics_params.return_value = {"n_permutations": 10}

        with (
            patch("ui.pages.erp_analysis_page.confirm_file_changes", return_value=False) as confirm,
            patch("ui.pages.erp_analysis_page.Worker") as worker,
        ):
            self.page.run_time_frequency()

        self.assertEqual(
            set(confirm.call_args.kwargs["replaced_paths"]),
            {self.paths[key] for key in ("tfr", "itc", "tfr_statistics")},
        )
        self.assertEqual(
            set(confirm.call_args.kwargs["deleted_paths"]),
            {self.paths[key] for key in ("tfr_mask", "itc_mask")},
        )
        worker.assert_not_called()

    def test_first_run_with_no_existing_outputs_needs_no_deletion_prompt(self):
        result = object()

        def save_power(path, _result, **_kwargs):
            Path(path).write_bytes(b"new power")

        with (
            patch("ui.widgets.tools.processing_confirmation.QMessageBox.question") as question,
            patch("ui.pages.erp_analysis_page.Worker", _ImmediateWorker),
            patch("ui.pages.erp_analysis_page.compute_event_tfrs", return_value=result),
            patch("ui.pages.erp_analysis_page.mne.time_frequency.write_tfrs", side_effect=save_power),
            patch.object(self.page, "update_ui_state"),
            patch.object(self.page, "open_time_frequency_widget"),
        ):
            self.page.run_time_frequency()

        question.assert_not_called()
        self.assertEqual(self.paths["tfr"].read_bytes(), b"new power")


if __name__ == "__main__":
    unittest.main()
