import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QMessageBox

from core.pipelines import build_analysis_paths
from core.processing import Preprocessor
from ui.widgets.run_ica_widget import RunICADialog


class _ImmediateWorker:
    def __init__(self, operation, **_kwargs):
        self.operation = operation
        self._error = None

    def exec_with_dialog(self, *_args, **_kwargs):
        return self.operation()


class TestRunICAConfirmation(unittest.TestCase):
    def make_dialog(self, root, mode, *, saved_results=True, existing_ica=False):
        stage_id = f"{mode}_ica"
        processor = Preprocessor.__new__(Preprocessor)
        processor.processing_order = [
            "raw", "filtered_raw", "continuous_ica", "epochs", "epochs_ica", "preprocessed"
        ]
        processor.output_dir = root
        processor.paths = {
            stage: root / f"subject_{stage}.fif" for stage in processor.processing_order
        }
        processor.has = lambda stage: processor.paths[stage].is_file()
        downstream = processor.processing_order[processor.processing_order.index(stage_id) + 1:]
        if saved_results:
            for stage in downstream:
                processor.paths[stage].write_bytes(b"saved result")
            for path in build_analysis_paths(
                processor.paths["preprocessed"], root, create_dirs=False
            ).values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"saved analysis")
        if existing_ica:
            processor.paths[stage_id].write_bytes(b"saved ICA")

        def run(**_kwargs):
            processor._clear_downstream_files(stage_id, verbose=False)
            processor.paths[stage_id].write_bytes(b"new ICA")

        operation = Mock(side_effect=run)
        setattr(processor, f"run_{mode}_ica", operation)
        dialog = SimpleNamespace(
            preprocessor=processor,
            mode=mode,
            settings_widget=SimpleNamespace(get_params=Mock(return_value={}), save_settings=Mock()),
            parent=lambda: None,
            accept=Mock(),
        )
        return dialog, operation

    def test_skipped_ica_requires_confirmation_and_cancel_preserves_every_file(self):
        for mode in ("continuous", "epochs"):
            for reply in (QMessageBox.StandardButton.No, QMessageBox.StandardButton.NoButton):
                with self.subTest(mode=mode, reply=reply), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    dialog, operation = self.make_dialog(root, mode)
                    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
                    self.assertFalse(dialog.preprocessor.has(f"{mode}_ica"))
                    with patch(
                        "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                        return_value=reply,
                    ) as question, patch("ui.widgets.run_ica_widget.Worker") as worker:
                        RunICADialog.run_ica(dialog)

                    question.assert_called_once()
                    for path in before:
                        self.assertIn(str(path), question.call_args.args[2])
                    self.assertEqual(question.call_args.args[4], QMessageBox.StandardButton.No)
                    worker.assert_not_called()
                    operation.assert_not_called()
                    dialog.settings_widget.save_settings.assert_not_called()
                    dialog.accept.assert_not_called()
                    self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_confirmed_run_deletes_downstream_files_only_after_approval(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dialog, operation = self.make_dialog(root, mode)
                saved_paths = [path for _, path in dialog.preprocessor.existing_downstream_files(f"{mode}_ica")]

                def approve(*_args):
                    operation.assert_not_called()
                    self.assertTrue(all(path.is_file() for path in saved_paths))
                    return QMessageBox.StandardButton.Yes

                with patch(
                    "ui.widgets.tools.processing_confirmation.QMessageBox.question", side_effect=approve
                ) as question, patch("ui.widgets.run_ica_widget.Worker", _ImmediateWorker), patch(
                    "ui.widgets.run_ica_widget.QMessageBox.information"
                ):
                    RunICADialog.run_ica(dialog)

                question.assert_called_once()
                operation.assert_called_once_with()
                self.assertTrue(all(not path.exists() for path in saved_paths))
                self.assertEqual(dialog.preprocessor.paths[f"{mode}_ica"].read_bytes(), b"new ICA")
                dialog.settings_widget.save_settings.assert_not_called()
                dialog.accept.assert_called_once_with()

    def test_first_run_without_saved_outputs_needs_no_deletion_prompt(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                dialog, operation = self.make_dialog(Path(directory), mode, saved_results=False)
                with patch(
                    "ui.widgets.tools.processing_confirmation.QMessageBox.question"
                ) as question, patch("ui.widgets.run_ica_widget.Worker", _ImmediateWorker), patch(
                    "ui.widgets.run_ica_widget.QMessageBox.information"
                ):
                    RunICADialog.run_ica(dialog)

                question.assert_not_called()
                operation.assert_called_once_with()
                dialog.settings_widget.save_settings.assert_not_called()

    def test_existing_ica_still_requires_overwrite_confirmation_without_downstream_files(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog, operation = self.make_dialog(
                Path(directory), "continuous", saved_results=False, existing_ica=True
            )
            with patch(
                "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question, patch("ui.widgets.run_ica_widget.Worker") as worker:
                RunICADialog.run_ica(dialog)

            self.assertIn(str(dialog.preprocessor.paths["continuous_ica"]), question.call_args.args[2])
            worker.assert_not_called()
            operation.assert_not_called()
            self.assertEqual(dialog.preprocessor.paths["continuous_ica"].read_bytes(), b"saved ICA")


if __name__ == "__main__":
    unittest.main()
