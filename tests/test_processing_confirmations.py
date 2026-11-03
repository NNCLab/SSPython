import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QMessageBox

from ui.widgets.preprocessing_widgets import (
    ApplyPreprocessingDialog,
    ArtifactRemovalDialog,
    EpochingDialog,
    FilterContinuousDialog,
    ReReferenceDialog,
)


class _ImmediateWorker:
    def __init__(self, operation, **_kwargs):
        self.operation = operation
        self._error = None

    def exec_with_dialog(self, *_args, **_kwargs):
        return self.operation()


class TestProcessingConfirmations(unittest.TestCase):
    CASES = (
        (ArtifactRemovalDialog, "filtered_raw", "run_artifact_removal", {}),
        (FilterContinuousDialog, "filtered_raw", "filter_continuous", {}),
        (EpochingDialog, "epochs", "run_epoching", {"mode": "fixed", "fixed_duration": 1.0}),
        (ReReferenceDialog, "epochs", "run_rereferencing", {"reference": "average"}),
        (ApplyPreprocessingDialog, "preprocessed", "filter_epochs", {}),
    )

    def dialog(self, params, downstream_files):
        preprocessor = Mock()
        preprocessor.has.return_value = False
        preprocessor.existing_downstream_files.return_value = downstream_files
        preprocessor.paths = {}
        preprocessor.epochs = SimpleNamespace(preload=True)
        return SimpleNamespace(
            preprocessor=preprocessor,
            settings_widget=SimpleNamespace(
                get_params=Mock(return_value=dict(params)), save_settings=Mock()
            ),
            parent=lambda: None,
            accept=Mock(),
            finished=SimpleNamespace(emit=Mock()),
        )

    def test_missing_stage_still_confirms_and_cancel_preserves_downstream_files(self):
        for dialog_class, stage_id, operation, params in self.CASES:
            with self.subTest(dialog=dialog_class.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "subject_desc-preprocessed_epo.fif"
                path.write_text("saved result", encoding="utf-8")
                dialog = self.dialog(params, [("preprocessed", path)])

                with patch(
                    "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.No,
                ) as confirm, patch("ui.widgets.preprocessing_widgets.Worker") as worker:
                    dialog_class.run_process(dialog)

                confirm.assert_called_once()
                self.assertIn(path.name, confirm.call_args.args[2])
                self.assertEqual(confirm.call_args.args[4], QMessageBox.StandardButton.No)
                dialog.preprocessor.existing_downstream_files.assert_called_once_with(stage_id)
                worker.assert_not_called()
                getattr(dialog.preprocessor, operation).assert_not_called()
                dialog.settings_widget.save_settings.assert_not_called()
                dialog.accept.assert_not_called()
                self.assertEqual(path.read_text(encoding="utf-8"), "saved result")

    def test_confirming_allows_processing_only_after_the_user_answers(self):
        for dialog_class, _stage_id, operation, params in self.CASES:
            with self.subTest(dialog=dialog_class.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "subject_desc-preprocessed_epo.fif"
                path.write_text("saved result", encoding="utf-8")
                dialog = self.dialog(params, [("preprocessed", path)])
                events = []

                def confirm(*_args):
                    self.assertTrue(path.exists())
                    events.append("confirmed")
                    return QMessageBox.StandardButton.Yes

                def run(**_kwargs):
                    events.append("processed")
                    path.unlink()

                getattr(dialog.preprocessor, operation).side_effect = run
                with patch(
                    "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                    side_effect=confirm,
                ), patch("ui.widgets.preprocessing_widgets.Worker", _ImmediateWorker):
                    dialog_class.run_process(dialog)

                self.assertEqual(events, ["confirmed", "processed"])
                dialog.settings_widget.save_settings.assert_not_called()
                self.assertFalse(path.exists())
                dialog.accept.assert_called_once_with()

    def test_new_processing_without_saved_outputs_does_not_need_a_deletion_prompt(self):
        for dialog_class, _stage_id, operation, params in self.CASES:
            if dialog_class is ReReferenceDialog:
                continue  # Re-referencing updates an existing epochs file.
            with self.subTest(dialog=dialog_class.__name__):
                dialog = self.dialog(params, [])
                with patch(
                    "ui.widgets.tools.processing_confirmation.QMessageBox.question"
                ) as confirm, patch("ui.widgets.preprocessing_widgets.Worker", _ImmediateWorker):
                    dialog_class.run_process(dialog)

                confirm.assert_not_called()
                getattr(dialog.preprocessor, operation).assert_called_once_with(**params)
                dialog.settings_widget.save_settings.assert_not_called()
                dialog.accept.assert_called_once_with()

    def test_orphaned_filter_checkpoint_requires_confirmation_before_replacement(self):
        for dialog_class in (ArtifactRemovalDialog, FilterContinuousDialog):
            with self.subTest(dialog=dialog_class.__name__), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "subject_desc-filterinput_raw.fif"
                checkpoint.write_text("unfiltered checkpoint", encoding="utf-8")
                dialog = self.dialog({}, [])
                dialog.preprocessor.paths["filter_input_raw"] = checkpoint
                with patch(
                    "ui.widgets.tools.processing_confirmation.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.No,
                ) as confirm, patch("ui.widgets.preprocessing_widgets.Worker") as worker:
                    dialog_class.run_process(dialog)

                confirm.assert_called_once()
                self.assertIn(checkpoint.name, confirm.call_args.args[2])
                worker.assert_not_called()
                self.assertEqual(checkpoint.read_text(encoding="utf-8"), "unfiltered checkpoint")

    def test_invalid_parameters_never_prompt_or_start_processing(self):
        for dialog_class, _stage_id, operation, params in self.CASES:
            with self.subTest(dialog=dialog_class.__name__):
                dialog = self.dialog(params, [])
                dialog.settings_widget.get_params.side_effect = ValueError("Invalid parameter")
                with patch(
                    "ui.widgets.preprocessing_widgets.confirm_processing_update"
                ) as confirm, patch("ui.widgets.preprocessing_widgets.Worker") as worker, patch(
                    "ui.widgets.preprocessing_widgets.QMessageBox.critical"
                ) as error:
                    dialog_class.run_process(dialog)

                error.assert_called_once()
                confirm.assert_not_called()
                worker.assert_not_called()
                getattr(dialog.preprocessor, operation).assert_not_called()
                dialog.settings_widget.save_settings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
