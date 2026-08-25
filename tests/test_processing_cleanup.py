import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QMessageBox

from core.pipelines import build_analysis_paths
from core.processing import Preprocessor
from ui.widgets.run_ica_widget import RunICADialog


class _CachedStage:
    def __init__(self):
        self.loaded = False
        self.closed = False

    def load_data(self):
        self.loaded = True

    def close(self):
        self.closed = True


class TestDownstreamCleanup(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.preprocessor = Preprocessor.__new__(Preprocessor)
        self.preprocessor.processing_order = [
            "raw",
            "filtered_raw",
            "continuous_ica",
            "epochs",
            "epochs_ica",
            "preprocessed",
        ]
        self.preprocessor.paths = {
            stage: root / f"{stage}.fif"
            for stage in self.preprocessor.processing_order
        }
        self.preprocessor.output_dir = root
        self.preprocessor.has = lambda stage: self.preprocessor.paths[stage].exists()
        for stage in ("epochs", "epochs_ica", "preprocessed"):
            self.preprocessor.paths[stage].touch()
        self.analysis_paths = build_analysis_paths(
            self.preprocessor.paths["preprocessed"],
            root,
            create_dirs=False,
        )
        for path in self.analysis_paths.values():
            path.touch()

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_continuous_ica_clears_all_downstream_files_and_caches(self):
        cached_epochs = _CachedStage()
        self.preprocessor._epochs = cached_epochs

        self.preprocessor._clear_downstream_files("continuous_ica", verbose=False)

        self.assertTrue(cached_epochs.loaded)
        self.assertTrue(cached_epochs.closed)
        self.assertFalse(hasattr(self.preprocessor, "_epochs"))
        for stage in ("epochs", "epochs_ica", "preprocessed"):
            self.assertFalse(self.preprocessor.paths[stage].exists())
        self.assertTrue(all(not path.exists() for path in self.analysis_paths.values()))

    def test_current_stage_cache_is_released_before_overwrite(self):
        cached_epochs = _CachedStage()
        cached_epochs.preload = False
        self.preprocessor._epochs = cached_epochs

        self.preprocessor._clear_downstream_files("epochs", verbose=False)

        self.assertTrue(cached_epochs.loaded)
        self.assertTrue(cached_epochs.closed)
        self.assertFalse(hasattr(self.preprocessor, "_epochs"))
        self.assertTrue(self.preprocessor.paths["epochs"].exists())
        self.assertFalse(self.preprocessor.paths["epochs_ica"].exists())
        self.assertFalse(self.preprocessor.paths["preprocessed"].exists())
        self.assertTrue(all(not path.exists() for path in self.analysis_paths.values()))

    def test_existing_downstream_files_reports_only_later_outputs(self):
        downstream_files = self.preprocessor.existing_downstream_files("epochs")

        self.assertEqual(
            downstream_files,
            [
                ("epochs_ica", self.preprocessor.paths["epochs_ica"]),
                ("preprocessed", self.preprocessor.paths["preprocessed"]),
                *self.analysis_paths.items(),
            ],
        )

    def test_recomputing_preprocessed_file_clears_analysis_derivatives(self):
        self.preprocessor._clear_downstream_files("preprocessed", verbose=False)

        self.assertTrue(self.preprocessor.paths["preprocessed"].exists())
        self.assertTrue(all(not path.exists() for path in self.analysis_paths.values()))

    def test_locked_file_deletion_is_retried(self):
        real_remove = os.remove
        attempts = 0

        def remove_after_retry(path):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise PermissionError("file is temporarily locked")
            real_remove(path)

        with patch("core.processing.os.remove", side_effect=remove_after_retry):
            self.preprocessor._clear_downstream_files("continuous_ica", verbose=False)

        expected_deletions = 3 + len(self.analysis_paths)
        self.assertEqual(attempts, expected_deletions + 1)
        for stage in ("epochs", "epochs_ica", "preprocessed"):
            self.assertFalse(self.preprocessor.paths[stage].exists())
        self.assertTrue(all(not path.exists() for path in self.analysis_paths.values()))

    def test_persistent_file_lock_aborts_cleanup(self):
        locked_path = self.preprocessor.paths["epochs"]

        with patch(
            "core.processing.os.remove",
            side_effect=PermissionError("file remains locked"),
        ):
            with self.assertRaisesRegex(PermissionError, "Close any plot or program"):
                self.preprocessor._clear_downstream_files(
                    "continuous_ica",
                    verbose=False,
                )

        self.assertTrue(locked_path.exists())


class TestRunICAStageLookup(unittest.TestCase):
    def test_continuous_mode_uses_pipeline_stage_identifier(self):
        preprocessor = Mock()
        preprocessor.has.return_value = True
        preprocessor.existing_downstream_files.return_value = [
            ("epochs", Path("subject_desc-epoched_epo.fif")),
        ]
        settings_widget = Mock()
        settings_widget.get_params.return_value = {}
        dialog = SimpleNamespace(
            preprocessor=preprocessor,
            mode="continuous",
            settings_widget=settings_widget,
        )

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            RunICADialog.run_ica(dialog)

        preprocessor.has.assert_called_once_with("continuous_ica")
        preprocessor.existing_downstream_files.assert_called_once_with("continuous_ica")
        self.assertIn("subject_desc-epoched_epo.fif", question.call_args.args[2])
        preprocessor.run_continuous_ica.assert_not_called()


if __name__ == "__main__":
    unittest.main()
