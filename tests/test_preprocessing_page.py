import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QMessageBox

from ui.pages.preprocessing_page import ProcessingPage, raw_inspection_decimation


class _ImmediateWorker:
    def __init__(self, operation, **_kwargs):
        self.operation = operation

    def exec_with_dialog(self, *_args):
        return self.operation()


class TestICASaving(unittest.TestCase):
    def test_save_ica_changes_invalidates_downstream_before_saving(self):
        preprocessor = Mock()
        page = SimpleNamespace(preprocessor=preprocessor)
        ica = Mock()
        path = Path("reviewed-ica.fif")
        operations = []
        preprocessor._clear_downstream_files.side_effect = (
            lambda stage_id: operations.append(("clear", stage_id))
        )
        ica.save.side_effect = lambda *_args, **_kwargs: operations.append(("save", path))

        with patch("ui.pages.preprocessing_page.Worker", _ImmediateWorker):
            saved = ProcessingPage._save_ica_changes(
                page,
                ica,
                path,
                "epochs_ica",
            )

        self.assertTrue(saved)
        self.assertEqual(
            operations,
            [("clear", "epochs_ica"), ("save", path)],
        )
        ica.save.assert_called_once_with(path, overwrite=True)
        preprocessor._clear_downstream_files.assert_called_once_with("epochs_ica")
        self.assertIs(preprocessor._epochs_ica, ica)
        preprocessor.add_description.assert_not_called()

    def test_ica_update_confirmation_names_downstream_files(self):
        preprocessor = Mock()
        preprocessor.existing_downstream_files.return_value = [
            ("preprocessed", Path("subject_desc-preprocessed_epo.fif")),
        ]
        page = SimpleNamespace(preprocessor=preprocessor)

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question:
            confirmed = ProcessingPage._confirm_ica_update(page, "epochs_ica")

        self.assertTrue(confirmed)
        self.assertIn("subject_desc-preprocessed_epo.fif", question.call_args.args[2])
        self.assertEqual(
            question.call_args.args[4],
            QMessageBox.StandardButton.No,
        )

    def test_cancelled_ica_review_restores_component_selection(self):
        ica = Mock()
        ica.exclude = [0]
        raw = Mock()
        raw.load_data.return_value = raw
        preprocessor = Mock()
        preprocessor.has.return_value = True
        preprocessor._get_last_continuous.return_value = raw
        preprocessor.continuous_ica = ica
        page = SimpleNamespace(
            preprocessor=preprocessor,
            _confirm_ica_update=Mock(),
            _save_ica_changes=Mock(),
            _refresh_after_step=Mock(),
        )

        def cancel_after_edit(*_args):
            ica.exclude = [1, 2]
            return False

        with patch(
            "ui.pages.preprocessing_page.run_ica_viewer",
            side_effect=cancel_after_edit,
        ):
            ProcessingPage.inspect_continuous_ica(page)

        self.assertEqual(ica.exclude, [0])
        page._save_ica_changes.assert_not_called()


class TestRawInspectionDecimation(unittest.TestCase):
    def test_uses_smallest_factor_at_or_below_display_ceiling(self):
        self.assertEqual(raw_inspection_decimation(4800, 1000), 5)
        self.assertEqual(raw_inspection_decimation(1001, 1000), 2)

    def test_does_not_decimate_data_already_within_ceiling(self):
        self.assertEqual(raw_inspection_decimation(1000, 1000), 1)
        self.assertEqual(raw_inspection_decimation(500, 1000), 1)

    def test_raw_viewer_receives_configured_decimation(self):
        raw = Mock()
        viz_raw = Mock()
        raw.copy.return_value = viz_raw
        viz_raw.info = {"bads": [], "sfreq": 4800.0}
        viz_raw.ch_names = ["EEG 001", "EEG 002"]
        viz_raw.annotations.copy.return_value = Mock()
        figure = Mock()
        viz_raw.plot.return_value = figure

        settings_store = Mock()
        settings_store.raw_inspect_max_sampling_hz.return_value = 1000
        preprocessor = Mock()
        preprocessor._get_last_continuous.return_value = raw
        page = SimpleNamespace(
            preprocessor=preprocessor,
            _raw_review_pending=False,
            _raw_review_figure=None,
            _after_raw_inspected=Mock(),
        )

        with patch(
            "ui.pages.preprocessing_page.get_settings_store",
            return_value=settings_store,
        ):
            ProcessingPage.inspect_raw_data(page)

        viz_raw.plot.assert_called_once_with(
            n_channels=2,
            duration=10,
            decim=5,
            use_opengl=None,
            splash=False,
            block=False,
        )
        figure.gotClosed.connect.assert_called_once_with(page._after_raw_inspected)


if __name__ == "__main__":
    unittest.main()
