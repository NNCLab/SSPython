import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtCore import QItemSelectionModel, QModelIndex, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QDrag, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMainWindow, QMessageBox

from core.channel_info import ChannelInfo, MontageValidationError
from core.conversion import load_conversion_info
from core.info_creation import create_info, read_electrodes_tsv
from core.pipelines import get_pipeline
from main import MainWindow
from ui.widgets.tools.create_info_dialog import CreateInfoDialog


class TestInfoCreation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)

    def write_tsv(self, content):
        path = self.directory / "sub-01_electrodes.tsv"
        path.write_text(content, encoding="utf-8-sig")
        return path

    def test_builtin_info_preserves_channel_order_types_and_sampling_frequency(self):
        info = create_info(
            ChannelInfo([" cz ", "EOG1", "Fz", "STI 014"], ["EEG", "eog", "eeg", "stim"]),
            4800,
            "easycap-M1",
        )
        self.assertEqual(info["ch_names"], ["cz", "EOG1", "Fz", "STI 014"])
        self.assertEqual(info.get_channel_types(), ["eeg", "eog", "eeg", "stim"])
        self.assertEqual(info["sfreq"], 4800)
        self.assertTrue(np.isfinite(info["chs"][0]["loc"][:3]).all())

    def test_invalid_channels_and_sampling_frequency_are_rejected(self):
        for channels, message in [
            (ChannelInfo([], []), "at least one"),
            (ChannelInfo([""], ["eeg"]), "empty"),
            (ChannelInfo(["Cz", "cz"], ["eeg", "eeg"]), "unique"),
            (ChannelInfo(["Cz"], ["invalid-type"]), "valid MNE"),
            (ChannelInfo(["Cz"], []), "same length"),
        ]:
            with self.subTest(channels=channels), self.assertRaisesRegex(ValueError, message):
                create_info(channels, 1000, "standard_1020")
        for sfreq in (0, -1, float("nan"), float("inf"), "invalid", None):
            with self.subTest(sfreq=sfreq), self.assertRaisesRegex(ValueError, "positive finite"):
                create_info(ChannelInfo(["Cz"], ["eeg"]), sfreq, "standard_1020")

    def test_required_channels_without_positions_are_rejected(self):
        with self.assertRaisesRegex(MontageValidationError, "Missing"):
            create_info(ChannelInfo(["Missing"], ["eeg"]), 1000, "standard_1020")
        montage = mne.channels.make_dig_montage(
            ch_pos={"Cz": np.array([np.nan, 0.0, 0.1])}, coord_frame="head"
        )
        with self.assertRaisesRegex(ValueError, "finite electrode position"):
            create_info(ChannelInfo(["Cz"], ["eeg"]), 1000, montage)

    def test_tsv_units_are_converted_to_meters_and_extra_columns_are_allowed(self):
        for unit, scale in [("m", 1.0), ("cm", 0.01), ("mm", 0.001)]:
            with self.subTest(unit=unit):
                path = self.write_tsv("name\tx\ty\tz\ttype\nCz\t0\t10\t90\tcup\nFz\t0\t60\t60\tcup\n")
                montage = read_electrodes_tsv(path, unit=unit)
                self.assertEqual(montage.ch_names, ["Cz", "Fz"])
                self.assertEqual(montage.get_positions()["coord_frame"], "head")
                np.testing.assert_allclose(montage.get_positions()["ch_pos"]["Cz"], [0, 10 * scale, 90 * scale])

    def test_unavailable_tsv_positions_cannot_satisfy_montage_coverage(self):
        path = self.write_tsv("name\tx\ty\tz\nCz\t0\t0\t0.09\nEOG1\tn/a\tn/a\tn/a\n")
        montage = read_electrodes_tsv(path)
        self.assertEqual(montage.ch_names, ["Cz"])
        create_info(ChannelInfo(["Cz", "EOG1"], ["eeg", "eog"]), 1000, montage)
        with self.assertRaisesRegex(MontageValidationError, "EOG1"):
            create_info(ChannelInfo(["Cz", "EOG1"], ["eeg", "eeg"]), 1000, montage)

    def test_malformed_tsv_files_are_rejected(self):
        cases = [
            ("", "empty"),
            ("name\tx\ty\nCz\t0\t0\n", "name, x, y, and z"),
            ("name\tx\ty\tz\tx\nCz\t0\t0\t0.09\t0\n", "unique"),
            ("name\tx\ty\tz\n\t0\t0\t0.09\n", "Missing electrode name"),
            ("name\tx\ty\tz\nCz\t0\t0\t0.09\ncz\t0\t0\t0.1\n", "Duplicate"),
            ("name\tx\ty\tz\nCz\t0\tbad\t0.09\n", "Invalid coordinates"),
            ("name\tx\ty\tz\nCz\t0\tn/a\t0.09\n", "Incomplete"),
            ("name\tx\ty\tz\nCz\t0\tinf\t0.09\n", "finite"),
            ("name\tx\ty\tz\nCz\t0\tnan\t0.09\n", "finite"),
            ("name\tx\ty\tz\nCz\tn/a\tn/a\tn/a\n", "no available"),
            ("name\tx\ty\tz\nCz\t0\t0\n", "number of fields"),
            ("name\tx\ty\tz\nCz\t0\t0\t0.09\textra\n", "number of fields"),
        ]
        for content, message in cases:
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, message):
                read_electrodes_tsv(self.write_tsv(content))
        with self.assertRaisesRegex(ValueError, "units"):
            read_electrodes_tsv(self.write_tsv("name\tx\ty\tz\nCz\t0\t0\t0.09\n"), unit="inch")

    def test_saved_standalone_info_is_usable_by_conversion_and_realtime(self):
        from ui.widgets.real_time_widget import load_realtime_info

        montage = read_electrodes_tsv(self.write_tsv("name\tx\ty\tz\nCz\t0\t0\t90\n"), unit="mm")
        info = create_info(ChannelInfo(["Cz", "EOG1"], ["eeg", "eog"]), 200, montage)
        path = self.directory / "recording-info.fif"
        info.save(path)
        for loader in (load_conversion_info, load_realtime_info):
            with self.subTest(loader=loader.__name__):
                loaded = loader(path)
                self.assertEqual(loaded["ch_names"], ["Cz", "EOG1"])
                self.assertEqual(loaded.get_channel_types(), ["eeg", "eog"])
                self.assertEqual(loaded["sfreq"], 200)
                np.testing.assert_allclose(loaded["chs"][0]["loc"][:3], [0, 0, 0.09])


class TestCreateInfoDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.dialog = CreateInfoDialog(output_directory=self.directory)
        self.addCleanup(self.dialog.close)

    def test_add_edit_and_remove_multiple_channels_preserves_order(self):
        for name in ("Fz", "Cz", "Pz", "EOG1"):
            self.dialog._append_channel(name)
        self.dialog.table.cellWidget(3, 1).setCurrentText("eog")
        selection = self.dialog.table.selectionModel()
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        selection.select(self.dialog.table.model().index(0, 0), flags)
        selection.select(self.dialog.table.model().index(2, 0), flags)
        self.dialog.remove_channels_button.click()
        info = self.dialog.get_info()
        self.assertEqual(info["ch_names"], ["Cz", "EOG1"])
        self.assertEqual(info.get_channel_types(), ["eeg", "eog"])

    def test_reordered_channels_keep_types_positions_and_saved_order(self):
        for name, kind in [("Fz", "eeg"), ("Pz", "eeg"), ("EOG1", "eog"), ("Cz", "eeg")]:
            self.dialog._append_channel(name, kind)
        original_info = self.dialog.get_info()
        eog_combo = self.dialog.table.cellWidget(2, 1)
        model = self.dialog.table.model()
        # Qt's internal drag-and-drop uses moveRow to move complete rows and
        # retain their persistent type editors.
        self.assertTrue(model.moveRow(QModelIndex(), 2, QModelIndex(), 0))
        self.assertTrue(model.moveRow(QModelIndex(), 3, QModelIndex(), 1))
        self.app.processEvents()
        self.assertIs(self.dialog.table.cellWidget(0, 1), eog_combo)
        info = self.dialog.get_info()
        self.assertEqual(info["ch_names"], ["EOG1", "Cz", "Fz", "Pz"])
        self.assertEqual(info.get_channel_types(), ["eog", "eeg", "eeg", "eeg"])
        original_positions = {
            ch["ch_name"]: ch["loc"][:3] for ch in original_info["chs"]
        }
        output_path = self.directory / "reordered-info.fif"
        with (
            patch("ui.widgets.tools.create_info_dialog.QFileDialog.getSaveFileName", return_value=(str(output_path), "")),
            patch("ui.widgets.tools.create_info_dialog.QMessageBox.information"),
        ):
            self.dialog.create_button.click()
        saved = load_conversion_info(output_path)
        self.assertEqual(saved["ch_names"], info["ch_names"])
        self.assertEqual(saved.get_channel_types(), info.get_channel_types())
        for ch in saved["chs"]:
            np.testing.assert_allclose(ch["loc"][:3], original_positions[ch["ch_name"]], equal_nan=True)

    def test_moving_multiple_rows_keeps_order_and_allows_removal_and_addition(self):
        for name, kind in [("Fz", "eeg"), ("EOG1", "eog"), ("Cz", "eeg"), ("STI 014", "stim")]:
            self.dialog._append_channel(name, kind)
        model = self.dialog.table.model()
        self.assertTrue(model.moveRows(QModelIndex(), 0, 2, QModelIndex(), 4))
        info = self.dialog.get_info()
        self.assertEqual(info["ch_names"], ["Cz", "STI 014", "Fz", "EOG1"])
        self.assertEqual(info.get_channel_types(), ["eeg", "stim", "eeg", "eog"])
        self.dialog.table.selectRow(3)
        self.dialog.remove_channels_button.click()
        self.dialog._append_channel("Pz")
        info = self.dialog.get_info()
        self.assertEqual(info["ch_names"], ["Cz", "STI 014", "Fz", "Pz"])
        self.assertEqual(info.get_channel_types(), ["eeg", "stim", "eeg", "eeg"])

    def test_native_drag_drop_moves_selected_rows_without_overwriting_channels(self):
        if self.app.platformName() == "offscreen":
            self.skipTest("Native drag-and-drop requires QT_QPA_PLATFORM=minimal or a desktop platform.")
        for name, kind in [("Fz", "eeg"), ("EOG1", "eog"), ("STI 014", "stim"), ("Cz", "eeg"), ("Pz", "eeg")]:
            self.dialog._append_channel(name, kind)
        self.dialog.show()
        self.app.processEvents()
        table = self.dialog.table
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        for row in (0, 2):
            table.selectionModel().select(table.model().index(row, 0), flags)
        mime = table.model().mimeData(table.selectedIndexes())
        accepted = []
        errors = []

        def drop_selected_rows():
            try:
                table.scrollToItem(table.item(4, 0))
                self.app.processEvents()
                point = QPoint(30, table.visualItemRect(table.item(4, 0)).bottom() - 1)
                enter = QDragEnterEvent(point, Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
                move = QDragMoveEvent(point, Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
                drop = QDropEvent(QPointF(point), Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
                for event in (enter, move, drop):
                    self.app.sendEvent(table.viewport(), event)
                    accepted.append(event.isAccepted())
            except Exception as exc:
                errors.append(exc)
            finally:
                QDrag.cancel()

        QTimer.singleShot(50, drop_selected_rows)
        table.startDrag(Qt.DropAction.MoveAction)
        self.assertEqual(errors, [])
        self.assertEqual(accepted, [True, True, True])
        info = self.dialog.get_info()
        self.assertEqual(info["ch_names"], ["EOG1", "Cz", "Pz", "Fz", "STI 014"])
        self.assertEqual(info.get_channel_types(), ["eog", "eeg", "eeg", "eeg", "stim"])
        self.assertEqual(table.rowCount(), 5)

    def test_add_montage_channels_preserves_existing_rows_and_avoids_duplicates(self):
        self.dialog.builtin_montage_combo.setCurrentText("easycap-M1")
        self.dialog._append_channel("cz")
        self.dialog._append_channel("EOG1", "eog")
        self.dialog.add_montage_channels_button.click()
        count = self.dialog.table.rowCount()
        self.dialog.add_montage_channels_button.click()
        info = self.dialog.get_info()
        self.assertEqual(self.dialog.table.rowCount(), count)
        self.assertEqual(info["ch_names"][:2], ["cz", "EOG1"])
        self.assertEqual(info.get_channel_types()[1], "eog")
        self.assertEqual(sum(name.casefold() == "cz" for name in info["ch_names"]), 1)

    def test_tsv_montage_can_populate_channels_and_create_info(self):
        path = self.directory / "electrodes.tsv"
        path.write_text("name\tx\ty\tz\nCz\t0\t0\t90\nFz\t0\t60\t60\n", encoding="utf-8")
        self.dialog.montage_source_combo.setCurrentIndex(1)
        self.dialog.electrodes_path_input.setText(str(path))
        self.dialog.coordinate_unit_combo.setCurrentIndex(2)
        self.dialog.add_montage_channels_button.click()
        self.dialog.sfreq_input.setValue(4800)
        info = self.dialog.get_info()
        self.assertEqual(info["ch_names"], ["Cz", "Fz"])
        self.assertEqual(info["sfreq"], 4800)
        np.testing.assert_allclose(info["chs"][0]["loc"][:3], [0, 0, 0.09])
        self.assertEqual(self.dialog.montage_stack.currentIndex(), 1)

    def test_invalid_settings_and_cancelled_save_leave_dialog_open(self):
        with (
            patch("ui.widgets.tools.create_info_dialog.QMessageBox.warning") as warning,
            patch("ui.widgets.tools.create_info_dialog.QFileDialog.getSaveFileName") as save_dialog,
        ):
            self.dialog.accept()
            warning.assert_called_once()
            save_dialog.assert_not_called()
        self.dialog._append_channel("Cz")
        with patch("ui.widgets.tools.create_info_dialog.QFileDialog.getSaveFileName", return_value=("", "")):
            self.dialog.accept()
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Rejected)
        self.assertIsNone(self.dialog.info)
        self.assertIsNone(self.dialog.saved_path)

    def test_save_creates_readable_info_and_accepts_dialog(self):
        self.dialog._append_channel("Cz")
        output_path = self.directory / "test-info"
        with (
            patch("ui.widgets.tools.create_info_dialog.QFileDialog.getSaveFileName", return_value=(str(output_path), "")),
            patch("ui.widgets.tools.create_info_dialog.QMessageBox.information"),
        ):
            self.dialog.accept()
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(self.dialog.saved_path, output_path.with_suffix(".fif"))
        self.assertIsInstance(self.dialog.info, mne.Info)
        self.assertEqual(load_conversion_info(self.dialog.saved_path)["ch_names"], ["Cz"])

    def test_save_commits_the_channel_name_being_edited(self):
        self.dialog.show()
        self.app.processEvents()
        self.dialog.add_channel_button.click()
        self.app.processEvents()
        editor = self.app.focusWidget()
        self.assertIsInstance(editor, QLineEdit)
        QTest.keyClicks(editor, "Cz")
        output_path = self.directory / "edited-info.fif"
        with (
            patch("ui.widgets.tools.create_info_dialog.QFileDialog.getSaveFileName", return_value=(str(output_path), "")),
            patch("ui.widgets.tools.create_info_dialog.QMessageBox.information"),
        ):
            self.dialog.create_button.click()
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(load_conversion_info(output_path)["ch_names"], ["Cz"])

    def test_appended_suffix_does_not_overwrite_without_confirmation(self):
        self.dialog._append_channel("Cz")
        output_path = self.directory / "test-info.fif"
        output_path.write_bytes(b"existing info")
        with (
            patch("ui.widgets.tools.create_info_dialog.QFileDialog.getSaveFileName", return_value=(str(output_path.with_suffix("")), "")),
            patch("ui.widgets.tools.create_info_dialog.QMessageBox.question", return_value=QMessageBox.StandardButton.No),
        ):
            self.dialog.accept()
        self.assertEqual(output_path.read_bytes(), b"existing info")
        self.assertIsNone(self.dialog.saved_path)

    def test_write_failure_leaves_dialog_available_for_retry(self):
        self.dialog._append_channel("Cz")
        output_path = self.directory / "test-info.fif"
        with (
            patch("ui.widgets.tools.create_info_dialog.QFileDialog.getSaveFileName", return_value=(str(output_path), "")),
            patch("mne.Info.save", side_effect=OSError("Cannot write file")),
            patch("ui.widgets.tools.create_info_dialog.QMessageBox.critical") as critical,
        ):
            self.dialog.accept()
        critical.assert_called_once()
        self.assertIsNone(self.dialog.saved_path)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Rejected)

    def test_file_menu_action_opens_create_info_dialog(self):
        # Exercise the real menu without constructing unrelated analysis pages
        # or accessing the user's saved workspace settings.
        window = MainWindow.__new__(MainWindow)
        QMainWindow.__init__(window)
        self.addCleanup(window.deleteLater)
        window.workspace_panel = SimpleNamespace(select_folder=lambda: None)
        window.current_pipeline = get_pipeline("standard")
        window.current_folder = self.directory
        window._setup_menu()
        action = next(action for action in window.file_menu.actions() if action.text() == "Create INFO")
        with patch("main.CreateInfoDialog") as dialog_class:
            action.trigger()
        dialog_class.assert_called_once_with(parent=window, output_directory=self.directory)
        dialog_class.return_value.exec.assert_called_once()


if __name__ == "__main__":
    unittest.main()
