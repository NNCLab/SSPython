import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.app_settings import SettingsStore
from ui.widgets.run_ica_widget import ICASettingsWidget


class TestICAIterationEditing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SettingsStore(QSettings(
            str(Path(directory.name) / "settings.ini"), QSettings.Format.IniFormat,
        ))
        patcher = patch("ui.widgets.run_ica_widget.get_settings_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def widget(self, mode):
        widget = ICASettingsWidget(mode, max_components=8)
        widget.advanced_settings.set_expanded(True)
        widget.show()
        self.app.processEvents()
        self.addCleanup(widget.close)
        return widget

    @staticmethod
    def type_iterations(widget, value):
        widget.max_iter_input.setFocus()
        widget.max_iter_input.selectAll()
        QTest.keyClicks(widget.max_iter_input, str(value))

    def test_typed_iteration_counts_survive_focus_loss(self):
        for mode in ("continuous", "epochs"):
            widget = self.widget(mode)
            for value in (1, 2, 50, 101, 1234, 20000):
                with self.subTest(mode=mode, iterations=value):
                    widget.max_iter_input.setValue(100)
                    self.type_iterations(widget, value)
                    widget.random_state_input.setFocus()
                    self.app.processEvents()
                    self.assertEqual(widget.max_iter_input.value(), value)
                    self.assertEqual(widget.get_params()["max_iter"], value)

    def test_save_commits_typed_iterations_while_editor_has_focus(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode):
                widget = self.widget(mode)
                self.type_iterations(widget, 2)
                widget.save_settings()
                self.assertEqual(self.store.get(widget.settings_path)["max_iter"], 2)
                reloaded = self.widget(mode)
                self.assertEqual(reloaded.max_iter_input.value(), 2)


if __name__ == "__main__":
    unittest.main()
