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


class TestICASettings(unittest.TestCase):
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

    def widget(self, mode, max_components=8):
        widget = ICASettingsWidget(mode, max_components=max_components)
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

    def test_manual_component_suggestions_survive_save_and_reload(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode):
                widget = self.widget(mode)
                widget.auto_n_components_checkbox.setChecked(False)
                widget.n_components_input.setValue(4)
                widget.save_settings()

                reloaded = self.widget(mode)
                self.assertFalse(reloaded.auto_n_components_checkbox.isChecked())
                self.assertTrue(reloaded.n_components_input.isEnabled())
                self.assertEqual(reloaded.get_params()["n_components"], 4)
                self.assertNotIn("auto_n_components", reloaded.get_params())

    def test_automatic_component_suggestions_survive_save_and_reload(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode):
                widget = self.widget(mode)
                widget.auto_n_components_checkbox.setChecked(False)
                widget.n_components_input.setValue(4)
                widget.save_settings()
                widget.auto_n_components_checkbox.setChecked(True)
                widget.save_settings()

                reloaded = self.widget(mode)
                self.assertTrue(reloaded.auto_n_components_checkbox.isChecked())
                self.assertFalse(reloaded.n_components_input.isEnabled())
                self.assertIsNone(reloaded.get_params()["n_components"])

    def test_legacy_automatic_flag_takes_precedence_over_component_count(self):
        for mode in ("continuous", "epochs"):
            for automatic, count, expected in (
                (True, 4, None), (False, 4, 4), (False, None, 8),
            ):
                with self.subTest(mode=mode, automatic=automatic, count=count):
                    self.store.set(f"pipelines/standard/ica/{mode}", {
                        "auto_n_components": automatic, "n_components": count,
                    })
                    widget = self.widget(mode)
                    self.assertEqual(widget.auto_n_components_checkbox.isChecked(), automatic)
                    self.assertEqual(widget.get_params()["n_components"], expected)

    def test_missing_component_count_uses_automatic_selection(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode):
                self.store.set(f"pipelines/standard/ica/{mode}", {"max_iter": 2000})
                widget = self.widget(mode)
                self.assertTrue(widget.auto_n_components_checkbox.isChecked())
                self.assertEqual(widget.n_components_input.value(), 8)
                self.assertIsNone(widget.get_params()["n_components"])

    def test_saved_component_count_respects_recording_maximum(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode):
                self.store.set(f"pipelines/standard/ica/{mode}", {"n_components": 64})
                widget = self.widget(mode)
                self.assertFalse(widget.auto_n_components_checkbox.isChecked())
                self.assertEqual(widget.get_params()["n_components"], 8)
                self.assertEqual(
                    self.store.get(widget.settings_path)["n_components"], 64,
                )

    def test_preferences_can_edit_manual_counts_without_recording_limit(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode):
                self.store.set(f"pipelines/standard/ica/{mode}", {"n_components": 256})
                widget = self.widget(mode, max_components=None)
                self.assertTrue(widget.n_components_input.isVisible())
                self.assertTrue(widget.n_components_input.isEnabled())
                self.assertEqual(widget.n_components_input.value(), 256)
                widget.n_components_input.setFocus()
                widget.n_components_input.selectAll()
                QTest.keyClicks(widget.n_components_input, "512")
                widget.save_settings()

                reloaded = self.widget(mode, max_components=None)
                self.assertFalse(reloaded.auto_n_components_checkbox.isChecked())
                self.assertEqual(reloaded.get_params()["n_components"], 512)

    def test_preferences_without_saved_count_start_automatic_and_allow_manual_entry(self):
        for mode in ("continuous", "epochs"):
            with self.subTest(mode=mode):
                widget = self.widget(mode, max_components=None)
                self.assertTrue(widget.auto_n_components_checkbox.isChecked())
                self.assertFalse(widget.n_components_input.isEnabled())
                widget.auto_n_components_checkbox.setChecked(False)
                self.assertTrue(widget.n_components_input.isEnabled())
                self.assertGreater(widget.get_params()["n_components"], 0)


if __name__ == "__main__":
    unittest.main()
