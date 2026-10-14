import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QProgressBar, QPushButton, QWidget

from utils import LoadingDialog, Worker, get_path


class TestLoadingDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widgets = []
        self.workers = []

    def tearDown(self):
        for worker in self.workers:
            worker.requestInterruption()
            self.assertTrue(worker.wait(3000), "Worker did not finish")
        for widget in reversed(self.widgets):
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def dialog(self, **kwargs):
        dialog = LoadingDialog("Working...", **kwargs)
        self.widgets.append(dialog)
        return dialog

    def worker(self, task, **kwargs):
        worker = Worker(task, **kwargs)
        self.workers.append(worker)
        return worker

    def test_busy_progress_ignores_internal_stage_percentages(self):
        dialog = self.dialog()
        dialog.show()
        dialog.update_progress(100)
        dialog.setLabelText("Saving results...")

        self.assertEqual((dialog.minimum(), dialog.maximum()), (0, 0))
        self.assertEqual(dialog.findChild(QProgressBar).text(), "")
        self.assertEqual(dialog.labelText(), "Saving results...")
        self.assertTrue(dialog.isVisible())

    def test_determinate_progress_does_not_reset_or_close_at_100(self):
        dialog = self.dialog(determinate=True)
        dialog.show()
        dialog.update_progress(45)
        self.assertEqual(dialog.value(), 45)
        dialog.update_progress(100)

        self.assertEqual(dialog.value(), 100)
        self.assertTrue(dialog.isVisible())

    def test_busy_animation_moves_in_both_themes(self):
        original_stylesheet = self.app.styleSheet()
        original_style = self.app.style().objectName()
        self.app.setStyle("Fusion")
        try:
            for theme in ("dark", "light"):
                with self.subTest(theme=theme):
                    self.app.setStyleSheet(
                        Path(get_path(f"style/{theme}_theme.qss")).read_text()
                    )
                    dialog = self.dialog()
                    dialog.resize(360, 110)
                    dialog.show()
                    QTest.qWait(80)
                    bar = dialog.findChild(QProgressBar)
                    first_frame = bar.grab().toImage()
                    QTest.qWait(160)
                    self.assertNotEqual(first_frame, bar.grab().toImage())
                    dialog.accept()
        finally:
            self.app.setStyleSheet(original_stylesheet)
            self.app.setStyle(original_style)

    def test_child_widget_owner_is_normalized_to_its_window(self):
        owner = QWidget()
        self.widgets.append(owner)
        page = QWidget(owner)
        dialog = self.dialog(parent=page)

        self.assertIs(dialog.parentWidget(), owner)
        self.assertTrue(dialog.isWindow())
        self.assertEqual(dialog.windowModality(), Qt.WindowModality.WindowModal)

    def test_dialog_is_raised_after_showing(self):
        dialog = self.dialog()
        with patch.object(dialog, "raise_") as raise_window, patch.object(
            dialog, "activateWindow"
        ) as activate_window:
            dialog.show()
            self.app.processEvents()

        raise_window.assert_called_once()
        activate_window.assert_called_once()

    def test_pending_raise_does_not_activate_a_finished_dialog(self):
        dialog = self.dialog()
        with patch.object(dialog, "raise_") as raise_window, patch.object(
            dialog, "activateWindow"
        ) as activate_window:
            dialog.show()
            dialog.accept()
            self.app.processEvents()

        raise_window.assert_not_called()
        activate_window.assert_not_called()

    def test_cancel_button_is_only_present_for_cancellable_work(self):
        self.assertIsNone(self.dialog().findChild(QPushButton))
        self.assertIsNotNone(self.dialog(cancellable=True).findChild(QPushButton))

    def test_silent_worker_keeps_busy_dialog_and_gui_responsive(self):
        release = threading.Event()
        self.addCleanup(release.set)
        result = object()
        observations = {}

        def task():
            release.wait(3)
            return result

        def observe():
            dialog = QApplication.activeModalWidget()
            observations["busy"] = (dialog.minimum(), dialog.maximum())
            observations["visible"] = dialog.isVisible()
            release.set()

        worker = self.worker(task)
        QTimer.singleShot(30, observe)
        self.assertIs(worker.exec_with_dialog(), result)
        self.assertEqual(observations, {"busy": (0, 0), "visible": True})
        self.assertIsNone(worker._result)

    def test_worker_preserves_lifetime_owner_and_uses_active_modal_dialog(self):
        owner = QWidget()
        self.widgets.append(owner)
        page = QWidget(owner)
        settings = QDialog(owner)
        self.widgets.append(settings)
        settings.open()
        self.app.processEvents()
        release = threading.Event()
        self.addCleanup(release.set)
        observations = {}

        def observe():
            dialog = QApplication.activeModalWidget()
            observations["parent"] = dialog.parentWidget()
            observations["transient_parent"] = dialog.windowHandle().transientParent()
            release.set()

        worker = self.worker(lambda: release.wait(3), parent=page)
        QTimer.singleShot(30, observe)
        worker.exec_with_dialog()

        self.assertIs(observations["parent"], settings)
        self.assertIs(observations["transient_parent"], settings.windowHandle())
        self.assertIs(worker.parent(), page)

    def test_explicit_dialog_owner_and_determinate_mode(self):
        owner = QWidget()
        self.widgets.append(owner)
        settings = QDialog(owner)
        self.widgets.append(settings)
        release = threading.Event()
        self.addCleanup(release.set)
        observations = {}

        def task():
            worker.progress.emit(60)
            release.wait(3)

        def observe():
            dialog = QApplication.activeModalWidget()
            observations["parent"] = dialog.parentWidget()
            observations["value"] = dialog.value()
            release.set()

        worker = self.worker(task, parent=owner)
        QTimer.singleShot(30, observe)
        worker.exec_with_dialog(dialog_parent=settings, determinate=True)

        self.assertIs(observations["parent"], settings)
        self.assertEqual(observations["value"], 60)
        self.assertIs(worker.parent(), owner)

    def test_fast_worker_finishes_without_reopening_the_dialog(self):
        worker = self.worker(lambda: "done")
        self.assertEqual(worker.exec_with_dialog(), "done")
        QTest.qWait(550)
        self.assertFalse(any(
            isinstance(widget, LoadingDialog) and widget.isVisible()
            for widget in QApplication.topLevelWidgets()
        ))

    def test_worker_error_closes_dialog_and_shows_error_with_same_owner(self):
        owner = QWidget()
        self.widgets.append(owner)

        def task():
            raise ValueError("Cannot load data")

        worker = self.worker(task, parent=owner)
        with patch("utils.ErrorDialog") as error_dialog:
            self.assertIsNone(worker.exec_with_dialog())

        self.assertIn("Cannot load data", error_dialog.call_args.args[0])
        self.assertIs(error_dialog.call_args.args[1], owner)
        error_dialog.return_value.exec.assert_called_once()

    def test_cancel_button_requests_cooperative_interruption(self):
        def task(is_cancelled):
            while True:
                is_cancelled()
                threading.Event().wait(0.005)

        worker = self.worker(task)

        def cancel():
            QApplication.activeModalWidget().findChild(QPushButton).click()

        QTimer.singleShot(30, cancel)
        self.assertIsNone(worker.exec_with_dialog())
        self.assertTrue(worker.wait(3000))
        self.assertIn("cancelled", worker._error)


if __name__ == "__main__":
    unittest.main()
