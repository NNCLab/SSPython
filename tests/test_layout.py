import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QBoxLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from main import MainWindow
from ui.pages.preprocessing_page import ProcessingPage
from ui.widgets.real_time_widget import RealTimeMainWidget


class TestResponsiveLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_shell_keeps_contextual_workspace_without_an_inspector(self):
        window = MainWindow()
        self.addCleanup(window.close)
        self.assertIsInstance(window.shell_splitter, QSplitter)
        self.assertEqual(window.shell_splitter.count(), 2)
        self.assertEqual(window.shell_splitter.indexOf(window.workspace_panel), 0)
        self.assertEqual(window.shell_splitter.widget(1).objectName(), "contentFrame")
        self.assertIsNone(window.findChild(QWidget, "inspectorPanel"))
        self.assertNotIn(
            "Toggle Derivative Inspector",
            [action.text() for action in window.view_menu.actions()],
        )

        for page_name, uses_workspace in (
            ("Preferences", False),
            ("Home", True),
            ("Preprocessing", True),
            ("Analysis", True),
            ("Real-Time", False),
        ):
            window.navigate_to_page(page_name)
            self.app.processEvents()
            self.assertEqual(window.workspace_panel.isHidden(), not uses_workspace)
            self.assertEqual(window.shell_splitter.count(), 2)

    def test_real_time_configuration_stacks_in_compact_layout(self):
        widget = RealTimeMainWidget()
        widget.set_compact_layout(True)
        self.assertEqual(
            widget.config_row.direction(),
            QBoxLayout.Direction.TopToBottom,
        )

        widget.set_compact_layout(False)
        self.assertEqual(
            widget.config_row.direction(),
            QBoxLayout.Direction.LeftToRight,
        )
        widget.close()

    def test_preferences_toolbox_headers_have_room_for_padded_text(self):
        window = MainWindow()
        page = window.page_lookup["Preferences"]
        headers = [
            button
            for button in page.tool_box.findChildren(QAbstractButton)
            if button.parentWidget() is page.tool_box
        ]

        self.assertEqual(len(headers), page.tool_box.count())
        for header in headers:
            self.assertGreaterEqual(
                header.minimumHeight(),
                header.fontMetrics().height() + 12,
            )
        window.close()

    def test_preferences_categories_fill_the_remaining_page_height(self):
        window = MainWindow()
        window.resize(1200, 900)
        window.show()
        window.navigate_to_page("Preferences")
        self.app.processEvents()
        page = window.page_lookup["Preferences"]

        tool_box_index = page.content_layout.indexOf(page.tool_box)
        self.assertEqual(page.content_layout.stretch(tool_box_index), 1)
        self.assertEqual(
            page.tool_box.sizePolicy().verticalPolicy(),
            QSizePolicy.Policy.Expanding,
        )

        bottom_margin = page.content_layout.contentsMargins().bottom()
        for category_index in range(page.tool_box.count()):
            page.tool_box.setCurrentIndex(category_index)
            self.app.processEvents()

            expected_bottom = page.content_container.height() - bottom_margin
            actual_bottom = page.tool_box.geometry().bottom() + 1
            self.assertLessEqual(abs(actual_bottom - expected_bottom), 1)
            self.assertEqual(
                page.tool_box.currentWidget().sizePolicy().verticalPolicy(),
                QSizePolicy.Policy.Expanding,
            )
        window.close()

    def test_preprocessing_stage_actions_wrap_in_compact_layout(self):
        page = ProcessingPage()
        stage = page.stage_sections["epochs"]
        stage.setParent(None)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(stage)
        layout.addStretch()
        self.addCleanup(host.close)
        self.addCleanup(page.close)

        for width, direction in (
            (1000, QBoxLayout.Direction.LeftToRight),
            (460, QBoxLayout.Direction.TopToBottom),
            (1000, QBoxLayout.Direction.LeftToRight),
        ):
            host.resize(width, 500)
            host.show()
            self.app.processEvents()
            self.app.processEvents()
            self.assertEqual(stage.row_layout.direction(), direction)
            buttons = stage.actions_widget.findChildren(QAbstractButton)
            self.assertEqual(len(buttons), 4)
            for index, button in enumerate(buttons):
                self.assertTrue(stage.actions_widget.rect().contains(button.geometry()))
                for other in buttons[index + 1:]:
                    self.assertFalse(button.geometry().intersects(other.geometry()))

    def test_preprocessing_renders_each_stage_heading_once(self):
        page = ProcessingPage()
        stage_titles = [
            label.text()
            for label in page.findChildren(QLabel)
            if label.objectName() == "workflowActionTitle"
        ]

        self.assertEqual(
            stage_titles,
            [
                "1. Raw Import",
                "2. Continuous Cleanup",
                "3. Continuous ICA",
                "4. Epoch Extraction",
                "5. Epoch ICA",
                "6. Preprocessed Output",
            ],
        )
        self.assertEqual(len(page.action_buttons), 12)
        self.assertFalse(page.workflow_container.findChildren(QGroupBox))
        page.close()


if __name__ == "__main__":
    unittest.main()
