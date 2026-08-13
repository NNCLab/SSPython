import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QWidget,
)

from main import MainWindow
from ui.pages.preprocessing_page import ProcessingPage, WorkflowDescriptionLabel
from ui.widgets.real_time_widget import RealTimeMainWidget
from ui.widgets.workspace_panel import DatasetInspectorPanel


class TestResponsiveLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_shell_uses_resizable_splitter_and_contextual_panels(self):
        window = MainWindow()
        self.assertIsInstance(window.shell_splitter, QSplitter)
        self.assertEqual(window.shell_splitter.indexOf(window.workspace_panel), 0)
        self.assertEqual(window.shell_splitter.indexOf(window.dataset_inspector_panel), 2)

        window.navigate_to_page("Preferences")
        self.app.processEvents()
        self.assertTrue(window.workspace_panel.isHidden())
        self.assertTrue(window.dataset_inspector_panel.isHidden())

        window.current_dataset = object()
        window.navigate_to_page("Home")
        self.app.processEvents()
        self.assertFalse(window.workspace_panel.isHidden())
        self.assertTrue(window.dataset_inspector_panel.isHidden())

        window.navigate_to_page("Preprocessing")
        self.app.processEvents()
        self.assertFalse(window.workspace_panel.isHidden())
        self.assertFalse(window.dataset_inspector_panel.isHidden())

        window.navigate_to_page("Real-Time")
        self.app.processEvents()
        self.assertTrue(window.workspace_panel.isHidden())
        self.assertTrue(window.dataset_inspector_panel.isHidden())
        window.close()

    def test_inspector_can_be_temporarily_compacted(self):
        panel = DatasetInspectorPanel()
        panel.set_responsive_collapsed(True)

        self.assertEqual(panel.minimumWidth(), panel.collapsed_width)
        self.assertEqual(panel.maximumWidth(), panel.collapsed_width)
        self.assertTrue(panel.content_container.isHidden())

        panel.set_responsive_collapsed(False)
        self.assertEqual(panel.minimumWidth(), 220)
        self.assertEqual(panel.maximumWidth(), 420)
        self.assertFalse(panel.content_container.isHidden())
        panel.close()

    def test_collapsed_inspector_returns_its_width_to_the_content(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        content = QWidget()
        panel = DatasetInspectorPanel()
        splitter.addWidget(content)
        splitter.addWidget(panel)
        splitter.resize(1000, 500)
        splitter.setSizes([700, 300])
        splitter.show()
        self.app.processEvents()

        content_width_before = content.width()
        panel.set_responsive_collapsed(True)
        self.app.processEvents()

        self.assertEqual(panel.width(), panel.collapsed_width)
        self.assertGreater(content.width(), content_width_before)
        self.assertGreaterEqual(panel.toggle_button.height(), panel.height() - 2)

        panel.set_responsive_collapsed(False)
        self.app.processEvents()
        self.assertGreaterEqual(panel.width(), 220)
        splitter.close()

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

    def test_preprocessing_descriptions_expand_to_their_wrapped_height(self):
        page = ProcessingPage()
        page.resize(520, 700)
        page.show()
        self.app.processEvents()

        descriptions = page.findChildren(WorkflowDescriptionLabel)
        self.assertTrue(descriptions)
        for description in descriptions:
            required_height = description.heightForWidth(description.width())
            self.assertGreaterEqual(description.height(), required_height)
        page.close()

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
                "Raw Import",
                "Continuous Cleanup",
                "Continuous ICA",
                "Epoch Extraction",
                "Epoch ICA",
                "Preprocessed Output",
            ],
        )
        self.assertEqual(len(page.action_buttons), 12)
        page.close()


if __name__ == "__main__":
    unittest.main()
