import os
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import mne
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QSizePolicy,
    QSplitter,
)

from main import MainWindow
from core.pipelines import DatasetRecord, get_pipeline
from utils import apply_theme
from ui.pages.preprocessing_page import ProcessingPage
from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.widgets.real_time_widget import RealTimeMainWidget
from ui.widgets.workspace_panel import WorkspacePanel
from ui.widgets.layout_helpers import dataset_label


class TestResponsiveLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_shell_keeps_workspace_without_a_details_panel(self):
        window = MainWindow()
        self.assertIsInstance(window.shell_splitter, QSplitter)
        self.assertEqual(window.shell_splitter.indexOf(window.workspace_panel), 0)
        self.assertEqual(window.shell_splitter.count(), 2)

        window.navigate_to_page("Preferences")
        self.app.processEvents()
        self.assertTrue(window.workspace_panel.isHidden())

        window.current_dataset = SimpleNamespace(display_name="sub-01_task-tms", source_path=Path("sub-01_task-tms_raw.fif"), pipeline=get_pipeline(None), stage_exists=lambda _: False)
        window.navigate_to_page("Home")
        self.app.processEvents()
        self.assertFalse(window.workspace_panel.isHidden())

        window.navigate_to_page("Preprocessing")
        self.app.processEvents()
        self.assertFalse(window.workspace_panel.isHidden())
        self.assertFalse(hasattr(window.page_lookup["Preprocessing"], "details_button"))
        self.assertNotIn("Toggle Derivative Inspector", [action.text() for action in window.view_menu.actions()])

        window.navigate_to_page("Real-Time")
        self.app.processEvents()
        self.assertTrue(window.workspace_panel.isHidden())
        window.close()

    def test_saved_step_names_and_info_buttons_inspect_their_own_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {stage: root / f"recording_{stage}.fif" for stage in ("raw", "epochs", "preprocessed")}
            for path in paths.values():
                path.touch()
            dataset = DatasetRecord(paths["raw"], get_pipeline(None), root, paths)
            window = MainWindow()
            self.addCleanup(window.close)
            page = window.page_lookup["Preprocessing"]
            with patch.object(page, "_ensure_preprocessor_loaded"), patch.object(page, "_refresh_plot_widgets"):
                page.preprocessor = Mock()
                page.current_dataset = dataset
                page.update_ui_state()
            with patch.object(window, "inspect_dataset_stage") as inspect:
                for stage_id in ("raw", "epochs", "preprocessed"):
                    link = page.stage_sections[stage_id].info_link
                    self.assertFalse(link.info_button.isHidden())
                    QTest.mouseClick(link.label, Qt.MouseButton.LeftButton)
                    inspect.assert_called_once_with(stage_id)
                    inspect.reset_mock()
                    link.info_button.click()
                    inspect.assert_called_once_with(stage_id)
                    inspect.reset_mock()
                    QTest.keyClick(link.label, Qt.Key.Key_Return)
                    inspect.assert_called_once_with(stage_id)
                    inspect.reset_mock()
                inspect.reset_mock()
                for stage_id in ("filtered_raw", "continuous_ica", "epochs_ica"):
                    link = page.stage_sections[stage_id].info_link
                    self.assertTrue(link.info_button.isHidden())
                    QTest.mouseClick(link.label, Qt.MouseButton.LeftButton)
                    QTest.keyClick(link.label, Qt.Key.Key_Return)
                inspect.assert_not_called()

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

    def test_preferences_categories_preserve_unsaved_edits(self):
        window = MainWindow()
        page = window.page_lookup["Preferences"]
        self.assertEqual(page.category_list.count(), 4)
        page.global_settings_widget.output_dir_input.setText("reviewed_outputs")
        page.category_list.setCurrentRow(2)
        self.assertEqual(page.category_stack.currentIndex(), 2)
        page.category_list.setCurrentRow(0)
        self.assertEqual(page.global_settings_widget.output_dir_input.text(), "reviewed_outputs")
        window.close()

    def test_preferences_categories_fill_the_remaining_page_height(self):
        window = MainWindow()
        window.resize(1200, 900)
        window.show()
        window.navigate_to_page("Preferences")
        self.app.processEvents()
        page = window.page_lookup["Preferences"]

        tool_box_index = page.content_layout.indexOf(page.settings_panel)
        self.assertEqual(page.content_layout.stretch(tool_box_index), 1)
        self.assertEqual(
            page.category_stack.sizePolicy().verticalPolicy(),
            QSizePolicy.Policy.Expanding,
        )

        bottom_margin = page.content_layout.contentsMargins().bottom()
        for category_index in range(page.category_stack.count()):
            page.category_list.setCurrentRow(category_index)
            self.app.processEvents()

            expected_bottom = page.content_container.height() - bottom_margin
            actual_bottom = page.settings_panel.geometry().bottom() + 1
            self.assertLessEqual(abs(actual_bottom - expected_bottom), 1)
            self.assertEqual(
                page.category_stack.currentWidget().sizePolicy().verticalPolicy(),
                QSizePolicy.Policy.Expanding,
            )
        window.close()

    def test_workspace_displays_and_wraps_the_exact_filename_stem(self):
        filename = "sub-0001_ses-followup_task-EyesOpen_acq-highdensity_run-03_recording-continuous_desc-original_raw.fif"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.touch()
            panel = WorkspacePanel()
            panel.set_context(workspace_root=Path(directory), pipeline=get_pipeline(None), output_root="derivatives")
            panel.resize(270, 600)
            panel.show()
            for width in (270, 240):
                panel.resize(width, 600)
                self.app.processEvents()
                row = panel.list_widget.itemWidget(panel.list_widget.item(0))
                self.assertEqual(row.title_label.text(), path.stem)
                self.assertEqual(row.title_label.toolTip(), str(path))
                self.assertGreaterEqual(row.title_label.height(), row.title_label.heightForWidth(row.title_label.width()))
                self.assertEqual(panel.list_widget.horizontalScrollBar().maximum(), 0)
            panel.close()

    def test_display_stems_preserve_internal_dots_and_remove_fif_compression_suffixes(self):
        dataset = SimpleNamespace(source_path=Path("subject.session.v2_raw.fif"))
        self.assertEqual(dataset_label(dataset), "subject.session.v2_raw")
        self.assertEqual(dataset_label(dataset, path=Path("subject.session.v2_desc-preprocessed_epo.fif.gz")), "subject.session.v2_desc-preprocessed_epo")

    def test_workspace_stays_visible_after_legacy_hidden_preference_and_resizing(self):
        original = MainWindow()
        original.settings_store.set("ui/main_window/workspace_visible", False)
        self.addCleanup(original.settings_store.remove, "ui/main_window/workspace_visible")
        original.close()
        window = MainWindow()
        self.addCleanup(window.close)
        window.show()
        for width in (1600, 1280, 1024):
            window.resize(width, 800)
            for page_name in ("Home", "Preprocessing", "Analysis"):
                window.navigate_to_page(page_name)
                self.app.processEvents()
                self.assertTrue(window.workspace_panel.isVisible(), page_name)
                self.assertGreaterEqual(window.workspace_panel.width(), window.workspace_panel.minimumWidth())
        self.assertNotIn("Toggle Datasets", [action.text() for action in window.view_menu.actions()])
        window.close()

    def test_preprocessing_renders_each_stage_heading_once(self):
        page = ProcessingPage()
        stage_titles = [section.title.text().split(". ", 1)[1] for section in page.stage_sections.values()]

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

    def test_empty_preprocessing_guides_dataset_selection(self):
        page = ProcessingPage()
        self.assertTrue(page.workflow_container.isHidden())
        self.assertFalse(page.empty_state.isHidden())
        self.assertEqual(page.choose_dataset_button.text(), "Open workspace")
        page.close()

    def test_preprocessing_operations_are_visible_and_run_on_one_click(self):
        calls = []
        def operation(page):
            calls.append(page)
        with patch.object(ProcessingPage, "filter_continuous", operation):
            page = ProcessingPage()
            page.current_dataset = SimpleNamespace(
                is_analysis_only=False, source_path=Path("recording_raw.fif"),
                stage_exists=lambda _: False, stage_status=lambda _: "pending",
            )
            page.preprocessor = Mock()
            with patch.object(page, "_ensure_preprocessor_loaded"), patch.object(page, "_is_action_available", return_value=True), patch.object(page, "_refresh_plot_widgets"):
                page.update_ui_state()
            page.resize(900, 800)
            page.show()
            self.app.processEvents()
            for button in page.action_buttons.values():
                self.assertTrue(button.isVisible(), button.text())
            page.action_buttons["filter_continuous"].click()
            self.assertEqual(calls, [page])
            page.close()

    def test_hiding_advanced_live_settings_preserves_parameters(self):
        widget = RealTimeMainWidget()
        widget.advanced_settings.set_expanded(True)
        widget.plot_settings_widget.refresh_rate_input.setValue(30)
        before = widget.plot_settings_widget.get_settings()
        widget.advanced_settings.set_expanded(False)
        widget.source_tabs.setCurrentIndex(1)
        widget.source_tabs.setCurrentIndex(0)
        self.assertEqual(widget.plot_settings_widget.get_settings(), before)
        widget.close()

    def test_loaded_analysis_fits_laptop_and_inspection_preserves_display_settings(self):
        apply_theme("dark")
        info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], 100, "eeg")
        info.set_montage("standard_1020")
        epochs = mne.EpochsArray(
            np.random.default_rng(1).normal(0, 1e-6, (2, 4, 51)), info,
            events=np.array([[10, 0, 1], [80, 0, 2]]),
            event_id={"Stimulus A": 1, "Stimulus B": 2}, tmin=-0.1,
        )
        original = epochs.get_data(copy=True)
        calls = []
        def compute_time_frequency(page):
            calls.append(page)
        with tempfile.TemporaryDirectory() as directory, patch.object(ErpAnalysisPage, "run_time_frequency", compute_time_frequency):
            path = Path(directory) / "sub-01_desc-preprocessed_epo.fif"
            epochs.save(path, overwrite=True)
            raw_path = Path(directory) / "sub-01_raw.fif"
            dataset = DatasetRecord(raw_path, get_pipeline(None), Path(directory), {"preprocessed": path})
            window = MainWindow()
            self.addCleanup(window.close)
            window.resize(1280, 800)
            window.show()
            window.navigate_to_page("Analysis")
            page = window.page_lookup["Analysis"]
            page.tep = SimpleNamespace(epochs=epochs, label="Subject 01", derivatives={})
            with patch.object(page, "_ensure_loaded"):
                window.set_current_dataset(dataset)
            self.app.processEvents()
            self.assertLessEqual(page.content_container.width(), page.scroll_area.viewport().width())
            plot = page.evoked_plot_widget
            plot_position = plot.canvas.mapTo(window, QPoint(0, 0))
            self.assertIn(path.stem, page.subtitle_label.text())
            self.assertNotIn(raw_path.stem, page.subtitle_label.text())
            self.assertNotIn(".fif", page.subtitle_label.text())
            self.assertEqual(page.subtitle_label.toolTip(), str(path))
            for button in (page.topoplot_button, page.psd_button, page.excitability_button, page.nf_button, page.tf_button, page.time_frequency_button, page.stc_button, page.source_estimate_button):
                self.assertTrue(button.isVisible(), button.text())
            page.tf_button.click()
            self.assertEqual(calls, [page])
            self.assertLess(plot_position.y(), 360)
            self.assertLessEqual(plot_position.y() + plot.canvas.height(), window.height())
            self.assertGreaterEqual(plot.canvas.axes.xaxis.label.get_window_extent(plot.canvas.get_renderer()).y0, 0)
            plot.event_selector.setCurrentIndex(plot.event_selector.findData(1))
            plot.baseline_toggle.click()
            plot.baseline_control.setValue((-0.1, 0))
            with patch.object(window, "inspect_dataset_stage") as inspect:
                QTest.mouseClick(page.subtitle_label, Qt.MouseButton.LeftButton)
                inspect.assert_called_once_with("preprocessed")
                page.subtitle_link.info_button.click()
                self.assertEqual(inspect.call_count, 2)
                QTest.keyClick(page.subtitle_label, Qt.Key.Key_Space)
                self.assertEqual(inspect.call_count, 3)
                self.assertTrue(window.workspace_panel.isVisible())
            self.app.processEvents()
            self.assertEqual(plot.event_selector.currentData(), 1)
            self.assertEqual(plot.baseline_control.value(), (-0.1, 0))
            np.testing.assert_array_equal(epochs.get_data(), original)
            self.assertLessEqual(page.content_container.width(), page.scroll_area.viewport().width())
            window.close()


if __name__ == "__main__":
    unittest.main()
