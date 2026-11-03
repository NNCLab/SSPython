import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.pipelines import discover_datasets, get_pipeline
from ui.widgets.workspace_panel import DatasetListItemWidget, WorkspacePanel


class TestWorkspaceProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_compact_progress_preserves_display_name_and_catalog_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "sub-01_task-tms_raw.fif"
            raw_path.touch()
            dataset = discover_datasets(root, get_pipeline("standard"), "derivatives")[0]
            catalog_path = dataset.paths[dataset.pipeline.analysis_ready_stage]
            widget = DatasetListItemWidget(dataset, display_path=catalog_path)

            self.assertEqual(widget.title_label.text(), dataset.display_name)
            self.assertEqual(widget.title_label.toolTip(), str(catalog_path))
            self.assertEqual(widget.toolTip(), str(catalog_path))
            self.assertIn(str(catalog_path), widget.progress_bar.toolTip())
            self.assertEqual(
                widget.progress_bar.value(), round(dataset.progress_fraction() * 100)
            )
            widget.close()

    def test_long_dataset_name_keeps_all_lines_visible_after_resize(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ("sub-01_" + "long_session_name_" * 8 + "raw.fif")).touch()
            panel = WorkspacePanel()
            panel.resize(440, 800)
            panel.set_context(
                workspace_root=root,
                pipeline=get_pipeline("standard"),
                output_root="derivatives",
            )
            panel.show()
            self.app.processEvents()
            self.app.processEvents()
            item = panel.list_widget.item(0)
            wide_height = item.sizeHint().height()

            panel.resize(250, 800)
            self.app.processEvents()
            self.app.processEvents()
            widget = panel.list_widget.itemWidget(item)
            required_height = widget.layout().totalHeightForWidth(widget.width())

            self.assertGreater(item.sizeHint().height(), wide_height)
            self.assertGreaterEqual(widget.height(), required_height)
            self.assertEqual(widget.title_label.text(), panel.current_dataset.display_name)
            panel.close()

    def test_rows_only_reserve_space_for_visible_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = get_pipeline("standard")
            for subject in ("01", "02"):
                (root / f"sub-{subject}_task-tms_raw.fif").touch()
            ready_dataset = discover_datasets(root, pipeline, "derivatives")[1]
            ready_path = ready_dataset.paths[pipeline.analysis_ready_stage]
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.touch()

            panel = WorkspacePanel()
            self.addCleanup(panel.close)
            panel.resize(440, 800)
            panel.set_context(
                workspace_root=root,
                pipeline=pipeline,
                output_root="derivatives",
            )
            panel.show()
            self.app.processEvents()
            self.app.processEvents()

            widgets = [
                panel.list_widget.itemWidget(panel.list_widget.item(index))
                for index in range(panel.list_widget.count())
            ]
            self.assertLess(widgets[0].height(), widgets[1].height())
            for widget in widgets:
                self.assertEqual(
                    widget.height(),
                    widget.layout().totalHeightForWidth(widget.width()),
                )


if __name__ == "__main__":
    unittest.main()
