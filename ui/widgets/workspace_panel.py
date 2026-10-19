from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QEvent, QTimer, Qt, Signal, QSize
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QStyle,
    QStyleOptionProgressBar,
    QVBoxLayout,
    QWidget,
)

from core.pipelines import (
    DatasetRecord,
    PipelineDefinition,
    discover_analysis_datasets,
    discover_datasets,
)
from ui.widgets.layout_helpers import FilenameLabel, dataset_label
from utils import themed_svg_icon


def display_name_for_path(path: Path | None, *, strip_extensions: bool = True) -> str:
    if path is None:
        return ""

    label = path.name or str(path)
    suffixes = "".join(path.suffixes)
    if strip_extensions and suffixes and path.is_file():
        return label[: -len(suffixes)]
    return label


class DatasetProgressBar(QProgressBar):
    def __init__(
        self,
        dataset: DatasetRecord,
        parent: QWidget | None = None,
        *,
        display_path: Path | None = None,
    ):
        super().__init__(parent)
        self.dataset = dataset
        self.setObjectName("datasetProgress")
        self.setProperty("selected", False)
        self.setRange(0, 100)
        self.setValue(int(round(dataset.progress_fraction() * 100)))
        self.setTextVisible(False)
        self.setMinimumHeight(32)
        self.setToolTip(
            f"{display_path or dataset.source_path}\n"
            f"{dataset.completed_stage_count()} of {len(dataset.pipeline.stages)} stages satisfied"
        )

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def paintEvent(self, event):
        option = QStyleOptionProgressBar()
        self.initStyleOption(option)
        option.textVisible = False

        painter = QPainter(self)
        self.style().drawControl(QStyle.ControlElement.CE_ProgressBar, option, painter, self)
        if self.property("compact"):
            return

        label = QFontMetrics(self.font()).elidedText(
            self.dataset.display_name,
            Qt.TextElideMode.ElideRight,
            max(self.width() - 16, 80),
        )
        text_rect = self.rect().adjusted(10, 0, -10, 0)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)


class DatasetListItemWidget(QFrame):
    def __init__(
        self,
        dataset: DatasetRecord,
        parent: QWidget | None = None,
        *,
        display_path: Path | None = None,
    ):
        super().__init__(parent)
        self.dataset = dataset
        self.setObjectName("datasetListItem")
        self.setProperty("selected", False)
        self.setToolTip(str(display_path or dataset.source_path))
        self.progress_bar = DatasetProgressBar(
            dataset, self, display_path=display_path
        )
        self.progress_bar.setProperty("compact", True)
        self.progress_bar.setMinimumHeight(3)
        self.progress_bar.setMaximumHeight(3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        self.title_label = FilenameLabel(dataset_label(dataset, path=display_path))
        self.title_label.setObjectName("datasetTitle")
        self.title_label.setToolTip(str(display_path or dataset.source_path))
        layout.addWidget(self.title_label)
        status = QLabel("Ready for analysis" if dataset.stage_exists(dataset.pipeline.analysis_ready_stage) else dataset.status_text())
        status.setObjectName("mutedLabel")
        layout.addWidget(status)
        layout.addWidget(self.progress_bar)

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.progress_bar.set_selected(selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class DatasetListWidget(QListWidget):
    """Keep each complete filename visible as the dataset browser is resized."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.viewport() and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self.update_row_heights)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_row_heights()

    def update_row_heights(self):
        for index in range(self.count()):
            item = self.item(index)
            widget = self.itemWidget(item)
            if widget is not None:
                widget.ensurePolished()
                widget.layout().invalidate()
                height = widget.layout().totalHeightForWidth(max(1, self.viewport().width() - 4))
                item_margins = max(0, self.visualItemRect(item).height() - widget.height())
                size = QSize(0, max(78, height + item_margins))
                if size != item.sizeHint():
                    item.setSizeHint(size)


class WorkspacePanel(QFrame):
    dataset_selected = Signal(object)
    folder_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("workspacePanel")

        self.pipeline: PipelineDefinition | None = None
        self.workspace_root: Path | None = None
        self.output_root = "derivatives"
        self.datasets: list[DatasetRecord] = []
        self.current_dataset: DatasetRecord | None = None
        self.analysis_mode = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Workspace")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.pipeline_badge = QLabel("No pipeline")
        self.pipeline_badge.setObjectName("pipelineBadge")
        layout.addWidget(self.pipeline_badge, 0, Qt.AlignmentFlag.AlignLeft)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)

        self.select_folder_button = QPushButton("Open Workspace")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("secondaryButton")
        controls_row.addWidget(self.select_folder_button, 1)
        controls_row.addWidget(self.refresh_button)
        layout.addLayout(controls_row)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Find a dataset…")
        layout.addWidget(self.filter_edit)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        self.folder_label = QLabel("No workspace selected.")
        self.folder_label.setObjectName("panelSubtitle")
        self.folder_label.setWordWrap(True)
        hbox.addWidget(self.folder_label)

        self.count_label = QLabel("(waiting...)")
        self.count_label.setObjectName("mutedLabel")
        hbox.addWidget(self.count_label)
        layout.addLayout(hbox)

        self.list_widget = DatasetListWidget()
        self.list_widget.setObjectName("workspaceList")
        layout.addWidget(self.list_widget, 1)

        self.select_folder_button.clicked.connect(self.select_folder)
        self.refresh_button.clicked.connect(self.refresh)
        self.filter_edit.textChanged.connect(self.populate_list)
        self.list_widget.currentItemChanged.connect(self._on_item_changed)

    def set_context(
        self,
        *,
        workspace_root: Path | None,
        pipeline: PipelineDefinition,
        output_root: str,
    ):
        self.workspace_root = workspace_root
        self.pipeline = pipeline
        self.output_root = output_root
        self.pipeline_badge.setText(f"{pipeline.name} pipeline")
        folder_label = display_name_for_path(workspace_root, strip_extensions=False) if workspace_root else "No workspace selected."
        self.folder_label.setText(folder_label)
        self.folder_label.setToolTip(str(workspace_root) if workspace_root else "")
        self.refresh()

    def select_folder(self):
        base_dir = str(self.workspace_root) if self.workspace_root else ""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Workspace", base_dir)
        if folder_path:
            self.folder_selected.emit(folder_path)

    def set_analysis_mode(self, enabled: bool):
        enabled = bool(enabled)
        if self.analysis_mode == enabled:
            return
        self.analysis_mode = enabled
        self.refresh()

    def _catalog_path(self, dataset: DatasetRecord) -> Path:
        if self.analysis_mode:
            return dataset.paths[dataset.pipeline.analysis_ready_stage]
        return dataset.source_path

    def refresh(self):
        selected_source_path = (
            self._catalog_path(self.current_dataset) if self.current_dataset else None
        )

        if self.pipeline is None:
            self.datasets = []
        elif self.analysis_mode:
            self.datasets = discover_analysis_datasets(
                self.workspace_root, self.pipeline, self.output_root
            )
        else:
            self.datasets = discover_datasets(self.workspace_root, self.pipeline, self.output_root)

        self.populate_list()

        restored = bool(
            selected_source_path is not None
            and self._restore_selection(selected_source_path)
        )
        if not restored and self.list_widget.count():
            self.list_widget.setCurrentRow(0)
            dataset = self.list_widget.item(0).data(Qt.ItemDataRole.UserRole)
            if self.current_dataset is not dataset:
                self._set_current_dataset(dataset)
        elif not restored:
            self._set_current_dataset(None)

    def populate_list(self):
        filter_text = self.filter_edit.text().lower().strip()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        visible_count = 0
        for dataset in self.datasets:
            catalog_path = self._catalog_path(dataset)
            haystack = " ".join(
                [
                    dataset.display_name.lower(),
                    catalog_path.name.lower(),
                    str(dataset.relative_dir).lower(),
                ]
            )
            if filter_text and filter_text not in haystack:
                continue

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, dataset)
            item.setSizeHint(QSize(0, 78))
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(
                item, DatasetListItemWidget(dataset, display_path=catalog_path)
            )
            visible_count += 1

        self.count_label.setText(f"({visible_count} file{'s' if visible_count != 1 else ''})")
        self.list_widget.update_row_heights()
        self.list_widget.blockSignals(False)

        if visible_count == 0:
            self._set_current_dataset(None)

    def _restore_selection(self, source_path: Path) -> bool:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            dataset = item.data(Qt.ItemDataRole.UserRole)
            if dataset and self._catalog_path(dataset) == source_path:
                self.list_widget.setCurrentRow(row)
                if self.current_dataset is not dataset:
                    self._set_current_dataset(dataset)
                return True
        return False

    def _on_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None):
        previous_widget = self.list_widget.itemWidget(previous) if previous is not None else None
        current_widget = self.list_widget.itemWidget(current) if current is not None else None
        if isinstance(previous_widget, DatasetListItemWidget):
            previous_widget.set_selected(False)
        if isinstance(current_widget, DatasetListItemWidget):
            current_widget.set_selected(True)
        dataset = current.data(Qt.ItemDataRole.UserRole) if current else None
        self._set_current_dataset(dataset)

    def _set_current_dataset(self, dataset: DatasetRecord | None):
        self.current_dataset = dataset
        self.dataset_selected.emit(dataset)
