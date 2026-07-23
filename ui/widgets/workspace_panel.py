from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, Qt, Signal, QSize, QUrl
from PySide6.QtGui import QDesktopServices, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMenu,
    QProgressBar,
    QPushButton,
    QStyle,
    QStyleOptionProgressBar,
    QVBoxLayout,
    QWidget,
)

from core.pipelines import DatasetRecord, PipelineDefinition, discover_datasets
from ui.widgets.tools.object_info_widget import ObjectDetailsDialog
from ui.widgets.tools.stage_io import load_stage_object
from utils import Worker, themed_svg_icon


def display_name_for_path(path: Path | None, *, strip_extensions: bool = True) -> str:
    if path is None:
        return ""

    label = path.name or str(path)
    suffixes = "".join(path.suffixes)
    if strip_extensions and suffixes and path.is_file():
        return label[: -len(suffixes)]
    return label


class DatasetProgressBar(QProgressBar):
    def __init__(self, dataset: DatasetRecord, parent: QWidget | None = None):
        super().__init__(parent)
        self.dataset = dataset
        self.setObjectName("datasetProgress")
        self.setProperty("selected", False)
        self.setRange(0, 100)
        self.setValue(int(round(dataset.progress_fraction() * 100)))
        self.setTextVisible(False)
        self.setMinimumHeight(32)
        self.setToolTip(
            f"{dataset.raw_path}\n{dataset.completed_stage_count()} of {len(dataset.pipeline.stages)} stages satisfied"
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

        label = QFontMetrics(self.font()).elidedText(
            self.dataset.display_name,
            Qt.TextElideMode.ElideRight,
            max(self.width() - 16, 80),
        )
        text_rect = self.rect().adjusted(10, 0, -10, 0)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)


class DatasetListItemWidget(QFrame):
    def __init__(self, dataset: DatasetRecord, parent: QWidget | None = None):
        super().__init__(parent)
        self.dataset = dataset
        self.setObjectName("datasetListItem")
        self.setProperty("selected", False)
        self.setToolTip(str(dataset.raw_path))
        self.progress_bar = DatasetProgressBar(dataset, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.progress_bar)

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.progress_bar.set_selected(selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class DiscreteProgressWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.segment_layout = QHBoxLayout(self)
        self.segment_layout.setContentsMargins(0, 0, 0, 0)
        self.segment_layout.setSpacing(4)

    def set_dataset(self, dataset: DatasetRecord | None):
        while self.segment_layout.count():
            item = self.segment_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if dataset is None:
            return

        for stage in dataset.pipeline.stages:
            segment = QFrame()
            segment.setObjectName("progressSegment")
            status = dataset.stage_status(stage.id)
            segment.setProperty("state", status)
            segment.setToolTip(
                str(dataset.paths[stage.id]) if status == "complete" else f"{stage.label}\n{stage.description}"
            )
            segment.setMinimumHeight(8)
            self.segment_layout.addWidget(segment, 1)


class StageStatusRow(QFrame):
    activated = Signal(str)
    context_requested = Signal(str, object)

    def __init__(
        self,
        dataset: DatasetRecord,
        stage_id: str,
        title: str,
        detail_text: str,
        detail_tooltip: str,
        *,
        optional: bool,
        state: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.stage_id = stage_id
        self.setObjectName("stageStatusRow")
        self.setProperty("state", state)
        self.setProperty("interactive", True)
        self.setToolTip(detail_tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        row_layout = QVBoxLayout(self)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        bullet = QFrame()
        bullet.setObjectName("stageDot")
        bullet.setProperty("state", state)
        bullet.setFixedSize(8, 8)
        bullet.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_row.addWidget(bullet, 0, Qt.AlignmentFlag.AlignVCenter)

        title_label = QLabel(title)
        title_label.setObjectName("stageStatusLabel")
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_row.addWidget(title_label, 1)

        if optional:
            optional_tag = QLabel("Optional")
            optional_tag.setObjectName("optionalTag")
            optional_tag.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            header_row.addWidget(optional_tag, 0, Qt.AlignmentFlag.AlignRight)

        detail_label = QLabel(detail_text)
        detail_label.setObjectName("mutedLabel")
        detail_label.setWordWrap(True)
        detail_label.setToolTip(detail_tooltip)
        detail_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        row_layout.addLayout(header_row)
        row_layout.addWidget(detail_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.stage_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        self.context_requested.emit(self.stage_id, event.globalPos())
        event.accept()


class StageStatusList(QFrame):
    stage_clicked = Signal(str)
    stage_context_requested = Signal(str, object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("stageStatusList")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(6)

    def set_dataset(self, dataset: DatasetRecord | None):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if dataset is None:
            placeholder = QLabel("Select a dataset to inspect its derivatives.")
            placeholder.setObjectName("mutedLabel")
            placeholder.setWordWrap(True)
            self.main_layout.addWidget(placeholder)
            return

        for stage in dataset.pipeline.stages:
            status = dataset.stage_status(stage.id)
            if status == "complete":
                detail_text = dataset.display_file_name(stage.id)
                detail_tooltip = str(dataset.paths[stage.id])
            elif status == "skipped":
                detail_text = "Skipped"
                detail_tooltip = stage.description
            else:
                detail_text = "Pending"
                detail_tooltip = stage.description

            row = StageStatusRow(
                dataset,
                stage.id,
                stage.label,
                detail_text,
                detail_tooltip,
                optional=stage.optional,
                state=status,
                parent=self,
            )
            row.activated.connect(self.stage_clicked)
            row.context_requested.connect(self.stage_context_requested)
            self.main_layout.addWidget(row)

        self.main_layout.addStretch()


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
        self.filter_edit.setPlaceholderText("Filter datasets, subject IDs, or runs")
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

        self.list_widget = QListWidget()
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

    def refresh(self):
        selected_raw_path = self.current_dataset.raw_path if self.current_dataset else None

        if self.pipeline is None:
            self.datasets = []
        else:
            self.datasets = discover_datasets(self.workspace_root, self.pipeline, self.output_root)

        self.populate_list()

        if selected_raw_path is not None:
            self._restore_selection(selected_raw_path)
        elif self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        else:
            self._set_current_dataset(None)

    def populate_list(self):
        filter_text = self.filter_edit.text().lower().strip()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        visible_count = 0
        for dataset in self.datasets:
            haystack = " ".join(
                [
                    dataset.display_name.lower(),
                    dataset.raw_path.name.lower(),
                    str(dataset.relative_dir).lower(),
                ]
            )
            if filter_text and filter_text not in haystack:
                continue

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, dataset)
            item.setSizeHint(QSize(0, 40))
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, DatasetListItemWidget(dataset))
            visible_count += 1

        self.count_label.setText(f"({visible_count} file{'s' if visible_count != 1 else ''})")
        self.list_widget.blockSignals(False)

        if visible_count == 0:
            self._set_current_dataset(None)

    def _restore_selection(self, raw_path: Path):
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            dataset = item.data(Qt.ItemDataRole.UserRole)
            if dataset and dataset.raw_path == raw_path:
                self.list_widget.setCurrentRow(row)
                return

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


class DatasetInspectorPanel(QFrame):
    derivatives_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("inspectorPanel")
        self.current_dataset: DatasetRecord | None = None
        self.collapsed = False
        self.expanded_width = 300
        self.collapsed_width = 40
        self._build_ui()
        self._apply_collapsed_state(animated=False)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 18, 10, 18)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        self.title_label = QLabel("Derivatives")
        self.title_label.setObjectName("panelTitle")
        header_row.addWidget(self.title_label, 1)

        self.toggle_button = QPushButton("")
        self.toggle_button.setObjectName("inspectorToggle")
        self.toggle_button.clicked.connect(self.toggle_collapsed)
        header_row.addWidget(self.toggle_button)
        layout.addLayout(header_row)

        self.content_container = QWidget()
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        self.progress_caption = QLabel("Pipeline progress")
        self.progress_caption.setObjectName("mutedLabel")
        content_layout.addWidget(self.progress_caption)

        self.progress_widget = DiscreteProgressWidget()
        content_layout.addWidget(self.progress_widget)

        self.stage_status_list = StageStatusList()
        self.stage_status_list.stage_clicked.connect(self._open_stage_details)
        self.stage_status_list.stage_context_requested.connect(self._show_stage_context_menu)
        content_layout.addWidget(self.stage_status_list, 1)

        layout.addWidget(self.content_container, 1)
        self.refresh_icons()

    def refresh_icons(self):
        self.toggle_button.setIcon(themed_svg_icon("assets/icons/menu.svg", size=20))
        self.toggle_button.setIconSize(QSize(20, 20))
        self.toggle_button.setObjectName('sidebarToggle')

    def toggle_collapsed(self):
        self.collapsed = not self.collapsed
        self._apply_collapsed_state(animated=True)

    def _apply_collapsed_state(self, *, animated: bool):
        target_width = self.collapsed_width if self.collapsed else self.expanded_width
        if not self.collapsed:
            self.content_container.setVisible(True)
            self.title_label.setVisible(True)
        self.toggle_button.setToolTip("Expand derivative inspector" if self.collapsed else "Collapse derivative inspector")

        if not animated:
            self.setMinimumWidth(target_width)
            self.setMaximumWidth(target_width)
            self.content_container.setVisible(not self.collapsed)
            self.title_label.setVisible(not self.collapsed)
            return

        animation_group = QParallelAnimationGroup(self)
        for prop in (b"minimumWidth", b"maximumWidth"):
            animation = QPropertyAnimation(self, prop)
            animation.setDuration(160)
            animation.setStartValue(self.width())
            animation.setEndValue(target_width)
            animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
            animation_group.addAnimation(animation)
        animation_group.finished.connect(
            lambda: (
                self.content_container.setVisible(not self.collapsed),
                self.title_label.setVisible(not self.collapsed),
            )
        )
        animation_group.start()
        self._animation = animation_group

    def set_dataset(self, dataset: DatasetRecord | None):
        self.current_dataset = dataset

        if dataset is None:
            self.progress_widget.set_dataset(None)
            self.stage_status_list.set_dataset(None)
            return
        self.progress_widget.set_dataset(dataset)
        self.stage_status_list.set_dataset(dataset)

    def _open_stage_details(self, stage_id: str):
        if self.current_dataset is None:
            return

        stage = self.current_dataset.stage_definition(stage_id)
        status = self.current_dataset.stage_status(stage_id)

        if status != "complete":
            dialog = ObjectDetailsDialog(
                stage.label,
                None,
                status_text=status.capitalize(),
                description=stage.description,
                parent=self,
            )
            dialog.exec()
            return

        stage_path = self.current_dataset.paths[stage_id]
        try:
            worker = Worker(
                lambda: load_stage_object(stage_id, stage_path, stage.data_kind),
                parent=self,
                add_loggers="mne",
            )
            mne_object = worker.exec_with_dialog(
                "Loading derivative",
                f"Loading {stage.label.lower()}...",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Unable to load derivative", str(exc))
            return

        dialog = ObjectDetailsDialog(
            stage.label,
            mne_object,
            file_path=str(stage_path),
            status_text="Saved",
            description=stage.description,
            parent=self,
        )
        dialog.exec()

    def _show_stage_context_menu(self, stage_id: str, global_pos):
        if self.current_dataset is None:
            return

        stage_path = self.current_dataset.paths.get(stage_id)
        if stage_path is None or not stage_path.exists():
            return

        menu = QMenu(self)
        open_location_action = menu.addAction("Open File Location")

        delete_paths = self._existing_derivative_paths_from_stage(stage_id)
        delete_action = menu.addAction("Delete This and Subsequent Derivatives")
        delete_action.setEnabled(bool(delete_paths))

        selected_action = menu.exec(global_pos)
        if selected_action == open_location_action:
            self._open_file_location(stage_path)
        elif selected_action == delete_action:
            self._confirm_and_delete_derivatives(stage_id)

    def _open_file_location(self, file_path: Path):
        location = file_path.parent if file_path.is_file() else file_path
        if not location.exists():
            QMessageBox.warning(
                self,
                "Location not found",
                f"The file location does not exist:\n{location}",
            )
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(location)))
        if not opened:
            QMessageBox.warning(
                self,
                "Unable to open location",
                f"Could not open the file location:\n{location}",
            )

    def _existing_derivative_paths_from_stage(self, stage_id: str) -> list[Path]:
        if self.current_dataset is None or stage_id == "raw":
            return []

        try:
            start_index = self.current_dataset.pipeline.stage_index(stage_id)
        except ValueError:
            return []

        paths: list[Path] = []
        for stage in self.current_dataset.pipeline.stages[start_index:]:
            if stage.id == "raw":
                continue
            path = self.current_dataset.paths.get(stage.id)
            if path is not None and path.exists() and path.is_file():
                paths.append(path)
        return paths

    def _confirm_and_delete_derivatives(self, stage_id: str):
        delete_paths = self._existing_derivative_paths_from_stage(stage_id)
        if not delete_paths:
            return

        stage = self.current_dataset.stage_definition(stage_id) if self.current_dataset else None
        stage_label = stage.label if stage else "selected stage"
        file_word = "file" if len(delete_paths) == 1 else "files"
        preview = "\n".join(path.name for path in delete_paths[:5])
        if len(delete_paths) > 5:
            preview += f"\n...and {len(delete_paths) - 5} more"

        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setWindowTitle("Delete derivatives?")
        message_box.setText(
            f"Delete {len(delete_paths)} derivative {file_word} from {stage_label} onward?"
        )
        message_box.setInformativeText(
            f"{preview}\n\nThis cannot be undone."
        )
        message_box.setDetailedText("\n".join(str(path) for path in delete_paths))
        delete_button = message_box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        message_box.setDefaultButton(cancel_button)
        message_box.exec()

        if message_box.clickedButton() != delete_button:
            return

        errors = self._delete_derivative_paths(delete_paths)
        self.set_dataset(self.current_dataset)
        self.derivatives_changed.emit()

        if errors:
            error_text = "\n".join(f"{path}: {error}" for path, error in errors)
            QMessageBox.warning(
                self,
                "Unable to delete some derivatives",
                error_text,
            )

    def _delete_derivative_paths(self, delete_paths: list[Path]) -> list[tuple[Path, str]]:
        errors: list[tuple[Path, str]] = []
        for path in delete_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                errors.append((path, str(exc)))
        return errors
