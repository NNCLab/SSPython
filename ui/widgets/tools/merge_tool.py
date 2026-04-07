from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.merge import (
    MergeEntry,
    merge_fif_files,
    read_annotation_descriptions,
    validate_merge_entries,
)
from utils import Worker


@dataclass
class MergeFileState:
    source_path: Path
    annotations: list[str] = field(default_factory=list)
    annotation_renames: dict[str, str] = field(default_factory=dict)


class AnnotationRenameDialog(QDialog):
    def __init__(
        self,
        *,
        source_path: Path,
        annotations: list[str],
        annotation_renames: dict[str, str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Rename Annotations: {source_path.name}")
        self.resize(640, 420)

        self.annotations = list(annotations)
        self.annotation_renames = dict(annotation_renames or {})

        layout = QVBoxLayout(self)
        description_label = QLabel(
            "Rename annotation labels before merging. "
            "Final labels must stay unique within this file."
        )
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        self.table = QTableWidget(len(self.annotations), 2)
        self.table.setHorizontalHeaderLabels(["Original", "Merged Label"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        for row, label in enumerate(self.annotations):
            original_item = QTableWidgetItem(label)
            original_item.setFlags(original_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            merged_item = QTableWidgetItem(self.annotation_renames.get(label, label))
            self.table.setItem(row, 0, original_item)
            self.table.setItem(row, 1, merged_item)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_annotation_renames(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for row, original in enumerate(self.annotations):
            merged_item = self.table.item(row, 1)
            mapping[original] = merged_item.text().strip() if merged_item else original
        return mapping

    def accept(self):
        renamed_labels = [label.strip() for label in self.get_annotation_renames().values()]
        if any(not label for label in renamed_labels):
            QMessageBox.warning(self, "Invalid Annotation Name", "Annotation names cannot be empty.")
            return
        if len(set(renamed_labels)) != len(renamed_labels):
            QMessageBox.warning(
                self,
                "Duplicate Annotation Name",
                "Merged annotation labels must remain unique within the file.",
            )
            return
        super().accept()


class MergeToolWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._files: list[MergeFileState] = []
        self._setup_ui()
        self._refresh_table()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        files_group = QGroupBox("FIF Files to Merge")
        files_layout = QVBoxLayout(files_group)

        controls_row = QHBoxLayout()
        self.add_button = QPushButton("Add FIF Files")
        self.remove_button = QPushButton("Remove Selected")
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        controls_row.addWidget(self.add_button)
        controls_row.addWidget(self.remove_button)
        controls_row.addWidget(self.move_up_button)
        controls_row.addWidget(self.move_down_button)
        controls_row.addStretch(1)
        files_layout.addLayout(controls_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Order", "File", "Annotations", "Edit"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        files_layout.addWidget(self.table)

        layout.addWidget(files_group, 1)

        output_group = QGroupBox("Merge Output")
        output_layout = QGridLayout(output_group)
        self.output_edit = QLineEdit()
        self.output_browse_button = QPushButton("Browse...")
        output_layout.addWidget(QLabel("Output FIF:"), 0, 0)
        output_layout.addWidget(self.output_edit, 0, 1)
        output_layout.addWidget(self.output_browse_button, 0, 2)
        layout.addWidget(output_group)

        footer_row = QHBoxLayout()
        self.summary_label = QLabel("No FIF files selected.")
        self.merge_button = QPushButton("Merge FIF Files")
        footer_row.addWidget(self.summary_label, 1)
        footer_row.addWidget(self.merge_button)
        layout.addLayout(footer_row)

        self.add_button.clicked.connect(self.add_files)
        self.remove_button.clicked.connect(self.remove_selected_file)
        self.move_up_button.clicked.connect(self.move_selected_file_up)
        self.move_down_button.clicked.connect(self.move_selected_file_down)
        self.output_browse_button.clicked.connect(self.select_output_path)
        self.merge_button.clicked.connect(self.run_merge)

    def add_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select FIF Files to Merge",
            "",
            "FIF files (*.fif)",
        )
        if not file_paths:
            return

        duplicate_paths = {state.source_path for state in self._files}
        rejected: list[str] = []
        for file_path in file_paths:
            source_path = Path(file_path)
            if source_path in duplicate_paths:
                continue
            try:
                annotations = read_annotation_descriptions(source_path)
            except ValueError as exc:
                rejected.append(str(exc))
                continue

            self._files.append(
                MergeFileState(
                    source_path=source_path,
                    annotations=annotations,
                )
            )
            duplicate_paths.add(source_path)

        if not self.output_edit.text().strip() and self._files:
            self.output_edit.setText(str(self._default_output_path()))

        self._refresh_table()
        if rejected:
            QMessageBox.warning(self, "Some Files Were Skipped", "\n".join(rejected))

    def remove_selected_file(self):
        row = self.table.currentRow()
        if row < 0:
            return
        del self._files[row]
        self._refresh_table()

    def move_selected_file_up(self):
        row = self.table.currentRow()
        if row <= 0:
            return
        self._files[row - 1], self._files[row] = self._files[row], self._files[row - 1]
        self._refresh_table(selected_row=row - 1)

    def move_selected_file_down(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._files) - 1:
            return
        self._files[row + 1], self._files[row] = self._files[row], self._files[row + 1]
        self._refresh_table(selected_row=row + 1)

    def select_output_path(self):
        suggested_path = self.output_edit.text().strip() or str(self._default_output_path())
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Merge Output FIF",
            suggested_path,
            "FIF files (*.fif)",
        )
        if not file_path:
            return
        output_path = Path(file_path)
        if output_path.suffix.lower() != ".fif":
            output_path = output_path.with_suffix(".fif")
        self.output_edit.setText(str(output_path))

    def run_merge(self):
        try:
            entries, output_path = self._collect_entries()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Merge Settings", str(exc))
            return

        worker = Worker(
            lambda: merge_fif_files(entries, output_path),
            parent=self,
            add_loggers="mne",
        )
        result = worker.exec_with_dialog(
            "Merging FIF Files",
            "Merging selected FIF recordings...",
        )
        if result is None:
            return

        QMessageBox.information(
            self,
            "Merge Complete",
            f"Merged file saved to:\n{result}",
        )

    def _collect_entries(self) -> tuple[list[MergeEntry], Path]:
        if len(self._files) < 2:
            raise ValueError("Select at least two FIF files to merge.")

        output_text = self.output_edit.text().strip()
        output_path = Path(output_text) if output_text else self._default_output_path()
        if output_path.suffix.lower() != ".fif":
            output_path = output_path.with_suffix(".fif")
            self.output_edit.setText(str(output_path))

        source_paths = {state.source_path.resolve() for state in self._files}
        if output_path.resolve() in source_paths:
            raise ValueError("The merge output path cannot overwrite one of the source files.")

        entries = [
            MergeEntry(
                source_path=state.source_path,
                annotation_renames=state.annotation_renames,
            )
            for state in self._files
        ]
        validate_merge_entries(entries)
        return entries, output_path

    def _refresh_table(self, *, selected_row: int | None = None):
        self.table.setRowCount(0)
        for row, state in enumerate(self._files):
            self.table.insertRow(row)

            order_item = QTableWidgetItem(str(row + 1))
            order_item.setFlags(order_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            file_item = QTableWidgetItem(str(state.source_path))
            file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            file_item.setToolTip(str(state.source_path))
            annotation_item = QTableWidgetItem(self._annotation_summary(state))
            annotation_item.setFlags(annotation_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            edit_button = QPushButton("Edit...")
            edit_button.clicked.connect(
                lambda checked=False, row_index=row: self.edit_annotations(row_index)
            )

            self.table.setItem(row, 0, order_item)
            self.table.setItem(row, 1, file_item)
            self.table.setItem(row, 2, annotation_item)
            self.table.setCellWidget(row, 3, edit_button)

        if selected_row is not None and 0 <= selected_row < self.table.rowCount():
            self.table.selectRow(selected_row)
        elif self.table.rowCount() > 0:
            self.table.selectRow(0)

        file_count = len(self._files)
        if file_count == 0:
            self.summary_label.setText("No FIF files selected.")
        else:
            self.summary_label.setText(f"{file_count} FIF file(s) ready for merge.")

    def edit_annotations(self, row: int):
        if row < 0 or row >= len(self._files):
            return

        state = self._files[row]
        if not state.annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                f"{state.source_path.name} does not contain annotations to rename.",
            )
            return

        dialog = AnnotationRenameDialog(
            source_path=state.source_path,
            annotations=state.annotations,
            annotation_renames=state.annotation_renames,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        state.annotation_renames = dialog.get_annotation_renames()
        self._refresh_table(selected_row=row)

    def _annotation_summary(self, state: MergeFileState) -> str:
        if not state.annotations:
            return "No annotations"

        rendered_labels = [
            state.annotation_renames.get(label, label)
            for label in state.annotations
        ]
        preview = ", ".join(rendered_labels[:3])
        if len(rendered_labels) > 3:
            preview += ", ..."
        return preview

    def _default_output_path(self) -> Path:
        if self._files:
            return self._files[0].source_path.parent / "merged_raw.fif"
        return Path.cwd() / "merged_raw.fif"
