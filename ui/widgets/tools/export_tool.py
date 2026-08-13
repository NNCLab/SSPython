from __future__ import annotations

from pathlib import Path
from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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
)

from core.exporting import evoked_filename


ExportMode = Literal["preprocessed", "evoked"]


class ExportFilesDialog(QDialog):
    SourcePathRole = Qt.ItemDataRole.UserRole + 1

    def __init__(
        self,
        source_paths: list[Path],
        workspace_root: Path,
        mode: ExportMode,
        parent=None,
    ):
        super().__init__(parent)
        if mode not in {"preprocessed", "evoked"}:
            raise ValueError(f"Unsupported export mode: {mode}")

        self.workspace_root = Path(workspace_root)
        self.mode = mode
        self.source_paths = [Path(path) for path in source_paths]

        export_name = "Preprocessed Files" if mode == "preprocessed" else "Evoked Files"
        self.setWindowTitle(f"Export {export_name}")
        self.setMinimumSize(820, 520)

        self._build_ui(export_name)
        self._populate_files()

    def _build_ui(self, export_name: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        heading = QLabel(f"Export {export_name}")
        heading.setObjectName("panelTitle")
        layout.addWidget(heading)

        description = QLabel(
            "Choose which preprocessed epochs files from the active workspace to export."
        )
        description.setObjectName("panelSubtitle")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Export", "Workspace file", "Export filename"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.itemChanged.connect(self._update_summary)
        layout.addWidget(self.table, 1)

        selection_row = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setObjectName("mutedLabel")
        selection_row.addWidget(self.summary_label, 1)
        self.select_all_button = QPushButton("Select All")
        self.select_none_button = QPushButton("Select None")
        self.select_all_button.setObjectName("secondaryButton")
        self.select_none_button.setObjectName("secondaryButton")
        self.select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.select_none_button.clicked.connect(lambda: self._set_all_checked(False))
        selection_row.addWidget(self.select_all_button)
        selection_row.addWidget(self.select_none_button)
        layout.addLayout(selection_row)

        destination_group = QGroupBox("Export Folder")
        destination_layout = QHBoxLayout(destination_group)
        self.destination_edit = QLineEdit()
        self.destination_edit.setPlaceholderText("Choose a destination folder...")
        self.destination_edit.setReadOnly(True)
        self.destination_button = QPushButton("Browse...")
        self.destination_button.clicked.connect(self._browse_destination)
        destination_layout.addWidget(self.destination_edit, 1)
        destination_layout.addWidget(self.destination_button)
        layout.addWidget(destination_group)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.export_button = self.button_box.addButton(
            f"Export {export_name}",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.export_button.setDefault(True)
        self.button_box.rejected.connect(self.reject)
        self.export_button.clicked.connect(self._accept_if_valid)
        layout.addWidget(self.button_box)

    def _populate_files(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row, source_path in enumerate(self.source_paths):
            self.table.insertRow(row)

            export_item = QTableWidgetItem()
            export_item.setFlags(
                (export_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsEditable
            )
            export_item.setCheckState(Qt.CheckState.Checked)
            export_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            try:
                display_path = source_path.relative_to(self.workspace_root)
            except ValueError:
                display_path = source_path
            source_item = QTableWidgetItem(str(display_path))
            source_item.setData(self.SourcePathRole, source_path)
            source_item.setToolTip(str(source_path))

            output_name = (
                source_path.name
                if self.mode == "preprocessed"
                else evoked_filename(source_path)
            )
            output_item = QTableWidgetItem(output_name)

            self.table.setItem(row, 0, export_item)
            self.table.setItem(row, 1, source_item)
            self.table.setItem(row, 2, output_item)

        self.table.blockSignals(False)
        self._update_summary()

    def _set_all_checked(self, checked: bool):
        check_state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(check_state)
        self.table.blockSignals(False)
        self._update_summary()

    def _update_summary(self):
        selected_count = len(self.selected_source_paths())
        total_count = self.table.rowCount()
        self.summary_label.setText(
            f"{selected_count} of {total_count} file{'s' if total_count != 1 else ''} selected."
        )

    def _browse_destination(self):
        current_value = self.destination_edit.text().strip()
        base_directory = current_value or str(self.workspace_root)
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            base_directory,
        )
        if folder:
            self.destination_edit.setText(folder)

    def selected_source_paths(self) -> list[Path]:
        selected_paths: list[Path] = []
        for row in range(self.table.rowCount()):
            export_item = self.table.item(row, 0)
            source_item = self.table.item(row, 1)
            if export_item.checkState() == Qt.CheckState.Checked:
                selected_paths.append(Path(source_item.data(self.SourcePathRole)))
        return selected_paths

    def destination_path(self) -> Path | None:
        destination = self.destination_edit.text().strip()
        return Path(destination) if destination else None

    def _accept_if_valid(self):
        if not self.selected_source_paths():
            QMessageBox.warning(
                self,
                "No files selected",
                "Select at least one file to export.",
            )
            return

        destination = self.destination_path()
        if destination is None:
            QMessageBox.warning(
                self,
                "No export folder selected",
                "Choose an export folder before continuing.",
            )
            return
        if destination.exists() and not destination.is_dir():
            QMessageBox.warning(
                self,
                "Invalid export folder",
                f"The selected path is not a folder:\n{destination}",
            )
            return

        self.accept()
