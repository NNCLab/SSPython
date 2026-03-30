from __future__ import annotations

from pathlib import Path

import mne
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
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
)

from core.conversion import (
    ConversionEntry,
    convert_files,
    default_output_filename,
    discover_convertible_files,
    normalize_output_filename,
)
from utils import Worker


class ConversionToolDialog(QDialog):
    BuiltinMontagePlaceholder = "Select montage..."
    CustomMontageLabel = "Custom montage file..."
    SourcePathRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Convert and Merge EEG Data")
        self.setMinimumSize(900, 640)
        self.source_folder: Path | None = None
        self.custom_montage_path: Path | None = None
        self.mat_channel_info_path: Path | None = None

        self._setup_ui()
        self._setup_connections()
        self._on_montage_selection_changed()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        source_group = QGroupBox("Source Folder")
        source_layout = QHBoxLayout(source_group)
        self.source_folder_edit = QLineEdit("No folder selected...")
        self.source_folder_edit.setReadOnly(True)
        self.source_folder_button = QPushButton("Select Folder")
        source_layout.addWidget(self.source_folder_edit, 1)
        source_layout.addWidget(self.source_folder_button)
        layout.addWidget(source_group)

        settings_group = QGroupBox("Conversion Settings")
        settings_layout = QGridLayout(settings_group)

        self.montage_combo = QComboBox()
        self.montage_combo.addItem(self.BuiltinMontagePlaceholder, None)
        for montage_name in sorted(mne.channels.get_builtin_montages()):
            self.montage_combo.addItem(montage_name, montage_name)
        self.montage_combo.addItem(self.CustomMontageLabel, self.CustomMontageLabel)

        self.custom_montage_edit = QLineEdit("No custom montage selected...")
        self.custom_montage_edit.setReadOnly(True)
        self.custom_montage_button = QPushButton("Browse...")

        self.mat_info_edit = QLineEdit("Optional unless .mat files are selected...")
        self.mat_info_edit.setReadOnly(True)
        self.mat_info_button = QPushButton("Select JSON")

        self.merge_name_edit = QLineEdit()
        self.merge_name_edit.setPlaceholderText("merged_recording_raw.fif")

        settings_layout.addWidget(QLabel("Montage:"), 0, 0)
        settings_layout.addWidget(self.montage_combo, 0, 1, 1, 2)
        settings_layout.addWidget(self.custom_montage_edit, 1, 1)
        settings_layout.addWidget(self.custom_montage_button, 1, 2)
        settings_layout.addWidget(QLabel("MAT Channel Info:"), 2, 0)
        settings_layout.addWidget(self.mat_info_edit, 2, 1)
        settings_layout.addWidget(self.mat_info_button, 2, 2)
        settings_layout.addWidget(QLabel("Merged Output Name:"), 3, 0)
        settings_layout.addWidget(self.merge_name_edit, 3, 1, 1, 2)
        layout.addWidget(settings_group)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Convert", "Merge", "Source", "Type", "Output File"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        footer_row = QHBoxLayout()
        self.summary_label = QLabel("No source folder loaded.")
        footer_row.addWidget(self.summary_label, 1)
        self.convert_button = QPushButton("Convert Selected Files")
        self.convert_button.setDefault(True)
        footer_row.addWidget(self.convert_button)
        layout.addLayout(footer_row)

    def _setup_connections(self):
        self.source_folder_button.clicked.connect(self.select_source_folder)
        self.montage_combo.currentIndexChanged.connect(
            self._on_montage_selection_changed
        )
        self.custom_montage_button.clicked.connect(self.select_custom_montage)
        self.mat_info_button.clicked.connect(self.select_mat_channel_info)
        self.convert_button.clicked.connect(self.run_conversion)

    def select_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Scan")
        if not folder:
            return

        self.source_folder = Path(folder)
        self.source_folder_edit.setText(str(self.source_folder))
        self.source_folder_edit.setToolTip(str(self.source_folder))

        worker = Worker(
            lambda: discover_convertible_files(self.source_folder),
            parent=self,
        )
        supported_files = worker.exec_with_dialog(
            "Scanning Folder",
            "Checking files that can be loaded by MNE...",
        )
        if supported_files is None:
            return
        self.populate_sources(supported_files)

    def select_custom_montage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Montage",
            "",
            "Montage files (*.fif *.loc *.locs *.elc *.csd *.sfp *.elp *.hpts *.txt);;All files (*)",
        )
        if not file_path:
            return

        self.custom_montage_path = Path(file_path)
        self.custom_montage_edit.setText(self.custom_montage_path.name)
        self.custom_montage_edit.setToolTip(str(self.custom_montage_path))

    def select_mat_channel_info(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select MAT Channel Info JSON",
            "",
            "JSON files (*.json)",
        )
        if not file_path:
            return

        self.mat_channel_info_path = Path(file_path)
        self.mat_info_edit.setText(self.mat_channel_info_path.name)
        self.mat_info_edit.setToolTip(str(self.mat_channel_info_path))

    def populate_sources(self, source_paths: list[Path]):
        self.table.setRowCount(0)
        for row, source_path in enumerate(source_paths):
            self.table.insertRow(row)

            convert_item = self._check_item(checked=True)
            merge_item = self._check_item(checked=False)
            source_item = QTableWidgetItem(
                str(source_path.relative_to(self.source_folder))
                if self.source_folder is not None
                else source_path.name
            )
            source_item.setData(self.SourcePathRole, source_path)
            source_item.setToolTip(str(source_path))
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            file_type_item = QTableWidgetItem(source_path.suffix.lower().lstrip(".") or "file")
            file_type_item.setFlags(file_type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            output_item = QTableWidgetItem(default_output_filename(source_path))

            self.table.setItem(row, 0, convert_item)
            self.table.setItem(row, 1, merge_item)
            self.table.setItem(row, 2, source_item)
            self.table.setItem(row, 3, file_type_item)
            self.table.setItem(row, 4, output_item)

        if source_paths:
            self.summary_label.setText(f"Found {len(source_paths)} convertible file(s).")
        else:
            self.summary_label.setText("No supported files found in the selected folder.")
            QMessageBox.information(
                self,
                "No Supported Files",
                "No files in the selected folder could be loaded with MNE or as .mat files.",
            )

    def run_conversion(self):
        try:
            entries, merged_output_path = self._collect_entries()
            builtin_montage, custom_montage_path = self._resolve_montage_selection()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid Conversion Settings", str(exc))
            return

        worker = Worker(
            lambda: convert_files(
                entries,
                builtin_montage=builtin_montage,
                custom_montage_path=custom_montage_path,
                mat_channel_info_path=self.mat_channel_info_path,
                merged_output_path=merged_output_path,
            ),
            parent=self,
            add_loggers="mne",
        )
        result = worker.exec_with_dialog(
            "Converting Files",
            "Converting selected recordings to FIF...",
        )
        if result is None:
            return

        converted_paths = result.get("converted", [])
        merged_path = result.get("merged")
        message_lines = [
            f"Converted {len(converted_paths)} file(s).",
        ]
        if merged_path is not None:
            message_lines.append(f"Merged file saved to:\n{merged_path}")
        else:
            message_lines.append("No merged output was requested.")
        QMessageBox.information(self, "Conversion Complete", "\n\n".join(message_lines))

    def _collect_entries(self) -> tuple[list[ConversionEntry], Path | None]:
        if self.source_folder is None:
            raise ValueError("Select a source folder before converting files.")
        if self.table.rowCount() == 0:
            raise ValueError("No source files are available for conversion.")

        entries: list[ConversionEntry] = []
        merge_count = 0
        seen_outputs: set[Path] = set()
        requires_mat_info = False

        for row in range(self.table.rowCount()):
            convert_item = self.table.item(row, 0)
            merge_item = self.table.item(row, 1)
            source_item = self.table.item(row, 2)
            output_item = self.table.item(row, 4)
            if not all([convert_item, merge_item, source_item, output_item]):
                continue

            if convert_item.checkState() != Qt.CheckState.Checked:
                continue

            source_path = source_item.data(self.SourcePathRole)
            if not source_path:
                continue
            source_path = Path(source_path)
            output_name = normalize_output_filename(output_item.text())
            output_item.setText(output_name)
            output_path = source_path.parent / output_name
            if output_path in seen_outputs:
                raise ValueError(f"Duplicate output file name detected: {output_name}")
            seen_outputs.add(output_path)

            should_merge = merge_item.checkState() == Qt.CheckState.Checked
            if should_merge:
                merge_count += 1
            if source_path.suffix.lower() == ".mat":
                requires_mat_info = True

            entries.append(
                ConversionEntry(
                    source_path=source_path,
                    output_path=output_path,
                    merge=should_merge,
                )
            )

        if not entries:
            raise ValueError("Select at least one file to convert.")
        if requires_mat_info and self.mat_channel_info_path is None:
            raise ValueError(
                "At least one selected file is a .mat file. Please provide the MAT channel info JSON."
            )

        merged_output_path = None
        if merge_count > 0:
            if merge_count < 2:
                raise ValueError("Select at least two files to create a merged output.")
            merge_name = normalize_output_filename(self.merge_name_edit.text() or "merged")
            self.merge_name_edit.setText(merge_name)
            merged_output_path = self.source_folder / merge_name
            if merged_output_path in seen_outputs:
                raise ValueError(
                    "The merged output name conflicts with an individual converted file."
                )

        return entries, merged_output_path

    def _resolve_montage_selection(self) -> tuple[str | None, Path | None]:
        montage_data = self.montage_combo.currentData()
        if not montage_data:
            raise ValueError("Select a montage before converting files.")
        if montage_data == self.CustomMontageLabel:
            if self.custom_montage_path is None:
                raise ValueError("Select a custom montage file before converting files.")
            return None, self.custom_montage_path
        return str(montage_data), None

    def _on_montage_selection_changed(self):
        is_custom = self.montage_combo.currentData() == self.CustomMontageLabel
        self.custom_montage_edit.setVisible(is_custom)
        self.custom_montage_button.setVisible(is_custom)

    @staticmethod
    def _check_item(*, checked: bool) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        return item


if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])
    dialog = ConversionToolDialog()
    dialog.show()
    app.exec()
