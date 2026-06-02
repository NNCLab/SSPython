from __future__ import annotations

from pathlib import Path

import mne
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
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
    QWidget,
)

from core.conversion import (
    ConversionEntry,
    convert_files,
    default_output_filename,
    discover_convertible_files,
    load_conversion_info,
    normalize_output_filename,
    validate_conversion_info,
    validate_source_for_info,
)
from ui.widgets.tools.merge_tool import MergeToolWidget
from utils import Worker


class ConversionWidget(QWidget):
    SourcePathRole = Qt.ItemDataRole.UserRole + 1

    def __init__(
        self,
        info: mne.Info | None = None,
        *,
        info_label: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.source_folder: Path | None = None
        self.conversion_info: mne.Info | None = None
        self.conversion_info_label: str | None = None

        self._setup_ui()
        self._setup_connections()
        self.set_conversion_info(info, label=info_label)

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

        self.info_edit = QLineEdit("No MNE info selected...")
        self.info_edit.setReadOnly(True)
        self.info_button = QPushButton("Select Info Source")

        settings_layout.addWidget(QLabel("MNE Info:"), 0, 0)
        settings_layout.addWidget(self.info_edit, 0, 1)
        settings_layout.addWidget(self.info_button, 0, 2)
        layout.addWidget(settings_group)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Convert", "Source", "Type", "Output File"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
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
        self.info_button.clicked.connect(self.select_info_source)
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

    def select_info_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select MNE Info Source",
            "",
            "MNE files (*.fif *.fif.gz);;All files (*)",
        )
        if not file_path:
            return

        try:
            info = load_conversion_info(Path(file_path))
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid MNE Info", str(exc))
            return

        self.set_conversion_info(info, label=Path(file_path).name)

    def populate_sources(self, source_paths: list[Path]):
        self.table.setRowCount(0)
        for row, source_path in enumerate(source_paths):
            self.table.insertRow(row)

            convert_item = self._check_item(checked=True)
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
            self.table.setItem(row, 1, source_item)
            self.table.setItem(row, 2, file_type_item)
            self.table.setItem(row, 3, output_item)

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
            entries = self._collect_entries()
            conversion_info = self._resolve_conversion_info()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Conversion Settings", str(exc))
            return

        if not self._ensure_info_ready(entries, conversion_info):
            return

        worker = Worker(
            lambda: convert_files(
                entries,
                info=conversion_info,
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
        QMessageBox.information(
            self,
            "Conversion Complete",
            f"Converted {len(converted_paths)} file(s).",
        )

    def _collect_entries(self) -> list[ConversionEntry]:
        if self.source_folder is None:
            raise ValueError("Select a source folder before converting files.")
        if self.table.rowCount() == 0:
            raise ValueError("No source files are available for conversion.")

        entries: list[ConversionEntry] = []
        seen_outputs: set[Path] = set()

        for row in range(self.table.rowCount()):
            convert_item = self.table.item(row, 0)
            source_item = self.table.item(row, 1)
            output_item = self.table.item(row, 3)
            if not all([convert_item, source_item, output_item]):
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

            entries.append(
                ConversionEntry(
                    source_path=source_path,
                    output_path=output_path,
                )
            )

        if not entries:
            raise ValueError("Select at least one file to convert.")

        return entries

    def _resolve_conversion_info(self) -> mne.Info:
        if self.conversion_info is None:
            raise ValueError("Select an MNE info source before converting files.")
        return validate_conversion_info(self.conversion_info)

    def _ensure_info_ready(
        self,
        entries: list[ConversionEntry],
        conversion_info: mne.Info,
    ) -> bool:
        for entry in entries:
            try:
                validate_source_for_info(
                    entry.source_path,
                    info=conversion_info,
                )
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    "Incompatible Source",
                    f"{entry.source_path.name} cannot be converted with the selected MNE info:\n{exc}",
                )
                return False

        return True

    def set_conversion_info(
        self,
        info: mne.Info | None,
        *,
        label: str | None = None,
    ):
        if info is None:
            self.conversion_info = None
            self.conversion_info_label = None
            self.info_edit.setText("No MNE info selected...")
            self.info_edit.setToolTip("Conversion requires an mne.Info object with a montage.")
            return

        conversion_info = info.copy()
        validate_conversion_info(conversion_info)
        self.conversion_info = conversion_info
        self.conversion_info_label = label
        info_label = label or "Provided MNE Info"
        channel_count = len(self.conversion_info["ch_names"])
        self.info_edit.setText(
            f"{info_label} ({channel_count} channels, {self.conversion_info['sfreq']:g} Hz)"
        )
        self.info_edit.setToolTip("Conversion info includes a montage.")

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


class ConvertToolDialog(QDialog):
    def __init__(
        self,
        info: mne.Info | None = None,
        *,
        info_label: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Convert EEG Data")
        self.setMinimumSize(980, 720)

        layout = QVBoxLayout(self)
        self.conversion_widget = ConversionWidget(
            info,
            info_label=info_label,
            parent=self,
        )
        layout.addWidget(self.conversion_widget)


class MergeToolDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge FIF Data")
        self.setMinimumSize(980, 720)

        layout = QVBoxLayout(self)
        self.merge_widget = MergeToolWidget(self)
        layout.addWidget(self.merge_widget)


if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])
    dialog = ConvertToolDialog()
    dialog.show()
    app.exec()
