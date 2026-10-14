from __future__ import annotations

import csv
from pathlib import Path

import mne
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.channel_info import COMMON_CHANNEL_TYPES, ChannelInfo
from core.info_creation import create_info, read_electrodes_tsv


class CreateInfoDialog(QDialog):
    def __init__(self, parent=None, *, output_directory: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Create INFO")
        self.resize(760, 680)
        self.output_directory = Path(output_directory) if output_directory else Path.cwd()
        self.info: mne.Info | None = None
        self.saved_path: Path | None = None

        layout = QVBoxLayout(self)
        description = QLabel(
            "Set the sampling frequency, edit channels in recording order, and choose "
            "their electrode positions. Save the INFO as a FIF file for conversion "
            "or real-time input."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        settings = QFormLayout()
        self.sfreq_input = QDoubleSpinBox()
        self.sfreq_input.setDecimals(3)
        self.sfreq_input.setRange(0.001, 10_000_000.0)
        self.sfreq_input.setValue(1000.0)
        self.sfreq_input.setSuffix(" Hz")
        settings.addRow("Sampling frequency:", self.sfreq_input)
        layout.addLayout(settings)

        montage_group = QGroupBox("Montage")
        montage_layout = QVBoxLayout(montage_group)
        self.montage_source_combo = QComboBox()
        self.montage_source_combo.addItems(["MNE built-in montage", "Electrodes TSV file"])
        montage_layout.addWidget(self.montage_source_combo)
        self.montage_stack = QStackedWidget()

        builtin_page = QWidget()
        builtin_layout = QFormLayout(builtin_page)
        builtin_layout.setContentsMargins(0, 0, 0, 0)
        self.builtin_montage_combo = QComboBox()
        self.builtin_montage_combo.addItems(mne.channels.get_builtin_montages())
        self.builtin_montage_combo.setCurrentText("standard_1020")
        builtin_layout.addRow("Montage:", self.builtin_montage_combo)
        self.montage_stack.addWidget(builtin_page)

        tsv_page = QWidget()
        tsv_layout = QFormLayout(tsv_page)
        tsv_layout.setContentsMargins(0, 0, 0, 0)
        file_row = QHBoxLayout()
        self.electrodes_path_input = QLineEdit()
        self.electrodes_path_input.setPlaceholderText("Select an electrodes.tsv file")
        self.electrodes_button = QPushButton("Browse...")
        file_row.addWidget(self.electrodes_path_input, 1)
        file_row.addWidget(self.electrodes_button)
        tsv_layout.addRow("Electrodes:", file_row)
        self.coordinate_unit_combo = QComboBox()
        self.coordinate_unit_combo.addItem("Meters (m)", "m")
        self.coordinate_unit_combo.addItem("Centimeters (cm)", "cm")
        self.coordinate_unit_combo.addItem("Millimeters (mm)", "mm")
        tsv_layout.addRow("Coordinate units:", self.coordinate_unit_combo)
        coordinate_note = QLabel(
            "Required columns: name, x, y, z (tab-separated). Coordinates must "
            "already be in MNE head space: x right, y forward, z up. "
            "Choose the units used by your file; coordsystem.json is not read. "
            "Rows with all three coordinates marked n/a are excluded."
        )
        coordinate_note.setWordWrap(True)
        tsv_layout.addRow(coordinate_note)
        self.montage_stack.addWidget(tsv_page)
        montage_layout.addWidget(self.montage_stack)
        layout.addWidget(montage_group)

        channels_group = QGroupBox("Channels (recording order)")
        channels_layout = QVBoxLayout(channels_group)
        actions = QHBoxLayout()
        self.add_channel_button = QPushButton("Add Channel")
        self.remove_channels_button = QPushButton("Remove Selected")
        self.add_montage_channels_button = QPushButton("Add Montage Channels")
        actions.addWidget(self.add_channel_button)
        actions.addWidget(self.remove_channels_button)
        actions.addStretch(1)
        actions.addWidget(self.add_montage_channels_button)
        channels_layout.addLayout(actions)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Channel Name", "Channel Type"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.table.setDragDropOverwriteMode(False)
        self.table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.table.setSupportedDragActions(Qt.DropAction.MoveAction)
        self.table.setDropIndicatorShown(True)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        channels_layout.addWidget(self.table, 1)
        channels_layout.addWidget(QLabel("Drag channel names to reorder. Double-click a name to edit it."))
        self.channel_count_label = QLabel()
        channels_layout.addWidget(self.channel_count_label)
        layout.addWidget(channels_group, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.create_button = self.button_box.addButton(
            "Create and Save INFO...", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.create_button.setDefault(True)
        layout.addWidget(self.button_box)

        self.montage_source_combo.currentIndexChanged.connect(self.montage_stack.setCurrentIndex)
        self.electrodes_button.clicked.connect(self._select_electrodes_file)
        self.add_channel_button.clicked.connect(self._add_channel)
        self.remove_channels_button.clicked.connect(self._remove_selected_channels)
        self.add_montage_channels_button.clicked.connect(self._add_montage_channels)
        self.table.itemSelectionChanged.connect(self._update_channel_state)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self._update_channel_state()

    def _append_channel(self, name: str, channel_type: str = "eeg"):
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsDropEnabled)
        self.table.setItem(row, 0, name_item)
        # Drops insert whole rows; neither column may be overwritten. An item
        # under the combo also lets Qt move its persistent widget with the row.
        type_item = QTableWidgetItem()
        type_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        self.table.setItem(row, 1, type_item)
        type_combo = QComboBox()
        type_combo.setEditable(True)
        type_combo.addItems(COMMON_CHANNEL_TYPES)
        type_combo.setCurrentText(channel_type)
        type_combo.setMinimumWidth(190)
        self.table.setCellWidget(row, 1, type_combo)
        self._update_channel_state()

    def _add_channel(self):
        self._append_channel("")
        item = self.table.item(self.table.rowCount() - 1, 0)
        self.table.setCurrentItem(item)
        self.table.editItem(item)

    def _remove_selected_channels(self):
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)
        self._update_channel_state()

    def _update_channel_state(self):
        self.channel_count_label.setText(f"{self.table.rowCount()} channel(s)")
        self.remove_channels_button.setEnabled(bool(self.table.selectionModel().selectedRows()))

    def _select_electrodes_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Electrodes TSV", str(self.output_directory), "TSV files (*.tsv)"
        )
        if path:
            self.electrodes_path_input.setText(path)

    def _resolve_montage(self) -> mne.channels.DigMontage:
        if self.montage_source_combo.currentIndex() == 0:
            return mne.channels.make_standard_montage(self.builtin_montage_combo.currentText())
        path = self.electrodes_path_input.text().strip()
        if not path:
            raise ValueError("Select an electrodes TSV file.")
        return read_electrodes_tsv(path, unit=self.coordinate_unit_combo.currentData())

    def _add_montage_channels(self):
        try:
            montage = self._resolve_montage()
        except (ValueError, OSError, UnicodeError, csv.Error) as exc:
            QMessageBox.warning(self, "Invalid Montage", str(exc))
            return
        existing = {
            self.table.item(row, 0).text().strip().casefold()
            for row in range(self.table.rowCount())
        }
        for name in montage.ch_names:
            if name.casefold() not in existing:
                self._append_channel(name)
                existing.add(name.casefold())

    def get_info(self) -> mne.Info:
        channel_info = ChannelInfo(
            ch_names=[self.table.item(row, 0).text() for row in range(self.table.rowCount())],
            ch_types=[
                self.table.cellWidget(row, 1).currentText()
                for row in range(self.table.rowCount())
            ],
        )
        return create_info(channel_info, self.sfreq_input.value(), self._resolve_montage())

    def accept(self):
        # Commit an active cell editor before collecting channel names.
        self.create_button.setFocus(Qt.FocusReason.OtherFocusReason)
        try:
            info = self.get_info()
        except (ValueError, OSError, UnicodeError, csv.Error) as exc:
            QMessageBox.warning(self, "Invalid INFO Settings", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save MNE INFO",
            str(self.output_directory / "recording-info.fif"),
            "MNE Info files (*.fif)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".fif":
            output_path = output_path.with_name(output_path.name + ".fif")
            if output_path.exists():
                response = QMessageBox.question(
                    self, "Replace INFO File", f"Replace the existing file?\n{output_path}"
                )
                if response != QMessageBox.StandardButton.Yes:
                    return
        try:
            info.save(output_path, overwrite=True)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return

        self.info = info
        self.saved_path = output_path
        QMessageBox.information(self, "INFO Saved", f"MNE INFO saved to:\n{output_path}")
        super().accept()
