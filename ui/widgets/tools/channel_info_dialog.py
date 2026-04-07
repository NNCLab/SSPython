from __future__ import annotations

from pathlib import Path

import mne
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.channel_info import (
    COMMON_CHANNEL_TYPES,
    ChannelInfo,
    ChannelIssue,
    identify_channel_info_issues,
    identify_montage_issues,
    load_channel_info,
    save_channel_info,
    validate_channel_info,
)


class ChannelInfoEditorDialog(QDialog):
    def __init__(
        self,
        *,
        source_label: str,
        montage_label: str,
        error_message: str,
        channel_info: ChannelInfo,
        issues: list[ChannelIssue],
        montage: mne.channels.DigMontage | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Fix Channel Names and Types")
        self.resize(860, 640)

        self.montage = montage
        self.saved_path: Path | None = None
        self._loading = False
        self._current_issue_rows: set[int] = set()

        layout = QVBoxLayout(self)

        summary_label = QLabel(
            f"{error_message}\n\n"
            f"Review the channel names and types for {source_label}. "
            f"Rows highlighted below do not match the selected montage ({montage_label}) "
            "or contain invalid channel metadata."
        )
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        actions_row = QHBoxLayout()
        self.load_button = QPushButton("Load JSON")
        self.save_button = QPushButton("Save JSON")
        actions_row.addWidget(self.load_button)
        actions_row.addWidget(self.save_button)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Status", "Channel Name", "Channel Type", "Issue"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.load_button.clicked.connect(self._load_from_json)
        self.save_button.clicked.connect(self._save_to_json)
        self.table.itemChanged.connect(self._on_item_changed)

        self._populate_table(channel_info)
        self._apply_issue_state(issues)

    def get_channel_info(self) -> ChannelInfo:
        return self._collect_channel_info()

    def accept(self):
        try:
            channel_info = validate_channel_info(self._collect_channel_info())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Channel Info", str(exc))
            return

        remaining_issues = self._calculate_issues(channel_info)
        if remaining_issues:
            response = QMessageBox.question(
                self,
                "Channels Still Flagged",
                "Some channels are still flagged against the selected montage. "
                "Use these channel settings anyway?",
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        super().accept()

    def _populate_table(self, channel_info: ChannelInfo):
        self._loading = True
        self.table.setRowCount(0)

        for row, (name, channel_type) in enumerate(
            zip(channel_info.ch_names, channel_info.ch_types, strict=False)
        ):
            self.table.insertRow(row)

            status_item = QTableWidgetItem("OK")
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            issue_item = QTableWidgetItem("")
            issue_item.setFlags(issue_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name_item = QTableWidgetItem(name)

            self.table.setItem(row, 0, status_item)
            self.table.setItem(row, 1, name_item)
            self.table.setCellWidget(row, 2, self._create_type_combo(channel_type))
            self.table.setItem(row, 3, issue_item)

        self._loading = False

    def _create_type_combo(self, channel_type: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(COMMON_CHANNEL_TYPES)
        combo.setCurrentText(channel_type)
        combo.currentTextChanged.connect(self._refresh_issues)
        return combo

    def _collect_channel_info(self) -> ChannelInfo:
        ch_names: list[str] = []
        ch_types: list[str] = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            type_combo = self.table.cellWidget(row, 2)
            if name_item is None or not isinstance(type_combo, QComboBox):
                continue
            ch_names.append(name_item.text().strip())
            ch_types.append(type_combo.currentText().strip().lower())
        return ChannelInfo(ch_names=ch_names, ch_types=ch_types)

    def _calculate_issues(self, channel_info: ChannelInfo) -> list[ChannelIssue]:
        if self.montage is None:
            return identify_channel_info_issues(channel_info)
        return identify_montage_issues(channel_info, self.montage)

    def _apply_issue_state(self, issues: list[ChannelIssue]):
        issue_map = {issue.index: issue.reason for issue in issues}
        self._current_issue_rows = set(issue_map)

        warning_color = QColor("#5f2222")
        clear_color = QColor(Qt.GlobalColor.transparent)
        for row in range(self.table.rowCount()):
            status_item = self.table.item(row, 0)
            name_item = self.table.item(row, 1)
            issue_item = self.table.item(row, 3)
            type_combo = self.table.cellWidget(row, 2)
            if status_item is None or name_item is None or issue_item is None:
                continue

            issue_text = issue_map.get(row, "")
            status_item.setText("Needs attention" if issue_text else "OK")
            issue_item.setText(issue_text)

            row_color = warning_color if issue_text else clear_color
            status_item.setBackground(row_color)
            name_item.setBackground(row_color)
            issue_item.setBackground(row_color)

            if isinstance(type_combo, QComboBox):
                type_combo.setStyleSheet(
                    "QComboBox { background-color: %s; }"
                    % (warning_color.name() if issue_text else "transparent")
                )

    def _refresh_issues(self):
        if self._loading:
            return
        self._apply_issue_state(self._calculate_issues(self._collect_channel_info()))

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._loading:
            return
        if item.column() == 1:
            self._refresh_issues()

    def _load_from_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Channel Info JSON",
            "",
            "JSON files (*.json)",
        )
        if not file_path:
            return

        try:
            channel_info = load_channel_info(Path(file_path))
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Channel Info", str(exc))
            return

        if len(channel_info.ch_names) != self.table.rowCount():
            QMessageBox.warning(
                self,
                "Channel Count Mismatch",
                "The loaded channel info does not match the number of channels in the current recording.",
            )
            return

        self.saved_path = None
        self._populate_table(channel_info)
        self._refresh_issues()

    def _save_to_json(self):
        try:
            channel_info = validate_channel_info(self._collect_channel_info())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Channel Info", str(exc))
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Channel Info JSON",
            "",
            "JSON files (*.json)",
        )
        if not file_path:
            return

        try:
            self.saved_path = save_channel_info(channel_info, Path(file_path))
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Channel Info Saved",
            f"Channel information saved to:\n{self.saved_path}",
        )
