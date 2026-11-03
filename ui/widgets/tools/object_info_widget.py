from __future__ import annotations

import json
from collections import Counter
from typing import Optional, Union

import mne
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.layout_helpers import FilenameLabel


MNEObject = Union[mne.io.Raw, mne.Epochs, mne.preprocessing.ICA]


def extract_event_counts(mne_object: MNEObject) -> list[tuple[str, int]]:
    if isinstance(mne_object, (mne.io.Raw, mne.io.RawArray)):
        annotations = getattr(mne_object, "annotations", None)
        if annotations is None or len(annotations) == 0:
            return []
        counts = Counter(str(description) for description in annotations.description if str(description).strip())
        return sorted(counts.items())

    if isinstance(mne_object, (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF)):
        if getattr(mne_object, "events", None) is None or len(mne_object.events) == 0:
            return []

        inverse_event_map: dict[int, list[str]] = {}
        for event_name, event_code in getattr(mne_object, "event_id", {}).items():
            inverse_event_map.setdefault(int(event_code), []).append(str(event_name))

        ids, counts = mne_object.events[:, 2], Counter(mne_object.events[:, 2])
        event_counts: list[tuple[str, int]] = []
        for event_code in sorted(set(int(event_id) for event_id in ids)):
            label = " / ".join(sorted(inverse_event_map.get(event_code, [str(event_code)])))
            event_counts.append((label, counts[event_code]))
        return event_counts

    return []


def _format_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "None"
        return ", ".join(str(item) for item in value)
    return str(value)


def _window_text(start: Optional[float], stop: Optional[float], *, scale: float = 1.0, suffix: str = "") -> str:
    if start is None and stop is None:
        return "None"
    left = "None" if start is None else f"{start * scale:.2f}{suffix}"
    right = "None" if stop is None else f"{stop * scale:.2f}{suffix}"
    return f"{left} to {right}"


def _description_entries(mne_object: MNEObject) -> list[dict]:
    info = getattr(mne_object, "info", None)
    description = info.get("description") if info is not None else None

    if not description:
        return []

    try:
        parsed = json.loads(description)
    except (json.JSONDecodeError, TypeError):
        return [{"description": str(description)}]

    if isinstance(parsed, list):
        return [entry for entry in parsed if isinstance(entry, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return [{"description": str(parsed)}]


def _humanize_key(key: str) -> str:
    return str(key).replace("_", " ").strip().title()


def _top_level_log_entry(entry: dict, index: int) -> tuple[str, object, str]:
    if len(entry) == 1:
        key, value = next(iter(entry.items()))
        timestamp = ""
        if isinstance(value, dict):
            timestamp = str(value.get("date", "") or "")
        return _humanize_key(str(key)), value, timestamp
    return f"Entry {index}", entry, ""


def build_summary_metrics(mne_object: MNEObject) -> list[tuple[str, str]]:
    if isinstance(mne_object, (mne.io.Raw, mne.io.RawArray)):
        return [
            ("Channels", str(len(mne_object.ch_names))),
            ("Time Points", str(mne_object.n_times)),
            ("Sampling", f"{mne_object.info['sfreq']:.2f} Hz"),
            ("Duration", f"{mne_object.times[-1]:.2f} s"),
        ]

    if isinstance(mne_object, (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF)):
        total_epochs = len(mne_object.drop_log)
        kept_epochs = len(mne_object)
        return [
            ("Epochs", f"{kept_epochs}/{total_epochs}"),
            (
                "Channels",
                f"{len(mne_object.ch_names) - len(mne_object.info['bads'])}/{len(mne_object.ch_names)}",
            ),
            ("Sampling", f"{mne_object.info['sfreq']:.2f} Hz"),
            ("Window", _window_text(mne_object.tmin, mne_object.tmax, scale=1e3, suffix=" ms")),
        ]

    if isinstance(mne_object, mne.preprocessing.ICA):
        n_components = int(mne_object.n_components_ or 0)
        excluded = len(mne_object.exclude)
        return [
            ("Components", f"{n_components - excluded}/{n_components}"),
            ("Excluded", str(excluded)),
            ("Samples", _format_value(getattr(mne_object, "n_samples_", None))),
            ("Method", _format_value(mne_object.method).upper()),
        ]

    return [("Status", "Unsupported object")]


def build_detail_sections(mne_object: MNEObject) -> list[tuple[str, object]]:
    sections: list[tuple[str, object]] = []

    event_counts = extract_event_counts(mne_object)
    if event_counts:
        sections.append(("Event Counts", {label: count for label, count in event_counts}))

    if isinstance(mne_object, (mne.io.Raw, mne.io.RawArray)):
        sections.append(
            (
                "Metadata",
                {
                    "Channels": len(mne_object.ch_names),
                    "Bad Channels": mne_object.info.get("bads", []),
                    "Sampling Frequency": f"{mne_object.info['sfreq']:.2f} Hz",
                    "High-pass": _format_value(mne_object.info.get("highpass")),
                    "Low-pass": _format_value(mne_object.info.get("lowpass")),
                    "Duration": f"{mne_object.times[-1]:.2f} s",
                },
            )
        )
        sections.append(("Channel Names", ", ".join(mne_object.ch_names)))

    elif isinstance(mne_object, (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF)):
        sections.append(
            (
                "Metadata",
                {
                    "Epochs": f"{len(mne_object)}/{len(mne_object.drop_log)}",
                    "Bad Channels": mne_object.info.get("bads", []),
                    "Sampling Frequency": f"{mne_object.info['sfreq']:.2f} Hz",
                    "Window": _window_text(mne_object.tmin, mne_object.tmax, scale=1e3, suffix=" ms"),
                    "Baseline": _window_text(
                        mne_object.baseline[0] if mne_object.baseline else None,
                        mne_object.baseline[1] if mne_object.baseline else None,
                        scale=1e3,
                        suffix=" ms",
                    ),
                },
            )
        )
        sections.append(("Channel Names", ", ".join(mne_object.ch_names)))

    elif isinstance(mne_object, mne.preprocessing.ICA):
        sections.append(
            (
                "Metadata",
                {
                    "Method": _format_value(mne_object.method),
                    "Components": _format_value(mne_object.n_components_),
                    "Excluded": _format_value(mne_object.exclude),
                    "PCA Components": _format_value(mne_object.n_pca_components),
                    "Channels": _format_value(mne_object.ch_names),
                },
            )
        )

    description_entries = _description_entries(mne_object)
    if description_entries:
        sections.append(("Processing Log", description_entries))

    return sections


class MetricTile(QFrame):
    def __init__(self, label: str, value: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("objectInfoMetric")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        label_widget = QLabel(label)
        label_widget.setObjectName("objectInfoMetricLabel")
        value_widget = QLabel(value)
        value_widget.setObjectName("objectInfoMetricValue")
        value_widget.setWordWrap(True)

        layout.addWidget(label_widget)
        layout.addWidget(value_widget)


class EventChip(QFrame):
    def __init__(self, label: str, count: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("objectInfoEventChip")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        event_label = QLabel(label)
        event_label.setObjectName("objectInfoEventLabel")
        count_label = QLabel(str(count))
        count_label.setObjectName("objectInfoEventCount")

        layout.addWidget(event_label)
        layout.addWidget(count_label)


class ObjectInfoWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mne_object: Optional[MNEObject] = None
        self.setObjectName("objectInfoCard")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.header_label = QLabel("")
        self.header_label.setObjectName("objectInfoTitle")
        self.header_label.hide()
        layout.addWidget(self.header_label)

        self.metrics_widget = QWidget()
        self.metrics_widget.setObjectName("objectInfoMetrics")
        self.metrics_layout = QGridLayout(self.metrics_widget)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setHorizontalSpacing(10)
        self.metrics_layout.setVerticalSpacing(10)
        layout.addWidget(self.metrics_widget)

        self.events_frame = QFrame()
        self.events_frame.setObjectName("objectInfoEvents")
        self.events_frame.hide()
        events_layout = QVBoxLayout(self.events_frame)
        events_layout.setContentsMargins(0, 0, 0, 0)
        events_layout.setSpacing(6)

        self.events_title = QLabel("Events")
        self.events_title.setObjectName("objectInfoSectionTitle")
        events_layout.addWidget(self.events_title)

        self.event_chips_widget = QWidget()
        self.event_chips_widget.setObjectName("objectInfoEventChips")
        self.event_chips_layout = QHBoxLayout(self.event_chips_widget)
        self.event_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.event_chips_layout.setSpacing(6)
        events_layout.addWidget(self.event_chips_widget)
        layout.addWidget(self.events_frame)

        self.hide()

    def update_info(
        self,
        mne_object: Optional[MNEObject],
        title: Optional[str] = None,
    ):
        if mne_object is None:
            self.clear_info()
            return

        self.mne_object = mne_object
        self.header_label.setText(title or self._object_type_name(mne_object))
        self.header_label.show()
        self._set_metrics(build_summary_metrics(mne_object))
        self._set_event_counts(extract_event_counts(mne_object))
        self.show()

    def clear_info(self):
        self.mne_object = None
        self.header_label.clear()
        self.header_label.hide()
        self._clear_layout(self.metrics_layout)
        self._clear_layout(self.event_chips_layout)
        self.events_frame.hide()
        self.hide()

    def detail_sections(self) -> list[tuple[str, object]]:
        if self.mne_object is None:
            return []
        return build_detail_sections(self.mne_object)

    def _set_metrics(self, metrics: list[tuple[str, str]]):
        self._clear_layout(self.metrics_layout)
        for index, (label, value) in enumerate(metrics):
            row, col = divmod(index, 2)
            self.metrics_layout.addWidget(MetricTile(label, value, self), row, col)

    def _set_event_counts(self, event_counts: list[tuple[str, int]]):
        self._clear_layout(self.event_chips_layout)
        if not event_counts:
            self.events_frame.hide()
            return

        for label, count in event_counts:
            self.event_chips_layout.addWidget(EventChip(label, count, self))
        self.event_chips_layout.addStretch()
        self.events_frame.show()

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _object_type_name(mne_object: MNEObject) -> str:
        if isinstance(mne_object, (mne.io.Raw, mne.io.RawArray)):
            return "Recording"
        if isinstance(mne_object, (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF)):
            return "Epochs"
        if isinstance(mne_object, mne.preprocessing.ICA):
            return "ICA"
        return "Object"


class ObjectDetailsDialog(QDialog):
    open_location_requested = Signal()
    delete_requested = Signal()

    def __init__(
        self,
        title: str,
        mne_object: Optional[MNEObject],
        *,
        file_path: str = "",
        status_text: str = "",
        description: str = "",
        allow_delete: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(940, 540 if mne_object is not None else 200)
        self.setMinimumSize(640, 380 if mne_object is not None else 0)
        self.setObjectName("fileInfoDialog")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(16)

        header_widget = QWidget()
        header_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        header = QVBoxLayout(header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        heading = QHBoxLayout()
        heading.setSpacing(12)
        title_label = FilenameLabel(title)
        title_label.setObjectName("fileInfoTitle")
        heading.addWidget(title_label, 1)

        if status_text:
            status_label = QLabel(status_text)
            status_label.setTextFormat(Qt.TextFormat.PlainText)
            status_label.setObjectName("fileInfoStatus")
            heading.addWidget(status_label, 0, Qt.AlignmentFlag.AlignTop)
        header.addLayout(heading)

        if description:
            description_label = QLabel(description)
            description_label.setTextFormat(Qt.TextFormat.PlainText)
            description_label.setObjectName("fileInfoDescription")
            description_label.setWordWrap(True)
            header.addWidget(description_label)

        if file_path:
            # Keep long paths on one quiet line, with native selection/copy support.
            self.path_field = QLineEdit(file_path)
            self.path_field.setObjectName("fileInfoPath")
            self.path_field.setReadOnly(True)
            self.path_field.setFrame(False)
            self.path_field.setCursorPosition(0)
            self.path_field.setToolTip(file_path)
            self.path_field.setAccessibleName("File path")
            header.addWidget(self.path_field)
        layout.addWidget(header_widget)

        if mne_object is not None:
            # Local import: the compact summary shares the extraction helpers above.
            from ui.widgets.tools.compact_object_info import CompactObjectInfoWidget

            self.summary_widget = CompactObjectInfoWidget(self)
            self.summary_widget.update_info(mne_object)
            self.summary_widget.events_label.hide()  # Listed once in the details below.
            self.summary_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.summary_widget.layout().setContentsMargins(0, 0, 0, 12)
            layout.addWidget(self.summary_widget)

            detail_sections = build_detail_sections(mne_object)
            processing_log = next(
                (
                    content
                    for section_title, content in detail_sections
                    if section_title == "Processing Log"
                ),
                None,
            )
            information_sections = self._information_sections(detail_sections, mne_object)

            self.details_scroll = self._build_sections_scroll(information_sections)
            if processing_log is not None:
                self.details_splitter = QSplitter(Qt.Orientation.Horizontal)
                self.details_splitter.setObjectName("fileInfoDetailsSplitter")
                self.details_splitter.setChildrenCollapsible(False)
                self.details_splitter.setHandleWidth(18)
                self.details_splitter.addWidget(self.details_scroll)

                self.processing_log_section = self._build_section(
                    "Processing Log",
                    processing_log,
                )
                self.processing_log_section.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Expanding,
                )
                self.details_splitter.addWidget(self.processing_log_section)
                self.details_splitter.setStretchFactor(0, 1)
                self.details_splitter.setStretchFactor(1, 1)
                self.details_splitter.setSizes([420, 460])
                layout.addWidget(self.details_splitter, 1)
            else:
                layout.addWidget(self.details_scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.setObjectName("fileInfoFooter")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        if file_path:
            self.open_location_button = buttons.addButton(
                "Open derivatives folder", QDialogButtonBox.ButtonRole.ActionRole
            )
            self.open_location_button.setObjectName("secondaryButton")
            self.open_location_button.clicked.connect(self.open_location_requested)
        if allow_delete:
            self.delete_button = buttons.addButton(
                "Delete this and subsequent files…", QDialogButtonBox.ButtonRole.ActionRole
            )
            self.delete_button.setObjectName("dangerButton")
            self.delete_button.setToolTip("Delete this saved step and its dependent files")
            self.delete_button.clicked.connect(self.delete_requested)
        layout.addWidget(buttons)
        if not hasattr(self, "details_splitter"):
            # Short files should not open into a large, mostly empty window.
            self.ensurePolished()
            content_width = self.width() - 48
            widgets = [header_widget, buttons]
            if mne_object is not None:
                widgets.extend([self.summary_widget, self.details_scroll.widget()])
            content_height = sum(
                widget.heightForWidth(content_width)
                if widget.hasHeightForWidth()
                else widget.sizeHint().height()
                for widget in widgets
            )
            height = content_height + 38 + (len(widgets) - 1) * layout.spacing()
            self.resize(self.width(), min(540, max(self.minimumHeight(), height)))

    @staticmethod
    def _information_sections(
        sections: list[tuple[str, object]], mne_object: MNEObject
    ) -> list[tuple[str, object]]:
        """Leave summary values in one place while retaining the detailed data."""
        if isinstance(mne_object, mne.preprocessing.ICA):
            summary_keys = {"Method", "Components"}
        elif isinstance(mne_object, (mne.io.Raw, mne.io.RawArray)):
            summary_keys = {"Channels", "Sampling Frequency", "Duration"}
        else:
            summary_keys = {"Epochs", "Sampling Frequency", "Window"}

        information = []
        for title, content in sections:
            if title == "Processing Log":
                continue
            if title == "Metadata":
                content = {key: value for key, value in content.items() if key not in summary_keys}
                content = {
                    "Excluded components" if key == "Excluded" else key: value
                    for key, value in content.items()
                }
                information.insert(0, ("Details", content))
            else:
                section_title = {"Event Counts": "Events", "Channel Names": "Channels"}.get(title, title)
                information.append((section_title, content))
        return information

    def resizeEvent(self, event):
        super().resizeEvent(event)
        splitter = getattr(self, "details_splitter", None)
        if splitter is not None:
            orientation = Qt.Orientation.Vertical if self.width() < 800 else Qt.Orientation.Horizontal
            if splitter.orientation() != orientation:
                splitter.setOrientation(orientation)
                if orientation == Qt.Orientation.Vertical:
                    body_height = max(260, self.height() - 220)
                    splitter.setSizes([int(body_height * 0.65), int(body_height * 0.35)])
                else:
                    splitter.setSizes([420, 460])

    def release_object(self):
        """Drop the dialog's reference to a file-backed MNE object."""
        summary_widget = getattr(self, "summary_widget", None)
        if summary_widget is not None:
            summary_widget.clear_info()

    def _build_sections_scroll(
        self,
        sections: list[tuple[str, object]],
    ) -> QScrollArea:
        sections_scroll = QScrollArea()
        sections_scroll.setObjectName("fileInfoDetailsScroll")
        sections_scroll.setWidgetResizable(True)
        sections_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        sections_widget = QWidget()
        sections_widget.setObjectName("fileInfoDetailsContent")
        sections_layout = QVBoxLayout(sections_widget)
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(18)

        for section_title, content in sections:
            sections_layout.addWidget(self._build_section(section_title, content))
        sections_layout.addStretch()

        sections_scroll.setWidget(sections_widget)
        return sections_scroll

    def _build_section(self, title: str, content: object) -> QWidget:
        section = QWidget()
        section.setObjectName("fileInfoHistorySection" if title == "Processing Log" else "fileInfoSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)

        title_label = QLabel("Processing history" if title == "Processing Log" else title)
        title_label.setObjectName("fileInfoSectionTitle")
        layout.addWidget(title_label)

        if isinstance(content, dict):
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(5)
            grid.setColumnMinimumWidth(0, 110)
            grid.setColumnStretch(1, 1)
            for row, (key, value) in enumerate(content.items()):
                key_label = QLabel(str(key))
                key_label.setTextFormat(Qt.TextFormat.PlainText)
                key_label.setObjectName("fileInfoDetailKey")
                key_label.setWordWrap(True)
                key_label.setMaximumWidth(160)
                value_label = self._value_label(value)
                grid.addWidget(key_label, row, 0, Qt.AlignmentFlag.AlignTop)
                grid.addWidget(value_label, row, 1, Qt.AlignmentFlag.AlignTop)
            layout.addLayout(grid)
            return section

        if title == "Processing Log" and isinstance(content, list):
            layout.addWidget(self._build_log_tree(content))
            return section

        text_content = content
        if isinstance(content, list):
            text_content = json.dumps(content, indent=2)

        layout.addWidget(self._value_label(text_content))
        return section

    @staticmethod
    def _value_label(value) -> QLabel:
        label = QLabel(_format_value(value))
        label.setObjectName("fileInfoDetailValue")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _build_log_tree(self, entries: list[dict]) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setObjectName("fileInfoHistoryTree")
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Operation", "Details"])
        tree.setRootIsDecorated(True)
        tree.setAlternatingRowColors(False)
        tree.setUniformRowHeights(False)
        tree.setWordWrap(True)
        tree.setMinimumSize(0, 100)
        tree.setIndentation(14)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        tree.setColumnWidth(0, 180)
        tree.header().setStretchLastSection(True)

        for index, entry in enumerate(entries, start=1):
            label, payload, timestamp = _top_level_log_entry(entry, index)
            top_item = QTreeWidgetItem([label, timestamp])
            if timestamp:
                top_item.setToolTip(1, timestamp)
            tree.addTopLevelItem(top_item)
            self._append_tree_children(top_item, payload, skip_keys={"date"})

        return tree

    def _append_tree_children(
        self,
        parent: QTreeWidgetItem,
        value: object,
        *,
        skip_keys: set[str] | None = None,
    ):
        if isinstance(value, dict):
            for key, child_value in value.items():
                if skip_keys and key in skip_keys:
                    continue
                child_item = QTreeWidgetItem([_humanize_key(str(key)), ""])
                parent.addChild(child_item)
                self._append_tree_children(child_item, child_value, skip_keys=skip_keys)
            return

        if isinstance(value, list):
            if not value:
                parent.setText(1, "None")
                return

            scalar_values = all(not isinstance(item, (dict, list, tuple, set)) for item in value)
            if scalar_values:
                parent.setText(1, _format_value(value))
                return

            for index, child_value in enumerate(value, start=1):
                child_item = QTreeWidgetItem([f"Item {index}", ""])
                parent.addChild(child_item)
                self._append_tree_children(child_item, child_value, skip_keys=skip_keys)
            return

        if isinstance(value, tuple):
            parent.setText(1, _format_value(list(value)))
            return

        if isinstance(value, set):
            parent.setText(1, _format_value(sorted(value)))
            return

        parent.setText(1, _format_value(value))
