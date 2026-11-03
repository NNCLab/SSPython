"""Unboxed object summaries for the preprocessing workflow."""

from html import escape
from math import ceil

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.widgets.layout_helpers import FilenameLabel, FlowLayout
from ui.widgets.tools.object_info_widget import (
    MNEObject,
    build_summary_metrics,
    extract_event_counts,
)


class _SummaryMetricLabel(QLabel):
    """Prefer a whole metric on one line, then wrap to the available width."""

    def sizeHint(self):
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(self.font())
        document.setHtml(self.text())
        return QSize(ceil(document.idealWidth()), ceil(document.size().height()))


class CompactObjectInfoWidget(QWidget):
    """Show a few wrapping metrics without holding an open data object."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stageSummary")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.metrics_widget = QWidget()
        self.metrics_widget.setObjectName("stageSummaryMetrics")
        self.metrics_layout = FlowLayout(self.metrics_widget, spacing=18)
        layout.addWidget(self.metrics_widget)

        self.events_label = FilenameLabel()
        self.events_label.setObjectName("stageSummaryEvents")
        layout.addWidget(self.events_label)
        self.metric_labels: list[QLabel] = []
        self.clear_info()

    def update_info(self, mne_object: MNEObject | None, title: str | None = None):
        self.clear_info()
        if mne_object is None:
            return

        self.setAccessibleName(title or "File summary")
        for name, value in build_summary_metrics(mne_object):
            label = _SummaryMetricLabel(
                f"<b>{escape(value)}</b> {escape(name.lower())}"
            )
            label.setObjectName("stageSummaryMetric")
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            policy = QSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
            )
            policy.setHeightForWidth(True)
            label.setSizePolicy(policy)
            label.setToolTip(f"{name}: {value}")
            self.metrics_layout.addWidget(label)
            self.metric_labels.append(label)

        counts = extract_event_counts(mne_object)
        self.events_label.setText(
            "Events · " + " · ".join(f"{name}: {count}" for name, count in counts)
            if counts
            else ""
        )
        self.events_label.setVisible(bool(counts))
        self.show()
        self.updateGeometry()

    def clear_info(self):
        while self.metrics_layout.count():
            widget = self.metrics_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.metric_labels.clear()
        self.events_label.clear()
        self.events_label.hide()
        self.setAccessibleName("")
        self.hide()
        self.updateGeometry()
