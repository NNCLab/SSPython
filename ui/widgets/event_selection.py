"""Shared analysis selector for event IDs present in epochs."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox

from core.analysis_events import event_label, event_options, select_event_epochs


class EventSelector(QComboBox):
    def __init__(self, epochs=None, parent=None):
        super().__init__(parent)
        self.set_epochs(epochs)
        self.setToolTip("Use all epochs or only epochs with the selected event_id.")

    def set_epochs(self, epochs):
        """Refresh options, retaining the selected ID when it is still present."""
        selected_code = self.currentData()
        blocker = QSignalBlocker(self)
        self.epochs = epochs
        self.clear()
        self.addItem("All events", None)
        if epochs is not None:
            for code, name, count in event_options(epochs):
                self.addItem(f"{name} [event_id={code}] ({count} epochs)", code)
        self.setCurrentIndex(max(0, self.findData(selected_code)))
        self.setEnabled(self.count() > 1)
        del blocker

    def selected_epochs(self):
        return select_event_epochs(self.epochs, self.currentData())

    def selected_label(self):
        return event_label(self.epochs, self.currentData())
