"""An explicit on/off control for an optional baseline interval."""

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from .optional_range_widget import OptionalRangeWidget


class BaselineControl(QWidget):
    valueChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.checkbox = QCheckBox("Apply baseline correction")
        self.interval = OptionalRangeWidget(
            ("Start:", "Stop:"), suffix=" ms", range=(None, None),
            scale=1e-3, parent=self,
        )
        self.interval.setToolTip(
            "Unchecked endpoints use the beginning or end of the epoch. "
            "Uncheck Apply baseline correction to keep the current data."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.interval)
        self.interval.setEnabled(False)
        self.checkbox.toggled.connect(self._on_toggled)
        self.interval.valueChanged.connect(lambda _: self.valueChanged.emit(self.value()))
        for spinbox in (self.interval.low_input, self.interval.high_input):
            spinbox.valueChanged.connect(self._resize_inputs)
        self._resize_inputs()

    def _resize_inputs(self):
        for spinbox in (self.interval.low_input, self.interval.high_input):
            width = spinbox.fontMetrics().horizontalAdvance(spinbox.text()) + 32
            spinbox.setMinimumWidth(max(104, width))

    def _on_toggled(self, enabled):
        self.interval.setEnabled(enabled)
        self.valueChanged.emit(self.value())

    def value(self):
        return self.interval.value() if self.checkbox.isChecked() else None

    def setValue(self, baseline, *, suggested=None):
        blocker = QSignalBlocker(self)
        self.interval.setValue(baseline if baseline is not None else suggested)
        self.checkbox.setChecked(baseline is not None)
        self.interval.setEnabled(baseline is not None)
        del blocker
