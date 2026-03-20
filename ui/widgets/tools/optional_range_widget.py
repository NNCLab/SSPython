from PySide6.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QDoubleSpinBox
from PySide6.QtCore import Signal
from typing import Tuple, Optional, List
import numpy as np


class OptionalRangeWidget(QWidget):
    """
    A custom widget for inputting an optional numerical range (e.g., low and high values).

    Each side of the range can be individually enabled or disabled with a checkbox.
    The widget's value is retrieved as a tuple of (low, high), where either can be None.
    """

    # Signal emitted when the value changes
    valueChanged = Signal(tuple)

    def __init__(
        self,
        labels: Tuple[str, str] = ("Low:", "High:"),
        suffix: str = "",
        required: Tuple[bool, bool] = (False, False),
        range: Tuple[float, float] = (0, None),
        scale: float = 1,
        parent: Optional[QWidget] = None,
    ):
        """
        Initializes the widget.

        Args:
            labels: A tuple of two strings for the low and high input labels.
            suffix: A string to append to the spin box values (e.g., " Hz", " ms").
            range: A tuple of (min, max) for the spin boxes.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.scale = scale

        # --- Create UI Components ---
        self.low_check = QCheckBox(labels[0])
        self.high_check = QCheckBox(labels[1])
        self.low_input = QDoubleSpinBox()
        self.high_input = QDoubleSpinBox()

        for spinbox in [self.low_input, self.high_input]:
            if range[0] is None:
                spinbox.setMinimum(-np.inf)
            if range[1] is None:
                spinbox.setMaximum(+np.inf)
            spinbox.setSuffix(suffix)

        # --- Layout ---
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # No extra margins
        layout.addWidget(self.low_check)
        layout.addWidget(self.low_input)
        layout.addSpacing(15)  # Add space between the two pairs
        layout.addWidget(self.high_check)
        layout.addWidget(self.high_input)
        layout.addStretch()

        # --- Connect Signals ---
        self.low_check.toggled.connect(self.low_input.setEnabled)
        self.high_check.toggled.connect(self.high_input.setEnabled)

        # Emit valueChanged Signal whenever the state changes
        self.low_check.toggled.connect(self._emit_value_changed)
        self.high_check.toggled.connect(self._emit_value_changed)
        self.low_input.valueChanged.connect(self._emit_value_changed)
        self.low_input.valueChanged.connect(self.adjust_spinbox_width)
        self.high_input.valueChanged.connect(self.adjust_spinbox_width)
        self.high_input.valueChanged.connect(self._emit_value_changed)
        self.adjust_spinbox_width()

        # --- First State ---
        self.low_check.setChecked(False)
        self.high_check.setChecked(False)

        # --- Required? ---
        if required[0]:
            self.low_check.setChecked(True)
            self.low_check.setDisabled(True)

        if required[1]:
            self.high_check.setChecked(True)
            self.high_check.setDisabled(True)

        self.low_input.setEnabled(self.low_check.isChecked())
        self.high_input.setEnabled(self.high_check.isChecked())

    def value(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Gets the current range as a tuple. Returns None for disabled parts.
        """
        low = (
            self.low_input.value() * self.scale if self.low_check.isChecked() else None
        )
        high = (
            self.high_input.value() * self.scale
            if self.high_check.isChecked()
            else None
        )
        return (low, high)

    def setValue(self, values: Tuple[Optional[float], Optional[float]]):
        """
        Sets the widget's state from a tuple.
        """
        low, high = values if values else (None, None)

        self.low_check.setChecked(low is not None)
        self.high_check.setChecked(high is not None)

        self.low_input.setValue(low / self.scale if low is not None else 0)
        self.high_input.setValue(high / self.scale if high is not None else 0)

    def _emit_value_changed(self):
        """Internal helper to emit the valueChanged Signal."""
        self.valueChanged.emit(self.value())

    def adjust_spinbox_width(self):
        """
        Adjusts the minimum width of a QDoubleSpinBox to fit its content.
        """
        # 1. Get the text for the min and max values
        prefix = self.low_input.prefix()
        suffix = self.low_input.suffix()
        decimals = self.low_input.decimals()

        # Format the numbers as they would appear in the spin box
        min_text = f"{prefix} {self.low_input.minimum():.{decimals}f}{suffix} "
        max_text = f"{prefix} {self.low_input.maximum():.{decimals}f}{suffix} "

        # 2. Get the font metrics from the internal line edit
        font_metrics = self.low_input.lineEdit().fontMetrics()

        # 3. Calculate the pixel width of both strings
        min_width = font_metrics.horizontalAdvance(min_text)
        max_width = font_metrics.horizontalAdvance(max_text)

        # 4. Determine the widest text and add a margin
        # The margin accounts for the up/down arrows and some internal padding.
        # 30px is usually a safe value.
        required_width = max(min_width, max_width) + 30

        # 5. Set the calculated minimum width
        self.low_input.setMinimumWidth(required_width)
        self.high_input.setMinimumWidth(required_width)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = OptionalRangeWidget()
    window.show()
    sys.exit(app.exec())
