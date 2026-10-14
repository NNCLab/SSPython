from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QToolButton, QVBoxLayout, QWidget


class AdvancedSettings(QWidget):
    """A disclosure that hides its contents without disabling their values."""

    def __init__(self, title="Advanced settings", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("advancedSettingsToggle")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setAccessibleName(title)
        layout.addWidget(self.toggle_button)
        self.content = QWidget()
        self.form = QFormLayout(self.content)
        self.form.setContentsMargins(12, 4, 0, 4)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addWidget(self.content)
        self.content.hide()
        self.toggle_button.toggled.connect(self.set_expanded)

    def set_expanded(self, expanded):
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(expanded)
