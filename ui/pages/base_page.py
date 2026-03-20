from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QLayout,
)  # Add QLayout
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt


class BasePage(QWidget):
    """
    A base class for pages providing a consistent, scrollable layout.
    The add_content method is now versatile and accepts both widgets and layouts.
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)

        base_layout = QVBoxLayout(self)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.setSpacing(0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        base_layout.addWidget(self.scroll_area)

        content_container = QWidget()
        self.scroll_area.setWidget(content_container)

        self.content_layout = QVBoxLayout(content_container)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(15)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.content_layout.addWidget(title_label)

        self.content_layout.addStretch()

    def add_content(self, item: QWidget | QLayout):
        """
        Adds a widget or a layout to the content area before the bottom stretch.
        This method intelligently handles both types.
        """
        if isinstance(item, QWidget):
            self.content_layout.insertWidget(self.content_layout.count() - 1, item)
        elif isinstance(item, QLayout):
            self.content_layout.insertLayout(self.content_layout.count() - 1, item)
        else:
            raise TypeError(
                f"add_content can only accept a QWidget or QLayout, not {type(item).__name__}"
            )

    def update_ui_state(self):
        """Updates the UI state. Can be overridden by subclasses."""
        pass

    def on_settings_updated(self):
        """Handles settings updates. Can be overridden by subclasses."""
        pass
