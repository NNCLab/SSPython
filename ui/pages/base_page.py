from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QLayout,
    QFrame,
    QHBoxLayout,
)
from PySide6.QtCore import Qt
from ui.widgets.layout_helpers import InspectionLink


class BasePage(QWidget):
    """
    A base class for pages providing a consistent, scrollable layout.
    The add_content method is now versatile and accepts both widgets and layouts.
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("pageRoot")

        self.base_layout = QVBoxLayout(self)
        self.base_layout.setContentsMargins(0, 0, 0, 0)
        self.base_layout.setSpacing(0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("pageScrollArea")
        self.scroll_area.viewport().setObjectName("pageScrollViewport")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.base_layout.addWidget(self.scroll_area, 1)

        self.content_container = QWidget()
        self.content_container.setObjectName("pageContent")
        self.content_container.setMaximumWidth(1440)
        self.scroll_area.setWidget(self.content_container)

        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(24, 24, 24, 24)
        self.content_layout.setSpacing(16)

        self.header_frame = QFrame()
        self.header_frame.setObjectName("pageHeader")
        header_layout = QVBoxLayout(self.header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("pageTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_row.addWidget(self.title_label, 1)
        header_layout.addLayout(title_row)

        self.subtitle_link = InspectionLink()
        self.subtitle_label = self.subtitle_link.label
        self.subtitle_label.setObjectName("pageSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_link.hide()
        header_layout.addWidget(self.subtitle_link)

        self.content_layout.addWidget(self.header_frame)

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

    def set_page_title(self, title: str):
        self.title_label.setText(title)

    def set_content_maximum_width(self, width: int):
        """Limit long pages while allowing them to shrink with the window."""
        self.content_container.setMaximumWidth(max(0, int(width)))

    def set_header_visible(self, visible: bool):
        self.header_frame.setVisible(visible)

    def set_footer(self, footer: QWidget):
        """Add a fixed footer below the scrollable page body."""
        footer.setParent(self)
        self.base_layout.addWidget(footer, 0)

    def set_page_subtitle(self, subtitle: str | None):
        if subtitle:
            self.subtitle_label.setText(subtitle)
            self.subtitle_link.show()
        else:
            self.subtitle_label.clear()
            self.subtitle_link.hide()

    def update_ui_state(self):
        """Updates the UI state. Can be overridden by subclasses."""
        pass

    def on_settings_updated(self):
        """Handles settings updates. Can be overridden by subclasses."""
        pass
