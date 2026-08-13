from .base_page import BasePage
from ..widgets.real_time_widget import RealTimeMainWidget


class RealTimePage(BasePage):
    """The page for real-time data collection and monitoring."""

    def __init__(self, parent=None):
        super().__init__("Real-time Data Visualization", parent)
        self.set_content_maximum_width(1200)
        self.widget = RealTimeMainWidget(self)
        self.add_content(self.widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "widget"):
            self.widget.set_compact_layout(event.size().width() < 1180)
