from .base_page import BasePage
from ..widgets.real_time_widget import RealTimeMainWidget


class RealTimePage(BasePage):
    """The page for real-time data collection and monitoring."""

    def __init__(self, parent=None):
        super().__init__("Real-time Data Visualization", parent)
        self.set_content_maximum_width(1200)
        self.header_frame.setObjectName("realTimeHeader")
        self.header_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.set_page_subtitle("Select an MNE info source, then launch the live visualizer.")
        self.content_layout.setSpacing(28)
        self.widget = RealTimeMainWidget(self)
        self.add_content(self.widget)
