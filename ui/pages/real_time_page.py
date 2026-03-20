from PySide6.QtWidgets import QLabel
from .base_page import BasePage
from ..widgets.real_time_widget import RealTimeMainWidget


class RealTimePage(BasePage):
    """The page for real-time data collection and monitoring."""

    def __init__(self, parent=None):
        super().__init__("<h1>Real-time Data Visualization", parent)
        self.widget = RealTimeMainWidget(self)
        self.add_content(self.widget)
