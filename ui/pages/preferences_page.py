import logging
from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QToolBox,
    QFrame,
    QLabel,
)
from PySide6.QtGui import QIcon
from .base_page import BasePage
from utils import get_path

# Global
from ..widgets.tools.global_preferences import GlobalSettingsWidget

# Epochs
from ..widgets.preprocessing_widgets import (
    EpochingSettingsWidget,
    ArtifactRemovalSettingsWidget,
    PreprocessingSettingsWidget,
)

# ICA
from ..widgets.run_ica_widget import ICASettingsWidget
from ..widgets.ica_widget import ICAPlottingSettingsWidget
from ..widgets.real_time_widget import RealTimeSettingsWidget, ConnectionWidget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferencesPage(BasePage):
    """
    A self-contained preferences page that creates and manages its own
    settings categories.
    """

    settings_saved = Signal()

    def __init__(self, settings: QSettings, parent: QWidget = None):
        super().__init__("Preferences", parent)
        self.settings = settings
        self.settings_widgets = []
        self.main_window = parent

        self._setup_ui()
        self._connect_signals()
        self._populate_categories()

    def _setup_ui(self):
        """Initializes widgets and layouts."""
        self.tool_box = QToolBox()
        self.tool_box.layout().setSpacing(10)
        self.add_content(self.tool_box)

        button_layout = QHBoxLayout()
        self.clear_all_button = QPushButton("Clear All Settings")
        self.revert_button = QPushButton("Revert Changes")
        self.save_button = QPushButton("Save Changes")

        self.clear_all_button.setToolTip(
            "Clear all saved settings from the application."
        )
        self.revert_button.setToolTip(
            "Reload the last saved settings, discarding any changes."
        )
        self.save_button.setToolTip("Save all current settings.")
        self.save_button.setDefault(True)

        button_layout.addWidget(self.clear_all_button)
        button_layout.addStretch()
        button_layout.addWidget(self.revert_button)
        button_layout.addWidget(self.save_button)
        self.add_content(button_layout)

    def _connect_signals(self):
        self.save_button.clicked.connect(self.save_all_settings)
        self.revert_button.clicked.connect(self.load_all_settings)
        self.clear_all_button.clicked.connect(self.clear_all_settings)

    def _populate_categories(self):
        """
        Creates and adds all the specific settings widgets to this page.
        """
        self.add_settings_category(
            GlobalSettingsWidget(),
            title="Global Settings",
            icon=QIcon(get_path("assets/icons/settings")),
        ),
        self.add_settings_category(
            ArtifactRemovalSettingsWidget(),
            EpochingSettingsWidget(),
            PreprocessingSettingsWidget(),
            title="TEP Preprocessing Steps",
            icon=QIcon(get_path("assets/icons/erp_preprocess")),
        ),
        self.add_settings_category(
            QLabel("Continuous"),
            ICASettingsWidget("continuous"),
            QLabel("Epochs"),
            ICASettingsWidget("epochs"),
            QLabel("Plotting"),
            ICAPlottingSettingsWidget(),
            title="ICA Settings",
            icon=QIcon(get_path("assets/icons/down_arrow")),
        ),
        self.add_settings_category(
            QLabel("Connection"),
            ConnectionWidget(),
            QLabel("Plotting"),
            RealTimeSettingsWidget(),
            title="Real-Time Settings",
            icon=QIcon(get_path("assets/icons/realtime")),
        )

    def add_settings_category(self, *widgets: QWidget, title: str, icon: QIcon = None):
        """
        Adds one or more settings widgets as a new collapsible category.

        Args:
            title (str): The title for the settings category.
            *widgets (QWidget): A variable number of widget instances to add.
            icon (QIcon, optional): An icon for the category tab.
        """
        if not widgets:
            raise ValueError("You must provide at least one widget.")

        # --- MODIFICATION START ---
        # 1. Validate all provided widgets and add them to the master list
        for widget in widgets:
            if isinstance(widget, QLabel):
                continue
            if not all(
                hasattr(widget, attr) for attr in ["load_settings", "save_settings"]
            ):
                raise TypeError(
                    f"Widget '{widget.__class__.__name__}' for category '{title}' is missing required methods."
                )
        self.settings_widgets.extend(widgets)
        container = QFrame()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        if icon:
            self.tool_box.addItem(container, icon, title)
        else:
            self.tool_box.addItem(container, title)
        for widget in widgets:
            if isinstance(widget, QLabel):
                continue
            widget.load_settings()

    def save_all_settings(self):
        logger.info("Saving all preference settings.")
        for widget in self.settings_widgets:
            if isinstance(widget, QLabel):
                continue
            widget.save_settings()
        self.settings_saved.emit()
        QMessageBox.information(
            self, "Success", "Settings have been saved successfully."
        )

    def load_all_settings(self):
        logger.info("Loading and applying all preference settings.")
        for widget in self.settings_widgets:
            if isinstance(widget, QLabel):
                continue
            widget.load_settings()

    def clear_all_settings(self):
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "Are you sure you want to clear all settings? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.warning("Clearing all application settings.")
            self.settings.clear()
            self.main_window.set_current_folder(None)
            self.settings.sync()
            self.settings_saved.emit()
            self.load_all_settings()
            self.save_all_settings()
            QMessageBox.information(
                self, "Settings Cleared", "All settings have been cleared."
            )
