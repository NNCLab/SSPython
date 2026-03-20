import sys
import logging
from pathlib import Path
from importlib import metadata

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QListWidgetItem,
    QFrame,
    QSplashScreen,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt, QSize, QSettings, Slot, Signal
from PySide6.QtGui import QIcon, QPixmap, QAction

from ui.pages.preprocessing_page import ProcessingPage
from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.pages.continuous_analysis_page import ContinuousAnalysisPage
from ui.pages.real_time_page import RealTimePage
from ui.pages.preferences_page import PreferencesPage
from ui.pages.home_page import HomePage
from utils import get_path, apply_theme, toggle_theme
from ui.widgets.tools.convert_brainamp import BrainampConverter
from ui.widgets.tools.convert_gtec import GtecConverter
from ui.widgets.tools.qss_helper import QSSEditorDialog

# --- Constants ---
APP_NAME = "SSPython"
try:
    APP_VERSION = metadata.version("sspython")
except metadata.PackageNotFoundError:
    # This is a fallback for when the package is not installed, e.g., when running from source
    APP_VERSION = "0.1.0-dev"
ORGANIZATION_NAME = "SSPython"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    The main application window, featuring a navigation list and a stacked widget
    to display different pages.
    """

    current_folder_changed = Signal(Path)

    # --- Constants for settings and object names ---
    GEOMETRY_SETTING = "geometry"
    CURRENT_FOLDER_SETTING = "current_folder"
    OUTPUT_DIR_SETTING = "global_settings/output_dir"
    THEME_SETTING = "theme"
    NAV_LIST_OBJECT_NAME = "navList"
    SIDEBAR_OBJECT_NAME = "sidebar"

    # --- Page Definitions ---
    PAGE_DEFINITIONS = [
        ("assets/icon.png", "SSPython", HomePage),
        None,  # Separator
        ("assets/icons/realtime.svg", "Real-Time", RealTimePage),
        None,  # Separator
        ("assets/icons/erp_preprocess.svg", "Preprocessing", ProcessingPage),
        None,  # Separator
        (
            "assets/icons/cont_analysis.svg",
            "Continuous Analysis",
            ContinuousAnalysisPage,
        ),
        ("assets/icons/erp_analysis.svg", "ERP Analysis", ErpAnalysisPage),
        None,  # Separator
        ("assets/icons/settings.svg", "Preferences", PreferencesPage),
    ]

    # Custom role to store the widget reference in a QListWidgetItem
    PageWidgetRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.settings = QSettings()
        self.current_folder = None
        self.output_dir = "derivatives"

        self._read_settings()
        self._setup_ui()
        self._setup_menu()
        self._populate_navigation()

    def _setup_ui(self):
        """Initializes the main UI layout and widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Sidebar
        sidebar_widget = self._create_sidebar()
        main_layout.addWidget(sidebar_widget)

        # Main content area
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.nav_list.currentItemChanged.connect(self._change_page)

    def _create_sidebar(self) -> QWidget:
        """Creates the sidebar widget containing navigation and a watermark."""
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName(self.SIDEBAR_OBJECT_NAME)
        sidebar_widget.setFixedWidth(240)

        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Navigation List
        self.nav_list = QListWidget()
        self.nav_list.setObjectName(self.NAV_LIST_OBJECT_NAME)
        self.nav_list.setIconSize(QSize(24, 24))
        sidebar_layout.addWidget(self.nav_list)

        # Watermark
        watermark = QLabel("Under Development.\nCouto, B.A.N (2025)")
        watermark.setStyleSheet("color: rgba(126, 126, 126, 50)")
        watermark.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        watermark.setMinimumHeight(20)
        sidebar_layout.addWidget(watermark)

        return sidebar_widget

    def _setup_menu(self):
        """Creates the main menu bar with its actions."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        self._add_action(file_menu, "Convert Brainamp data", self._convert_brainamp_data)
        self._add_action(file_menu, "Convert gTEC data", self._convert_gtec_data)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close)

        # View Menu
        view_menu = menu_bar.addMenu("&View")
        self._add_action(view_menu, "Toggle Light/Dark Theme", toggle_theme)
        self._add_action(view_menu, "Toggle Fullscreen", lambda: self.showFullScreen() if not self.isFullScreen() else self.showNormal(), "F11")
        view_menu.addSeparator()
        self._add_action(view_menu, "Toggle QSS Editor (Test)", self._open_qss_dialog)

    def _add_action(self, menu, text, slot, shortcut=None):
        """Helper to create and add an action to a menu."""
        action = QAction(text, self)
        action.triggered.connect(slot)
        menu.addAction(action)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _populate_navigation(self):
        """Creates and adds pages to the navigation list and stacked widget."""
        for page_def in self.PAGE_DEFINITIONS:
            if page_def is None:
                self._add_separator()
                continue

            icon_path, name, PageClass = page_def
            page_widget = (
                PreferencesPage(self.settings, self)
                if PageClass == PreferencesPage
                else PageClass(self)
            )

            if isinstance(page_widget, PreferencesPage):
                page_widget.settings_saved.connect(self._on_settings_saved)

            self.stacked_widget.addWidget(page_widget)

            item = QListWidgetItem(QIcon(get_path(icon_path)), name)
            item.setSizeHint(QSize(0, 50))
            item.setData(self.PageWidgetRole, page_widget)
            self.nav_list.addItem(item)

        self.nav_list.setCurrentRow(0)

    def _add_separator(self):
        """Adds a visual separator to the navigation list."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        item = QListWidgetItem(self.nav_list)
        item.setSizeHint(separator.sizeHint())
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
        self.nav_list.setItemWidget(item, separator)

    @Slot(QListWidgetItem)
    def _change_page(self, current_item: QListWidgetItem):
        """Changes the visible page based on navigation list selection."""
        if current_item and (page_widget := current_item.data(self.PageWidgetRole)):
            self.stacked_widget.setCurrentWidget(page_widget)
            page_widget.update_ui_state()

    def _read_settings(self):
        """Reads and applies saved application settings."""
        if geometry := self.settings.value(self.GEOMETRY_SETTING):
            self.restoreGeometry(geometry)
        else:
            self.setGeometry(100, 100, 1280, 800)  # Default size

        if folder_str := self.settings.value(self.CURRENT_FOLDER_SETTING):
            folder_path = Path(folder_str)
            if folder_path.exists():
                self.current_folder = folder_path

        self.output_dir = self.settings.value(self.OUTPUT_DIR_SETTING, "derivatives")

    def _write_settings(self):
        """Saves application settings like geometry and paths."""
        self.settings.setValue(self.GEOMETRY_SETTING, self.saveGeometry())
        if self.current_folder:
            self.settings.setValue(self.CURRENT_FOLDER_SETTING, str(self.current_folder))
        self.settings.setValue(self.OUTPUT_DIR_SETTING, self.output_dir)

    @Slot(str)
    def set_current_folder(self, folder_path: str | None):
        """Sets the current working folder and notifies other widgets."""
        if folder_path is None:
            self.current_folder = None
            return

        path = Path(folder_path)
        if not (path.exists() and path.is_dir()):
            logger.warning(f"Attempted to set invalid folder: {folder_path}")
            return

        if self.current_folder != path:
            self.current_folder = path
            logger.info(f"Current folder changed to: {self.current_folder}")
            self.current_folder_changed.emit(self.current_folder)
            self._write_settings()

    @Slot()
    def _on_settings_saved(self):
        """Notifies all pages that settings have been updated."""
        logger.info("MainWindow received settings_saved Signal. Notifying all pages.")
        self.output_dir = self.settings.value(self.OUTPUT_DIR_SETTING, "derivatives")

        for i in range(self.stacked_widget.count()):
            widget = self.stacked_widget.widget(i)
            widget.on_settings_updated()

    def closeEvent(self, event):
        """Saves settings before exiting the application."""
        self._write_settings()
        super().closeEvent(event)

    # --- Data Conversion Dialogs ---
    @Slot()
    def _convert_brainamp_data(self):
        """Shows the Brainamp data conversion dialog."""
        logger.info("Triggered 'Convert Brainamp data' action.")
        dialog = BrainampConverter(self)
        if dialog.exec():
            logger.info("Brainamp conversion completed successfully.")

    @Slot()
    def _convert_gtec_data(self):
        """Shows the gTEC data conversion dialog."""
        logger.info("Triggered 'Convert gTEC data' action.")
        dialog = GtecConverter(self)
        if dialog.exec():
            logger.info("gTEC conversion completed successfully.")

    # --- Dialogs ---
    def _open_qss_dialog(self):
        """Opens the QSS editor dialog."""
        dialog = QSSEditorDialog(self)
        dialog.exec()

def main():
    """Main function to set up and run the application."""
    app = QApplication(sys.argv)
    QApplication.setOrganizationName(ORGANIZATION_NAME)
    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(APP_VERSION)

    # --- Theme & Icon ---
    apply_theme()
    app.setWindowIcon(QIcon(get_path("assets/icon.png")))
    app.setStyle("Fusion")

    # --- Splash Screen ---
    splash = QSplashScreen(
        QPixmap(get_path("assets/icon.png")),
        Qt.WindowType.WindowStaysOnTopHint
    )
    splash.show()
    splash.showMessage(
        "Loading application...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
    )
    app.processEvents()

    window = MainWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
