import logging
import sys
from importlib import metadata
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, Qt, QSettings, Signal, QSize, Slot
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplashScreen,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_settings_store
from core.pipelines import all_pipelines, get_pipeline
from ui.pages.continuous_analysis_page import ContinuousAnalysisPage
from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.pages.home_page import HomePage
from ui.pages.preferences_page import PreferencesPage
from ui.pages.preprocessing_page import ProcessingPage
from ui.pages.real_time_page import RealTimePage
from ui.widgets.tools.convert_brainamp import BrainampConverter
from ui.widgets.tools.convert_gtec import GtecConverter
from ui.widgets.tools.qss_helper import QSSEditorDialog
from ui.widgets.workspace_panel import DatasetInspectorPanel, WorkspacePanel
from utils import apply_theme, get_path, themed_svg_icon, toggle_theme as toggle_app_theme

import os
os.environ["MNE_FORCE_EAGER"] = "1"

APP_NAME = "SSPython"
ORGANIZATION_NAME = "SSPython"

try:
    APP_VERSION = metadata.version("sspython")
except metadata.PackageNotFoundError:
    APP_VERSION = "0.1.1"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    current_folder_changed = Signal(object)
    current_pipeline_changed = Signal(object)
    current_dataset_changed = Signal(object)

    GEOMETRY_SETTING = "ui/main_window/geometry"
    SIDEBAR_COLLAPSED_SETTING = "ui/main_window/sidebar_collapsed"
    INSPECTOR_COLLAPSED_SETTING = "ui/main_window/inspector_collapsed"

    NAV_LIST_OBJECT_NAME = "navList"
    SIDEBAR_OBJECT_NAME = "sidebar"

    PAGE_DEFINITIONS = [
        ("assets/icon.png", "Home", HomePage),
        None,
        ("assets/icons/realtime.svg", "Real-Time", RealTimePage),
        None,
        ("assets/icons/erp_preprocess.svg", "Preprocessing", ProcessingPage),
        None,
        ("assets/icons/cont_analysis.svg", "Continuous Analysis", ContinuousAnalysisPage),
        ("assets/icons/erp_analysis.svg", "ERP Analysis", ErpAnalysisPage),
        None,
        ("assets/icons/settings.svg", "Preferences", PreferencesPage),
    ]

    PageWidgetRole = Qt.ItemDataRole.UserRole + 1
    PageIconRole = Qt.ItemDataRole.UserRole + 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")

        self.settings = QSettings()
        self.settings_store = get_settings_store(self.settings)

        self.current_folder = self.settings_store.current_folder()
        if self.current_folder and not self.current_folder.exists():
            self.current_folder = None

        self.output_root = self.settings_store.output_root()
        self.current_pipeline = get_pipeline(self.settings_store.current_pipeline_id())
        self.current_dataset = None
        self.sidebar_expanded_width = 236
        self.sidebar_collapsed_width = 84
        self.sidebar_collapsed = self.settings_store.get(
            self.SIDEBAR_COLLAPSED_SETTING,
            False,
            value_type=bool,
        )
        self.inspector_collapsed = self.settings_store.get(
            self.INSPECTOR_COLLAPSED_SETTING,
            False,
            value_type=bool,
        )

        self.page_lookup: dict[str, QWidget] = {}

        self._setup_ui()
        self._setup_menu()
        self._populate_navigation()
        self._read_window_state()
        self.refresh_workspace()

    def _setup_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("appShell")
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar_widget = self._create_sidebar()
        layout.addWidget(self.sidebar_widget)

        self.workspace_panel = WorkspacePanel(self)
        self.workspace_panel.folder_selected.connect(self.set_current_folder)
        self.workspace_panel.dataset_selected.connect(self.set_current_dataset)
        layout.addWidget(self.workspace_panel)

        content_frame = QFrame()
        content_frame.setObjectName("contentFrame")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)
        layout.addWidget(content_frame, 1)

        self.dataset_inspector_panel = DatasetInspectorPanel(self)
        self.dataset_inspector_panel.collapsed = self.inspector_collapsed
        self.dataset_inspector_panel._apply_collapsed_state(animated=False)
        layout.addWidget(self.dataset_inspector_panel)

        self.nav_list.currentItemChanged.connect(self._change_page)
        self._apply_sidebar_state(animated=False)

    def _create_sidebar(self) -> QWidget:
        sidebar_widget = QFrame()
        sidebar_widget.setObjectName(self.SIDEBAR_OBJECT_NAME)

        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(14, 18, 14, 18)
        sidebar_layout.setSpacing(12)

        chrome_row = QHBoxLayout()
        chrome_row.setContentsMargins(0, 0, 0, 0)
        chrome_row.setSpacing(8)

        self.sidebar_brand = QLabel("SSPython")
        self.sidebar_brand.setObjectName("sidebarBrand")
        chrome_row.addWidget(self.sidebar_brand, 1)

        self.toggle_sidebar_button = QPushButton("")
        self.toggle_sidebar_button.setObjectName("sidebarToggle")
        self.toggle_sidebar_button.clicked.connect(self.toggle_sidebar)
        chrome_row.addWidget(self.toggle_sidebar_button)
        sidebar_layout.addLayout(chrome_row)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName(self.NAV_LIST_OBJECT_NAME)
        self.nav_list.setIconSize(QSize(20, 20))
        self.nav_list.setSpacing(0)
        sidebar_layout.addWidget(self.nav_list, 1)

        self.theme_button = QPushButton("Theme")
        self.theme_button.setObjectName("sidebarActionButton")
        self.theme_button.clicked.connect(self.toggle_theme)
        sidebar_layout.addWidget(self.theme_button)

        self.sidebar_footer = QLabel("Navigate, analyze, and visualize\nyour EEG data with ease. \nMade by: Couto, BAN.")
        self.sidebar_footer.setObjectName("sidebarFooter")
        self.sidebar_footer.setWordWrap(True)
        sidebar_layout.addWidget(self.sidebar_footer)

        return sidebar_widget

    def _setup_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        self._add_action(file_menu, "Open Workspace", self.workspace_panel.select_folder, "Ctrl+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "Convert Brainamp data", self._convert_brainamp_data)
        self._add_action(file_menu, "Convert gTEC data", self._convert_gtec_data)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close)

        view_menu = menu_bar.addMenu("&View")
        self._add_action(view_menu, "Toggle Sidebar", self.toggle_sidebar, "Ctrl+B")
        self._add_action(view_menu, "Toggle Derivative Inspector", self.toggle_derivative_inspector, "Ctrl+I")
        self._add_action(view_menu, "Toggle Light/Dark Theme", self.toggle_theme)
        self._add_action(
            view_menu,
            "Toggle Fullscreen",
            lambda: self.showFullScreen() if not self.isFullScreen() else self.showNormal(),
            "F11",
        )
        view_menu.addSeparator()

        pipeline_menu = menu_bar.addMenu("&Pipeline")
        self.pipeline_actions = {}
        for pipeline in all_pipelines():
            action = self._add_action(
                pipeline_menu,
                pipeline.name,
                lambda checked=False, pipeline_id=pipeline.id: self.set_current_pipeline(pipeline_id),
            )
            action.setCheckable(True)
            action.setChecked(pipeline.id == self.current_pipeline.id)
            self.pipeline_actions[pipeline.id] = action

    def _add_action(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(slot)
        menu.addAction(action)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _populate_navigation(self):
        for page_def in self.PAGE_DEFINITIONS:
            if page_def is None:
                separator = QListWidgetItem("")
                separator.setFlags(Qt.ItemFlag.NoItemFlags)
                separator.setSizeHint(QSize(0, 12))
                self.nav_list.addItem(separator)
                continue

            icon_path, name, page_class = page_def

            if page_class == PreferencesPage:
                page_widget = PreferencesPage(self.settings_store, self)
                page_widget.settings_saved.connect(self._on_settings_saved)
            else:
                page_widget = page_class(self)

            self.page_lookup[name] = page_widget
            self.stacked_widget.addWidget(page_widget)

            item = QListWidgetItem(name)
            item.setSizeHint(QSize(0, 44))
            item.setData(self.PageWidgetRole, page_widget)
            item.setData(self.PageIconRole, icon_path)
            self.nav_list.addItem(item)

        self._refresh_icons()
        self.nav_list.setCurrentRow(0)

    def _read_window_state(self):
        geometry = self.settings_store.get(self.GEOMETRY_SETTING, None)
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.setGeometry(120, 80, 1600, 920)

    def _write_window_state(self):
        self.settings_store.set(self.GEOMETRY_SETTING, self.saveGeometry())
        self.settings_store.set(self.SIDEBAR_COLLAPSED_SETTING, self.sidebar_collapsed)
        self.settings_store.set(self.INSPECTOR_COLLAPSED_SETTING, self.dataset_inspector_panel.collapsed)
        self.settings_store.set_current_folder(self.current_folder)
        self.settings_store.set("workspace/output_root", self.output_root)
        self.settings_store.set_current_pipeline_id(self.current_pipeline.id)
        self.settings_store.sync()

    @Slot(QListWidgetItem)
    def _change_page(self, current_item: QListWidgetItem):
        if current_item is None:
            return

        page_widget = current_item.data(self.PageWidgetRole)
        if page_widget:
            self.stacked_widget.setCurrentWidget(page_widget)
            self._apply_page_chrome_visibility(current_item.text())
            if hasattr(page_widget, "update_ui_state"):
                page_widget.update_ui_state()

    def _apply_page_chrome_visibility(self, page_name: str):
        show_workspace_shell = page_name != "Real-Time"
        self.workspace_panel.setVisible(show_workspace_shell)
        self.dataset_inspector_panel.setVisible(show_workspace_shell)

    @Slot(str)
    def set_current_folder(self, folder_path: str | None):
        if not folder_path:
            self.current_folder = None
            self.current_folder_changed.emit(None)
            self.refresh_workspace()
            return

        path = Path(folder_path)
        if not (path.exists() and path.is_dir()):
            logger.warning("Attempted to set invalid workspace: %s", folder_path)
            return

        self.current_folder = path
        self.settings_store.set_current_folder(path)
        self.settings_store.sync()
        self.current_folder_changed.emit(path)
        self.refresh_workspace()

    @Slot(object)
    def set_current_dataset(self, dataset):
        self.current_dataset = dataset
        self.dataset_inspector_panel.set_dataset(dataset)
        self.current_dataset_changed.emit(dataset)

    def set_current_pipeline(self, pipeline_id: str):
        pipeline = get_pipeline(pipeline_id)
        if pipeline.id == self.current_pipeline.id and self.workspace_panel.pipeline is not None:
            return

        self.current_pipeline = pipeline
        self.settings_store.set_current_pipeline_id(pipeline.id)
        self.settings_store.sync()
        for action_pipeline_id, action in self.pipeline_actions.items():
            action.setChecked(action_pipeline_id == pipeline.id)
        self.current_pipeline_changed.emit(pipeline)
        self.refresh_workspace()

        current_page = self.stacked_widget.currentWidget()
        if current_page and hasattr(current_page, "update_ui_state"):
            current_page.update_ui_state()

    def refresh_workspace(self):
        self.output_root = self.settings_store.output_root()
        self.workspace_panel.set_context(
            workspace_root=self.current_folder,
            pipeline=self.current_pipeline,
            output_root=self.output_root,
        )

    def toggle_sidebar(self):
        self.sidebar_collapsed = not self.sidebar_collapsed
        self._apply_sidebar_state(animated=True)

    def toggle_derivative_inspector(self):
        self.dataset_inspector_panel.toggle_collapsed()

    def toggle_theme(self):
        toggle_app_theme()
        self._refresh_icons()
        for page_name, page_widget in self.page_lookup.items():
            target = getattr(page_widget, "widget", page_widget)
            if hasattr(target, "refresh_theme"):
                target.refresh_theme()

    def _icon_for_path(self, icon_path: str, *, size: int = 20) -> QIcon:
        if icon_path.lower().endswith(".svg"):
            return themed_svg_icon(icon_path, size=size)
        return QIcon(get_path(icon_path))

    def _refresh_icons(self):
        self.toggle_sidebar_button.setIcon(themed_svg_icon("assets/icons/menu.svg", size=20))
        self.toggle_sidebar_button.setIconSize(QSize(20, 20))
        self.theme_button.setIcon(themed_svg_icon("assets/icons/theme.svg", size=20))
        self.theme_button.setIconSize(QSize(20, 20))

        for row in range(self.nav_list.count()):
            item = self.nav_list.item(row)
            if item is None:
                continue
            icon_path = item.data(self.PageIconRole)
            if icon_path:
                item.setIcon(self._icon_for_path(icon_path, size=20))

        if hasattr(self.dataset_inspector_panel, "refresh_icons"):
            self.dataset_inspector_panel.refresh_icons()

    def _apply_sidebar_state(self, *, animated: bool):
        target_width = self.sidebar_collapsed_width if self.sidebar_collapsed else self.sidebar_expanded_width
        self.sidebar_brand.setVisible(not self.sidebar_collapsed)
        self.sidebar_footer.setVisible(not self.sidebar_collapsed)
        self.theme_button.setText("" if self.sidebar_collapsed else "Theme")
        self.toggle_sidebar_button.setToolTip("Expand sidebar" if self.sidebar_collapsed else "Collapse sidebar")

        if not animated:
            self.sidebar_widget.setMinimumWidth(target_width)
            self.sidebar_widget.setMaximumWidth(target_width)
            return

        group = QParallelAnimationGroup(self)
        for prop in (b"minimumWidth", b"maximumWidth"):
            animation = QPropertyAnimation(self.sidebar_widget, prop)
            animation.setDuration(180)
            animation.setStartValue(self.sidebar_widget.width())
            animation.setEndValue(target_width)
            animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
            group.addAnimation(animation)
        group.start()
        self._sidebar_animation = group

    def navigate_to_page(self, page_name: str):
        for row in range(self.nav_list.count()):
            item = self.nav_list.item(row)
            if item and item.text() == page_name:
                self.nav_list.setCurrentRow(row)
                return

    @Slot()
    def _on_settings_saved(self):
        updated_output_root = self.settings_store.output_root()
        updated_pipeline = get_pipeline(self.settings_store.current_pipeline_id())

        self.output_root = updated_output_root
        if updated_pipeline.id != self.current_pipeline.id:
            self.current_pipeline = updated_pipeline
            for action_pipeline_id, action in self.pipeline_actions.items():
                action.setChecked(action_pipeline_id == updated_pipeline.id)
            self.current_pipeline_changed.emit(updated_pipeline)

        self.refresh_workspace()

        for index in range(self.stacked_widget.count()):
            widget = self.stacked_widget.widget(index)
            if hasattr(widget, "on_settings_updated"):
                widget.on_settings_updated()

    def closeEvent(self, event):
        self._write_window_state()
        super().closeEvent(event)

    @Slot()
    def _convert_brainamp_data(self):
        dialog = BrainampConverter(self)
        dialog.exec()

    @Slot()
    def _convert_gtec_data(self):
        dialog = GtecConverter(self)
        dialog.exec()

    def _open_qss_dialog(self):
        dialog = QSSEditorDialog(self)
        dialog.exec()


def main():
    app = QApplication(sys.argv)
    QApplication.setOrganizationName(ORGANIZATION_NAME)
    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(APP_VERSION)

    apply_theme()
    app.setWindowIcon(QIcon(get_path("assets/icon.png")))
    app.setStyle("Fusion")

    splash = QSplashScreen(QPixmap(get_path("assets/icon.png")), Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage(
        "Initializing SSPython... Setting up your workspace.",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
    )
    app.processEvents()

    window = MainWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
