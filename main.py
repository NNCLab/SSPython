import logging
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSettings,
    QSize,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSplashScreen,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_settings_store
from core.exporting import (
    evoked_export_targets,
    export_evoked_files,
    export_preprocessed_files,
    preprocessed_export_targets,
)
from core.pipelines import all_pipelines, get_pipeline
from core.version import __version__
from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.pages.home_page import HomePage
from ui.pages.preferences_page import PreferencesPage
from ui.pages.preprocessing_page import ProcessingPage
from ui.pages.real_time_page import RealTimePage
from ui.widgets.tools.conversion_tool import ConvertToolDialog, MergeToolDialog
from ui.widgets.tools.export_tool import ExportFilesDialog
from ui.widgets.tools.qss_helper import QSSEditorDialog
from ui.widgets.workspace_panel import DatasetInspectorPanel, WorkspacePanel
from utils import Worker, apply_theme, get_path, themed_svg_icon, toggle_theme as toggle_app_theme

os.environ["MNE_FORCE_EAGER"] = "1"

APP_NAME = "SSPython"
ORGANIZATION_NAME = "SSPython"
APP_VERSION = __version__

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _validate_frozen_dependencies() -> None:
    """Exercise dynamic imports and native resources used by packaged features."""
    import inspect
    from importlib.resources import files
    from tempfile import TemporaryDirectory

    import pyvista
    import pyvistaqt
    import torch
    from h5io import read_hdf5, write_hdf5
    from mne_icalabel.iclabel.network.torch import ICLabelNet
    from mne_lsl.lsl.load_liblsl import load_liblsl
    from mne.viz.backends.renderer import _TimeInteraction
    from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderer

    lsl = load_liblsl()
    if lsl.lsl_library_version() <= 0:
        raise RuntimeError("The bundled liblsl library did not initialize.")

    network_file = files("mne_icalabel.iclabel.network") / "assets" / "ICLabelNet.pt"
    network = ICLabelNet()
    network.load_state_dict(torch.load(network_file, weights_only=True))

    h5io_payload = {
        "version": 1,
        "channels": ["C3", "C4"],
        "values": [1.25, 2.5],
    }
    with TemporaryDirectory(prefix="sspython-h5io-") as temp_directory:
        h5io_file = Path(temp_directory) / "roundtrip.h5"
        write_hdf5(h5io_file, h5io_payload, overwrite=True)
        if read_hdf5(h5io_file) != h5io_payload:
            raise RuntimeError("The bundled h5io backend failed its round-trip check.")

    renderer_source = inspect.getsource(_TimeInteraction._enable_time_interaction)
    if "@_auto_weakref" not in renderer_source:
        raise RuntimeError("The bundled MNE STC renderer source is unavailable.")

    # Keep these references live so the imports cannot be optimized away.
    if not pyvista.__version__ or not pyvistaqt.__version__ or vtkOpenGLRenderer is None:
        raise RuntimeError("The bundled PyVista/VTK backend did not initialize.")


class MainWindow(QMainWindow):
    current_folder_changed = Signal(object)
    current_pipeline_changed = Signal(object)
    current_dataset_changed = Signal(object)


    GEOMETRY_SETTING = "ui/main_window/geometry"
    SPLITTER_STATE_SETTING = "ui/main_window/splitter_state"
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
        ("assets/icons/erp_analysis.svg", "Analysis", ErpAnalysisPage),
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
        self._compact_layout_active = False

        self.page_lookup: dict[str, QWidget] = {}

        self._setup_ui()
        self._setup_menu()
        self._populate_navigation()
        self._read_window_state()
        self._apply_responsive_layout(self.width(), force=True)
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

        self.shell_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.shell_splitter.setObjectName("shellSplitter")
        self.shell_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.shell_splitter, 1)

        self.workspace_panel = WorkspacePanel(self)
        self.workspace_panel.folder_selected.connect(self.set_current_folder)
        self.workspace_panel.dataset_selected.connect(self.set_current_dataset)
        self.shell_splitter.addWidget(self.workspace_panel)

        content_frame = QFrame()
        content_frame.setObjectName("contentFrame")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("pageStack")
        content_layout.addWidget(self.stacked_widget)
        self.shell_splitter.addWidget(content_frame)

        self.dataset_inspector_panel = DatasetInspectorPanel(self)
        self.dataset_inspector_panel.collapsed = self.inspector_collapsed
        self.dataset_inspector_panel._apply_collapsed_state(animated=False)
        self.dataset_inspector_panel.derivatives_about_to_be_deleted.connect(
            self._release_derivative_resources
        )
        self.dataset_inspector_panel.derivatives_changed.connect(self.refresh_workspace)
        self.shell_splitter.addWidget(self.dataset_inspector_panel)
        self.shell_splitter.setStretchFactor(0, 0)
        self.shell_splitter.setStretchFactor(1, 1)
        self.shell_splitter.setStretchFactor(2, 0)
        self.shell_splitter.setSizes([320, 980, 300])

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

        self.file_menu = menu_bar.addMenu("&File")
        self._add_action(self.file_menu, "Open Workspace", self.workspace_panel.select_folder, "Ctrl+O")
        self.file_menu.addSeparator()
        self.export_menu = self.file_menu.addMenu("Export")
        self._add_action(
            self.export_menu,
            "Export Preprocessed Files...",
            self._export_preprocessed_files,
        )
        self._add_action(
            self.export_menu,
            "Export Evoked Files...",
            self._export_evoked_files,
        )
        self.file_menu.addSeparator()
        self.conversion_menu = self.file_menu.addMenu("Convert / Merge EEG data")
        self._add_action(self.conversion_menu, "Convert EEG data", self._open_convert_tool)
        self._add_action(self.conversion_menu, "Merge FIF data", self._open_merge_tool)
        self.file_menu.addSeparator()
        self._add_action(self.file_menu, "Exit", self.close)

        self.view_menu = menu_bar.addMenu("&View")
        self._add_action(self.view_menu, "Toggle Sidebar", self.toggle_sidebar, "Ctrl+B")
        self._add_action(self.view_menu, "Toggle Derivative Inspector", self.toggle_derivative_inspector, "Ctrl+I")
        self._add_action(self.view_menu, "Toggle Light/Dark Theme", self.toggle_theme)
        self._add_action(
            self.view_menu,
            "Toggle Fullscreen",
            lambda: self.showFullScreen() if not self.isFullScreen() else self.showNormal(),
            "F11",
        )
        self.view_menu.addSeparator()

        self.pipeline_menu = menu_bar.addMenu("&Pipeline")
        self.pipeline_actions = {}
        for pipeline in all_pipelines():
            action = self._add_action(
                self.pipeline_menu,
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

        splitter_state = self.settings_store.get(self.SPLITTER_STATE_SETTING, None)
        if splitter_state:
            self.shell_splitter.restoreState(splitter_state)

    def _write_window_state(self):
        self.settings_store.set(self.GEOMETRY_SETTING, self.saveGeometry())
        self.settings_store.set(self.SPLITTER_STATE_SETTING, self.shell_splitter.saveState())
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
        uses_workspace = page_name in {"Home", "Preprocessing", "Analysis"}
        uses_inspector = page_name in {"Preprocessing", "Analysis"}
        if uses_workspace:
            self.workspace_panel.set_analysis_mode(page_name == "Analysis")
        show_inspector = uses_inspector and self.current_dataset is not None
        self.workspace_panel.setVisible(uses_workspace)
        self.dataset_inspector_panel.setVisible(show_inspector)

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
        current_item = self.nav_list.currentItem()
        if current_item is not None:
            self._apply_page_chrome_visibility(current_item.text())
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

    @Slot(object)
    def _release_derivative_resources(self, paths):
        """Ask every page to drop references to files about to be deleted."""
        for page in self.page_lookup.values():
            release = getattr(page, "release_derivative_resources", None)
            if callable(release):
                release(paths)

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def _apply_responsive_layout(self, width: int, *, force: bool = False):
        compact_layout = width < 1320
        if not force and compact_layout == self._compact_layout_active:
            return
        self._compact_layout_active = compact_layout
        if hasattr(self, "dataset_inspector_panel"):
            self.dataset_inspector_panel.set_responsive_collapsed(compact_layout)

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
    def _open_convert_tool(self):
        dialog = ConvertToolDialog(parent=self)
        dialog.exec()

    def _open_merge_tool(self):
        dialog = MergeToolDialog(self)
        dialog.exec()

    def _export_preprocessed_files(self):
        self._open_workspace_export_dialog(
            mode="preprocessed",
            export_label="preprocessed",
            target_builder=preprocessed_export_targets,
            exporter=export_preprocessed_files,
        )

    def _export_evoked_files(self):
        self._open_workspace_export_dialog(
            mode="evoked",
            export_label="evoked",
            target_builder=evoked_export_targets,
            exporter=export_evoked_files,
        )

    def _workspace_preprocessed_paths(self) -> list[Path]:
        source_paths: list[Path] = []
        seen_paths: set[str] = set()
        for dataset in self.workspace_panel.datasets:
            if not dataset.stage_exists("preprocessed"):
                continue
            source_path = dataset.paths["preprocessed"]
            source_key = str(source_path.resolve()).casefold()
            if source_key not in seen_paths:
                seen_paths.add(source_key)
                source_paths.append(source_path)
        return source_paths

    def _open_workspace_export_dialog(
        self,
        *,
        mode,
        export_label,
        target_builder,
        exporter,
    ):
        if self.current_folder is None:
            QMessageBox.warning(
                self,
                "No workspace open",
                "Open a workspace before exporting files.",
            )
            return

        source_paths = self._workspace_preprocessed_paths()
        if not source_paths:
            QMessageBox.information(
                self,
                "No preprocessed files",
                "No preprocessed epochs files were found in the active workspace.",
            )
            return

        dialog = ExportFilesDialog(
            source_paths,
            self.current_folder,
            mode,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        source_paths = dialog.selected_source_paths()
        destination = dialog.destination_path()
        if destination is None:
            return

        try:
            targets = target_builder(source_paths, destination)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Unable to export", str(exc))
            return

        existing_targets = [target for _, target in targets if target.exists()]
        overwrite = False
        if existing_targets:
            file_word = "file" if len(existing_targets) == 1 else "files"
            response = QMessageBox.question(
                self,
                "Replace existing files?",
                f"{len(existing_targets)} export {file_word} already exist. Replace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        worker = Worker(
            exporter,
            source_paths,
            destination,
            overwrite=overwrite,
            parent=self,
            add_loggers=["", "mne"],
        )
        exported_paths = worker.exec_with_dialog(
            "Exporting files",
            f"Exporting {len(source_paths)} {export_label} file(s)...",
        )
        if exported_paths is None:
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Exported {len(exported_paths)} {export_label} file(s) to:\n{destination}",
        )

    def _open_qss_dialog(self):
        dialog = QSSEditorDialog(self)
        dialog.exec()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    smoke_test = "--smoke-test" in args
    qt_args = [arg for arg in args if arg != "--smoke-test"]

    app = QApplication(qt_args)
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

    if smoke_test:
        _validate_frozen_dependencies()
        QTimer.singleShot(0, app.quit)

    return app.exec()


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        smoke_log_path = os.environ.get("SSPYTHON_SMOKE_LOG")
        if smoke_log_path:
            Path(smoke_log_path).write_text(traceback.format_exc(), encoding="utf-8")
        raise
    raise SystemExit(exit_code)
