import logging
from pathlib import Path
from PySide6.QtCore import QSettings, Qt, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from .base_page import BasePage
from core.processing import Preprocessor
from ..widgets.psd_plot import PSDPlotWidget
from ..widgets.tools.object_info_widget import ObjectInfoWidget
from utils import Worker

logger = logging.getLogger(__name__)


class ContinuousAnalysisPage(BasePage):
    """
    A page for performing analyses on continuous (raw) data, such as
    displaying the Power Spectral Density (PSD).
    """

    def __init__(self, parent=None):
        super().__init__("<h1>Continuous Analysis</h1>", parent)
        self.main_window = parent
        self.settings = QSettings()

        # Data & widget placeholders
        self.preprocessor: Preprocessor | None = None
        self.current_file_path: Path | None = None
        self.raw_info_widget = ObjectInfoWidget()

        self._setup_ui()
        self._setup_connections()
        self._connect_to_main_window()

    # UI Creation
    def _setup_ui(self):
        """Creates and arranges all UI components."""
        file_group = self._create_file_group()
        self.visualization_group = self._create_visualization_group()

        groups = {
            "File Selection": file_group,
            "Visualization": self.visualization_group,
        }

        for title, widget in groups.items():
            title_widget = QLabel(f"<h2>{title}</h2>")
            widget.setTitle("")
            self.add_content(title_widget)
            self.add_content(widget)

        self.update_ui_state()

    def _create_file_group(self) -> QGroupBox:
        """Creates the file selection group box for raw files."""
        group = QGroupBox()
        layout = QHBoxLayout(group)
        vbox = QVBoxLayout()
        self.select_folder_button = QPushButton("Select Data Folder")
        self.file_filter = QLineEdit()
        self.file_filter.setPlaceholderText("Filter raw files by name...")
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.setMinimumHeight(150)
        for widget in [self.select_folder_button, self.file_filter, self.file_list]:
            vbox.addWidget(widget)
        layout.addLayout(vbox)
        layout.addWidget(self.raw_info_widget)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        """Creates the group box for the PSD plot."""
        group = QGroupBox()
        layout = QVBoxLayout(group)
        self.psd_plot_widget = PSDPlotWidget()
        self.psd_plot_widget.scrolled.connect(
            lambda e: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 50
                if e == "up"
                else self.scroll_area.verticalScrollBar().value() + 50
            )
        )
        layout.addWidget(self.psd_plot_widget)
        return group

    # Connections and State Management
    def _setup_connections(self):
        """Connects all widget signals to slots."""
        self.select_folder_button.clicked.connect(self.select_folder)
        self.file_list.itemClicked.connect(self.on_file_selected)
        self.file_filter.textChanged.connect(self.update_file_list)

    def _connect_to_main_window(self):
        """Connects to signals from the main window."""
        if self.main_window:
            self.main_window.current_folder_changed.connect(self.on_folder_updated)
            self.on_folder_updated(self.main_window.current_folder)

    def update_ui_state(self):
        """Updates UI elements based on whether data is loaded."""
        has_data = self.preprocessor is not None

        if has_data:
            continuous_data = self.preprocessor._get_last_continuous()
            self.raw_info_widget.update_info(continuous_data, "Continuous data: ")
            self.psd_plot_widget.update_plot(continuous_data, self.preprocessor.label)
        else:
            self.raw_info_widget.update_info(None)
            self.psd_plot_widget.update_plot(None)

        self.psd_plot_widget.setVisible(has_data)
        self.update_file_list()

    @Slot()
    def on_settings_updated(self):
        """This slot is called by MainWindow when application settings have changed."""
        logger.info("Continuous Analysis Page is updating its settings.")
        self.preprocessor = None
        self.update_ui_state()

    # File Handling Slots
    @Slot(Path)
    def on_folder_updated(self, folder_path: Path):
        """Updates the UI when the main application folder changes."""
        if folder_path:
            self.select_folder_button.setText(f"Folder: {folder_path.name}")
        else:
            self.select_folder_button.setText("Select Data Folder")
        self.update_ui_state()

    def select_folder(self):
        """Opens a dialog to select a new data folder."""
        if not self.main_window:
            return

        folder_path_str = QFileDialog.getExistingDirectory(
            self, "Select EEG Data Folder", str(self.main_window.current_folder)
        )
        if folder_path_str:
            self.main_window.set_current_folder(folder_path_str)

    def update_file_list(self):
        """Filters and displays the list of raw .fif files."""
        self.file_list.clear()
        folder = self.main_window.current_folder
        if not (folder and folder.is_dir()):
            return

        try:
            files = sorted(folder.rglob("*preprocessed-epo.fif"))
            filter_text = self.file_filter.text().lower().split()
            for f in files:
                if all(word in f.name.lower() for word in filter_text):
                    label = "_".join(f.name.split("_")[:-1])
                    item = QListWidgetItem(label)
                    item.setData(Qt.ItemDataRole.UserRole, f)
                    self.file_list.addItem(item)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not read files from folder: {e}"
            )

    def on_file_selected(self, item: QListWidgetItem):
        """Handles file selection by loading it in a background thread."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path:
            return

        self.current_file_path = file_path
        self.preprocessor = None
        self.update_ui_state()
        output_dir = self.main_window.current_folder / self.main_window.output_dir

        worker = Worker(
            lambda: Preprocessor(file_path, output_dir), parent=self, add_loggers="mne"
        )
        self.preprocessor = worker.exec_with_dialog(
            "Please wait", "Loading continuous data..."
        )
        self.update_ui_state()
