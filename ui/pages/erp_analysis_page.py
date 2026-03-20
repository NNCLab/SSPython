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
from core.processing import TMSEEGAnalysis
from ..widgets.evoked_plot import EvokedPlotWidget
from ..widgets.tools.object_info_widget import ObjectInfoWidget
from ..widgets.tep_analysis.natural_frequency import NaturalFrequencyApp
from ..widgets.tep_analysis.excitability import ExcitabilityApp
from utils import Worker, parse_tuple

logger = logging.getLogger(__name__)


class ErpAnalysisPage(BasePage):
    """
    A page for performing post-preprocessing analyses, such as Excitability,
    Natural Frequency, and Time-Frequency analysis.
    """

    def __init__(self, parent=None):
        super().__init__("<h1>ERP/TEP Analysis</h1>", parent)
        self.main_window = parent
        self.settings = QSettings()

        # Data & widget placeholders
        self.tep: TMSEEGAnalysis | None = None
        self.current_file_path: Path | None = None
        self.epochs_info_widget = ObjectInfoWidget()

        self._setup_ui()
        self._setup_connections()
        self._connect_to_main_window()

    # UI Creation
    def _setup_ui(self):
        """Creates and arranges all UI components."""
        file_group = self._create_file_group()
        self.visualization_group = self._create_visualization_group()
        self.analysis_group = self._create_analysis_group()

        groups = {
            "File Selection": file_group,
            "Visualization": self.visualization_group,
            "Analysis Methods": self.analysis_group,
        }

        for title, widget in groups.items():
            title_widget = QLabel(f"<h2>{title}</h2>")
            widget.setTitle("")
            self.add_content(title_widget)
            self.add_content(widget)

        self.update_ui_state()

    def _create_file_group(self) -> QGroupBox:
        """Creates the file selection group box for preprocessed files."""
        group = QGroupBox()
        layout = QHBoxLayout(group)
        vbox = QVBoxLayout()
        self.select_folder_button = QPushButton("Select Data Folder")
        self.file_filter = QLineEdit()
        self.file_filter.setPlaceholderText("Filter preprocessed files by name...")
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.setMinimumHeight(150)
        for widget in [self.select_folder_button, self.file_filter, self.file_list]:
            vbox.addWidget(widget)
        layout.addLayout(vbox)
        layout.addWidget(self.epochs_info_widget)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)
        self.evoked_plot_widget = EvokedPlotWidget()
        self.evoked_plot_widget.scrolled.connect(
            lambda e: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 50
                if e == "up"
                else self.scroll_area.verticalScrollBar().value() + 50
            )
        )
        self.topoplot_button = QPushButton("Topoplot")
        layout.addWidget(self.evoked_plot_widget)
        layout.addWidget(self.topoplot_button)
        return group

    def _create_analysis_group(self) -> QGroupBox:
        """Creates the group box for analysis action buttons."""
        group = QGroupBox()
        layout = QVBoxLayout(group)
        self.excitability_button = QPushButton("Response Amplitude Analysis")
        self.nf_button = QPushButton("Natural Frequency Analysis")
        self.tf_button = QPushButton("Run Time-Frequency Analysis")
        self.tf_button.setDisabled(True)

        for widget in [self.excitability_button, self.nf_button, self.tf_button]:
            layout.addWidget(widget)
        return group

    # Connections and State Management
    def _setup_connections(self):
        """Connects all widget signals to slots."""
        self.select_folder_button.clicked.connect(self.select_folder)
        self.file_list.itemClicked.connect(self.on_file_selected)
        self.file_filter.textChanged.connect(self.update_file_list)
        self.topoplot_button.clicked.connect(self.topoplot)
        self.excitability_button.clicked.connect(self.run_excitability)
        self.nf_button.clicked.connect(self.run_natural_frequency)
        self.tf_button.clicked.connect(self.run_time_frequency)

    def _connect_to_main_window(self):
        """Connects to signals from the main window."""
        if self.main_window:
            self.main_window.current_folder_changed.connect(self.on_folder_updated)
            self.on_folder_updated(self.main_window.current_folder)

    def update_ui_state(self):
        """Updates UI elements based on whether preprocessed data is loaded."""
        has_data = self.tep is not None and self.tep.epochs is not None

        if has_data:
            self.epochs_info_widget.update_info(
                self.tep.epochs, "Preprocessed epoched: "
            )
        else:
            self.epochs_info_widget.update_info(None)
        if has_data:
            self.evoked_plot_widget.update_plot(self.tep.evoked, self.tep.label)

        # Enable/disable analysis buttons
        self.topoplot_button.setEnabled(has_data)
        self.excitability_button.setEnabled(has_data)
        self.evoked_plot_widget.setVisible(has_data)
        self.nf_button.setEnabled(has_data)
        self.tf_button.setEnabled(has_data)
        self.update_file_list()

    @Slot()
    def on_settings_updated(self):
        """This Slot is called by MainWindow when application settings have changed."""
        logger.info(f"TEP Analysis Page is updating its settings.")
        self.tep = None
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
        """
        Filters and displays the list of raw .fif files.
        This method is now 'read-only' regarding application state.
        It does NOT change self.preprocessor or call update_ui_state().
        """
        # Store the current selection's file path to restore it later
        current_selection_path = None
        if self.file_list.currentItem():
            current_selection_path = self.file_list.currentItem().data(
                Qt.ItemDataRole.UserRole
            )

        self.file_list.blockSignals(True)
        self.file_list.clear()

        folder = self.main_window.current_folder
        if folder is None:
            self.file_list.blockSignals(False)
            return

        if not folder.is_dir():
            self.file_list.blockSignals(False)
            self.main_window.current_folder
            return

        try:
            files = sorted(
                (folder / self.main_window.output_dir).rglob("*_preprocessed-epo.fif")
            )
            filter_text = self.file_filter.text().lower().split()

            item_to_reselect = None
            for f in files:
                if all(word in f.name.lower() for word in filter_text):
                    label = "_".join(f.name.split("_")[:-1])
                    item = QListWidgetItem(label)
                    item.setData(Qt.ItemDataRole.UserRole, f)
                    item.setToolTip(str(f))
                    self.file_list.addItem(item)

                    # Check if this is the item we need to re-select
                    if f == current_selection_path:
                        item_to_reselect = item

            # Restore the previous selection
            if item_to_reselect:
                self.file_list.setCurrentItem(item_to_reselect)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not read files from folder: {e}"
            )
        finally:
            # IMPORTANT: Always unblock signals
            self.file_list.blockSignals(False)

    def on_file_selected(self, item: QListWidgetItem):
        """Handles file selection by loading it in a background thread."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path:
            return
        self.current_file_path = file_path
        self.tep = None
        self.update_ui_state()
        output_dir = self.main_window.current_folder / self.main_window.output_dir
        worker = Worker(
            lambda: TMSEEGAnalysis(file_path, output_dir=output_dir, preload=True),
            parent=self,
            add_loggers="mne",
        )
        self.tep = worker.exec_with_dialog(
            "Please wait", "Loading preprocessed data..."
        )
        self.update_ui_state()

    # Analysis Execution
    def topoplot(self):
        if not self.tep:
            return
        xlim = QSettings().value("plot_settings/evoked_xlim", "(-200, 500)")
        xlim = parse_tuple(xlim, float, scale=1e-3)
        evoked = self.tep.evoked.copy().apply_baseline().crop(*xlim)
        fig = evoked.plot_topo(
            title=self.tep.label,
        )
        for line in fig.axes[0].lines:
            line.set_linewidth(1.25)

    def run_excitability(self):
        if not self.tep:
            return
        self.exc_app = ExcitabilityApp(self.tep.epochs, parent=self)
        logger.info("Launched Excitability Analysis App.")

    def run_natural_frequency(self):
        if not self.tep:
            return
        nf_app = NaturalFrequencyApp(self.tep.epochs, parent=self)
        nf_app.exec()

    def run_time_frequency(self):
        """Runs the time-frequency analysis in a background thread."""
        if not self.tep:
            return
        worker = Worker(
            lambda: self.tep.run_tfr_analysis(overwrite=True),
            parent=self,
            add_loggers="mne",
        )
        worker.exec_with_dialog("Please wait", "Running Time-Frequency Analysis...")
        self.update_ui_state()
