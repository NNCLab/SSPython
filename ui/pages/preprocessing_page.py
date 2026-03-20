import logging
from pathlib import Path
from PySide6.QtCore import QSettings, Qt, Slot
from PySide6.QtWidgets import (
    QGridLayout,
    QFileDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QProgressDialog,
)
from .base_page import BasePage
from core.processing import Preprocessor
from ..widgets.evoked_plot import EvokedPlotWidget
from ..widgets.tools.object_info_widget import ObjectInfoWidget
from ..widgets.run_ica_widget import RunICADialog
from ..widgets.preprocessing_widgets import (
    ArtifactRemovalDialog,
    FilterContinuousDialog,
    EpochingDialog,
    ReReferenceDialog,
    ApplyPreprocessingDialog,
)
from ..widgets.ica_widget import run_ica_viewer
from utils import Worker

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ProcessingPage(BasePage):
    """
    Page for handling the TEP preprocessing workflow, using ObjectInfoWidget
    to display details of MNE objects with an efficient update strategy.
    """

    def __init__(self, parent=None):
        super().__init__("<h1>Event Related Potentials Preprocessing</h1>", parent)
        self.main_window = parent
        self.preprocessor: Preprocessor | None = None
        self.settings = QSettings()

        # Placeholders for our dynamic info widgets
        self.raw_info_widget = ObjectInfoWidget()
        self.continuous_ica_info_widget = ObjectInfoWidget()
        self.epochs_info_widget = ObjectInfoWidget()
        self.epochs_ica_info_widget = ObjectInfoWidget()
        self.preprocessed_info_widget = ObjectInfoWidget()

        self._setup_ui()
        self._setup_connections()
        self._connect_to_main_window()

    # --- UI ----
    def _setup_ui(self):
        """Creates and arranges all UI components."""

        self.plot_widget = EvokedPlotWidget()
        self.plot_widget.scrolled.connect(
            lambda e: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 50
                if e == "up"
                else self.scroll_area.verticalScrollBar().value() + 50
            )
        )
        self.plot_widget.setVisible(False)
        self.raw_evoked_plot_widget = EvokedPlotWidget()

        file_group = self._create_file_group()
        self.continuous_group = self._create_continuous_group()
        self.epochs_group = self._create_epochs_group()
        self.preprocessed_group = self._create_preprocessed_group()

        groups = {
            "File Selection": file_group,
            "Continuous": self.continuous_group,
            "Epochs": self.epochs_group,
            "Preprocessing": self.preprocessed_group,
        }

        for title, widget in groups.items():
            title_widget = QLabel("<h2>" + title + "</h2>")
            title_widget.setAlignment(Qt.AlignmentFlag.AlignLeft)
            widget.setTitle("")
            self.add_content(title_widget)
            self.add_content(widget)

        self.update_ui_state()

    def _create_file_group(self) -> QGroupBox:
        """Creates the file selection group box."""
        group = QGroupBox("1. File Selection")
        layout = QVBoxLayout(group)
        self.select_folder_button = QPushButton("Select Data Folder")
        self.file_filter = QLineEdit()
        self.file_filter.setPlaceholderText("Filter files by name (e.g., 'sub-01')...")
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.setMinimumHeight(200)
        for widget in [self.select_folder_button, self.file_filter, self.file_list]:
            layout.addWidget(widget)
        return group

    def _create_continuous_group(self) -> QGroupBox:
        group = QGroupBox("1. Continuous")
        layout = QVBoxLayout(group)
        self.inspect_raw_button = QPushButton("Inspect Raw Data (Optional)")
        self.artifact_removal_button = QPushButton(
            "Stimulation Artifact Removal (Optional)"
        )
        self.filter_continuous_button = QPushButton("Filter && Resample (Optional)")
        vbox = QVBoxLayout()
        vbox.addWidget(self.inspect_raw_button)
        vbox.addWidget(self.artifact_removal_button)
        vbox.addWidget(self.filter_continuous_button)

        hbox = QHBoxLayout()
        hbox.addWidget(self.raw_info_widget)
        hbox.addLayout(vbox)
        layout.addLayout(hbox)

        self.run_continuous_ica_button = QPushButton("Run Continuous ICA (Optional)")
        self.inspect_continuous_ica_sources_button = QPushButton(
            "Inspect Continuous ICA Sources (Optional)"
        )
        self.inspect_continuous_ica_button = QPushButton(
            "Inspect Continuous ICA (Optional)"
        )
        vbox = QVBoxLayout()
        vbox.addWidget(self.run_continuous_ica_button)
        vbox.addWidget(self.inspect_continuous_ica_sources_button)
        vbox.addWidget(self.inspect_continuous_ica_button)

        hbox = QHBoxLayout()
        hbox.addWidget(self.continuous_ica_info_widget)
        hbox.addLayout(vbox)
        layout.addLayout(hbox)

        self.segment_button = QPushButton("Segment into Epochs")
        layout.addWidget(self.segment_button)
        return group

    def _create_epochs_group(self) -> QGroupBox:
        group = QGroupBox("3. Epochs")
        layout = QVBoxLayout(group)
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        self.plot_raw_evoked_button = QPushButton("Plot Raw Evoked (Optional)")
        self.inspect_epochs_button = QPushButton("Inspect Epochs")
        self.rereference_epochs_button = QPushButton("Re-referencing")
        vbox.addWidget(self.plot_raw_evoked_button)
        vbox.addWidget(self.inspect_epochs_button)
        vbox.addWidget(self.rereference_epochs_button)
        hbox.addWidget(self.epochs_info_widget)
        hbox.addLayout(vbox)
        layout.addLayout(hbox)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        self.run_epochs_ica_button = QPushButton("Run ICA")
        self.inspect_epochs_ica_button = QPushButton("Inspect ICA")
        vbox.addWidget(self.run_epochs_ica_button)
        vbox.addWidget(self.inspect_epochs_ica_button)
        hbox.addWidget(self.epochs_ica_info_widget)
        hbox.addLayout(vbox)
        layout.addLayout(hbox)
        return group

    def _create_preprocessed_group(self) -> QGroupBox:
        group = QGroupBox("5. Preprocessing")
        layout = QVBoxLayout(group)

        self.apply_filters_button = QPushButton("Apply ICA && Filters")

        hbox = QHBoxLayout()
        hbox.addWidget(self.preprocessed_info_widget)

        vbox = QVBoxLayout()
        vbox.addWidget(self.apply_filters_button)
        hbox.addLayout(vbox)

        layout.addLayout(hbox)
        layout.addWidget(self.plot_widget)
        return group

    def _setup_connections(self):
        """Connects all widget signals to slots."""
        self.select_folder_button.clicked.connect(self.select_folder)
        self.file_list.itemClicked.connect(self.on_file_selected)
        self.file_filter.textChanged.connect(self.update_file_list)

        self.inspect_raw_button.clicked.connect(self.inspect_raw_data)
        self.artifact_removal_button.clicked.connect(self.artifact_removal)
        self.filter_continuous_button.clicked.connect(self.filter_continuous)

        self.run_continuous_ica_button.clicked.connect(self.run_continuous_ica)
        self.inspect_continuous_ica_sources_button.clicked.connect(
            self.inspect_continuous_ica_sources
        )
        self.inspect_continuous_ica_button.clicked.connect(self.inspect_continuous_ica)
        self.segment_button.clicked.connect(self.segment_epochs)

        self.plot_raw_evoked_button.clicked.connect(self.plot_raw_evoked)
        self.inspect_epochs_button.clicked.connect(self.inspect_epochs)
        self.rereference_epochs_button.clicked.connect(self.rereference_epochs)

        self.run_epochs_ica_button.clicked.connect(self.run_epochs_ica)
        self.inspect_epochs_ica_button.clicked.connect(self.inspect_epochs_ica)

        self.apply_filters_button.clicked.connect(self.apply_filters)

    def _connect_to_main_window(self):
        """Connects to signals from the main window."""
        if self.main_window:
            self.main_window.current_folder_changed.connect(self.on_folder_updated)
            self.on_folder_updated(self.main_window.current_folder)

    def update_ui_state(self):
        """Centralized method to update UI elements based on preprocessor state."""
        has_raw = bool(self.preprocessor and self.preprocessor.has("raw"))
        has_continuous_ica = bool(
            self.preprocessor and self.preprocessor.has("continuous_ica")
        )
        has_epochs = bool(self.preprocessor and self.preprocessor.has("epochs"))
        has_epochs_ica = bool(self.preprocessor and self.preprocessor.has("epochs_ica"))
        has_preprocessed = bool(
            self.preprocessor and self.preprocessor.has("preprocessed")
        )

        if has_raw:
            self.raw_info_widget.update_info(self.preprocessor._get_last_continuous())
        if has_continuous_ica:
            self.continuous_ica_info_widget.update_info(
                self.preprocessor.continuous_ica
            )
        if has_epochs:
            self.epochs_info_widget.update_info(self.preprocessor.epochs)
        if has_epochs_ica:
            self.epochs_ica_info_widget.update_info(self.preprocessor.epochs_ica)
        if has_preprocessed:
            self.preprocessed_info_widget.update_info(self.preprocessor.preprocessed)

        # Update button states
        self.inspect_raw_button.setEnabled(has_raw)
        self.artifact_removal_button.setEnabled(has_raw)
        self.filter_continuous_button.setEnabled(has_raw)
        self.segment_button.setEnabled(has_raw)

        self.run_continuous_ica_button.setEnabled(has_raw)
        self.inspect_continuous_ica_button.setEnabled(has_continuous_ica)
        self.inspect_continuous_ica_sources_button.setEnabled(has_continuous_ica)

        self.plot_raw_evoked_button.setEnabled(has_epochs)
        self.inspect_epochs_button.setEnabled(has_epochs)
        self.rereference_epochs_button.setEnabled(
            has_epochs and not self.preprocessor.epochs.proj
        )

        self.run_epochs_ica_button.setEnabled(has_epochs)
        self.inspect_epochs_ica_button.setEnabled(has_epochs_ica)

        self.apply_filters_button.setEnabled(has_epochs)

        self.plot_widget.setVisible(has_preprocessed)
        if has_preprocessed:
            if self.preprocessor.preprocessed is not None:
                evoked = self.preprocessor.preprocessed.average()
                self.plot_widget.update_plot(evoked, label=self.preprocessor.label)

        self.update_file_list()

    # --- Connections ---
    @Slot(Path)
    def on_folder_updated(self, folder_path: Path):
        """Updates the UI when the main folder changes."""
        if not folder_path:
            return
        if not folder_path.is_dir():
            return
        self.select_folder_button.setText(f"Folder: {folder_path.name}")
        self.update_file_list()

    def select_folder(self):
        """Opens a dialog to select a new data folder."""
        if not self.main_window:
            return

        folder_path = QFileDialog.getExistingDirectory(
            self, "Select EEG Data Folder", str(self.main_window.current_folder)
        )
        if folder_path:
            self.main_window.set_current_folder(folder_path)

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
            return

        try:
            files = sorted(folder.rglob("*_raw.fif"))
            filter_text = self.file_filter.text().lower().split()

            item_to_reselect = None
            for f in files:
                if all(word in f.name.lower() for word in filter_text):
                    label = "_".join(f.name.split("_")[:-1])
                    finished_path = (
                        self.main_window.current_folder
                        / self.main_window.output_dir
                        / "preprocessed"
                        / f"{label}_preprocessed-epo.fif"
                    )
                    prefix = "✅ " if finished_path.exists() else "❌ "

                    item = QListWidgetItem(prefix + label)
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

    @Slot()
    def on_settings_updated(self):
        """This Slot is called by MainWindow when application settings have changed."""
        logger.info(f"TEP Preprocessing Page is updating its settings.")
        if self.main_window.current_folder is not None:
            output_dir = self.main_window.current_folder / self.main_window.output_dir
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
        else:
            output_dir = None
            self.preprocessor = None

        if self.preprocessor:
            self.preprocessor = Worker(
                lambda: Preprocessor(self.preprocessor.raw_filepath, output_dir),
                parent=self,
            ).exec_with_dialog("Loading data", "Loading raw data...")
        else:
            self.preprocessor = None

        self.update_ui_state()

    def on_file_selected(self, item: QListWidgetItem):
        """Handles file selection by loading data for the chosen file."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path:
            return
        self.preprocessor = None
        self.update_ui_state()

        output_dir = self.main_window.current_folder / self.main_window.output_dir

        # Load the new preprocessor object
        worker = Worker(
            lambda: Preprocessor(file_path, output_dir), parent=self, add_loggers="mne"
        )
        self.preprocessor = worker.exec_with_dialog(
            "Loading data", "Loading raw data..."
        )
        if self.preprocessor:
            logger.info(f"Successfully loaded {self.preprocessor.label}")
        self.update_ui_state()

    # --- Buttons and Functions ---
    def inspect_raw_data(self):
        if not self.preprocessor:
            return
        n_ch = len(self.preprocessor.raw.ch_names)
        self.preprocessor.raw.plot(
            n_channels=n_ch, duration=10, use_opengl=None, splash=False, block=True
        )
        self.update_ui_state()

    def artifact_removal(self):
        dialog = ArtifactRemovalDialog(self.preprocessor, self)
        dialog.exec()
        self.update_ui_state()

    def filter_continuous(self):
        if not self.preprocessor.has("raw"):
            return
        dialog = FilterContinuousDialog(self.preprocessor, self)
        dialog.exec()
        self.update_ui_state()

    def run_continuous_ica(self):
        if not self.preprocessor.has("raw"):
            return
        dialog = RunICADialog(self.preprocessor, mode="continuous", parent=self)
        dialog.exec()
        self.update_ui_state()

    def inspect_continuous_ica_sources(self):
        fig = self.preprocessor.continuous_ica.plot_sources(
            self.preprocessor._get_last_continuous().load_data(),
            block=False,
            splash=False,
        )
        fig.gotClosed.connect(lambda: self.update_ui_state())

    def inspect_continuous_ica(self):
        if not self.preprocessor.has("continuous_ica"):
            return
        raw = self.preprocessor._get_last_continuous().load_data()
        ica = run_ica_viewer(self.preprocessor.continuous_ica, raw, self)
        if ica:
            self.preprocessor.continuous_ica.exclude = ica.exclude
            worker = Worker(
                lambda: self.preprocessor.continuous_ica.save(
                    self.preprocessor.paths["continuous_ica"], overwrite=True
                ),
                parent=self,
                add_loggers="mne",
            )
            worker.exec_with_dialog("Please wait", "Saving ICA...")
        self.update_ui_state()

    def segment_epochs(self):
        dialog = EpochingDialog(self.preprocessor, self)
        dialog.exec()
        self.update_ui_state()

    def plot_raw_evoked(self):
        """Calculates and displays the raw evoked response."""
        if not self.preprocessor or not self.preprocessor.has("epochs"):
            QMessageBox.warning(self, "No Data", "Epochs data is not available.")
            return
        try:
            label = f"<b>{self.preprocessor.label}</b>"
            evoked = self.preprocessor.epochs.average().pick(
                [
                    ch
                    for ch in self.preprocessor.epochs.ch_names
                    if ch not in self.preprocessor.epochs.info["bads"]
                ]
            )
            self.raw_evoked_plot_widget.update_plot(evoked, label=label)
            self.raw_evoked_plot_widget.activateWindow()
        except Exception as e:
            QMessageBox.critical(
                self, "Plotting Error", f"An error occurred while plotting: {e}"
            )
            logger.error(f"Error plotting raw evoked: {e}")

    def inspect_epochs(self):
        """
        Shows a non-blocking interactive plot for inspecting epochs.
        The logic to handle changes is in the _after_epochs_inspected Slot.
        """
        if not self.preprocessor or not self.preprocessor.has("epochs"):
            QMessageBox.warning(
                self, "Warning", "No epochs data available for inspection."
            )
            return

        try:
            if not self.preprocessor.epochs.preload:
                Worker(
                    lambda: self.preprocessor.epochs.load_data(), parent=self
                ).exec_with_dialog("Please wait", "Loading epoched data...")
            self.viz_epochs = self.preprocessor.epochs.copy()
            self.original_bads = self.preprocessor.epochs.info["bads"].copy()
            self.original_selection = self.preprocessor.epochs.selection.copy()

            n_channels_str = self.settings.value(
                "erp_preprocessing/inspect/n_channels", "auto"
            )
            n_channels = (
                len(self.viz_epochs.ch_names)
                if n_channels_str == "auto"
                else int(n_channels_str)
            )
            fig = self.viz_epochs.plot(
                picks=self.preprocessor.epochs.ch_names,
                n_channels=n_channels,
                events=True,
                event_id=True,
                event_color=None,
                n_epochs=self.settings.value(
                    "erp_preprocessing/inspect/n_epochs", 10, type=int
                ),
                precompute=True,
                use_opengl=True,
                splash=False,
            )
            fig.mne.view.scene().sigMouseClicked.connect(fig._redraw) # Importat fix for MNE-QT-Browser
            fig.gotClosed.connect(self._after_epochs_inspected)
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open epochs plot: {e}")
            return

    @Slot()
    def _after_epochs_inspected(self):
        """
        This Slot is executed AFTER the user closes the MNE plot window.
        It compares the modified epochs with the original and prompts the user to save.
        """
        try:
            newly_marked_bads = [
                ch
                for ch in self.viz_epochs.info["bads"]
                if ch not in self.original_bads
            ]
            newly_unmarked_bads = [
                ch
                for ch in self.viz_epochs.ch_names
                if (ch in self.original_bads)
                and (ch not in self.viz_epochs.info["bads"])
            ]
            original_set = set(self.original_selection)
            new_set = set(self.viz_epochs.selection)
            num_dropped = len(original_set - new_set)

            if not newly_marked_bads and num_dropped == 0 and not newly_unmarked_bads:
                QMessageBox.information(
                    self, "No Changes", "No epochs or channels were modified."
                )
                return

            # 2. Build the confirmation message.
            message = "The following changes were made:\n\n"
            if newly_unmarked_bads:
                message += (
                    f" • Restored bad channels: {', '.join(newly_unmarked_bads)}\n"
                )
            if num_dropped > 0:
                message += f" • Epochs to drop: {num_dropped}\n"
            if newly_marked_bads:
                message += f" • New bad channels: {', '.join(newly_marked_bads)}\n"
            message += "\nWould you like to save these changes to the epochs file?"

            # 3. Ask for confirmation.
            reply = QMessageBox.question(
                self,
                "Apply Changes?",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                logger.info("Applying changes to epochs...")
                logger.info(
                    f"Epochs object updated. New count: {len(self.preprocessor.epochs)}"
                )
                logger.info(
                    f"Updated bad channels: {self.preprocessor.epochs.info['bads']}"
                )
                self.preprocessor.update_epochs(self.viz_epochs)
                QMessageBox.information(self, "Success", "Changes have been saved.")
            else:
                logger.info("Epoch changes were discarded by the user.")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while updating epochs: {e}"
            )
        finally:
            self.update_ui_state()

    def rereference_epochs(self):
        if not self.preprocessor.has("epochs"):
            return
        dialog = ReReferenceDialog(self.preprocessor, self)
        dialog.exec()
        self.update_ui_state()

    def run_epochs_ica(self):
        if not self.preprocessor.has("epochs"):
            return
        if not self.preprocessor.epochs.preload:
            Worker(
                lambda: self.preprocessor.epochs.load_data(), parent=self
            ).exec_with_dialog("Please wait", "Loading epoched data...")
        dialog = RunICADialog(self.preprocessor, mode="epochs", parent=self)
        dialog.exec()
        self.update_ui_state()

    def inspect_epochs_ica(self):
        if not self.preprocessor.has("epochs_ica"):
            return
        if not self.preprocessor.epochs.preload:
            Worker(
                lambda: self.preprocessor.epochs.load_data(), parent=self
            ).exec_with_dialog("Please wait", "Loading epoched data...")
        ica = run_ica_viewer(
            self.preprocessor.epochs_ica, self.preprocessor.epochs, self
        )
        if ica:
            self.preprocessor.epochs_ica.exclude = ica.exclude
            worker = Worker(
                lambda: self.preprocessor.epochs_ica.save(
                    self.preprocessor.paths["epochs_ica"], overwrite=True
                ),
                parent=self,
                add_loggers="mne",
            )
            worker.exec_with_dialog("Please wait", "Saving ICA...")
        self.update_ui_state()

    def apply_filters(self):
        dialog = ApplyPreprocessingDialog(self.preprocessor, self)
        dialog.finished.connect(self.update_ui_state)
        dialog.exec()
        self.update_ui_state()
