import logging
from datetime import datetime

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.processing import Preprocessor
from ui.widgets.evoked_plot import EvokedPlotWidget
from ui.widgets.ica_widget import run_ica_viewer
from ui.widgets.preprocessing_widgets import (
    ApplyPreprocessingDialog,
    ArtifactRemovalDialog,
    EpochingDialog,
    FilterContinuousDialog,
    ReReferenceDialog,
)
from ui.widgets.run_ica_widget import RunICADialog
from ui.widgets.tools.object_info_widget import ObjectInfoWidget
from utils import Worker

from .base_page import BasePage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ProcessingPage(BasePage):
    def __init__(self, parent=None):
        super().__init__("Preprocessing", parent)
        self.main_window = parent
        self.current_dataset = None
        self.preprocessor: Preprocessor | None = None
        self._needs_reload = False

        self.raw_info_widget = ObjectInfoWidget()
        self.continuous_ica_info_widget = ObjectInfoWidget()
        self.epochs_info_widget = ObjectInfoWidget()
        self.epochs_ica_info_widget = ObjectInfoWidget()
        self.preprocessed_info_widget = ObjectInfoWidget()

        self._setup_ui()
        self._setup_connections()
        self._configure_workflow_buttons()
        self._connect_to_main_window()
        self.update_ui_state()

    def _setup_ui(self):
        self.context_label = QLabel("Choose a dataset from the workspace panel.")
        self.context_label.setObjectName("contextCard")
        self.context_label.setWordWrap(True)
        self.add_content(self.context_label)

        self.plot_widget = EvokedPlotWidget()
        self.plot_widget.scrolled.connect(
            lambda direction: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 50
                if direction == "up"
                else self.scroll_area.verticalScrollBar().value() + 50
            )
        )
        self.plot_widget.setVisible(False)
        self.raw_evoked_plot_widget = EvokedPlotWidget()

        self.continuous_group = self._create_continuous_group()
        self.epochs_group = self._create_epochs_group()
        self.preprocessed_group = self._create_preprocessed_group()

        for title, widget in (
            ("Continuous Processing", self.continuous_group),
            ("Epoching and ICA", self.epochs_group),
            ("Preprocessed Output", self.preprocessed_group),
        ):
            title_widget = QLabel(f"<h2>{title}</h2>")
            self.add_content(title_widget)
            self.add_content(widget)

    def _create_continuous_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        first_row = QHBoxLayout()
        first_row.addWidget(self.raw_info_widget, 1)

        first_actions = QVBoxLayout()
        self.inspect_raw_button = QPushButton("Inspect Raw Data")
        self.artifact_removal_button = QPushButton("Remove Stimulation Artifact")
        self.filter_continuous_button = QPushButton("Filter and Resample")
        for button in (
            self.inspect_raw_button,
            self.artifact_removal_button,
            self.filter_continuous_button,
        ):
            first_actions.addWidget(button)
        first_row.addLayout(first_actions, 1)
        layout.addLayout(first_row)

        second_row = QHBoxLayout()
        second_row.addWidget(self.continuous_ica_info_widget, 1)

        second_actions = QVBoxLayout()
        self.run_continuous_ica_button = QPushButton("Run Continuous ICA")
        self.inspect_continuous_ica_button = QPushButton("Inspect Continuous ICA")
        for button in (
            self.run_continuous_ica_button,
            self.inspect_continuous_ica_button,
        ):
            second_actions.addWidget(button)
        second_row.addLayout(second_actions, 1)
        layout.addLayout(second_row)

        self.segment_button = QPushButton("Segment into Epochs")
        layout.addWidget(self.segment_button)
        return group

    def _create_epochs_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        first_row = QHBoxLayout()
        first_row.addWidget(self.epochs_info_widget, 1)

        first_actions = QVBoxLayout()
        self.plot_raw_evoked_button = QPushButton("Plot Raw Evoked")
        self.inspect_epochs_button = QPushButton("Inspect Epochs")
        self.rereference_epochs_button = QPushButton("Re-reference Epochs")
        for button in (
            self.plot_raw_evoked_button,
            self.inspect_epochs_button,
            self.rereference_epochs_button,
        ):
            first_actions.addWidget(button)
        first_row.addLayout(first_actions, 1)
        layout.addLayout(first_row)

        second_row = QHBoxLayout()
        second_row.addWidget(self.epochs_ica_info_widget, 1)

        second_actions = QVBoxLayout()
        self.run_epochs_ica_button = QPushButton("Run Epoch ICA")
        self.inspect_epochs_ica_button = QPushButton("Inspect Epoch ICA")
        second_actions.addWidget(self.run_epochs_ica_button)
        second_actions.addWidget(self.inspect_epochs_ica_button)
        second_row.addLayout(second_actions, 1)
        layout.addLayout(second_row)
        return group

    def _create_preprocessed_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(self.preprocessed_info_widget, 1)

        actions = QVBoxLayout()
        self.apply_filters_button = QPushButton("Apply ICA and Final Filters")
        actions.addWidget(self.apply_filters_button)
        row.addLayout(actions)

        layout.addLayout(row)
        layout.addWidget(self.plot_widget)
        return group

    def _setup_connections(self):
        self.inspect_raw_button.clicked.connect(self.inspect_raw_data)
        self.artifact_removal_button.clicked.connect(self.artifact_removal)
        self.filter_continuous_button.clicked.connect(self.filter_continuous)
        self.run_continuous_ica_button.clicked.connect(self.run_continuous_ica)
        self.inspect_continuous_ica_button.clicked.connect(self.inspect_continuous_ica)
        self.segment_button.clicked.connect(self.segment_epochs)

        self.plot_raw_evoked_button.clicked.connect(self.plot_raw_evoked)
        self.inspect_epochs_button.clicked.connect(self.inspect_epochs)
        self.rereference_epochs_button.clicked.connect(self.rereference_epochs)
        self.run_epochs_ica_button.clicked.connect(self.run_epochs_ica)
        self.inspect_epochs_ica_button.clicked.connect(self.inspect_epochs_ica)
        self.apply_filters_button.clicked.connect(self.apply_filters)

    def _configure_workflow_buttons(self):
        self._workflow_button_specs = {
            self.inspect_raw_button: {"stage_id": "raw", "role": "inspect"},
            self.artifact_removal_button: {"stage_id": "filtered_raw", "role": "process"},
            self.filter_continuous_button: {"stage_id": "filtered_raw", "role": "process"},
            self.run_continuous_ica_button: {"stage_id": "continuous_ica", "role": "process"},
            self.inspect_continuous_ica_button: {"stage_id": "continuous_ica", "role": "inspect"},
            self.segment_button: {"stage_id": "epochs", "role": "process"},
            self.plot_raw_evoked_button: {"stage_id": "epochs", "role": "inspect"},
            self.inspect_epochs_button: {"stage_id": "epochs", "role": "inspect"},
            self.rereference_epochs_button: {"stage_id": "epochs", "role": "process"},
            self.run_epochs_ica_button: {"stage_id": "epochs_ica", "role": "process"},
            self.inspect_epochs_ica_button: {"stage_id": "epochs_ica", "role": "inspect"},
            self.apply_filters_button: {"stage_id": "preprocessed", "role": "process"},
        }

        stage_lookup = {}
        if self.main_window is not None:
            stage_lookup = {
                stage.id: stage for stage in self.main_window.current_pipeline.stages
            }

        for button, spec in self._workflow_button_specs.items():
            stage = stage_lookup.get(spec["stage_id"])
            button.setProperty("workflowButton", True)
            button.setProperty("workflowRole", spec["role"])
            button.setProperty("stepState", "locked")
            button.setProperty("optionalStep", bool(stage and stage.optional))
            if stage is not None:
                tooltip = f"{stage.label}\n{stage.description}"
                if stage.optional:
                    tooltip += "\nOptional step."
                button.setToolTip(tooltip)

    def _refresh_workflow_button_states(self, availability: dict[QPushButton, bool]):
        for button, spec in self._workflow_button_specs.items():
            is_available = availability.get(button, False)
            if self.current_dataset is None:
                step_state = "locked"
            else:
                stage_state = self.current_dataset.stage_status(spec["stage_id"])
                if spec["role"] == "inspect":
                    step_state = "complete" if is_available else "locked"
                elif stage_state == "complete":
                    step_state = "complete"
                elif stage_state == "skipped":
                    step_state = "skipped"
                elif is_available:
                    step_state = "available"
                else:
                    step_state = "locked"

            button.setProperty("stepState", step_state)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _connect_to_main_window(self):
        if not self.main_window:
            return
        self.main_window.current_dataset_changed.connect(self.on_dataset_changed)
        self.main_window.current_pipeline_changed.connect(self.on_pipeline_changed)
        self.current_dataset = self.main_window.current_dataset
        self._needs_reload = True

    @Slot(object)
    def on_dataset_changed(self, dataset):
        self.current_dataset = dataset
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()

    @Slot(object)
    def on_pipeline_changed(self, pipeline):
        self._needs_reload = True
        self._configure_workflow_buttons()
        if self.isVisible():
            self.update_ui_state()

    def _ensure_preprocessor_loaded(self):
        if not self.current_dataset:
            self.preprocessor = None
            self._needs_reload = False
            return

        if not self._needs_reload and self.preprocessor is not None:
            return

        worker = Worker(
            lambda: Preprocessor(self.current_dataset.raw_path, self.current_dataset.derivative_root),
            parent=self,
            add_loggers="mne",
        )
        self.preprocessor = worker.exec_with_dialog("Loading data", "Loading pipeline dataset...")
        self._needs_reload = False

    def _refresh_after_step(self):
        self._needs_reload = True
        if self.main_window:
            self.main_window.refresh_workspace()
        if self.isVisible():
            self.update_ui_state()

    def _save_ica_changes(self, ica_object, path, log_key: str):
        if not self.preprocessor:
            return

        self.preprocessor.add_description(
            ica_object,
            {
                log_key: {
                    "excluded_components": list(ica_object.exclude),
                    "excluded_count": len(ica_object.exclude),
                    "date": datetime.now().isoformat(),
                }
            },
        )
        worker = Worker(
            lambda: ica_object.save(path, overwrite=True),
            parent=self,
            add_loggers="mne",
        )
        worker.exec_with_dialog("Please wait", "Saving ICA...")

    def update_ui_state(self):
        pipeline_name = self.main_window.current_pipeline.name if self.main_window else "Pipeline"
        if self.current_dataset:
            self.set_page_subtitle(f"{pipeline_name} pipeline · {self.current_dataset.display_name}")
            self.context_label.hide()
        else:
            self.set_page_subtitle("Select a dataset in the workspace panel to start preprocessing.")
            self.context_label.setText("Choose a dataset from the workspace panel.")
            self.context_label.setToolTip("")
            self.context_label.show()

        self._ensure_preprocessor_loaded()

        has_raw = bool(self.preprocessor and self.preprocessor.has("raw"))
        has_continuous_ica = bool(self.preprocessor and self.preprocessor.has("continuous_ica"))
        has_epochs = bool(self.preprocessor and self.preprocessor.has("epochs"))
        has_epochs_ica = bool(self.preprocessor and self.preprocessor.has("epochs_ica"))
        has_preprocessed = bool(self.preprocessor and self.preprocessor.has("preprocessed"))

        if has_raw:
            self.raw_info_widget.update_info(self.preprocessor._get_last_continuous(), "Recording summary")
        else:
            self.raw_info_widget.clear_info()

        if has_continuous_ica:
            self.continuous_ica_info_widget.update_info(self.preprocessor.continuous_ica, "ICA summary")
        else:
            self.continuous_ica_info_widget.clear_info()

        if has_epochs:
            self.epochs_info_widget.update_info(self.preprocessor.epochs, "Epoch summary")
        else:
            self.epochs_info_widget.clear_info()

        if has_epochs_ica:
            self.epochs_ica_info_widget.update_info(self.preprocessor.epochs_ica, "Epoch ICA summary")
        else:
            self.epochs_ica_info_widget.clear_info()

        if has_preprocessed:
            self.preprocessed_info_widget.update_info(self.preprocessor.preprocessed, "Ready output")
        else:
            self.preprocessed_info_widget.clear_info()

        button_availability = {
            self.inspect_raw_button: has_raw,
            self.artifact_removal_button: has_raw,
            self.filter_continuous_button: has_raw,
            self.segment_button: has_raw,
            self.run_continuous_ica_button: has_raw,
            self.inspect_continuous_ica_button: has_continuous_ica,
            self.plot_raw_evoked_button: has_epochs,
            self.inspect_epochs_button: has_epochs,
            self.rereference_epochs_button: has_epochs and not self.preprocessor.epochs.proj if has_epochs else False,
            self.run_epochs_ica_button: has_epochs,
            self.inspect_epochs_ica_button: has_epochs_ica,
            self.apply_filters_button: has_epochs,
        }
        for button, is_enabled in button_availability.items():
            button.setEnabled(is_enabled)
        self._refresh_workflow_button_states(button_availability)

        self.plot_widget.setVisible(has_preprocessed)
        if has_preprocessed and self.preprocessor.preprocessed is not None:
            self.plot_widget.update_plot(
                self.preprocessor.preprocessed,
                label=self.preprocessor.label,
            )
        else:
            self.plot_widget.update_plot(None)

    @Slot()
    def on_settings_updated(self):
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()

    def inspect_raw_data(self):
        if not self.preprocessor:
            return
        n_channels = len(self.preprocessor.raw.ch_names)
        self.preprocessor.raw.plot(
            n_channels=n_channels,
            duration=10,
            use_opengl=None,
            splash=False,
            block=True,
        )
        self.update_ui_state()

    def artifact_removal(self):
        if not self.preprocessor:
            return
        dialog = ArtifactRemovalDialog(self.preprocessor, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_after_step()

    def filter_continuous(self):
        if not self.preprocessor or not self.preprocessor.has("raw"):
            return
        dialog = FilterContinuousDialog(self.preprocessor, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_after_step()

    def run_continuous_ica(self):
        if not self.preprocessor or not self.preprocessor.has("raw"):
            return
        dialog = RunICADialog(self.preprocessor, mode="continuous", parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_after_step()

    def inspect_continuous_ica(self):
        if not self.preprocessor or not self.preprocessor.has("continuous_ica"):
            return
        raw = self.preprocessor._get_last_continuous().load_data()
        ica = run_ica_viewer(self.preprocessor.continuous_ica, raw, self)
        if ica:
            self.preprocessor.continuous_ica.exclude = ica.exclude
            self._save_ica_changes(
                self.preprocessor.continuous_ica,
                self.preprocessor.paths["continuous_ica"],
                "continuous_ica_review",
            )
            self._refresh_after_step()

    def segment_epochs(self):
        if not self.preprocessor:
            return
        dialog = EpochingDialog(self.preprocessor, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_after_step()

    def plot_raw_evoked(self):
        if not self.preprocessor or not self.preprocessor.has("epochs"):
            QMessageBox.warning(self, "No Data", "Epoch data is not available.")
            return

        try:
            good_channels = [
                channel
                for channel in self.preprocessor.epochs.ch_names
                if channel not in self.preprocessor.epochs.info["bads"]
            ]
            epochs = self.preprocessor.epochs.copy()
            if good_channels:
                epochs.pick(good_channels)
            self.raw_evoked_plot_widget.update_plot(
                epochs,
                label=f"<b>{self.preprocessor.label}</b>",
            )
            self.raw_evoked_plot_widget.activateWindow()
        except Exception as exc:
            QMessageBox.critical(self, "Plotting Error", f"An error occurred while plotting: {exc}")

    def inspect_epochs(self):
        if not self.preprocessor or not self.preprocessor.has("epochs"):
            QMessageBox.warning(self, "Warning", "No epochs data available for inspection.")
            return

        try:
            if not self.preprocessor.epochs.preload:
                Worker(lambda: self.preprocessor.epochs.load_data(), parent=self).exec_with_dialog(
                    "Please wait",
                    "Loading epoched data...",
                )

            self.viz_epochs = self.preprocessor.epochs.copy()
            self.original_bads = self.preprocessor.epochs.info["bads"].copy()
            self.original_selection = self.preprocessor.epochs.selection.copy()

            figure = self.viz_epochs.plot(
                picks=self.preprocessor.epochs.ch_names,
                n_channels=len(self.viz_epochs.ch_names),
                events=True,
                event_id=True,
                event_color=None,
                n_epochs=10,
                precompute=True,
                use_opengl=True,
                splash=False,
            )
            figure.mne.view.scene().sigMouseClicked.connect(figure._redraw)
            figure.gotClosed.connect(self._after_epochs_inspected)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open epochs plot: {exc}")

    @Slot()
    def _after_epochs_inspected(self):
        did_save = False
        try:
            newly_marked_bads = [
                channel
                for channel in self.viz_epochs.info["bads"]
                if channel not in self.original_bads
            ]
            newly_unmarked_bads = [
                channel
                for channel in self.viz_epochs.ch_names
                if channel in self.original_bads and channel not in self.viz_epochs.info["bads"]
            ]
            original_set = set(self.original_selection)
            new_set = set(self.viz_epochs.selection)
            num_dropped = len(original_set - new_set)

            if not newly_marked_bads and not newly_unmarked_bads and num_dropped == 0:
                QMessageBox.information(self, "No Changes", "No epochs or channels were modified.")
                return

            message = "The following changes were made:\n\n"
            if newly_unmarked_bads:
                message += f" - Restored bad channels: {', '.join(newly_unmarked_bads)}\n"
            if num_dropped:
                message += f" - Epochs to drop: {num_dropped}\n"
            if newly_marked_bads:
                message += f" - New bad channels: {', '.join(newly_marked_bads)}\n"
            message += "\nSave these changes to the epochs file?"

            reply = QMessageBox.question(
                self,
                "Apply Changes?",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.preprocessor.update_epochs(self.viz_epochs)
                did_save = True
                QMessageBox.information(self, "Success", "Changes have been saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"An error occurred while updating epochs: {exc}")
        finally:
            if did_save:
                self._refresh_after_step()
            elif self.isVisible():
                self.update_ui_state()

    def rereference_epochs(self):
        if not self.preprocessor or not self.preprocessor.has("epochs"):
            return
        dialog = ReReferenceDialog(self.preprocessor, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_after_step()

    def run_epochs_ica(self):
        if not self.preprocessor or not self.preprocessor.has("epochs"):
            return
        if not self.preprocessor.epochs.preload:
            Worker(lambda: self.preprocessor.epochs.load_data(), parent=self).exec_with_dialog(
                "Please wait",
                "Loading epoched data...",
            )
        dialog = RunICADialog(self.preprocessor, mode="epochs", parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_after_step()

    def inspect_epochs_ica(self):
        if not self.preprocessor or not self.preprocessor.has("epochs_ica"):
            return
        if not self.preprocessor.epochs.preload:
            Worker(lambda: self.preprocessor.epochs.load_data(), parent=self).exec_with_dialog(
                "Please wait",
                "Loading epoched data...",
            )
        ica = run_ica_viewer(self.preprocessor.epochs_ica, self.preprocessor.epochs, self)
        if ica:
            self.preprocessor.epochs_ica.exclude = ica.exclude
            self._save_ica_changes(
                self.preprocessor.epochs_ica,
                self.preprocessor.paths["epochs_ica"],
                "epochs_ica_review",
            )
            self._refresh_after_step()

    def apply_filters(self):
        if not self.preprocessor:
            return
        dialog = ApplyPreprocessingDialog(self.preprocessor, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_after_step()
