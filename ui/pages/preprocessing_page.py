import logging
from collections import Counter
from datetime import datetime

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.processing import Preprocessor
from core.pipelines import get_pipeline
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
from ui.widgets.tools.stage_io import load_stage_object
from utils import Worker

from .base_page import BasePage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _annotation_entries(annotations) -> list[tuple[float, float, str, tuple[str, ...]]]:
    ch_names = getattr(annotations, "ch_names", None)
    if ch_names is None:
        ch_names = [()] * len(annotations)

    entries = []
    for onset, duration, description, annotation_channels in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        ch_names,
    ):
        if isinstance(annotation_channels, str):
            channels = (annotation_channels,)
        else:
            channels = tuple(str(channel) for channel in annotation_channels)
        entries.append((float(onset), float(duration), str(description), channels))
    return entries


def _annotation_counter(annotations) -> Counter:
    return Counter(_annotation_entries(annotations))


def _is_bad_annotation(entry: tuple[float, float, str, tuple[str, ...]]) -> bool:
    return entry[2].lower().startswith("bad")


class ProcessingPage(BasePage):
    def __init__(self, parent=None):
        super().__init__("Preprocessing", parent)
        self.main_window = parent
        self.current_dataset = None
        self.preprocessor: Preprocessor | None = None
        self._needs_reload = False

        self.action_buttons: dict[str, QPushButton] = {}
        self._workflow_button_specs = {}
        self.info_widgets_by_stage: dict[str, ObjectInfoWidget] = {}
        self.plot_widgets_by_stage: dict[str, EvokedPlotWidget] = {}
        self._raw_review_pending = False
        self._raw_review_figure = None

        self._setup_ui()
        self._connect_to_main_window()
        self.update_ui_state()

    def _setup_ui(self):
        self.raw_evoked_plot_widget = EvokedPlotWidget()
        self.workflow_container = QWidget()
        self.workflow_layout = QVBoxLayout(self.workflow_container)
        self.workflow_layout.setContentsMargins(0, 0, 0, 0)
        self.workflow_layout.setSpacing(16)
        self.add_content(self.workflow_container)
        self._build_pipeline_workflow()

    def _current_pipeline(self):
        if self.main_window is not None:
            return self.main_window.current_pipeline
        return get_pipeline(None)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            widget = item.widget()
            if child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()
            if widget is not None:
                widget.deleteLater()

    def _build_pipeline_workflow(self):
        self._clear_layout(self.workflow_layout)
        self.action_buttons = {}
        self._workflow_button_specs = {}
        self.info_widgets_by_stage = {}
        self.plot_widgets_by_stage = {}

        pipeline = self._current_pipeline()
        if pipeline is None:
            return

        for section in pipeline.workflow_sections:
            title_widget = QLabel(f"<h2>{section.title}</h2>")
            self.workflow_layout.addWidget(title_widget)

            group = QGroupBox()
            group_layout = QVBoxLayout(group)

            for panel in section.panels:
                if panel.info_stage_id is None:
                    for action in panel.actions:
                        group_layout.addWidget(self._create_workflow_button(action))
                else:
                    row = QHBoxLayout()
                    info_widget = ObjectInfoWidget()
                    self.info_widgets_by_stage[panel.info_stage_id] = info_widget
                    setattr(self, f"{panel.info_stage_id}_info_widget", info_widget)
                    row.addWidget(info_widget, 1)

                    actions_layout = QVBoxLayout()
                    for action in panel.actions:
                        actions_layout.addWidget(self._create_workflow_button(action))
                    row.addLayout(actions_layout, 1)
                    group_layout.addLayout(row)

                if panel.plot_stage_id is not None:
                    plot_widget = EvokedPlotWidget()
                    self._connect_plot_scrolling(plot_widget)
                    plot_widget.setVisible(False)
                    self.plot_widgets_by_stage[panel.plot_stage_id] = plot_widget
                    if panel.plot_stage_id == "preprocessed":
                        self.plot_widget = plot_widget
                    group_layout.addWidget(plot_widget)

            self.workflow_layout.addWidget(group)

        self.workflow_layout.addStretch()
        self._configure_workflow_buttons()

    def _create_workflow_button(self, action) -> QPushButton:
        button = QPushButton(action.label)
        handler = getattr(self, action.handler, None)
        if handler is not None:
            button.clicked.connect(handler)
        else:
            button.setEnabled(False)
            button.setToolTip(f"No handler is available for `{action.handler}`.")
        self.action_buttons[action.id] = button
        self._workflow_button_specs[button] = action
        setattr(self, f"{action.id}_button", button)
        return button

    def _connect_plot_scrolling(self, plot_widget: EvokedPlotWidget):
        plot_widget.scrolled.connect(
            lambda direction: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 50
                if direction == "up"
                else self.scroll_area.verticalScrollBar().value() + 50
            )
        )

    def _configure_workflow_buttons(self):
        stage_lookup = {}
        if self.main_window is not None:
            stage_lookup = {
                stage.id: stage for stage in self.main_window.current_pipeline.stages
            }

        for button, action in self._workflow_button_specs.items():
            stage = stage_lookup.get(action.stage_id)
            button.setProperty("workflowButton", True)
            button.setProperty("workflowRole", action.role)
            button.setProperty("stepState", "locked")
            button.setProperty("optionalStep", bool(stage and stage.optional))
            if stage is not None:
                tooltip = f"{action.label}\n{stage.label}: {stage.description}"
                if stage.optional:
                    tooltip += "\nOptional step."
                button.setToolTip(tooltip)

    def _refresh_workflow_button_states(self, availability: dict[QPushButton, bool]):
        for button, action in self._workflow_button_specs.items():
            is_available = availability.get(button, False)
            if self.current_dataset is None:
                step_state = "locked"
            else:
                stage_state = self._stage_status(action.stage_id)
                if action.role == "inspect":
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
        self._build_pipeline_workflow()
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

    def _has_stage(self, stage_id: str | None) -> bool:
        if not self.preprocessor or not stage_id:
            return False
        if self.preprocessor.has(stage_id):
            return True
        return bool(self.current_dataset and self.current_dataset.stage_exists(stage_id))

    def _is_action_available(self, action) -> bool:
        if not self.preprocessor:
            return False

        is_available = self._has_stage(action.requires_stage_id)
        if action.id == "rereference_epochs" and is_available:
            return not self.preprocessor.epochs.proj
        return is_available

    def _stage_definition(self, stage_id: str | None):
        if stage_id is None:
            return None
        pipeline = self._current_pipeline()
        return next((stage for stage in pipeline.stages if stage.id == stage_id), None)

    def _stage_status(self, stage_id: str) -> str:
        if self.current_dataset is None:
            return "pending"
        try:
            return self.current_dataset.stage_status(stage_id)
        except (StopIteration, ValueError):
            return "pending"

    def _stage_object_for_info(self, stage_id: str):
        if not self.preprocessor:
            return None, ""

        stage = self._stage_definition(stage_id)
        label = f"{stage.label} summary" if stage is not None else "Stage summary"

        try:
            if stage_id == "raw" and self.preprocessor.has("raw"):
                return self.preprocessor._get_last_continuous(), "Current continuous file"
            if stage_id == "filtered_raw" and self.preprocessor.has("filtered_raw"):
                return self.preprocessor.filtered_raw, label
            if stage_id == "continuous_ica" and self.preprocessor.has("continuous_ica"):
                return self.preprocessor.continuous_ica, label
            if stage_id == "epochs" and self.preprocessor.has("epochs"):
                return self.preprocessor.epochs, label
            if stage_id == "epochs_ica" and self.preprocessor.has("epochs_ica"):
                return self.preprocessor.epochs_ica, label
            if stage_id == "preprocessed" and self.preprocessor.has("preprocessed"):
                return self.preprocessor.preprocessed, label

            if self.current_dataset is None or stage is None:
                return None, ""
            if not self.current_dataset.stage_exists(stage_id):
                return None, ""
            return load_stage_object(stage_id, self.current_dataset.paths[stage_id], stage.data_kind), label
        except (FileNotFoundError, ValueError):
            return None, ""

        return None, ""

    def _refresh_info_widgets(self):
        for stage_id, info_widget in self.info_widgets_by_stage.items():
            stage_object, label = self._stage_object_for_info(stage_id)
            if stage_object is None:
                info_widget.clear_info()
            else:
                info_widget.update_info(stage_object, label)

    def _refresh_plot_widgets(self):
        for stage_id, plot_widget in self.plot_widgets_by_stage.items():
            stage_object, _ = self._stage_object_for_info(stage_id)
            stage = self._stage_definition(stage_id)
            can_plot = stage is not None and stage.data_kind == "epochs"
            plot_widget.setVisible(stage_object is not None and can_plot)
            if stage_object is None or not can_plot:
                plot_widget.update_plot(None)
            else:
                plot_widget.update_plot(
                    stage_object,
                    label=self.preprocessor.label if self.preprocessor else None,
                )

    def update_ui_state(self):
        pipeline_name = self.main_window.current_pipeline.name if self.main_window else "Pipeline"
        if self.current_dataset:
            self.set_page_subtitle(f"{pipeline_name} pipeline · {self.current_dataset.display_name}")
        else:
            self.set_page_subtitle("Select a dataset in the workspace panel to start preprocessing.")

        self._ensure_preprocessor_loaded()

        self._refresh_info_widgets()

        button_availability = {
            button: self._is_action_available(action)
            for button, action in self._workflow_button_specs.items()
        }
        for button, is_enabled in button_availability.items():
            button.setEnabled(is_enabled)
        self._refresh_workflow_button_states(button_availability)

        self._refresh_plot_widgets()

    @Slot()
    def on_settings_updated(self):
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()

    def inspect_raw_data(self):
        if not self.preprocessor:
            return
        if self._raw_review_pending:
            if self._raw_review_figure is not None and hasattr(self._raw_review_figure, "activateWindow"):
                self._raw_review_figure.activateWindow()
            return
        try:
            self.viz_raw = self.preprocessor._get_last_continuous().copy()
            self.original_raw_bads = self.viz_raw.info["bads"].copy()
            self.original_raw_annotations = self.viz_raw.annotations.copy()
            self._raw_review_pending = True

            figure = self.viz_raw.plot(
                n_channels=len(self.viz_raw.ch_names),
                duration=10,
                use_opengl=None,
                splash=False,
                block=False,
            )
            self._raw_review_figure = figure
            if hasattr(figure, "gotClosed"):
                figure.gotClosed.connect(self._after_raw_inspected)
            elif hasattr(figure, "canvas") and hasattr(figure.canvas, "mpl_connect"):
                figure.canvas.mpl_connect("close_event", lambda event: self._after_raw_inspected())
            else:
                self._raw_review_pending = False
                self._raw_review_figure = None
                QMessageBox.warning(
                    self,
                    "Unable to Track Inspection",
                    "The raw inspection window does not expose a close event, so changes cannot be saved from this viewer.",
                )
        except Exception as exc:
            self._raw_review_pending = False
            self._raw_review_figure = None
            QMessageBox.critical(self, "Error", f"Failed to inspect raw data: {exc}")

    @Slot()
    def _after_raw_inspected(self):
        if not self._raw_review_pending:
            return
        self._raw_review_pending = False
        self._raw_review_figure = None
        did_save = False
        try:
            newly_marked_bads = [
                channel
                for channel in self.viz_raw.info["bads"]
                if channel not in self.original_raw_bads
            ]
            newly_unmarked_bads = [
                channel
                for channel in self.viz_raw.ch_names
                if channel in self.original_raw_bads and channel not in self.viz_raw.info["bads"]
            ]

            original_annotations = _annotation_counter(self.original_raw_annotations)
            reviewed_annotations = _annotation_counter(self.viz_raw.annotations)
            added_annotations = reviewed_annotations - original_annotations
            removed_annotations = original_annotations - reviewed_annotations
            added_annotation_count = sum(added_annotations.values())
            removed_annotation_count = sum(removed_annotations.values())
            added_bad_annotation_count = sum(
                count for entry, count in added_annotations.items() if _is_bad_annotation(entry)
            )
            removed_bad_annotation_count = sum(
                count for entry, count in removed_annotations.items() if _is_bad_annotation(entry)
            )

            if (
                not newly_marked_bads
                and not newly_unmarked_bads
                and added_annotation_count == 0
                and removed_annotation_count == 0
            ):
                QMessageBox.information(self, "No Changes", "No channels or annotations were modified.")
                return

            message = "The following changes were made:\n\n"
            if newly_unmarked_bads:
                message += f" - Restored bad channels: {', '.join(newly_unmarked_bads)}\n"
            if newly_marked_bads:
                message += f" - New bad channels: {', '.join(newly_marked_bads)}\n"
            if added_annotation_count:
                message += f" - Annotations added: {added_annotation_count}\n"
            if removed_annotation_count:
                message += f" - Annotations removed: {removed_annotation_count}\n"
            if added_bad_annotation_count or removed_bad_annotation_count:
                message += (
                    " - Bad annotation changes: "
                    f"+{added_bad_annotation_count}/-{removed_bad_annotation_count}\n"
                )
            message += "\nSave these changes to the filtered raw derivative?"
            message += "\nThe source raw file will not be modified."

            reply = QMessageBox.question(
                self,
                "Apply Changes?",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.preprocessor.update_raw_review(self.viz_raw)
                did_save = True
                QMessageBox.information(self, "Success", "Changes have been saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"An error occurred while updating raw data: {exc}")
        finally:
            if did_save:
                self._refresh_after_step()
            elif self.isVisible():
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
            self.raw_evoked_plot_widget.update_plot(
                self.preprocessor.epochs,
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
