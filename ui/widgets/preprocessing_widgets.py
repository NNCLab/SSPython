import logging
import sys
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QCheckBox,
    QAbstractItemView,
    QDialogButtonBox,
    QDialog,
    QMessageBox,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QRadioButton,
    QLabel,
    QScrollArea,
)
from .evoked_plot import EvokedPlotDialog
import mne
from core.processing import Preprocessor
from utils import parse_numeric, to_display_string, Worker
from .tools.optional_range_widget import OptionalRangeWidget
from pathlib import Path

from core.app_settings import get_settings_store

logger = logging.getLogger(__name__)


def _bounded_dialog_size(
    hint: QSize,
    *,
    min_size: QSize,
    max_size: QSize,
    width_padding: int = 96,
    height_padding: int = 120,
) -> QSize:
    width = min(max(hint.width() + width_padding, min_size.width()), max_size.width())
    height = min(
        max(hint.height() + height_padding, min_size.height()),
        max_size.height(),
    )
    return QSize(width, height)


def _selected_channels_button_text(
    selected_channels: list[str], empty_text: str
) -> str:
    display_text = ", ".join(selected_channels)
    if len(display_text) > 30:
        return f"{len(selected_channels)} channels selected"
    return display_text or empty_text


# Channel Selection
class ChannelSelectionDialog(QDialog):
    """A dialog to select channels from a list."""

    def __init__(
        self,
        all_channels: list[str],
        selected_channels: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Custom Reference Channels")
        self.setMinimumSize(380, 340)
        self.resize(420, 460)
        self.setObjectName("appDialog")
        layout = QVBoxLayout(self)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter channels")
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.addItems(all_channels)
        layout.addWidget(self.list_widget)
        self.search_input.textChanged.connect(self.filter_channels)

        # Pre-select channels if they were already chosen
        if selected_channels:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item.text() in selected_channels:
                    item.setSelected(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_channels(self) -> list[str]:
        """Returns the text of the selected items."""
        return [item.text() for item in self.list_widget.selectedItems()]

    def filter_channels(self, text: str):
        needle = text.strip().lower()
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            item.setHidden(bool(needle) and needle not in item.text().lower())


class BaseProcessingDialog(QDialog):
    """A base dialog class to reduce boilerplate for processing dialogues."""

    finished = Signal()

    def __init__(
        self,
        preprocessor: Preprocessor,
        title: str,
        settings_widget: QWidget,
        parent=None,
    ):
        super().__init__(parent)
        self.preprocessor = preprocessor
        self.setWindowTitle(title)
        self.setModal(True)
        self.setObjectName("appDialog")
        self.settings_widget = settings_widget

        layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.viewport().setAutoFillBackground(False)

        scroll_contents = QWidget()
        scroll_layout = QVBoxLayout(scroll_contents)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)
        scroll_layout.addWidget(self.settings_widget)
        scroll_layout.addStretch()
        self.scroll_area.setWidget(scroll_contents)

        layout.addWidget(self.scroll_area, 1)

        hbox = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        self.reset_button = QPushButton("Reset")

        hbox.addStretch()
        hbox.addWidget(self.reset_button)
        hbox.addWidget(self.cancel_button)
        hbox.addWidget(self.apply_button)
        layout.addLayout(hbox)

        self.apply_button.clicked.connect(self.run_process)
        self.cancel_button.clicked.connect(self.reject)
        if hasattr(self.settings_widget, "load_settings"):
            self.reset_button.clicked.connect(self.settings_widget.load_settings)
        else:
            self.reset_button.hide()

        self._apply_compact_size()

    def _apply_compact_size(
        self,
        *,
        min_size: QSize = QSize(520, 320),
        max_size: QSize = QSize(760, 720),
    ):
        self.layout().activate()
        self.adjustSize()
        dialog_size = _bounded_dialog_size(
            self.sizeHint(),
            min_size=min_size,
            max_size=max_size,
            width_padding=40,
            height_padding=32,
        )
        self.setMinimumSize(min_size)
        self.resize(dialog_size)

    def run_process(self):
        """To be implemented by subclasses to perform the actual processing logic."""
        raise NotImplementedError("Subclasses must implement run_process()")


class BaseSettingsWidget(QWidget):
    """A base widget for managing QSettings load/save logic."""

    SETTINGS_PATH = ""
    LEGACY_KEYS: tuple[str, ...] = ()

    def __init__(self, parent=None, pipeline_id: str | None = None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self.pipeline_id = pipeline_id or self.settings_store.current_pipeline_id()

    def get_settings_path(self) -> str:
        return self.SETTINGS_PATH.format(pipeline_id=self.pipeline_id)

    def set_pipeline(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.load_settings()

    def save_settings(self):
        params = self.get_params()
        self.settings_store.set(self.get_settings_path(), params)
        self.settings_store.sync()

    def load_settings(self):
        saved_params = self.settings_store.get(
            self.get_settings_path(),
            {},
            legacy_keys=self.LEGACY_KEYS,
        ) or {}

        params = self.get_defaults().copy()
        if isinstance(saved_params, dict):
            params.update(saved_params)

        self.set_params(params)

    def get_params(self) -> dict:
        """Returns a dictionary of the current settings from the UI."""
        raise NotImplementedError("Subclasses must implement get_params()")

    def set_params(self, params: dict):
        """Applies a dictionary of parameters to the UI."""
        raise NotImplementedError("Subclasses must implement set_params()")

    def get_defaults(self) -> dict:
        """Returns a dictionary of default parameters."""
        return {}


# Artifact Removal
class ArtifactRemovalSettingsWidget(BaseSettingsWidget):
    """
    A widget for configuring artifact removal settings.
    """

    SETTINGS_PATH = "pipelines/{pipeline_id}/preprocessing/artifact_removal"
    LEGACY_KEYS = ("erp_preprocessing/artifact_removal",)

    def __init__(self, show_events_list=False, parent=None):
        super().__init__(parent)
        self.show_events_list = show_events_list
        self.event_id_map = {}

        layout = QVBoxLayout(self)
        self.init_form(layout)
        self.load_settings()

    def init_form(self, main_layout):
        artifact_group = QGroupBox("Artifact Removal")
        artifact_layout = QFormLayout(artifact_group)

        self.window_edit = OptionalRangeWidget(
            ("Start: ", "Stop"),
            "ms",
            (True, True),
            range=(None, None),
            scale=1e-3,
            parent=self,
        )
        self.smoothing_edit = OptionalRangeWidget(
            ("Start: ", "Stop"),
            "ms",
            (True, True),
            range=(None, None),
            scale=1e-3,
            parent=self,
        )
        self.span_spinbox = QSpinBox()
        self.span_spinbox.setRange(1, 100)

        artifact_layout.addRow("Window:", self.window_edit)
        artifact_layout.addRow("Smoothing:", self.smoothing_edit)
        artifact_layout.addRow("Span (samples):", self.span_spinbox)

        if self.show_events_list:
            self.events_list = QListWidget()
            self.events_list.setSelectionMode(QListWidget.NoSelection)
            artifact_layout.addRow("Events for Artifact Removal:", self.events_list)
        else:
            self.events_list = None

        main_layout.addWidget(artifact_group)

    def populate_events_list(self, raw_data):
        """Discover events from the raw data and populate the checklist."""
        if not self.show_events_list or self.events_list is None:
            return

        try:
            if not raw_data.annotations:
                raise ValueError("No annotations found in raw data.")
            _, self.event_id_map = mne.events_from_annotations(raw_data)
        except Exception as e:
            logger.warning(f"Could not get events from annotations: {e}")
            return

        self.events_list.clear()
        for description in sorted(self.event_id_map.keys()):
            item = QListWidgetItem(description)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.events_list.addItem(item)

    def get_event_selection(self):
        """Get the currently selected event IDs from the list."""
        if not self.show_events_list or self.events_list is None:
            return []

        selected_events = []
        for i in range(self.events_list.count()):
            item = self.events_list.item(i)
            if item.checkState() == Qt.Checked:
                event_description = item.text()
                selected_events.append(self.event_id_map[event_description])
        return selected_events

    def get_defaults(self) -> dict:
        return {
            "window": (-0.002, 0.006),
            "smoothing": (-0.002, 0.002),
            "span": 5,
            "notch": None,
            "bandpass": (0.5, None),
        }

    def set_params(self, params: dict):
        self.window_edit.setValue(params.get("window"))
        self.smoothing_edit.setValue(params.get("smoothing"))
        self.span_spinbox.setValue(int(params.get("span")))

    def get_params(self) -> dict:
        """Returns a dictionary of the current settings from the UI."""
        params = {
            "span": int(self.span_spinbox.value()),
            "event_id": self.get_event_selection(),
            "window": self.window_edit.value(),
            "smoothing": self.smoothing_edit.value(),
        }
        logger.info(f"params:{params}")
        return params


class ArtifactRemovalDialog(BaseProcessingDialog):
    """A dialog for running the artifact removal process."""

    def __init__(self, preprocessor: Preprocessor, parent=None):
        settings_widget = ArtifactRemovalSettingsWidget(show_events_list=True)
        super().__init__(preprocessor, "Artifact Removal", settings_widget, parent)
        if self.preprocessor.has("raw"):
            self.settings_widget.populate_events_list(self.preprocessor.raw)

    def run_process(self):
        try:
            params = self.settings_widget.get_params()
            self.settings_widget.save_settings()
            logger.info(f"Running artifact removal with parameters: {params}")
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", str(e))
            return

        if self.preprocessor.has("continuous_ica") or self.preprocessor.has("epochs"):
            reply = QMessageBox.question(
                self,
                "Warning",
                "Downstream data already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        worker = Worker(
            lambda: self.preprocessor.run_artifact_removal(**params),
            parent=self.parent(),
            add_loggers="tqdm",
        )
        worker.exec_with_dialog("Please wait", "Running Artifact Removal")
        self.accept()


# Continuous Filter
class FilterContinuousWidget(BaseSettingsWidget):
    SETTINGS_PATH = "pipelines/{pipeline_id}/preprocessing/continuous_filter"
    LEGACY_KEYS = ("erp_preprocessing/continuous_filter",)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.init_form(layout)
        self.load_settings()

    def init_form(self, main_layout):
        filter_group = QGroupBox("Continuous Filter")
        filter_layout = QFormLayout(filter_group)

        self.resample_edit = QLineEdit()
        self.notch_edit = QLineEdit()
        self.filter_edit = OptionalRangeWidget(suffix="Hz", parent=self)

        filter_layout.addRow("Notch (Hz)", self.notch_edit)
        filter_layout.addRow("Bandpass", self.filter_edit)
        filter_layout.addRow("Resample (Hz)", self.resample_edit)

        main_layout.addWidget(filter_group)

    def get_defaults(self) -> dict:
        return {"notch": None, "bandpass": (0.5, None), "resample": None}

    def set_params(self, params: dict):
        self.notch_edit.setText(to_display_string(params.get("notch")))
        self.filter_edit.setValue(params.get("bandpass"))
        self.resample_edit.setText(to_display_string(params.get("resample")))

    def get_params(self) -> dict:
        """Returns a dictionary of the current settings from the UI."""
        params = {
            "notch": parse_numeric(self.notch_edit.text()),
            "bandpass": self.filter_edit.value(),
            "resample": parse_numeric(self.resample_edit.text()),
        }
        logger.info(f"params:{params}")
        return params


class FilterContinuousDialog(BaseProcessingDialog):
    def __init__(self, preprocessor: Preprocessor, parent=None):
        settings_widget = FilterContinuousWidget(parent=parent)
        super().__init__(preprocessor, "Filter & Resampling", settings_widget, parent)

    def run_process(self):
        try:
            params = self.settings_widget.get_params()
            self.settings_widget.save_settings()
            logger.info(f"Running epoching with parameters: {params}")
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", str(e))
            return

        if self.preprocessor.has("epochs"):
            reply = QMessageBox.question(
                self,
                "Warning",
                "Epochs data already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return
        worker = Worker(
            lambda: self.preprocessor.filter_continuous(**params), parent=self.parent()
        )
        worker.exec_with_dialog("Please wait", "Running Epoching")
        self.accept()


# Epoching
class EpochingSettingsWidget(BaseSettingsWidget):
    """A widget for configuring epoch segmentation settings."""

    SETTINGS_PATH = "pipelines/{pipeline_id}/preprocessing/epoching"
    LEGACY_KEYS = ("erp_preprocessing/epoching",)

    def __init__(self, show_events_list=False, parent=None):
        super().__init__(parent)
        self.detrend_map = {"Constant (DC)": 0, "Linear": 1, "No detrend": None}
        self.mode_map = {
            "Event-based": "event",
            "Fixed-length (Resting State)": "fixed",
        }
        self.show_events_list = show_events_list
        self.event_id_map: dict[str, int] = {}
        self._pending_event_selection: dict[str, int] | list[int] | int | None = None

        layout = QVBoxLayout(self)
        self.init_form(layout)
        self.load_settings()

        # Trigger the initial UI state update based on loaded settings
        self.update_ui_visibility(self.mode_combobox.currentText())

    def init_form(self, main_layout):
        epoching_group = QGroupBox("Epoching Configuration")
        epoching_layout = QFormLayout(epoching_group)

        # --- 1. Mode Selection ---
        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(["Event-based", "Fixed-length (Resting State)"])
        self.mode_combobox.currentTextChanged.connect(self.update_ui_visibility)
        epoching_layout.addRow("Epoching Mode:", self.mode_combobox)

        # --- 2. Common Settings ---
        self.resample_edit = QLineEdit()
        self.pick_eeg_only_checkbox = QCheckBox()
        self.notch_input = QLineEdit()
        self.bandpass_input = OptionalRangeWidget(
            suffix="Hz", required=(False, False), range=(0, None), parent=self
        )
        self.detrend_combobox = QComboBox()
        self.detrend_combobox.addItems(list(self.detrend_map.keys()))
        self.average_reference_checkbox = QCheckBox()

        epoching_layout.addRow("Notch (Hz):", self.notch_input)
        epoching_layout.addRow("Bandpass (low, high) Hz:", self.bandpass_input)
        epoching_layout.addRow("Detrend:", self.detrend_combobox)
        epoching_layout.addRow("Pick only EEG channels:", self.pick_eeg_only_checkbox)
        epoching_layout.addRow("Resample (Hz):", self.resample_edit)
        epoching_layout.addRow(
            "Apply average reference:", self.average_reference_checkbox
        )

        # --- 3. Event-Based Specifics ---
        self.tlim_label = QLabel("Time Limits (tmin, tmax) ms:")
        self.tlim_edit = OptionalRangeWidget(
            ("Start:", "Stop:"),
            "ms",
            required=(True, True),
            range=(None, None),
            parent=self,
            scale=1e-3,
        )

        self.events_label = QLabel("Events for Segmentation:")
        self.events_list = QListWidget(self)
        self.events_list.setSelectionMode(QListWidget.NoSelection)
        self.events_list.setFixedHeight(150)  # Limit height

        # Add Event-Based widgets to layout
        epoching_layout.addRow(self.events_label, self.events_list)
        epoching_layout.addRow(self.tlim_label, self.tlim_edit)
        # Now control visibility based on the flag
        if not self.show_events_list:
            self.events_label.hide()
            self.events_list.hide()

        # --- 4. Fixed-Length Specifics ---
        self.fixed_dur_label = QLabel("Duration (s):")
        self.fixed_dur_edit = QLineEdit()
        self.fixed_dur_edit.setPlaceholderText("e.g. 2.0")

        self.fixed_overlap_label = QLabel("Overlap (s):")
        self.fixed_overlap_edit = QLineEdit()
        self.fixed_overlap_edit.setPlaceholderText("e.g. 0.0 for no overlap")

        # Add Fixed-Length widgets to layout
        epoching_layout.addRow(self.fixed_dur_label, self.fixed_dur_edit)
        epoching_layout.addRow(self.fixed_overlap_label, self.fixed_overlap_edit)

        main_layout.addWidget(epoching_group)
        main_layout.addStretch()

    def update_ui_visibility(self, mode_text):
        """Shows/Hides widgets based on the selected mode."""
        is_event_based = mode_text == "Event-based"

        # Use the helper to hide the entire row including the label
        self.tlim_label.setVisible(is_event_based)
        self.tlim_edit.setVisible(is_event_based)
        if self.events_list:
            # If the mode is not event-based OR show_events_list is false, hide it
            visible = is_event_based and self.show_events_list
            self.events_label.setVisible(visible)
            self.events_list.setVisible(visible)

        # Toggle Fixed Length Widgets
        self.fixed_dur_label.setVisible(not is_event_based)
        self.fixed_dur_edit.setVisible(not is_event_based)
        self.fixed_overlap_label.setVisible(not is_event_based)
        self.fixed_overlap_edit.setVisible(not is_event_based)

    def populate_events_list(self, raw_data):
        if not self.show_events_list or self.events_list is None:
            return

        try:
            # Safety check: if raw has no annotations, user might BE forced to use Fixed-length
            if not raw_data.annotations:
                self.mode_combobox.setCurrentText("Fixed-length (Resting State)")
                # Optional: Disable the combobox if you want to force it
                self.mode_combobox.setEnabled(False)

            events, self.event_id_map = mne.events_from_annotations(raw_data)
        except Exception as e:
            # If no events found, we don't crash, just log
            logger.warning(f"Could not extract events: {e}")
            return

        self.events_list.clear()
        for description in sorted(self.event_id_map.keys()):
            item = QListWidgetItem(description)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.events_list.addItem(item)
        self._apply_event_selection(self._pending_event_selection)

    def _apply_event_selection(
        self, selection: dict[str, int] | list[int] | int | None
    ):
        if self.events_list is None or self.events_list.count() == 0:
            return

        if selection is None:
            for index in range(self.events_list.count()):
                self.events_list.item(index).setCheckState(Qt.Checked)
            return

        if isinstance(selection, dict):
            selected_ids = {int(value) for value in selection.values()}
        elif isinstance(selection, (list, tuple, set)):
            selected_ids = {int(value) for value in selection}
        else:
            selected_ids = {int(selection)}

        for index in range(self.events_list.count()):
            item = self.events_list.item(index)
            event_id = self.event_id_map.get(item.text())
            item.setCheckState(
                Qt.Checked
                if event_id in selected_ids
                else Qt.Unchecked
            )

    def get_event_selection(self):
        if (
            not self.show_events_list
            or self.events_list is None
            or not self.events_list.isVisible()
        ):
            return None  # Return empty if hidden

        selected_events: dict[str, int] = {}
        for i in range(self.events_list.count()):
            item = self.events_list.item(i)
            if item.checkState() == Qt.Checked:
                event_description = item.text()
                selected_events[event_description] = self.event_id_map[event_description]
        return selected_events or None

    def get_defaults(self) -> dict:
        return {
            "mode": "event",
            "resample": None,
            "tlim": (-0.800, 0.800),
            "fixed_duration": 2.0,
            "fixed_overlap": 0.5,
            "detrend": None,
            "event_id": None,
            "notch": None,
            "bandpass": (0.5, None),
            "pick_eeg_only": True,
            "reference": None,
        }

    def set_params(self, params: dict):
        defaults = self.get_defaults()
        mode = params.get("mode", "event")
        display_mode = next(
            (
                label
                for label, value in self.mode_map.items()
                if mode in {label, value}
            ),
            "Event-based",
        )
        self.mode_combobox.setCurrentText(display_mode)
        self.resample_edit.setText(to_display_string(params.get("resample")))
        tlim = params.get("tlim")
        if (
            not isinstance(tlim, (list, tuple))
            or len(tlim) != 2
            or tlim[0] is None
            or tlim[1] is None
        ):
            tlim = defaults["tlim"]
        fixed_duration = params.get("fixed_duration")
        if fixed_duration is None:
            fixed_duration = defaults["fixed_duration"]
        fixed_overlap = params.get("fixed_overlap")
        if fixed_overlap is None:
            fixed_overlap = defaults["fixed_overlap"]
        self.tlim_edit.setValue(tlim)
        self.fixed_dur_edit.setText(to_display_string(fixed_duration))
        self.fixed_overlap_edit.setText(to_display_string(fixed_overlap))

        detrend_val = params.get("detrend")
        # Reverse lookup for detrend combobox
        for k, v in self.detrend_map.items():
            if v == detrend_val:
                self.detrend_combobox.setCurrentText(k)
                break

        self.pick_eeg_only_checkbox.setChecked(params.get("pick_eeg_only"))
        self.notch_input.setText(to_display_string(params.get("notch")))
        self.bandpass_input.setValue(params.get("bandpass"))
        self.average_reference_checkbox.setChecked(
            params.get("reference") == "average"
        )
        self._pending_event_selection = params.get("event_id")
        self._apply_event_selection(self._pending_event_selection)

    def get_params(self) -> dict:
        mode = self.mode_combobox.currentText()
        mode_value = self.mode_map[mode]
        is_fixed = mode_value == "fixed"

        return {
            "mode": mode_value,
            # Event based params
            "event_id": self.get_event_selection() if not is_fixed else None,
            "tlim": self.tlim_edit.value() if not is_fixed else None,
            # Fixed length params
            "fixed_duration": (
                parse_numeric(self.fixed_dur_edit.text()) if is_fixed else None
            ),
            "fixed_overlap": (
                parse_numeric(self.fixed_overlap_edit.text()) if is_fixed else 0.0
            ),
            # Common params
            "resample": parse_numeric(self.resample_edit.text()),
            "detrend": self.detrend_map[self.detrend_combobox.currentText()],
            "pick_eeg_only": self.pick_eeg_only_checkbox.isChecked(),
            "notch": parse_numeric(self.notch_input.text()),
            "bandpass": self.bandpass_input.value(),
            "reference": (
                "average" if self.average_reference_checkbox.isChecked() else None
            ),
        }


class EpochingDialog(BaseProcessingDialog):
    def __init__(self, preprocessor, parent=None):
        settings_widget = EpochingSettingsWidget(show_events_list=True)
        super().__init__(preprocessor, "Epoch Segmentation", settings_widget, parent)

        # Populate events if raw data exists
        if self.preprocessor.has("raw"):
            self.settings_widget.populate_events_list(self.preprocessor.raw)
            # Force a visibility refresh after populating data
            current_mode = self.settings_widget.mode_combobox.currentText()
            self.settings_widget.update_ui_visibility(current_mode)
        self._apply_compact_size(
            min_size=QSize(560, 380),
            max_size=QSize(760, 700),
        )

    def run_process(self):
        try:
            params = self.settings_widget.get_params()
            # Basic validation for fixed length
            if params["mode"] == "fixed":
                if not params["fixed_duration"] or params["fixed_duration"] <= 0:
                    raise ValueError("Duration must be a positive number.")
            else:
                tlim = params.get("tlim")
                if (
                    tlim is None
                    or not isinstance(tlim, (list, tuple))
                    or len(tlim) != 2
                    or tlim[0] is None
                    or tlim[1] is None
                ):
                    raise ValueError(
                        "Time limits must include both start and stop values for event-based epoching."
                    )
                if tlim[0] >= tlim[1]:
                    raise ValueError(
                        "Time limits start must be before stop for event-based epoching."
                    )

            self.settings_widget.save_settings()  # Save for next time
            logger.info(f"Running epoching: {params}")
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", str(e))
            return

        if self.preprocessor.has("epochs"):
            reply = QMessageBox.question(
                self,
                "Warning",
                "Epochs data already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        worker = Worker(
            lambda: self.preprocessor.run_epoching(**params), parent=self.parent()
        )
        worker.exec_with_dialog("Please wait", "Running Epoching")
        self.accept()


# Re-referencing
class ReReferenceSettingsWidget(BaseSettingsWidget):
    """A widget for configuring re-referencing settings."""

    SETTINGS_PATH = "pipelines/{pipeline_id}/preprocessing/re_reference"
    LEGACY_KEYS = ("erp_preprocessing/re_reference",)

    def __init__(self, ch_names: list[str], parent=None):
        super().__init__(parent)
        self._selected_ref_channels = []
        self.ch_names = ch_names

        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.init_form(layout)
        self.load_settings()

    def init_form(self, layout):
        ref_group = QGroupBox("Reference")
        ref_layout = QVBoxLayout(ref_group)
        self.average_ref_radio = QRadioButton("Average of all channels")
        self.no_ref_radio = QRadioButton("Keep current reference")

        custom_ref_layout = QHBoxLayout()
        self.custom_ref_radio = QRadioButton("Custom channel(s):")
        self.select_channels_button = QPushButton("Select Channels...")
        self.select_channels_button.setEnabled(False)
        custom_ref_layout.addWidget(self.custom_ref_radio)
        custom_ref_layout.addWidget(self.select_channels_button)
        custom_ref_layout.addStretch()

        ref_layout.addWidget(self.average_ref_radio)
        ref_layout.addWidget(self.no_ref_radio)
        ref_layout.addLayout(custom_ref_layout)

        self.custom_ref_radio.toggled.connect(self.select_channels_button.setEnabled)
        self.select_channels_button.clicked.connect(self.show_channel_selection_dialog)

        layout.addWidget(ref_group)

    def show_channel_selection_dialog(self):
        dialog = ChannelSelectionDialog(
            self.ch_names, self._selected_ref_channels, self
        )
        if dialog.exec():
            self._selected_ref_channels = dialog.get_selected_channels()
            self.select_channels_button.setText(
                _selected_channels_button_text(
                    self._selected_ref_channels, "Select Channels..."
                )
            )

    def get_defaults(self) -> dict:
        return {
            "reference": None,
        }

    def set_params(self, params: dict):
        reference = params.get("reference")
        if reference == "average":
            self.average_ref_radio.setChecked(True)
        elif isinstance(reference, list):
            self.custom_ref_radio.setChecked(True)
            self._selected_ref_channels = reference
            self.select_channels_button.setText(
                _selected_channels_button_text(
                    self._selected_ref_channels, "Select Channels..."
                )
            )
        else:
            self.no_ref_radio.setChecked(True)

    def get_params(self) -> dict:
        params = {}
        if self.average_ref_radio.isChecked():
            params["reference"] = "average"
        elif self.no_ref_radio.isChecked():
            params["reference"] = None
        elif self.custom_ref_radio.isChecked():
            if not self._selected_ref_channels:
                raise ValueError(
                    "Custom reference selected, but no channels were chosen."
                )
            params["reference"] = self._selected_ref_channels
        return params


class ReReferenceDialog(BaseProcessingDialog):
    """A dialog for running the re-referencing process."""

    def __init__(self, preprocessor: Preprocessor, parent=None):
        settings_widget = ReReferenceSettingsWidget(
            ch_names=preprocessor.epochs.ch_names
        )
        super().__init__(preprocessor, "Re-reference", settings_widget, parent)

    def run_process(self):
        try:
            params = self.settings_widget.get_params()
            self.settings_widget.save_settings()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", str(e))
            return

        reply = QMessageBox.question(
            self,
            "Overwrite Data?",
            "Re-referencing will overwrite current epochs data. Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return
        worker = Worker(
            lambda: self.preprocessor.run_rereferencing(**params), parent=self.parent()
        )
        worker.exec_with_dialog("Please wait", "Running Re-referencing")
        self.accept()


# Apply Filters
class PreprocessingSettingsWidget(BaseSettingsWidget):
    """
    A reusable widget for configuring preprocessing steps with an improved UI.
    """

    SETTINGS_PATH = "pipelines/{pipeline_id}/preprocessing/apply"
    LEGACY_KEYS = ("erp_preprocessing/apply_params",)

    def __init__(self, preprocessor: Preprocessor | None = None, parent=None):
        super().__init__(parent)
        self.preprocessor = preprocessor
        self.ch_names = self.preprocessor.epochs.ch_names if self.preprocessor else []
        self._selected_bad_channels = []
        self._selected_ref_channels = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.init_form(layout)
        self.load_settings()

    def init_form(self, layout):
        groupbox = QGroupBox("Preprocessing")
        form_layout = QFormLayout(groupbox)

        # --- Extra bads ---
        self.plot_button = QPushButton("Quick plot")
        self.select_bad_channels_button = QPushButton("Extra Bad Channels")
        if self.preprocessor:
            self.plot_button.clicked.connect(self.quick_plot_evoked)
            self.select_bad_channels_button.clicked.connect(
                self.show_bad_channel_selection
            )
            form_layout.addRow(self.plot_button)
            form_layout.addRow("Extra bads:", self.select_bad_channels_button)

        self.bandpass_input = OptionalRangeWidget(suffix="Hz", parent=self)

        # --- Reference Selection ---
        ref_group = QGroupBox("Reference")
        ref_layout = QVBoxLayout(ref_group)
        self.average_ref_radio = QRadioButton("Average of all channels")
        self.no_ref_radio = QRadioButton("Keep current reference")

        custom_ref_layout = QHBoxLayout()
        self.custom_ref_radio = QRadioButton("Custom channel(s):")
        self.select_ref_channels_button = QPushButton("Select Channels...")
        self.select_ref_channels_button.setEnabled(False)
        custom_ref_layout.addWidget(self.custom_ref_radio)
        custom_ref_layout.addWidget(self.select_ref_channels_button)
        custom_ref_layout.addStretch()

        ref_layout.addWidget(self.average_ref_radio)
        ref_layout.addWidget(self.no_ref_radio)
        ref_layout.addLayout(custom_ref_layout)

        self.custom_ref_radio.toggled.connect(
            self.select_ref_channels_button.setEnabled
        )
        self.select_ref_channels_button.clicked.connect(
            self.show_ref_channel_selection_dialog
        )

        # --- Other widgets ---
        self.interpolate_bad_channels_check = QCheckBox()
        self.resample_input = QLineEdit()
        self.baseline_input = OptionalRangeWidget(
            ("Start:", "Stop:"),
            suffix="ms",
            required=(False, False),
            range=(None, None),
            parent=self,
            scale=1e-3,
        )

        # --- Add rows to form ---
        form_layout.addRow("Bandpass Filter:", self.bandpass_input)
        form_layout.addRow(
            "Interpolate Bad Channels:", self.interpolate_bad_channels_check
        )
        form_layout.addRow(ref_group)
        form_layout.addRow("Resample (Hz or 'None'):", self.resample_input)
        form_layout.addRow("Baseline Correction:", self.baseline_input)

        layout.addWidget(groupbox)

    def quick_plot_evoked(self):
        dialog = EvokedPlotDialog(parent=self)
        if self.preprocessor.has("epochs_ica"):
            logger.info(f"Applying ICA to epochs data...")
            epochs = self.preprocessor.epochs_ica.apply(
                self.preprocessor.epochs.load_data().copy()
            )
        else:
            logger.info(f"Preparing epochs data...")
            epochs = self.preprocessor.epochs.copy()
        dialog.update_plot(epochs, label=self.preprocessor.label)
        dialog.exec()

    def show_bad_channel_selection(self):
        dialog = ChannelSelectionDialog(
            self.ch_names, self._selected_bad_channels, self
        )
        if dialog.exec():
            self._selected_bad_channels = dialog.get_selected_channels()
            self.select_bad_channels_button.setText(
                _selected_channels_button_text(
                    self._selected_bad_channels, "Extra Bad Channels..."
                )
            )

    def show_ref_channel_selection_dialog(self):
        dialog = ChannelSelectionDialog(
            self.ch_names, self._selected_ref_channels, self
        )
        if dialog.exec():
            self._selected_ref_channels = dialog.get_selected_channels()
            self.select_ref_channels_button.setText(
                _selected_channels_button_text(
                    self._selected_ref_channels, "Select Channels..."
                )
            )

    def get_defaults(self) -> dict:
        return {
            "bads": None,
            "bandpass": (1, 80),
            "interpolate_bad_channels": True,
            "reference": None,
            "resample": None,
            "baseline": (None, 0),
        }

    def set_params(self, params: dict):
        self.bandpass_input.setValue(params.get("bandpass"))
        self.baseline_input.setValue(params.get("baseline"))
        self.interpolate_bad_channels_check.setChecked(
            params.get("interpolate_bad_channels")
        )
        self.resample_input.setText(str(params.get("resample", "None")))

        reference = params.get("reference")
        if reference == "average":
            self.average_ref_radio.setChecked(True)
        elif isinstance(reference, list):
            self.custom_ref_radio.setChecked(True)
            self._selected_ref_channels = reference
            self.select_ref_channels_button.setText(
                _selected_channels_button_text(
                    self._selected_ref_channels, "Select Channels..."
                )
            )
        else:
            self.no_ref_radio.setChecked(True)

    def get_params(self) -> dict:
        params = {}
        params["bads"] = self._selected_bad_channels
        params["bandpass"] = self.bandpass_input.value()
        params["interpolate_bad_channels"] = (
            self.interpolate_bad_channels_check.isChecked()
        )
        if self.average_ref_radio.isChecked():
            params["reference"] = "average"
        elif self.no_ref_radio.isChecked():
            params["reference"] = None
        elif self.custom_ref_radio.isChecked():
            if not self._selected_ref_channels:
                raise ValueError(
                    "Custom reference selected, but no channels were chosen."
                )
            params["reference"] = self._selected_ref_channels
        params["baseline"] = self.baseline_input.value()
        params["resample"] = parse_numeric(self.resample_input.text(), int)
        return params


class ApplyPreprocessingDialog(BaseProcessingDialog):
    """
    A dialog for applying preprocessing steps. (MODIFIED)
    """

    def __init__(
        self, preprocessor: Preprocessor, parent=None
    ):  # preprocessor: ERPpreprocessor
        settings_widget = PreprocessingSettingsWidget(preprocessor=preprocessor)
        super().__init__(preprocessor, "Apply Preprocessing", settings_widget, parent)

        if preprocessor.epochs.proj:
            self.settings_widget.average_ref_radio.setEnabled(False)
            self.settings_widget.custom_ref_radio.setEnabled(False)
            self.settings_widget.select_ref_channels_button.setEnabled(False)
            self.settings_widget.no_ref_radio.setEnabled(False)
            self.settings_widget.no_ref_radio.setChecked(True)

    def load_settings(self):
        """Delegates loading settings directly to the widget."""
        self.settings_widget.load_settings()

    def save_settings(self):
        """Delegates saving settings directly to the widget."""
        self.settings_widget.save_settings()

    def run_process(self):
        """Validates input and starts the background worker."""
        params = self.settings_widget.get_params()
        self.settings_widget.save_settings()

        if self.preprocessor.has("preprocessed"):
            reply = QMessageBox.question(
                self,
                "Overwrite Data?",
                "Preprocessed data already exists. Do you want to overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        if not self.preprocessor.epochs.preload:
            Worker(
                lambda: self.preprocessor.epochs.load_data(), parent=self.parent()
            ).exec_with_dialog("Loading...", "Loading epoched data...")

        logger.info(f"Running preprocessing with params: {params}")

        worker = Worker(
            lambda: self.preprocessor.filter_epochs(**params), parent=self.parent()
        )
        worker.exec_with_dialog("Loading", "Applying final preprocessing steps...")
        self.finished.emit()
        self.accept()


# Example Usage
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set up QSettings
    QApplication.setOrganizationName("SSPython")
    QApplication.setApplicationName("SSPy")
    main_settings = QSettings()

    raw_path = Path("/Users/brunoandrynascimentocouto/Desktop/Test Dataset")
    raw_files = sorted(list(raw_path.rglob("*raw.fif")))
    raw_file = raw_files[0]
    preprocessor = Preprocessor(raw_file, raw_path / "derivatives")

    dialog = ArtifactRemovalDialog(preprocessor)
    dialog.exec()

    dialog = EpochingDialog(preprocessor)
    dialog.exec()

    preprocessor.epochs.average().plot()
    app.close()
    sys.exit(app.exec())
