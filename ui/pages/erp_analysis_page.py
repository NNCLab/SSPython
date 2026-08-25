import logging
import json
from datetime import datetime
from pathlib import Path

import mne

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.mne_compat import (
    install_mne_brain_empty_label_compat,
    install_mne_brain_vertex_picker_compat,
)
from core.processing import TMSEEGAnalysis
from ui.widgets.evoked_plot import EvokedPlotWidget
from ui.widgets.psd_plot import PSDPlotSettingsDialog
from ui.widgets.tep_analysis.excitability import ExcitabilityApp
from ui.widgets.tep_analysis.natural_frequency import NaturalFrequencyApp
from ui.widgets.source_estimate_widget import (
    ComputeSTCSettingsDialog,
    STCSourceConfig,
    compute_stc,
    metadata_to_source_config,
    prepare_stc_source_model,
)
from ui.widgets.time_frequency_widget import ComputeTFRSettingsDialog, TimeFrequencyDialog
from ui.widgets.tools.object_info_widget import ObjectInfoWidget
from utils import Worker

from .base_page import BasePage

logger = logging.getLogger(__name__)


class TopoplotTimeWindowDialog(QDialog):
    """Collect a time window constrained to the loaded epoch limits."""

    def __init__(self, epochs: mne.BaseEpochs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Topoplot Time Window")
        self.setModal(True)
        self.setMinimumWidth(420)

        times = epochs.times
        self.time_bounds = (float(times[0]), float(times[-1]))
        sample_step = 1.0 / float(epochs.info["sfreq"])

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.start_input = self._create_time_input(self.time_bounds[0], sample_step)
        self.end_input = self._create_time_input(self.time_bounds[1], sample_step)
        form.addRow("Start:", self.start_input)
        form.addRow("End:", self.end_input)
        layout.addLayout(form)

        bounds_label = QLabel(
            f"Available range: {self.time_bounds[0]:.6f} to "
            f"{self.time_bounds[1]:.6f} s"
        )
        bounds_label.setObjectName("analysisMethodDescription")
        layout.addWidget(bounds_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Plot")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _create_time_input(self, value: float, step: float) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox(self)
        spinbox.setDecimals(6)
        spinbox.setRange(*self.time_bounds)
        spinbox.setSingleStep(step)
        spinbox.setSuffix(" s")
        spinbox.setValue(value)
        return spinbox

    def accept(self):
        tmin, tmax = self.get_time_window()
        if tmin >= tmax:
            QMessageBox.warning(
                self,
                "Invalid Time Window",
                "Start time must be earlier than end time.",
            )
            return
        super().accept()

    def get_time_window(self) -> tuple[float, float]:
        return float(self.start_input.value()), float(self.end_input.value())


class ErpAnalysisPage(BasePage):
    def __init__(self, parent=None):
        super().__init__("Analysis", parent)
        self.set_content_maximum_width(1360)
        self.main_window = parent
        self.current_dataset = None
        self.tep: TMSEEGAnalysis | None = None
        self._needs_reload = False
        self.time_frequency_dialogs: list[TimeFrequencyDialog] = []
        self.source_estimate_dialogs: list[ComputeSTCSettingsDialog] = []
        self.source_estimate_brain = None

        self.context_label = QLabel("Select a dataset that has reached the preprocessed stage.")
        self.context_label.setObjectName("contextCard")
        self.context_label.setWordWrap(True)
        self.add_content(self.context_label)

        self.epochs_info_widget = ObjectInfoWidget()
        self.visualization_group = self._create_visualization_group()
        self.analysis_group = self._create_analysis_group()

        self.visualization_title = QLabel("Visualization")
        self.visualization_title.setObjectName("sectionTitle")
        self.add_content(self.visualization_title)
        self.add_content(self.visualization_group)
        self.analysis_title = QLabel("Analysis methods")
        self.analysis_title.setObjectName("sectionTitle")
        self.add_content(self.analysis_title)
        self.add_content(self.analysis_group)

        self._connect_to_main_window()
        self.update_ui_state()

    def _create_visualization_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)
        layout.addWidget(self.epochs_info_widget)

        self.evoked_plot_widget = EvokedPlotWidget()
        self.evoked_plot_widget.scrolled.connect(
            lambda direction: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 50
                if direction == "up"
                else self.scroll_area.verticalScrollBar().value() + 50
            )
        )
        layout.addWidget(self.evoked_plot_widget)

        self.topoplot_button = QPushButton("Topoplot")
        self.psd_button = QPushButton("Plot PSD")
        self.topoplot_button.setObjectName("secondaryButton")
        self.psd_button.setObjectName("secondaryButton")
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.topoplot_button)
        action_row.addWidget(self.psd_button)
        layout.addLayout(action_row)
        return group

    def _create_analysis_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(24)
        layout.setVerticalSpacing(18)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.excitability_button = QPushButton("Response Amplitude Analysis")
        self.nf_button = QPushButton("Natural Frequency Analysis")
        self.tf_button = QPushButton("Compute TFR")
        self.time_frequency_button = QPushButton("Plot TFR")
        self.stc_button = QPushButton("Compute STC")
        self.source_estimate_button = QPushButton("Plot STC")

        self._add_analysis_method(
            layout,
            0,
            0,
            "Response amplitude",
            "Measure response strength over a selected time window.",
            self.excitability_button,
        )
        self._add_analysis_method(
            layout,
            0,
            1,
            "Natural frequency",
            "Estimate dominant response frequencies across channels.",
            self.nf_button,
        )
        self._add_analysis_method(
            layout,
            1,
            0,
            "Time-frequency response",
            "Compute a TFR derivative or open an existing result.",
            self.tf_button,
            self.time_frequency_button,
        )
        self._add_analysis_method(
            layout,
            1,
            1,
            "Source estimate",
            "Compute and inspect a cortical source estimate.",
            self.stc_button,
            self.source_estimate_button,
        )
        return group

    @staticmethod
    def _add_analysis_method(layout, row, column, title, description, *buttons):
        method_layout = QVBoxLayout()
        method_layout.setContentsMargins(0, 0, 0, 0)
        method_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("analysisMethodTitle")
        method_layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setObjectName("analysisMethodDescription")
        description_label.setWordWrap(True)
        method_layout.addWidget(description_label)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 2, 0, 0)
        for button in buttons:
            button.setProperty("analysisAction", True)
            button_row.addWidget(button)
        button_row.addStretch(1)
        method_layout.addLayout(button_row)
        layout.addLayout(method_layout, row, column)

    def _connect_to_main_window(self):
        if not self.main_window:
            return
        self.main_window.current_dataset_changed.connect(self.on_dataset_changed)
        self.main_window.current_pipeline_changed.connect(self.on_pipeline_changed)
        self.current_dataset = self.main_window.current_dataset
        self._needs_reload = True

        self.topoplot_button.clicked.connect(self.topoplot)
        self.psd_button.clicked.connect(self.plot_psd)
        self.time_frequency_button.clicked.connect(self.open_time_frequency_widget)
        self.excitability_button.clicked.connect(self.run_excitability)
        self.nf_button.clicked.connect(self.run_natural_frequency)
        self.tf_button.clicked.connect(self.run_time_frequency)
        self.stc_button.clicked.connect(self.run_source_estimate)
        self.source_estimate_button.clicked.connect(self.plot_source_estimate)

    @Slot(object)
    def on_dataset_changed(self, dataset):
        self.current_dataset = dataset
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()

    @Slot(object)
    def on_pipeline_changed(self, pipeline):
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()

    def _ensure_loaded(self):
        if not self.current_dataset or not self.current_dataset.stage_exists("preprocessed"):
            self.tep = None
            self._needs_reload = False
            return

        if not self._needs_reload and self.tep is not None:
            return

        worker = Worker(
            lambda: TMSEEGAnalysis(
                self.current_dataset.paths["preprocessed"],
                output_dir=self.current_dataset.derivative_root,
                preload=True,
            ),
            parent=self,
            add_loggers="mne",
        )
        self.tep = worker.exec_with_dialog("Please wait", "Loading preprocessed data...")
        self._needs_reload = False

    def update_ui_state(self):
        pipeline_name = self.main_window.current_pipeline.name if self.main_window else "Pipeline"
        if not self.current_dataset:
            self.set_page_subtitle("Select a dataset in the workspace panel to inspect preprocessed epochs.")
            self.context_label.setText("Select a dataset that has reached the preprocessed stage.")
            self.context_label.setToolTip("")
            self.context_label.show()
        elif not self.current_dataset.stage_exists("preprocessed"):
            self.set_page_subtitle(f"{pipeline_name} pipeline · {self.current_dataset.display_name}")
            self.context_label.setText(
                "This dataset is not at the preprocessed stage yet. Complete preprocessing first."
            )
            self.context_label.setToolTip(str(self.current_dataset.raw_path))
            self.context_label.show()
        else:
            self.set_page_subtitle(f"{pipeline_name} pipeline · {self.current_dataset.display_name}")
            self.context_label.hide()

        self._ensure_loaded()

        has_data = self.tep is not None and self.tep.epochs is not None
        if has_data:
            self.epochs_info_widget.update_info(self.tep.epochs, "Preprocessed epochs")
            self.evoked_plot_widget.update_plot(self.tep.epochs, self.tep.label)
        else:
            self.epochs_info_widget.clear_info()
            self.evoked_plot_widget.update_plot(None)

        self.topoplot_button.setEnabled(has_data)
        self.psd_button.setEnabled(has_data)
        self.time_frequency_button.setEnabled(has_data and self._has_time_frequency_derivatives())
        self.source_estimate_button.setEnabled(has_data and self._has_source_estimate_derivative())
        self.excitability_button.setEnabled(has_data)
        self.nf_button.setEnabled(has_data)
        self.tf_button.setEnabled(has_data)
        self.stc_button.setEnabled(has_data)
        self.evoked_plot_widget.setVisible(has_data)
        self.visualization_title.setVisible(has_data)
        self.visualization_group.setVisible(has_data)
        self.analysis_title.setVisible(has_data)
        self.analysis_group.setVisible(has_data)

    def release_derivative_resources(self, paths):
        """Drop analysis objects derived from a file that will be deleted."""
        if self.tep is None:
            return
        target_paths = {Path(path).resolve() for path in paths}
        if self.tep.data_input.resolve() not in target_paths:
            return

        self.epochs_info_widget.clear_info()
        self.evoked_plot_widget.update_plot(None)
        self.tep = None
        self._needs_reload = True

    @Slot()
    def on_settings_updated(self):
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()

    def topoplot(self):
        if not self.tep or self.tep.epochs is None:
            return
        dialog = TopoplotTimeWindowDialog(self.tep.epochs, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        tmin, tmax = dialog.get_time_window()
        event_names = list(self.tep.epochs.event_id.keys()) if self.tep.epochs.event_id else []
        if len(event_names) > 1:
            evokeds = [
                self.tep.epochs[event_name]
                .average()
                .apply_baseline()
                .crop(tmin=tmin, tmax=tmax)
                for event_name in event_names
            ]
            colors = [f"C{index % 10}" for index in range(len(evokeds))]
            figure = mne.viz.plot_evoked_topo(
                evokeds,
                color=colors,
                title=self.tep.label,
                legend=True,
                show=True,
            )
        else:
            figure = (
                self.tep.evoked.copy()
                .apply_baseline()
                .crop(tmin=tmin, tmax=tmax)
                .plot_topo(title=self.tep.label)
            )
        for axes in figure.axes:
            for line in axes.lines:
                line.set_linewidth(1.25)

    def run_excitability(self):
        if not self.tep:
            return
        self.exc_app = ExcitabilityApp(self.tep.epochs, parent=self)

    def run_natural_frequency(self):
        if not self.tep:
            return
        dialog = NaturalFrequencyApp(self.tep.epochs, parent=self)
        dialog.exec()

    def plot_psd(self):
        if not self.tep or self.tep.epochs is None:
            return

        dialog = PSDPlotSettingsDialog(self.tep.epochs, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            self.tep.epochs.plot_psd(**dialog.get_plot_params(), show=True)
        except Exception as exc:
            logger.exception("PSD plotting failed")
            QMessageBox.critical(self, "PSD Plot Failed", str(exc))

    def _available_time_frequency_derivatives(self) -> dict[str, object]:
        if not self.tep or not hasattr(self.tep, "derivatives"):
            return {}
        derivatives = self.tep.derivatives or {}
        available: dict[str, object] = {}
        if derivatives.get("tfr") is not None:
            available["Power (TFR)"] = derivatives["tfr"]
        if derivatives.get("itc") is not None:
            available["Inter-trial coherence (ITC)"] = derivatives["itc"]
        return available

    def _has_time_frequency_derivatives(self) -> bool:
        return bool(self._available_time_frequency_derivatives())

    def _available_source_estimate(self):
        if not self.tep or not hasattr(self.tep, "derivatives"):
            return None
        derivatives = self.tep.derivatives or {}
        return derivatives.get("stc")

    def _has_source_estimate_derivative(self) -> bool:
        return self._available_source_estimate() is not None

    def _forget_time_frequency_dialog(self, dialog: TimeFrequencyDialog):
        if dialog in self.time_frequency_dialogs:
            self.time_frequency_dialogs.remove(dialog)

    def _show_time_frequency_dialog(self):
        if not self.tep or self.tep.epochs is None:
            return
        tfr_derivatives = self._available_time_frequency_derivatives()
        if not tfr_derivatives:
            return
        dialog = TimeFrequencyDialog(
            self.tep.epochs,
            label=self.tep.label,
            tfr_derivatives=tfr_derivatives,
            parent=self,
        )
        dialog.finished.connect(lambda: self._forget_time_frequency_dialog(dialog))
        self.time_frequency_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def open_time_frequency_widget(self):
        if not self.tep or self.tep.epochs is None:
            return
        if not self._has_time_frequency_derivatives():
            return
        self._show_time_frequency_dialog()

    def run_time_frequency(self):
        if not self.tep:
            return
        dialog = ComputeTFRSettingsDialog(self.tep.epochs, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        compute_params = dialog.get_compute_params()

        def compute_and_save():
            result = self.tep.epochs.copy().compute_tfr(**compute_params)
            if isinstance(result, tuple):
                tfr, itc = result
            else:
                tfr, itc = result, None

            self.tep.derivatives["tfr"] = tfr
            tfr.save(self.tep.paths["tfr"], overwrite=True, verbose=False)
            if itc is not None:
                self.tep.derivatives["itc"] = itc
                itc.save(self.tep.paths["itc"], overwrite=True, verbose=False)
            return result

        worker = Worker(
            compute_and_save,
            parent=self,
            add_loggers="mne",
        )
        worker.exec_with_dialog("Please wait", "Running Time-Frequency Analysis...")
        if getattr(worker, "_error", None):
            return
        if hasattr(self.tep, "_get_derivatives"):
            self.tep.derivatives = self.tep._get_derivatives()
        self._needs_reload = True
        self.update_ui_state()
        self.open_time_frequency_widget()

    def run_source_estimate(self):
        if not self.tep or self.tep.epochs is None:
            return
        dialog = ComputeSTCSettingsDialog(self.tep.epochs, parent=self)
        dialog.accepted.connect(lambda: self._compute_source_estimate_from_dialog(dialog))
        dialog.finished.connect(lambda: self._forget_source_estimate_dialog(dialog))
        self.source_estimate_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _compute_source_estimate_from_dialog(self, dialog: ComputeSTCSettingsDialog):
        if not self.tep or self.tep.epochs is None:
            return
        compute_params = dialog.get_compute_params()

        def compute_and_save():
            stc, source_model = compute_stc(self.tep.epochs, **compute_params)
            self.tep.paths["stc"].parent.mkdir(parents=True, exist_ok=True)
            stc.save(self.tep.paths["stc"], ftype="h5", overwrite=True, verbose=False)

            metadata = self._stc_metadata(compute_params, source_model)
            self.tep.paths["stc_metadata"].write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            return stc, metadata

        worker = Worker(
            compute_and_save,
            parent=self,
            add_loggers="mne",
        )
        result = worker.exec_with_dialog("Please wait", "Computing source time course...")
        if result is None or getattr(worker, "_error", None):
            return

        stc, metadata = result
        self.tep.derivatives["stc"] = stc
        self.tep.derivatives["stc_metadata"] = metadata
        if hasattr(self.tep, "_get_derivatives"):
            self.tep.derivatives = self.tep._get_derivatives()
        self._needs_reload = True
        self.update_ui_state()

    def _forget_source_estimate_dialog(self, dialog: ComputeSTCSettingsDialog):
        if dialog in self.source_estimate_dialogs:
            self.source_estimate_dialogs.remove(dialog)

    def plot_source_estimate(self):
        if not self.tep:
            return
        stc = self._available_source_estimate()
        if stc is None:
            return

        source_config = self._source_config_for_stc_plot(stc)
        if source_config is None:
            QMessageBox.warning(
                self,
                "Missing STC Metadata",
                "This STC does not include enough source-space metadata to plot. "
                "Recompute it with the source files selected.",
            )
            return

        def prepare_plot_source_model():
            source_model = prepare_stc_source_model(source_config)
            source_spaces = mne.read_source_spaces(source_model.src, verbose=False)
            return source_model, source_spaces

        worker = Worker(
            prepare_plot_source_model,
            parent=self,
            add_loggers="mne",
        )
        plot_source_model = worker.exec_with_dialog(
            "Please wait",
            "Preparing source model files...",
        )
        if plot_source_model is None:
            return
        source_model, source_spaces = plot_source_model

        try:
            install_mne_brain_vertex_picker_compat()
            install_mne_brain_empty_label_compat()
            self.source_estimate_brain = stc.plot(
                subject=source_model.subject,
                subjects_dir=source_model.subjects_dir,
                src=source_spaces,
                backend="pyvistaqt",
                hemi=self._stc_plot_hemi(stc),
                colormap="turbo",
                views="dorsal",
                initial_time=self._stc_initial_time(stc),
                time_unit="ms",
                size=(800, 800),
                smoothing_steps=5,
                time_viewer=True,
                surface="pial",
                title=self.tep.label,
            )
        except Exception as exc:
            logger.exception("STC plotting failed")
            QMessageBox.critical(self, "STC Plot Failed", str(exc))

    def _source_config_for_stc_plot(self, stc) -> STCSourceConfig | None:
        derivatives = self.tep.derivatives if self.tep and self.tep.derivatives else {}
        source_config = metadata_to_source_config(derivatives.get("stc_metadata"))
        if source_config is not None:
            return source_config

        subject = getattr(stc, "subject", None) or "fsaverage"
        if subject == "fsaverage":
            return STCSourceConfig(mode="fsaverage")
        return None

    @staticmethod
    def _stc_plot_hemi(stc) -> str:
        vertices = getattr(stc, "vertices", [])
        if len(vertices) > 1 and len(vertices[0]) and len(vertices[1]):
            return "both"
        if len(vertices) > 1 and len(vertices[1]):
            return "rh"
        return "lh"

    @staticmethod
    def _stc_initial_time(stc) -> float | None:
        vertices = getattr(stc, "vertices", [])
        hemi = "rh" if len(vertices) > 1 and len(vertices[1]) else "lh"
        try:
            _, time_max = stc.get_peak(hemi=hemi)
        except Exception:
            return None
        return float(time_max)

    @classmethod
    def _stc_metadata(cls, compute_params: dict[str, object], source_model) -> dict[str, object]:
        return {
            "created": datetime.now().isoformat(),
            "source": source_model.to_metadata(),
            "compute": {
                key: cls._json_safe(value)
                for key, value in compute_params.items()
                if key != "source_config"
            },
        }

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, STCSourceConfig):
            return {
                "mode": value.mode,
                "subject": value.subject,
                "subjects_dir": cls._json_safe(value.subjects_dir),
                "src": cls._json_safe(value.src),
                "bem": cls._json_safe(value.bem),
                "trans": cls._json_safe(value.trans),
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        return value
