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
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.mne_compat import (
    install_mne_brain_empty_label_compat,
    install_mne_brain_vertex_picker_compat,
)
from core.processing import TMSEEGAnalysis
from core.baseline import validate_baseline
from core.tfr_statistics import compute_tfr_statistics, save_tfr_statistics
from core.analysis_events import (
    compute_event_tfrs,
    event_label,
    event_options,
    select_event_epochs,
)
from ui.widgets.event_selection import EventSelector
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
from ui.widgets.layout_helpers import FlowLayout, dataset_label
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
        self.event_selector = EventSelector(epochs, self)
        form.addRow("Event:", self.event_selector)
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

    def get_event_code(self) -> int | None:
        return self.event_selector.currentData()


class ErpAnalysisPage(BasePage):
    def __init__(self, parent=None):
        super().__init__("Analysis", parent)
        self.set_content_maximum_width(1360)
        self.main_window = parent
        self.subtitle_link.activated.connect(self._inspect_input)
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
        self.select_input_button = QPushButton("Open workspace")
        self.select_input_button.clicked.connect(self._select_input)
        empty_actions = FlowLayout()
        empty_actions.addWidget(self.select_input_button)
        self.add_content(empty_actions)

        self.epochs_info_widget = ObjectInfoWidget(compact=True)
        self.add_content(self.epochs_info_widget)
        self.visualization_group = self._create_visualization_group()
        self.analysis_group = self._create_analysis_group()

        self.add_content(self.analysis_group)
        self.add_content(self.visualization_group)
        self.content_layout.setStretch(self.content_layout.indexOf(self.visualization_group), 1)
        self.content_layout.setStretch(self.content_layout.count() - 1, 0)
        self.evoked_plot_widget.canvas.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )

        self._connect_to_main_window()
        self.update_ui_state()

    def _inspect_input(self):
        if self.main_window and self.current_dataset and self.current_dataset.stage_exists("preprocessed"):
            self.main_window.inspect_dataset_stage("preprocessed")

    def _select_input(self):
        if not self.main_window:
            return
        if not self.main_window.current_folder:
            self.main_window.workspace_panel.select_folder()
        elif not self.main_window.workspace_panel.datasets:
            self.main_window.navigate_to_page("Preprocessing")
        else:
            self.main_window.focus_workspace()

    def _create_visualization_group(self) -> QWidget:
        group = QWidget()
        group.setObjectName("plotSurface")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)

        self.evoked_plot_widget = EvokedPlotWidget()
        self.evoked_plot_widget.title_label_action.setVisible(False)
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
        return group

    def _create_analysis_group(self) -> QWidget:
        group = QWidget()
        group.setObjectName("analysisActions")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.analysis_title = QLabel("Analysis tools")
        self.analysis_title.setObjectName("analysisMethodTitle")
        layout.addWidget(self.analysis_title)
        self.excitability_button = QPushButton("Response amplitude")
        self.nf_button = QPushButton("Natural frequency")
        self.tf_button = QPushButton("Compute time-frequency")
        self.time_frequency_button = QPushButton("Plot time-frequency")
        self.stc_button = QPushButton("Compute source estimate")
        self.source_estimate_button = QPushButton("Plot source estimate")
        button_row = FlowLayout()
        for button, description in (
            (self.topoplot_button, "Inspect event responses and topographic maps."),
            (self.psd_button, "Inspect the power spectral density of preprocessed epochs."),
            (self.excitability_button, "Measure response amplitude over a selected time window."),
            (self.nf_button, "Estimate dominant response frequencies across channels."),
            (self.tf_button, "Compute a time-frequency response (TFR/ITC)."),
            (self.time_frequency_button, "Open a saved time-frequency result. Compute time-frequency first if no result exists."),
            (self.stc_button, "Compute a cortical source estimate (STC)."),
            (self.source_estimate_button, "Open a saved source estimate. Compute a source estimate first if no result exists."),
        ):
            button.setProperty("analysisAction", True)
            button.setToolTip(description)
            button_row.addWidget(button)
        layout.addLayout(button_row)
        return group

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
        self.subtitle_link.set_available(bool(self.current_dataset and self.current_dataset.stage_exists("preprocessed")))
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
            self.context_label.setToolTip(str(self.current_dataset.source_path))
            self.context_label.show()
        else:
            self.set_page_subtitle(f"{pipeline_name} pipeline · {self.current_dataset.display_name}")
            self.context_label.hide()
        if self.current_dataset:
            input_path = self.current_dataset.paths.get("preprocessed", self.current_dataset.source_path)
            self.set_page_subtitle(f"{dataset_label(self.current_dataset, path=input_path)} · {pipeline_name} pipeline")
            self.subtitle_label.setToolTip(str(input_path))

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
        self.visualization_group.setVisible(has_data)
        self.analysis_title.setVisible(has_data)
        self.analysis_group.setVisible(has_data)
        self.select_input_button.setVisible(not has_data)
        has_workspace = bool(self.main_window and self.main_window.current_folder)
        has_candidates = bool(self.main_window and self.main_window.workspace_panel.datasets)
        self.select_input_button.setText("Open workspace" if not has_workspace else "Choose dataset" if has_candidates else "Open Preprocessing")

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
        event_code = dialog.get_event_code()
        try:
            validate_baseline(
                self.tep.epochs.times, self.evoked_plot_widget.baseline_control.value(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Baseline", str(exc))
            return
        options = event_options(self.tep.epochs)
        if event_code is None and len(options) > 1:
            evokeds = []
            for code, _, _ in options:
                evoked = (
                    self.evoked_plot_widget.apply_display_baseline(
                        select_event_epochs(self.tep.epochs, code).average()
                    ).crop(tmin=tmin, tmax=tmax)
                )
                evoked.comment = event_label(self.tep.epochs, code)
                evokeds.append(evoked)
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
                self.evoked_plot_widget.apply_display_baseline(
                    select_event_epochs(self.tep.epochs, event_code).average()
                )
                .crop(tmin=tmin, tmax=tmax)
                .plot_topo(title=f"{self.tep.label} - {event_label(self.tep.epochs, event_code)}")
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
            figure = dialog.get_selected_epochs().plot_psd(**dialog.get_plot_params(), show=True)
            figure.suptitle(f"{self.tep.label} - {dialog.event_selector.selected_label()}")
        except Exception as exc:
            logger.exception("PSD plotting failed")
            QMessageBox.critical(self, "PSD Plot Failed", str(exc))

    def _available_time_frequency_derivatives(self) -> dict[str, object]:
        if not self.tep or not hasattr(self.tep, "derivatives"):
            return {}
        derivatives = self.tep.derivatives or {}
        available: dict[str, object] = {}
        if derivatives.get("tfr") is not None:
            available["Power (TFR)"] = derivatives.get("tfr_events", derivatives["tfr"])
        if derivatives.get("itc") is not None:
            available["Inter-trial coherence (ITC)"] = derivatives.get("itc_events", derivatives["itc"])
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
            tfr_statistics={"Power (TFR)": self.tep.derivatives.get("tfr_statistics")},
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
        statistics_params = dialog.get_statistics_params()
        event_code = dialog.event_selector.currentData()
        analysis = self.tep

        def compute_and_save():
            result = compute_event_tfrs(analysis.epochs, event_code=event_code, **compute_params)
            if isinstance(result, tuple):
                tfr, itc = result
            else:
                tfr, itc = result, None

            statistics = None if statistics_params is None else compute_tfr_statistics(
                analysis.epochs, tfr, event_code=event_code,
                compute_params=compute_params, settings=statistics_params,
            )

            mne.time_frequency.write_tfrs(analysis.paths["tfr"], tfr, overwrite=True, verbose=False)
            save_tfr_statistics(analysis.paths["tfr_statistics"], statistics)
            analysis.derivatives["tfr_statistics"] = statistics
            for key in ("tfr_mask", "itc_mask"):
                analysis.paths[key].unlink(missing_ok=True)
                analysis.derivatives[key] = None
            analysis.derivatives["tfr"] = tfr[0] if isinstance(tfr, list) else tfr
            analysis.derivatives.pop("tfr_events", None)
            if isinstance(tfr, list):
                analysis.derivatives["tfr_events"] = tfr
            analysis.derivatives.pop("itc_events", None)
            if itc is not None:
                mne.time_frequency.write_tfrs(analysis.paths["itc"], itc, overwrite=True, verbose=False)
                analysis.derivatives["itc"] = itc[0] if isinstance(itc, list) else itc
                if isinstance(itc, list):
                    analysis.derivatives["itc_events"] = itc
            else:
                analysis.paths["itc"].unlink(missing_ok=True)
                analysis.derivatives["itc"] = None
            return result

        worker = Worker(
            compute_and_save,
            parent=self,
            add_loggers="mne",
        )
        worker.exec_with_dialog("Please wait", "Running Time-Frequency Analysis...")
        if getattr(worker, "_error", None):
            return
        if hasattr(analysis, "_get_derivatives"):
            analysis.derivatives = analysis._get_derivatives()
        if self.tep is analysis:
            self._needs_reload = True
            self.update_ui_state()
            self.open_time_frequency_widget()

    def run_source_estimate(self):
        if not self.tep or self.tep.epochs is None:
            return
        analysis = self.tep
        dialog = ComputeSTCSettingsDialog(analysis.epochs, parent=self)
        dialog.accepted.connect(lambda: self._compute_source_estimate_from_dialog(dialog, analysis))
        dialog.finished.connect(lambda: self._forget_source_estimate_dialog(dialog))
        self.source_estimate_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _compute_source_estimate_from_dialog(
        self, dialog: ComputeSTCSettingsDialog, analysis: TMSEEGAnalysis | None = None,
    ):
        analysis = analysis if analysis is not None else self.tep
        if not analysis or analysis.epochs is None:
            return
        compute_params = dialog.get_compute_params()

        def compute_and_save():
            stc, source_model = compute_stc(analysis.epochs, **compute_params)
            analysis.paths["stc"].parent.mkdir(parents=True, exist_ok=True)
            stc.save(analysis.paths["stc"], ftype="h5", overwrite=True, verbose=False)

            metadata = self._stc_metadata(compute_params, source_model)
            analysis.paths["stc_metadata"].write_text(
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
        analysis.derivatives["stc"] = stc
        analysis.derivatives["stc_metadata"] = metadata
        if hasattr(analysis, "_get_derivatives"):
            analysis.derivatives = analysis._get_derivatives()
        if self.tep is analysis:
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
                title=self._stc_plot_title(),
            )
        except Exception as exc:
            logger.exception("STC plotting failed")
            QMessageBox.critical(self, "STC Plot Failed", str(exc))

    def _stc_plot_title(self) -> str:
        metadata = self.tep.derivatives.get("stc_metadata") or {}
        if "condition" not in metadata.get("compute", {}):
            return self.tep.label
        condition = metadata.get("compute", {}).get("condition")
        epochs = getattr(self.tep, "epochs", None)
        if epochs is None:
            return self.tep.label
        if isinstance(condition, str):
            if condition not in epochs.event_id:
                return f"{self.tep.label} - {condition}"
            condition = epochs.event_id[condition]
        return f"{self.tep.label} - {event_label(epochs, condition)}"

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
