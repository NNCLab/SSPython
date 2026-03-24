import logging

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout

from core.processing import TMSEEGAnalysis
from ui.widgets.evoked_plot import EvokedPlotWidget
from ui.widgets.tep_analysis.excitability import ExcitabilityApp
from ui.widgets.tep_analysis.natural_frequency import NaturalFrequencyApp
from ui.widgets.tools.object_info_widget import ObjectInfoWidget
from utils import Worker

from .base_page import BasePage

logger = logging.getLogger(__name__)


class ErpAnalysisPage(BasePage):
    def __init__(self, parent=None):
        super().__init__("ERP / TEP Analysis", parent)
        self.main_window = parent
        self.current_dataset = None
        self.tep: TMSEEGAnalysis | None = None
        self._needs_reload = False

        self.context_label = QLabel("Select a dataset that has reached the preprocessed stage.")
        self.context_label.setObjectName("contextCard")
        self.context_label.setWordWrap(True)
        self.add_content(self.context_label)

        self.epochs_info_widget = ObjectInfoWidget()
        self.visualization_group = self._create_visualization_group()
        self.analysis_group = self._create_analysis_group()

        self.add_content(QLabel("<h2>Visualization</h2>"))
        self.add_content(self.visualization_group)
        self.add_content(QLabel("<h2>Analysis Methods</h2>"))
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
        layout.addWidget(self.topoplot_button)
        return group

    def _create_analysis_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)
        self.excitability_button = QPushButton("Response Amplitude Analysis")
        self.nf_button = QPushButton("Natural Frequency Analysis")
        self.tf_button = QPushButton("Run Time-Frequency Analysis")

        layout.addWidget(self.excitability_button)
        layout.addWidget(self.nf_button)
        layout.addWidget(self.tf_button)
        return group

    def _connect_to_main_window(self):
        if not self.main_window:
            return
        self.main_window.current_dataset_changed.connect(self.on_dataset_changed)
        self.main_window.current_pipeline_changed.connect(self.on_pipeline_changed)
        self.current_dataset = self.main_window.current_dataset
        self._needs_reload = True

        self.topoplot_button.clicked.connect(self.topoplot)
        self.excitability_button.clicked.connect(self.run_excitability)
        self.nf_button.clicked.connect(self.run_natural_frequency)
        self.tf_button.clicked.connect(self.run_time_frequency)

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
            self.evoked_plot_widget.update_plot(self.tep.evoked, self.tep.label)
        else:
            self.epochs_info_widget.clear_info()
            self.evoked_plot_widget.update_plot(None)

        self.topoplot_button.setEnabled(has_data)
        self.excitability_button.setEnabled(has_data)
        self.nf_button.setEnabled(has_data)
        self.tf_button.setEnabled(has_data)
        self.evoked_plot_widget.setVisible(has_data)

    @Slot()
    def on_settings_updated(self):
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()

    def topoplot(self):
        if not self.tep:
            return
        figure = self.tep.evoked.copy().apply_baseline().plot_topo(title=self.tep.label)
        for line in figure.axes[0].lines:
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

    def run_time_frequency(self):
        if not self.tep:
            return
        worker = Worker(
            lambda: self.tep.run_tfr_analysis(overwrite=True),
            parent=self,
            add_loggers="mne",
        )
        worker.exec_with_dialog("Please wait", "Running Time-Frequency Analysis...")
        self._needs_reload = True
        self.update_ui_state()
