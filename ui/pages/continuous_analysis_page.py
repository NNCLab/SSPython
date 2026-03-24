import logging

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout

from core.processing import Preprocessor
from ui.widgets.psd_plot import PSDPlotWidget
from ui.widgets.tools.object_info_widget import ObjectInfoWidget
from utils import Worker

from .base_page import BasePage

logger = logging.getLogger(__name__)


class ContinuousAnalysisPage(BasePage):
    def __init__(self, parent=None):
        super().__init__("Continuous Analysis", parent)
        self.main_window = parent
        self.current_dataset = None
        self.preprocessor: Preprocessor | None = None
        self._needs_reload = False

        self.context_label = QLabel("Select a dataset from the workspace panel.")
        self.context_label.setObjectName("contextCard")
        self.context_label.setWordWrap(True)
        self.add_content(self.context_label)

        self.raw_info_widget = ObjectInfoWidget()
        self.visualization_group = self._create_visualization_group()
        self.add_content(QLabel("<h2>Power Spectral Density</h2>"))
        self.add_content(self.visualization_group)

        self._connect_to_main_window()
        self.update_ui_state()

    def _create_visualization_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)
        layout.addWidget(self.raw_info_widget)

        self.psd_plot_widget = PSDPlotWidget()
        self.psd_plot_widget.scrolled.connect(
            lambda direction: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 50
                if direction == "up"
                else self.scroll_area.verticalScrollBar().value() + 50
            )
        )
        layout.addWidget(self.psd_plot_widget)
        return group

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
        if self.isVisible():
            self.update_ui_state()

    def _ensure_loaded(self):
        if not self.current_dataset:
            self.preprocessor = None
            self._needs_reload = False
            return

        if not self._needs_reload and self.preprocessor is not None:
            return

        worker = Worker(
            lambda: Preprocessor(
                self.current_dataset.raw_path,
                self.current_dataset.derivative_root,
            ),
            parent=self,
            add_loggers="mne",
        )
        self.preprocessor = worker.exec_with_dialog(
            "Please wait",
            "Loading continuous data...",
        )
        self._needs_reload = False

    def update_ui_state(self):
        pipeline_name = self.main_window.current_pipeline.name if self.main_window else "Pipeline"
        if self.current_dataset:
            self.set_page_subtitle(f"{pipeline_name} pipeline · {self.current_dataset.display_name}")
            self.context_label.hide()
        else:
            self.set_page_subtitle("Select a dataset in the workspace panel to inspect continuous data.")
            self.context_label.setText("Select a dataset from the workspace panel.")
            self.context_label.setToolTip("")
            self.context_label.show()

        self._ensure_loaded()

        if self.preprocessor is not None:
            continuous_data = self.preprocessor._get_last_continuous()
            self.raw_info_widget.update_info(continuous_data, "Recording summary")
            self.psd_plot_widget.update_plot(continuous_data, self.preprocessor.label)
            self.psd_plot_widget.setVisible(True)
        else:
            self.raw_info_widget.clear_info()
            self.psd_plot_widget.update_plot(None)
            self.psd_plot_widget.setVisible(False)

    @Slot()
    def on_settings_updated(self):
        self._needs_reload = True
        if self.isVisible():
            self.update_ui_state()
