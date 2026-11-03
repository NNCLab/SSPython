import logging

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QBoxLayout,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from utils import apply_theme

from .base_page import BasePage
from ..widgets.auto_label import ICALabelingSettingsWidget
from ..widgets.ica_widget import ICAPlottingSettingsWidget
from ..widgets.preprocessing_widgets import (
    ArtifactRemovalSettingsWidget,
    FilterContinuousWidget,
    EpochingSettingsWidget,
    PreprocessingSettingsWidget,
)
from ..widgets.psd_plot import PSDPlotSettingsWidget
from ..widgets.real_time_widget import ConnectionWidget, RealTimeSettingsWidget
from ..widgets.run_ica_widget import ICASettingsWidget
from ..widgets.tools.global_preferences import GlobalSettingsWidget

logger = logging.getLogger(__name__)


class PreferencesPage(BasePage):
    settings_saved = Signal()

    def __init__(self, settings_store, parent: QWidget | None = None):
        super().__init__("Preferences", parent)
        self.set_content_maximum_width(1120)
        self.header_frame.setObjectName("flatPageHeader")
        self.header_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(24)
        self.settings_store = settings_store
        self.main_window = parent
        self.settings_widgets: list[QWidget] = []

        self._build_ui()
        self._populate_categories()
        self._connect_signals()
        self._sync_pipeline_summary()

        if self.main_window:
            self.main_window.current_pipeline_changed.connect(self.on_pipeline_changed)

    def _build_ui(self):
        self.summary_label = self.subtitle_label
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        self.summary_label.show()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("preferencesCategories")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.tabs.tabBar().setObjectName("preferencesTabBar")
        self.tabs.tabBar().setAccessibleName("Preference categories")
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.add_content(self.tabs)
        self.content_layout.setStretch(self.content_layout.indexOf(self.tabs), 1)

        footer = QFrame()
        footer.setObjectName("preferencesFooter")
        self.button_row = QHBoxLayout(footer)
        self.button_row.setContentsMargins(24, 12, 24, 12)
        self.button_row.setSpacing(8)
        self.clear_all_button = QPushButton("Reset Settings")
        self.clear_all_button.setObjectName("dangerButton")
        self.revert_button = QPushButton("Reload Saved")
        self.revert_button.setObjectName("secondaryButton")
        self.save_button = QPushButton("Save Changes")
        self.save_button.setDefault(True)
        self.save_button.setMinimumWidth(136)

        self.button_row.addWidget(self.clear_all_button)
        self.button_row.addStretch()
        self.button_row.addWidget(self.revert_button)
        self.button_row.addWidget(self.save_button)
        self.set_footer(footer)

    def _populate_categories(self):
        self.global_settings_widget = GlobalSettingsWidget()
        self.global_settings_widget.pipeline_combo.currentIndexChanged.connect(self._sync_pipeline_preview)
        self.psd_plot_widget = PSDPlotSettingsWidget()

        self.artifact_widget = ArtifactRemovalSettingsWidget()
        self.continuous_filter_widget = FilterContinuousWidget()
        self.epoching_widget = EpochingSettingsWidget()
        self.apply_widget = PreprocessingSettingsWidget()
        self.ica_continuous_widget = ICASettingsWidget("continuous")
        self.ica_epochs_widget = ICASettingsWidget("epochs")
        self.ica_plotting_widget = ICAPlottingSettingsWidget()
        self.ica_labeling_widget = ICALabelingSettingsWidget()
        self.ica_continuous_widget.findChild(QGroupBox).setTitle("Continuous ICA")
        self.ica_epochs_widget.findChild(QGroupBox).setTitle("Epoch ICA")
        self.ica_plotting_widget.findChild(QGroupBox).setTitle("ICA plotting")
        self.connection_widget = ConnectionWidget()
        self.real_time_plot_widget = RealTimeSettingsWidget()

        self.add_settings_category("General", self.global_settings_widget, self.psd_plot_widget)
        self.add_settings_category(
            "Preprocessing",
            self.artifact_widget,
            self.continuous_filter_widget,
            self.epoching_widget,
            self.apply_widget,
        )
        self.add_settings_category(
            "ICA",
            self.ica_continuous_widget,
            self.ica_epochs_widget,
            self.ica_plotting_widget,
            self.ica_labeling_widget,
        )
        self.add_settings_category(
            "Real-time",
            self.connection_widget,
            self.real_time_plot_widget,
        )

        self.load_all_settings()

    def _connect_signals(self):
        self.save_button.clicked.connect(self.save_all_settings)
        self.revert_button.clicked.connect(self.load_all_settings)
        self.clear_all_button.clicked.connect(self.clear_all_settings)

    def add_settings_category(self, title: str, *widgets: QWidget):
        container = QWidget()
        container.setObjectName("flatSection")
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 16, 0, 24)
        layout.setSpacing(24)
        for widget in widgets:
            self._prepare_settings_widget(widget)
            layout.addWidget(widget)
            self.settings_widgets.append(widget)
        layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("preferencesTabScroll")
        scroll_area.viewport().setObjectName("preferencesTabViewport")
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(container)
        self.tabs.addTab(scroll_area, title)

    @staticmethod
    def _prepare_settings_widget(widget: QWidget):
        """Flatten embedded forms while retaining their shared dialog presentation."""
        widget.setObjectName("flatSection")
        policy = widget.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
        widget.setSizePolicy(policy)
        widget.layout().setContentsMargins(0, 0, 0, 0)
        for group in widget.findChildren(QGroupBox):
            if group.layout() is not None:
                group.layout().setContentsMargins(0, 12, 0, 0)
        for form in widget.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setHorizontalSpacing(20)
            form.setVerticalSpacing(10)
            # Wrapped forms need room to grow beyond their unwrapped size hint.
            host = form.parentWidget()
            policy = host.sizePolicy()
            policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
            host.setSizePolicy(policy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "button_row"):
            compact = event.size().width() < 560
            self.button_row.setDirection(
                QBoxLayout.Direction.TopToBottom
                if compact
                else QBoxLayout.Direction.LeftToRight
            )
            wrap_policy = (
                QFormLayout.RowWrapPolicy.WrapAllRows
                if compact
                else QFormLayout.RowWrapPolicy.WrapLongRows
            )
            for form in self.tabs.findChildren(QFormLayout):
                if form.rowWrapPolicy() != wrap_policy:
                    form.setRowWrapPolicy(wrap_policy)

    def _active_pipeline_id(self) -> str:
        return self.global_settings_widget.pipeline_combo.currentData()

    def _sync_pipeline_preview(self):
        pipeline_id = self._active_pipeline_id()
        for widget in self.settings_widgets:
            if hasattr(widget, "set_pipeline"):
                widget.set_pipeline(pipeline_id)
        self._sync_pipeline_summary()

    def _sync_pipeline_summary(self):
        pipeline_name = self.global_settings_widget.pipeline_combo.currentText()
        output_root = self.global_settings_widget.output_dir_input.text().strip() or "derivatives"
        self.summary_label.setText(
            f"Defaults for the {pipeline_name} pipeline · Output folder: {output_root}"
        )

    def load_all_settings(self):
        for widget in self.settings_widgets:
            if hasattr(widget, "load_settings"):
                widget.load_settings()
        self._sync_pipeline_preview()

    def save_all_settings(self):
        try:
            # Validate every form before any form writes its suggestions.
            for widget in self.settings_widgets:
                if hasattr(widget, "validate_settings"):
                    widget.validate_settings()
                elif hasattr(widget, "get_params"):
                    widget.get_params()
                elif hasattr(widget, "get_settings"):
                    widget.get_settings()
            for widget in self.settings_widgets:
                if hasattr(widget, "save_settings"):
                    widget.save_settings()
            apply_theme()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid Settings", str(exc))
            return

        self.settings_saved.emit()
        QMessageBox.information(self, "Saved", "Preferences were saved successfully.")

    def clear_all_settings(self):
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all saved application settings to their defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.settings_store.clear()
        if self.main_window:
            self.main_window.set_current_folder(None)

        self.load_all_settings()
        apply_theme()
        self.settings_saved.emit()
        QMessageBox.information(self, "Reset Complete", "Preferences were reset to defaults.")

    def on_pipeline_changed(self, pipeline):
        index = self.global_settings_widget.pipeline_combo.findData(pipeline.id)
        if index >= 0:
            self.global_settings_widget.pipeline_combo.blockSignals(True)
            self.global_settings_widget.pipeline_combo.setCurrentIndex(index)
            self.global_settings_widget.pipeline_combo.blockSignals(False)
        self._sync_pipeline_preview()
