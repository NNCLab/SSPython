import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QListWidget,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from utils import apply_theme

from .base_page import BasePage
from ..widgets.ica_widget import ICAPlottingSettingsWidget
from ..widgets.preprocessing_widgets import (
    ArtifactRemovalSettingsWidget,
    FilterContinuousWidget,
    EpochingSettingsWidget,
    PreprocessingSettingsWidget,
)
from ..widgets.real_time_widget import ConnectionWidget, RealTimeSettingsWidget
from ..widgets.run_ica_widget import ICASettingsWidget
from ..widgets.tools.global_preferences import GlobalSettingsWidget

logger = logging.getLogger(__name__)


class PreferencesPage(BasePage):
    settings_saved = Signal()

    def __init__(self, settings_store, parent: QWidget | None = None):
        super().__init__("Preferences", parent)
        self.set_content_maximum_width(1120)
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
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("mutedLabel")
        self.summary_label.setWordWrap(True)
        self.add_content(self.summary_label)
        self.settings_panel = QWidget()
        self.category_layout = QHBoxLayout(self.settings_panel)
        self.category_layout.setContentsMargins(0, 0, 0, 0)
        self.category_layout.setSpacing(20)
        self.category_list = QListWidget()
        self.category_list.setObjectName("preferencesCategories")
        self.category_list.setFixedWidth(200)
        self.category_stack = QStackedWidget()
        self.category_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.category_layout.addWidget(self.category_list)
        self.category_layout.addWidget(self.category_stack, 1)
        self.category_list.currentRowChanged.connect(self.category_stack.setCurrentIndex)
        self.add_content(self.settings_panel)
        self.content_layout.setStretch(self.content_layout.indexOf(self.settings_panel), 1)
        self.content_layout.setStretch(self.content_layout.count() - 1, 0)

        footer = QFrame()
        footer.setObjectName("pageFooter")
        button_row = QHBoxLayout(footer)
        button_row.setContentsMargins(24, 12, 24, 12)
        self.clear_all_button = QPushButton("Reset Settings")
        self.clear_all_button.setObjectName("dangerButton")
        self.revert_button = QPushButton("Reload Saved")
        self.revert_button.setObjectName("secondaryButton")
        self.save_button = QPushButton("Save Changes")
        self.save_button.setDefault(True)

        button_row.addWidget(self.clear_all_button)
        button_row.addStretch()
        button_row.addWidget(self.revert_button)
        button_row.addWidget(self.save_button)
        self.set_footer(footer)

    def _populate_categories(self):
        self.global_settings_widget = GlobalSettingsWidget()
        self.global_settings_widget.pipeline_combo.currentIndexChanged.connect(self._sync_pipeline_preview)

        self.artifact_widget = ArtifactRemovalSettingsWidget()
        self.continuous_filter_widget = FilterContinuousWidget()
        self.epoching_widget = EpochingSettingsWidget()
        self.apply_widget = PreprocessingSettingsWidget()
        self.ica_continuous_widget = ICASettingsWidget("continuous")
        self.ica_epochs_widget = ICASettingsWidget("epochs")
        self.ica_plotting_widget = ICAPlottingSettingsWidget()
        self.connection_widget = ConnectionWidget()
        self.real_time_plot_widget = RealTimeSettingsWidget()

        self.add_settings_category("Workspace and Appearance", self.global_settings_widget)
        self.add_settings_category(
            "Preprocessing Defaults",
            self.artifact_widget,
            self.continuous_filter_widget,
            self.epoching_widget,
            self.apply_widget,
        )
        self.add_settings_category(
            "ICA Defaults",
            self.ica_continuous_widget,
            self.ica_epochs_widget,
            self.ica_plotting_widget,
        )
        self.add_settings_category(
            "Real-Time",
            self.connection_widget,
            self.real_time_plot_widget,
        )

        self.load_all_settings()
        self.category_list.setCurrentRow(0)

    def _connect_signals(self):
        self.save_button.clicked.connect(self.save_all_settings)
        self.revert_button.clicked.connect(self.load_all_settings)
        self.clear_all_button.clicked.connect(self.clear_all_settings)

    def add_settings_category(self, title: str, *widgets: QWidget):
        container = QFrame()
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        for widget in widgets:
            layout.addWidget(widget)
            self.settings_widgets.append(widget)
        layout.addStretch(1)
        for form in container.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.category_stack.addWidget(scroll)
        self.category_list.addItem(title)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "category_layout"):
            compact = event.size().width() < 720
            self.category_layout.setDirection(QHBoxLayout.Direction.TopToBottom if compact else QHBoxLayout.Direction.LeftToRight)
            if compact:
                self.category_list.setMinimumWidth(0)
                self.category_list.setMaximumWidth(16777215)
            else:
                self.category_list.setFixedWidth(200)
            self.category_list.setMaximumHeight(164 if compact else 16777215)

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
            f"Editing defaults for the {pipeline_name} pipeline. "
            f"Derivatives are saved in the workspace’s {output_root} folder."
        )

    def load_all_settings(self):
        for widget in self.settings_widgets:
            if hasattr(widget, "load_settings"):
                widget.load_settings()
        self._sync_pipeline_preview()

    def save_all_settings(self):
        try:
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
