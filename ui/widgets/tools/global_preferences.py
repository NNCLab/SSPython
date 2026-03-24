import logging
from typing import Any

import matplotlib.style
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import DEFAULT_OUTPUT_ROOT, get_settings_store
from core.pipelines import all_pipelines
from utils import parse_tuple, to_display_string

logger = logging.getLogger(__name__)


class PlotSettingsWidget(QWidget):
    SETTINGS_PATH = "appearance/plots/global"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self.style_name_map: dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._build_ui(layout)

    def _build_ui(self, layout: QVBoxLayout):
        groupbox = QGroupBox("Plotting")
        form_layout = QFormLayout(groupbox)

        self.matplotlib_style_combo = QComboBox()
        self.cmap_input = QComboBox()
        self.evoked_xlim_input = QLineEdit()

        self._populate_styles_combo()

        form_layout.addRow("Matplotlib Style:", self.matplotlib_style_combo)
        form_layout.addRow("Evoked Plot X-Limits (ms):", self.evoked_xlim_input)
        form_layout.addRow("Default Colormap:", self.cmap_input)

        layout.addWidget(groupbox)

    def _populate_styles_combo(self):
        available_styles = [style for style in matplotlib.style.available if not style.startswith("_")]
        available_styles.append("default")

        for style_name in sorted(set(available_styles)):
            display_name = "Default" if style_name == "default" else " ".join(
                style_name.replace("-v0_8", "").split("-")
            ).capitalize()
            self.style_name_map[display_name] = style_name

        self.matplotlib_style_combo.addItems(self.style_name_map.keys())
        self.cmap_input.addItems(list(matplotlib.colormaps))

    def load_settings(self):
        defaults = {"style": "default", "evoked_xlim": (-200, 500), "cmap": "turbo"}
        params = self.settings_store.get(self.SETTINGS_PATH, defaults)
        if not isinstance(params, dict):
            params = defaults

        saved_style = params.get("style", "default")
        display_style = next(
            (display for display, actual in self.style_name_map.items() if actual == saved_style),
            "Default",
        )
        self.matplotlib_style_combo.setCurrentText(display_style)
        self.evoked_xlim_input.setText(to_display_string(params.get("evoked_xlim")))
        self.cmap_input.setCurrentText(params.get("cmap", "turbo"))

    def save_settings(self):
        params = self.get_params()
        self.settings_store.set(self.SETTINGS_PATH, params)
        self.settings_store.set("plot_settings/plot_params", params)

    def clear_settings(self):
        self.settings_store.remove(self.SETTINGS_PATH)
        self.settings_store.remove("plot_settings/plot_params")
        self.settings_store.sync()

    def get_params(self) -> dict[str, Any]:
        selected_display = self.matplotlib_style_combo.currentText()
        return {
            "style": self.style_name_map.get(selected_display, "default"),
            "evoked_xlim": parse_tuple(self.evoked_xlim_input.text(), float),
            "cmap": self.cmap_input.currentText(),
        }


class GlobalSettingsWidget(QWidget):
    SETTINGS_PATH = "workspace"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._build_ui(layout)
        self.load_settings()

    def _build_ui(self, layout: QVBoxLayout):
        groupbox = QGroupBox("Application Settings")
        form_layout = QFormLayout(groupbox)

        self.plot_settings_widget = PlotSettingsWidget()

        self.pipeline_combo = QComboBox()
        for pipeline in all_pipelines():
            self.pipeline_combo.addItem(pipeline.name, pipeline.id)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])

        self.output_dir_input = QLineEdit()
        self.n_jobs_input = QLineEdit()
        self.n_jobs_input.setPlaceholderText("-1 for all cores")

        form_layout.addRow("Active Pipeline:", self.pipeline_combo)
        form_layout.addRow("Theme:", self.theme_combo)
        form_layout.addRow("Derivative Root Folder:", self.output_dir_input)
        form_layout.addRow("Parallel Jobs (n_jobs):", self.n_jobs_input)
        form_layout.addRow(self.plot_settings_widget)

        layout.addWidget(groupbox)

    def load_settings(self):
        self.plot_settings_widget.load_settings()

        output_root = self.settings_store.output_root()
        n_jobs = self.settings_store.get("workspace/n_jobs", -1)
        pipeline_id = self.settings_store.current_pipeline_id()
        theme = self.settings_store.get("appearance/theme", "dark")

        index = self.pipeline_combo.findData(pipeline_id)
        if index >= 0:
            self.pipeline_combo.setCurrentIndex(index)

        self.theme_combo.setCurrentText(theme)
        self.output_dir_input.setText(output_root)
        self.n_jobs_input.setText(str(n_jobs))

    def save_settings(self):
        self.plot_settings_widget.save_settings()

        output_root = self.output_dir_input.text().strip() or DEFAULT_OUTPUT_ROOT
        try:
            n_jobs = int(self.n_jobs_input.text())
        except (TypeError, ValueError) as exc:
            raise ValueError("Parallel Jobs (n_jobs) must be an integer.") from exc

        self.settings_store.set("workspace/current_pipeline", self.pipeline_combo.currentData())
        self.settings_store.set("appearance/theme", self.theme_combo.currentText())
        self.settings_store.set("workspace/output_root", output_root)
        self.settings_store.set("workspace/n_jobs", n_jobs)
        self.settings_store.set("global_settings/output_dir", output_root)
        self.settings_store.sync()

    def clear_settings(self):
        self.plot_settings_widget.clear_settings()
        for key in (
            "workspace/current_pipeline",
            "appearance/theme",
            "workspace/output_root",
            "workspace/n_jobs",
            "global_settings/output_dir",
        ):
            self.settings_store.remove(key)
        self.settings_store.sync()
        self.load_settings()
