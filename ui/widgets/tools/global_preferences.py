import sys
import matplotlib.style
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QGroupBox,
    QComboBox,
    QCheckBox,
)
from PySide6.QtCore import QSettings
from typing import Any, Tuple, Union, Dict
from utils import parse_tuple, to_display_string
import logging

logger = logging.getLogger(__name__)


# --- Modular Plot Settings Widget ---
class PlotSettingsWidget(QWidget):
    """A dedicated widget for configuring plotting parameters."""

    GROUP_NAME = "plot_settings"
    SETTINGS_KEY = "plot_params"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.style_name_map = {}
        self.settings = QSettings()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.init_form(layout)

    def init_form(self, layout: QVBoxLayout):
        """Initializes the form widgets for plotting."""
        groupbox = QGroupBox("Plotting")
        form_layout = QFormLayout(groupbox)

        self.matplotlib_style_combo = QComboBox()
        self.cmap_input = QComboBox()
        self._populate_styles_combo()

        self.font_size_input = QLineEdit()
        self.evoked_xlim_input = QLineEdit()

        form_layout.addRow("Matplotlib Style:", self.matplotlib_style_combo)
        form_layout.addRow("Evoked Plot X-Limits (start, end):", self.evoked_xlim_input)
        form_layout.addRow("Default Colormap (cmap):", self.cmap_input)

        layout.addWidget(groupbox)

    def _populate_styles_combo(self):
        """
        Dynamically populates the QComboBox with available matplotlib styles,
        using user-friendly names.
        """
        # Get styles, filtering out internal ones (like '_mpl-gallery')
        available_styles = [
            s for s in matplotlib.style.available if not s.startswith("_")
        ]
        available_styles.append("default")  # Ensure default is always an option

        # Create a mapping from a user-friendly name to the actual style name
        for style_name in sorted(list(set(available_styles))):
            if style_name == "default":
                display_name = "Default"
            else:
                # e.g., 'seaborn-v0_8-darkgrid' -> 'Seaborn darkgrid'
                display_name = " ".join(
                    style_name.replace("-v0_8", "").split("-")
                ).capitalize()

            self.style_name_map[display_name] = style_name

        self.cmap_input.addItems(list(matplotlib.colormaps))

        # Add the user-friendly names to the combo box
        self.matplotlib_style_combo.addItems(self.style_name_map.keys())

    def load_settings(self):
        """Loads plot parameters from QSettings."""
        self.settings.beginGroup(self.GROUP_NAME)
        default_params = {
            "style": "default",
            "evoked_xlim": (-200, 500),
            "cmap": "turbo",
        }
        params = self.settings.value(self.SETTINGS_KEY, default_params)
        if not isinstance(params, dict):
            logger.info(
                f"Could not load '{self.SETTINGS_KEY}'; falling back to defaults."
            )
            params = default_params
        self.settings.endGroup()

        # Find the display name corresponding to the saved style name
        saved_style_name = params.get("style")
        display_name_to_set = "Default"  # Fallback
        for display, actual in self.style_name_map.items():
            if actual == saved_style_name:
                display_name_to_set = display
                break
        self.matplotlib_style_combo.setCurrentText(display_name_to_set)
        self.evoked_xlim_input.setText(to_display_string(params.get("evoked_xlim")))
        self.cmap_input.setCurrentText(params.get("cmap"))

    def save_settings(self):
        """Saves current plot settings to QSettings."""
        params = self.get_params()
        self.settings.beginGroup(self.GROUP_NAME)
        self.settings.setValue(self.SETTINGS_KEY, params)
        self.settings.endGroup()

    def clear_settings(self):
        """Removes plot settings from QSettings."""
        self.settings.beginGroup(self.GROUP_NAME)
        self.settings.remove(self.SETTINGS_KEY)
        self.settings.endGroup()
        self.settings.sync()
        logger.info(
            f"Settings key '{self.SETTINGS_KEY}' in group '{self.GROUP_NAME}' has been cleared."
        )

    def get_params(self) -> Dict[str, Any]:
        """Returns a dictionary of the current plot settings."""
        params = {}
        # Get the selected display name and map it back to the actual style name
        selected_display_name = self.matplotlib_style_combo.currentText()
        params["style"] = self.style_name_map.get(selected_display_name, "default")
        params["evoked_xlim"] = parse_tuple(self.evoked_xlim_input.text(), float)
        params["cmap"] = self.cmap_input.currentText()
        return params


# --- Main Global Settings Widget ---


class GlobalSettingsWidget(QWidget):
    """
    A reusable widget for configuring global settings, now incorporating
    the modular PlotSettingsWidget.
    """

    GROUP_NAME = "global_settings"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # self.setLayout(layout) is not needed when passed in constructor
        self.init_form(layout)
        self.load_settings()  # Load settings upon initialization

    def init_form(self, layout: QVBoxLayout):
        """Initializes and lays out the form widgets."""
        groupbox = QGroupBox("Global Settings")
        form_layout = QFormLayout(groupbox)

        # Instantiate the dedicated plot settings widget
        self.plot_settings_widget = PlotSettingsWidget()

        # --- General & Output Sub-Group ---
        general_groupbox = QGroupBox("General")
        general_layout = QFormLayout(general_groupbox)

        self.output_dir_input = QLineEdit()
        self.n_jobs_input = QLineEdit()
        self.n_jobs_input.setPlaceholderText("-1 for all cores")

        general_layout.addRow("Output Directory:", self.output_dir_input)
        general_layout.addRow("Parallel Jobs (n_jobs):", self.n_jobs_input)

        # Add sub-widgets and groups to the main layout
        form_layout.addRow(self.plot_settings_widget)
        form_layout.addRow(general_groupbox)

        layout.addWidget(groupbox)

    def load_settings(self):
        """Loads all parameters from QSettings and populates the UI."""
        # Load settings for the plot widget
        self.plot_settings_widget.load_settings()

        # Load settings for this widget's own parameters
        self.settings.beginGroup(self.GROUP_NAME)

        # Load values one by one, providing a default for each
        output_dir = self.settings.value("output_dir", "derivatives")
        n_jobs = self.settings.value("n_jobs", -1)

        self.output_dir_input.setText(output_dir)
        self.n_jobs_input.setText(str(n_jobs))

        self.settings.endGroup()
        logger.info(f"Loaded settings for group '{self.GROUP_NAME}'.")

    def save_settings(self):
        """Saves all current settings to QSettings."""
        # Save settings from the child widget first
        self.plot_settings_widget.save_settings()

        # Get only the 'global' parameters to save in this group
        try:
            params = self.get_params()["global"]
        except ValueError as e:
            logger.error(f"Validation error during save: {e}")
            # Optionally, show a message box to the user
            return

        self.settings.beginGroup(self.GROUP_NAME)
        for key, value in params.items():
            self.settings.setValue(key, value)
        self.settings.endGroup()
        logger.info(f"Saved settings for group '{self.GROUP_NAME}'.")

    def clear_settings(self):
        """Removes all settings associated with this widget's group."""
        # Clear settings for the child widget
        self.plot_settings_widget.clear_settings()

        # Correctly clear only the settings within the current group
        self.settings.beginGroup(self.GROUP_NAME)
        self.settings.remove("")  # An empty key removes the entire group
        self.settings.endGroup()

        self.settings.sync()  # Ensure changes are written to storage
        logger.info(f"Settings in group '{self.GROUP_NAME}' have been cleared.")

        # Reload the default settings into the UI
        self.load_settings()

    def get_params(self) -> Dict[str, Any]:
        """
        Returns a nested dictionary of all settings from the UI.
        Raises ValueError on validation failure.
        """
        all_params = {}
        # Get params from the dedicated plotting widget
        all_params["plotting"] = self.plot_settings_widget.get_params()

        # Get params from this widget
        global_params = {}
        global_params["output_dir"] = self.output_dir_input.text().strip()
        if not global_params["output_dir"]:
            raise ValueError("Output Directory cannot be empty.")

        try:
            global_params["n_jobs"] = int(self.n_jobs_input.text())
        except (ValueError, TypeError):
            raise ValueError("Parallel Jobs (n_jobs) must be an integer.")

        all_params["global"] = global_params
        return all_params


# --- Example Usage ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    QApplication.setOrganizationName("SSPython")
    QApplication.setApplicationName("SSPy")

    window = QWidget()
    window.setWindowTitle("Global Settings Widget Demo")
    main_layout = QVBoxLayout(window)

    settings_widget = GlobalSettingsWidget()
    main_layout.addWidget(settings_widget)

    settings = QSettings()
    settings_widget.load_settings(settings)

    window.show()

    def on_quit():
        try:
            logger.info("Attempting to save settings...")
            settings_widget.save_settings(settings)
            logger.info("Settings saved successfully.")
            # Demonstrate the nested dictionary structure
            import json

            logger.info("\nRetrieved parameters:")
            logger.info(json.dumps(settings_widget.get_params(), indent=2))
        except ValueError as e:
            logger.info(f"\nError saving settings: {e}")

    app.aboutToQuit.connect(on_quit)

    sys.exit(app.exec())
