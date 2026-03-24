from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QLabel,
    QVBoxLayout,
    QLineEdit,
    QGroupBox,
    QMessageBox,
    QComboBox,
    QRadioButton,
    QApplication,
)
from PySide6.QtCore import Qt
from core.processing import Preprocessor
from utils import Worker, to_display_string, parse_numeric
from typing import Literal
import sys
from pathlib import Path
import logging

from core.app_settings import get_settings_store

logger = logging.getLogger(__name__)


class ICASettingsWidget(QWidget):
    """
    A reusable widget for configuring ICA parameters.
    It manages its own UI elements and can load/save its state
    to a QSettings object.
    """

    def __init__(
        self,
        mode: Literal["continuous", "epochs"],
        max_components: int | None = None,
        ch_names: list[str] | None = None,
        parent=None,
        pipeline_id: str | None = None,
    ):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self.mode = mode
        self.SETTINGS_KEY = mode
        self.pipeline_id = pipeline_id or self.settings_store.current_pipeline_id()
        self.settings_path = f"pipelines/{self.pipeline_id}/ica/{mode}"

        self.max_components = max_components
        self.ica_method = [
            "fastica",
            "infomax",
            "picard",
            "infomax (extended)",
            "picard (extended)",
        ]
        self.ch_names = ch_names if ch_names else []
        self._selected_ref_channels = []

        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.init_form(layout)
        self.load_settings()

    def init_form(self, layout):
        """Initializes and lays out the form widgets."""
        group = QGroupBox("ICA Parameters")
        layout.addWidget(group)

        form_layout = QFormLayout()

        self.ica_method_input = QComboBox()
        self.ica_method_input.addItems(self.ica_method)
        self.auto_n_components_checkbox = QCheckBox("Auto Detect Number of Components")
        self.n_components_input = QSpinBox()
        self.random_state_input = QLineEdit()
        self.random_state_input.setPlaceholderText("Integer or 'None'")
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(100, 10000)
        self.max_iter_input.setSingleStep(50)

        if self.max_components is not None:
            self.n_components_input.setRange(1, self.max_components)
        else:
            self.n_components_input.setVisible(False)

        self.auto_n_components_checkbox.toggled.connect(
            self.n_components_input.setDisabled
        )

        form_layout.addRow(self.auto_n_components_checkbox)
        if self.max_components is not None:
            form_layout.addRow(QLabel("Number of Components:"), self.n_components_input)
        form_layout.addRow(QLabel("Random State:"), self.random_state_input)
        form_layout.addRow(QLabel("Max Iterations:"), self.max_iter_input)
        form_layout.addRow(QLabel("Method:"), self.ica_method_input)
        group.setLayout(form_layout)
        layout.addStretch()

    def load_settings(self):
        """Loads ICA parameters from a QSettings object into the UI."""
        defaults = {
            "auto_n_components": True,
            "random_state": None,
            "max_iter": 1000,
            "method": "fastica",
        }
        saved_params = self.settings_store.get(
            self.settings_path,
            {},
            legacy_keys=(f"ica/{self.mode}",),
        ) or {}
        params = defaults.copy()
        params.update(saved_params)

        self.auto_n_components_checkbox.setChecked(params.get("auto_n_components"))
        if self.max_components is not None:
            self.n_components_input.setValue(self.max_components)
        self.random_state_input.setText(to_display_string(params.get("random_state")))
        self.max_iter_input.setValue(params.get("max_iter"))
        self.ica_method_input.setCurrentText(params.get("method"))

    def save_settings(self):
        """Saves the current UI state to a QSettings object."""
        params = self.get_params()
        self.settings_store.set(self.settings_path, params)
        self.settings_store.set(f"ica/{self.SETTINGS_KEY}", params)
        self.settings_store.sync()

    def set_pipeline(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.settings_path = f"pipelines/{self.pipeline_id}/ica/{self.mode}"
        self.load_settings()

    def get_params(self) -> dict:
        """Parses UI controls to get ICA parameters."""

        if self.auto_n_components_checkbox.isChecked():
            n_components = None
        else:
            n_components = self.n_components_input.value()

        params = {
            "n_components": n_components,
            "random_state": parse_numeric(self.random_state_input.text(), int),
            "max_iter": self.max_iter_input.value(),
            "method": self.ica_method_input.currentText(),
        }

        return params

    def clear_settings(self):
        """Removes all settings associated with this widget group."""
        self.settings_store.remove(self.settings_path)
        self.settings_store.remove(f"ica/{self.SETTINGS_KEY}")
        self.settings_store.sync()


class RunICADialog(QDialog):
    """
    A dialog for running the ICA process. It uses ICASettingsWidget for the UI
    and adds dialog-specific controls and the logic to execute the task.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        mode: Literal["continuous", "epochs"],
        parent=None,
    ):
        super().__init__(parent)
        self.preprocessor = preprocessor
        self.mode = mode

        self.setWindowTitle("Independent Component Analysis")
        self.setModal(True)
        self.resize(560, 360)
        self.setObjectName("appDialog")

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Create and add the settings widget
        if mode == "continuous":
            raw = self.preprocessor._get_last_continuous()
            self.max_components = len(raw.info["ch_names"]) - len(raw.info["bads"])
            if self.preprocessor.raw.proj:
                self.max_components -= 1
            ch_names = list(self.preprocessor.raw.ch_names)
        elif mode == "epochs":
            self.max_components = len(self.preprocessor.epochs.info["ch_names"]) - len(
                self.preprocessor.epochs.info["bads"]
            )
            if self.preprocessor.epochs.proj:
                self.max_components -= 1
            ch_names = list(self.preprocessor.epochs.ch_names)
        else:
            raise ValueError("'mode' needs to be 'continuous' or 'epochs'.")

        self.settings_widget = ICASettingsWidget(
            mode,
            self.max_components,
            ch_names,
            parent=self,
            pipeline_id=get_settings_store().current_pipeline_id(),
        )
        self.settings_widget.load_settings()
        layout.addWidget(self.settings_widget)

        # Add dialog-specific buttons
        hbox = QHBoxLayout()
        self.apply_button = QPushButton("Run")
        self.apply_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        self.reset_button = QPushButton("Reset Defaults")

        hbox.addStretch()
        hbox.addWidget(self.reset_button)
        hbox.addWidget(self.cancel_button)
        hbox.addWidget(self.apply_button)
        layout.addLayout(hbox)

        # Connect signals
        self.apply_button.clicked.connect(self.run_ica)
        self.cancel_button.clicked.connect(self.reject)
        self.reset_button.clicked.connect(
            self.settings_widget.load_settings
        )

    def run_ica(self):
        ica_params = self.settings_widget.get_params()
        self.settings_widget.save_settings()

        if self.preprocessor.has(self.mode + "-ica"):
            reply = QMessageBox.question(
                self,
                "Overwrite ICA?",
                f"{self.mode.split('-')[0].capitalize()} ICA data already exists. Do you want to overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        logger.info(f"Running {self.mode} ICA with parameters: {ica_params}")
        if self.mode == "continuous":
            worker = Worker(
                lambda: self.preprocessor.run_continuous_ica(**ica_params),
                parent=self.parent(),
                add_loggers="mne",
            )
        elif self.mode == "epochs":
            worker = Worker(
                lambda: self.preprocessor.run_epochs_ica(**ica_params),
                parent=self.parent(),
                add_loggers="mne",
            )
        worker.exec_with_dialog("Processing", "Running ICA...")
        QMessageBox.information(self, "Done", "Finished running ICA.")
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

    # The dialog now requires the settings object
    dialog = RunICADialog(preprocessor, main_settings)
    dialog.finished.connect(app.quit)
    dialog.show()

    sys.exit(app.exec())
