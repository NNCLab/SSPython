import sys
import mne
import mne_icalabel
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QLabel,
    QDialogButtonBox,
    QGroupBox,
    QProgressDialog,
)
from utils import Worker
from PySide6.QtCore import Qt, Signal, QCoreApplication
import numpy as np

from core.app_settings import get_settings_store

# Define the labels that can be automatically excluded
EXCLUDABLE_LABELS = [
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]

# --- CHANGE 1: Added a dictionary to map labels to emojis ---
LABEL_EMOJIS = {
    "brain": "🧠",
    "muscle artifact": "💪",
    "eye blink": "👀",
    "heart beat": "❤️",
    "line noise": "⚡️",
    "channel noise": "📡",
    "other": "❓",
}


class ICALabelingSettingsWidget(QWidget):
    """Saved suggestions for which ICLabel classes to exclude."""

    GROUP_SETTINGS = "ica_label"
    SETTINGS_PATH = f"{GROUP_SETTINGS}/exclude"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self.label_checkboxes = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        group_box = QGroupBox("ICA label exclusion suggestions")
        group_layout = QVBoxLayout(group_box)
        for label in EXCLUDABLE_LABELS:
            emoji = LABEL_EMOJIS.get(label, "")
            checkbox = QCheckBox(f"{emoji} {label.replace('_', ' ').title()}".strip())
            group_layout.addWidget(checkbox)
            self.label_checkboxes[label] = checkbox
        layout.addWidget(group_box)
        self.load_settings()

    def load_settings(self):
        for label, checkbox in self.label_checkboxes.items():
            checkbox.setChecked(self.settings_store.get(
                f"{self.SETTINGS_PATH}/{label}", True, value_type=bool,
            ))

    def get_settings(self) -> dict[str, bool]:
        return {
            label: checkbox.isChecked()
            for label, checkbox in self.label_checkboxes.items()
        }

    def save_settings(self):
        for label, selected in self.get_settings().items():
            self.settings_store.set(f"{self.SETTINGS_PATH}/{label}", selected)
        self.settings_store.sync()


class ICALabelingDialog(QDialog):
    """
    A dialog to show ICLabel results and allow the user to select components for exclusion.
    """

    GROUP_SETTINGS = ICALabelingSettingsWidget.GROUP_SETTINGS

    def __init__(self, ica, epochs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ICA Component Selection")
        self.setGeometry(200, 200, 800, 600)  # Increased default width for emojis
        self.setMinimumSize(700, 500)

        self.ica = ica
        self.epochs = epochs
        self.ic_labels = None
        # --- Layout ---
        self.layout = QVBoxLayout(self)
        suggestions_label = QLabel(
            "Suggestions from Preferences. Changes apply to this run only."
        )
        suggestions_label.setWordWrap(True)
        self.layout.addWidget(suggestions_label)

        # --- Label Selection GroupBox ---
        self._create_label_selection_box()

        # --- Results Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setSortingEnabled(True)
        self.table.setHorizontalHeaderLabels(
            ["Component", "Label", "Probability", "Exclude"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.layout.addWidget(self.table)

        # --- Dialog Buttons (OK/Cancel) ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Apply Selection"
        )
        self.layout.addWidget(self.button_box)

        QApplication.processEvents()
        self.run_and_populate()

    def _create_label_selection_box(self):
        """Creates the GroupBox with checkboxes for each excludable label."""
        self.label_settings_widget = ICALabelingSettingsWidget(self)
        self.label_settings_widget.findChild(QGroupBox).setTitle(
            "Labels to exclude for this run"
        )
        self.label_checkboxes = self.label_settings_widget.label_checkboxes
        for checkbox in self.label_checkboxes.values():
            checkbox.toggled.connect(self._update_component_checkboxes)
        self.layout.addWidget(self.label_settings_widget)

    def run_and_populate(self):
        """Runs the mne_icalabel algorithm and fills the results table."""
        try:
            worker = Worker(
                lambda: mne_icalabel.label_components(
                    self.epochs, self.ica, method="iclabel"
                ),
                parent=self,
            )
            self.ic_labels = worker.exec_with_dialog(
                "ICA Auto-label",
                "Detecting ICA components...\nThis is optimally designed for infomax (extended).",
            )
            labels_to_exclude = self.get_selected_labels_to_exclude()
            labels = self.ic_labels["labels"]
            probs = self.ic_labels["y_pred_proba"]

            self.table.setRowCount(len(labels))

            for i, (label, prob) in enumerate(zip(labels, probs)):
                self.table.setItem(i, 0, QTableWidgetItem(f"ICA{i:03}"))

                # --- CHANGE 3: Add emoji to the label in the table as well ---
                emoji = LABEL_EMOJIS.get(label, "")
                table_label_text = f"{emoji} {label.replace('_', ' ').title()}".strip()
                self.table.setItem(i, 1, QTableWidgetItem(table_label_text))

                self.table.setItem(i, 2, QTableWidgetItem(f"{prob:.3f}"))

                checkbox = QCheckBox()
                checkbox.setChecked(label in labels_to_exclude)

                cell_widget = QWidget()
                cell_layout = QHBoxLayout(cell_widget)
                cell_layout.addWidget(checkbox)
                cell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                self.table.setCellWidget(i, 3, cell_widget)

        except Exception as e:
            error_label = QLabel(f"An error occurred during labeling: {e}")
            error_label.setStyleSheet("color: red;")
            self.layout.insertWidget(0, error_label)

    def get_selected_labels_to_exclude(self):
        """Returns a set of labels currently checked in the label selection box."""
        return {
            label
            for label, checkbox in self.label_checkboxes.items()
            if checkbox.isChecked()
        }

    def _update_component_checkboxes(self):
        """Updates all component checkboxes based on the current label selections."""
        if not self.ic_labels:
            return

        labels_to_exclude = self.get_selected_labels_to_exclude()
        # Use the original labels from the ica object for comparison
        component_labels = self.ic_labels["labels"]

        for i, component_label in enumerate(component_labels):
            cell_widget = self.table.cellWidget(i, 3)
            checkbox = cell_widget.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(component_label in labels_to_exclude)

    def get_excluded_indices(self):
        """
        Reads the state of the component checkboxes and returns a list of indices to exclude,
        correctly handling sorting.
        """
        exclude_idx = []
        for i in range(self.table.rowCount()):
            cell_widget = self.table.cellWidget(i, 3)
            checkbox = cell_widget.findChild(QCheckBox)

            if checkbox and checkbox.isChecked():
                component_item = self.table.item(i, 0)
                if component_item:
                    component_name = component_item.text()  # e.g., "ICA005"
                    component_index = int(component_name[3:])
                    exclude_idx.append(component_index)
        return exclude_idx


class ICALabelingWidget(QWidget):
    """
    A QWidget that provides a button to launch a dialog for selecting ICA components.
    """

    finished_label = Signal(mne.preprocessing.ICA)

    def __init__(self, ica, epochs, parent=None):
        super().__init__(parent)
        self.ica = ica
        self.epochs = epochs
        if not self.epochs.preload:
            self.epochs.load_data()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label_button = QPushButton("Run Auto-label")
        self.label_button.clicked.connect(self.open_labeling_dialog)
        layout.addWidget(self.label_button)

    def open_labeling_dialog(self):
        """Creates and shows the component selection dialog."""
        dialog = ICALabelingDialog(self.ica, self.epochs, self)

        if dialog.exec():
            exclude_indices = dialog.get_excluded_indices()
            self.ica.exclude = exclude_indices
            self.finished_label.emit(self.ica)
        else:
            print("Dialog cancelled.")


# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Set Organization and Application name for QSettings to work
    QCoreApplication.setOrganizationName("SSPython")
    QCoreApplication.setApplicationName("SSPy")

    # 1. Create dummy MNE data for demonstration
    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:32]
    sfreq = 200
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)
    n_times = 1000
    raw = mne.io.RawArray(np.random.randn(len(ch_names), n_times), info)

    events = mne.make_fixed_length_events(raw, duration=1)
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, preload=True)
    epochs.filter(1.0, 40.0, verbose=False)

    # 2. Fit ICA on the dummy data
    print("Fitting ICA on dummy data...")
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto")
    ica.fit(epochs)
    print("ICA fitting complete.")

    # 3. Setup the Qt Application
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle("ICA Widget Demo")

    central_widget = QWidget()
    main_layout = QVBoxLayout(central_widget)

    info_label = QLabel(
        "This is a demo application. The widget below is self-contained."
    )
    main_layout.addWidget(info_label)

    # 4. Instantiate the widget with the fitted ICA and epochs
    ica_widget = ICALabelingWidget(ica=ica, epochs=epochs)
    main_layout.addWidget(ica_widget)
    main_layout.addStretch()

    # 5. Connect to the widget's Signal
    def handle_exclusion_selection(ica):
        print("\n--- Main Application received Signal ---")
        print(f"The user selected these components to exclude: {ica.exclude}")
        print("Labels for all components:")

    ica_widget.finished_label.connect(handle_exclusion_selection)

    main_window.setCentralWidget(central_widget)
    main_window.setGeometry(100, 100, 400, 150)
    main_window.show()

    sys.exit(app.exec())
