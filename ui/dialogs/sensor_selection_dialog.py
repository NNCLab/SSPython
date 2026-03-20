import sys
import mne
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QListWidget,
    QPushButton,
    QLabel,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, Slot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils import use_plot_style


class SensorSelectionDialog(QDialog):
    """
    A PySide6 Dialog for interactively selecting MNE sensor channels.

    This dialog displays a topomap of sensor locations and allows the user
    to select channels using a lasso tool. The names of the selected channels
    are displayed in a list widget.
    """

    def __init__(self, epochs, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.selected_channels = []

        self.setWindowTitle("Interactive Sensor Selector")
        self.setMinimumSize(600, 700)

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Matplotlib Figure and Canvas ---
        # We create a Figure instance to host our plot.
        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # --- Dialog Buttons ---
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # --- Plotting and Lasso Setup ---
        self.setup_sensor_plot()
        self.setup_lasso()

    @use_plot_style
    def setup_sensor_plot(self):
        """
        Creates the MNE sensor plot on the matplotlib canvas.
        """
        # Use the existing MNE plotting function, but target our figure's axes.
        # We add a single axes to our figure.
        self.ax = self.fig.add_subplot(111)
        self.epochs.plot_sensors(
            kind="topomap", axes=self.ax, show=False, show_names=True
        )
        self.fig.tight_layout(pad=2.0)  # Add padding for title

        # Get sensor positions and channel names for lasso logic
        self.collection = self.ax.collections[0]
        self.sensor_pos = self.collection.get_offsets()
        self.ch_names = self.epochs.ch_names

        self.canvas.draw()

    def setup_lasso(self):
        """
        Initializes the Matplotlib LassoSelector widget.
        """
        # The lasso instance needs to be a member of the class to prevent
        # it from being garbage collected.
        self.lasso = LassoSelector(self.ax, self.on_lasso_select, button=1)

    @Slot(list)
    def on_lasso_select(self, verts):
        """
        Callback function for the LassoSelector.
        Updates the list widget and highlights selected sensors.
        """
        p = Path(verts)
        indices = np.nonzero(p.contains_points(self.sensor_pos))[0]
        self.selected_channels = []

        if len(indices) > 0:
            self.selected_channels = [self.ch_names[i] for i in indices]

        # Highlight the selected sensors
        facecolors = self.collection.get_facecolors()
        facecolors[:, :] = (0, 0, 0, 1)  # Reset all to black
        for i in indices:
            facecolors[i] = (1, 0, 0, 1)  # Highlight selected in red

        # Redraw the canvas to show the changes
        self.canvas.draw_idle()

    def get_selected_channels(self):
        """
        Returns the list of final selected channel names.
        """
        return self.selected_channels


def main():
    """
    Main function to load data and run the application.
    """
    # --- 1. Load Sample Data (same as original script) ---
    print("Loading MNE sample data...")
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (
        sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
    raw.pick_types(meg="mag", eeg=False, stim=False, eog=False, exclude="bads")
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, preload=True, verbose=False)
    print("Data loaded.")

    # --- 2. Initialize and run the PySide6 Application ---
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = SensorSelectionDialog(epochs)

    # Show the dialog and check if the user clicked "OK"
    if dialog.exec():
        final_selection = dialog.get_selected_channels()
        print("\n--- Dialog Accepted ---")
        if final_selection:
            print(f"Final selected channels ({len(final_selection)}):")
            print(", ".join(final_selection))
        else:
            print("No channels were selected.")
    else:
        print("\n--- Dialog Canceled ---")

    # The sys.exit(app.exec()) is not needed here as dialog.exec() runs the event loop.


if __name__ == "__main__":
    main()
