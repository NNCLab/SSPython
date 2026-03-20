import sys
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt, QTimer, QSettings, Signal
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import mne
from utils import update_toolbar_color  # Assuming utils.py is in the same directory
import logging

logger = logging.getLogger(__name__)


class MplCanvas(FigureCanvas):
    """A custom Matplotlib canvas widget that integrates with PySide6."""

    def __init__(self, parent=None, figsize=None, dpi=100):
        fig = Figure(figsize=figsize, dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)


class PSDPlotWidget(QWidget):
    """
    A widget for displaying MNE Power Spectral Density (PSD) plots with interactive features.
    """

    scrolled = Signal(str)

    def __init__(self, figsize=None, dpi=100, params={}, progress_dialog=None):
        super().__init__()

        self.setWindowTitle("Power Spectral Density (PSD) Plot")
        self.progress_dialog = progress_dialog

        # --- UI Setup ---
        self.canvas = MplCanvas(self, figsize=figsize, dpi=dpi)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.title_label = QLabel("Waiting for data...")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setMaximumHeight(30)

        layout = QVBoxLayout(self)
        self.setMinimumSize(400, 500)
        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

        # --- Instance Variables ---
        self.params = params
        self.spectrum = None  # Will store the MNE Spectrum object
        self.ch_names = []
        self.drag_start_coords = None
        self.drag_span = None
        self.picked_artists_in_click = []
        self.original_zorders = {}

        # Throttling for smooth dragging
        self.throttle_interval = 0.04  # (0.04s ≈ 25 FPS)
        self.last_update_time = 0

        self.default_params = {
            "fmin": 0,
            "fmax": 50,
            "average": False,
            "spatial_colors": True,
            "cmap": "turbo",
        }

        self.update_plot()

    def update_plot(self, mne_obj: mne.io.Raw | mne.Epochs | None = None, label=None):
        """
        Clears the axes and plots the PSD for the given MNE object (Raw or Epochs).
        """
        # --- Clear and Style ---
        self.reapply_style()
        self.canvas.axes.cla()
        self.params = QSettings().value(
            "plot_settings/psd_plot_params", self.default_params
        )
        logger.info(self.params)
        update_toolbar_color(self.toolbar)

        if mne_obj is not None:
            self.toolbar.show()
            self.canvas.show()
            self.canvas.axes.grid(True)

            # --- Compute and Plot PSD ---
            self.spectrum = mne_obj.compute_psd(
                fmin=self.params.get("fmin", 0), fmax=self.params.get("fmax", 50)
            )
            self.spectrum.plot(
                average=False,
                spatial_colors=self.params.get("spatial_colors", True),
                axes=self.canvas.axes,
                show=False,
            )

            # --- Final Touches ---
            self.canvas.axes.set_xlabel("Frequency (Hz)")
            self.canvas.axes.set_ylabel(r"PSD (dB/Hz)")
            self.canvas.axes.set_title("")  # Remove MNE's default title
            self.title_label.setText(label or "Power Spectral Density")
            self.canvas.draw()
        else:
            self.title_label.setText("No Data Available")
            self.toolbar.hide()
            self.canvas.hide()
            self.canvas.draw()

    # === Utility Methods ===
    def closeEvent(self, event):
        """Overrides close event to also close any associated dialogs."""
        if self.progress_dialog:
            self.progress_dialog.close()
        event.accept()
        super().closeEvent(event)

    def reapply_style(self):
        """Re-applies the current matplotlib style to the canvas for dynamic theming."""
        logger.info("Re-applying plot style...")
        fig = self.canvas.figure
        ax = self.canvas.axes

        fig.set_facecolor(plt.rcParams["figure.facecolor"])
        ax.set_facecolor(plt.rcParams["axes.facecolor"])

        for spine in ax.spines.values():
            spine.set_color(plt.rcParams["axes.edgecolor"])

        ax.tick_params(axis="x", colors=plt.rcParams["xtick.color"])
        ax.tick_params(axis="y", colors=plt.rcParams["ytick.color"])

        ax.xaxis.label.set_color(plt.rcParams["text.color"])
        ax.yaxis.label.set_color(plt.rcParams["text.color"])

        update_toolbar_color(self.toolbar)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:64]
    sfreq = 500
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create 10 seconds of random data
    n_samples = 10 * sfreq
    data = np.random.randn(len(ch_names), n_samples) * 1e-6

    # Add a 10 Hz sine wave to a few channels
    time_vec = np.arange(n_samples) / sfreq
    for i in [5, 10, 15]:
        data[i, :] += 2 * 1e-6 * np.sin(2 * np.pi * 10 * time_vec)

    raw_data = mne.io.RawArray(data, info)
    raw_data.set_montage(montage)

    # --- Create and show the widget ---
    main_widget = PSDPlotWidget()
    main_widget.update_plot(raw_data, label="Random Raw Data PSD (with 10 Hz peak)")
    main_widget.show()

    sys.exit(app.exec())
