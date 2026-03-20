import sys
import uuid
import time
import numpy as np
import mne
import mne_lsl
from scipy.spatial.distance import pdist

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QDialog,
    QDockWidget,
    QToolBar,
    QLineEdit,
    QFormLayout,
    QMessageBox,
    QHBoxLayout,
    QDialogButtonBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QProgressDialog,
    QGroupBox,
    QFileDialog,
    QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, QSettings
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from .tools.optional_range_widget import OptionalRangeWidget


class ConnectionManager:
    """Handles the connection to an LSL stream."""

    def __init__(self, params):
        self.params = params
        self.stream = None
        self.epochs = None

    def connect_to_stream(self):
        """Establishes a connection to the LSL stream."""
        try:
            stream_duration = self.params.get("stream_duration")
            stream_name = self.params.get("stream_name")
            tmin, tmax = self.params.get("tlim")
            montage_path = self.params.get("montage_path")
            event_id_str = self.params.get("event_id")

            event_id = None
            if event_id_str:
                try:
                    # Attempt to convert the event ID string to an integer
                    event_id = int(event_id_str)
                    print(f"Using Event ID: {event_id}")
                except ValueError:
                    print(
                        f"Warning: Invalid Event ID '{event_id_str}'. Must be an integer. Capturing all events instead."
                    )
            else:
                print("No Event ID provided. Capturing all events.")

            self.stream = mne_lsl.stream.StreamLSL(stream_duration, name=stream_name)
            self.stream.connect(
                acquisition_delay=0.1, processing_flags="all", timeout=5
            )

            if montage_path:
                try:
                    if montage_path.endswith(".fif"):
                        montage = mne.channels.read_dig_fif(montage_path)
                    else:
                        montage = mne.channels.read_custom_montage(montage_path)
                    self.stream.set_montage(montage)
                    print(f"Successfully loaded montage from {montage_path}")
                except Exception as e:
                    print(f"Could not load montage file: {e}")

            self.epochs = mne_lsl.stream.EpochsStream(
                self.stream,
                bufsize=10,
                event_id=event_id,
                event_channels=self.params.get("event_channel"),
                tmin=tmin,
                tmax=tmax,
                baseline=(None, 0),
                picks="eeg",
            )
            self.epochs.connect(acquisition_delay=0.1)
            print("Connection successful.")
            return self.stream, self.epochs
        except Exception as e:
            print(f"Failed to connect to stream: {e}")
            return None, None


class ConnectionWorker(QObject):
    """Worker thread for handling the LSL stream connection."""

    finished = Signal(object, object)  # Emits stream and epochs on success
    error = Signal(str)  # Emits error message on failure

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Tries to connect to the stream."""
        try:
            conn_manager = ConnectionManager(self.params)
            stream, epochs = conn_manager.connect_to_stream()
            if stream and epochs:
                self.finished.emit(stream, epochs)
            else:
                self.error.emit(
                    "Failed to connect to the LSL stream. Please check the stream name and ensure it is available."
                )
        except Exception as e:
            self.error.emit(f"An error occurred during connection: {e}")


class ConnectionWidget(QWidget):
    """A widget to get stream connection parameters from the user."""

    GROUP_NAME = "real_time"
    SETTINGS_KEY = "connection"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Initializes the user interface for the connection settings."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        group_box = QGroupBox("Connection Settings")
        main_layout.addWidget(group_box)

        form_layout = QFormLayout(group_box)

        self.stream_name_input = QLineEdit()
        self.stream_duration_input = QSpinBox()
        self.stream_duration_input.setRange(1, 60 * 5)
        self.tlim_input = OptionalRangeWidget(
            ("Start:", "End:"),
            range=(None, None),
            suffix=" ms",
            required=(True, True),
            scale=1e-3,
            parent=self,
        )
        self.event_channel_input = QLineEdit()
        self.event_id_input = QLineEdit()

        form_layout.addRow(QLabel("Stream Name:"), self.stream_name_input)
        form_layout.addRow(QLabel("Stream Duration (s):"), self.stream_duration_input)
        form_layout.addRow(QLabel("Epoching (t_start, t_end):"), self.tlim_input)
        form_layout.addRow(QLabel("Event Channel:"), self.event_channel_input)
        form_layout.addRow(QLabel("Event ID:"), self.event_id_input)

        self.montage_path_input = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_for_montage)

        montage_layout = QHBoxLayout()
        montage_layout.addWidget(self.montage_path_input)
        montage_layout.addWidget(browse_button)
        form_layout.addRow(QLabel("Montage File:"), montage_layout)

    def _browse_for_montage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Montage File",
            "",
            "Montage Files (*.loc *.locs *.elc *.sfp *.hpts *.txt *-dig.fif)",
        )
        if file_path:
            self.montage_path_input.setText(file_path)

    def get_settings(self) -> dict:
        """Returns the current settings as a dictionary."""
        return {
            "stream_name": self.stream_name_input.text(),
            "stream_duration": self.stream_duration_input.value(),
            "tlim": self.tlim_input.value(),
            "montage_path": self.montage_path_input.text(),
            "event_channel": self.event_channel_input.text(),
            "event_id": self.event_id_input.text(),
        }

    def set_settings(self, params: dict):
        """Sets the UI components from a settings dictionary."""
        self.stream_name_input.setText(params.get("stream_name"))
        self.stream_duration_input.setValue(int(params.get("stream_duration")))
        self.montage_path_input.setText(params.get("montage_path"))
        self.event_channel_input.setText(params.get("event_channel"))
        self.event_id_input.setText(params.get("event_id"))
        self.tlim_input.setValue(params.get("tlim"))

    def save_settings(self):
        """Saves the current settings to QSettings under the group."""
        params = self.get_settings()
        self.settings.beginGroup(self.GROUP_NAME)
        self.settings.setValue(self.SETTINGS_KEY, params)
        self.settings.endGroup()

    def load_settings(self):
        """Loads settings from QSettings, applying defaults for missing values."""
        self.settings.beginGroup(self.GROUP_NAME)

        default_params = {
            "stream_name": "EEGStream",
            "stream_duration": 5,
            "tlim": (-0.1, 0.3),
            "montage_path": "",
            "event_channel": "STI 014",
            "event_id": "1",
        }

        saved_params = self.settings.value(self.SETTINGS_KEY, {}) or {}
        self.settings.endGroup()

        # Merge defaults with saved params
        params = default_params.copy()
        params.update(saved_params)

        self.set_settings(params)


class RealTimeSettingsWidget(QWidget):
    """A widget to configure real-time plotting parameters."""

    GROUP_NAME = "real_time"
    SETTINGS_KEY = "plot_settings"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Initializes the user interface for the plot settings."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        group_box = QGroupBox("Real-Time Plot Settings")
        main_layout.addWidget(group_box)

        form_layout = QFormLayout(group_box)
        form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.refresh_rate_input = QSpinBox()
        self.refresh_rate_input.setSuffix(" Hz")
        self.refresh_rate_input.setRange(1, 60)
        form_layout.addRow("Refresh Rate:", self.refresh_rate_input)

        form_layout.addRow(QLabel("<br><b>Real-Time Pre-processing</b>"))

        self.art_rem_input = OptionalRangeWidget(
            ("Start:", "End:"),
            range=(None, None),
            suffix=" ms",
            required=(True, True),
            scale=1e-3,
        )
        form_layout.addRow("Artifact Removal:", self.art_rem_input)

        self.apply_bandpass = QCheckBox("Apply Bandpass Filter")
        self.bandpass_input = OptionalRangeWidget(
            labels=("Low:", "High:"), suffix=" Hz", range=(0.1, 200)
        )
        form_layout.addRow(self.apply_bandpass, self.bandpass_input)

        self.apply_notch = QCheckBox("Apply Notch Filter")
        self.notch_input = QLineEdit()
        form_layout.addRow(self.apply_notch, self.notch_input)

        self.apply_bandpass.toggled.connect(self.bandpass_input.setEnabled)
        self.apply_notch.toggled.connect(self.notch_input.setEnabled)

    def get_settings(self) -> dict:
        """Returns the current settings as a dictionary."""
        try:
            notch_freqs = [
                float(f.strip())
                for f in self.notch_input.text().split(",")
                if f.strip()
            ]
        except (ValueError, TypeError):
            notch_freqs = []

        return {
            "refresh_rate": self.refresh_rate_input.value(),
            "art_rem": self.art_rem_input.value(),
            "apply_bandpass": self.apply_bandpass.isChecked(),
            "bandpass_range": self.bandpass_input.value(),
            "apply_notch": self.apply_notch.isChecked(),
            "notch_freqs": notch_freqs,
        }

    def set_settings(self, params: dict):
        """Sets the UI components from a settings dictionary."""
        self.refresh_rate_input.setValue(params.get("refresh_rate"))
        self.apply_bandpass.setChecked(params.get("apply_bandpass"))
        self.apply_notch.setChecked(params.get("apply_notch"))

        self.art_rem_input.setValue(params.get("art_rem"))
        self.bandpass_input.setValue(params.get("bandpass_range"))

        notch_freqs = params.get("notch_freqs", [])
        self.notch_input.setText(
            ", ".join(
                map(str, notch_freqs if type(notch_freqs) == list else [notch_freqs])
            )
        )

        self.bandpass_input.setEnabled(self.apply_bandpass.isChecked())
        self.notch_input.setEnabled(self.apply_notch.isChecked())

    def save_settings(self):
        """Saves the current parameters to QSettings under the group."""
        params = self.get_settings()
        self.settings.beginGroup(self.GROUP_NAME)
        self.settings.setValue(self.SETTINGS_KEY, params)
        self.settings.endGroup()

    def load_settings(self):
        """Loads parameters from QSettings, applying defaults, and updates the UI."""
        self.settings.beginGroup(self.GROUP_NAME)

        default_params = {
            "refresh_rate": 24,
            "art_rem": (-0.005, 0.005),
            "apply_bandpass": False,
            "bandpass_range": (8.0, 80.0),
            "apply_notch": False,
            "notch_freqs": [50.0],
        }

        saved_params = self.settings.value(self.SETTINGS_KEY, {}) or {}
        self.settings.endGroup()

        params = default_params.copy()
        params.update(saved_params)

        self.set_settings(params)


class RealTimeSettings(QDialog):
    """A dialog for configuring real-time plot settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time Plot Settings")

        # Main layout
        layout = QVBoxLayout(self)

        # The widget with all the controls
        self.settings_widget = RealTimeSettingsWidget(self)

        # Standard OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(self.settings_widget)
        layout.addWidget(button_box)

    def get_settings(self) -> dict:
        """Returns the settings from the internal widget."""
        return self.settings_widget.get_settings()


class SingleChannelPlot(QDialog):
    """A dialog for plotting data from a single EEG channel."""

    closed = Signal()

    def __init__(self, ch_name, parent):
        super().__init__(parent)
        self.ch_name = ch_name
        self.parent = parent
        self.setup_ui()
        self.update_plot()
        self.show()

    def setup_ui(self):
        self.setWindowTitle(f"Channel: {self.ch_name}")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        figure = Figure(layout="constrained")
        self.canvas = FigureCanvas(figure)
        self.ax = figure.add_subplot(111)
        toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

    def update_plot(self):
        """Updates the plot with new data."""
        idx = self.parent.epochs.info["ch_names"].index(self.ch_name)
        self.ax.clear()
        self.ax.set_ylabel(r"$\mu$V")
        self.ax.set_xlabel("Time (ms)")

        data = self.parent._processed_data[:, idx, :] * 1e6
        evoked = np.nanmean(data, axis=0)
        ch_std = np.nanstd(data, axis=0)
        times = self.parent.times * 1e3
        t0_idx = np.argmin(abs(times))

        self.ax.fill_between(
            times, evoked - ch_std, evoked + ch_std, color="C0", alpha=0.5
        )
        self.ax.plot(times, evoked, lw=2, color="C0")
        self.ax.axvline(0, linestyle="--", color="k")
        self.ax.set_xlim(times[0], times[-1])
        self.ax.grid(True)
        self.ax.set_title(
            rf"{self.ch_name} ($n=${data.shape[0]} | $\bar{{\sigma}}_R$={np.nanmean(ch_std[t0_idx:]):.1f} $\mu V$)"
        )
        self.canvas.draw()

    def closeEvent(self, event):
        """Emits a closed Signal when the dialog is closed."""
        self.closed.emit()
        super().closeEvent(event)


class TopomapPlot(QDialog):
    """A dialog for displaying a topographical plot."""

    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.show()

    def update_plot(self, data, position, title="", names=None, cmap="turbo"):
        """Updates the topomap with new data."""
        self.ax.clear()
        im, _ = mne.viz.plot_topomap(
            data, position, names=names, cmap=cmap, show=False, axes=self.ax
        )
        if not hasattr(self, "cbar"):
            self.cbar = self.fig.colorbar(im, ax=self.ax)
        else:
            self.cbar.update_normal(im)
        self.ax.set_title(title)
        self.canvas.draw()

    def closeEvent(self, event):
        """Emits a closed Signal when the dialog is closed."""
        self.closed.emit()
        super().closeEvent(event)


class RealTimeERP(QWidget):
    GEOMETRY_SETTING = "real_time/geometry"
    """Main widget for real-time ERP visualization."""

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.stream = None
        self.epochs = None
        self._raw_data = None
        self.ch_names = []
        self.coords_2d = []
        self.colors = []
        self.times = []
        self.opened_single_channels = {}
        self.bads = []
        self.topomap_dialog = None
        self.drag_span = None

        self.setup_ui()
        self.setup_connections()
        self.setWindowTitle("Real-Time TEP Visualization")
        self._read_settings()

    # --- UI ---
    def setup_ui(self):
        """Initializes the user interface."""
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        main_layout = QVBoxLayout(self)
        self.toolbar = self._create_toolbar()
        self.topo_widget = self._create_topo_widget()
        self.evoked_widget = self._create_evoked_widget()

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.topo_widget, 2)
        main_layout.addWidget(self.evoked_widget, 1)

    def _read_settings(self):
        """Reads and applies saved application settings."""
        geometry = QSettings().value(self.GEOMETRY_SETTING)
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.setGeometry(100, 100, 1280, 800)  # Default size

    def _create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addAction("🧠 Toggle Topoview").triggered.connect(
            lambda: self.topo_widget.setVisible(not self.topo_widget.isVisible())
        )
        toolbar.addAction("🦋 Toggle Butterfly Plot").triggered.connect(
            lambda: self.evoked_widget.setVisible(not self.evoked_widget.isVisible())
        )
        toolbar.addAction("⚙️ Settings ").triggered.connect(self.update_settings)
        return toolbar

    def _create_topo_widget(self):
        dock = QDockWidget("Topoplot")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.topo_fig = Figure(layout="constrained")
        self.topo_canvas = FigureCanvas(self.topo_fig)
        self.topo_ax = self.topo_fig.add_subplot(111)
        layout.addWidget(NavigationToolbar(self.topo_canvas, self))
        layout.addWidget(self.topo_canvas)
        dock.setWidget(widget)
        return dock

    def _create_evoked_widget(self):
        dock = QDockWidget("Evoked Potentials")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.evoked_fig = Figure(layout="constrained")
        self.evoked_canvas = FigureCanvas(self.evoked_fig)
        self.evoked_ax = self.evoked_fig.add_subplot(111)
        layout.addWidget(self.evoked_canvas)
        layout.addWidget(NavigationToolbar(self.evoked_canvas, self))
        dock.setWidget(widget)
        return dock

    # --- Connection ---
    def setup_connections(self):
        """Connects UI element signals to slots."""
        self.topo_canvas.mpl_connect("pick_event", self.on_topo_pick)
        self.evoked_canvas.mpl_connect("button_press_event", self.on_evoked_press)
        self.evoked_canvas.mpl_connect("motion_notify_event", self.on_evoked_motion)
        self.evoked_canvas.mpl_connect("button_release_event", self.on_evoked_release)

    def start_visualization(self):
        """Starts the connection process in a separate thread with a progress dialog."""
        self.thread = QThread(self)
        self.worker = ConnectionWorker(self.params)
        self.worker.moveToThread(self.thread)

        self.progress_dialog = QProgressDialog(
            "Connecting to LSL stream...", "Cancel", 0, 0, self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setWindowTitle("Connecting")

        # Connections
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_connection_success)
        self.worker.error.connect(self._on_connection_failure)
        self.progress_dialog.canceled.connect(
            self.close
        )  # Close main window if user cancels

        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.progress_dialog.exec()

    def _on_connection_success(self, stream, epochs):
        """Handles successful connection and continues initialization."""
        self.progress_dialog.accept()
        self.stream = stream
        self.epochs = epochs

        self.ch_names = self.epochs.info["ch_names"]
        # Use a generic montage if none is found
        if self.stream.get_montage() is None:
            try:
                print("No montage found in stream. Using a standard montage.")
                montage = mne.channels.make_standard_montage("easycap-M1")
                self.stream.set_montage(montage)
                # Ensure info is updated in epochs object as well
                self.epochs.info = self.stream.info
            except:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Could not use standard montage. Please check data format or provide the correct montage file.",
                )
                self.closeEvent()

        self.info = self.epochs.info

        self.decimate = self.params.get("decimate", 5)
        self._raw_data = np.zeros((0, len(self.ch_names), len(self.epochs.times)))
        coords_3d = np.array([ch["loc"][:3] for ch in self.info["chs"]])
        self.coords_2d = self.project_electrodes_to_2d(coords_3d)
        self.colors = self.to_rgb(coords_3d)
        self.times = self.epochs.times[:: self.decimate]
        self.sfreq = self.stream.info["sfreq"]

        self.setup_topo_plot()
        self.setup_evoked_plot()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1000 / 24)  # ~24 FPS

    def _on_connection_failure(self, error_message):
        """Handles failed connection."""
        self.progress_dialog.reject()
        QMessageBox.critical(self, "Connection Failed", error_message)
        self.close()

    # --- Plots ---
    def setup_topo_plot(self):
        """Sets up the initial topoplot."""
        self.topo_ax.clear()
        self.update_axis_scales()
        self.topo_lines = []
        t0_idx = np.argmin(abs(self.times))
        for i, (x, y) in enumerate(self.coords_scaled):
            plot_x = 0.5 + x * 0.5 - self.topo_ax_size_fig / 2
            plot_y = 0.5 + y * 0.5 - self.topo_ax_size_fig / 2
            self.topo_ax.plot(
                [
                    plot_x + self.scaled_times[t0_idx],
                    plot_x + self.scaled_times[t0_idx],
                ],
                [plot_y, plot_y + self.topo_ax_height],
                color=plt.rcParams["text.color"],
                linewidth=1,
                linestyle="--",
                alpha=0.5,
            )
            (line,) = self.topo_ax.plot(
                self.scaled_times + plot_x,
                np.zeros_like(self.scaled_times) + plot_y,
                color=self.colors[i],
                label=self.ch_names[i],
                lw=2,
            )
            line.set_picker(5)
            self.topo_ax.text(
                plot_x + self.topo_ax_width / 2,
                plot_y + self.topo_ax_height,
                self.ch_names[i],
                ha="center",
                va="bottom",
                fontsize=8,
            )
            self.topo_lines.append(line)
        self.topo_ax.set_axis_off()
        self.topo_canvas.draw()

    def setup_evoked_plot(self):
        """Sets up the initial evoked plot."""
        self.evoked_ax.clear()
        times = self.times * 1e3
        self.evoked_lines = [
            self.evoked_ax.plot(
                times,
                np.zeros_like(times),
                color=self.colors[i],
                label=self.ch_names[i],
            )[0]
            for i in range(len(self.ch_names))
        ]
        self.evoked_ax.set_xlim(times[0], times[-1])
        self.evoked_ax.set_xlabel("Time (ms)")
        self.evoked_ax.set_ylabel(r"Potential ($\mu$V)")
        self.evoked_ax.grid(True)
        self.evoked_canvas.draw()

    def update_plots(self):
        """Periodically gets new data, adds it to the raw buffer, and triggers processing."""
        if self.epochs.n_new_epochs == 0:
            return
        new_data = self.epochs.get_data(n_epochs=self.epochs.n_new_epochs)
        self._raw_data = np.concatenate([self._raw_data, new_data], axis=0)
        self._process_and_update_plots()

    def _process_and_update_plots(self):
        """
        Applies all processing from self.params to a copy of the raw data
        and then updates all visualizations.
        """
        if self._raw_data.shape[0] == 0:
            return

        # 1. Start with a fresh copy of the raw data
        plot_data = self._raw_data.copy()
        original_sfreq = self.stream.info["sfreq"]

        # 6. Artifact Removal
        art_rem_start, art_rem_end = self.params.get("art_rem", (0, 0))
        if art_rem_end > art_rem_start:
            original_times = self.epochs.times
            start_idx = np.searchsorted(original_times, art_rem_start)
            end_idx = np.searchsorted(original_times, art_rem_end)
            width = end_idx - start_idx
            if end_idx > start_idx:
                print(f"Interpolating artifact from {art_rem_start}s to {art_rem_end}s")
                plot_data[:, :, start_idx:end_idx] = plot_data[
                    :, :, start_idx - width : start_idx
                ]
                # # For each epoch and channel, replace the artifact segment
                # for i in range(plot_data.shape[0]): # epochs
                #     for j in range(plot_data.shape[1]): # channels
                #         # Simple linear interpolation
                #         start_val = plot_data[i, j, start_idx - 1]
                #         end_val = plot_data[i, j, end_idx]
                #         interp_line = np.linspace(start_val, end_val, end_idx - start_idx)
                #         plot_data[i, j, start_idx:end_idx] = interp_line

        # 2. Apply Bandpass Filter (if enabled)
        if self.params.get("apply_bandpass", False):
            low, high = self.params.get("bandpass_range", (0, 0))
            if high > low:
                # Filter at the original sampling frequency to prevent aliasing
                plot_data = mne.filter.filter_data(
                    plot_data,
                    method="iir",
                    iir_params=dict(order=3, ftype="butter", phase="zero-double"),
                    sfreq=original_sfreq,
                    l_freq=low,
                    h_freq=high,
                    verbose=False,
                )
                print(f"Applying bandpass filter: {low}-{high} Hz")

        # 3. Apply Notch Filter (if enabled)
        if self.params.get("apply_notch", False):
            freqs = self.params.get("notch_freqs", [])
            if freqs:
                plot_data = mne.filter.notch_filter(
                    plot_data,
                    method="iir",
                    iir_params=dict(order=3, ftype="butter", phase="zero-double"),
                    Fs=original_sfreq,
                    freqs=freqs,
                    verbose=False,
                )
                print(f"Applying notch filter at: {freqs} Hz")

        # 4. Apply Referencing
        ref_method = self.params.get("reference", "average")
        if ref_method == "average":
            # Using np.nanmean to be robust against potential NaNs
            plot_data -= np.nanmean(plot_data, axis=1, keepdims=True)

        # 5. Decimate the data AFTER filtering
        plot_data = plot_data[:, :, :: self.decimate]

        # 7. Update the processed data attribute
        self._processed_data = plot_data
        self.setWindowTitle(f"Real-Time TEP - Epochs: {self._processed_data.shape[0]}")

        # 8. Update all plots
        self.update_topo_plot()
        self.update_evoked_plot()
        for ch, plot in self.opened_single_channels.items():
            plot.update_plot()
        if self.topomap_dialog:
            self.update_topomap()

    def update_topo_plot(self):
        mean_data = np.nanmean(self._processed_data, axis=0)
        for i, line in enumerate(self.topo_lines):
            y_offset = 0.5 + self.coords_scaled[i, 1] * 0.5 - self.topo_ax_size_fig / 2
            scaled_data = self.scale_data(mean_data[i, :], self.topo_ax_height)
            line.set_ydata(scaled_data + y_offset)
            line.set_alpha(0.25 if self.ch_names[i] in self.bads else 1.0)
        self.topo_ax.set_title(f"Epochs: {self._raw_data.shape[0]}")
        self.topo_canvas.draw()

    def update_evoked_plot(self):
        mean_data = np.nanmean(self._processed_data, axis=0) * 1e6
        for i, line in enumerate(self.evoked_lines):
            if self.ch_names[i] in self.bads:
                line.set_ydata(np.full_like(self.times, np.nan))
            else:
                line.set_ydata(mean_data[i, :])
        self.evoked_ax.relim()
        self.evoked_ax.autoscale_view(True, True, True)
        self.evoked_canvas.draw()

    def on_topo_pick(self, event):
        ch_name = event.artist.get_label()
        if event.mouseevent.button == 1:  # Left-click to open single channel plot
            if ch_name not in self.opened_single_channels:
                plot = SingleChannelPlot(ch_name, self)
                plot.closed.connect(lambda: self.opened_single_channels.pop(ch_name))
                self.opened_single_channels[ch_name] = plot
            else:
                self.opened_single_channels[ch_name].activateWindow()
        elif event.mouseevent.button == 3:  # Right-click to mark as bad
            if ch_name in self.bads:
                self.bads.remove(ch_name)
            else:
                self.bads.append(ch_name)
            self.update_topo_plot()
            self.update_evoked_plot()

    def on_evoked_press(self, event):
        if event.inaxes != self.evoked_ax or event.button != 1:
            return
        if self.drag_span:
            self.drag_span.remove()
        self.drag_span = self.evoked_ax.axvspan(
            event.xdata, event.xdata, color="gray", alpha=0.3
        )
        self.evoked_canvas.draw()

    def on_evoked_motion(self, event):
        if not self.drag_span or event.xdata is None:
            return
        start_x = self.drag_span.get_x()
        current_x = event.xdata
        if current_x is None:
            return
        left = min(start_x, current_x)
        right = max(start_x, current_x)
        self.drag_span.set_x(left)
        self.drag_span.set_width(right - left)
        self.evoked_canvas.draw_idle()

    def on_evoked_release(self, event):
        if not self.drag_span:
            return
        start = self.drag_span.get_x()
        end = self.drag_span.get_width() + start
        self.current_topomap_time_range = [min(start, end), max(start, end)]
        self.drag_span.remove()
        self.drag_span = None
        self.evoked_canvas.draw()

        if (
            abs(self.current_topomap_time_range[0] - self.current_topomap_time_range[1])
            > 1
        ):  # Min time diff
            if not self.topomap_dialog:
                self.topomap_dialog = TopomapPlot(self)
                self.topomap_dialog.closed.connect(
                    lambda: setattr(self, "topomap_dialog", None)
                )
            self.update_topomap()
            self.topomap_dialog.show()

    def update_topomap(self):
        """Updates the external topomap dialog."""
        if not self.topomap_dialog or not hasattr(self, "current_topomap_time_range"):
            return

        start_t, end_t = self.current_topomap_time_range
        time_mask = (self.times * 1e3 >= start_t) & (self.times * 1e3 <= end_t)

        valid_chs_mask = [
            i for i, ch in enumerate(self.ch_names) if ch not in self.bads
        ]
        if not valid_chs_mask:
            return
        data = self._processed_data[:, :, time_mask]
        data = np.nanmean(data[:, valid_chs_mask, :], axis=(0, 2)) * 1e6
        pos = np.array(
            [
                self.coords_2d[i]
                for i, ch in enumerate(self.ch_names)
                if ch not in self.bads
            ]
        )
        pos /= np.max(np.linalg.norm(pos)) * 2
        names = [ch for ch in self.ch_names if ch not in self.bads]
        title = rf"{start_t:.1f} to {end_t:.1f} ms | n={self._processed_data.shape[0]}"
        self.topomap_dialog.update_plot(data, pos, title=title, names=names)

    def update_axis_scales(self):
        """Calculates scaling factors for topo subplots."""
        min_dist = np.min(pdist(self.coords_2d)) if self.coords_2d.shape[0] > 1 else 0.2
        scaling_factor = 1 / (1 + min_dist / 2)
        self.coords_scaled = self.coords_2d * scaling_factor
        self.topo_ax_size_fig = min_dist * 0.5 * scaling_factor
        self.topo_ax_width = self.topo_ax_height = self.topo_ax_size_fig
        self.scaled_times = np.linspace(0, self.topo_ax_width, len(self.times))

    # --- Static ---
    @staticmethod
    def project_electrodes_to_2d(positions_3d):
        """Projects 3D electrode positions to a 2D plane."""
        center = np.mean(positions_3d, axis=0)
        centered_pos = positions_3d - center
        d = np.sqrt(np.sum(centered_pos**2, axis=1))
        # Handle the case of a point at the origin
        d[d == 0] = 1.0
        theta = np.arctan2(centered_pos[:, 1], centered_pos[:, 0])
        phi = np.arccos(centered_pos[:, 2] / d)
        x_2d = phi * np.cos(theta)
        y_2d = phi * np.sin(theta)
        coords_2d = np.column_stack([x_2d, y_2d])
        # Normalize so the largest radius is 1
        max_radius = np.max(np.linalg.norm(coords_2d, axis=1))
        if max_radius > 0:
            coords_2d /= max_radius
        return coords_2d

    @staticmethod
    def to_rgb(positions_3d):
        """Converts 3D positions to RGB colors."""
        xyz = positions_3d.copy()
        xyz -= xyz.min(axis=0)
        xyz /= xyz.max(axis=0)
        return xyz

    @staticmethod
    def scale_data(x, height=1):
        """Scales data to a specified height."""
        ptp = np.nanmax(x) - np.nanmin(x)
        return height * (x - np.nanmin(x)) / ptp if ptp > 0 else np.zeros_like(x)

    # --- Settings & Close ---
    def update_settings(self):
        dialog = RealTimeSettings(self)
        if dialog.exec():
            params = dialog.get_settings()
            print("Dialog accepted. New settings:", params)
            self.params.update(params)
            self._process_and_update_plots()
        else:
            print("Dialog cancelled.")

    def closeEvent(self, event):
        """Handles widget close event."""
        if hasattr(self, "timer"):
            self.timer.stop()
        # if hasattr(self, 'thread') and self.thread.isRunning():
        #     self.thread.quit()
        #     self.thread.wait()  # Wait for the thread to finish
        for window in list(self.opened_single_channels.values()):
            window.close()
        if self.topomap_dialog:
            self.topomap_dialog.close()
        if self.epochs is not None:
            if self.epochs.connected:
                self.epochs.disconnect()
        if self.stream is not None:
            if self.stream.connected:
                self.stream.disconnect()

        QSettings().setValue(self.GEOMETRY_SETTING, self.saveGeometry())
        super().closeEvent(event)


class RealTimeMainWidget(QWidget):
    """The main entry point widget for the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rt_erp_widget = None
        self.setup_ui()
        self.setWindowTitle("LSL Stream Visualizer")

    def setup_ui(self):
        """Sets up the UI for the main widget."""
        layout = QVBoxLayout(self)
        self.connection_widget = ConnectionWidget()
        layout.addWidget(self.connection_widget)

        self.connect_button = QPushButton("Connect to LSL Stream")
        self.connect_button.clicked.connect(self.launch_visualizer)
        # self.connect_button.setEnabled(False)

        layout.addWidget(self.connect_button)
        layout.addStretch()  # Add stretch to push widgets to the top

    def launch_visualizer(self):
        """Launches the real-time ERP visualizer."""
        params = self.connection_widget.get_settings()
        params.update({"reference": "average"})
        params.update(RealTimeSettingsWidget().get_settings())
        if self.rt_erp_widget:
            self.rt_erp_widget.close()

        self.rt_erp_widget = RealTimeERP(params, self)
        self.rt_erp_widget.show()
        self.rt_erp_widget.start_visualization()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = RealTimeMainWidget()
    main_window.show()
    sys.exit(app.exec())
