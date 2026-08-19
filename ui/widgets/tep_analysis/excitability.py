import sys
import numpy as np
import mne
import matplotlib
from enum import Enum, auto

# Set the backend to PySide6
matplotlib.use("QtAgg")

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QDialog,
    QGridLayout,
    QFrame,
    QMessageBox,
)
from PySide6.QtGui import QIcon, QScreen
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from ui.dialogs.sensor_selection_dialog import SensorSelectionDialog
import logging
from utils import use_plot_style

logger = logging.getLogger(__name__)


# --- Enum for analysis modes for better code safety and readability ---
class AnalysisMode(Enum):
    GMFP = auto()
    LFP = auto()
    P2P = auto()


class ExcitabilityApp(QDialog):
    """
    An application for performing excitability analysis on MNE Epochs data.

    This widget allows visualizing evoked potentials, selecting time windows and
    channels, and calculating metrics like Global Mean Field Power (GMFP),
    Local Field Power (LFP), and Peak-to-Peak (P2P) amplitude.
    """

    def __init__(self, epochs: mne.Epochs, parent=None):
        super().__init__(parent)
        self.setModal(True)

        # --- Data ---
        self.epochs = epochs

        # --- State Parameters ---
        self.span = np.array([20, 200])
        self.xlim = np.array([-100, 400])
        self.mode = AnalysisMode.GMFP
        self.selected_channels: list[str] = []

        # --- UI Initialization ---
        self._setup_ui()
        self._setup_plots()
        self.show()

    def _setup_ui(self) -> None:
        """Initializes the main UI window and layout."""
        self.setWindowTitle("Excitability Analysis")
        # Assuming 'sspy/resources/excitability.png' is in the correct path
        self.setWindowIcon(QIcon("sspy/resources/excitability.png"))
        self.setGeometry(100, 100, 1280, 920)
        self.move(self.screen().geometry().center() - self.frameGeometry().center())

        # --- Create main components ---
        control_panel = self._create_control_panel()
        self.canvas = self._create_plot_canvas()

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(self.canvas, 4)

    def _create_control_panel(self) -> QFrame:
        """Creates the top control panel with buttons and inputs."""
        card = QFrame()
        card.setObjectName("card")
        vbox = QVBoxLayout(card)

        # --- Mode Buttons ---
        self.mode_buttons = {
            AnalysisMode.GMFP: QPushButton("Global Mean Field Power"),
            AnalysisMode.LFP: QPushButton("Local Field Power"),
            AnalysisMode.P2P: QPushButton("Peak to Peak"),
        }
        self.mode_buttons[AnalysisMode.GMFP].clicked.connect(
            lambda: self._change_mode(AnalysisMode.GMFP)
        )
        self.mode_buttons[AnalysisMode.LFP].clicked.connect(
            lambda: self._change_mode(AnalysisMode.LFP)
        )
        self.mode_buttons[AnalysisMode.P2P].clicked.connect(
            lambda: self._change_mode(AnalysisMode.P2P)
        )

        for button in self.mode_buttons.values():
            vbox.addWidget(button)

        # --- Inputs Grid ---
        grid = QGridLayout()
        self.time_window_input = QLineEdit()
        self.channels_input = QLineEdit()
        self.xlim_input = QLineEdit()
        self.select_channels_button = QPushButton("Select Channels")

        grid.addWidget(QLabel("Time window:"), 0, 0)
        grid.addWidget(self.time_window_input, 0, 1)
        grid.addWidget(self.select_channels_button, 1, 0)
        grid.addWidget(self.channels_input, 1, 1)
        grid.addWidget(QLabel("Plot limits:"), 2, 0)
        grid.addWidget(self.xlim_input, 2, 1)

        self.time_window_input.returnPressed.connect(self._update_view)
        self.channels_input.returnPressed.connect(self._update_view)
        self.xlim_input.returnPressed.connect(self._update_view)
        self.select_channels_button.clicked.connect(self._select_roi)
        vbox.addLayout(grid)

        # --- Outcome Label ---
        self.outcome_label = QLabel("")
        self.outcome_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.outcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.outcome_label.setStyleSheet("font-size: 16pt")
        vbox.addWidget(self.outcome_label, alignment=Qt.AlignmentFlag.AlignCenter)

        return card

    def _create_plot_canvas(self) -> FigureCanvas:
        """Creates the Matplotlib figure and canvas."""
        self.figure = Figure(layout="constrained")
        return FigureCanvas(self.figure)

    @use_plot_style
    def _setup_plots(self) -> None:
        """Sets up the initial state of the Matplotlib plots and axes."""
        self.xlim_input.setText(f"{self.xlim[0]} {self.xlim[1]}")
        self.time_window_input.setText(f"{self.span[0]} {self.span[1]}")

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 3, width_ratios=[2, 1, 0.2])
        self.ax = self.figure.add_subplot(gs[0, 0])
        self.outcome_ax = self.figure.add_subplot(gs[1, 0], sharex=self.ax)
        self.topo_ax = self.figure.add_subplot(gs[0, 1])
        self.power_ax = self.figure.add_subplot(gs[1, 1])
        self.cax1 = self.figure.add_subplot(gs[0, 2])
        self.cax2 = self.figure.add_subplot(gs[1, 2])

        # --- Plot initial evoked potential ---
        self.epochs.average().plot(
            axes=self.ax, show=False, selectable=False, time_unit="ms"
        )
        self.ax.grid(True)
        self.ax.set_xlabel("")
        self.ax.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels

        # --- Span Selectors ---
        self.span_selector = SpanSelector(
            self.ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="red"),
        )
        self.span_selector2 = SpanSelector(
            self.outcome_ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="red"),
        )

        # --- Initial view update ---
        self._update_view()

    def _update_view(self) -> None:
        """Orchestrates the full update of the plot based on current parameters."""
        if not self._update_params_from_ui():
            return

        # --- Clear previous plot elements ---

        # Manually remove the vspan from the top axis (`self.ax`), as this
        # axis is persistent and not cleared.
        if hasattr(self, "current_vspan1") and self.current_vspan1.axes:
            self.current_vspan1.remove()

        # Clear the other axes. This automatically removes all artists on them,
        # including the old `current_vspan2` on `outcome_ax`.
        self.outcome_ax.clear()
        self.topo_ax.clear()
        self.power_ax.clear()
        self.cax1.clear()
        self.cax2.clear()

        # --- Plot new data ---
        self._plot_outcome_and_update_label()
        self._plot_topographies()
        self._finalize_plot_aesthetics()
        self.canvas.draw()

    def _update_params_from_ui(self) -> bool:
        """Reads and validates user input from QLineEdit widgets."""
        try:
            span_vals = [float(x) for x in self.time_window_input.text().split()]
            xlim_vals = [float(x) for x in self.xlim_input.text().split()]

            if len(span_vals) != 2 or span_vals[0] >= span_vals[1]:
                QMessageBox.warning(self, "Error", "Invalid time window input.")
                return False
            self.span = np.array(span_vals)

            if len(xlim_vals) != 2 or xlim_vals[0] >= xlim_vals[1]:
                QMessageBox.warning(self, "Error", "Invalid plot limits input.")
                return False
            self.xlim = np.array(xlim_vals)

            self.selected_channels = (
                self.channels_input.text().replace(" ", "").split(",")
            )

            if self.mode in [AnalysisMode.LFP, AnalysisMode.P2P]:
                if not self.selected_channels:
                    QMessageBox.warning(
                        self, "Error", "Please select channels for LFP or P2P mode."
                    )
                    return False
                if any(ch not in self.epochs.ch_names for ch in self.selected_channels):
                    QMessageBox.warning(self, "Error", "Invalid channel selection.")
                    return False
        except ValueError:
            QMessageBox.warning(
                self, "Error", "Inputs must be space-separated numbers."
            )
            return False
        return True

    def _plot_outcome_and_update_label(self) -> None:
        """Calculates and plots the outcome data (GMFP, LFP, P2P)."""
        times_ms = self.epochs.times * 1e3
        span_idx = np.where((times_ms >= self.span[0]) & (times_ms <= self.span[1]))[0]

        if self.mode == AnalysisMode.GMFP:
            data = np.sqrt(np.mean(self.epochs.average().data ** 2, axis=0)) * 1e6
            auc = np.trapezoid(data[span_idx], x=times_ms[span_idx])
            self.outcome_label.setText(f"AUC (GMFP): {auc:.2f} µV·ms")
            self.outcome_ax.plot(times_ms, data, label="GMFP")

        elif self.mode == AnalysisMode.LFP:
            data = (
                np.sqrt(
                    np.mean(
                        self.epochs.copy().pick(self.selected_channels).average().data
                        ** 2,
                        axis=0,
                    )
                )
                * 1e6
            )
            auc = np.trapezoid(data[span_idx], x=times_ms[span_idx])
            self.outcome_label.setText(f"AUC (LFP): {auc:.2f} µV·ms")
            self.outcome_ax.plot(
                times_ms, data, label=f"LFP ({', '.join(self.selected_channels)})"
            )

        elif self.mode == AnalysisMode.P2P:
            data = (
                self.epochs.copy().pick(self.selected_channels).average().data.mean(0)
                * 1e6
            )
            peak_indices = self.epochs.time_as_index(self.span * 1e-3)
            p2p_data = data[peak_indices[0] : peak_indices[1] + 1]

            min_val, max_val = p2p_data.min(), p2p_data.max()
            min_idx = np.argmin(p2p_data) + peak_indices[0]
            max_idx = np.argmax(p2p_data) + peak_indices[0]
            p2p = max_val - min_val

            self.outcome_label.setText(f"Peak to Peak: {p2p:.2f} µV")
            self.outcome_ax.plot(
                times_ms, data, label=f"Avg ({', '.join(self.selected_channels)})"
            )
            self.outcome_ax.plot(
                times_ms[[min_idx, max_idx]], [min_val, max_val], "ro", markersize=5
            )
            self.outcome_ax.annotate(
                f"{min_val:.2f} µV at {times_ms[min_idx]:.1f} ms",
                xy=(times_ms[min_idx], min_val),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.7
                ),  # Added bbox here
            )
            self.outcome_ax.annotate(
                f"{max_val:.2f} µV at {times_ms[max_idx]:.1f} ms",
                xy=(times_ms[max_idx], max_val),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                bbox=dict(
                    boxstyle="round,pad=00.3", fc="white", ec="black", lw=0.5, alpha=0.7
                ),  # Added bbox here
            )

    def _plot_topographies(self) -> None:
        """Calculates and plots the topographies for the selected time window."""
        span_idx = self.epochs.time_as_index(self.span * 1e-3)
        sliced_data = self.epochs.average().data[:, span_idx[0] : span_idx[1] + 1]

        # Topography of mean activity
        mean_topo_data = sliced_data.mean(axis=1)
        im1 = mne.viz.plot_topomap(
            mean_topo_data,
            self.epochs.info,
            axes=self.topo_ax,
            cmap="turbo",
            show=False,
        )[0]
        self.figure.colorbar(im1, cax=self.cax1, label="µV")
        self.topo_ax.set_title("Mean Activity")

        # Topography of power
        power_topo_data = np.sqrt(np.mean(sliced_data**2, axis=1))
        im2 = mne.viz.plot_topomap(
            power_topo_data,
            self.epochs.info,
            axes=self.power_ax,
            cmap="turbo",
            show=False,
        )[0]
        self.figure.colorbar(im2, cax=self.cax2, label="µV (RMS)")
        self.power_ax.set_title("RMS Power")

    def _finalize_plot_aesthetics(self) -> None:
        """Sets final plot limits, labels, grids, and visual spans."""
        # --- Visual Spans ---
        self.current_vspan1 = self.ax.axvspan(
            self.span[0], self.span[1], alpha=0.2, color="gray"
        )
        self.current_vspan2 = self.outcome_ax.axvspan(
            self.span[0], self.span[1], alpha=0.2, color="gray"
        )

        # --- Common Aesthetics ---
        self.ax.set_xlim(self.xlim)
        self.outcome_ax.set_xlabel("Time (ms)")
        self.outcome_ax.grid(True)
        self.outcome_ax.legend(loc="upper right")
        self.outcome_ax.axvline(0, color="k", linestyle="--", linewidth=1)
        self.ax.axvline(0, color="k", linestyle="--", linewidth=1)
        self.figure.suptitle("Evoked Potential Analysis")

    # --- Event Handlers / Slots ---
    def _on_span_select(self, xmin: float, xmax: float) -> None:
        """Callback for the Matplotlib SpanSelector."""
        self.time_window_input.setText(f"{xmin:.2f} {xmax:.2f}")
        self._update_view()

    def _change_mode(self, mode: AnalysisMode) -> None:
        """Changes the analysis mode and updates the view."""
        self.mode = mode
        for m, button in self.mode_buttons.items():
            is_active = m == self.mode
            button.setStyleSheet("background-color: lightblue" if is_active else "")
        self._update_view()

    def _select_roi(self) -> None:
        try:
            dialog = SensorSelectionDialog(self.epochs)
            if dialog.exec():
                self.roi_selection_list = dialog.get_selected_channels()
                self.channels_input.setText(", ".join(self.roi_selection_list))
        except Exception as e:
            logger.error(f"Could not open sensor plot: {e}", exc_info=True)
            QMessageBox.warning(
                self, "Plotting Error", "Could not display sensor selection plot."
            )
        finally:
            self._update_view()


if __name__ == "__main__":
    # --- Create Dummy MNE Data for Demonstration ---
    sfreq = 1000
    n_channels = 32
    n_epochs = 100
    tmin, tmax = -0.8, 0.8

    data = np.random.randn(n_epochs, n_channels, int(sfreq * (tmax - tmin)))
    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:n_channels]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, tmin=tmin)
    epochs.set_montage(montage)

    # --- Run Application ---
    app = QApplication(sys.argv)
    ex = ExcitabilityApp(epochs)
    sys.exit(app.exec())
