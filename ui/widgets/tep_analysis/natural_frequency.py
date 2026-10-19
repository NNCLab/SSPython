import mne
import numpy as np
import logging
import traceback
from typing import List, Dict
from tqdm import tqdm
from PySide6.QtWidgets import (
    QDialog,
    QComboBox,
    QHBoxLayout,
    QSplitter,
    QFrame,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QPushButton,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import Qt, Slot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ui.dialogs.sensor_selection_dialog import SensorSelectionDialog
from ui.widgets.event_selection import EventSelector
from utils import use_plot_style, Worker, parse_tuple
from core.tfr_statistics import bootstrap_power

logger = logging.getLogger(__name__)


def TFstat(
    TF_trials: np.ndarray,
    times: np.ndarray,
    baseline: List[float] = [-0.5, -0.05],
    ch_name: str = None,
    alpha: float = 0.05,
    n_permutations: int = 1000,
    fdr_correction: bool = True,
    verbose=True,
) -> np.ndarray:
    """Compute statistical mask for time-frequency power representation."""
    return bootstrap_power(
        TF_trials[:, np.newaxis], times, baseline=baseline, alpha=alpha,
        n_permutations=n_permutations, fdr_correction=fdr_correction,
        verbose=verbose,
    )[0]


def calculate_nf_data(
    epochs: mne.Epochs,
    roi: List[str],
    freqs: np.ndarray,
    n_cycles: float,
    baseline: List[float],
    response: List[float],
    n_permutations: int,
    baseline_mode: str | None = "mean",
) -> Dict:
    """
    Calculates the natural frequency and returns all data required for plotting.
    MODIFIED to align with the logic of the provided class methods.
    """
    times = epochs.times

    # 1. Compute TFR only on the selected ROI
    ep_roi = epochs.copy().pick(roi, verbose=False)
    tfr = ep_roi.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False,
        verbose=False,
    )
    # tfr.data shape: (n_trials, n_channels, n_freqs, n_times)

    # 2. Compute statistical mask for each channel in the ROI individually
    stat_mask = np.zeros(tfr.data.shape[1:])  # (n_channels, n_freqs, n_times)
    pbar = tqdm(
        enumerate(tfr.ch_names),
        desc="Running permutations per channel...",
        total=len(tfr.ch_names),
    )
    for i, ch_name in pbar:
        stat_mask[i, :, :] = TFstat(
            tfr.data[:, i, :, :],
            times,
            baseline=baseline,
            n_permutations=n_permutations,
            ch_name=ch_name,
        )

    avg_tfr = tfr.data.mean(axis=0)  # (n_channels, n_freqs, n_times)
    if baseline_mode is not None:
        avg_tfr = mne.baseline.rescale(
            avg_tfr, times, baseline, mode=baseline_mode, copy=False, verbose=False,
        )

    response_idx = np.where((times >= response[0]) & (times <= response[1]))[0]
    masked_tfr = avg_tfr * stat_mask  # (n_channels, n_freqs, n_times)
    masked_avg_tfr = masked_tfr.mean(axis=0)  # (n_freqs, n_times)

    # check if there's any non False in stat_mask
    if np.all(stat_mask[:, :, response_idx] == 0):
        logger.warning(
            "No significant power found. Plotting unmasked power distribution instead."
        )
        agg_freq = avg_tfr[:, :, response_idx].mean(
            axis=(0, 2)
        )  # Average over channels and the response window
    else:
        # 5. Aggregate power over the response window to find the natural frequency
        agg_freq = np.nansum(
            masked_avg_tfr[:, response_idx], axis=1
        )  # Sum over response window

    idx_nat_freq = np.nanargmax(agg_freq)
    nat_freq = freqs[idx_nat_freq]
    agg_freq_plot = agg_freq

    evoked_all = epochs.average()
    evoked_roi = ep_roi.average()

    return {
        "nat_freq": nat_freq,
        "avg_tfr": avg_tfr,
        "stat_mask": stat_mask,
        "agg_freq": agg_freq_plot,
        "times": times,
        "freqs": freqs,
        "roi": roi,
        "evoked_all": evoked_all.data,
        "evoked_roi": evoked_roi.data,
    }


class NaturalFrequencyApp(QDialog):
    """A dialog for calculating and inspecting Natural Frequency from MNE Epochs."""

    def __init__(self, epochs: mne.Epochs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Natural Frequency Inspection")
        self.setMinimumSize(1024, 768)
        self.setModal(True)

        self.epochs = epochs
        self.roi_selection_list = None

        self._init_ui()
        self._setup_connections()
        self.show()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        params_widget = self._create_parameters_group()
        splitter.addWidget(params_widget)

        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.toolbar)
        splitter.addWidget(plot_frame)

        splitter.setSizes([300, 700])

    def _create_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Parameters")
        layout = QVBoxLayout(group)
        form_layout = QFormLayout()
        self.event_selector = EventSelector(self.epochs, self)
        form_layout.addRow("Event:", self.event_selector)

        self.roi_button = QPushButton("Select ROI from 2D Layout")
        self.channels_input = QLineEdit()
        self.channels_input.setPlaceholderText("e.g., C3 C5 FC1 or use button")
        form_layout.addRow(self.roi_button)
        form_layout.addRow("Selected Channels:", self.channels_input)

        start = max(float(self.epochs.times[0]), -0.5)
        stop = min(float(self.epochs.times[-1]), -0.05)
        if stop <= start:
            start, stop = float(self.epochs.times[0]), float(self.epochs.times[-1])
        self.baseline_input = QLineEdit(f"{start * 1000:g}, {stop * 1000:g}")
        self.baseline_mode_combo = QComboBox()
        self.baseline_mode_combo.addItems(["mean", "ratio", "logratio", "percent", "zscore", "zlogratio"])
        self.baseline_mode_combo.addItem("None", None)
        self.baseline_mode_combo.setToolTip(
            "MNE baseline correction applied to mean trial power per channel. "
            "The baseline interval is also required for statistics."
        )
        self.response_input = QLineEdit("20, 120")
        self.freqs_input = QLineEdit("8, 45")
        self.n_cycles_input = QLineEdit("3.5")
        self.fig_xlim_input = QLineEdit("-100, 300")
        self.n_perms_input = QLineEdit("500")

        form_layout.addRow("Baseline (ms):", self.baseline_input)
        form_layout.addRow("Baseline correction:", self.baseline_mode_combo)
        form_layout.addRow("Response Window (ms):", self.response_input)
        form_layout.addRow("Frequencies (Hz):", self.freqs_input)
        form_layout.addRow("Morlet Cycles:", self.n_cycles_input)
        form_layout.addRow("Plot Time Range (ms):", self.fig_xlim_input)
        form_layout.addRow("Permutations:", self.n_perms_input)

        layout.addLayout(form_layout)
        layout.addStretch()
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.run_button)
        return group

    def _setup_connections(self):
        self.roi_button.clicked.connect(self.select_roi)
        self.run_button.clicked.connect(self.run_analysis)
        self.event_selector.currentIndexChanged.connect(self._on_event_changed)

    def _on_event_changed(self):
        self.figure.clear()
        self.figure.suptitle(f"{self.event_selector.selected_label()} - Run analysis to update")
        self.canvas.draw()

    @Slot()
    def select_roi(self):
        try:
            dialog = SensorSelectionDialog(self.epochs)
            if dialog.exec():
                self.roi_selection_list = dialog.get_selected_channels()
                self.channels_input.setText(" ".join(self.roi_selection_list))
        except Exception as e:
            logger.error(f"Could not open sensor plot: {e}", exc_info=True)
            QMessageBox.warning(
                self, "Plotting Error", "Could not display sensor selection plot."
            )

    def run_analysis(self):
        """Parses parameters and runs the calculation in a background thread."""
        self.roi = self.channels_input.text().split()
        if not self.roi:
            QMessageBox.critical(
                self, "ROI Error", "Please select a valid region of interest."
            )
            return

        freq_parts = list(map(float, self.freqs_input.text().split(",")))
        self.freqs = np.arange(freq_parts[0], freq_parts[1] + 1)
        self.n_cycles = float(self.n_cycles_input.text())
        self.baseline = parse_tuple(self.baseline_input.text(), scale=1e-3)
        self.response = parse_tuple(self.response_input.text(), scale=1e-3)
        self.fig_xlim = parse_tuple(self.fig_xlim_input.text(), scale=1e-3)
        self.n_permutations = int(self.n_perms_input.text())
        self.baseline_mode = (None if self.baseline_mode_combo.currentText() == "None"
                              else self.baseline_mode_combo.currentText())

        if (
            len(self.baseline) != 2
            or len(self.response) != 2
            or len(self.fig_xlim) != 2
        ):
            raise ValueError("Time windows must contain two numbers.")

        try:
            self.run_button.setEnabled(False)
            self.run_button.setText("Running...")
            self.event_selector.setEnabled(False)
            selected_epochs = self.event_selector.selected_epochs()
            self.analysis_event_label = self.event_selector.selected_label()

            worker = Worker(
                lambda: self._update_plot(selected_epochs, self.figure), parent=self
            )
            worker.exec_with_dialog("Loading", "Calculating Natural Frequency...")
            self._on_calculation_finished()

        except Exception as e:
            logger.error(f"An error occurred during calculation: {e}", exc_info=True)
            self.run_button.setEnabled(True)
            self.run_button.setText("Run Analysis")
        finally:
            self.event_selector.setEnabled(self.event_selector.count() > 1)

    def _update_plot(self, epochs, figure):
        """Updates the plot with the results from the calculation."""
        data = calculate_nf_data(
            epochs,
            self.roi,
            self.freqs,
            self.n_cycles,
            self.baseline,
            self.response,
            self.n_permutations,
            self.baseline_mode,
        )

        nat_freq = data["nat_freq"]
        times = data["times"]
        freqs = data["freqs"]
        roi = data["roi"]
        avg_tfr = data["avg_tfr"].mean(0)
        stat_mask = data["stat_mask"]
        agg_freq = data["agg_freq"]

        figure.clear()
        gs = figure.add_gridspec(
            2, 2, width_ratios=(10, 3), height_ratios=(1, 3), hspace=0.05, wspace=0.05
        )
        ax_erp = figure.add_subplot(gs[0, 0])
        ax_main = figure.add_subplot(gs[1, 0], sharex=ax_erp)
        ax_power = figure.add_subplot(gs[1, 1], sharey=ax_main)

        title = (
            f"Natural Frequency: {nat_freq:.1f} Hz"
            if not np.isnan(nat_freq)
            else "Natural Frequency: N/A"
        )
        figure.suptitle(f"{title} - {self.analysis_event_label}", fontsize=14)

        # --- TEP Plot (Top) ---
        ax_erp.plot(times, data["evoked_all"].T * 1e6, color="black", alpha=0.3, lw=1.5)
        for i, ch_data in enumerate(data["evoked_roi"]):
            ax_erp.plot(times, ch_data * 1e6, label=roi[i], lw=2)
        ax_erp.axvline(0, color=plt.rcParams["text.color"], linestyle="--", lw=1)
        ax_erp.set_ylabel("Amplitude (µV)")
        ax_erp.legend(
            loc="upper right",
            fontsize="small",
            ncol=len(roi) // 2 if len(roi) > 2 else 1,
        )
        ax_erp.tick_params(axis="x", labelbottom=False)
        ax_erp.spines[["top", "right", "bottom"]].set_visible(False)
        ax_erp.set_xlim(self.fig_xlim)
        ax_erp.grid(True, linestyle=":")

        # --- Main TFR Plot (Center) ---
        mask_sum = stat_mask.sum(axis=0)
        max_count = np.max(mask_sum)
        if max_count > 0:
            alpha = mask_sum / max_count
            alpha = 0.5 + 0.5 * alpha  # Scale to be between 0.5 and 1.0
            alpha[mask_sum == 0] = 0.25  # Set non-significant to 0.25
        else:
            alpha = np.full_like(mask_sum, 0.25, dtype=float)

        t_idx_plot = epochs.time_as_index(self.fig_xlim)
        vmin, vmax = np.quantile(
            avg_tfr[:, t_idx_plot[0] : t_idx_plot[1]], [0.05, 0.995]
        )
        im = ax_main.imshow(
            avg_tfr,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            alpha=alpha,
        )
        ax_main.axvline(0, color=plt.rcParams["text.color"], linestyle="--", lw=1)
        if not np.isnan(nat_freq):
            ax_main.axhline(nat_freq, color="r", linestyle="--", lw=1)
        ax_main.axvspan(*self.response, color="r", alpha=0.1)
        ax_main.set_xlabel("Time (s)")
        ax_main.set_ylabel("Frequency (Hz)")

        # --- Power Spectrum Plot (Right) ---
        ax_power.plot(agg_freq, freqs, color=plt.rcParams["text.color"], lw=1.5)
        if not np.isnan(nat_freq):
            ax_power.axhline(nat_freq, color="r", linestyle="--", lw=1)
            ax_power.text(
                ax_power.get_xlim()[1] * 0.1,
                nat_freq,
                f" {nat_freq:.1f} Hz",
                color="r",
                va="bottom",
                ha="left",
            )
        ax_power.spines[["top", "right", "left"]].set_visible(False)
        ax_power.tick_params(axis="y", labelleft=False)
        ax_power.set_xlabel("Power", fontsize="small")
        ax_power.grid(True, axis="x", linestyle=":")

        divider = make_axes_locatable(ax_power)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        mode = self.baseline_mode or "uncorrected"
        figure.colorbar(im, cax=cax, label=f"Power ({mode})")

    def _on_calculation_finished(self):
        logger.info("Calculation finished, received result. Now updating plot.")
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Analysis")
        self.canvas.draw()

    def closeEvent(self, event):
        """Ensure the MNE plot is closed when this dialog closes."""
        super().closeEvent(event)


if __name__ == "__main__":
    from pathlib import Path
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    data_path = Path("C:/Users/Bruno Couto/Documents/TEST DATASET")  # Example path
    files = sorted(list(data_path.rglob("*M1_preprocessed-epo.fif")))
    epochs = mne.read_epochs(files[0])
    ex = NaturalFrequencyApp(epochs)
    sys.exit(app.exec())
