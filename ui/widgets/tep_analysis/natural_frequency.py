import mne
import numpy as np
import logging
import traceback
from typing import List, Dict
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection
from PySide6.QtWidgets import (
    QDialog,
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
from utils import use_plot_style, Worker, parse_tuple

logger = logging.getLogger(__name__)


# --- Statistical Functions ---
def bootstr2(
    tf_data: np.ndarray,
    basesamps: np.ndarray,
    nperm: int = 1000,
    ch_name: str = None,
    verbose=True,
) -> np.ndarray:
    """bootstraps by shuffling temporal samples."""
    n_trials, n_freqs, _ = np.shape(tf_data)
    boots = np.zeros((nperm, n_freqs, 2))
    x = np.zeros([n_trials, n_freqs, len(basesamps)])

    pbar = tqdm(
        range(nperm),
        total=nperm,
        desc=f"Running permutations{' for ' + ch_name if ch_name is not None else ''}...",
        disable=not verbose,
    )
    for n in pbar:
        for t in range(n_trials):
            np.random.shuffle(basesamps)
            x[t, :, :] = tf_data[t, :, basesamps].T

        boots[n, :, 0] = np.max((np.mean(x, axis=0)), axis=1)
        boots[n, :, 1] = np.min((np.mean(x, axis=0)), axis=1)

    for f in range(n_freqs):
        boots[:, f, 0] = np.sort(boots[:, f, 0])
        boots[:, f, 1] = np.sort(boots[:, f, 1])

    return boots


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
    nTrials, nFreq, nTimes = TF_trials.shape
    TF_avg = np.mean(TF_trials, axis=0)
    baseline_idxs = np.where((times >= baseline[0]) & (times <= baseline[1]))[0]

    boots = bootstr2(
        TF_trials, baseline_idxs, nperm=n_permutations, verbose=verbose, ch_name=ch_name
    )

    pvals = np.zeros_like(TF_avg)
    for f in range(nFreq):
        for t in range(nTimes):
            p1 = np.sum(boots[:, f, 0] >= TF_avg[f, t]) / n_permutations
            p2 = np.sum(boots[:, f, 1] <= TF_avg[f, t]) / n_permutations
            pvals[f, t] = min(p1, p2)

    if fdr_correction:
        rejected, pvals_corrected = fdrcorrection(pvals.ravel(), alpha=alpha / 2)
        pvals = pvals_corrected.reshape(pvals.shape)

    mask = (pvals < alpha).astype(int)
    return mask


def calculate_nf_data(
    epochs: mne.Epochs,
    roi: List[str],
    freqs: np.ndarray,
    n_cycles: float,
    baseline: List[float],
    response: List[float],
    n_permutations: int,
) -> Dict:
    """
    Calculates the natural frequency and returns all data required for plotting.
    MODIFIED to align with the logic of the provided class methods.
    """
    times = epochs.times
    baseline_idx = np.where((times >= baseline[0]) & (times <= baseline[1]))[0]

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
    avg_tfr -= avg_tfr[:, :, baseline_idx].mean(
        axis=-1, keepdims=True
    )  # baseline correction

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

        self.roi_button = QPushButton("Select ROI from 2D Layout")
        self.channels_input = QLineEdit()
        self.channels_input.setPlaceholderText("e.g., C3 C5 FC1 or use button")
        form_layout.addRow(self.roi_button)
        form_layout.addRow("Selected Channels:", self.channels_input)

        self.baseline_input = QLineEdit("-500, -50")
        self.response_input = QLineEdit("20, 120")
        self.freqs_input = QLineEdit("8, 45")
        self.n_cycles_input = QLineEdit("3.5")
        self.fig_xlim_input = QLineEdit("-100, 300")
        self.n_perms_input = QLineEdit("500")

        form_layout.addRow("Baseline (ms):", self.baseline_input)
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

        if (
            len(self.baseline) != 2
            or len(self.response) != 2
            or len(self.fig_xlim) != 2
        ):
            raise ValueError("Time windows must contain two numbers.")

        try:
            self.run_button.setEnabled(False)
            self.run_button.setText("Running...")

            worker = Worker(
                lambda: self._update_plot(self.epochs, self.figure), parent=self
            )
            worker.exec_with_dialog("Loading", "Calculating Natural Frequency...")
            self._on_calculation_finished()

        except Exception as e:
            logger.error(f"An error occurred during calculation: {e}", exc_info=True)
            self.run_button.setEnabled(True)
            self.run_button.setText("Run Analysis")

    def _update_plot(self, epochs, figure):
        """Updates the plot with the results from the calculation."""
        data = calculate_nf_data(
            self.epochs,
            self.roi,
            self.freqs,
            self.n_cycles,
            self.baseline,
            self.response,
            self.n_permutations,
        )

        nat_freq = data["nat_freq"]
        times = data["times"]
        freqs = data["freqs"]
        roi = data["roi"]
        avg_tfr = data["avg_tfr"].mean(0)
        stat_mask = data["stat_mask"]
        agg_freq = data["agg_freq"]

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
        figure.suptitle(title, fontsize=14)

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
        figure.colorbar(im, cax=cax, label="Power Change")

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
