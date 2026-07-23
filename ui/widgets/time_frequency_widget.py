from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RangeSlider
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.dialogs.sensor_selection_dialog import SensorSelectionDialog
from ui.widgets.tools.optional_range_widget import OptionalRangeWidget
from utils import Worker, theme_tokens, update_toolbar_color

logger = logging.getLogger(__name__)

POWER_DB_MODE = "Power (dB)"
RAW_POWER_MODE = "Raw power"
ZSCORE_MODE = "z-score"

def parse_tfr_freqs(text: str, *, method: str = "morlet") -> np.ndarray | str:
    cleaned = (text or "").strip().lower()
    if cleaned == "auto":
        if method != "stockwell":
            raise ValueError("Frequency value 'auto' is only valid for Stockwell TFR.")
        return "auto"
    if not cleaned:
        raise ValueError("Frequencies are required.")

    normalized = cleaned.replace(";", ",").replace(" ", ",")
    if ":" in normalized:
        parts = [part for part in normalized.split(":") if part]
        if len(parts) not in (2, 3):
            raise ValueError("Use start:stop or start:stop:step for frequency ranges.")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 1.0
        if step <= 0:
            raise ValueError("Frequency step must be greater than zero.")
        freqs = np.arange(start, stop + step * 0.5, step, dtype=float)
    else:
        freqs = np.array(
            [float(part) for part in normalized.split(",") if part],
            dtype=float,
        )

    if freqs.size == 0:
        raise ValueError("Frequencies are required.")
    if np.any(freqs <= 0):
        raise ValueError("Frequencies must be greater than zero.")
    if method == "stockwell" and freqs.size != 2:
        raise ValueError("Stockwell frequencies must be 'auto' or exactly two values: fmin, fmax.")
    return freqs


def parse_tfr_numeric_list(text: str) -> float | list[float]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Value is required.")
    normalized = cleaned.replace(";", ",").replace(" ", ",")
    values = [float(part) for part in normalized.split(",") if part]
    if not values:
        raise ValueError("Value is required.")
    return values[0] if len(values) == 1 else values


def parse_tfr_picks(text: str):
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    picks = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not picks:
        return None
    if len(picks) == 1:
        return picks[0]
    return picks


def parse_channel_selection(text: str, available_channels: list[str]) -> list[str] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    if any(separator in cleaned for separator in ",;\n"):
        raw_parts = [
            part.strip()
            for part in cleaned.replace(";", ",").replace("\n", ",").split(",")
            if part.strip()
        ]
    elif cleaned in available_channels:
        raw_parts = [cleaned]
    else:
        raw_parts = [part.strip() for part in cleaned.split() if part.strip()]

    if not raw_parts:
        return None

    available_lookup = {name.lower(): name for name in available_channels}
    selected = []
    invalid = []
    for part in raw_parts:
        channel_name = available_lookup.get(part.lower())
        if channel_name is None:
            invalid.append(part)
        elif channel_name not in selected:
            selected.append(channel_name)

    if invalid:
        raise ValueError(f"Unknown channel(s): {', '.join(invalid)}")
    return selected or None


class ComputeTFRSettingsDialog(QDialog):
    """Collects parameters for ``mne.Epochs.compute_tfr``."""

    VERBOSE_VALUES = {
        "Default": None,
        "True": True,
        "False": False,
        "ERROR": "ERROR",
        "WARNING": "WARNING",
        "INFO": "INFO",
        "DEBUG": "DEBUG",
    }

    def __init__(self, epochs: mne.BaseEpochs, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.setWindowTitle("Compute Time-Frequency")
        self.setModal(True)
        self.setMinimumWidth(660)

        layout = QVBoxLayout(self)
        layout.addWidget(self._create_compute_group())
        layout.addWidget(self._create_method_group())

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Compute")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.method_combo.currentTextChanged.connect(self._sync_method_fields)
        self.output_combo.currentTextChanged.connect(self._sync_output_constraints)
        self.average_checkbox.toggled.connect(self._sync_output_constraints)
        self.return_itc_checkbox.toggled.connect(self._sync_output_constraints)
        self._sync_method_fields()
        self._sync_output_constraints()

    def _create_compute_group(self) -> QGroupBox:
        group = QGroupBox("mne.Epochs.compute_tfr")
        form = QFormLayout(group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["morlet", "multitaper", "stockwell"])

        self.freqs_input = QLineEdit("8:45:1")
        self.freqs_input.setPlaceholderText("8:45:1, 8,10,12, or auto for Stockwell")

        times = np.asarray(self.epochs.times, dtype=float)
        self.time_range_input = OptionalRangeWidget(
            labels=("tmin:", "tmax:"),
            suffix=" s",
            range=(float(times[0]), float(times[-1])) if times.size else (None, None),
            parent=self,
        )
        for spinbox in (self.time_range_input.low_input, self.time_range_input.high_input):
            spinbox.setDecimals(4)
            spinbox.setSingleStep(0.01)
            if times.size:
                spinbox.setRange(float(times[0]), float(times[-1]))
        self.time_range_input.adjust_spinbox_width()

        self.picks_input = QLineEdit()
        self.picks_input.setPlaceholderText("Blank for good data channels, or e.g. eeg, Cz, Pz")

        self.proj_checkbox = QCheckBox("Apply SSP projections")
        self.output_combo = QComboBox()
        self.output_combo.addItems(["power", "complex", "phase"])
        self.average_checkbox = QCheckBox("Average across epochs")
        self.return_itc_checkbox = QCheckBox("Return ITC")

        self.decim_input = QSpinBox()
        self.decim_input.setRange(1, 100000)
        self.decim_input.setValue(1)

        self.n_jobs_checkbox = QCheckBox("Set")
        self.n_jobs_input = QSpinBox()
        self.n_jobs_input.setRange(-1, 256)
        self.n_jobs_input.setValue(1)
        n_jobs_row = QWidget()
        n_jobs_layout = QHBoxLayout(n_jobs_row)
        n_jobs_layout.setContentsMargins(0, 0, 0, 0)
        n_jobs_layout.addWidget(self.n_jobs_checkbox)
        n_jobs_layout.addWidget(self.n_jobs_input)
        n_jobs_layout.addStretch()
        self.n_jobs_input.setEnabled(False)
        self.n_jobs_checkbox.toggled.connect(self.n_jobs_input.setEnabled)

        self.verbose_combo = QComboBox()
        self.verbose_combo.addItems(list(self.VERBOSE_VALUES.keys()))

        form.addRow("method:", self.method_combo)
        form.addRow("freqs:", self.freqs_input)
        form.addRow("time window:", self.time_range_input)
        form.addRow("picks:", self.picks_input)
        form.addRow(self.proj_checkbox)
        form.addRow("output:", self.output_combo)
        form.addRow(self.average_checkbox)
        form.addRow(self.return_itc_checkbox)
        form.addRow("decim:", self.decim_input)
        form.addRow("n_jobs:", n_jobs_row)
        form.addRow("verbose:", self.verbose_combo)
        return group

    def _create_method_group(self) -> QGroupBox:
        group = QGroupBox("Method Keywords")
        form = QFormLayout(group)

        self.n_cycles_input = QLineEdit("3.5")
        self.n_cycles_input.setPlaceholderText("Scalar or comma-separated values")
        self.use_fft_checkbox = QCheckBox("use_fft")
        self.use_fft_checkbox.setChecked(True)
        self.zero_mean_checkbox = QCheckBox("zero_mean")
        self.zero_mean_checkbox.setChecked(True)
        self.time_bandwidth_input = QDoubleSpinBox()
        self.time_bandwidth_input.setRange(0.5, 100.0)
        self.time_bandwidth_input.setDecimals(2)
        self.time_bandwidth_input.setValue(4.0)
        self.n_fft_checkbox = QCheckBox("Set n_fft")
        self.n_fft_input = QSpinBox()
        self.n_fft_input.setRange(1, 10_000_000)
        self.n_fft_input.setEnabled(False)
        self.n_fft_checkbox.toggled.connect(self.n_fft_input.setEnabled)
        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(0.01, 100.0)
        self.width_input.setDecimals(2)
        self.width_input.setValue(1.0)

        self.n_cycles_label = QLabel("n_cycles:")
        self.use_fft_label = QLabel("")
        self.zero_mean_label = QLabel("")
        self.time_bandwidth_label = QLabel("time_bandwidth:")
        self.n_fft_label = QLabel("n_fft:")
        self.width_label = QLabel("width:")

        self.n_fft_row = QWidget()
        n_fft_layout = QHBoxLayout(self.n_fft_row)
        n_fft_layout.setContentsMargins(0, 0, 0, 0)
        n_fft_layout.addWidget(self.n_fft_checkbox)
        n_fft_layout.addWidget(self.n_fft_input)
        n_fft_layout.addStretch()

        form.addRow(self.n_cycles_label, self.n_cycles_input)
        form.addRow(self.use_fft_label, self.use_fft_checkbox)
        form.addRow(self.zero_mean_label, self.zero_mean_checkbox)
        form.addRow(self.time_bandwidth_label, self.time_bandwidth_input)
        form.addRow(self.n_fft_label, self.n_fft_row)
        form.addRow(self.width_label, self.width_input)
        return group

    def _set_row_visible(self, label: QLabel, widget: QWidget, visible: bool):
        label.setVisible(visible)
        widget.setVisible(visible)

    def _sync_method_fields(self):
        method = self.method_combo.currentText()
        is_stockwell = method == "stockwell"
        is_multitaper = method == "multitaper"
        self._set_row_visible(self.n_cycles_label, self.n_cycles_input, not is_stockwell)
        self._set_row_visible(self.use_fft_label, self.use_fft_checkbox, not is_stockwell)
        self._set_row_visible(self.zero_mean_label, self.zero_mean_checkbox, not is_stockwell)
        self._set_row_visible(self.time_bandwidth_label, self.time_bandwidth_input, is_multitaper)
        self._set_row_visible(self.n_fft_label, self.n_fft_row, is_stockwell)
        self._set_row_visible(self.width_label, self.width_input, is_stockwell)
        if is_stockwell and self.freqs_input.text().strip() == "8:45:1":
            self.freqs_input.setText("8, 45")
            self.average_checkbox.setChecked(True)
        elif not is_stockwell and self.freqs_input.text().strip().lower() == "auto":
            self.freqs_input.setText("8:45:1")

    def _sync_output_constraints(self):
        if self.return_itc_checkbox.isChecked():
            self.average_checkbox.setChecked(True)
            self.output_combo.setCurrentText("power")
        self.output_combo.setEnabled(not self.return_itc_checkbox.isChecked())

    def _validation_error(self) -> str | None:
        try:
            params = self.get_compute_params()
        except ValueError as exc:
            return str(exc)

        freqs = params["freqs"]
        if isinstance(freqs, np.ndarray):
            nyquist = float(self.epochs.info["sfreq"]) / 2.0
            if np.any(freqs > nyquist):
                return f"Frequencies cannot exceed Nyquist ({nyquist:.2f} Hz)."
            if params["method"] != "stockwell" and freqs.size > 1 and np.any(np.diff(freqs) <= 0):
                return "Frequencies must be strictly increasing."

        tmin = params.get("tmin")
        tmax = params.get("tmax")
        if tmin is not None and tmax is not None and tmin >= tmax:
            return "tmin must be lower than tmax."

        if params["return_itc"] and not params["average"]:
            return "return_itc requires average=True."
        if params["average"] and params["output"] != "power":
            return "average=True is only compatible with output='power'."
        return None

    def accept(self):
        error = self._validation_error()
        if error:
            QMessageBox.warning(self, "Invalid TFR Parameters", error)
            return
        super().accept()

    def get_compute_params(self) -> dict[str, Any]:
        method = self.method_combo.currentText()
        tmin, tmax = self.time_range_input.value()
        params: dict[str, Any] = {
            "method": method,
            "freqs": parse_tfr_freqs(self.freqs_input.text(), method=method),
            "tmin": tmin,
            "tmax": tmax,
            "picks": parse_tfr_picks(self.picks_input.text()),
            "proj": self.proj_checkbox.isChecked(),
            "output": self.output_combo.currentText(),
            "average": self.average_checkbox.isChecked(),
            "return_itc": self.return_itc_checkbox.isChecked(),
            "decim": int(self.decim_input.value()),
            "n_jobs": int(self.n_jobs_input.value()) if self.n_jobs_checkbox.isChecked() else None,
            "verbose": self.VERBOSE_VALUES[self.verbose_combo.currentText()],
        }

        if method in ("morlet", "multitaper"):
            params["n_cycles"] = parse_tfr_numeric_list(self.n_cycles_input.text())
            params["use_fft"] = self.use_fft_checkbox.isChecked()
            params["zero_mean"] = self.zero_mean_checkbox.isChecked()
        if method == "multitaper":
            params["time_bandwidth"] = float(self.time_bandwidth_input.value())
        if method == "stockwell":
            if self.n_fft_checkbox.isChecked():
                params["n_fft"] = int(self.n_fft_input.value())
            params["width"] = float(self.width_input.value())
        return params


@dataclass
class TimeFrequencyDisplayData:
    times_ms: np.ndarray
    freqs_hz: np.ndarray
    values: np.ndarray
    erp_uV: np.ndarray
    erp_channel_uV: np.ndarray
    erp_channel_names: list[str]
    spectrum: np.ndarray
    units: str
    channel_label: str
    event_label: str
    n_epochs: int


def _coerce_tfr(tfr: Any):
    if isinstance(tfr, (list, tuple)):
        return tfr[0] if tfr else None
    return tfr


def _finite_range(values: np.ndarray, fallback: tuple[float, float] = (-1.0, 1.0)) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if math.isclose(lower, upper):
        pad = max(1.0, abs(lower) * 0.1)
        return lower - pad, upper + pad
    return lower, upper


def _percentile_clim(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lower, upper = np.percentile(finite, [2.0, 98.0])
    lower = float(lower)
    upper = float(upper)
    if math.isclose(lower, upper):
        pad = max(1.0, abs(lower) * 0.1)
        return lower - pad, upper + pad
    return lower, upper


def _centers_to_extent(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (0.0, 1.0)
    if values.size == 1:
        center = float(values[0])
        return center - 0.5, center + 0.5
    first_step = float(values[1] - values[0])
    last_step = float(values[-1] - values[-2])
    return float(values[0] - first_step / 2.0), float(values[-1] + last_step / 2.0)


def _baseline_mask(times_s: np.ndarray, baseline_s: tuple[float | None, float | None]) -> np.ndarray:
    start, stop = baseline_s
    start = float(times_s[0]) if start is None else float(start)
    stop = float(times_s[-1]) if stop is None else float(stop)
    if start > stop:
        start, stop = stop, start
    return (times_s >= start) & (times_s <= stop)


def apply_time_frequency_transform(
    power: np.ndarray,
    times_s: np.ndarray,
    *,
    mode: str = POWER_DB_MODE,
    baseline_s: tuple[float | None, float | None] = (None, 0.0),
    apply_baseline: bool = True,
) -> tuple[np.ndarray, str]:
    """Convert raw MNE power values into display-ready spectrogram values."""
    power = np.asarray(power, dtype=float)
    times_s = np.asarray(times_s, dtype=float)
    eps = np.finfo(float).tiny

    if mode == RAW_POWER_MODE:
        values = power * 1e12
        units = "Power (uV^2)"
        if apply_baseline:
            mask = _baseline_mask(times_s, baseline_s)
            if np.any(mask):
                baseline = np.nanmean(values[:, mask], axis=1, keepdims=True)
                values = values - baseline
                units = "Power change (uV^2)"
        return values, units

    mask = _baseline_mask(times_s, baseline_s) if apply_baseline else np.ones(times_s.shape, dtype=bool)
    if not np.any(mask):
        mask = np.ones(times_s.shape, dtype=bool)

    baseline_values = power[:, mask]
    baseline_mean = np.nanmean(baseline_values, axis=1, keepdims=True)
    baseline_mean = np.clip(baseline_mean, eps, None)

    if mode == ZSCORE_MODE:
        baseline_std = np.nanstd(baseline_values, axis=1, keepdims=True)
        baseline_std = np.where(baseline_std <= eps, 1.0, baseline_std)
        return (power - baseline_mean) / baseline_std, "Power (z-score)"

    ratio = np.clip(power, eps, None) / baseline_mean
    return 10.0 * np.log10(ratio), "Power (dB)"


def _data_channel_names(info: mne.Info, restrict_to: list[str] | None = None) -> list[str]:
    picks = mne.pick_types(
        info,
        meg=True,
        eeg=True,
        seeg=True,
        ecog=True,
        fnirs=False,
        emg=False,
        stim=False,
        eog=False,
        misc=False,
        exclude="bads",
    )
    names = [info["ch_names"][index] for index in picks]
    if not names:
        bads = set(info.get("bads", []))
        names = [name for name in info["ch_names"] if name not in bads]
    if restrict_to is not None:
        allowed = set(restrict_to)
        names = [name for name in names if name in allowed]
    return names


def _pick_indices(ch_names: list[str], channel_names: str | list[str] | None) -> list[int]:
    if channel_names is None:
        return list(range(len(ch_names)))
    if isinstance(channel_names, str):
        channel_names = [channel_names]
    missing = [channel_name for channel_name in channel_names if channel_name not in ch_names]
    if missing:
        raise ValueError(f"Channel(s) not present in the time-frequency data: {', '.join(missing)}")
    return [ch_names.index(channel_name) for channel_name in channel_names]


def _event_name_lookup(epochs: mne.BaseEpochs) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for name, code in (epochs.event_id or {}).items():
        lookup.setdefault(int(code), str(name))
    return lookup


def _subset_epochs(epochs: mne.BaseEpochs, event_code: int | None) -> mne.BaseEpochs:
    if event_code is None or epochs.events is None:
        return epochs
    indices = np.flatnonzero(epochs.events[:, 2] == int(event_code))
    return epochs[indices]


def _extract_tfr_power(
    tfr: Any,
    *,
    channel_names: str | list[str] | None,
    event_code: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    tfr = _coerce_tfr(tfr)
    if tfr is None:
        raise ValueError("No time-frequency data is loaded.")

    data = np.asarray(tfr.data, dtype=float)
    ch_names = list(tfr.ch_names)
    pick_indices = _pick_indices(ch_names, channel_names)
    n_epochs = int(getattr(tfr, "nave", 1) or 1)

    if data.ndim == 4:
        if event_code is not None and getattr(tfr, "events", None) is not None:
            event_mask = np.asarray(tfr.events)[:, 2] == int(event_code)
            data = data[event_mask]
        n_epochs = int(data.shape[0])
        if n_epochs == 0:
            raise ValueError("No epochs match the selected event.")
        matrix = np.nanmean(data[:, pick_indices, :, :], axis=(0, 1))
    elif data.ndim == 3:
        matrix = np.nanmean(data[pick_indices, :, :], axis=0)
    else:
        raise ValueError(f"Unsupported TFR data shape: {data.shape}")

    if channel_names is None:
        channel_label = "All data channels"
    elif isinstance(channel_names, str):
        channel_label = channel_names
    elif len(channel_names) == 1:
        channel_label = channel_names[0]
    else:
        channel_label = f"{len(channel_names)} selected channels"
    return np.asarray(tfr.times, dtype=float), np.asarray(tfr.freqs, dtype=float), matrix, n_epochs, channel_label


def _erp_traces_uV(
    epochs: mne.BaseEpochs,
    *,
    channel_names: str | list[str] | None,
    event_code: int | None,
) -> tuple[list[str], np.ndarray]:
    selected_epochs = _subset_epochs(epochs, event_code)
    if len(selected_epochs) == 0:
        if channel_names is None:
            selected_names = _data_channel_names(epochs.info)
        elif isinstance(channel_names, str):
            selected_names = [channel_names]
        else:
            selected_names = list(channel_names)
        return selected_names, np.full((len(selected_names), len(epochs.times)), np.nan, dtype=float)

    evoked = selected_epochs.average()
    data_channel_names = _data_channel_names(evoked.info)
    if channel_names is not None:
        pick_indices = _pick_indices(evoked.ch_names, channel_names)
    else:
        pick_indices = [
            evoked.ch_names.index(name) for name in data_channel_names if name in evoked.ch_names
        ]
    if not pick_indices:
        pick_indices = list(range(len(evoked.ch_names)))
    selected_names = [evoked.ch_names[index] for index in pick_indices]
    return selected_names, evoked.data[pick_indices, :] * 1e6


def _spatial_trace_colors(info: mne.Info, channel_names: list[str]) -> list:
    if not channel_names:
        return []

    picks = []
    for channel_name in channel_names:
        try:
            picks.append(info["ch_names"].index(channel_name))
        except ValueError:
            break

    if len(picks) == len(channel_names) and len(picks) > 1:
        locs = np.array([info["chs"][index]["loc"][:3] for index in picks], dtype=float)
        if (
            locs.shape == (len(channel_names), 3)
            and np.all(np.isfinite(locs))
            and np.any(np.abs(locs) > 0.0)
            and np.any(np.ptp(locs, axis=0) > 0.0)
        ):
            rgb = locs.copy()
            rgb -= np.nanmin(rgb, axis=0)
            rgb /= np.maximum(np.nanmax(rgb, axis=0), 1e-16)
            rgb[rgb.sum(axis=1) > 2.5] = rgb[rgb.sum(axis=1) > 2.5] - 0.3
            return [tuple(np.clip(color, 0.0, 1.0)) for color in rgb]

    cmap = colormaps.get_cmap("tab20")
    return [cmap(index % cmap.N) for index in range(len(channel_names))]


def build_time_frequency_display(
    *,
    epochs: mne.BaseEpochs,
    tfr: Any,
    channel_name: str | list[str] | None = None,
    channel_names: str | list[str] | None = None,
    event_code: int | None = None,
    transform_mode: str = POWER_DB_MODE,
    baseline_s: tuple[float | None, float | None] = (None, 0.0),
    apply_baseline: bool = True,
) -> TimeFrequencyDisplayData:
    selected_channels = channel_names if channel_names is not None else channel_name
    times_s, freqs_hz, power, n_epochs, channel_label = _extract_tfr_power(
        tfr,
        channel_names=selected_channels,
        event_code=event_code,
    )
    values, units = apply_time_frequency_transform(
        power,
        times_s,
        mode=transform_mode,
        baseline_s=baseline_s,
        apply_baseline=apply_baseline,
    )
    event_lookup = _event_name_lookup(epochs)
    event_label = event_lookup.get(int(event_code), f"Event {event_code}") if event_code is not None else "All events"
    erp_channel_names, erp_full_uV = _erp_traces_uV(
        epochs,
        channel_names=selected_channels,
        event_code=event_code,
    )
    if erp_full_uV.size:
        erp_channel_uV = np.vstack(
            [np.interp(times_s, epochs.times, trace_uV) for trace_uV in erp_full_uV]
        )
        erp_uV = np.nanmean(erp_channel_uV, axis=0)
    else:
        erp_channel_uV = np.empty((0, len(times_s)), dtype=float)
        erp_uV = np.full_like(times_s, np.nan, dtype=float)
    spectrum = np.nanmean(values, axis=1)
    return TimeFrequencyDisplayData(
        times_ms=times_s * 1000.0,
        freqs_hz=freqs_hz,
        values=values,
        erp_uV=erp_uV,
        erp_channel_uV=erp_channel_uV,
        erp_channel_names=erp_channel_names,
        spectrum=spectrum,
        units=units,
        channel_label=channel_label,
        event_label=event_label,
        n_epochs=n_epochs,
    )


class TimeFrequencyWidget(QWidget):
    timeWindowChanged = Signal(float, float)

    def __init__(
        self,
        epochs: mne.BaseEpochs,
        *,
        label: str | None = None,
        tfr: Any = None,
        tfr_derivatives: dict[str, Any] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.epochs = epochs
        self.label = label or "Time-Frequency"
        self.tfr_derivatives = self._normalize_tfr_derivatives(tfr_derivatives, tfr)
        self.current_derivative_key = next(iter(self.tfr_derivatives), None)
        self.tfr = self.tfr_derivatives.get(self.current_derivative_key)
        self.display_data: TimeFrequencyDisplayData | None = None
        self._clim: tuple[float, float] | None = None
        self._full_clim: tuple[float, float] = (-1.0, 1.0)
        self._slider: RangeSlider | None = None
        self._slider_cid = None
        self._xlim_callback_cid = None
        self._suppress_xlim_signal = False
        self._canvas_motion_cid = None
        self._canvas_leave_cid = None
        self._canvas_press_cid = None
        self._canvas_release_cid = None
        self._roi_drag_start: tuple[float, float] | None = None
        self._roi_drag_active = False
        self._updating_roi_controls = False
        self.roi_time_ms: tuple[float, float] | None = None
        self.roi_freq_hz: tuple[float, float] | None = None

        self.main_ax = None
        self.erp_ax = None
        self.spectrum_ax = None
        self.time_power_ax = None
        self.colorbar_ax = None
        self.slider_ax = None
        self.image_artist = None
        self.spectrum_line = None
        self.time_power_line = None
        self.erp_channel_lines = []
        self.erp_average_line = None
        self.roi_rectangle: Rectangle | None = None
        self.roi_time_span = None
        self.crosshair_vline = None
        self.crosshair_hline = None
        self.tooltip = None
        self.erp_hover_vline = None
        self.erp_hover_hline = None
        self.time_power_hover_vline = None
        self.time_power_hover_hline = None
        self.spectrum_hover_vline = None
        self.spectrum_hover_hline = None
        self.erp_tooltip = None
        self.time_power_tooltip = None
        self.spectrum_tooltip = None

        self._init_ui()
        self._connect_events()
        self._populate_selectors()
        self._configure_time_spinboxes()
        self._render()

    @staticmethod
    def _normalize_tfr_derivatives(
        tfr_derivatives: dict[str, Any] | None,
        tfr: Any = None,
    ) -> dict[str, Any]:
        derivatives: dict[str, Any] = {}
        for label, derivative in (tfr_derivatives or {}).items():
            coerced = _coerce_tfr(derivative)
            if coerced is not None:
                derivatives[str(label)] = coerced
        if not derivatives:
            coerced = _coerce_tfr(tfr)
            if coerced is not None:
                derivatives["TFR"] = coerced
        return derivatives

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        self.header_frame = QFrame()
        self.header_frame.setObjectName("tfHeader")
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(8)

        self.title_label = QLabel(self.label)
        self.title_label.setObjectName("panelSubtitle")
        self.context_label = QLabel("")
        self.context_label.setObjectName("mutedLabel")
        self.context_label.setWordWrap(True)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.context_label, 1)

        self.controls_toggle_button = QPushButton("Controls")
        self.controls_toggle_button.setCheckable(True)
        self.controls_toggle_button.setChecked(True)
        self.controls_toggle_button.setToolTip("Show or hide the controls pane")
        header_layout.addWidget(self.controls_toggle_button)
        root_layout.addWidget(self.header_frame)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.splitter, 1)

        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        update_toolbar_color(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.toolbar)
        self.splitter.addWidget(plot_frame)

        self.settings_panel = self._create_settings_panel()
        self.splitter.addWidget(self.settings_panel)
        self.splitter.setSizes([980, 290])

    def _create_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(290)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(10)

        display_group = QGroupBox("Display")
        display_form = QFormLayout(display_group)
        self.derivative_combo = QComboBox()
        self.channels_input = QLineEdit()
        self.channels_input.setPlaceholderText("Blank for all, or e.g. Cz, Pz")
        self.channel_picker_button = QPushButton("Select...")
        channel_row = QWidget()
        channel_layout = QHBoxLayout(channel_row)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(6)
        channel_layout.addWidget(self.channels_input, 1)
        channel_layout.addWidget(self.channel_picker_button)
        self.event_combo = QComboBox()
        self.transform_combo = QComboBox()
        self.transform_combo.addItems([POWER_DB_MODE, RAW_POWER_MODE, ZSCORE_MODE])
        display_form.addRow("Derivative:", self.derivative_combo)
        display_form.addRow("Channels:", channel_row)
        display_form.addRow("Event:", self.event_combo)
        display_form.addRow("Scale:", self.transform_combo)

        self.baseline_checkbox = QCheckBox("Baseline correction")
        self.baseline_checkbox.setChecked(True)
        self.baseline_start_input = QDoubleSpinBox()
        self.baseline_stop_input = QDoubleSpinBox()
        for spinbox in (self.baseline_start_input, self.baseline_stop_input):
            spinbox.setDecimals(1)
            spinbox.setSingleStep(10.0)
            spinbox.setSuffix(" ms")
        display_form.addRow(self.baseline_checkbox)
        display_form.addRow("Baseline start:", self.baseline_start_input)
        display_form.addRow("Baseline stop:", self.baseline_stop_input)
        layout.addWidget(display_group)

        color_group = QGroupBox("Color Limits")
        color_form = QFormLayout(color_group)
        self.clim_min_input = QDoubleSpinBox()
        self.clim_max_input = QDoubleSpinBox()
        for spinbox in (self.clim_min_input, self.clim_max_input):
            spinbox.setDecimals(3)
            spinbox.setRange(-1e9, 1e9)
            spinbox.setSingleStep(0.25)
        self.auto_clim_button = QPushButton("Auto CLim")
        color_form.addRow("Min:", self.clim_min_input)
        color_form.addRow("Max:", self.clim_max_input)
        color_form.addRow(self.auto_clim_button)
        layout.addWidget(color_group)

        roi_group = QGroupBox("ROI")
        roi_form = QFormLayout(roi_group)
        self.roi_time_start_input = QDoubleSpinBox()
        self.roi_time_stop_input = QDoubleSpinBox()
        self.roi_freq_low_input = QDoubleSpinBox()
        self.roi_freq_high_input = QDoubleSpinBox()
        for spinbox in (self.roi_time_start_input, self.roi_time_stop_input):
            spinbox.setDecimals(1)
            spinbox.setSingleStep(5.0)
            spinbox.setSuffix(" ms")
            spinbox.setRange(-1e9, 1e9)
        for spinbox in (self.roi_freq_low_input, self.roi_freq_high_input):
            spinbox.setDecimals(2)
            spinbox.setSingleStep(1.0)
            spinbox.setSuffix(" Hz")
            spinbox.setRange(0.0, 1e9)
        self.roi_mean_label = QLabel("Mean: n/a")
        self.roi_mean_label.setObjectName("mutedLabel")
        roi_form.addRow("Time start:", self.roi_time_start_input)
        roi_form.addRow("Time stop:", self.roi_time_stop_input)
        roi_form.addRow("Freq low:", self.roi_freq_low_input)
        roi_form.addRow("Freq high:", self.roi_freq_high_input)
        roi_form.addRow(self.roi_mean_label)
        layout.addWidget(roi_group)

        layout.addStretch()
        return panel

    def _connect_events(self):
        self.controls_toggle_button.toggled.connect(self._set_controls_visible)
        self.derivative_combo.currentIndexChanged.connect(self._on_derivative_changed)
        self.channels_input.editingFinished.connect(self._on_display_setting_changed)
        self.channel_picker_button.clicked.connect(self._open_channel_picker)
        self.event_combo.currentIndexChanged.connect(self._on_display_setting_changed)
        self.transform_combo.currentIndexChanged.connect(self._on_transform_changed)
        self.baseline_checkbox.toggled.connect(self._on_transform_changed)
        self.baseline_start_input.valueChanged.connect(self._on_transform_changed)
        self.baseline_stop_input.valueChanged.connect(self._on_transform_changed)
        self.clim_min_input.valueChanged.connect(self._on_clim_spinbox_changed)
        self.clim_max_input.valueChanged.connect(self._on_clim_spinbox_changed)
        self.auto_clim_button.clicked.connect(self._auto_scale_clim)
        for spinbox in (
            self.roi_time_start_input,
            self.roi_time_stop_input,
            self.roi_freq_low_input,
            self.roi_freq_high_input,
        ):
            spinbox.valueChanged.connect(self._on_roi_control_changed)
        self._canvas_press_cid = self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self._canvas_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas_release_cid = self.canvas.mpl_connect("button_release_event", self._on_button_release)
        self._canvas_leave_cid = self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)

    def _set_controls_visible(self, visible: bool):
        self.settings_panel.setVisible(bool(visible))
        if visible:
            self.splitter.setSizes([980, 290])

    def _populate_selectors(self):
        self._populate_derivative_selector()
        self._populate_channel_selector()
        self._populate_event_selector()

    def _populate_derivative_selector(self):
        current_key = self.current_derivative_key
        if current_key not in self.tfr_derivatives:
            current_key = next(iter(self.tfr_derivatives), None)

        self.derivative_combo.blockSignals(True)
        self.derivative_combo.clear()
        selected_index = 0
        for index, label in enumerate(self.tfr_derivatives):
            self.derivative_combo.addItem(label, label)
            if label == current_key:
                selected_index = index
        if self.derivative_combo.count() > 0:
            self.derivative_combo.setCurrentIndex(selected_index)
            self.current_derivative_key = self.derivative_combo.currentData()
            self.tfr = self.tfr_derivatives.get(self.current_derivative_key)
        else:
            self.current_derivative_key = None
            self.tfr = None
        self.derivative_combo.setEnabled(self.derivative_combo.count() > 1)
        self.derivative_combo.blockSignals(False)

    def _populate_channel_selector(self):
        available_channels = self._available_channel_names()
        self.channels_input.setPlaceholderText(
            "Blank for all, or e.g. " + ", ".join(available_channels[:2])
            if available_channels
            else "Blank for all channels"
        )

    def _populate_event_selector(self):
        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        self.event_combo.addItem("All events", None)
        if self.epochs.events is not None and len(self.epochs.events) > 0:
            names_by_code = _event_name_lookup(self.epochs)
            for event_code in sorted({int(code) for code in self.epochs.events[:, 2]}):
                count = int(np.sum(self.epochs.events[:, 2] == event_code))
                label = names_by_code.get(event_code, f"Event {event_code}")
                self.event_combo.addItem(f"{label} ({count})", event_code)
        self.event_combo.setVisible(self.event_combo.count() > 2)
        self.event_combo.blockSignals(False)

    def _configure_time_spinboxes(self):
        times_ms = np.asarray(self.epochs.times, dtype=float) * 1000.0
        if times_ms.size == 0:
            return
        tmin = float(times_ms[0])
        tmax = float(times_ms[-1])
        for spinbox in (self.baseline_start_input, self.baseline_stop_input):
            spinbox.setRange(tmin, tmax)

        baseline_stop = 0.0 if tmin <= 0.0 <= tmax else min(tmax, tmin + (tmax - tmin) * 0.25)
        baseline_start = max(tmin, baseline_stop - max(20.0, min(200.0, (tmax - tmin) * 0.2)))
        if baseline_start >= baseline_stop:
            baseline_start = tmin
            baseline_stop = min(tmax, tmin + max(1.0, (tmax - tmin) * 0.25))
        self.baseline_start_input.setValue(baseline_start)
        self.baseline_stop_input.setValue(baseline_stop)

    def _available_channel_names(self) -> list[str]:
        tfr_channels = list(getattr(self.tfr, "ch_names", [])) if self.tfr is not None else None
        channel_names = _data_channel_names(self.epochs.info, restrict_to=tfr_channels)
        if not channel_names and tfr_channels:
            channel_names = tfr_channels
        return channel_names

    def _selected_channels(self) -> list[str] | None:
        return parse_channel_selection(self.channels_input.text(), self._available_channel_names())

    def _set_selected_channels(self, channel_names: list[str] | None):
        self.channels_input.setText(", ".join(channel_names or []))
        self._on_display_setting_changed()

    def _open_channel_picker(self):
        available_channels = self._available_channel_names()
        if not available_channels:
            QMessageBox.warning(self, "No Channels", "No plottable channels are available for this TFR.")
            return
        try:
            picker_epochs = self.epochs.copy().pick(available_channels)
            dialog = SensorSelectionDialog(picker_epochs, parent=self)
        except Exception as exc:
            logger.exception("Could not open sensor selection dialog.")
            QMessageBox.warning(self, "Sensor Selection Failed", str(exc))
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected_channels = dialog.get_selected_channels()
        if selected_channels:
            self._set_selected_channels(selected_channels)

    def _selected_event_code(self) -> int | None:
        value = self.event_combo.currentData()
        return None if value is None else int(value)

    def _baseline_s(self) -> tuple[float | None, float | None]:
        return (self.baseline_start_input.value() / 1000.0, self.baseline_stop_input.value() / 1000.0)

    def _on_derivative_changed(self):
        self.current_derivative_key = self.derivative_combo.currentData()
        self.tfr = self.tfr_derivatives.get(self.current_derivative_key)
        self._populate_channel_selector()
        self._clim = None
        self._render()

    def _on_display_setting_changed(self):
        self._clim = None
        self._render()

    def _on_transform_changed(self):
        self._clim = None
        self._render()

    def _on_clim_spinbox_changed(self):
        if self.display_data is None:
            return
        lower = float(self.clim_min_input.value())
        upper = float(self.clim_max_input.value())
        if lower >= upper:
            return
        self._set_clim((lower, upper), update_spinboxes=False, update_slider=True)

    def _auto_scale_clim(self):
        if self.display_data is None:
            return
        self._set_clim(_percentile_clim(self.display_data.values), update_spinboxes=True, update_slider=True)

    def _set_clim(
        self,
        clim: tuple[float, float],
        *,
        update_spinboxes: bool,
        update_slider: bool,
    ):
        lower, upper = sorted((float(clim[0]), float(clim[1])))
        if math.isclose(lower, upper):
            upper = lower + 1.0
        self._clim = (lower, upper)
        if self.image_artist is not None:
            self.image_artist.set_clim(lower, upper)
        if update_spinboxes:
            self._set_clim_spinbox_values(lower, upper)
        if update_slider and self._slider is not None:
            try:
                self._slider.set_val((lower, upper))
            except Exception:
                logger.debug("Could not synchronize matplotlib range slider.", exc_info=True)
        self.canvas.draw_idle()

    def _set_clim_spinbox_values(self, lower: float, upper: float):
        self.clim_min_input.blockSignals(True)
        self.clim_max_input.blockSignals(True)
        self.clim_min_input.setValue(lower)
        self.clim_max_input.setValue(upper)
        self.clim_min_input.blockSignals(False)
        self.clim_max_input.blockSignals(False)

    def _render(self):
        self.figure.clear()
        self._slider = None
        self._slider_cid = None
        self.image_artist = None
        self.spectrum_line = None
        self.time_power_line = None
        self.erp_channel_lines = []
        self.erp_average_line = None
        self.roi_rectangle = None
        self.roi_time_span = None
        self.crosshair_vline = None
        self.crosshair_hline = None
        self.tooltip = None
        self.erp_hover_vline = None
        self.erp_hover_hline = None
        self.time_power_hover_vline = None
        self.time_power_hover_hline = None
        self.spectrum_hover_vline = None
        self.spectrum_hover_hline = None
        self.erp_tooltip = None
        self.time_power_tooltip = None
        self.spectrum_tooltip = None

        if self.tfr is None:
            self._render_empty_state("No time-frequency data loaded.")
            return

        try:
            self.display_data = build_time_frequency_display(
                epochs=self.epochs,
                tfr=self.tfr,
                channel_names=self._selected_channels(),
                event_code=self._selected_event_code(),
                transform_mode=self.transform_combo.currentText(),
                baseline_s=self._baseline_s(),
                apply_baseline=self.baseline_checkbox.isChecked(),
            )
        except Exception as exc:
            logger.exception("Could not render time-frequency data.")
            self.display_data = None
            self._render_empty_state(str(exc))
            return

        self._draw_display_data()

    def _render_empty_state(self, message: str):
        tokens = theme_tokens()
        self.display_data = None
        self.context_label.setText("No TFR display")
        self.roi_mean_label.setText("Mean: n/a")
        self.main_ax = self.figure.add_subplot(111)
        self.main_ax.set_facecolor(tokens["plot_background"])
        self.figure.patch.set_facecolor(tokens["panel"])
        self.main_ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.main_ax.transAxes,
            color=tokens["muted"],
        )
        self.main_ax.set_axis_off()
        self.canvas.draw_idle()

    def _plot_evoked_traces(self, tokens: dict[str, str]):
        if self.display_data is None or self.erp_ax is None:
            return

        times_ms = self.display_data.times_ms
        channel_names = list(self.display_data.erp_channel_names)
        channel_data = np.asarray(self.display_data.erp_channel_uV, dtype=float)
        if channel_data.ndim != 2 or channel_data.shape[0] == 0:
            self.erp_average_line, = self.erp_ax.plot(
                times_ms,
                self.display_data.erp_uV,
                color=tokens["text"],
                linewidth=1.2,
            )
            return

        selected_channels = self._selected_channels()
        available_channel_count = len(self._available_channel_names())
        selected_count = len(selected_channels or [])
        show_all_channels = selected_channels is None or (
            available_channel_count > 0 and selected_count >= available_channel_count
        )
        show_subset_average = (
            selected_channels is not None
            and len(channel_names) > 1
            and (available_channel_count == 0 or len(channel_names) < available_channel_count)
        )

        colors = _spatial_trace_colors(self.epochs.info, channel_names)
        if show_all_channels:
            for trace_uV, color in zip(channel_data, colors):
                line, = self.erp_ax.plot(
                    times_ms,
                    trace_uV,
                    color=color,
                    linewidth=0.9,
                    alpha=0.76,
                )
                self.erp_channel_lines.append(line)
            return

        if show_subset_average:
            for trace_uV, color in zip(channel_data, colors):
                line, = self.erp_ax.plot(
                    times_ms,
                    trace_uV,
                    color=color,
                    linewidth=0.9,
                    alpha=0.38,
                )
                self.erp_channel_lines.append(line)
            self.erp_average_line, = self.erp_ax.plot(
                times_ms,
                self.display_data.erp_uV,
                color="black",
                linewidth=1.7,
                alpha=0.95,
                zorder=3.0,
            )
            return

        trace_uV = channel_data[0] if channel_data.shape[0] == 1 else self.display_data.erp_uV
        line, = self.erp_ax.plot(times_ms, trace_uV, color="black", linewidth=1.3)
        self.erp_channel_lines.append(line)

    def _draw_display_data(self):
        assert self.display_data is not None
        tokens = theme_tokens()
        self.figure.patch.set_facecolor(tokens["panel"])
        grid = self.figure.add_gridspec(
            3,
            5,
            width_ratios=(8.0, 1.25, 0.34, 0.55, 0.45),
            height_ratios=(1.35, 6.8, 1.35),
            left=0.075,
            right=0.975,
            top=0.955,
            bottom=0.105,
            hspace=0.09,
            wspace=0.16,
        )
        self.erp_ax = self.figure.add_subplot(grid[0, 0])
        self.main_ax = self.figure.add_subplot(grid[1, 0], sharex=self.erp_ax)
        self.spectrum_ax = self.figure.add_subplot(grid[1, 1], sharey=self.main_ax)
        self.colorbar_ax = self.figure.add_subplot(grid[1, 2])
        spacer_ax = self.figure.add_subplot(grid[1, 3])
        spacer_ax.set_axis_off()
        self.slider_ax = self.figure.add_subplot(grid[1, 4])
        self.time_power_ax = self.figure.add_subplot(grid[2, 0], sharex=self.main_ax)
        for empty_cell in (
            grid[0, 1],
            grid[0, 2],
            grid[0, 3],
            grid[0, 4],
            grid[2, 1],
            grid[2, 2],
            grid[2, 3],
            grid[2, 4],
        ):
            blank_ax = self.figure.add_subplot(empty_cell)
            blank_ax.set_axis_off()

        for axis in (self.spectrum_ax, self.main_ax, self.erp_ax, self.time_power_ax):
            axis.set_facecolor(tokens["plot_background"])
            axis.tick_params(colors=tokens["text"])
            axis.xaxis.label.set_color(tokens["text"])
            axis.yaxis.label.set_color(tokens["text"])
            for spine in axis.spines.values():
                spine.set_color(tokens["border"])
            axis.grid(True, color=tokens["grid"], alpha=0.28, linewidth=0.7)

        times_ms = self.display_data.times_ms
        freqs_hz = self.display_data.freqs_hz
        time_extent = _centers_to_extent(times_ms)
        freq_extent = _centers_to_extent(freqs_hz)
        self._full_clim = _finite_range(self.display_data.values)
        if self._clim is None:
            self._clim = _percentile_clim(self.display_data.values)
        self._set_clim_spinbox_values(*self._clim)

        self.image_artist = self.main_ax.imshow(
            self.display_data.values,
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="turbo",
            extent=[time_extent[0], time_extent[1], freq_extent[0], freq_extent[1]],
            vmin=self._clim[0],
            vmax=self._clim[1],
        )
        self.main_ax.axvline(0.0, color=tokens["muted"], linestyle="--", linewidth=1.0)
        self.main_ax.set_ylabel("Frequency (Hz)")
        self.main_ax.tick_params(axis="x", labelbottom=False)

        self.spectrum_line, = self.spectrum_ax.plot([], [], color=tokens["text"], linewidth=1.4)
        self.spectrum_ax.set_xlabel("Avg")
        self.spectrum_ax.tick_params(axis="y", labelleft=False)

        self._plot_evoked_traces(tokens)
        self.erp_ax.axvline(0.0, color=tokens["muted"], linestyle="--", linewidth=1.0)
        self.erp_ax.set_ylabel("uV")
        self.erp_ax.tick_params(axis="x", labelbottom=False)

        self.time_power_line, = self.time_power_ax.plot([], [], color=tokens["accent_soft"], linewidth=1.3)
        self.time_power_ax.axvline(0.0, color=tokens["muted"], linestyle="--", linewidth=1.0)
        self.time_power_ax.set_xlabel("Time (ms)")
        self.time_power_ax.set_ylabel("Avg")

        colorbar = self.figure.colorbar(self.image_artist, cax=self.colorbar_ax)
        colorbar.set_label(self.display_data.units)
        colorbar.ax.yaxis.label.set_color(tokens["text"])
        colorbar.ax.tick_params(colors=tokens["text"])
        self._create_range_slider()

        self.crosshair_vline = self.main_ax.axvline(
            times_ms[0],
            color=tokens["text"],
            linewidth=0.8,
            alpha=0.72,
            visible=False,
        )
        self.crosshair_hline = self.main_ax.axhline(
            freqs_hz[0],
            color=tokens["text"],
            linewidth=0.8,
            alpha=0.72,
            visible=False,
        )
        self.tooltip = self.main_ax.annotate(
            "",
            xy=(times_ms[0], freqs_hz[0]),
            xytext=(12, 12),
            textcoords="offset points",
            color=tokens["text"],
            bbox=dict(boxstyle="round,pad=0.28", fc=tokens["panel"], ec=tokens["border"], alpha=0.94),
            fontsize=9,
            visible=False,
        )
        marginal_line_style = dict(
            color=tokens["text"],
            linewidth=0.8,
            alpha=0.68,
            visible=False,
            zorder=5.0,
        )
        self.erp_hover_vline = self.erp_ax.axvline(times_ms[0], **marginal_line_style)
        self.erp_hover_hline = self.erp_ax.axhline(0.0, **marginal_line_style)
        self.time_power_hover_vline = self.time_power_ax.axvline(times_ms[0], **marginal_line_style)
        self.time_power_hover_hline = self.time_power_ax.axhline(0.0, **marginal_line_style)
        self.spectrum_hover_vline = self.spectrum_ax.axvline(0.0, **marginal_line_style)
        self.spectrum_hover_hline = self.spectrum_ax.axhline(freqs_hz[0], **marginal_line_style)
        self.erp_tooltip = self._create_axis_tooltip(self.erp_ax)
        self.time_power_tooltip = self._create_axis_tooltip(self.time_power_ax)
        self.spectrum_tooltip = self._create_axis_tooltip(
            self.spectrum_ax,
            fixed_position=(0.04, 0.96),
        )
        self.roi_rectangle = Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            fill=False,
            edgecolor=tokens["accent_soft"],
            linewidth=1.4,
            linestyle="-",
            zorder=3.5,
        )
        self.main_ax.add_patch(self.roi_rectangle)
        self.roi_time_span = self.time_power_ax.axvspan(
            0.0,
            0.0,
            facecolor=tokens["accent_fill"],
            alpha=0.28,
            visible=False,
            zorder=0.2,
        )
        self._ensure_roi_bounds()
        self._configure_roi_controls()
        self._update_roi_outputs(draw=False)
        self._style_auxiliary_axes()
        self._connect_xlim_callback()
        self._update_header_text()
        self.canvas.draw_idle()

    def _create_axis_tooltip(self, axis, *, fixed_position: tuple[float, float] | None = None):
        tokens = theme_tokens()
        if fixed_position is None:
            tooltip = axis.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(12, 12),
                textcoords="offset points",
                color=tokens["text"],
                bbox=dict(boxstyle="round,pad=0.28", fc=tokens["panel"], ec=tokens["border"], alpha=0.94),
                fontsize=9,
                visible=False,
                zorder=20.0,
            )
        else:
            tooltip = axis.annotate(
                "",
                xy=fixed_position,
                xycoords="axes fraction",
                xytext=(0, 0),
                textcoords="offset points",
                ha="left",
                va="top",
                color=tokens["text"],
                bbox=dict(boxstyle="round,pad=0.28", fc=tokens["panel"], ec=tokens["border"], alpha=0.96),
                fontsize=9,
                visible=False,
                zorder=20.0,
                annotation_clip=False,
            )
        tooltip.set_clip_on(False)
        tooltip._tf_fixed_position = fixed_position is not None
        return tooltip

    def _style_auxiliary_axes(self):
        if self.spectrum_ax is not None:
            self.spectrum_ax.spines["top"].set_visible(False)
            self.spectrum_ax.spines["left"].set_visible(False)
        if self.erp_ax is not None:
            self.erp_ax.spines["top"].set_visible(False)
            self.erp_ax.spines["right"].set_visible(False)
        if self.time_power_ax is not None:
            self.time_power_ax.spines["top"].set_visible(False)
            self.time_power_ax.spines["right"].set_visible(False)

    def _default_roi(self) -> tuple[tuple[float, float], tuple[float, float]]:
        assert self.display_data is not None
        times = self.display_data.times_ms
        freqs = self.display_data.freqs_hz
        time_min = float(times[0])
        time_max = float(times[-1])
        freq_min = float(freqs[0])
        freq_max = float(freqs[-1])

        time_start = 0.0 if time_min <= 0.0 <= time_max else time_min
        time_stop = min(time_max, time_start + 120.0)
        if time_stop <= time_start:
            time_start, time_stop = time_min, time_max

        freq_low = max(freq_min, 8.0)
        freq_high = min(freq_max, 12.0)
        if freq_high <= freq_low:
            freq_low, freq_high = freq_min, freq_max
        return (time_start, time_stop), (freq_low, freq_high)

    def _clamp_roi(
        self,
        time_start: float,
        time_stop: float,
        freq_low: float,
        freq_high: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        assert self.display_data is not None
        times = self.display_data.times_ms
        freqs = self.display_data.freqs_hz
        time_min = float(times[0])
        time_max = float(times[-1])
        freq_min = float(freqs[0])
        freq_max = float(freqs[-1])

        left, right = sorted((float(time_start), float(time_stop)))
        low, high = sorted((float(freq_low), float(freq_high)))
        left = float(np.clip(left, time_min, time_max))
        right = float(np.clip(right, time_min, time_max))
        low = float(np.clip(low, freq_min, freq_max))
        high = float(np.clip(high, freq_min, freq_max))

        if right <= left:
            idx = int(np.clip(np.searchsorted(times, left), 0, len(times) - 1))
            if idx < len(times) - 1:
                right = float(times[idx + 1])
            elif idx > 0:
                left = float(times[idx - 1])
        if high <= low:
            idx = int(np.clip(np.searchsorted(freqs, low), 0, len(freqs) - 1))
            if idx < len(freqs) - 1:
                high = float(freqs[idx + 1])
            elif idx > 0:
                low = float(freqs[idx - 1])
        return (left, right), (low, high)

    def _ensure_roi_bounds(self):
        if self.display_data is None:
            return
        if self.roi_time_ms is None or self.roi_freq_hz is None:
            self.roi_time_ms, self.roi_freq_hz = self._default_roi()
            return
        self.roi_time_ms, self.roi_freq_hz = self._clamp_roi(
            self.roi_time_ms[0],
            self.roi_time_ms[1],
            self.roi_freq_hz[0],
            self.roi_freq_hz[1],
        )

    def _configure_roi_controls(self):
        if self.display_data is None or self.roi_time_ms is None or self.roi_freq_hz is None:
            return
        times = self.display_data.times_ms
        freqs = self.display_data.freqs_hz
        self._updating_roi_controls = True
        try:
            for spinbox in (self.roi_time_start_input, self.roi_time_stop_input):
                spinbox.setRange(float(times[0]), float(times[-1]))
            for spinbox in (self.roi_freq_low_input, self.roi_freq_high_input):
                spinbox.setRange(float(freqs[0]), float(freqs[-1]))
            self.roi_time_start_input.setValue(self.roi_time_ms[0])
            self.roi_time_stop_input.setValue(self.roi_time_ms[1])
            self.roi_freq_low_input.setValue(self.roi_freq_hz[0])
            self.roi_freq_high_input.setValue(self.roi_freq_hz[1])
        finally:
            self._updating_roi_controls = False

    def _roi_masks(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.display_data is not None
        assert self.roi_time_ms is not None
        assert self.roi_freq_hz is not None
        time_start, time_stop = self.roi_time_ms
        freq_low, freq_high = self.roi_freq_hz
        time_mask = (self.display_data.times_ms >= time_start) & (self.display_data.times_ms <= time_stop)
        freq_mask = (self.display_data.freqs_hz >= freq_low) & (self.display_data.freqs_hz <= freq_high)
        if not np.any(time_mask):
            time_mask[int(np.argmin(np.abs(self.display_data.times_ms - time_start)))] = True
        if not np.any(freq_mask):
            freq_mask[int(np.argmin(np.abs(self.display_data.freqs_hz - freq_low)))] = True
        return time_mask, freq_mask

    def _update_roi_outputs(self, *, draw: bool):
        if (
            self.display_data is None
            or self.roi_time_ms is None
            or self.roi_freq_hz is None
            or self.spectrum_line is None
            or self.time_power_line is None
        ):
            return
        time_mask, freq_mask = self._roi_masks()
        values = self.display_data.values
        spectrum = np.nanmean(values[:, time_mask], axis=1)
        time_power = np.nanmean(values[freq_mask, :], axis=0)
        roi_values = values[np.ix_(freq_mask, time_mask)]
        roi_mean = float(np.nanmean(roi_values)) if roi_values.size else np.nan

        self.spectrum_line.set_data(spectrum, self.display_data.freqs_hz)
        self.time_power_line.set_data(self.display_data.times_ms, time_power)
        self.spectrum_ax.relim()
        self.spectrum_ax.autoscale_view(scalex=True, scaley=False)
        self.time_power_ax.relim()
        self.time_power_ax.autoscale_view(scalex=False, scaley=True)

        time_start, time_stop = self.roi_time_ms
        freq_low, freq_high = self.roi_freq_hz
        if self.roi_rectangle is not None:
            self.roi_rectangle.set_x(time_start)
            self.roi_rectangle.set_y(freq_low)
            self.roi_rectangle.set_width(time_stop - time_start)
            self.roi_rectangle.set_height(freq_high - freq_low)
            self.roi_rectangle.set_visible(True)
        if self.roi_time_span is not None:
            self.roi_time_span.set_x(time_start)
            self.roi_time_span.set_width(time_stop - time_start)
            self.roi_time_span.set_visible(True)

        mean_text = "n/a" if not np.isfinite(roi_mean) else f"{roi_mean:.3f} {self.display_data.units}"
        self.roi_mean_label.setText(f"Mean: {mean_text}")
        if draw:
            self.canvas.draw_idle()

    def _set_roi_from_values(
        self,
        time_start: float,
        time_stop: float,
        freq_low: float,
        freq_high: float,
        *,
        update_controls: bool,
        draw: bool,
    ):
        if self.display_data is None:
            return
        self.roi_time_ms, self.roi_freq_hz = self._clamp_roi(
            time_start,
            time_stop,
            freq_low,
            freq_high,
        )
        if update_controls:
            self._configure_roi_controls()
        self._update_roi_outputs(draw=draw)

    def _on_roi_control_changed(self):
        if self._updating_roi_controls or self.display_data is None:
            return
        self._set_roi_from_values(
            self.roi_time_start_input.value(),
            self.roi_time_stop_input.value(),
            self.roi_freq_low_input.value(),
            self.roi_freq_high_input.value(),
            update_controls=True,
            draw=True,
        )

    def _create_range_slider(self):
        if self.slider_ax is None or self._clim is None:
            return
        lower_bound, upper_bound = self._full_clim
        if math.isclose(lower_bound, upper_bound):
            lower_bound -= 1.0
            upper_bound += 1.0
        lower, upper = self._clim
        lower = float(np.clip(lower, lower_bound, upper_bound))
        upper = float(np.clip(upper, lower_bound, upper_bound))
        if lower >= upper:
            lower, upper = lower_bound, upper_bound

        self.slider_ax.set_facecolor(theme_tokens()["panel"])
        self._slider = RangeSlider(
            self.slider_ax,
            "",
            lower_bound,
            upper_bound,
            valinit=(lower, upper),
            orientation="vertical",
            valfmt="%.2f",
            facecolor="#2f7e8d",
        )
        self._slider.valtext.set_visible(False)
        self._slider_cid = self._slider.on_changed(self._on_range_slider_changed)

    def _on_range_slider_changed(self, value):
        lower, upper = float(value[0]), float(value[1])
        self._set_clim((lower, upper), update_spinboxes=True, update_slider=False)

    def _connect_xlim_callback(self):
        if self.main_ax is None:
            return
        self._xlim_callback_cid = self.main_ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

    def _on_xlim_changed(self, axis):
        if self._suppress_xlim_signal:
            return
        left, right = axis.get_xlim()
        self.timeWindowChanged.emit(float(left), float(right))

    def set_time_window(self, start_ms: float, stop_ms: float):
        if self.main_ax is None:
            return
        self._suppress_xlim_signal = True
        try:
            self.main_ax.set_xlim(float(start_ms), float(stop_ms))
            if self.erp_ax is not None:
                self.erp_ax.set_xlim(float(start_ms), float(stop_ms))
            if self.time_power_ax is not None:
                self.time_power_ax.set_xlim(float(start_ms), float(stop_ms))
            self.canvas.draw_idle()
        finally:
            self._suppress_xlim_signal = False

    def _update_header_text(self):
        if self.display_data is None:
            return
        self.context_label.setText(
            f"{self.current_derivative_key or 'TFR'} | {self.display_data.channel_label} | {self.display_data.event_label} | "
            f"Epochs: {self.display_data.n_epochs} | Units: {self.display_data.units}"
        )

    def _event_to_data(self, event) -> tuple[float, float] | None:
        if self.main_ax is None or self.display_data is None:
            return None
        if event.xdata is not None and event.ydata is not None:
            x_value = float(event.xdata)
            y_value = float(event.ydata)
        elif event.x is not None and event.y is not None:
            try:
                x_value, y_value = self.main_ax.transData.inverted().transform((event.x, event.y))
            except Exception:
                return None
        else:
            return None
        times = self.display_data.times_ms
        freqs = self.display_data.freqs_hz
        return (
            float(np.clip(x_value, float(times[0]), float(times[-1]))),
            float(np.clip(y_value, float(freqs[0]), float(freqs[-1]))),
        )

    def _on_button_press(self, event):
        if event.inaxes is not self.main_ax or event.button != MouseButton.LEFT:
            self._roi_drag_start = None
            self._roi_drag_active = False
            return
        coords = self._event_to_data(event)
        if coords is None:
            return
        self._roi_drag_start = coords
        self._roi_drag_active = True

    def _on_button_release(self, event):
        if not self._roi_drag_active or self._roi_drag_start is None:
            return
        start = self._roi_drag_start
        self._roi_drag_start = None
        self._roi_drag_active = False
        coords = self._event_to_data(event)
        if coords is None:
            return
        if abs(coords[0] - start[0]) < 0.5 and abs(coords[1] - start[1]) < 0.25:
            return
        self._set_roi_from_values(
            start[0],
            coords[0],
            start[1],
            coords[1],
            update_controls=True,
            draw=True,
        )

    def _hide_hover_artists(self, *, except_axis=None):
        if except_axis is not self.main_ax:
            for artist in (self.crosshair_vline, self.crosshair_hline, self.tooltip):
                if artist is not None:
                    artist.set_visible(False)
        for axis, artists in (
            (self.erp_ax, (self.erp_tooltip, self.erp_hover_vline, self.erp_hover_hline)),
            (
                self.time_power_ax,
                (self.time_power_tooltip, self.time_power_hover_vline, self.time_power_hover_hline),
            ),
            (
                self.spectrum_ax,
                (self.spectrum_tooltip, self.spectrum_hover_vline, self.spectrum_hover_hline),
            ),
        ):
            if axis is not except_axis:
                for artist in artists:
                    if artist is not None:
                        artist.set_visible(False)

    def _show_axis_tooltip(self, tooltip, x_value: float, y_value: float, text: str):
        if tooltip is None:
            return
        if not getattr(tooltip, "_tf_fixed_position", False):
            tooltip.xy = (x_value, y_value)
        tooltip.set_text(text)
        tooltip.set_visible(True)
        self.canvas.draw_idle()

    def _show_erp_hover(self, event):
        if self.display_data is None or event.xdata is None:
            return
        times_ms = self.display_data.times_ms
        index = int(np.argmin(np.abs(times_ms - float(event.xdata))))
        time_value = float(times_ms[index])
        amp_value = float(self.display_data.erp_uV[index])
        if self.erp_hover_vline is not None:
            self.erp_hover_vline.set_xdata([time_value, time_value])
            self.erp_hover_vline.set_visible(True)
        if self.erp_hover_hline is not None:
            self.erp_hover_hline.set_ydata([amp_value, amp_value])
            self.erp_hover_hline.set_visible(True)
        self._show_axis_tooltip(
            self.erp_tooltip,
            time_value,
            amp_value,
            f"Time: {time_value:.0f} ms\nAmplitude: {amp_value:.2f} uV",
        )

    def _show_time_power_hover(self, event):
        if self.time_power_line is None or event.xdata is None:
            return
        x_data = np.asarray(self.time_power_line.get_xdata(), dtype=float)
        y_data = np.asarray(self.time_power_line.get_ydata(), dtype=float)
        if x_data.size == 0 or y_data.size == 0:
            return
        index = int(np.argmin(np.abs(x_data - float(event.xdata))))
        time_value = float(x_data[index])
        power_value = float(y_data[index])
        units = self.display_data.units if self.display_data is not None else "Power"
        if self.time_power_hover_vline is not None:
            self.time_power_hover_vline.set_xdata([time_value, time_value])
            self.time_power_hover_vline.set_visible(True)
        if self.time_power_hover_hline is not None:
            self.time_power_hover_hline.set_ydata([power_value, power_value])
            self.time_power_hover_hline.set_visible(True)
        self._show_axis_tooltip(
            self.time_power_tooltip,
            time_value,
            power_value,
            f"Time: {time_value:.0f} ms\nAvg power: {power_value:.2f} {units}",
        )

    def _show_spectrum_hover(self, event):
        if self.spectrum_line is None or event.ydata is None:
            return
        x_data = np.asarray(self.spectrum_line.get_xdata(), dtype=float)
        y_data = np.asarray(self.spectrum_line.get_ydata(), dtype=float)
        if x_data.size == 0 or y_data.size == 0:
            return
        index = int(np.argmin(np.abs(y_data - float(event.ydata))))
        power_value = float(x_data[index])
        freq_value = float(y_data[index])
        units = self.display_data.units if self.display_data is not None else "Power"
        if self.spectrum_hover_vline is not None:
            self.spectrum_hover_vline.set_xdata([power_value, power_value])
            self.spectrum_hover_vline.set_visible(True)
        if self.spectrum_hover_hline is not None:
            self.spectrum_hover_hline.set_ydata([freq_value, freq_value])
            self.spectrum_hover_hline.set_visible(True)
        self._show_axis_tooltip(
            self.spectrum_tooltip,
            power_value,
            freq_value,
            f"Freq: {freq_value:.1f} Hz\nAvg power: {power_value:.2f} {units}",
        )

    def _on_mouse_move(self, event):
        if self._roi_drag_active and self._roi_drag_start is not None:
            coords = self._event_to_data(event)
            if coords is not None and self.roi_rectangle is not None:
                time_range, freq_range = self._clamp_roi(
                    self._roi_drag_start[0],
                    coords[0],
                    self._roi_drag_start[1],
                    coords[1],
                )
                self.roi_rectangle.set_x(time_range[0])
                self.roi_rectangle.set_width(time_range[1] - time_range[0])
                self.roi_rectangle.set_y(freq_range[0])
                self.roi_rectangle.set_height(freq_range[1] - freq_range[0])
                self.roi_rectangle.set_visible(True)
                self.canvas.draw_idle()

        if event.inaxes is self.erp_ax:
            self._hide_hover_artists(except_axis=self.erp_ax)
            self._show_erp_hover(event)
            return
        if event.inaxes is self.time_power_ax:
            self._hide_hover_artists(except_axis=self.time_power_ax)
            self._show_time_power_hover(event)
            return
        if event.inaxes is self.spectrum_ax:
            self._hide_hover_artists(except_axis=self.spectrum_ax)
            self._show_spectrum_hover(event)
            return

        if (
            self.display_data is None
            or event.inaxes is not self.main_ax
            or event.xdata is None
            or event.ydata is None
            or self.crosshair_vline is None
            or self.crosshair_hline is None
            or self.tooltip is None
        ):
            self._hide_hover_artists()
            return

        self._hide_hover_artists(except_axis=self.main_ax)
        times_ms = self.display_data.times_ms
        freqs_hz = self.display_data.freqs_hz
        if times_ms.size == 0 or freqs_hz.size == 0:
            return

        time_index = int(np.argmin(np.abs(times_ms - float(event.xdata))))
        freq_index = int(np.argmin(np.abs(freqs_hz - float(event.ydata))))
        time_value = float(times_ms[time_index])
        freq_value = float(freqs_hz[freq_index])
        power_value = float(self.display_data.values[freq_index, time_index])

        self.crosshair_vline.set_xdata([time_value, time_value])
        self.crosshair_hline.set_ydata([freq_value, freq_value])
        self.crosshair_vline.set_visible(True)
        self.crosshair_hline.set_visible(True)
        self.tooltip.xy = (time_value, freq_value)
        self.tooltip.set_text(
            f"Time: {time_value:.0f} ms\nFreq: {freq_value:.1f} Hz\nPower: {power_value:.2f}"
        )
        self.tooltip.set_visible(True)
        self.canvas.draw_idle()

    def _on_axes_leave(self, event):
        if event.inaxes not in (self.main_ax, self.erp_ax, self.time_power_ax, self.spectrum_ax):
            return
        self._hide_hover_artists()
        self.canvas.draw_idle()

    @Slot()
    def compute_time_frequency(self):
        event_code = self._selected_event_code()
        epochs = _subset_epochs(self.epochs, event_code)
        if len(epochs) == 0:
            QMessageBox.warning(self, "No Epochs", "No epochs match the selected event.")
            return

        dialog = ComputeTFRSettingsDialog(epochs, parent=self)
        selected_channels = self._selected_channels()
        if selected_channels:
            dialog.picks_input.setText(", ".join(selected_channels))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        compute_params = dialog.get_compute_params()

        def compute():
            display_epochs = epochs.copy()
            return display_epochs.compute_tfr(**compute_params)

        worker = Worker(compute, parent=self, add_loggers="mne")
        result = worker.exec_with_dialog("Please wait", "Computing time-frequency display...")
        if result is None:
            return
        if isinstance(result, tuple):
            power, itc = result
            self.tfr_derivatives["Computed power"] = power
            self.tfr_derivatives["Computed ITC"] = itc
            self.current_derivative_key = "Computed power"
            self.tfr = power
        else:
            label = "Computed display TFR"
            self.tfr_derivatives[label] = result
            self.current_derivative_key = label
            self.tfr = result
        self._populate_selectors()
        self._clim = None
        self._render()


class TimeFrequencyDialog(QDialog):
    def __init__(
        self,
        epochs: mne.BaseEpochs | None = None,
        *,
        label: str | None = None,
        tfr: Any = None,
        tfr_derivatives: dict[str, Any] | None = None,
        widget: TimeFrequencyWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Time-Frequency Analysis")
        self.setModal(False)
        self.setMinimumSize(900, 640)
        self.resize(1180, 780)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        if widget is None:
            if epochs is None:
                raise ValueError("epochs are required when widget is not provided.")
            widget = TimeFrequencyWidget(
                epochs,
                label=label,
                tfr=tfr,
                tfr_derivatives=tfr_derivatives,
                parent=self,
            )
        else:
            widget.setParent(self)
        self.widget = widget
        layout.addWidget(self.widget)
