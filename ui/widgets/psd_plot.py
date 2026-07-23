from __future__ import annotations

import logging
import math
from typing import Any

import mne
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_settings_store
from ui.widgets.tools.optional_range_widget import OptionalRangeWidget

logger = logging.getLogger(__name__)


class PSDPlotSettingsWidget(QWidget):
    """User-facing controls for the relevant ``mne.Epochs.plot_psd`` parameters."""

    SETTINGS_PATH = "appearance/plots/psd"

    def __init__(self, epochs: mne.BaseEpochs | None = None, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.settings_store = get_settings_store()
        self.nyquist = self._nyquist_frequency()
        self.time_bounds = self._epoch_time_bounds()
        self.default_params = self._default_params()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._build_ui(layout)
        self.load_settings()

    def _nyquist_frequency(self) -> float:
        if self.epochs is None:
            return 50.0
        return float(self.epochs.info["sfreq"]) / 2.0

    def _epoch_time_bounds(self) -> tuple[float, float]:
        if self.epochs is None or len(self.epochs.times) == 0:
            return (0.0, 0.0)
        return (float(self.epochs.times[0]), float(self.epochs.times[-1]))

    def _default_params(self) -> dict[str, Any]:
        return {
            "fmin": 0.0,
            "fmax": min(50.0, self.nyquist),
            "tmin": None,
            "tmax": None,
            "picks": "",
            "method": "auto",
            "proj": False,
            "average": False,
            "dB": True,
            "estimate": "power",
            "xscale": "linear",
            "spatial_colors": True,
            "exclude_bads": True,
        }

    def _build_ui(self, layout: QVBoxLayout):
        groupbox = QGroupBox("PSD Parameters")
        form_layout = QFormLayout(groupbox)

        self.frequency_range_input = OptionalRangeWidget(
            labels=("Min:", "Max:"),
            suffix=" Hz",
            required=(True, False),
            range=(0.0, None),
            parent=self,
        )
        self._configure_range_spinboxes(
            self.frequency_range_input,
            decimals=2,
            step=1.0,
            minimum=0.0,
            maximum=self.nyquist,
        )

        self.time_range_input = OptionalRangeWidget(
            labels=("Start:", "End:"),
            suffix=" s",
            required=(False, False),
            range=(None, None),
            parent=self,
        )
        self._configure_range_spinboxes(
            self.time_range_input,
            decimals=4,
            step=0.01,
            minimum=self.time_bounds[0],
            maximum=self.time_bounds[1],
        )

        self.picks_input = QLineEdit()
        self.picks_input.setPlaceholderText("Blank for all data channels, or e.g. eeg, Cz, Pz")

        self.method_input = QComboBox()
        self.method_input.addItems(["auto", "welch", "multitaper"])

        self.estimate_input = QComboBox()
        self.estimate_input.addItems(["power", "amplitude"])

        self.xscale_input = QComboBox()
        self.xscale_input.addItems(["linear", "log"])

        self.proj_input = QCheckBox("Apply projectors")
        self.average_input = QCheckBox("Average spectra")
        self.db_input = QCheckBox("Use dB scale")
        self.spatial_colors_input = QCheckBox("Use spatial colors")
        self.exclude_bads_input = QCheckBox("Exclude bad channels")

        form_layout.addRow("Frequency range:", self.frequency_range_input)
        form_layout.addRow("Time range:", self.time_range_input)
        form_layout.addRow("Channels:", self.picks_input)
        form_layout.addRow("PSD method:", self.method_input)
        form_layout.addRow("Estimate:", self.estimate_input)
        form_layout.addRow("X axis scale:", self.xscale_input)
        form_layout.addRow(self.proj_input)
        form_layout.addRow(self.average_input)
        form_layout.addRow(self.db_input)
        form_layout.addRow(self.spatial_colors_input)
        form_layout.addRow(self.exclude_bads_input)

        layout.addWidget(groupbox)

    def _configure_range_spinboxes(
        self,
        widget: OptionalRangeWidget,
        *,
        decimals: int,
        step: float,
        minimum: float | None,
        maximum: float | None,
    ):
        for spinbox in (widget.low_input, widget.high_input):
            spinbox.setDecimals(decimals)
            spinbox.setSingleStep(step)
            if minimum is not None:
                spinbox.setMinimum(minimum)
            if maximum is not None:
                spinbox.setMaximum(maximum)
        widget.adjust_spinbox_width()

    def _finite_or_none(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric_value):
            return None
        return numeric_value

    def _load_raw_settings(self) -> dict[str, Any]:
        saved = self.settings_store.get(
            self.SETTINGS_PATH,
            {},
            legacy_keys=("plot_settings/psd_plot_params",),
        )
        if not isinstance(saved, dict):
            logger.warning("Ignoring invalid PSD plot settings: %r", saved)
            return {}
        return saved

    def _merged_settings(self) -> dict[str, Any]:
        saved = self._load_raw_settings()
        params = self.default_params.copy()
        params.update(saved)

        if "exclude" in saved and "exclude_bads" not in saved:
            params["exclude_bads"] = saved.get("exclude") == "bads"

        fmax = self._finite_or_none(params.get("fmax"))
        if fmax is not None:
            params["fmax"] = min(fmax, self.nyquist)

        return params

    def load_settings(self):
        params = self._merged_settings()

        self.frequency_range_input.setValue(
            (
                self._finite_or_none(params.get("fmin")) or 0.0,
                self._finite_or_none(params.get("fmax")),
            )
        )
        self.time_range_input.setValue(
            (
                self._finite_or_none(params.get("tmin")),
                self._finite_or_none(params.get("tmax")),
            )
        )
        self.picks_input.setText(str(params.get("picks", "") or ""))
        self.method_input.setCurrentText(params.get("method", "auto"))
        self.estimate_input.setCurrentText(params.get("estimate", "power"))
        self.xscale_input.setCurrentText(params.get("xscale", "linear"))
        self.proj_input.setChecked(bool(params.get("proj", False)))
        self.average_input.setChecked(bool(params.get("average", False)))
        self.db_input.setChecked(bool(params.get("dB", True)))
        self.spatial_colors_input.setChecked(bool(params.get("spatial_colors", True)))
        self.exclude_bads_input.setChecked(bool(params.get("exclude_bads", True)))

    def get_settings(self) -> dict[str, Any]:
        fmin, fmax = self.frequency_range_input.value()
        tmin, tmax = self.time_range_input.value()
        return {
            "fmin": float(fmin if fmin is not None else 0.0),
            "fmax": self._finite_or_none(fmax),
            "tmin": self._finite_or_none(tmin),
            "tmax": self._finite_or_none(tmax),
            "picks": self.picks_input.text().strip(),
            "method": self.method_input.currentText(),
            "proj": self.proj_input.isChecked(),
            "average": self.average_input.isChecked(),
            "dB": self.db_input.isChecked(),
            "estimate": self.estimate_input.currentText(),
            "xscale": self.xscale_input.currentText(),
            "spatial_colors": self.spatial_colors_input.isChecked(),
            "exclude_bads": self.exclude_bads_input.isChecked(),
        }

    def save_settings(self):
        params = self.get_settings()
        self.settings_store.set(self.SETTINGS_PATH, params)
        self.settings_store.set("plot_settings/psd_plot_params", params)
        self.settings_store.sync()

    def _parse_picks(self, value: str):
        cleaned_value = value.strip()
        if not cleaned_value:
            return None
        picks = [part.strip() for part in cleaned_value.split(",") if part.strip()]
        if not picks:
            return None
        if len(picks) == 1:
            return picks[0]
        return picks

    def get_plot_params(self) -> dict[str, Any]:
        settings = self.get_settings()
        params: dict[str, Any] = {
            "fmin": settings["fmin"],
            "method": settings["method"],
            "proj": settings["proj"],
            "average": settings["average"],
            "dB": settings["dB"],
            "estimate": settings["estimate"],
            "xscale": settings["xscale"],
            "spatial_colors": settings["spatial_colors"],
            "exclude": "bads" if settings["exclude_bads"] else (),
        }

        for key in ("fmax", "tmin", "tmax"):
            if settings[key] is not None:
                params[key] = settings[key]

        picks = self._parse_picks(settings["picks"])
        if picks is not None:
            params["picks"] = picks

        return params

    def validation_error(self) -> str | None:
        settings = self.get_settings()
        fmin = settings["fmin"]
        fmax = settings["fmax"]
        tmin = settings["tmin"]
        tmax = settings["tmax"]

        if fmin < 0:
            return "Frequency minimum must be zero or greater."
        if fmax is not None and fmin >= fmax:
            return "Frequency minimum must be lower than frequency maximum."
        if fmax is not None and fmax > self.nyquist:
            return f"Frequency maximum cannot exceed the Nyquist frequency ({self.nyquist:.2f} Hz)."
        if tmin is not None and tmin < self.time_bounds[0]:
            return f"Time minimum cannot be earlier than {self.time_bounds[0]:.4f} s."
        if tmax is not None and tmax > self.time_bounds[1]:
            return f"Time maximum cannot be later than {self.time_bounds[1]:.4f} s."
        if tmin is not None and tmax is not None and tmin >= tmax:
            return "Time minimum must be lower than time maximum."

        return None


class PSDPlotSettingsDialog(QDialog):
    """Dialog that collects PSD parameters before plotting epochs."""

    def __init__(self, epochs: mne.BaseEpochs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot PSD")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        self.settings_widget = PSDPlotSettingsWidget(epochs, parent=self)
        layout.addWidget(self.settings_widget)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Plot")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def accept(self):
        error = self.settings_widget.validation_error()
        if error:
            QMessageBox.warning(self, "Invalid PSD Parameters", error)
            return
        self.settings_widget.save_settings()
        super().accept()

    def get_plot_params(self) -> dict[str, Any]:
        return self.settings_widget.get_plot_params()
