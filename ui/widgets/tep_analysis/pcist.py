"""PCIst calculation and linked threshold/transition-matrix inspection."""

import mne
import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from core.external.pci_st import distance2transition
from core.pcist import PCIST_DEFAULTS, calculate_pcist
from ui.widgets.event_selection import EventSelector
from utils import Worker, update_toolbar_color, use_plot_style


class PCIstDialog(QDialog):
    """Compute PCIst on an evoked response, then explore each retained PC."""

    def __init__(self, epochs, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.result = None
        self.setWindowTitle("PCIst — State Transition Complexity")
        self.setMinimumSize(960, 680)
        self.resize(1380, 940)
        self.parameter_inputs = {}
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(splitter)
        parameters = QWidget()
        parameters_layout = QVBoxLayout(parameters)
        parameters_layout.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton("Calculate PCIst")
        self.run_button.clicked.connect(self.run_analysis)
        parameters_layout.addWidget(self.run_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(290)
        scroll.setWidget(self._build_parameters())
        parameters_layout.addWidget(scroll, 1)
        splitter.addWidget(parameters)

        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        self.result_label = QLabel("PCIst: —")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        viewer_layout.addWidget(self.result_label)
        self.stale_label = QLabel("")
        self.stale_label.setWordWrap(True)
        self.stale_label.hide()
        viewer_layout.addWidget(self.stale_label)

        self.inspector = QWidget()
        inspector_layout = QHBoxLayout(self.inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.addWidget(QLabel("Component:"))
        self.component_selector = QComboBox()
        inspector_layout.addWidget(self.component_selector, 1)
        self.optimal_button = QPushButton("Optimal threshold ε*")
        inspector_layout.addWidget(self.optimal_button)
        viewer_layout.addWidget(self.inspector)
        self.threshold_controls = QWidget()
        threshold_layout = QHBoxLayout(self.threshold_controls)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setToolTip("Move through the calculated thresholds, or click the threshold curve.")
        threshold_layout.addWidget(self.threshold_slider, 1)
        self.threshold_label = QLabel("Threshold ε")
        threshold_layout.addWidget(self.threshold_label)
        viewer_layout.addWidget(self.threshold_controls)
        self.component_label = QLabel("")
        self.component_label.setWordWrap(True)
        self.component_label.hide()
        viewer_layout.addWidget(self.component_label)

        self.tabs = QTabWidget()
        explorer = QWidget()
        explorer_layout = QVBoxLayout(explorer)
        self.figure = Figure(figsize=(9, 8), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        update_toolbar_color(self.toolbar)
        explorer_layout.addWidget(self.canvas, 1)
        explorer_layout.addWidget(self.toolbar)
        self.tabs.addTab(explorer, "Threshold explorer")
        details = QWidget()
        details_layout = QVBoxLayout(details)
        self.detail_figure = Figure(figsize=(9, 8), layout="constrained")
        self.detail_canvas = FigureCanvas(self.detail_figure)
        self.detail_toolbar = NavigationToolbar(self.detail_canvas, self)
        update_toolbar_color(self.detail_toolbar)
        details_layout.addWidget(self.detail_canvas, 1)
        details_layout.addWidget(self.detail_toolbar)
        self.tabs.addTab(details, "Evoked response & components")
        viewer_layout.addWidget(self.tabs, 1)
        self.canvas.setToolTip(
            "Click a curve to select a threshold. NST densities are scaled by the response sample count.\n"
            "ΔNST = NST response − k × NST baseline. PCIst sums positive peak contributions."
        )
        splitter.addWidget(viewer)
        splitter.setSizes([320, 1060])
        self.inspector.setEnabled(False)
        self.threshold_controls.setEnabled(False)
        self.component_selector.currentIndexChanged.connect(self._on_component_changed)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.optimal_button.clicked.connect(self._select_optimal_threshold)
        self.canvas.mpl_connect("button_press_event", self._on_curve_click)
        self.detail_canvas.mpl_connect("button_press_event", self._on_contribution_click)

    def _build_parameters(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        selection = QGroupBox("Evoked input")
        selection_form = self._form(selection)
        self.event_selector = EventSelector(self.epochs, self)
        self.event_selector.setToolTip("The selected event's trials are averaged before calculating PCIst.")
        self.event_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.event_selector.setMinimumContentsLength(12)
        selection_form.addRow("Event:", self.event_selector)
        self.channels_input = QLineEdit()
        self.channels_input.setPlaceholderText("All good EEG channels")
        self.channels_input.setToolTip("Comma-separated EEG channel names. Bad channels are excluded.")
        selection_form.addRow("Channels:", self.channels_input)
        select_button = QPushButton("Select channels…")
        select_button.clicked.connect(self._select_channels)
        selection_form.addRow(select_button)
        layout.addWidget(selection)
        self.event_selector.currentIndexChanged.connect(self._mark_stale)
        self.channels_input.textChanged.connect(self._mark_stale)

        windows = QGroupBox("Time windows (ms)")
        form = self._form(windows)
        lower, upper = np.asarray(self.epochs.times)[[0, -1]] * 1000
        self.time_bounds = (float(lower), float(upper))
        for name, label in (("baseline_window", "Baseline:"), ("response_window", "Response:")):
            inputs = []
            for endpoint, value in zip(("start", "end"), PCIST_DEFAULTS[name]):
                spin = QDoubleSpinBox()
                spin.setDecimals(3)
                spin.setRange(float(lower), float(upper))
                spin.setSingleStep(1.0)
                spin.setValue(float(np.clip(value, lower, upper)))
                spin.setToolTip(f"Available: {lower:g} to {upper:g} ms. Window ends are exclusive.")
                spin.valueChanged.connect(self._mark_stale)
                form.addRow(f"{label[:-1]} {endpoint}:", spin)
                inputs.append(spin)
            self.parameter_inputs[name] = inputs
        layout.addWidget(windows)

        reduction = QGroupBox("Component selection")
        form = self._form(reduction)
        self._add_number(form, "max_var", "Variance retained (%):", 0.001, 100, 3)
        self._add_number(form, "min_snr", "Minimum SNR:", 0, 1e6, 3)
        self._add_number(form, "n_components", "SVD components:", 0, len(self.epochs.ch_names), special="Auto")
        self.parameter_inputs["min_snr"].setToolTip("Retain PCs whose response/baseline RMS ratio is strictly greater than this value.")
        layout.addWidget(reduction)

        transitions = QGroupBox("State transitions")
        form = self._form(transitions)
        self._add_number(form, "k", "Noise weight k:", 1.001, 100, 3)
        self._add_number(form, "n_steps", "Threshold steps:", 2, 10000)
        self._add_number(form, "max_thr_p", "Maximum threshold factor:", 0.001, 100, 3)
        self.parameter_inputs["max_thr_p"].setToolTip("Upper threshold = maximum response distance × this factor.")
        self._add_check(form, "embed", "Time-delay embedding")
        self._add_number(form, "L", "Embedding dimensions L:", 1, 1000)
        self._add_number(form, "tau", "Embedding delay (samples):", 1, 100000)
        self.parameter_inputs["tau"].setToolTip("Samples at the analysis sampling rate, after optional resampling.")
        self.parameter_inputs["embed"].toggled.connect(self._update_embedding_controls)
        self._update_embedding_controls()
        layout.addWidget(transitions)

        preprocessing = QGroupBox("Optional preprocessing")
        form = self._form(preprocessing)
        self._add_check(form, "avgref", "Average reference")
        self._add_check(form, "baseline_corr", "Baseline correction")
        self.parameter_inputs["baseline_corr"].setToolTip(
            "Subtracts the mean of all original samples before −50 ms, "
            "independently of the analysis baseline window."
        )
        self._add_number(form, "resample", "Resample (Hz):", 0, 100000, 3, special="Original sampling")
        layout.addWidget(preprocessing)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _form(parent):
        form = QFormLayout(parent)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _add_number(self, form, name, label, minimum, maximum, decimals=None, special=None):
        spin = QSpinBox() if decimals is None else QDoubleSpinBox()
        if decimals is not None:
            spin.setDecimals(decimals)
            spin.setSingleStep(0.1)
        spin.setRange(minimum, maximum)
        if special:
            spin.setSpecialValueText(special)
        spin.setValue(PCIST_DEFAULTS[name] or 0)
        spin.valueChanged.connect(self._mark_stale)
        self.parameter_inputs[name] = spin
        form.addRow(label, spin)

    def _add_check(self, form, name, label):
        check = QCheckBox(label)
        check.setChecked(PCIST_DEFAULTS[name])
        check.toggled.connect(self._mark_stale)
        self.parameter_inputs[name] = check
        form.addRow(check)

    def _update_embedding_controls(self, *_):
        enabled = self.parameter_inputs["embed"].isChecked()
        for name in ("L", "tau"):
            self.parameter_inputs[name].setEnabled(enabled)

    def _select_channels(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("PCIst EEG channels")
        layout = QVBoxLayout(dialog)
        channels = QListWidget()
        selected = {name.strip() for name in self.channels_input.text().split(",") if name.strip()}
        for index in mne.pick_types(self.epochs.info, eeg=True, meg=False, exclude="bads"):
            name = self.epochs.ch_names[index]
            item = QListWidgetItem(name, channels)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if not selected or name in selected else Qt.CheckState.Unchecked)
        layout.addWidget(channels)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            names = [channels.item(i).text() for i in range(channels.count()) if channels.item(i).checkState() == Qt.CheckState.Checked]
            if not names:
                QMessageBox.warning(self, "PCIst channels", "Select at least one EEG channel.")
                return
            self.channels_input.setText(", ".join(names))

    def get_parameters(self):
        params = {}
        for name, control in self.parameter_inputs.items():
            if isinstance(control, list):
                # Qt rounds spinbox limits to their display precision. Preserve
                # exact epoch endpoints when that rounding falls outside them.
                params[name] = tuple(float(np.clip(spin.value(), *self.time_bounds)) for spin in control)
            elif isinstance(control, QCheckBox):
                params[name] = control.isChecked()
            else:
                params[name] = control.value()
        for name in ("n_components", "resample"):
            params[name] = params[name] or None
        return params

    def _mark_stale(self, *_):
        if self.result is not None:
            self.stale_label.setText("Inputs changed. Recalculate to update.")
            self.stale_label.show()

    def run_analysis(self):
        params = self.get_parameters()
        picks = [name.strip() for name in self.channels_input.text().split(",") if name.strip()] or None
        self.run_button.setEnabled(False)
        try:
            worker = Worker(
                calculate_pcist, self.epochs, event_code=self.event_selector.currentData(),
                picks=picks, parent=self, **params,
            )
            result = worker.exec_with_dialog("PCIst", "Calculating components and state transitions…")
            if result is not None:
                self.set_result(result)
        finally:
            self.run_button.setEnabled(True)

    def set_result(self, result):
        self.result = result
        self.stale_label.clear()
        self.stale_label.hide()
        self.component_label.show()
        count = result["n_dims"]
        self.result_label.setText(
            f"PCIst: {result['PCI']:.6g}  ·  {count} retained components  ·  "
            f"{result['n_epochs']} epochs  ·  {len(result['ch_names'])} EEG channels\n{result['event_label']}"
        )
        blocker = QSignalBlocker(self.component_selector)
        self.component_selector.clear()
        for i, contribution in enumerate(result["dNST"]):
            rank = result.get("component_indices", np.arange(count))[i] + 1
            self.component_selector.addItem(f"PC {rank}  ·  ΔNST = {contribution:.4g}", i)
        del blocker
        self.inspector.setEnabled(bool(count))
        self.threshold_controls.setEnabled(bool(count))
        if count:
            self._on_component_changed(0)
        else:
            self.threshold_label.setText("No threshold available")
            self.component_label.setText("No components passed the variance/SNR selection.")
            self.figure.clear()
            self.curve_ax = None
            self.distance_axes = []
            self.transition_axes = []
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No retained components", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            self._render_details()

    def _on_component_changed(self, index):
        if self.result is None or index < 0 or not self.result["n_dims"]:
            return
        blocker = QSignalBlocker(self.threshold_slider)
        self.threshold_slider.setRange(0, self.result["thresholds"].shape[0] - 1)
        optimal = int(np.argmax(self.result["NST_diff"][:, index]))
        self.threshold_slider.setValue(optimal)
        del blocker
        self._render_explorer()
        self._render_details()

    def _select_optimal_threshold(self):
        if self.result is not None and self.result["n_dims"]:
            component = self.component_selector.currentIndex()
            self.threshold_slider.setValue(int(np.argmax(self.result["NST_diff"][:, component])))

    @use_plot_style
    def _render_explorer(self):
        result = self.result
        component = self.component_selector.currentIndex()
        self.figure.clear()
        self.figure.set_facecolor(rcParams["figure.facecolor"])
        grid = self.figure.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
        self.curve_ax = self.figure.add_subplot(grid[0, :])
        thresholds = result["thresholds"][:, component]
        scale = len(result["response_times"])
        k = result["params"]["k"]
        self.curve_ax.plot(thresholds, result["NST_resp"][:, component] * scale, color="#4c91cf", label="NST response")
        self.curve_ax.plot(thresholds, k * result["NST_base"][:, component] * scale, color="#888888", label=f"k × NST baseline (k={k:g})")
        self.curve_ax.plot(thresholds, result["NST_diff"][:, component] * scale, color="#e35667", label="ΔNST (signed)")
        optimum = result["max_thresholds"][component]
        self.curve_ax.axvline(optimum, color="#e35667", linestyle="--", linewidth=1, label="Optimal ε*")
        self.selected_line = self.curve_ax.axvline(optimum, color="#ba8c39", linestyle=":", linewidth=1.5, label="Selected ε")
        self.curve_ax.axhline(0, color="#888888", linewidth=0.5)
        self.curve_ax.set(xlabel="Threshold ε (µV)", ylabel="Scaled NST")
        self.curve_ax.legend(loc="upper right", fontsize=8, ncols=2)
        self.distance_axes = [self.figure.add_subplot(grid[1, i]) for i in range(2)]
        self.transition_axes = [self.figure.add_subplot(grid[2, i]) for i in range(2)]
        distance_max = max(float(np.max(result["D_base"][component])), float(np.max(result["D_resp"][component])), np.finfo(float).eps)
        for ax, key, times_key, title in zip(self.distance_axes, ("D_base", "D_resp"), ("baseline_times", "response_times"), ("Baseline distances", "Response distances")):
            im = ax.imshow(result[key][component], origin="lower", extent=self._extent(result[times_key]), interpolation="nearest", cmap="viridis", vmin=0, vmax=distance_max, aspect="auto")
            ax.set(title=title, xlabel="Time (ms)", ylabel="Time (ms)")
            self.figure.colorbar(im, ax=ax, label="µV", fraction=0.046, pad=0.025)
        self.transition_images = []
        for ax, times_key, title in zip(self.transition_axes, ("baseline_times", "response_times"), ("Baseline transitions", "Response transitions")):
            n = len(result[times_key])
            im = ax.imshow(np.zeros((n, n)), origin="lower", extent=self._extent(result[times_key]), interpolation="nearest", cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
            self.transition_images.append(im)
            ax.set(title=title, xlabel="Time (ms)", ylabel="Time (ms)")
        self._on_threshold_changed(self.threshold_slider.value())

    @staticmethod
    def _extent(times):
        step = (times[1] - times[0]) / 2
        return (times[0] - step, times[-1] + step, times[0] - step, times[-1] + step)

    def _on_threshold_changed(self, index):
        if self.result is None or not self.result["n_dims"]:
            return
        result = self.result
        component = self.component_selector.currentIndex()
        threshold = result["thresholds"][index, component]
        self.threshold_label.setText(f"ε = {threshold:.5g} µV  ({index + 1}/{len(result['thresholds'])})")
        for image, key in zip(self.transition_images, ("D_base", "D_resp")):
            image.set_data(distance2transition(result[key][component], threshold))
        self.selected_line.set_xdata([threshold, threshold])
        scale = len(result["response_times"])
        delta = result["NST_diff"][index, component] * scale
        variance = result["var_exp"][component]
        self.component_label.setText(
            f"Contribution: {result['dNST'][component]:.5g}  ·  Selected ΔNST: {delta:.5g}  ·  "
            f"Optimal ε*: {result['max_thresholds'][component]:.5g} µV  ·  "
            f"SNR: {result['snrs'][component]:.4g}  ·  Variance: {variance:.3g}%"
        )
        self.canvas.draw_idle()

    def _on_curve_click(self, event):
        if self.result is None or not self.result["n_dims"] or event.inaxes is not self.curve_ax or event.xdata is None or event.button != 1 or self.toolbar.mode:
            return
        thresholds = self.result["thresholds"][:, self.component_selector.currentIndex()]
        self.threshold_slider.setValue(int(np.argmin(np.abs(thresholds - event.xdata))))

    @use_plot_style
    def _render_details(self):
        result = self.result
        self.detail_figure.clear()
        self.detail_figure.set_facecolor(rcParams["figure.facecolor"])
        evoked_ax, component_ax, self.contribution_ax = self.detail_figure.subplots(3, 1)
        times = result["times"]
        evoked_ax.plot(times, result["signal_evk"].T, linewidth=0.65, alpha=0.65)
        evoked_ax.set(title="Preprocessed evoked response", ylabel="Amplitude (µV)")
        component = self.component_selector.currentIndex()
        if result["n_dims"]:
            rank = result.get("component_indices", np.arange(result["n_dims"]))[component] + 1
            component_ax.plot(times, result["signal_svd"][component], color="#4c91cf", linewidth=0.9)
            component_ax.set_title(f"PC {rank}")
            ranks = result.get("component_indices", np.arange(result["n_dims"])) + 1
            colors = ["#e35667" if i == component else "#4c91cf" for i in range(result["n_dims"])]
            self.contribution_ax.bar(np.arange(result["n_dims"]), result["dNST"], color=colors)
            self.contribution_ax.set_xticks(np.arange(result["n_dims"]), [f"PC {rank}" for rank in ranks])
        else:
            component_ax.text(0.5, 0.5, "No retained components", ha="center", transform=component_ax.transAxes)
        component_ax.set(xlabel="Time (ms)", ylabel="Amplitude (µV)")
        for ax in (evoked_ax, component_ax):
            ax.axvspan(*result["params"]["baseline_window"], color="#888888", alpha=0.15, label="Baseline")
            ax.axvspan(*result["params"]["response_window"], color="#4c91cf", alpha=0.12, label="Response")
            ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
        evoked_ax.legend(loc="upper right", fontsize=8)
        self.contribution_ax.set(title="Component contributions (click to select)", xlabel="Retained principal component", ylabel="ΔNST contribution")
        self.detail_canvas.draw_idle()

    def _on_contribution_click(self, event):
        if self.result is None or not self.result["n_dims"] or event.inaxes is not self.contribution_ax or event.xdata is None or event.button != 1 or self.detail_toolbar.mode:
            return
        component = int(round(event.xdata))
        if 0 <= component < self.result["n_dims"]:
            self.component_selector.setCurrentIndex(component)
