import sys
import mne
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QLineEdit,
    QCheckBox,
    QLabel,
    QFrame,
    QStackedWidget,
    QListWidgetItem,
    QScrollArea,
    QPushButton,
    QGridLayout,
    QDialog,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QSplashScreen,
    QMessageBox,
    QGroupBox,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from utils import get_path, Worker
from .tools.optional_range_widget import OptionalRangeWidget
from .auto_label import ICALabelingWidget
import scipy
import logging
from dataclasses import dataclass, asdict

from core.app_settings import get_settings_store

logger = logging.getLogger(__file__)


class ICAPlottingSettingsWidget(QWidget):
    """
    A reusable widget for configuring plotting and auto-rejection settings.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        pipeline_id = self.settings_store.current_pipeline_id()
        self.settings_path = f"pipelines/{pipeline_id}/ica/plotting"
        self.init_form()
        self.load_settings()

    def init_form(self):
        layout = QVBoxLayout(self)

        # Plotting Parameters Group
        plot_box = QGroupBox("Plotting Parameters")
        plot_layout = QFormLayout(plot_box)
        layout.addWidget(plot_box)

        self.number_of_columns_input = QSpinBox()
        self.number_of_columns_input.setMinimum(1)
        self.number_of_columns_input.setToolTip("Number of columns in the home grid.")
        plot_layout.addRow("Number of Columns:", self.number_of_columns_input)

        self.decimate_input = QSpinBox()
        self.decimate_input.setMinimum(1)
        self.decimate_input.setToolTip(
            "Downsample the inst data to reduce memory and improve plotting speed."
        )
        plot_layout.addRow("Decimate Factor:", self.decimate_input)

        self.cmap_input = QComboBox()
        self.cmap_input.addItems(
            ["turbo", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r"]
        )
        self.cmap_input.setToolTip("Colormap for topography plots.")
        plot_layout.addRow("Colormap (cmap):", self.cmap_input)

        self.psd_input = OptionalRangeWidget(
            suffix="Hz", required=(True, False), parent=self
        )
        plot_layout.addRow("PSD Limits: ", self.psd_input)

        self.plot_type_input = QComboBox()
        self.plot_type_input.addItems(["Overlay", "Side-by-side"])
        self.plot_type_input.setToolTip(
            "Type of plot to display on the (overlay, side-by-side)."
        )
        plot_layout.addRow("Plot Type:", self.plot_type_input)

    def get_settings(self):
        """Returns the settings as a dictionary."""
        return {
            "number_of_columns": self.number_of_columns_input.value(),
            "decimate": self.decimate_input.value(),
            "cmap": self.cmap_input.currentText(),
            "psd_flim": self.psd_input.value(),
            "plot_type": self.plot_type_input.currentText(),
        }

    def load_settings(self):
        """Loads parameters from QSettings and populates the UI widgets."""
        default_params = {
            "number_of_columns": 5,
            "decimate": 1,
            "cmap": "turbo",
            "psd_flim": (0.0, 80.0),
            "plot_type": "Overlay",
        }

        save_settings = self.settings_store.get(
            self.settings_path,
            {},
            legacy_keys=("ica/plotting",),
        ) or {}
        params = default_params.copy()
        params.update(save_settings)
        self.number_of_columns_input.setValue(params.get("number_of_columns", 5))
        self.decimate_input.setValue(params.get("decimate", 5))
        self.cmap_input.setCurrentText(params.get("cmap", "turbo"))
        self.psd_input.setValue(params.get("psd_flim"))
        self.plot_type_input.setCurrentText(params.get("plot_type", "Overlay"))

    def save_settings(self):
        """Retrieves current settings from UI and saves them to QSettings."""
        params = self.get_settings()
        self.settings_store.set(self.settings_path, params)
        self.settings_store.set("ica/plotting", params)
        self.settings_store.sync()

    def set_pipeline(self, pipeline_id: str):
        self.settings_path = f"pipelines/{pipeline_id}/ica/plotting"
        self.load_settings()


class ICAPlottingDialog(QDialog):
    """A dialog window for configuring plotting settings. It uses IcaPlottingSettingsWidget for the UI and adds dialog-specific controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ICA Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)

        self.settings_widget = ICAPlottingSettingsWidget()
        layout.addWidget(self.settings_widget)

        # Standard OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def accept(self):
        """Saves settings when the user clicks 'OK'."""
        super().accept()

    def get_settings(self):
        """Returns the current settings from the widget."""
        return self.settings_widget.get_settings()


# --- Base Matplotlib Canvas Widget ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.fig.set_tight_layout(True)
        super().__init__(self.fig)
        self.fig.set_layout_engine("constrained")

        # Add this line to pass mouse events to the parent widget
        self.setAttribute(Qt.WA_TransparentForMouseEvents)


# --- Unchanged Overlay Plot Widget ---
class OverlayPlotWidget(QWidget):
    def __init__(self, ica, inst, plot_type="Overlay", baseline=(None, 0), parent=None):
        super().__init__(parent)
        self.ica = ica
        self.evoked = inst.copy().average()
        self.var_original = np.var(self.evoked.data)
        self.baseline = baseline

        self.times = self.evoked.times * 1e3
        self.canvas = MplCanvas(self, dpi=100)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.plot_type = plot_type
        if self.plot_type == "Side-by-side":
            self.start_sidebyside_plot()
        else:
            self.start_overlay_plot()

    def start_overlay_plot(self):
        self.primary_lines = []
        self.secondary_lines = []
        self.ax = self.canvas.fig.subplots()
        original_data = self.evoked.data
        for i in range(original_data.shape[0]):
            (line,) = self.ax.plot(
                self.times,
                original_data[i, :],
                color=plt.rcParams["text.color"],
                alpha=1,
            )
            self.primary_lines.append(line)
            (line,) = self.ax.plot(
                self.times, original_data[i, :], color="red", alpha=1
            )
            self.secondary_lines.append(line)
        for line in self.primary_lines:
            line.set_zorder(1)
        for line in self.secondary_lines:
            line.set_zorder(0)
        self.ax.set_title("ICA Overlay")
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(self.times[0], self.times[-1])
        self.ax.grid(True, linestyle=":")

    def update_overlay(self, detail_component_index=None):
        # primary -> cleaner
        # secondary -> less clean
        is_detail_page = detail_component_index is not None
        var_original = self.var_original.copy() if self.var_original > 0 else 1.0
        self.ax.set_title("")  # Clear the main title

        # --- Data Calculation ---
        if is_detail_page:
            is_excluded = detail_component_index in self.ica.exclude
            if is_excluded:
                primary_evoked = (
                    self.ica.apply(self.evoked.copy(), verbose=False)
                    .apply_baseline(self.baseline)
                    .data
                )
                comparison_evoked = np.zeros_like(primary_evoked) * np.nan
            else:
                comparison_evoked = (
                    self.ica.apply(self.evoked.copy(), verbose=False)
                    .apply_baseline(self.baseline)
                    .data
                )
                ica = self.ica.copy()
                ica.exclude = list(set(self.ica.exclude) | {detail_component_index})
                primary_evoked = (
                    ica.apply(self.evoked.copy(), verbose=False)
                    .apply_baseline(self.baseline)
                    .data
                )
                title_prefix = f"- C{detail_component_index:03}"
                power_comparison_pct = (np.var(comparison_evoked) / var_original) * 100

            power_primary_pct = (np.var(primary_evoked) / var_original) * 100

        else:
            comparison_evoked = self.evoked.copy().data
            primary_evoked = (
                self.ica.copy()
                .apply(self.evoked.copy(), verbose=False)
                .apply_baseline(self.baseline)
                .data
            )
            power_primary_pct = (np.var(primary_evoked) / var_original) * 100

        # --- Plotting Data ---
        for i in range(len(self.secondary_lines)):
            self.primary_lines[i].set_ydata(primary_evoked[i, :])
            self.secondary_lines[i].set_ydata(comparison_evoked[i, :])

        # --- Legend Drawing Logic ---
        if is_detail_page:
            if is_excluded:
                lines = [self.primary_lines[0]]
                labels = [f"Current ({power_primary_pct:.1f}%)"]
            else:
                label_primary = f"{title_prefix}\n(var:{power_primary_pct:.1f}%)"
                label_secondary = f"Current\n(var: {power_comparison_pct:.1f}%)"
                lines = [self.secondary_lines[0], self.primary_lines[0]]
                labels = [label_secondary, label_primary]
            self.ax.legend(lines, labels, loc="upper right", fontsize="x-small")
        else:
            lines = [self.secondary_lines[0], self.primary_lines[0]]
            labels = [
                "Original",
                f"n:-{len(self.ica.exclude)} (var:{power_primary_pct:.1f}%)",
            ]
            self.ax.legend(lines, labels, loc="upper right", fontsize="x-small")

        # --- Figure Updates ---
        if is_detail_page:
            if is_excluded:
                self.canvas.figure.set_facecolor("#E57373")
            else:
                self.canvas.figure.set_facecolor("#81C784")
        else:
            self.canvas.figure.set_facecolor(plt.rcParams["axes.facecolor"])

        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw()

    def start_sidebyside_plot(self):
        self.axs = self.canvas.fig.subplots(1, 2, sharey=True)
        left_evoked = self.evoked.copy()
        right_evoked = self.ica.apply(self.evoked.copy())
        left_evoked.plot(axes=self.axs[0])
        right_evoked.plot(axes=self.axs[1])
        self.axs[0].set_title("Cleaned")
        self.axs[1].set_title("Cleaned (Excluded: 0)")

    def update_sidebyside(self, detail_component_index=None):
        # axs[0] -> cleaned
        # axs[1] -> unclean
        is_detail_page = detail_component_index is not None
        is_excluded = False
        var_original = self.var_original.copy() if self.var_original > 0 else 1.0

        if is_detail_page:
            # --- Left Plot (axs[0]): Current Cleaned Signal ---
            clean_evoked = (
                self.ica.apply(self.evoked.copy(), verbose=False)
                .apply_baseline(self.baseline)
                .data
            )
            var_current_clean = np.var(clean_evoked)

            # Calculate the percentage of the original Signal power that is REMAINING.
            power_current_pct = (var_current_clean / var_original) * 100
            self.axs[0].set_title(f"Cleaned ({power_current_pct:.1f}%)")

            # --- Right Plot (axs[1]): "What If" Comparison ---
            is_excluded = detail_component_index in self.ica.exclude

            if is_excluded:
                # "What if" we ADD the component BACK?
                exclude_list_comparison = list(
                    set(self.ica.exclude) - {detail_component_index}
                )
                title_prefix = f"+ C{detail_component_index:03}"
            else:
                # "What if" we REMOVE the component?
                exclude_list_comparison = set(self.ica.exclude) | {
                    detail_component_index
                }
                title_prefix = f"- C{detail_component_index:03}"

            # Calculate the "what if" Signal.
            ica = self.ica.copy()
            ica.exclude = exclude_list_comparison
            uncleaned_evoked = (
                ica.apply(self.evoked.copy(), verbose=False)
                .apply_baseline(self.baseline)
                .data
            )
            var_comparison = np.var(uncleaned_evoked)

            # Calculate the power that WOULD REMAIN in this "what if" scenario.
            power_comparison_pct = (var_comparison / var_original) * 100
            self.axs[1].set_title(f"{title_prefix} ({power_comparison_pct:.1f}%)")

            for l, line in enumerate(self.axs[0].lines):
                line.set_ydata(clean_evoked[l, :])
            for l, line in enumerate(self.axs[1].lines):
                line.set_ydata(uncleaned_evoked[l, :])
            if is_excluded:
                self.canvas.figure.set_facecolor("#E57373")
            else:
                self.canvas.figure.set_facecolor("#81C784")
        else:
            # Home Page
            cleaned = self.ica.apply(self.evoked.copy()).apply_baseline(self.baseline)
            cleaned_var = np.var(cleaned.data)
            power_pct = (cleaned_var / var_original) * 100
            self.axs[0].set_title("Original")
            self.axs[1].set_title(
                f"Cleaned (Excluded: {len(self.ica.exclude)} | {power_pct:.1f}%)"
            )
            for l, line in enumerate(self.axs[0].lines):
                line.set_ydata(self.evoked.data[l, :])
            for l, line in enumerate(self.axs[1].lines):
                line.set_ydata(cleaned.data[l, :])
            self.canvas.figure.set_facecolor(plt.rcParams["axes.facecolor"])

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw()

    def update_plot(self, detail_component_index=None):
        if self.plot_type == "Side-by-side":
            self.update_sidebyside(detail_component_index=detail_component_index)
        else:
            self.update_overlay(detail_component_index=detail_component_index)


# --- Widget for a single item in the home grid ---
class ComponentGridItem(QFrame):
    component_selected = Signal(int)
    component_status_toggled = Signal(int)

    def __init__(self, ica, component_index, source_evoked, cmap="turbo", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setMinimumHeight(320)
        self.ica = ica
        self.index = component_index
        self.cmap = cmap
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.canvas = MplCanvas(self, dpi=100)
        self.canvas.figure.set_facecolor("none")
        self.ax_topo, self.ax_ts = self.canvas.fig.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}
        )
        layout.addWidget(self.label)
        layout.addWidget(self.canvas)
        self.update_plot(source_evoked)
        self.update_status_style()

    def update_plot(self, source_evoked):
        self.ax_topo.clear()
        self.ax_ts.clear()
        mne.viz.plot_ica_components(
            self.ica,
            picks=[self.index],
            axes=[self.ax_topo],
            show=False,
            cmap=self.cmap,
        )
        source_data_for_comp = source_evoked.get_data(picks=[self.index]).flatten()
        times = source_evoked.times
        self.ax_ts.axvline(0, color="C0", linestyle="--", alpha=0.5)
        self.ax_ts.plot(
            times, source_data_for_comp, linewidth=0.5, color=plt.rcParams["text.color"]
        )
        self.ax_ts.set_xlim(times[0], times[-1])
        self.ax_ts.set_yticks([])
        self.ax_ts.set_xticks([])
        self.ax_topo.set_title("")
        self.canvas.draw()

    def update_status_style(self):
        is_excluded = self.index in self.ica.exclude
        status = "(Drop)" if is_excluded else "(Keep)"
        self.label.setText(f"<b>Component {self.index:03}</b> {status}")
        self.setStyleSheet(
            f"background-color: {'#E57373' if is_excluded else '#81C784'}"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.component_selected.emit(self.index)
        elif event.button() == Qt.RightButton:
            self.component_status_toggled.emit(self.index)
        super().mousePressEvent(event)


# --- Home Page ---
class HomePage(QWidget):
    component_selected = Signal(int)
    component_status_toggled = Signal(int)

    def __init__(self, ica, inst, params, parent=None):
        super().__init__(parent)
        self.ica = ica
        self.params = params
        self.grid_items = {}
        self.source_evoked = ica.get_sources(inst.average())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        grid_container = QWidget()
        self.grid_layout = QGridLayout(grid_container)
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(grid_container)
        self.populate_grid()
        self.scroll_area.viewport().installEventFilter(self)

    def populate_grid(self):
        n_components = self.ica.n_components_
        n_cols = self.params.get("number_of_columns", 5)
        cmap = self.params.get("cmap", "turbo")
        for i in range(n_components):
            item_widget = ComponentGridItem(self.ica, i, self.source_evoked, cmap=cmap)
            item_widget.component_selected.connect(self.component_selected.emit)
            item_widget.component_status_toggled.connect(
                self.component_status_toggled.emit
            )
            row, col = divmod(i, n_cols)
            self.grid_layout.addWidget(item_widget, row, col)
            self.grid_items[i] = item_widget

    def update_all_item_styles(self):
        for item in self.grid_items.values():
            item.update_status_style()


# --- Detail page ---
class DetailPage(QWidget):
    def __init__(self, ica, inst, parent=None):
        super().__init__(parent)
        self.ica = ica
        self.inst = inst
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas(self, dpi=120)
        layout.addWidget(self.canvas)
        gs = GridSpec(
            4,
            3,
            figure=self.canvas.fig,
            height_ratios=[0.2, 2, 1, 2],
            width_ratios=[1, 0.05, 1],
        )
        self.axes_list = [
            self.canvas.fig.add_subplot(gs[1:3, 0]),
            self.canvas.fig.add_subplot(gs[1, 2]),
            self.canvas.fig.add_subplot(gs[2, 2]),
            self.canvas.fig.add_subplot(gs[3, 0]),
            self.canvas.fig.add_subplot(gs[3, 2]),
        ]

    def show_component(self, component_index, params):
        for ax in self.axes_list:
            ax.clear()
        for ax in self.axes_list[5:]:  # Remove sum of variance appended axis
            self.canvas.figure.delaxes(ax)
        self.axes_list = self.axes_list[:5]
        cmap = params.get("cmap", "viridis")
        psd_flim = list(params.get("psd_flim", (0, 80)))
        psd_flim[0] = psd_flim[0] if psd_flim[0] is not None else 0
        psd_flim[1] = psd_flim[1] if psd_flim[1] is not None else np.inf

        try:
            fig = self.ica.plot_properties(
                self.inst,
                picks=component_index,
                axes=self.axes_list,
                topomap_args={"cmap": cmap},
                image_args={"cmap": cmap},
                psd_args={"fmin": psd_flim[0], "fmax": psd_flim[1]},
                show=False,
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while plotting the component: {e}"
            )
            return

        self.axes_list = fig[0].axes

        main_color = plt.rcParams["text.color"]

        # EVK
        fig[0].axes[2].lines[0].set_color(main_color)
        fig[0].axes[2].collections[0].set_color(main_color)
        # PSD
        fig[0].axes[3].lines[0].set_color(main_color)
        fig[0].axes[3].collections[0].set_color(main_color)
        # Variance
        fig[0].axes[4].collections[0].set_color(main_color)
        fig[0].axes[4].collections[1].set_color(main_color)
        # Sum of Variance
        fig[0].axes[5].lines[0].set_color(main_color)
        for patch in fig[0].axes[5].patches:
            patch.set_color(main_color)

        for ax in self.axes_list:
            ax.set_facecolor(plt.rcParams["axes.facecolor"])

        is_excluded = component_index in self.ica.exclude
        if is_excluded:
            self.canvas.figure.set_facecolor("#E57373B0")
        else:
            self.canvas.figure.set_facecolor("#81C784B0")

        self.canvas.draw()


# --- Main viewer  ---
class ICAViewer(QDialog):
    def __init__(self, ica, inst, parent=None, **params):
        super().__init__(parent)
        self.ica = ica
        self.params = params
        self.setModal(True)
        self.setWindowTitle("ICA Viewer")
        self.baseline = (None, 0)
        inst = inst.copy().pick(
            [ch for ch in inst.info["ch_names"] if ch not in inst.info["bads"]]
        )

        if type(inst) in [mne.io.Raw, mne.io.RawArray]:
            inst = mne.make_fixed_length_epochs(inst, duration=2)

        self.inst = inst
        self.baseline = (
            (inst.tmin, 0) if np.sign(inst.tmin) == -1 else (inst.tmin, inst.tmax)
        )

        # Safely apply baseline
        tmin, tmax = inst.times.min(), inst.times.max()
        if self.baseline[0] is not None and self.baseline[1] is not None:
            if (
                self.baseline[0] >= tmin
                and self.baseline[1] <= tmax
                and self.baseline[0] < self.baseline[1]
            ):
                inst.apply_baseline(self.baseline)
                logger.info(f"ICA Viewer: Applied baseline {self.baseline}")

        decimate_factor = self.params.get("decimate", 1)
        if decimate_factor > 1:
            logger.info(
                f"Decimating inst by a factor of {decimate_factor} (new sfreq: {inst.info['sfreq']/decimate_factor:.2f} Hz)..."
            )
            self.inst = inst.decimate(decimate_factor)
        flim = self.params.get("psd_flim", (0, 80))
        self.params["psd_flim"] = [
            max(flim[0], 0),
            min(flim[1], inst.info["sfreq"] / 2),
        ]
        print(self.params)
        print(inst.info["sfreq"])

        self.current_component_index = None

        self.initUI()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setup_connections()
        self.adjustSize()

    def initUI(self):
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        main_layout = QHBoxLayout(self)
        # --- Left Lists ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(200)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.kept_label = QLabel("<b>Kept</b>")
        self.kept_list = QListWidget()
        self.dropped_label = QLabel("<b>Dropped</b>")
        self.dropped_list = QListWidget()
        left_layout.addWidget(self.kept_label)
        left_layout.addWidget(self.kept_list, 2)
        left_layout.addWidget(self.dropped_label)
        left_layout.addWidget(self.dropped_list, 2)
        left_layout.addStretch(1)

        # Auto Reject
        # auto_reject_widget = AutoRejectionWidget(self.ica, self.inst, settings_only=False) # OLD
        self.ica_labeling_widget = ICALabelingWidget(self.ica, self.inst, parent=self)
        left_layout.addWidget(self.ica_labeling_widget)

        # Buttons
        button_box = QDialogButtonBox()
        self.home_button = button_box.addButton("🏠 Home", QDialogButtonBox.ResetRole)
        self.accept_button = button_box.addButton(
            "✔ Accept", QDialogButtonBox.AcceptRole
        )
        self.cancel_button = button_box.addButton(
            "✖ Cancel", QDialogButtonBox.RejectRole
        )
        left_layout.addWidget(self.home_button)
        left_layout.addStretch(1)
        left_layout.addWidget(self.accept_button)
        left_layout.addWidget(self.cancel_button)

        # --- Right Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.overlay_plot = OverlayPlotWidget(
            self.ica,
            self.inst,
            plot_type=self.params.get("plot_type", "Overlay"),
            baseline=self.baseline,
        )
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setContentsMargins(0, 0, 0, 0)
        self.home_page = HomePage(self.ica, self.inst, self.params)
        self.detail_page = DetailPage(self.ica, self.inst)
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.detail_page)
        right_layout.addWidget(self.overlay_plot, 1)
        right_layout.addWidget(self.stacked_widget, 3)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        self.update_component_lists()
        self.show_home_view()

    def setup_connections(self):
        for list_widget in [self.kept_list, self.dropped_list]:
            list_widget.itemClicked.connect(
                lambda item: self.show_detail_view(item.data(Qt.UserRole))
            )
            list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
            list_widget.customContextMenuRequested.connect(self.handle_list_right_click)

        self.ica_labeling_widget.finished_label.connect(self.auto_label)
        self.home_page.component_selected.connect(self.show_detail_view)
        self.home_page.component_status_toggled.connect(
            self.toggle_component_status_from_click
        )
        self.home_button.clicked.connect(self.show_home_view)
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def auto_label(self, ica):
        self.ica = ica
        self.update_component_lists()
        self.update_list_selection()
        self.overlay_plot.update_plot(
            detail_component_index=self.current_component_index
        )
        self.home_page.update_all_item_styles()
        if self.stacked_widget.currentWidget() == self.detail_page:
            self.detail_page.show_component(self.current_component_index, self.params)

    def handle_list_right_click(self, pos):
        """Toggles the component status when an item in the list is right-clicked."""
        list_widget = self.sender()
        if item := list_widget.itemAt(pos):
            index = item.data(Qt.UserRole)
            self.toggle_component_status_from_click(index)

    def keyPressEvent(self, event):
        """Handles keyboard navigation for the entire dialog."""
        key = event.key()
        if key == Qt.Key_Right:
            self.navigate_cycle(1)
        elif key == Qt.Key_Left:
            self.navigate_cycle(-1)
        elif key == Qt.Key_Space and self.current_component_index is not None:
            self.toggle_current_component()
        elif key in (Qt.Key_Escape, Qt.Key_H):
            self.show_home_view()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
        else:
            super().keyPressEvent(event)

    def navigate_cycle(self, delta):
        """Navigates between home and components in a cycle."""
        if self.ica.n_components_ == 0:
            return

        n_comps = self.ica.n_components_
        current_index = self.current_component_index

        if delta > 0:  # Moving Right (->)
            if current_index is None:  # Currently on Home page
                self.show_detail_view(0)
            elif current_index == n_comps - 1:  # On the last component
                self.show_home_view()
            else:  # On any other component
                self.show_detail_view(current_index + 1)
        elif delta < 0:  # Moving Left (<-)
            if current_index is None:  # Currently on Home page
                self.show_detail_view(n_comps - 1)
            elif current_index == 0:  # On the first component
                self.show_home_view()
            else:  # On any other component
                self.show_detail_view(current_index - 1)

    def toggle_current_component(self):
        if self.current_component_index is not None:
            self.toggle_component_status_from_click(self.current_component_index)

    def toggle_component_status_from_click(self, index):
        self.on_component_status_changed(index, index not in self.ica.exclude)

    def on_component_status_changed(self, index, is_excluded):
        if is_excluded and index not in self.ica.exclude:
            self.ica.exclude.append(index)
        elif not is_excluded and index in self.ica.exclude:
            self.ica.exclude.remove(index)
        self.ica.exclude.sort()
        self.overlay_plot.update_plot(
            detail_component_index=self.current_component_index
        )
        self.update_component_lists()
        self.home_page.update_all_item_styles()
        if self.stacked_widget.currentWidget() == self.detail_page:
            self.detail_page.show_component(self.current_component_index, self.params)

    def show_detail_view(self, component_index):
        self.setFocus()
        self.current_component_index = component_index
        self.overlay_plot.update_plot(detail_component_index=component_index)
        self.detail_page.show_component(component_index, self.params)
        self.stacked_widget.setCurrentWidget(self.detail_page)
        self.update_list_selection()

    def show_home_view(self):
        self.setFocus()  # Set focus when returning home too
        self.current_component_index = None
        self.overlay_plot.update_plot(detail_component_index=None)
        self.stacked_widget.setCurrentWidget(self.home_page)
        self.update_list_selection()

    @staticmethod
    def label_to_list(ica):
        data = ica.labels_
        if data == {}:
            return False
        flat_list = [
            (int(index), label) for label, indices in data.items() for index in indices
        ]
        flat_list.sort(key=lambda pair: pair[0])
        result_list = [label for index, label in flat_list]
        return result_list

    def update_component_lists(self):
        self.kept_list.clear()
        self.dropped_list.clear()
        labels = self.label_to_list(self.ica)
        for i in range(self.ica.n_components_):
            item = QListWidgetItem(
                f"Component {i:03}" + f"({labels[i]})"
                if labels
                else f"Component {i:03}"
            )
            item.setData(Qt.UserRole, i)
            if i in self.ica.exclude:
                self.dropped_list.addItem(item)
            else:
                self.kept_list.addItem(item)
        self.kept_label.setText(f"<b>Kept ({self.kept_list.count()})</b>")
        self.dropped_label.setText(f"<b>Dropped ({self.dropped_list.count()})</b>")
        self.update_list_selection()

    def update_list_selection(self):
        self.kept_list.clearSelection()
        self.dropped_list.clearSelection()
        if self.current_component_index is None:
            return
        target_list = (
            self.dropped_list
            if self.current_component_index in self.ica.exclude
            else self.kept_list
        )
        for i in range(target_list.count()):
            item = target_list.item(i)
            if item.data(Qt.UserRole) == self.current_component_index:
                item.setSelected(True)
                target_list.scrollToItem(item)
                break

    def closeEvent(self, event):
        self.accept()

    def accept(self):
        super().accept()

    def reject(self):
        super().reject()


def run_ica_viewer(ica, inst, parent=None):
    """ICAViewer helper function."""
    dialog = ICAPlottingDialog(parent)
    if dialog.exec():  # Plot parameters
        params = dialog.get_settings()
    else:
        return False

    pixmap = QPixmap(get_path("assets/icon.png"))
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage(
        "Initializing ICA Viewer...\nThis may take a moment.",
        Qt.AlignCenter | Qt.AlignBottom,
        Qt.black,
    )
    logger.info(f"Running ICA Viewer with parameters: {params}")
    dialog = ICAViewer(ica, inst, parent, **params)
    splash.finish(dialog)
    if dialog.exec():
        return dialog.ica
    else:
        return False


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QWidget
    import mne
    import numpy as np

    app = QApplication(sys.argv)
    main_window = QWidget()
    dimensions = (50, 64, 50)
    tmin = -0.5
    sfreq = 1000
    data = np.random.rand(*dimensions)
    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[: dimensions[1]]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    info.set_montage(montage)
    inst = mne.EpochsArray(data, info, tmin=tmin)
    ica = mne.preprocessing.ICA()
    ica.fit(inst)
    dialog = run_ica_viewer(ica, inst, main_window)
    if dialog:
        print(f"ICA Updated: {dialog.exclude}")
    sys.exit()
