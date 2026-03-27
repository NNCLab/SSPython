import sys
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QLabel,
    QToolBar,
    QRadioButton,
    QComboBox,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, QTimer, Signal, QSignalBlocker
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import mne
from utils import update_toolbar_color
import logging

from core.app_settings import get_settings_store

logger = logging.getLogger(__name__)


class MplCanvas(FigureCanvas):
    """A custom Matplotlib canvas widget that integrates with PySide6."""

    def __init__(self, parent=None, figsize=None, dpi=100):
        # Create a new Matplotlib figure
        fig = Figure(figsize=figsize, dpi=dpi)
        # Add an axes to the figure for plotting
        self.axes = fig.add_subplot(111)
        # Call the parent constructor
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

    def reset_axes(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        return self.axes


class TopomapWidget(QWidget):
    """A widget dedicated to displaying an MNE topomap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, figsize=(4, 4), dpi=100)
        self._colorbar = None
        layout.addWidget(self.canvas)
        # Prevent the layout from having extra margins
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def plot(self, data, info, cmap="turbo", title="Topomap"):
        """Plots the topomap data on the canvas."""
        axes = self.canvas.reset_axes()
        self._colorbar = None

        im, _ = mne.viz.plot_topomap(
            data,
            info,
            cmap=cmap,
            show=False,  # Important: MNE should not show the plot itself
            axes=axes,
        )

        # Add a colorbar to the figure
        self._colorbar = self.canvas.figure.colorbar(
            im, ax=axes, shrink=0.8, label=r"$\mu$V"
        )
        axes.set_title(title)
        self.canvas.draw()


class TopomapDialog(QDialog):
    """A dialog to display the TopomapWidget."""

    def __init__(self, data, info, cmap, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Topographical Map")
        self.setModal(False)
        self.setContentsMargins(0, 0, 0, 0)

        # Create the plot widget
        self.topomap_widget = TopomapWidget()
        self.topomap_widget.plot(data, info, cmap, title)

        # Set layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.topomap_widget)

    def closeEvent(self, *args, **kwargs):
        self.accept()


class EvokedPlotWidget(QWidget):
    """
    The main widget for displaying evoked potential plots with interactive events.
    """

    scrolled = Signal(str)

    def __init__(
        self, figsize=None, dpi=100, params=None, progress_dialog=None, parent=None
    ):
        super().__init__(parent)

        self.setWindowTitle("Evoked Potential Plot")
        self.progress_dialog = progress_dialog
        self.figsize = figsize
        self.dpi = dpi
        self.params = params or {}
        self.source_data = None
        self.epochs = None
        self.evoked = None
        self.display_evoked = None
        self.evoked_lines = []
        self.zero_line = None
        self.label = None
        self.selected_event_name = None
        self.open_topomaps = []

        self.setMinimumSize(400, 500)

        self.init_ui()
        self.init_params()
        self.connect_events()
        self.update_plot()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.reference_widget = QRadioButton("Average Reference")
        self.event_label = QLabel("Event:")
        self.event_selector = QComboBox()
        self.event_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.event_label.hide()
        self.event_selector.hide()

        self.topbar = QToolBar()
        self.topbar.setMovable(False)

        self.topbar.addWidget(self.reference_widget)
        self.topbar.addWidget(self.event_label)
        self.topbar.addWidget(self.event_selector)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.topbar.addWidget(spacer)

        self.title_label = QLabel("Waiting for data...")

        self.canvas = MplCanvas(self, figsize=self.figsize, dpi=self.dpi)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.topbar.addWidget(self.title_label)

        layout.addWidget(self.topbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

    def init_params(self):
        # Initialize instance variables
        self.params = self.params
        self.drag_start_coords = None
        self.drag_press_event = None
        self.drag_active = False
        self.drag_threshold_px = 6
        self.picked_artists_in_click = []
        self.original_zorders = {}
        self.ch_names = []
        self.evoked = None
        self.drag_span = None
        self.throttle_interval = 0.04  # (0.04s ≈ 25 FPS)
        self.last_update_time = 0

        self.default_params = {
            "baseline": (None, 0),
            "evoked_xlim": (-200, 500),
            "spatial_colors": True,
            "time_unit": "ms",
            "gfp": False,
            "cmap": "turbo",
        }

    def _load_plot_params(self):
        self.params = get_settings_store().get(
            "appearance/plots/global",
            self.default_params,
            legacy_keys=("plot_settings/plot_params",),
        )
        update_toolbar_color(self.toolbar)

    def _reset_canvas(self):
        self.canvas.reset_axes()
        self.evoked_lines = []
        self.zero_line = None
        self.original_zorders = {}
        self.drag_start_coords = None
        self.drag_press_event = None
        self.drag_active = False
        self.picked_artists_in_click.clear()
        self.drag_span = None

    def _set_event_selector_visible(self, visible: bool):
        self.event_label.setVisible(visible)
        self.event_selector.setVisible(visible)

    def _clear_event_selector(self):
        blocker = QSignalBlocker(self.event_selector)
        self.event_selector.clear()
        del blocker
        self.selected_event_name = None
        self._set_event_selector_visible(False)

    def _populate_event_selector(self, epochs: mne.BaseEpochs):
        event_names = list(epochs.event_id.keys()) if epochs.event_id else []
        if not event_names:
            self._clear_event_selector()
            return

        preferred_event = (
            self.selected_event_name if self.selected_event_name in event_names else None
        )
        counts = {
            event_name: int(np.sum(epochs.events[:, 2] == event_code))
            for event_name, event_code in epochs.event_id.items()
        }

        blocker = QSignalBlocker(self.event_selector)
        self.event_selector.clear()
        if len(event_names) > 1:
            self.event_selector.addItem("All events", None)
        for event_name in event_names:
            self.event_selector.addItem(
                f"{event_name} ({counts.get(event_name, 0)})",
                event_name,
            )

        if preferred_event is not None:
            index = self.event_selector.findData(preferred_event)
            if index >= 0:
                self.event_selector.setCurrentIndex(index)
        else:
            self.event_selector.setCurrentIndex(0)
        del blocker

        self.selected_event_name = self.event_selector.currentData()
        self._set_event_selector_visible(True)
        self.event_selector.setEnabled(self.event_selector.count() > 1)

    def _get_epochs_for_display(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        display_epochs = epochs.copy()
        good_channels = [
            channel
            for channel in display_epochs.ch_names
            if channel not in display_epochs.info["bads"]
        ]
        if good_channels:
            display_epochs.pick(good_channels)
        if self.selected_event_name:
            display_epochs = display_epochs[self.selected_event_name]
        return display_epochs

    def _resolve_current_evoked(self) -> mne.Evoked | None:
        if self.source_data is None:
            return None
        if isinstance(self.source_data, mne.Evoked):
            return self.source_data
        if isinstance(self.source_data, mne.BaseEpochs):
            return self._get_epochs_for_display(self.source_data).average()
        return None

    def _build_title_text(self) -> str:
        base_label = self.label or "Evoked Potential Plot"
        if self.epochs is None:
            return base_label
        event_label = self.selected_event_name or "All events"
        return f"{base_label} - {event_label}"

    def _prepare_display_evoked(self, evoked: mne.Evoked) -> mne.Evoked:
        display_evoked = evoked.copy()
        blocker = QSignalBlocker(self.reference_widget)

        if evoked.proj:
            self.reference_widget.setChecked(True)
            self.reference_widget.setEnabled(False)
        else:
            self.reference_widget.setEnabled(True)
            if self.reference_widget.isChecked():
                display_evoked.set_eeg_reference("average", projection=True)
                display_evoked.apply_proj()
        del blocker

        baseline = self.params.get("baseline", (None, 0)) or (None, 0)
        tmin, tmax = display_evoked.times.min(), display_evoked.times.max()
        baseline_tmin = baseline[0] if baseline[0] is not None else tmin
        baseline_tmax = baseline[1] if baseline[1] is not None else tmax

        if (
            baseline_tmin < tmin
            or baseline_tmax > tmax
            or baseline_tmin >= baseline_tmax
        ):
            logger.warning(
                "Baseline %s is outside of data time range [%.3f, %.3f]. Skipping baseline correction.",
                baseline,
                tmin,
                tmax,
            )
        else:
            display_evoked.apply_baseline(baseline)

        return display_evoked

    def _plot_display_evoked(self, evoked: mne.Evoked):
        evoked.plot(
            axes=self.canvas.axes,
            show=False,
            selectable=False,
            xlim=self.params.get("evoked_xlim", (-200, 500)),
            spatial_colors=self.params.get("spatial_colors", True),
            time_unit=self.params.get("time_unit", "ms"),
            gfp=self.params.get("gfp", False),
        )

    def _style_axes(self):
        axes = self.canvas.axes
        axes.grid(True, alpha=0.24, linewidth=0.8)
        axes.set_title("")
        axes.set_xlabel(f'Time ({self.params.get("time_unit", "ms")})')
        axes.set_ylabel(r"Amplitude ($\mu$V)")
        axes.margins(x=0.01)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)

    def _register_evoked_lines(self, evoked: mne.Evoked):
        stds = np.std(evoked.data, axis=1)
        z_order_indices = np.argsort(stds)
        self.evoked_lines = list(self.canvas.axes.get_lines())
        self.original_zorders = {}

        for i, line in enumerate(self.evoked_lines):
            if i >= len(z_order_indices):
                break
            z_order = int(z_order_indices[i])
            line.set_zorder(z_order)
            line.set_picker(5)
            line.set_linewidth(1.5)
            line.set_label(self.ch_names[i])
            self.original_zorders[line] = z_order

    def _add_zero_line(self):
        self.zero_line = self.canvas.axes.axvline(
            0,
            linestyle="--",
            color=plt.rcParams["text.color"],
            linewidth=1.0,
            alpha=0.75,
        )

    def _show_empty_state(self):
        self.evoked = None
        self.display_evoked = None
        self.title_label.setText("No Data Available")
        self.toolbar.hide()
        self.canvas.hide()
        self.canvas.draw()

    def _restore_line_state(self):
        if self.canvas.axes.get_legend():
            self.canvas.axes.get_legend().remove()

        for line in self.evoked_lines:
            line.set_linewidth(1.5)
            line.set_alpha(1.0)
            line.set_zorder(self.original_zorders.get(line, 0))

    def _render_current_view(self):
        self.reference_widget.setEnabled(True)
        self._load_plot_params()
        self._reset_canvas()
        self.reapply_style()

        evoked = self._resolve_current_evoked()
        if evoked is None:
            self._show_empty_state()
            return

        self.evoked = evoked
        self.ch_names = list(evoked.ch_names)
        self.display_evoked = self._prepare_display_evoked(evoked)

        self.toolbar.show()
        self.canvas.show()
        self._plot_display_evoked(self.display_evoked)
        self._style_axes()
        self._register_evoked_lines(self.display_evoked)
        self._add_zero_line()
        self.canvas.axes.autoscale_view(scalex=False, scaley=True)
        self.title_label.setText(self._build_title_text())
        self.canvas.draw()
        self.show()
        self.raise_()

    def update_plot(
        self, data: mne.BaseEpochs | mne.Evoked | None = None, label=None
    ):
        """
        Clears the axes and plots the new evoked potential data.
        """
        self.source_data = data
        self.label = label
        self.epochs = data if isinstance(data, mne.BaseEpochs) else None
        if self.epochs is not None:
            self._populate_event_selector(self.epochs)
        else:
            self._clear_event_selector()
        self._render_current_view()

    # === Event Handler Methods ===
    def connect_events(self):
        """Connects the required Matplotlib events to their callback methods."""
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", lambda e: self.scrolled.emit(e.button))
        self.reference_widget.toggled.connect(self._render_current_view)
        self.event_selector.currentIndexChanged.connect(
            self._on_event_selection_changed
        )

    def _on_event_selection_changed(self):
        self.selected_event_name = self.event_selector.currentData()
        self._render_current_view()

    def _clear_drag_span(self):
        if self.drag_span is not None:
            self.drag_span.remove()
            self.drag_span = None

    def on_button_press(self, event):
        """Callback for when a mouse button is pressed."""
        if event.inaxes is not self.canvas.axes:
            return
        if event.button == 1:
            self.picked_artists_in_click.clear()
            self.drag_press_event = event
            self.drag_start_coords = (event.xdata, event.ydata)
            self.drag_active = False

    def on_mouse_move(self, event):
        """Callback for mouse movement. Updates the axvspan during a drag."""
        current_time = time.time()
        if (current_time - self.last_update_time) < self.throttle_interval:
            # If not enough time has passed, do nothing
            return

        if (
            self.drag_press_event is None
            or self.drag_start_coords is None
            or event.inaxes is not self.canvas.axes
        ):
            return

        if (
            event.x is None
            or event.y is None
            or self.drag_press_event.x is None
            or self.drag_press_event.y is None
        ):
            return

        if not self.drag_active:
            drag_distance = np.hypot(
                event.x - self.drag_press_event.x,
                event.y - self.drag_press_event.y,
            )
            if drag_distance < self.drag_threshold_px:
                return

            self.drag_active = True
            self.picked_artists_in_click.clear()
            start_x = self.drag_start_coords[0]
            if start_x is None:
                return
            self.drag_span = self.canvas.axes.axvspan(
                start_x,
                start_x,
                facecolor=plt.rcParams["text.color"],
                alpha=0.3,
            )

        if self.drag_span is None:
            return

        current_x = event.xdata
        if current_x is None:
            return

        start_x = self.drag_start_coords[0]
        left = min(start_x, current_x)
        right = max(start_x, current_x)

        self.drag_span.set_x(left)
        self.drag_span.set_width(right - left)

        self.canvas.draw_idle()
        self.last_update_time = current_time

    def on_button_release(self, event):
        """Callback for when a mouse button is released."""
        if event.button != 1 or self.drag_press_event is None:
            return

        start_coords = self.drag_start_coords
        was_drag = self.drag_active
        self.drag_press_event = None
        self.drag_start_coords = None
        self.drag_active = False
        self._clear_drag_span()

        if (
            was_drag
            and start_coords is not None
            and start_coords[0] is not None
            and event.xdata is not None
        ):
            self.picked_artists_in_click.clear()
            self.handle_drag_selection(start=start_coords, end=(event.xdata, event.ydata))
            self.canvas.draw_idle()
            return

        QTimer.singleShot(0, self._process_pick_event)

    def on_pick(self, event):
        """Callback for a pick event. Just adds the picked artist to a list."""
        if (
            self.drag_press_event is not None
            and event.mouseevent.button == 1
            and event.artist in self.evoked_lines
        ):
            self.picked_artists_in_click.append(event.artist)

    def _process_pick_event(self):
        """
        This function runs after a short delay following any left-click.
        It checks if any artists were collected by on_pick. If not, it was a background click.
        """
        if not self.picked_artists_in_click:
            self.handle_line_click(None)  # No artists picked -> background click
            return

        top_artist = max(
            self.picked_artists_in_click, key=lambda artist: artist.get_zorder()
        )
        self.picked_artists_in_click.clear()
        self.handle_line_click(top_artist)

    def handle_drag_selection(self, start, end):
        """Called when a drag selection is completed. Plots a topomap in a dialog."""
        start_x, _ = start
        end_x, _ = end

        if start_x > end_x:
            start_x, end_x = end_x, start_x

        if self.display_evoked is None:
            return

        scale = 1e-3 if self.params.get("time_unit", "ms") == "ms" else 1
        idx = self.display_evoked.time_as_index((start_x * scale, end_x * scale))
        if idx[0] == idx[1]:
            return
        data = self.display_evoked.data[:, idx[0] : idx[1]].mean(axis=1)

        event_prefix = (
            f"{self.selected_event_name} - " if self.selected_event_name else ""
        )
        title = (
            f"{event_prefix}Topomap ({start_x:.2f} to {end_x:.2f}) "
            f"{self.params.get('time_unit', 'ms')}"
        )

        dialog = TopomapDialog(
            data=data,
            info=self.display_evoked.info,
            cmap=self.params.get("cmap", "turbo"),
            title=title,
            parent=self,
        )
        self.open_topomaps.append(dialog)
        dialog.finished.connect(
            lambda: self.open_topomaps.remove(dialog)
            if dialog in self.open_topomaps
            else None
        )
        dialog.show()

    def handle_line_click(self, line_artist):
        """Called for the single, top-most line that was clicked, or None for a background click."""
        if line_artist is None:
            self._restore_line_state()
            self.canvas.draw()
            return

        self.canvas.axes.legend(
            handles=[line_artist], fontsize="small", loc="upper right"
        )

        max_zorder = len(self.original_zorders) + 1
        line_artist.set_linewidth(3)
        line_artist.set_alpha(1.0)
        line_artist.set_zorder(max_zorder)

        for line in self.evoked_lines:
            if line is not line_artist:
                line.set_linewidth(1.5)
                line.set_alpha(0.3)
                line.set_zorder(self.original_zorders.get(line, 0))

        self.canvas.draw()

    def closeEvent(self, event):
        """
        Overrides the default close event to also close the progress dialog.
        """
        if self.progress_dialog:
            self.progress_dialog.close()  # Or .accept()

        if len(self.open_topomaps) > 0:
            for topomap in list(self.open_topomaps):
                topomap.close()

        event.accept()
        super().closeEvent(event)

    def reapply_style(self):
        """
        Re-applies the current matplotlib style to the canvas and all its elements.
        This is useful for dynamically updating the plot theme (e.g., light/dark mode).
        """
        logger.debug("Re-applying plot style...")

        # Manually update the colors of the figure and axes from the new rcParams
        fig = self.canvas.figure
        ax = self.canvas.axes

        fig.set_facecolor(plt.rcParams["figure.facecolor"])
        ax.set_facecolor(plt.rcParams["axes.facecolor"])

        # Update border (spines) and tick colors
        ax.spines["bottom"].set_color(plt.rcParams["axes.edgecolor"])
        ax.spines["top"].set_color(plt.rcParams["axes.edgecolor"])
        ax.spines["right"].set_color(plt.rcParams["axes.edgecolor"])
        ax.spines["left"].set_color(plt.rcParams["axes.edgecolor"])
        ax.tick_params(axis="x", colors=plt.rcParams["xtick.color"])
        ax.tick_params(axis="y", colors=plt.rcParams["ytick.color"])

        # Update the toolbar's colors (your existing utility function)
        update_toolbar_color(self.toolbar)


class EvokedPlotDialog(QDialog):
    """
    A dialog that wraps the EvokedPlotWidget to show it as a standalone,
    modal window.
    """

    def __init__(
        self, parent=None, figsize=None, dpi=100, params={}, progress_dialog=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Evoked Potential Plot")
        self.setMinimumSize(500, 600)  # Give the dialog a reasonable default size

        # 1. Create an instance of your main plotting widget
        self.plot_widget = EvokedPlotWidget(figsize, dpi, params, progress_dialog)

        # 2. Create standard dialog buttons (e.g., OK)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(
            self.accept
        )  # Connect OK to the dialog's accept Slot

        # 3. Set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)
        layout.addWidget(button_box)

    def update_plot(
        self, data: mne.BaseEpochs | mne.Evoked | None = None, label=None
    ):
        """
        A convenience method to pass data directly to the contained widget.
        This is called "delegation".
        """
        self.plot_widget.update_plot(data, label)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[::2][:32]
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
    evoked_data = mne.EvokedArray(np.random.randn(32, 1000) * 1e-6, info, tmin=-0.2)
    evoked_data.set_montage(montage)

    main_widget = EvokedPlotWidget()
    main_widget.update_plot(evoked_data, label="Random Evoked Data")
    main_widget.show()

    sys.exit(app.exec())
