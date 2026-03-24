import sys
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QLabel,
    QToolBar,
    QRadioButton,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, QTimer, Signal
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


class TopomapWidget(QWidget):
    """A widget dedicated to displaying an MNE topomap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, figsize=(4, 4), dpi=100)
        layout.addWidget(self.canvas)
        # Prevent the layout from having extra margins
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def plot(self, data, info, cmap="turbo", title="Topomap"):
        """Plots the topomap data on the canvas."""
        self.canvas.axes.clear()  # Clear previous plot

        im, _ = mne.viz.plot_topomap(
            data,
            info,
            cmap=cmap,
            show=False,  # Important: MNE should not show the plot itself
            axes=self.canvas.axes,
        )

        # Add a colorbar to the figure
        self.canvas.figure.colorbar(
            im, ax=self.canvas.axes, shrink=0.8, label=r"$\mu$V"
        )
        self.canvas.axes.set_title(title)
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
        self, figsize=None, dpi=100, params={}, progress_dialog=None, parent=None
    ):
        super().__init__(parent)

        self.setWindowTitle("Evoked Potential Plot")
        self.progress_dialog = progress_dialog
        self.figsize = figsize
        self.dpi = dpi
        self.params = params
        self.evoked = None
        self.label = None
        self.open_topomaps = []

        self.setMinimumSize(400, 500)

        self.init_ui()
        self.init_params()
        self.connect_events()
        self.update_plot()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.reference_widget = QRadioButton("Average Reference")

        self.topbar = QToolBar()
        self.topbar.setMovable(False)

        self.topbar.addWidget(self.reference_widget)
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

    def update_plot(self, evoked: mne.Evoked | None = None, label=None):
        """
        Clears the axes and plots the new evoked potential data.
        """
        self.reference_widget.setEnabled(True)
        self.reapply_style()

        self.canvas.axes.cla()
        for ax in self.canvas.figure.axes[1:]:
            self.canvas.figure.delaxes(ax)
        self.params = get_settings_store().get(
            "appearance/plots/global",
            self.default_params,
            legacy_keys=("plot_settings/plot_params",),
        )
        logger.info("Evoked plot: ", self.params)
        update_toolbar_color(self.toolbar)

        if evoked is not None and evoked:
            self.label = label
            self.evoked = evoked
            evoked_to_plot = self.evoked.copy()

            if self.evoked.proj:  # Data is already referenced
                self.reference_widget.setChecked(True)
                self.reference_widget.setEnabled(False)
            else:  # Data is not referenced, so the user has control.
                self.reference_widget.setEnabled(True)
                if self.reference_widget.isChecked():
                    evoked_to_plot.set_eeg_reference("average", projection=True)
                    evoked_to_plot.apply_proj()

            self.toolbar.show()
            self.canvas.show()
            self.canvas.axes.grid(True)

            # Calculate standard deviation for each dataset to determine z-order
            stds = np.std(evoked.data, axis=1)
            z_order_indices = np.argsort(stds)
            self.ch_names = evoked.ch_names

            # Plot the evoked data using MNE's plotting function
            baseline = self.params.get("baseline", (None, 0)) or (None, 0)

            # Check if the baseline period is valid for the given evoked data
            tmin, tmax = evoked_to_plot.times.min(), evoked_to_plot.times.max()
            baseline_tmin = baseline[0] if baseline[0] is not None else tmin
            baseline_tmax = baseline[1] if baseline[1] is not None else tmax

            if (
                baseline_tmin < tmin
                or baseline_tmax > tmax
                or baseline_tmin >= baseline_tmax
            ):
                logger.warning(
                    f"Baseline {baseline} is outside of data time range [{tmin:.3f}, {tmax:.3f}]. Skipping baseline correction."
                )
            else:
                evoked_to_plot.apply_baseline(baseline)
            fig = evoked_to_plot.plot(
                axes=self.canvas.axes,
                show=False,
                selectable=False,
                xlim=self.params.get("evoked_xlim", (-200, 500)),
                spatial_colors=self.params.get("spatial_colors", True),
                time_unit=self.params.get("time_unit", "ms"),
                gfp=self.params.get("gfp", False),
            )

            data_lines = self.canvas.axes.get_lines()

            self.original_zorders = {}
            for i, line in enumerate(data_lines):
                if i >= len(z_order_indices):
                    break
                z_order = z_order_indices[i]
                line.set_zorder(z_order)
                line.set_picker(5)  # Increased picker tolerance for easier clicking
                line.set_linewidth(1.5)
                line.set_label(self.ch_names[i])
                self.original_zorders[line] = z_order

            self.canvas.axes.axvline(
                0, linestyle="--", color=plt.rcParams["text.color"]
            )
            self.canvas.axes.set_xlabel(f'Time ({self.params.get("time_unit", "ms")})')
            self.canvas.axes.set_ylabel(r"Amplitude ($\mu$V)")
            self.canvas.axes.set_title("")
            self.canvas.axes.autoscale_view(scalex=False, scaley=True)
            self.title_label.setText(self.label or "Evoked Potential Plot")
            self.canvas.draw()
            self.show()
            self.raise_()
        else:
            self.title_label.setText("No Data Available")
            self.toolbar.hide()
            self.canvas.hide()
            self.canvas.draw()

    # === Event Handler Methods ===
    def connect_events(self):
        """Connects the required Matplotlib events to their callback methods."""
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", lambda e: self.scrolled.emit(e.button))
        self.reference_widget.toggled.connect(
            lambda: self.update_plot(self.evoked, self.label)
        )

    def on_button_press(self, event):
        """Callback for when a mouse button is pressed."""
        if event.inaxes is not self.canvas.axes:
            return
        if event.button == 3:
            if self.drag_span:
                self.drag_span.remove()
                self.drag_span = None
            self.drag_start_coords = (event.xdata, event.ydata)
            self.drag_span = self.canvas.axes.axvspan(
                self.drag_start_coords[0],
                self.drag_start_coords[0],
                facecolor=plt.rcParams["text.color"],
                alpha=0.3,
            )
            self.canvas.draw()
        elif event.button == 1:
            QTimer.singleShot(10, self._process_pick_event)

    def on_mouse_move(self, event):
        """Callback for mouse movement. Updates the axvspan during a drag."""
        current_time = time.time()
        if (current_time - self.last_update_time) < self.throttle_interval:
            # If not enough time has passed, do nothing
            return

        if (
            self.drag_start_coords is None
            or self.drag_span is None
            or event.inaxes is not self.canvas.axes
        ):
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
        """Callback for when a mouse button is released. Ends the drag operation on right-click."""
        if self.drag_start_coords is not None and event.button == 3:
            if self.drag_span:
                self.drag_span.remove()
                self.drag_span = None
                self.canvas.draw()

            if event.xdata is not None:
                start_x, start_y = self.drag_start_coords
                end_x, end_y = event.xdata, event.ydata
                self.drag_start_coords = None

                if abs(start_x - end_x) > 0.01:
                    self.handle_drag_selection(
                        start=(start_x, start_y), end=(end_x, end_y)
                    )

    def on_pick(self, event):
        """Callback for a pick event. Just adds the picked artist to a list."""
        if event.mouseevent.button == 1:
            self.picked_artists_in_click.append(event.artist)

    def _process_pick_event(self):
        """
        This function runs after a short delay following any left-click.
        It checks if any artists were collected by on_pick. If not, it was a background click.
        """
        if not self.picked_artists_in_click:
            print("Background click detected.")
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

        scale = 1e-3 if self.params.get("time_unit", "ms") == "ms" else 1
        idx = self.evoked.time_as_index((start_x * scale, end_x * scale))
        data = self.evoked.data[:, idx[0] : idx[1]].mean(axis=1)

        title = f"Topomap ({start_x:.2f} to {end_x:.2f}) {self.params.get('time_unit', 'ms')}"

        dialog = TopomapDialog(
            data=data,
            info=self.evoked.info,
            cmap=self.params.get("cmap", "turbo"),
            title=title,
            parent=self,
        )
        self.open_topomaps.append(dialog)
        dialog.finished.connect(lambda: self.open_topomaps.remove(dialog))
        dialog.show()

        print(
            f"Drag Event: Displayed topomap for time range {start_x:.2f} to {end_x:.2f}"
        )

    def handle_line_click(self, line_artist):
        """Called for the single, top-most line that was clicked, or None for a background click."""
        if line_artist is None:
            if self.canvas.axes.get_legend():
                self.canvas.axes.get_legend().remove()

            for line in self.canvas.axes.get_lines()[1:]:
                line.set_linewidth(1.5)
                line.set_alpha(1.0)
                line.set_zorder(self.original_zorders.get(line, 0))
            self.canvas.draw()
            return

        line_label = line_artist.get_label()
        print(f"Click Event: You clicked on the '{line_label}' line.")

        self.canvas.axes.legend(
            handles=[line_artist], fontsize="small", loc="upper right"
        )

        max_zorder = len(self.original_zorders) + 1
        line_artist.set_linewidth(3)
        line_artist.set_alpha(1.0)
        line_artist.set_zorder(max_zorder)

        for line in self.canvas.axes.get_lines()[1:]:
            if line is not line_artist:
                line.set_linewidth(1.5)
                line.set_alpha(0.3)
                line.set_zorder(self.original_zorders.get(line, 0))

        self.canvas.draw()

    def closeEvent(self, event):
        """
        Overrides the default close event to also close the progress dialog.
        """
        print("Closing the plot widget and the progress dialog.")
        if self.progress_dialog:
            self.progress_dialog.close()  # Or .accept()

        if len(self.open_topomaps) > 0:
            for topomap in self.open_topomaps:
                topomap.close()

        event.accept()
        super().closeEvent(event)

    def reapply_style(self):
        """
        Re-applies the current matplotlib style to the canvas and all its elements.
        This is useful for dynamically updating the plot theme (e.g., light/dark mode).
        """
        logger.info("Re-applying plot style...")

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

    def update_plot(self, evoked: mne.Evoked | None = None, label=None):
        """
        A convenience method to pass data directly to the contained widget.
        This is called "delegation".
        """
        self.plot_widget.update_plot(evoked, label)


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
