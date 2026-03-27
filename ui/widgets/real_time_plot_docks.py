from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pyqtgraph as pg
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QDockWidget, QVBoxLayout, QWidget


class BasePlotDock(QDockWidget):
    DISPLAY_POINTS_PER_PIXEL = 1.35
    DISPLAY_MIN_POINTS = 320
    DISPLAY_MAX_POINTS = 6000

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setObjectName(title.replace(" ", "") + "Dock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setWidget(container)

        self._draw_pending = False

    def request_draw(self):
        if self._draw_pending:
            return
        self._draw_pending = True
        QTimer.singleShot(0, self._flush_draw)

    def _flush_draw(self):
        self._draw_pending = False
        self.canvas.draw_idle()

    def _display_indices(self, sample_count: int) -> np.ndarray | slice:
        sample_count = int(sample_count)
        if sample_count <= 0:
            return slice(0, 0)

        logical_width = max(1, int(self.canvas.width()))
        dpr = float(self.canvas.devicePixelRatioF()) if hasattr(self.canvas, "devicePixelRatioF") else 1.0
        pixel_width = max(1, int(round(logical_width * max(1.0, dpr))))
        max_points = int(np.clip(
            np.ceil(pixel_width * self.DISPLAY_POINTS_PER_PIXEL),
            self.DISPLAY_MIN_POINTS,
            self.DISPLAY_MAX_POINTS,
        ))
        if sample_count <= max_points:
            return slice(None)

        stride = max(1, int(np.ceil(sample_count / max_points)))
        indices = np.arange(0, sample_count, stride, dtype=int)
        if indices[-1] != sample_count - 1:
            indices = np.append(indices, sample_count - 1)
        return indices

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rerender_cached_data()

    def _rerender_cached_data(self):
        """Subclasses can reapply cached full-resolution data after a resize."""
        return


class RawMonitorDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Raw Data Monitor", parent)
        self.setObjectName("RawDataMonitorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.hideButtons()
        self.plot_item.showGrid(x=True, y=False, alpha=0.22)
        self.plot_item.setClipToView(True)
        self.plot_item.setDownsampling(auto=True, mode="peak")
        self.plot_item.setMouseEnabled(x=True, y=False)
        self.plot_item.setMenuEnabled(False)
        layout.addWidget(self.plot_widget)
        self.setWidget(container)

        self.ch_names: list[str] = []
        self.colors: list[tuple[float, float, float, float]] = []
        self.raw_offsets = np.array([], dtype=float)
        self.curves = []
        self.latest_time_axis = np.array([], dtype=float)
        self.latest_scaled_data = np.empty((0, 0), dtype=float)
        self._visible_count = 0

    @staticmethod
    def _pen_from_color(color: tuple[float, float, float, float]):
        rgba = [int(np.clip(component, 0.0, 1.0) * 255) for component in color[:4]]
        if len(rgba) == 3:
            rgba.append(255)
        return pg.mkPen(color=tuple(rgba), width=1.2)

    def _update_y_axis(self):
        if not self.ch_names:
            return
        n_show = self._visible_count or len(self.ch_names)
        n_show = max(1, min(n_show, len(self.ch_names)))
        max_y = len(self.ch_names)
        min_y = max_y - n_show
        tick_pairs = [(float(self.raw_offsets[index]), self.ch_names[index]) for index in range(n_show)]
        self.plot_item.getAxis("left").setTicks([tick_pairs])
        self.plot_item.setYRange(min_y - 0.5, max_y - 0.5, padding=0.0)

    def configure(self, ch_names: list[str], colors: list[tuple[float, float, float, float]], stream_duration: float):
        self.ch_names = list(ch_names)
        self.colors = list(colors)
        self.raw_offsets = np.arange(len(self.ch_names), dtype=float)[::-1]
        self._visible_count = len(self.ch_names)

        self.plot_item.clear()
        self.plot_item.setLabel("bottom", "Time (s)")
        self.plot_item.setXRange(-float(stream_duration), 0.0, padding=0.0)
        self.plot_item.setLimits(xMin=-float(stream_duration), xMax=0.0)
        self.curves = [
            self.plot_item.plot([], [], pen=self._pen_from_color(self.colors[index]), connect="finite")
            for index in range(len(self.ch_names))
        ]
        for curve in self.curves:
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
        self._update_y_axis()

    def set_visible_channels(self, n_show: int):
        if not self.ch_names:
            return
        self._visible_count = max(1, min(int(n_show), len(self.ch_names)))
        self._update_y_axis()

    def update_data(self, time_axis: np.ndarray, scaled_data: np.ndarray):
        self.latest_time_axis = np.asarray(time_axis, dtype=float)
        self.latest_scaled_data = np.asarray(scaled_data, dtype=float)
        if self.latest_scaled_data.size == 0 or self.latest_time_axis.size == 0:
            return
        if self.latest_scaled_data.shape[0] != len(self.curves):
            return
        for index, curve in enumerate(self.curves):
            curve.setData(self.latest_time_axis, self.latest_scaled_data[index, :] + self.raw_offsets[index], connect="finite")

    def refresh_theme(self, tokens: dict[str, str], colors: list[tuple[float, float, float, float]]):
        self.colors = list(colors)
        self.plot_widget.setBackground(tokens["plot_background"])
        axis_pen = pg.mkPen(tokens["border"])
        text_pen = pg.mkPen(tokens["text"])
        for axis_name in ("bottom", "left"):
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(text_pen)
        self.plot_item.showGrid(x=True, y=False, alpha=0.22)
        for index, curve in enumerate(self.curves):
            curve.setPen(self._pen_from_color(self.colors[index]))
        self._update_y_axis()


class EvokedButterflyDock(BasePlotDock):
    roi_changed = Signal(float, float)
    channel_left_clicked = Signal(str)
    channel_right_clicked = Signal(str)
    CLICK_DRAG_THRESHOLD_PX = 6
    CHANNEL_PICK_TOLERANCE_RATIO = 0.045

    def __init__(self, parent=None):
        super().__init__("Evoked Potentials", parent)
        self.figure.subplots_adjust(left=0.1, right=0.99, bottom=0.12, top=0.98)
        self.ax = self.figure.add_subplot(111)
        self.times_ms = np.array([], dtype=float)
        self.ch_names: list[str] = []
        self.colors: list[tuple[float, float, float, float]] = []
        self.lines = []
        self.roi_region: tuple[float, float] | None = None
        self.latest_mean_data_uV = np.empty((0, 0), dtype=float)
        self.latest_bads: list[str] = []
        self.latest_global_limits = (-10.0, 10.0)
        self._mouse_press = None
        self._drag_span = None
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)

    def configure(self, times_ms: np.ndarray, ch_names: list[str], colors: list[tuple[float, float, float, float]]):
        self.times_ms = np.asarray(times_ms, dtype=float)
        self.ch_names = list(ch_names)
        self.colors = list(colors)

        self.ax.clear()
        self.lines = [
            self.ax.plot(
                self.times_ms,
                np.full_like(self.times_ms, np.nan, dtype=float),
                linewidth=1.3,
                color=self.colors[index],
            )[0]
            for index in range(len(self.ch_names))
        ]
        self.ax.set_xlim(float(self.times_ms[0]), float(self.times_ms[-1]))
        self.ax.set_ylim(-10.0, 10.0)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Potential (uV)")
        self._drag_span = self.ax.axvspan(0.0, 0.0, facecolor="#2f7e8d", alpha=0.22, visible=False, zorder=0.1)
        self.set_roi_region(self._default_roi_region(), emit=False)

    def _default_roi_region(self) -> tuple[float, float]:
        if self.times_ms.size == 0:
            return (0.0, 0.0)
        min_x = float(self.times_ms[0])
        max_x = float(self.times_ms[-1])
        span = max(10.0, min(30.0, (max_x - min_x) * 0.08))
        start_ms = 0.0 if min_x <= 0.0 <= max_x else min_x
        end_ms = min(max_x, start_ms + span)
        if end_ms <= start_ms:
            end_ms = min(max_x, start_ms + 1.0)
        return (start_ms, end_ms)

    def set_roi_region(self, region: tuple[float, float], *, emit: bool):
        if self.times_ms.size == 0:
            return
        min_x = float(self.times_ms[0])
        max_x = float(self.times_ms[-1])
        start_ms, end_ms = sorted((float(region[0]), float(region[1])))
        start_ms = float(np.clip(start_ms, min_x, max_x))
        end_ms = float(np.clip(end_ms, min_x, max_x))
        if end_ms <= start_ms:
            return
        self.roi_region = (start_ms, end_ms)
        if emit:
            self.roi_changed.emit(start_ms, end_ms)

    def _clamped_xdata(self, event) -> float | None:
        if self.times_ms.size == 0:
            return None
        min_x = float(self.times_ms[0])
        max_x = float(self.times_ms[-1])
        if event.xdata is not None:
            return float(np.clip(float(event.xdata), min_x, max_x))
        if event.x is None or event.y is None:
            return None
        try:
            xdata = float(self.ax.transData.inverted().transform((event.x, event.y))[0])
        except Exception:
            return None
        return float(np.clip(xdata, min_x, max_x))

    def _set_drag_span(self, start_ms: float, end_ms: float, *, visible: bool):
        if self._drag_span is None:
            return
        left = float(min(start_ms, end_ms))
        right = float(max(start_ms, end_ms))
        self._drag_span.set_x(left)
        self._drag_span.set_width(max(0.0, right - left))
        self._drag_span.set_visible(visible and right > left)
        self.request_draw()

    def _on_button_press(self, event):
        if event.inaxes is not self.ax:
            self._mouse_press = None
            return
        if event.button not in (MouseButton.LEFT, MouseButton.RIGHT):
            self._mouse_press = None
            return
        if event.x is None or event.y is None or event.xdata is None or event.ydata is None:
            self._mouse_press = None
            return
        self._mouse_press = {
            "button": event.button,
            "x": float(event.x),
            "y": float(event.y),
            "xdata": float(event.xdata),
            "ydata": float(event.ydata),
        }
        if event.button == MouseButton.LEFT:
            self._set_drag_span(float(event.xdata), float(event.xdata), visible=False)

    def _on_mouse_move(self, event):
        press = self._mouse_press
        if press is None or press["button"] != MouseButton.LEFT:
            return
        current_x = self._clamped_xdata(event)
        if current_x is None:
            return
        moved_px = 0.0
        if event.x is not None and event.y is not None:
            moved_px = float(np.hypot(float(event.x) - press["x"], float(event.y) - press["y"]))
        self._set_drag_span(press["xdata"], current_x, visible=moved_px > 1.0)

    def _on_button_release(self, event):
        press = self._mouse_press
        self._mouse_press = None
        if press is None or event.button != press["button"]:
            return
        current_x = self._clamped_xdata(event)
        self._set_drag_span(press["xdata"], current_x if current_x is not None else press["xdata"], visible=False)
        if event.x is None or event.y is None:
            return

        dx = float(event.x) - press["x"]
        dy = float(event.y) - press["y"]
        moved_px = float(np.hypot(dx, dy))

        if (
            press["button"] == MouseButton.LEFT
            and current_x is not None
            and moved_px > self.CLICK_DRAG_THRESHOLD_PX
            and abs(float(current_x) - press["xdata"]) > 0.25
        ):
            self.set_roi_region((press["xdata"], float(current_x)), emit=True)
            return

        if moved_px > self.CLICK_DRAG_THRESHOLD_PX:
            return

        ch_name = self._pick_channel(event)
        if ch_name is None:
            return
        if press["button"] == MouseButton.LEFT:
            self.channel_left_clicked.emit(ch_name)
        elif press["button"] == MouseButton.RIGHT:
            self.channel_right_clicked.emit(ch_name)

    def _pick_channel(self, event) -> str | None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return None
        if self.latest_mean_data_uV.size == 0 or self.times_ms.size == 0:
            return None

        idx = int(np.clip(np.searchsorted(self.times_ms, float(event.xdata)), 0, len(self.times_ms) - 1))
        if idx > 0 and abs(self.times_ms[idx - 1] - float(event.xdata)) <= abs(self.times_ms[idx] - float(event.xdata)):
            idx -= 1

        values = self.latest_mean_data_uV[:, idx]
        valid_mask = np.isfinite(values) & np.array([name not in self.latest_bads for name in self.ch_names], dtype=bool)
        if not np.any(valid_mask):
            return None

        y_range = self.ax.get_ylim()
        tolerance = max(0.5, abs(float(y_range[1]) - float(y_range[0])) * self.CHANNEL_PICK_TOLERANCE_RATIO)
        diffs = np.abs(values - float(event.ydata))
        diffs[~valid_mask] = np.inf
        best_index = int(np.argmin(diffs))
        if not np.isfinite(diffs[best_index]) or diffs[best_index] > tolerance:
            return None
        return self.ch_names[best_index]

    def update_data(
        self,
        mean_data_uV: np.ndarray,
        bads: list[str],
        global_min: float,
        global_max: float,
    ):
        self.latest_mean_data_uV = np.asarray(mean_data_uV, dtype=float)
        self.latest_bads = list(bads)
        self.latest_global_limits = (float(global_min), float(global_max))
        self._rerender_cached_data()

    def _rerender_cached_data(self):
        if self.latest_mean_data_uV.size == 0 or self.times_ms.size == 0:
            return
        indices = self._display_indices(self.times_ms.size)
        time_axis = self.times_ms[indices]
        for index, line in enumerate(self.lines):
            if self.ch_names[index] in self.latest_bads:
                line.set_data(time_axis, np.full_like(time_axis, np.nan))
            else:
                line.set_data(time_axis, self.latest_mean_data_uV[index, indices])

        global_min, global_max = self.latest_global_limits
        y_padding = max(1.0, (global_max - global_min) * 0.12)
        self.ax.set_ylim(global_min - y_padding, global_max + y_padding)
        self.request_draw()

    def refresh_theme(self, tokens: dict[str, str], colors: list[tuple[float, float, float, float]]):
        self.colors = list(colors)
        self.figure.patch.set_facecolor(tokens["panel"])
        self.ax.set_facecolor(tokens["plot_background"])
        self.ax.tick_params(colors=tokens["text"])
        self.ax.xaxis.label.set_color(tokens["text"])
        self.ax.yaxis.label.set_color(tokens["text"])
        self.ax.title.set_color(tokens["text"])
        for spine in self.ax.spines.values():
            spine.set_color(tokens["border"])
        self.ax.grid(False)
        self.ax.grid(True, axis="x", color=tokens["grid"], alpha=0.28, linewidth=0.7)
        self.ax.grid(True, axis="y", color=tokens["grid"], alpha=0.28, linewidth=0.7)
        for index, line in enumerate(self.lines):
            line.set_color(self.colors[index])
            line.set_linewidth(1.3)
        if self._drag_span is not None:
            self._drag_span.set_facecolor(tokens["roi"])
            self._drag_span.set_alpha(0.22)
        self.request_draw()


class ChannelLayoutDock(BasePlotDock):
    channel_left_clicked = Signal(str)
    channel_right_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Channel Layout", parent)
        self.figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        self.host_ax = self.figure.add_subplot(111)
        self.times_ms = np.array([], dtype=float)
        self.ch_names: list[str] = []
        self.colors: list[tuple[float, float, float, float]] = []
        self.channel_axes = []
        self.channel_lines = []
        self.axis_to_channel: dict[object, str] = {}
        self.head_outline: Circle | None = None
        self.scale_mode = "Global Auto-Scale"
        self.latest_mean_data_uV = np.empty((0, 0), dtype=float)
        self.latest_bads: list[str] = []
        self.latest_global_limits = (-10.0, 10.0)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def configure(
        self,
        times_ms: np.ndarray,
        ch_names: list[str],
        colors: list[tuple[float, float, float, float]],
        positions: np.ndarray,
    ):
        self.times_ms = np.asarray(times_ms, dtype=float)
        self.ch_names = list(ch_names)
        self.colors = list(colors)

        self.figure.clear()
        self.host_ax = self.figure.add_subplot(111)
        self.host_ax.set_aspect("equal")
        self.host_ax.set_xlim(0.0, 1.0)
        self.host_ax.set_ylim(0.0, 1.0)
        self.host_ax.set_xticks([])
        self.host_ax.set_yticks([])
        for spine in self.host_ax.spines.values():
            spine.set_visible(False)

        self.head_outline = Circle((0.5, 0.5), 0.47, fill=False, linewidth=1.2)
        self.host_ax.add_patch(self.head_outline)

        self.channel_axes = []
        self.channel_lines = []
        self.axis_to_channel = {}

        for index, ch_name in enumerate(self.ch_names):
            if index >= positions.shape[0]:
                break
            left, bottom, width, height = positions[index]
            inset_ax = self.host_ax.inset_axes([left, bottom, width, height])
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_title(ch_name, fontsize=7, pad=1)
            inset_ax.margins(x=0.04, y=0.12)
            line, = inset_ax.plot(
                self.times_ms,
                np.full_like(self.times_ms, np.nan, dtype=float),
                linewidth=1.2,
                color=self.colors[index],
            )
            self.channel_axes.append(inset_ax)
            self.channel_lines.append(line)
            self.axis_to_channel[inset_ax] = ch_name

        self._update_axis_limits(-10.0, 10.0)

    @staticmethod
    def _centered_limits(min_value: float, max_value: float) -> tuple[float, float]:
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            return (-10.0, 10.0)
        span = max(1.0, float(max_value - min_value))
        padding = max(1.0, span * 0.18)
        center = float(min_value + max_value) * 0.5
        half_range = max(1.0, span * 0.5 + padding)
        return center - half_range, center + half_range

    def _global_limits(self, global_min: float, global_max: float) -> tuple[float, float]:
        return self._centered_limits(global_min, global_max)

    def _local_limits(self, channel_data: np.ndarray, fallback_limits: tuple[float, float]) -> tuple[float, float]:
        finite_values = np.asarray(channel_data, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return fallback_limits
        return self._centered_limits(float(np.min(finite_values)), float(np.max(finite_values)))

    def _update_axis_limits(self, global_min: float, global_max: float, mean_data_uV: np.ndarray | None = None):
        if self.times_ms.size == 0:
            return
        fallback_limits = self._global_limits(global_min, global_max)
        for index, axis in enumerate(self.channel_axes):
            axis.set_xlim(float(self.times_ms[0]), float(self.times_ms[-1]))
            if self.scale_mode == "Local Auto-Scale" and mean_data_uV is not None and index < mean_data_uV.shape[0]:
                y_min, y_max = self._local_limits(mean_data_uV[index], fallback_limits)
            else:
                y_min, y_max = fallback_limits
            axis.set_ylim(y_min, y_max)

    def update_data(
        self,
        mean_data_uV: np.ndarray,
        bads: list[str],
        global_min: float,
        global_max: float,
        scale_mode: str,
    ):
        self.latest_mean_data_uV = np.asarray(mean_data_uV, dtype=float)
        self.latest_bads = list(bads)
        self.latest_global_limits = (float(global_min), float(global_max))
        self.scale_mode = scale_mode
        self._rerender_cached_data()

    def _rerender_cached_data(self):
        if self.times_ms.size == 0 or self.latest_mean_data_uV.size == 0:
            return
        global_min, global_max = self.latest_global_limits
        indices = self._display_indices(self.times_ms.size)
        time_axis = self.times_ms[indices]
        mean_data_uV = self.latest_mean_data_uV[:, indices]
        self._update_axis_limits(global_min, global_max, mean_data_uV)
        for index, line in enumerate(self.channel_lines):
            line.set_data(time_axis, mean_data_uV[index])
            is_bad = self.ch_names[index] in self.latest_bads
            line.set_linestyle("--" if is_bad else "-")
            line.set_alpha(0.4 if is_bad else 1.0)
        self.request_draw()

    def refresh_theme(
        self,
        tokens: dict[str, str],
        colors: list[tuple[float, float, float, float]],
        bads: list[str],
    ):
        self.colors = list(colors)
        self.figure.patch.set_facecolor(tokens["panel"])
        self.host_ax.set_facecolor(tokens["plot_background"])
        self.host_ax.set_xticks([])
        self.host_ax.set_yticks([])
        for spine in self.host_ax.spines.values():
            spine.set_visible(False)
        if self.head_outline is not None:
            self.head_outline.set_edgecolor(tokens["border"])

        for index, axis in enumerate(self.channel_axes):
            axis.set_facecolor(tokens["plot_background"])
            axis.tick_params(colors=tokens["text"], length=0)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(tokens["border"])
            is_bad = self.ch_names[index] in bads
            line = self.channel_lines[index]
            line.set_color(self.colors[index])
            line.set_linewidth(1.2)
            line.set_linestyle("--" if is_bad else "-")
            line.set_alpha(0.4 if is_bad else 1.0)
            axis.title.set_color(tokens["muted"] if is_bad else mcolors.to_hex(self.colors[index]))
            axis.title.set_alpha(0.82 if is_bad else 1.0)

        self.request_draw()

    def _on_canvas_click(self, event):
        if event.x is None or event.y is None:
            return
        ch_name = self.axis_to_channel.get(event.inaxes)
        if ch_name is None:
            for axis, axis_channel in self.axis_to_channel.items():
                if axis.get_window_extent().contains(event.x, event.y):
                    ch_name = axis_channel
                    break
        if ch_name is None:
            return
        if event.button == MouseButton.LEFT:
            self.channel_left_clicked.emit(ch_name)
        elif event.button == MouseButton.RIGHT:
            self.channel_right_clicked.emit(ch_name)
