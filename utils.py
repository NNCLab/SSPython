import re
import sys
import os
import inspect
from functools import lru_cache, wraps
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Signal, Qt, QThread, QObject, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QProgressDialog, QMessageBox, QWidget
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import logging
import traceback
from typing import Any, Optional, Tuple, Type, Callable
import matplotlib.pyplot as plt

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional at import time
    pg = None

from core.app_settings import get_settings_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


THEME_TOKENS = {
    "dark": {
        "background": "#0c1114",
        "panel": "#151d21",
        "border": "#223038",
        "text": "#e7ecef",
        "muted": "#84939a",
        "accent": "#3a8a9b",
        "accent_soft": "#6db3c0",
        "accent_fill": "#235f6c",
        "selection": "#1a3037",
        "icon": "#e7ecef",
        "plot_background": "#11181c",
        "plot_foreground": "#e7ecef",
        "grid": "#2a3941",
        "roi": "#3a8a9b",
    },
    "light": {
        "background": "#edf1f2",
        "panel": "#ffffff",
        "border": "#d7e0e3",
        "text": "#16242a",
        "muted": "#64747c",
        "accent": "#2f7e8d",
        "accent_soft": "#5f97a3",
        "accent_fill": "#d8eaee",
        "selection": "#dfeff3",
        "icon": "#16242a",
        "plot_background": "#ffffff",
        "plot_foreground": "#16242a",
        "grid": "#c7d5da",
        "roi": "#2f7e8d",
    },
}


def get_path(relative_path):
    """
    Get the absolute path to a resource, works for dev and for PyInstaller.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Not running in a PyInstaller bundle, use the current directory
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def current_theme_name() -> str:
    settings_store = get_settings_store()
    return settings_store.get("appearance/theme", "dark", legacy_keys=("theme",))


def theme_tokens(theme: str | None = None) -> dict[str, str]:
    theme_name = theme or current_theme_name()
    return dict(THEME_TOKENS.get(theme_name, THEME_TOKENS["dark"]))


def apply_pyqtgraph_theme(theme: str | None = None):
    if pg is None:
        return

    tokens = theme_tokens(theme)
    pg.setConfigOption("background", tokens["plot_background"])
    pg.setConfigOption("foreground", tokens["plot_foreground"])


@lru_cache(maxsize=64)
def _render_svg_pixmap(svg_path: str, color_hex: str, width: int, height: int) -> QPixmap:
    svg_text = Path(svg_path).read_text(encoding="utf-8")
    if "fill=" in svg_text:
        svg_text = re.sub(r'fill="[^"]*"', f'fill="{color_hex}"', svg_text)
    else:
        svg_text = svg_text.replace("<svg ", f'<svg fill="{color_hex}" ', 1)

    renderer = QSvgRenderer(QByteArray(svg_text.encode("utf-8")))
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    return pixmap


def themed_svg_icon(relative_path: str, *, color: str | QColor | None = None, size: int | QSize = 20) -> QIcon:
    if isinstance(size, QSize):
        width, height = size.width(), size.height()
    else:
        width = height = int(size)

    svg_path = get_path(relative_path)
    tint = QColor(color or theme_tokens()["icon"]).name()
    return QIcon(_render_svg_pixmap(svg_path, tint, width, height))


# === Style ===
def apply_theme(theme: str | None = None):
    """
    Loads and applies a QSS stylesheet to the application.
    If no theme is provided, it reads from settings.
    """
    app = QApplication.instance()
    if not app:
        return

    settings_store = get_settings_store()
    if theme is None:
        theme = settings_store.get("appearance/theme", "dark", legacy_keys=("theme",))

    # Handle Qt.ColorScheme enum
    if isinstance(theme, Qt.ColorScheme):
        if theme == Qt.ColorScheme.Dark:
            theme = "dark"
        else:
            theme = "light"

    qss_path = get_path(f"style/{theme}_theme.qss")
    try:
        with open(qss_path, "r") as f:
            stylesheet = f.read()
        app.setStyleSheet(stylesheet)
        app.setProperty("currentTheme", theme)
        settings_store.set("appearance/theme", theme)
        settings_store.set("theme", theme)
    except FileNotFoundError:
        logger.warning(f"Stylesheet not found at: {qss_path}")

    apply_pyqtgraph_theme(theme)

    # Apply a corresponding matplotlib style
    mpl_style = "dark_background" if theme == "dark" else "default"
    plt.style.use(mpl_style)
    settings_store.set("appearance/plot_style", mpl_style)
    settings_store.set("plot_style", mpl_style)
    settings_store.sync()


def toggle_theme():
    """Toggles between light and dark themes."""
    settings_store = get_settings_store()
    current_theme = settings_store.get("appearance/theme", "dark", legacy_keys=("theme",))
    new_theme = "light" if current_theme == "dark" else "dark"
    apply_theme(new_theme)


def use_plot_style(plot_func):
    """Decorator to apply the current matplotlib style before plotting."""

    @wraps(plot_func)
    def wrapper(*args, **kwargs):
        # This is where the magic happens
        settings_store = get_settings_store()
        style_to_use = settings_store.get(
            "appearance/plot_style",
            "default",
            legacy_keys=("plot_style",),
        )
        plt.style.use(style_to_use)
        # Now, run the original plotting function
        return plot_func(*args, **kwargs)

    return wrapper


def update_toolbar_color(toolbar: NavigationToolbar2QT):
    """
    Updates the toolbar icons to match the current theme.
    
    .. warning::
        This function is highly fragile as it relies on the internal `_actions`
        and `_icon` attributes of Matplotlib's `NavigationToolbar2QT`. These
        are not public APIs and may change or be removed in future versions
        of Matplotlib, which would break this function.
    """
    for _, _, image_file, callback in toolbar.toolitems:
        if not image_file:
            continue
        tool = toolbar._actions.get(callback)
        if tool:
            tool.setIcon(toolbar._icon(image_file + ".png"))


# === Parsers ===
def parse_tuple(
    text: Optional[str], dtype: Type = float, scale: float = 1
) -> Optional[Tuple[Any, ...]]:
    """Parses a comma-separated string into a tuple of a specified type."""
    if not text or not text.strip():
        return None
    cleaned_text = text.strip().strip("[]()")
    try:
        parts = cleaned_text.split(",")
        result = [
            None if p.strip().lower() == "none" else dtype(p.strip()) for p in parts
        ]
        # multiply by scale
        result = [i * scale if i is not None else None for i in result]
        return tuple(result)
    except (ValueError, IndexError):
        raise ValueError(
            f"Invalid input for tuple: '{text}'. "
            "Expected a comma-separated format like '1, 2' or '(None, 5)'."
        )


def parse_numeric(
    text: Optional[str], dtype: Type[int] | Type[float] = float
) -> Optional[int | float]:
    """Parses a string into an integer or float."""
    if not text or not text.strip():
        return None
    cleaned_text = text.strip().lower()
    if cleaned_text in ["none", ""]:
        return None
    try:
        return dtype(cleaned_text)
    except (ValueError, IndexError):
        raise ValueError(
            f"Invalid numeric input: '{text}'. Expected a number like '123' or '45.6'."
        )


def to_display_string(value):
    """
    Converts a parameter value into a string suitable for a QLineEdit.
    Handles None, lists, tuples, and other types.
    """
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        # Converts (1.0, 45.0) into "1.0, 45.0"
        # Also converts (None, 0) into "None, 0"
        parts = (str(v) if v is not None else "None" for v in value)
        return ", ".join(parts)
    return str(value)


# === Worker ===
class TqdmWriter(QObject):
    """A stream-like object that captures tqdm's output and emits Qt signals."""

    progress = Signal(int)
    progress_text = Signal(str)

    def write(self, text: str):
        """
        Parses tqdm output to extract percentage and description.
        - Looks for a pattern like "Description: 75%|███ | 3/4 [00:01<00:00, 2.99it/s]"
        - Extracts description and percentage.
        """
        # The regex is improved to be more robust.
        # It handles an optional description, the percentage, and the progress bar.
        # It's non-greedy `(.*?)` to handle various tqdm formats.
        match = re.search(r"^(.*?):.*?\s+(\d+)%\|", text.strip())
        if not match and "%" in text:
            # Fallback for formats without a description
            match = re.search(r"(\d+)%\|", text.strip())

        if match:
            groups = match.groups()
            if len(groups) == 2:
                description = groups[0].strip()
                if description:
                    self.progress_text.emit(description)
                percentage = int(groups[1])
            else:
                percentage = int(groups[0])
            self.progress.emit(percentage)

    def flush(self):
        """tqdm requires a flush method, but we don't need to do anything."""
        pass


class QtLogHandler(logging.Handler, QObject):
    """
    A custom logging handler that emits a Qt Signal for each log record.
    Inherits from QObject to handle signals.
    """

    new_record = Signal(str)

    def __init__(self, parent: QObject | None = None):
        # Initialize both parent classes correctly.
        logging.Handler.__init__(self)
        QObject.__init__(self, parent)

    def emit(self, record: logging.LogRecord):
        """Formats the log record and emits it as a Signal."""
        msg = self.format(record)
        self.new_record.emit(msg)


class InterruptionRequestedError(Exception):
    """Custom exception to signal that the task was cancelled."""

    pass

# (other imports remain the same)

# ... (rest of the file is the same until the Worker class) ...

class Worker(QThread):
    """
    A professional QThread worker for running tasks in the background.
    This worker is designed for cooperative cancellation. It checks if the
    target function can accept an `is_cancelled` keyword argument. If so,
    it passes a callable that should be checked periodically to allow for
    graceful termination.
    Signals:
        - finished(object): Emitted when the task completes successfully.
        - error(str): Emitted when the task raises an exception.
        - cancelled(): Emitted when the task is cancelled.
        - progress(int): For tqdm-style integer progress (0-100).
        - progress_text(str): For text updates.
    """
    progress = Signal(int)
    progress_text = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        task_function: Callable,
        *args,
        parent: QWidget | None = None,
        add_loggers: list[str] | str | None = None,
        **kwargs,
    ):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.task_function = task_function
        self.args = args
        self.kwargs = kwargs
        self._result: Any = None
        self._error: str | None = None
        self._is_cancellable: bool = False

        # Setup progress and logging handlers
        self.tqdm_writer = TqdmWriter()
        self.log_handler = QtLogHandler()
        self.tqdm_writer.progress.connect(self.progress)
        self.tqdm_writer.progress_text.connect(self.progress_text)
        self.log_handler.new_record.connect(self.progress_text)

        # Normalize loggers to a list
        if isinstance(add_loggers, str):
            self.loggers_to_add = [add_loggers]
        elif add_loggers is None:
            self.loggers_to_add = [""]  # Default to root logger
        else:
            self.loggers_to_add = add_loggers

        # Check if the task function is cancellable
        try:
            sig = inspect.signature(self.task_function)
            self._is_cancellable = "is_cancelled" in sig.parameters
        except (ValueError, TypeError):
            self._is_cancellable = False

    def run(self):
        """
        Executes the task, redirecting stderr and handling logs.
        Conditionally injects `is_cancelled` if the task supports it.
        """
        active_loggers = [logging.getLogger(name) for name in self.loggers_to_add]
        original_stderr = sys.stderr
        try:
            for logger_instance in active_loggers:
                logger_instance.addHandler(self.log_handler)
            sys.stderr = self.tqdm_writer

            task_kwargs = self.kwargs.copy()
            if self._is_cancellable:
                task_kwargs["is_cancelled"] = self.check_interruption

            self._result = self.task_function(*self.args, **task_kwargs)

        except InterruptionRequestedError:
            self._error = "Task was cancelled by the user."
            self.logger.info(self._error)
            self.cancelled.emit()
            return

        except Exception:
            self._error = traceback.format_exc()
            self.logger.error(f"An error occurred in the worker:\n{self._error}")
            self.error.emit(self._error)

        else:
            self.finished.emit(self._result)

        finally:
            sys.stderr = original_stderr
            for logger_instance in active_loggers:
                logger_instance.removeHandler(self.log_handler)
    
    def exec_with_dialog(
        self,
        title: str = "Processing...",
        label: str = "Starting task...",
        is_cancellable: bool | None = None,
        show_error_msg: bool = True,
    ) -> Any:
        """
        Convenience method to run the worker and show a modal progress dialog.
        The cancel button is only shown if the task is cancellable.
        """
        parent = self.parent()
        if not isinstance(parent, QWidget):
            parent = None

        show_cancel_button = self._is_cancellable
        if is_cancellable is not None:
            show_cancel_button = is_cancellable and self._is_cancellable

        dialog = QProgressDialog(label, "Cancel", 0, 100, parent)
        dialog.setWindowTitle(title)
        dialog.setMinimumDuration(500)
        dialog.setAutoClose(False)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)

        if not show_cancel_button:
            dialog.setCancelButton(None)
        else:
            dialog.canceled.connect(self.requestInterruption)

        self.progress.connect(dialog.setValue)
        self.progress_text.connect(dialog.setLabelText)
        self.finished.connect(dialog.accept)
        self.error.connect(dialog.reject)
        self.cancelled.connect(dialog.reject)

        self.start()
        dialog_code = dialog.exec()

        if dialog_code == QProgressDialog.DialogCode.Rejected and self._error:
            if not self.isInterruptionRequested() and show_error_msg:
                QMessageBox.critical(
                    parent, "Error", f"An error occurred:\n{self._error}"
                )
            return None

        return self._result

    def check_interruption(self) -> bool:
        """
        Checks if an interruption has been requested and raises an exception if it has.
        This function is passed to the task.
        """
        if self.isInterruptionRequested():
            raise InterruptionRequestedError
        return False



