import sys
import uuid
import time
import numpy as np
import mne
import mne_lsl
import pyqtgraph as pg
from collections import deque
from scipy.spatial.distance import pdist

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QDialog,
    QDockWidget,
    QToolBar,
    QLineEdit,
    QFormLayout,
    QMessageBox,
    QHBoxLayout,
    QDialogButtonBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QProgressDialog,
    QGroupBox,
    QFileDialog,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
    QMainWindow,
    QFrame,
    QScrollArea,
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, QSettings
from PySide6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .tools.optional_range_widget import OptionalRangeWidget
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import json

from core.app_settings import get_settings_store
from utils import apply_pyqtgraph_theme, current_theme_name, theme_tokens


DEFAULT_REAL_TIME_MONTAGE = "easycap-M1"
CHANNEL_TYPE_OPTIONS = ["eeg", "eog", "stim", "reference", "times", "bad"]


def builtin_montage_names() -> list[str]:
    return ["None"] + sorted(mne.channels.get_builtin_montages())


def suggest_channel_type(channel_name: str) -> str:
    name = channel_name.strip().lower()
    if any(token in name for token in ("stim", "sti", "trigger", "trig", "marker", "status", "event")):
        return "stim"
    if any(token in name for token in ("eog", "veog", "heog")):
        return "eog"
    if "time" in name:
        return "times"
    if any(token in name for token in ("ref", "masto", "ear")):
        return "reference"
    return "eeg"


def build_epoch_stream_configuration(
    channel_settings: list[dict],
) -> tuple[int | dict[str, int] | None, str | list[str] | None, list[str]]:
    event_channels: list[str] = []
    event_name_map: dict[str, int] = {}
    bads: list[str] = []

    for channel in channel_settings:
        channel_name = channel.get("name", "")
        enabled = bool(channel.get("enabled", True))
        channel_type = str(channel.get("type", "eeg"))
        event_id = int(channel.get("event_id", 0) or 0)
        event_name = str(channel.get("event_name", "") or "").strip()

        if not enabled:
            bads.append(channel_name)
            continue

        if channel_type == "bad":
            bads.append(channel_name)
            continue

        if channel_type == "stim":
            event_channels.append(channel_name)
            if event_id > 0:
                label = event_name or f"Event {event_id}"
                suffix = 2
                original_label = label
                while label in event_name_map and event_name_map[label] != event_id:
                    label = f"{original_label} ({suffix})"
                    suffix += 1
                event_name_map[label] = event_id

    if event_name_map:
        if len(event_name_map) == 1:
            only_label, only_value = next(iter(event_name_map.items()))
            event_id_config = only_value if only_label == f"Event {only_value}" else {only_label: only_value}
        else:
            event_id_config = event_name_map
    else:
        event_id_config = None

    if len(event_channels) == 1:
        event_channel_config: str | list[str] | None = event_channels[0]
    else:
        event_channel_config = event_channels or None

    return event_id_config, event_channel_config, bads


def normalize_event_channels(event_channels: str | list[str] | None) -> list[str]:
    if event_channels is None:
        return []
    if isinstance(event_channels, str):
        return [event_channels]
    return [channel for channel in event_channels if channel]


def detect_regular_stream_event_ids(
    stim_data: np.ndarray,
    event_channels: str | list[str] | None,
    sfreq: float,
) -> list[int]:
    channels = normalize_event_channels(event_channels)
    if not channels or stim_data.size == 0:
        return []

    info = mne.create_info(channels, sfreq, ch_types=["stim"] * len(channels))
    raw = mne.io.RawArray(stim_data, info, verbose=False)

    detected_ids: set[int] = set()
    try:
        events = mne.find_events(
            raw,
            stim_channel=channels if len(channels) > 1 else channels[0],
            shortest_event=1,
            min_duration=0,
            verbose=False,
        )
        detected_ids.update(int(event_id) for event_id in np.unique(events[:, 2]) if int(event_id) > 0)
    except Exception:
        pass

    if not detected_ids:
        detected_ids.update(int(value) for value in np.unique(stim_data.astype(int)) if int(value) > 0)

    return sorted(detected_ids)


def resolve_regular_stream_event_id(
    stream,
    event_id: int | dict[str, int] | None,
    event_channels: str | list[str] | None,
    *,
    timeout_s: float = 2.5,
) -> int | dict[str, int]:
    if event_id is not None:
        return event_id

    channels = normalize_event_channels(event_channels)
    if not channels:
        raise ValueError("No stim channel was selected for epoching.")

    deadline = time.monotonic() + max(timeout_s, 0.5)
    detected_ids: list[int] = []
    while time.monotonic() < deadline and not detected_ids:
        stim_data, _ = stream.get_data(picks=channels)
        detected_ids = detect_regular_stream_event_ids(stim_data, channels, stream.info["sfreq"])
        if not detected_ids:
            time.sleep(0.15)

    if not detected_ids:
        raise ValueError(
            "No events were detected on the selected stim channel(s). "
            "Check the trigger channel selection or set a specific Event ID."
        )

    if len(detected_ids) == 1:
        return detected_ids[0]

    return {f"Event {event_id_value}": event_id_value for event_id_value in detected_ids}


def build_trace_colors(count: int) -> list:
    theme_name = current_theme_name()
    if count <= 0:
        return []

    if theme_name == "dark":
        return [
            pg.intColor(
                index,
                hues=max(count, 8),
                minHue=165,
                maxHue=245,
                sat=170,
                minValue=160,
                maxValue=255,
            )
            for index in range(count)
        ]

    return [
        pg.intColor(
            index,
            hues=max(count, 8),
            minHue=165,
            maxHue=245,
            sat=145,
            minValue=90,
            maxValue=180,
        )
        for index in range(count)
    ]


def channel_grid_positions(coords_2d: np.ndarray) -> list[tuple[int, int]]:
    if coords_2d.size == 0:
        return []

    n_channels = coords_2d.shape[0]
    grid_size = max(5, int(np.ceil(np.sqrt(n_channels))) + 2)
    x_values = coords_2d[:, 0]
    y_values = coords_2d[:, 1]

    if np.ptp(x_values) == 0:
        x_scaled = np.full(n_channels, 0.5)
    else:
        x_scaled = (x_values - np.min(x_values)) / np.ptp(x_values)

    if np.ptp(y_values) == 0:
        y_scaled = np.full(n_channels, 0.5)
    else:
        y_scaled = (y_values - np.min(y_values)) / np.ptp(y_values)

    occupied: set[tuple[int, int]] = set()
    positions: list[tuple[int, int]] = []
    for x_norm, y_norm in zip(x_scaled, y_scaled):
        preferred_row = int(round((1.0 - y_norm) * (grid_size - 1)))
        preferred_col = int(round(x_norm * (grid_size - 1)))
        candidates = [
            (preferred_row + row_offset, preferred_col + col_offset)
            for radius in range(grid_size)
            for row_offset in range(-radius, radius + 1)
            for col_offset in range(-radius, radius + 1)
        ]
        for row, col in candidates:
            if 0 <= row < grid_size and 0 <= col < grid_size and (row, col) not in occupied:
                occupied.add((row, col))
                positions.append((row, col))
                break
        else:
            positions.append((preferred_row, preferred_col))
    return positions

class PlayerWidget(QWidget):
    """A widget to play a file as an LSL stream."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = None
        self.setWindowTitle("LSL File Player")
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Select a .fif file to stream...")
        self.browse_button = QPushButton("Browse...")
        file_layout.addWidget(self.file_path_input)
        file_layout.addWidget(self.browse_button)
        layout.addRow("File Path:", file_layout)

        # Stream settings
        self.stream_name_input = QLineEdit("SSPy-Player")
        layout.addRow("Stream Name:", self.stream_name_input)
        
        self.chunk_size_input = QSpinBox()
        self.chunk_size_input.setRange(1, 4096)
        self.chunk_size_input.setValue(256)
        self.chunk_size_input.setSuffix(" samples")
        layout.addRow("Chunk Size:", self.chunk_size_input)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Streaming")
        self.stop_button = QPushButton("Stop Streaming")
        self.stop_button.setEnabled(False)
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addRow(button_layout)
        
        # Connections
        self.browse_button.clicked.connect(self._browse_file)
        self.start_button.clicked.connect(self.start_streaming)
        self.stop_button.clicked.connect(self.stop_streaming)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select FIF File", "", "FIF Files (*.fif *.fif.gz)"
        )
        if file_path:
            self.file_path_input.setText(file_path)
        
    def start_streaming(self):
        file_path = self.file_path_input.text()
        if not file_path:
            QMessageBox.warning(self, "No File", "Please select a file to stream.")
            return
            
        stream_name = self.stream_name_input.text()
        chunk_size = self.chunk_size_input.value()
        
        try:
            self.player = mne_lsl.player.PlayerLSL(
                file_path,
                chunk_size=chunk_size,
                name=stream_name
            ).start()
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.browse_button.setEnabled(False)
            self.file_path_input.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error Starting Player", f"Could not start the LSL player: {e}")

    def stop_streaming(self):
        if self.player:
            try:
                self.player.stop()
                self.player = None
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.browse_button.setEnabled(True)
                self.file_path_input.setEnabled(True)
            except Exception as e:
                 QMessageBox.critical(self, "Error Stopping Player", f"Could not stop the LSL player: {e}")

    def closeEvent(self, event):
        self.stop_streaming()
        super().closeEvent(event)

class StreamInfoWorker(QObject):
    """Worker to find an LSL stream and get its info."""
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, stream_name):
        super().__init__()
        self.stream_name = stream_name

    def run(self):
        try:
            # Short duration, we just want the info
            stream = mne_lsl.stream.StreamLSL(1, name=self.stream_name)
            stream.connect(acquisition_delay=0.1, processing_flags="all", timeout=5)
            ch_names = stream.info["ch_names"]
            stream.disconnect()
            self.finished.emit(ch_names)
        except Exception as e:
            self.error.emit(f"Could not find or connect to stream '{self.stream_name}': {e}")

class ConnectionWidget(QWidget):
    """A widget to get stream connection parameters from the user."""

    SETTINGS_PATH = "real_time/connection"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self.ch_names = []
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Initializes the user interface for the connection settings."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # --- Connection Group ---
        conn_group_box = QGroupBox("Connection Settings")
        form_layout = QFormLayout(conn_group_box)
        self.stream_name_input = QLineEdit()
        self.stream_duration_input = QSpinBox()
        self.stream_duration_input.setRange(1, 60 * 5)
        self.tlim_input = OptionalRangeWidget(
            ("Start:", "End:"),
            range=(None, None),
            suffix=" ms",
            required=(True, True),
            scale=1e-3,
            parent=self,
        )
        self.default_montage_combo = QComboBox()
        self.default_montage_combo.addItems(builtin_montage_names())
        self.find_channels_button = QPushButton("Find Channels")
        form_layout.addRow("Stream Name:", self.stream_name_input)
        form_layout.addRow("Stream Duration (s):", self.stream_duration_input)
        form_layout.addRow("Epoching (t_start, t_end):", self.tlim_input)
        form_layout.addRow("Fallback Montage:", self.default_montage_combo)
        form_layout.addRow(self.find_channels_button)
        main_layout.addWidget(conn_group_box)
        
        # --- Channel Config Group ---
        chan_group_box = QGroupBox("Channel Configuration")
        chan_layout = QVBoxLayout(chan_group_box)
        helper_label = QLabel(
            "Find channels first. Stim channels are auto-detected by name. Leave Event ID as Auto unless you want to restrict epoching to a specific trigger code."
        )
        helper_label.setObjectName("mutedLabel")
        helper_label.setWordWrap(True)
        chan_layout.addWidget(helper_label)
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(5)
        self.channel_table.setHorizontalHeaderLabels(["Enabled", "Name", "Type", "Event ID", "Event Name"])
        self.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.channel_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.channel_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.channel_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.channel_table.verticalHeader().setVisible(False)
        chan_layout.addWidget(self.channel_table)
        
        save_load_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Config")
        self.load_button = QPushButton("Load Config")
        save_load_layout.addStretch()
        save_load_layout.addWidget(self.save_button)
        save_load_layout.addWidget(self.load_button)
        chan_layout.addLayout(save_load_layout)
        
        main_layout.addWidget(chan_group_box)
        
        # --- Connections ---
        self.find_channels_button.clicked.connect(self.find_channels)
        self.save_button.clicked.connect(self.save_channel_config)
        self.load_button.clicked.connect(self.load_channel_config)

    def find_channels(self):
        stream_name = self.stream_name_input.text()
        if not stream_name:
            QMessageBox.warning(self, "Missing Stream Name", "Please enter a stream name to find.")
            return
            
        self.find_channels_button.setEnabled(False)
        self.find_channels_button.setText("Finding...")

        self.thread = QThread(self)
        self.worker = StreamInfoWorker(stream_name)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_channels_found)
        self.worker.error.connect(self.on_find_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_channels_found(self, ch_names):
        self.ch_names = ch_names
        self.channel_table.clearContents()
        self.channel_table.setRowCount(len(ch_names))

        for i, name in enumerate(ch_names):
            # Enabled checkbox
            enabled_check = QCheckBox()
            enabled_check.setChecked(True)
            self.channel_table.setCellWidget(i, 0, self._create_centered_widget(enabled_check))

            # Name
            name_item = QTableWidgetItem(name)
            self.channel_table.setItem(i, 1, name_item)

            # Type combo
            type_combo = QComboBox()
            type_combo.addItems(CHANNEL_TYPE_OPTIONS)
            self.channel_table.setCellWidget(i, 2, type_combo)
            type_combo.currentTextChanged.connect(lambda text, row=i: self._on_type_changed(text, row))
            type_combo.setCurrentText(suggest_channel_type(name))

            # Event ID & Name
            spin_box = QSpinBox()
            spin_box.setRange(0, 999999)
            spin_box.setSpecialValueText("Auto")
            spin_box.setValue(0)
            self.channel_table.setCellWidget(i, 3, spin_box)
            self.channel_table.setItem(i, 4, QTableWidgetItem())

            self._on_type_changed(type_combo.currentText(), i)

        self.channel_table.resizeRowsToContents()

        self.find_channels_button.setEnabled(True)
        self.find_channels_button.setText("Find Channels")

    def on_find_error(self, error_message):
        QMessageBox.critical(self, "Error Finding Stream", error_message)
        self.find_channels_button.setEnabled(True)
        self.find_channels_button.setText("Find Channels")

    def _create_centered_widget(self, widget):
        centered_widget = QWidget()
        layout = QHBoxLayout(centered_widget)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0,0,0,0)
        return centered_widget
        
    def _on_type_changed(self, text, row):
        is_stim = (text == 'stim')
        
        # Event ID SpinBox
        id_widget = self.channel_table.cellWidget(row, 3)
        if id_widget:
            id_widget.setEnabled(is_stim)
            if not is_stim:
                id_widget.setValue(0)

        # Event Name Item
        name_item = self.channel_table.item(row, 4)
        if name_item:
            flags = name_item.flags()
            if is_stim:
                flags |= Qt.ItemIsEditable
            else:
                flags &= ~Qt.ItemIsEditable
                name_item.setText("")
            name_item.setFlags(flags)
                
    def save_channel_config(self):
        if not self.ch_names:
            QMessageBox.warning(self, "No Channels", "Find channels before saving configuration.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Channel Configuration", "", "JSON Files (*.json)")
        if not path:
            return

        config = self.get_channel_settings()
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            QMessageBox.critical(self, "Error Saving File", f"Could not save configuration file: {e}")

    def load_channel_config(self):
        if not self.ch_names:
            QMessageBox.warning(self, "No Channels", "Find channels before loading a configuration.")
            return
            
        path, _ = QFileDialog.getOpenFileName(self, "Load Channel Configuration", "", "JSON Files (*.json)")
        if not path:
            return

        try:
            with open(path, 'r') as f:
                config = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load or parse configuration file: {e}")
            return
        
        self.set_channel_settings(config)

    def get_channel_settings(self):
        channels_config = []
        for i in range(self.channel_table.rowCount()):
            name = self.channel_table.item(i, 1).text()
            
            enabled_widget = self.channel_table.cellWidget(i, 0)
            enabled = enabled_widget.findChild(QCheckBox).isChecked()
            
            type_widget = self.channel_table.cellWidget(i, 2)
            ch_type = type_widget.currentText()
            
            id_widget = self.channel_table.cellWidget(i, 3)
            event_id = id_widget.value()
            
            event_name = self.channel_table.item(i, 4).text()
            
            channels_config.append({
                "name": name,
                "enabled": enabled,
                "type": ch_type,
                "event_id": event_id,
                "event_name": event_name,
            })
        return channels_config

    def set_channel_settings(self, config):
        if len(config) == self.channel_table.rowCount():
            for i, ch_config in enumerate(config):
                self.channel_table.item(i, 1).setText(ch_config.get('name', ''))
                
                enabled_widget = self.channel_table.cellWidget(i, 0)
                enabled_widget.findChild(QCheckBox).setChecked(ch_config.get('enabled', True))
                
                type_widget = self.channel_table.cellWidget(i, 2)
                type_widget.setCurrentText(ch_config.get('type', 'eeg'))

                id_widget = self.channel_table.cellWidget(i, 3)
                id_widget.setValue(ch_config.get('event_id', 0))
                
                self.channel_table.item(i, 4).setText(ch_config.get('event_name', ''))
        else:
            config_map = {item['name']: item for item in config}
            for i in range(self.channel_table.rowCount()):
                name = self.channel_table.item(i, 1).text()
                if name in config_map:
                    ch_config = config_map[name]
                    
                    enabled_widget = self.channel_table.cellWidget(i, 0)
                    enabled_widget.findChild(QCheckBox).setChecked(ch_config.get('enabled', True))
                    
                    type_widget = self.channel_table.cellWidget(i, 2)
                    type_widget.setCurrentText(ch_config.get('type', 'eeg'))

                    id_widget = self.channel_table.cellWidget(i, 3)
                    id_widget.setValue(ch_config.get('event_id', 0))
                    
                    self.channel_table.item(i, 4).setText(ch_config.get('event_name', ''))

    def get_settings(self) -> dict:
        base_settings = {
            "stream_name": self.stream_name_input.text(),
            "stream_duration": self.stream_duration_input.value(),
            "tlim": self.tlim_input.value(),
            "default_montage": self.default_montage_combo.currentText(),
        }
        channel_settings = self.get_channel_settings()
        event_id, event_channels, bads = build_epoch_stream_configuration(channel_settings)

        base_settings['event_id'] = event_id
        base_settings['event_channels'] = event_channels
        base_settings['bads'] = bads
        base_settings['channels'] = channel_settings

        return base_settings

    def save_settings(self):
        """Saves the current settings to QSettings under the group."""
        params = {
            "stream_name": self.stream_name_input.text(),
            "stream_duration": self.stream_duration_input.value(),
            "tlim": self.tlim_input.value(),
            "default_montage": self.default_montage_combo.currentText(),
        }
        self.settings_store.set(self.SETTINGS_PATH, params)
        self.settings_store.set("real_time/connection", params)
        self.settings_store.sync()

    def load_settings(self):
        """Loads settings from QSettings, applying defaults for missing values."""
        default_params = {
            "stream_name": "EEGStream",
            "stream_duration": 5,
            "tlim": (-0.1, 0.3),
            "default_montage": DEFAULT_REAL_TIME_MONTAGE,
        }
        saved_params = self.settings_store.get(
            self.SETTINGS_PATH,
            {},
            legacy_keys=("real_time/connection",),
        ) or {}
        params = default_params.copy()
        params.update(saved_params)
        self.stream_name_input.setText(params.get("stream_name"))
        self.stream_duration_input.setValue(int(params.get("stream_duration")))
        self.tlim_input.setValue(params.get("tlim"))
        self.default_montage_combo.setCurrentText(params.get("default_montage", DEFAULT_REAL_TIME_MONTAGE))

class ConnectionManager:

    """Handles the connection to an LSL stream."""

    def __init__(self, params):
        self.params = params
        self.stream = None
        self.epochs = None

    def connect_to_stream(self):
        """Establishes a connection to the LSL stream."""
        try:
            stream_duration = self.params.get("stream_duration")
            stream_name = self.params.get("stream_name")
            tmin, tmax = self.params.get("tlim")
            event_id = self.params.get("event_id")
            event_channels = self.params.get("event_channels")
            event_channel_list = normalize_event_channels(event_channels)

            self.stream = mne_lsl.stream.StreamLSL(stream_duration, name=stream_name)
            self.stream.connect(
                acquisition_delay=0.1, processing_flags="all", timeout=5
            )

            # Apply channel names and types
            if self.params.get("channels"):
                ch_rename_map = {}
                ch_types = {}
                for idx, ch in enumerate(self.params["channels"]):
                    if idx < len(self.stream.ch_names):
                        orig_name = self.stream.ch_names[idx]
                        new_name = ch["name"]
                        if orig_name != new_name:
                            ch_rename_map[orig_name] = new_name
                        
                        c_type = ch["type"] if ch["type"] in ["eeg", "eog", "stim", "misc"] else "misc"
                        ch_types[new_name] = c_type

                if ch_rename_map:
                    try:
                        self.stream.rename_channels(ch_rename_map)
                        print(f"Renamed channels: {ch_rename_map}")
                    except Exception as e:
                        print(f"Warning: could not rename channels: {e}")

                if ch_types:
                    try:
                        self.stream.set_channel_types(ch_types)
                        print(f"Set channel types: {ch_types}")
                    except Exception as e:
                        print(f"Warning: could not set channel types: {e}")

            bads = self.params.get('bads', [])
            if bads:
                self.stream.info['bads'] = bads

            default_montage = self.params.get("default_montage")
            if default_montage and default_montage != "None" and self.stream.get_montage() is None:
                try:
                    self.stream.set_montage(mne.channels.make_standard_montage(default_montage))
                except Exception as exc:
                    print(f"Warning: could not apply montage '{default_montage}': {exc}")

            if self.params.get("apply_bandpass", False):
                low, high = self.params.get("bandpass_range", (None, None))
                if low is not None or high is not None:
                    self.stream.filter(low, high, picks="eeg", iir_params=None, verbose=False)

            if self.params.get("apply_notch", False):
                for freq in self.params.get("notch_freqs", []):
                    self.stream.notch_filter(freq, picks="eeg", iir_params=None, verbose=False)

            event_id = resolve_regular_stream_event_id(
                self.stream,
                event_id,
                event_channel_list,
                timeout_s=min(max(float(stream_duration), 1.5), 4.0),
            )
            self.params["event_id"] = event_id

            self.epochs = mne_lsl.stream.EpochsStream(
                self.stream,
                bufsize=10,
                event_id=event_id,
                event_channels=event_channel_list[0] if len(event_channel_list) == 1 else event_channel_list,
                tmin=tmin,
                tmax=tmax,
                baseline=(None, 0),
                picks="eeg",
            )
            self.epochs.connect(acquisition_delay=0.1)
            print("Connection successful.")
            return self.stream, self.epochs
        except Exception as e:
            print(f"Failed to connect to stream: {e}")
            return None, None

class ConnectionWorker(QObject):
    """Worker thread for handling the LSL stream connection."""

    finished = Signal(object, object)  # Emits stream and epochs on success
    error = Signal(str)  # Emits error message on failure

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Tries to connect to the stream."""
        try:
            conn_manager = ConnectionManager(self.params)
            stream, epochs = conn_manager.connect_to_stream()
            if stream and epochs:
                self.finished.emit(stream, epochs)
            else:
                self.error.emit(
                    "Failed to connect to the LSL stream. Please check the stream name and ensure it is available."
                )
        except Exception as e:
            self.error.emit(f"An error occurred during connection: {e}")

class RealTimeSettingsWidget(QWidget):
    """A widget to configure real-time plotting parameters."""

    SETTINGS_PATH = "real_time/plotting"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Initializes the user interface for the plot settings."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        group_box = QGroupBox("Real-Time Plot Settings")
        main_layout.addWidget(group_box)

        form_layout = QFormLayout(group_box)
        form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.refresh_rate_input = QSpinBox()
        self.refresh_rate_input.setSuffix(" Hz")
        self.refresh_rate_input.setRange(1, 60)
        form_layout.addRow("Refresh Rate:", self.refresh_rate_input)

        form_layout.addRow(QLabel("<br><b>Real-Time Pre-processing</b>"))

        self.art_rem_input = OptionalRangeWidget(
            ("Start:", "End:"),
            range=(None, None),
            suffix=" ms",
            required=(True, True),
            scale=1e-3,
        )
        form_layout.addRow("Artifact Removal:", self.art_rem_input)

        self.apply_bandpass = QCheckBox("Apply Bandpass Filter")
        self.bandpass_input = OptionalRangeWidget(
            labels=("Low:", "High:"), suffix=" Hz", range=(0.1, 200)
        )
        form_layout.addRow(self.apply_bandpass, self.bandpass_input)

        self.apply_notch = QCheckBox("Apply Notch Filter")
        self.notch_input = QLineEdit()
        form_layout.addRow(self.apply_notch, self.notch_input)

        self.apply_bandpass.toggled.connect(self.bandpass_input.setEnabled)
        self.apply_notch.toggled.connect(self.notch_input.setEnabled)

    def get_settings(self) -> dict:
        """Returns the current settings as a dictionary."""
        try:
            notch_freqs = [
                float(f.strip())
                for f in self.notch_input.text().split(",")
                if f.strip()
            ]
        except (ValueError, TypeError):
            notch_freqs = []

        return {
            "refresh_rate": self.refresh_rate_input.value(),
            "art_rem": self.art_rem_input.value(),
            "apply_bandpass": self.apply_bandpass.isChecked(),
            "bandpass_range": self.bandpass_input.value(),
            "apply_notch": self.apply_notch.isChecked(),
            "notch_freqs": notch_freqs,
        }

    def set_settings(self, params: dict):
        """Sets the UI components from a settings dictionary."""
        self.refresh_rate_input.setValue(params.get("refresh_rate"))
        self.apply_bandpass.setChecked(params.get("apply_bandpass"))
        self.apply_notch.setChecked(params.get("apply_notch"))

        self.art_rem_input.setValue(params.get("art_rem"))
        self.bandpass_input.setValue(params.get("bandpass_range"))

        notch_freqs = params.get("notch_freqs", [])
        self.notch_input.setText(
            ", ".join(
                map(str, notch_freqs if type(notch_freqs) == list else [notch_freqs])
            )
        )

        self.bandpass_input.setEnabled(self.apply_bandpass.isChecked())
        self.notch_input.setEnabled(self.apply_notch.isChecked())

    def save_settings(self):
        """Saves the current parameters to QSettings under the group."""
        params = self.get_settings()
        self.settings_store.set(self.SETTINGS_PATH, params)
        self.settings_store.set("real_time/plot_settings", params)
        self.settings_store.sync()

    def load_settings(self):
        """Loads parameters from QSettings, applying defaults, and updates the UI."""
        default_params = {
            "refresh_rate": 24,
            "art_rem": (-0.005, 0.005),
            "apply_bandpass": False,
            "bandpass_range": (8.0, 80.0),
            "apply_notch": False,
            "notch_freqs": [50.0],
        }

        saved_params = self.settings_store.get(
            self.SETTINGS_PATH,
            {},
            legacy_keys=("real_time/plot_settings",),
        ) or {}

        params = default_params.copy()
        params.update(saved_params)

        self.set_settings(params)

class RealTimeSettings(QDialog):
    """A dialog for configuring real-time plot settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time Plot Settings")
        self.resize(640, 560)
        self.setMinimumSize(560, 420)
        self.setObjectName("appDialog")

        # Main layout
        layout = QVBoxLayout(self)

        # The widget with all the controls
        self.settings_widget = RealTimeSettingsWidget(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.viewport().setAutoFillBackground(False)
        self.scroll_area.setWidget(self.settings_widget)

        # Standard OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(button_box)

    def get_settings(self) -> dict:
        """Returns the settings from the internal widget."""
        return self.settings_widget.get_settings()

class DataProcessingWorker(QObject):
    """Processes EEG data in a separate thread."""

    data_ready = Signal(dict)
    finished = Signal()

    def __init__(self, stream, epochs, params, parent=None):
        super().__init__(parent)
        self.stream = stream
        self.epochs = epochs
        self.params = params
        self.is_running = True
        self.max_epochs = params.get("max_epochs", 200)
        self._processed_epochs = deque(maxlen=self.max_epochs)
        self.original_times = self.epochs.times.copy()
        self.original_sfreq = self.epochs.info["sfreq"]
        self.decimate = self.params.get("decimate", 5)
        self.times = self.original_times[:: self.decimate]
        self.raw_picks = mne.pick_types(self.stream.info, eeg=True, exclude=())

    def run(self):
        """Starts the data processing loop."""
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_and_update)
        self.timer.start(max(1, int(round(1000 / self.params.get("refresh_rate", 24)))))

    def stop(self):
        """Stops the data processing loop."""
        self.is_running = False

    def _process_and_update(self):
        if not self.is_running:
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()
            self.finished.emit()
            return

        # Always try to get a raw chunk for continuous visualization
        emit_dict = {}
        try:
            raw_chunk = self._peek_raw_data()
            if raw_chunk is not None:
                emit_dict["raw_data"] = raw_chunk
        except Exception:
            pass

        new_epoch_count = getattr(self.epochs, "n_new_epochs", 0)
        if new_epoch_count > 0:
            try:
                new_data = self.epochs.get_data(n_epochs=new_epoch_count)
                if new_data is not None and new_data.shape[0] > 0:
                    processed_batch = self._process_epoch_batch(new_data)
                    for epoch_index in range(processed_batch.shape[0]):
                        self._processed_epochs.append(processed_batch[epoch_index])
            except Exception as e:
                print(f"Error fetching epoch data: {e}")

        if self._processed_epochs:
            # Create a consistent snapshot for this processing iteration
            plot_data = np.stack(self._processed_epochs, axis=0)

            # Decimate after processing so the on-screen data stays light.
            processed_data = plot_data[:, :, :: self.decimate]

            # Calculate mean for plotting
            mean_data = np.nanmean(processed_data, axis=0)
            
            # Add epoch data to the emit dictionary
            emit_dict["epoch_data"] = {
                "processed_data": processed_data,
                "mean_data": mean_data,
                "n_epochs": processed_data.shape[0]
            }
        
        # Emit data for UI thread
        self.data_ready.emit(emit_dict)

    def _process_epoch_batch(self, epoch_batch: np.ndarray) -> np.ndarray:
        plot_data = epoch_batch.copy()

        art_rem_start, art_rem_end = self.params.get("art_rem", (0, 0))
        if art_rem_end > art_rem_start:
            start_idx = np.searchsorted(self.original_times, art_rem_start, side="left")
            end_idx = np.searchsorted(self.original_times, art_rem_end, side="right")
            if end_idx > start_idx and start_idx > 0 and end_idx < len(self.original_times) - 1:
                interp_x = [start_idx - 1, end_idx]
                for epoch_index in range(plot_data.shape[0]):
                    for channel_index in range(plot_data.shape[1]):
                        interp_y = [
                            plot_data[epoch_index, channel_index, start_idx - 1],
                            plot_data[epoch_index, channel_index, end_idx],
                        ]
                        plot_data[epoch_index, channel_index, start_idx:end_idx] = np.interp(
                            np.arange(start_idx, end_idx),
                            interp_x,
                            interp_y,
                        )

        if self.params.get("reference", "average") == "average" and plot_data.shape[1] > 1:
            plot_data -= np.nanmean(plot_data, axis=1, keepdims=True)

        return plot_data

    def _peek_raw_data(self) -> np.ndarray | None:
        buffer = getattr(self.stream, "_buffer", None)
        timestamps = getattr(self.stream, "_timestamps", None)
        if buffer is None or timestamps is None:
            return None

        n_samples = int(np.count_nonzero(timestamps))
        if n_samples <= 0:
            return None

        recent_samples = buffer[-n_samples:, :]
        return recent_samples[:, self.raw_picks].T

    def update_params(self, new_params):
        """Update processing parameters."""
        self.params.update(new_params)
        if hasattr(self, "timer"):
            self.timer.setInterval(max(1, int(round(1000 / self.params.get("refresh_rate", 24)))))

class SingleChannelPlot(QDialog):
    """A dialog for plotting data from a single EEG channel using pyqtgraph."""
    closed = Signal()

    def __init__(self, ch_name, times, parent):
        super().__init__(parent)
        self.ch_name = ch_name
        self.times = times * 1e3 # Convert to ms
        self.parent = parent
        self.setup_ui()
        self.show()

    def setup_ui(self):
        self.setWindowTitle(f"Channel: {self.ch_name}")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        layout = QVBoxLayout(self)
        self.position_label = QLabel(self.parent.channel_position_text(self.ch_name))
        self.position_label.setObjectName("mutedLabel")
        self.position_label.setWordWrap(True)
        layout.addWidget(self.position_label)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setLabel('left', 'Potential (uV)')
        self.plot_widget.setLabel('bottom', 'Time (ms)')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        self.mean_curve = self.plot_widget.plot(name='Mean')
        self.std_fill = pg.FillBetweenItem(pg.PlotDataItem(), pg.PlotDataItem())
        self.plot_widget.addItem(self.std_fill)
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.plot_widget.addItem(self.v_line)
        self.v_line.setPos(0)
        self.refresh_theme()

    def refresh_theme(self):
        tokens = theme_tokens()
        accent_color = QColor(tokens["accent_soft"])
        self.plot_widget.setBackground(tokens["plot_background"])
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen(tokens["text"]))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen(tokens["text"]))
        self.plot_widget.getAxis("left").setPen(pg.mkPen(tokens["border"]))
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen(tokens["border"]))
        self.mean_curve.setPen(pg.mkPen(accent_color, width=2))
        self.v_line.setPen(pg.mkPen(tokens["accent"], style=Qt.DashLine))

    def update_plot(self, data):
        """Updates the plot with new data."""
        ch_idx = self.parent.ch_names.index(self.ch_name)
        channel_data = data[:, ch_idx, :] * 1e6  # Convert to uV

        evoked = np.nanmean(channel_data, axis=0)
        ch_std = np.nanstd(channel_data, axis=0)
        t0_idx = np.argmin(abs(self.times))

        self.mean_curve.setData(self.times, evoked)

        # Update FillBetweenItem
        if self.std_fill:
            self.plot_widget.removeItem(self.std_fill)
        upper_bound = pg.PlotDataItem(self.times, evoked + ch_std)
        lower_bound = pg.PlotDataItem(self.times, evoked - ch_std)
        fill_color = QColor(theme_tokens()["accent_fill"])
        fill_color.setAlpha(92)
        self.std_fill = pg.FillBetweenItem(upper_bound, lower_bound, brush=pg.mkBrush(fill_color))
        self.plot_widget.addItem(self.std_fill)

        self.plot_widget.setTitle(
            f"{self.ch_name} (n={data.shape[0]} | "
            f"R_std={np.nanmean(ch_std[t0_idx:]):.1f} uV)"
        )

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

class TopomapPlot(QDialog):
    """A dialog for displaying a topographical plot using Matplotlib."""
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.fig = Figure(layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.cbar = None
        self.refresh_theme()
        self.show()

    def update_plot(self, data, position, title="", names=None, cmap="turbo"):
        self.ax.clear()
        im, _ = mne.viz.plot_topomap(
            data, position, names=names, cmap=cmap, show=False, axes=self.ax
        )
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=self.ax)
        else:
            self.cbar.update_normal(im)
        self.ax.set_title(title)
        self.refresh_theme()
        self.canvas.draw()

    def refresh_theme(self):
        tokens = theme_tokens()
        self.fig.patch.set_facecolor(tokens["panel"])
        self.ax.set_facecolor(tokens["plot_background"])
        self.ax.title.set_color(tokens["text"])
        if self.cbar is not None:
            self.cbar.ax.yaxis.set_tick_params(color=tokens["text"])
            for tick_label in self.cbar.ax.get_yticklabels():
                tick_label.set_color(tokens["text"])

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

class RealTimeDockWidget(QDockWidget):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)

    def event(self, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.NonClientAreaMouseButtonDblClick:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.setFloating(True)
                self.showFullScreen()
            return True
        return super().event(event)

class RealTimeERP(QMainWindow):
    GEOMETRY_SETTING = "real_time/geometry"
    """Main widget for real-time ERP visualization."""

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.stream = None
        self.epochs = None
        self._processed_data = None
        self.ch_names = []
        self.coords_2d = []
        self.channel_coords = {}
        self.colors = []
        self.times = []
        self.opened_single_channels = {}
        self.bads = self.params.get('bads', [])
        self.topomap_dialog = None
        self.raw_plot_buffer = None
        self.active_montage_name = None
        self.topo_label_items = {}
        self.theme_name = current_theme_name()
        self.tokens = theme_tokens(self.theme_name)

        self.setup_ui()
        self.setWindowTitle("Real-Time TEP Visualization")
        self._read_settings()

    def setup_ui(self):
        apply_pyqtgraph_theme(self.theme_name)
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AnimatedDocks)

        self.raw_widget = self._create_raw_widget()
        self.topo_widget = self._create_topo_widget()
        self.evoked_widget = self._create_evoked_widget()
        self.toolbar = self._create_toolbar()

        self.addToolBar(self.toolbar)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.raw_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.topo_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.evoked_widget)

        dummy = QWidget()
        dummy.hide()
        self.setCentralWidget(dummy)
        self.refresh_theme()

    def _read_settings(self):
        geometry = QSettings().value(self.GEOMETRY_SETTING)
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.setGeometry(100, 100, 1280, 960)
        state = QSettings().value(self.GEOMETRY_SETTING + "_state")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        QSettings().setValue(self.GEOMETRY_SETTING, self.saveGeometry())
        QSettings().setValue(self.GEOMETRY_SETTING + "_state", self.saveState())
        super().closeEvent(event)

    def refresh_theme(self):
        self.theme_name = current_theme_name()
        self.tokens = theme_tokens(self.theme_name)
        apply_pyqtgraph_theme(self.theme_name)

        accent_brush = QColor(self.tokens["roi"])
        accent_brush.setAlpha(82)
        accent_pen = pg.mkPen(self.tokens["accent"], width=1.4)

        for plot_widget in [getattr(self, "raw_plot_widget", None), getattr(self, "evoked_plot", None)]:
            if plot_widget is None:
                continue
            plot_widget.setBackground(self.tokens["plot_background"])
            plot_widget.showGrid(x=True, y=plot_widget is self.evoked_plot, alpha=0.16)
            plot_widget.setMouseEnabled(x=True, y=False)
            plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)
            for axis_name in ("left", "bottom"):
                axis = plot_widget.getAxis(axis_name)
                axis.setTextPen(pg.mkPen(self.tokens["text"]))
                axis.setPen(pg.mkPen(self.tokens["border"]))

        if hasattr(self, "topo_plot_widget"):
            self.topo_plot_widget.setBackground(self.tokens["plot_background"])

        if hasattr(self, "roi"):
            self.roi.setBrush(pg.mkBrush(accent_brush))
            self.roi.setPen(accent_pen)
            hover_pen = QColor(self.tokens["accent_soft"])
            hover_pen.setAlpha(210)
            self.roi.setHoverBrush(pg.mkBrush(accent_brush))
            self.roi.setHoverPen(pg.mkPen(hover_pen, width=1.6))

        if self.ch_names:
            self.colors = build_trace_colors(len(self.ch_names))
            for index, curve in enumerate(getattr(self, "raw_curves", [])):
                curve.setPen(pg.mkPen(self.colors[index]))
            for index, line in enumerate(getattr(self, "evoked_lines", [])):
                line.setPen(pg.mkPen(self.colors[index], width=1.3))
            for channel_name, curve in getattr(self, "topo_plots", {}).items():
                curve.setPen(pg.mkPen(self.colors[self.ch_names.index(channel_name)], width=1.2))
            for channel_name, label in self.topo_label_items.items():
                label.setColor(self.colors[self.ch_names.index(channel_name)])
            for plot_item in getattr(self, "topo_plot_items", {}).values():
                plot_item.setBackground(self.tokens["plot_background"])

        if self.topomap_dialog is not None:
            self.topomap_dialog.refresh_theme()
        for dialog in self.opened_single_channels.values():
            dialog.refresh_theme()

    def _create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addAction("Raw").triggered.connect(lambda: self.raw_widget.setVisible(not self.raw_widget.isVisible()))
        toolbar.addAction("Channel Layout").triggered.connect(lambda: self.topo_widget.setVisible(not self.topo_widget.isVisible()))
        toolbar.addAction("Butterfly").triggered.connect(lambda: self.evoked_widget.setVisible(not self.evoked_widget.isVisible()))
        
        toolbar.addSeparator()
        
        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(["Global Auto-Scale", "Local Auto-Scale"])
        toolbar.addWidget(QLabel(" Scale: "))
        toolbar.addWidget(self.scale_mode_combo)
        
        self.n_chan_spinbox = QSpinBox()
        self.n_chan_spinbox.setRange(1, 256)
        self.n_chan_spinbox.setValue(32)
        self.n_chan_spinbox.valueChanged.connect(self._update_n_channels_shown)
        toolbar.addWidget(QLabel(" View Chans: "))
        toolbar.addWidget(self.n_chan_spinbox)
        
        toolbar.addSeparator()
        toolbar.addAction("Settings").triggered.connect(self.update_settings)
        return toolbar

    def _create_raw_widget(self):
        dock = RealTimeDockWidget("Raw Data Monitor", self)
        dock.setObjectName("RawDockWidget")
        self.raw_plot_widget = pg.PlotWidget()
        self.raw_plot_widget.setMenuEnabled(False)
        dock.setWidget(self.raw_plot_widget)
        return dock

    def _create_topo_widget(self):
        dock = RealTimeDockWidget("Topoplot", self)
        dock.setObjectName("TopoDockWidget")
        self.topo_plot_widget = pg.GraphicsLayoutWidget()
        dock.setWidget(self.topo_plot_widget)
        return dock

    def _create_evoked_widget(self):
        dock = RealTimeDockWidget("Evoked Potentials", self)
        dock.setObjectName("EvokedDockWidget")
        self.evoked_plot = pg.PlotWidget()
        self.evoked_plot.setMenuEnabled(False)
        self.evoked_plot.setLabel('left', 'Potential (uV)')
        self.evoked_plot.setLabel('bottom', 'Time (ms)')
        self.evoked_plot.showGrid(x=True, y=True)
        # Add ROI for topomap selection
        self.roi = pg.LinearRegionItem()
        self.roi.sigRegionChanged.connect(self.on_roi_changed)
        self.evoked_plot.addItem(self.roi)
        dock.setWidget(self.evoked_plot)
        return dock

    def start_visualization(self):
        self.conn_thread = QThread(self)
        self.conn_worker = ConnectionWorker(self.params)
        self.conn_worker.moveToThread(self.conn_thread)

        self.progress_dialog = QProgressDialog("Connecting to LSL stream...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)

        self.conn_thread.started.connect(self.conn_worker.run)
        self.conn_worker.finished.connect(self._on_connection_success)
        self.conn_worker.error.connect(self._on_connection_failure)
        self.progress_dialog.canceled.connect(self.close)

        self.conn_worker.finished.connect(self.conn_thread.quit)
        self.conn_worker.error.connect(self.conn_thread.quit)
        self.conn_thread.finished.connect(self.conn_worker.deleteLater)
        self.conn_thread.finished.connect(self.conn_thread.deleteLater)

        self.conn_thread.start()
        self.progress_dialog.exec()

    def _on_connection_success(self, stream, epochs):
        self.progress_dialog.accept()
        self.stream = stream
        self.epochs = epochs

        self._apply_visual_montage()

        eeg_picks = mne.pick_types(self.epochs.info, eeg=True, exclude=())
        self.info = mne.pick_info(self.epochs.info, eeg_picks)
        self.ch_names = self.info["ch_names"]
        self.sfreq = self.info["sfreq"]
        coords_3d = []
        for ch in self.info["chs"]:
            loc = ch.get("loc", np.zeros(12))
            if loc is not None and len(loc) >= 3 and not np.all(loc[:3] == 0):
                coords_3d.append(loc[:3])
            else:
                coords_3d.append([0.0, 0.0, 0.0])
        coords_3d = np.array(coords_3d)
        self.coords_2d = self.project_electrodes_to_2d(coords_3d)
        self.channel_coords = {
            channel_name: tuple(float(value) for value in self.coords_2d[index])
            for index, channel_name in enumerate(self.ch_names)
        }
        self.colors = build_trace_colors(len(self.ch_names))

        self.n_chan_spinbox.setValue(len(self.ch_names))

        # Start data processing worker thread
        self.data_thread = QThread(self)
        self.data_worker = DataProcessingWorker(self.stream, self.epochs, self.params)
        self.data_worker.moveToThread(self.data_thread)

        self.times = self.data_worker.times

        self.setup_plots()

        # Connections for data worker
        self.data_worker.data_ready.connect(self._on_data_ready)
        self.data_worker.finished.connect(self.data_thread.quit)
        self.data_thread.started.connect(self.data_worker.run)
        self.data_thread.finished.connect(self.data_worker.deleteLater)
        self.data_thread.finished.connect(self.data_thread.deleteLater)
        self.data_thread.start()

    def _on_connection_failure(self, error_message):
        self.progress_dialog.reject()
        QMessageBox.critical(self, "Connection Failed", error_message)
        self.close()

    def setup_plots(self):
        # Raw Plot Setup
        self._setup_raw_plot()
        # Evoked Plot Setup
        self.evoked_lines = [
            self.evoked_plot.plot(self.times * 1e3, np.zeros_like(self.times), pen=pg.mkPen(color=self.colors[i]))
            for i in range(len(self.ch_names))
        ]
        self.evoked_plot.setXRange(self.times[0] * 1e3, self.times[-1] * 1e3)
        self.evoked_plot.setLimits(xMin=self.times[0] * 1e3, xMax=self.times[-1] * 1e3)
        self.evoked_plot.setYRange(-10, 10)

        # Topo Plot Setup
        self.topo_plots = {}
        self.topo_plot_items = {}
        self.topo_label_items = {}
        self.topo_plot_widget.clear()
        plot_positions = channel_grid_positions(self.coords_2d)
        for i, name in enumerate(self.ch_names):
            row, col = plot_positions[i] if i < len(plot_positions) else (i // 8, i % 8)
            p = self.topo_plot_widget.addPlot(row=row, col=col)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideAxis('left')
            p.hideAxis('bottom')
            p.getViewBox().setMouseEnabled(x=False, y=False)
            p.setXRange(self.times[0] * 1e3, self.times[-1] * 1e3, padding=0)
            p.setLimits(xMin=self.times[0] * 1e3, xMax=self.times[-1] * 1e3)
            curve = p.plot(self.times * 1e3, np.zeros_like(self.times), pen=self.colors[i])
            p.setYRange(-10, 10, padding=0.1)
            text = pg.TextItem(name, color=self.colors[i], anchor=(0.5, 1))
            p.addItem(text)
            text.setPos(self.times[len(self.times)//2]*1e3, 10)
            p.scene().sigMouseClicked.connect(
                lambda ev, plot_item=p, ch=name: self._handle_topo_scene_click(ev, plot_item, ch)
            )
            self.topo_plots[name] = curve
            self.topo_plot_items[name] = p
            self.topo_label_items[name] = text
        self.topo_plot_widget.ci.layout.setSpacing(0)
        self.topo_plot_widget.ci.layout.setContentsMargins(0,0,0,0)
        self.refresh_theme()

    def _setup_raw_plot(self):
        self.raw_plot_widget.clear()
        self.raw_curves = []
        n_chans = len(self.ch_names)

        self.raw_plot_widget.setLabel('bottom', 'Time (s)')
        self.raw_plot_widget.showGrid(x=True, y=False)
        self.raw_offsets = np.arange(n_chans)[::-1]
        
        yax = self.raw_plot_widget.getAxis('left')
        ticks = [list(zip(self.raw_offsets, self.ch_names))]
        yax.setTicks(ticks)
        
        stream_duration = self.params.get("stream_duration", 5)
        self.raw_plot_widget.setLimits(xMin=-stream_duration, xMax=0, yMin=-0.5, yMax=n_chans - 0.5)
        self.raw_plot_widget.setXRange(-stream_duration, 0, padding=0)
        self.raw_plot_widget.setMouseEnabled(x=True, y=False)
        self.raw_plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        for i in range(n_chans):
            curve = self.raw_plot_widget.plot(pen=self.colors[i])
            self.raw_curves.append(curve)
            
        self._update_n_channels_shown()
        
    def _update_n_channels_shown(self):
        if not self.ch_names:
            return
        n_show = self.n_chan_spinbox.value()
        max_y = len(self.ch_names)
        min_y = max_y - n_show
        self.raw_plot_widget.setYRange(min_y - 0.5, max_y - 0.5, padding=0)
        self.raw_plot_widget.getViewBox().setMouseEnabled(y=False, x=True)

    @staticmethod
    def to_rgb(positions_3d):
        xyz = positions_3d.copy()
        if xyz.shape[0] > 1:
            xyz -= xyz.min(axis=0)
            max_vals = xyz.max(axis=0)
            max_vals[max_vals == 0] = 1 # Avoid division by zero
            xyz /= max_vals
        return xyz * 255

    def _on_data_ready(self, data_dict):
        if "epoch_data" in data_dict:
            self._update_epoch_plots(data_dict["epoch_data"])
        
        if "raw_data" in data_dict:
            self._update_raw_plot(data_dict["raw_data"])

    def _update_epoch_plots(self, epoch_data):
        self._processed_data = epoch_data["processed_data"]
        mean_data = epoch_data["mean_data"]
        n_epochs = epoch_data["n_epochs"]
        self.setWindowTitle(f"Real-Time TEP - Epochs: {n_epochs}")
        self.evoked_widget.setWindowTitle(f"Evoked Potentials (Epochs: {n_epochs})")

        # Update Evoked Plot
        for i, line in enumerate(self.evoked_lines):
            if self.ch_names[i] in self.bads:
                line.setData(self.times * 1e3, np.full_like(self.times, np.nan))
            else:
                line.setData(self.times * 1e3, mean_data[i] * 1e6)

        # Get global min and max for consistent scaling of topo sparklines
        valid_data = [mean_data[i] for i in range(len(self.ch_names)) if self.ch_names[i] not in self.bads]
        
        global_min, global_max = -10.0, 10.0
        if valid_data:
            arr = np.array(valid_data)
            if np.any(np.isfinite(arr)):
                global_min = np.nanmin(arr) * 1e6
                global_max = np.nanmax(arr) * 1e6

        if global_min >= global_max:
            global_min -= 1.0
            global_max += 1.0

        # Update Topo Plot
        for name, plot_item in self.topo_plots.items():
            ch_idx = self.ch_names.index(name)
            alpha = 0.25 if name in self.bads else 1.0
            plot_item.setOpacity(alpha)
            plot_item.setData(self.times * 1e3, mean_data[ch_idx] * 1e6)
            
            p = self.topo_plot_items[name]
            p.setYRange(global_min, global_max, padding=0.1)
            label_item = self.topo_label_items.get(name)
            if label_item is not None:
                label_item.setPos(self.times[len(self.times)//2]*1e3, global_max)

        # Update any open single channel plots
        for ch_name, plot_dialog in self.opened_single_channels.items():
            if self._processed_data is not None:
                plot_dialog.update_plot(self._processed_data)

    def _update_raw_plot(self, raw_chunk):
        if raw_chunk is None or raw_chunk.shape[1] == 0:
            return
        
        n_samples = raw_chunk.shape[1]
        n_chans = len(self.raw_curves)
        
        if n_chans != raw_chunk.shape[0]:
             return # Mismatch between buffer and incoming data

        time_axis = np.linspace(-n_samples / self.sfreq, 0, n_samples)
        scale_mode = self.scale_mode_combo.currentText()
        
        # Mean center the chunk to prevent massive DC offsets from breaking the scale
        means = np.nanmean(raw_chunk, axis=1, keepdims=True)
        centered_chunk = raw_chunk - means
        
        if scale_mode == "Global Auto-Scale":
            global_std = np.nanstd(centered_chunk)
            scale = 1.0 / (global_std * 6) if (not np.isnan(global_std) and global_std > 0) else 1.0
            for i, curve in enumerate(self.raw_curves):
                curve.setData(time_axis, (centered_chunk[i, :] * scale) + self.raw_offsets[i])
        else:
            for i, curve in enumerate(self.raw_curves):
                y_data = centered_chunk[i, :]
                local_std = np.nanstd(y_data)
                if not np.isnan(local_std) and local_std > 0:
                    scale = 1.0 / (local_std * 6)
                    curve.setData(time_axis, (y_data * scale) + self.raw_offsets[i])
                else:
                    curve.setData(time_axis, y_data + self.raw_offsets[i])

    def update_settings(self):
        """Opens the settings dialog and applies changes."""
        dialog = RealTimeSettings(self)
        if dialog.exec():
            new_params = dialog.get_settings()
            dialog.settings_widget.save_settings()
            filter_restart_needed = any(
                self.params.get(key) != new_params.get(key)
                for key in ("apply_bandpass", "bandpass_range", "apply_notch", "notch_freqs")
            )
            self.params.update(new_params)
            if hasattr(self, "data_worker"):
                self.data_worker.update_params(self.params)
            if filter_restart_needed:
                QMessageBox.information(
                    self,
                    "Restart Required for Filters",
                    "Filter changes are applied when the live stream connects. Relaunch the visualizer to use the new filter settings.",
                )

    def on_roi_changed(self):
        """Handles changes in the ROI selection to update the topomap."""
        if self._processed_data is None:
            return

        min_x, max_x = self.roi.getRegion()
        min_s = min_x / 1000.0
        max_s = max_x / 1000.0

        start_idx = np.searchsorted(self.times, min_s)
        end_idx = np.searchsorted(self.times, max_s)

        if start_idx >= end_idx:
            return

        mean_data = np.nanmean(self._processed_data[:, :, start_idx:end_idx], axis=(0, 2))

        if self.topomap_dialog is None:
            self.topomap_dialog = TopomapPlot(self)
            self.topomap_dialog.closed.connect(self._on_topomap_closed)
            self.topomap_dialog.show()

        self.topomap_dialog.update_plot(
            mean_data,
            self.info, # Use info object for better MNE topomaps
            title=f"Avg: {min_x:.0f} to {max_x:.0f} ms",
            names=self.ch_names
        )

    def _on_topomap_closed(self):
        self.topomap_dialog = None

    def _handle_topo_scene_click(self, event, plot_item, ch_name):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not plot_item.sceneBoundingRect().contains(event.scenePos()):
            return
        self.on_topo_pick(ch_name)

    def on_topo_pick(self, ch_name):
        """Opens a single channel plot when a topoplot trace is clicked."""
        if ch_name in self.opened_single_channels:
            self.opened_single_channels[ch_name].raise_()
            return

        dialog = SingleChannelPlot(ch_name, self.times, self)
        dialog.closed.connect(lambda ch=ch_name: self.opened_single_channels.pop(ch, None))
        self.opened_single_channels[ch_name] = dialog
        if self._processed_data is not None:
            dialog.update_plot(self._processed_data)

    @staticmethod
    def project_electrodes_to_2d(coords_3d):
        """Projects 3D electrode coordinates to 2D."""
        x, y, z = coords_3d.T
        r = np.sqrt(x**2 + y**2 + z**2)
        r[r == 0] = 1.0 # avoid division by zero
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.column_stack((theta * np.cos(phi), theta * np.sin(phi)))

    def _apply_visual_montage(self):
        if self.stream is None or self.epochs is None:
            return

        selected_montage = self.params.get("default_montage")
        if self.stream.get_montage() is not None:
            self.active_montage_name = (
                selected_montage if selected_montage and selected_montage != "None" else "Stream montage"
            )
            return

        self.active_montage_name = None

    def channel_position_text(self, ch_name: str) -> str:
        coords = self.channel_coords.get(ch_name)
        if coords is None:
            return "No channel position available."
        if all(abs(value) < 1e-9 for value in coords) and not self.active_montage_name:
            return "No channel position available."

        montage_text = self.active_montage_name or "No montage"
        return f"Montage: {montage_text} | projected x={coords[0]:.3f}, y={coords[1]:.3f}"

class RealTimeMainWidget(QWidget):
    """The main entry point widget for the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rt_erp_widget = None
        self.setup_ui()
        self.setWindowTitle("LSL Stream Visualizer")

    def setup_ui(self):
        """Sets up the UI for the main widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.setMinimumWidth(640)

        self.connection_widget = ConnectionWidget()
        self.plot_settings_widget = RealTimeSettingsWidget()
        self.player_widget = PlayerWidget()

        live_card = self._build_card("Connect to Live Stream")
        live_layout = live_card.layout()
        live_help = QLabel(
            "Find channels, confirm which inputs are EEG or stim, then launch the live visualizer."
        )
        live_help.setObjectName("mutedLabel")
        live_help.setWordWrap(True)
        live_layout.addWidget(live_help)

        config_row = QHBoxLayout()
        config_row.setContentsMargins(0, 0, 0, 0)
        config_row.setSpacing(16)
        config_row.addWidget(self.connection_widget, 2)
        config_row.addWidget(self.plot_settings_widget, 1)
        live_layout.addLayout(config_row)

        self.launch_button = QPushButton("Launch Live Visualizer")
        self.launch_button.setDefault(True)
        self.launch_button.clicked.connect(self.launch_visualizer)
        live_layout.addWidget(self.launch_button)
        layout.addWidget(live_card)

        file_card = self._build_card("Play from File")
        file_card.layout().addWidget(self.player_widget)
        layout.addWidget(file_card)
        layout.addStretch()

    def _build_card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("pipelineCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        card_layout.addWidget(title_label)
        return card

    def refresh_theme(self):
        if self.rt_erp_widget is not None:
            self.rt_erp_widget.refresh_theme()

    def launch_visualizer(self):
        """Launches the real-time ERP visualizer."""
        params = self.connection_widget.get_settings()
        if not params.get("channels"):
            QMessageBox.warning(self, "No Channel Configuration", "Find channels before launching the live visualizer.")
            return
        if not params.get("event_channels"):
            QMessageBox.warning(
                self,
                "No Event Channel",
                "Mark at least one enabled channel as 'stim' so epochs can be built from incoming events.",
            )
            return

        self.connection_widget.save_settings()
        self.plot_settings_widget.save_settings()
        params.update(self.plot_settings_widget.get_settings())
        if self.rt_erp_widget:
            self.rt_erp_widget.close()

        self.rt_erp_widget = RealTimeERP(params, self)
        self.rt_erp_widget.show()
        self.rt_erp_widget.start_visualization()

    def closeEvent(self, event):
        self.player_widget.close()
        if self.rt_erp_widget:
            self.rt_erp_widget.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_pyqtgraph_theme(current_theme_name())
    app.setStyle("Fusion")
    main_window = RealTimeMainWidget()
    main_window.show()
    sys.exit(app.exec())
