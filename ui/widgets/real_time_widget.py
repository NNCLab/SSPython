import sys
import uuid
import time
import numpy as np
import mne
import mne_lsl
import matplotlib.colors as mcolors
from dataclasses import dataclass
from collections import deque
from functools import lru_cache
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QDialog,
    QToolBar,
    QLineEdit,
    QFormLayout,
    QMessageBox,
    QHBoxLayout,
    QDialogButtonBox,
    QLabel,
    QSpinBox,
    QProgressDialog,
    QGroupBox,
    QFileDialog,
    QCheckBox,
    QMainWindow,
    QComboBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, QSettings, Slot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from shiboken6 import isValid
from .tools.optional_range_widget import OptionalRangeWidget
from .real_time_plot_docks import ChannelLayoutDock, EvokedButterflyDock, MEPDock, RawMonitorDock
from scipy import signal
from scipy.spatial import distance

from core.app_settings import get_settings_store
from utils import apply_theme, current_theme_name, theme_tokens


DEFAULT_LIVE_EVENT_ID_MAX = 65535
MEP_THRESHOLD_UV = 50.0
DEFAULT_MEP_WINDOW_MS = (15.0, 50.0)
DEFAULT_SNR_BASELINE_WINDOW = (-0.1, 0.0)
DEFAULT_SNR_RESPONSE_WINDOW = (0.0, 0.1)
REALTIME_AMPLITUDE_SCALE_OPTIONS = (
    ("Volts (V)", 1.0),
    ("Microvolts (uV)", 1e-6),
)


def normalize_realtime_amplitude_scale(value) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("microvolts", "uv").replace("microvolt", "uv")
        if "uv" in normalized or "micro volts" in normalized:
            return 1e-6
        if normalized in {"v", "volts", "volt"} or normalized.startswith("volts"):
            return 1.0
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


@dataclass(slots=True)
class RawRenderPayload:
    time_axis: np.ndarray
    scaled_data: np.ndarray


@dataclass(slots=True)
class EpochRenderPayload:
    mean_data: np.ndarray
    std_data: np.ndarray
    n_epochs: int
    buffered_epochs: int
    total_epochs: int
    global_min: float
    global_max: float


@dataclass(slots=True)
class MEPRenderPayload:
    mean_data: np.ndarray
    std_data: np.ndarray
    emg_names: list[str]
    n_epochs: int


def normalize_live_channel_type(channel_type: str | None) -> str:
    normalized = str(channel_type or "eeg").strip().lower()
    if normalized in {"eeg", "emg", "stim", "bad"}:
        return normalized
    return "bad"


def build_epoch_stream_configuration(
    channel_settings: list[dict],
) -> tuple[int | dict[str, int] | None, str | list[str] | None, list[str]]:
    event_channels: list[str] = []
    event_name_map: dict[str, int] = {}
    bads: list[str] = []

    for channel in channel_settings:
        channel_name = channel.get("name", "")
        enabled = bool(channel.get("enabled", True))
        channel_type = normalize_live_channel_type(channel.get("type", "eeg"))
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


def build_mep_trace_uV(
    mean_emg_data: np.ndarray,
    emg_names: list[str],
    active_channel: str,
    reference_channel: str | None,
) -> np.ndarray:
    data = np.asarray(mean_emg_data, dtype=float)
    if data.ndim != 2 or not emg_names or active_channel not in emg_names:
        return np.array([], dtype=float)

    active_idx = emg_names.index(active_channel)
    trace = data[active_idx].copy()
    if reference_channel and reference_channel in emg_names and reference_channel != active_channel:
        trace = trace - data[emg_names.index(reference_channel)]
    return trace * 1e6


def mep_measurement_mask(
    times_ms: np.ndarray,
    window_ms: tuple[float, float] = DEFAULT_MEP_WINDOW_MS,
) -> np.ndarray:
    times = np.asarray(times_ms, dtype=float)
    if times.size == 0:
        return np.array([], dtype=bool)

    start_ms, end_ms = sorted((float(window_ms[0]), float(window_ms[1])))
    mask = (times >= start_ms) & (times <= end_ms)
    if np.any(mask):
        return mask

    post_stim_mask = times >= 0.0
    if np.any(post_stim_mask):
        return post_stim_mask
    return np.ones(times.shape, dtype=bool)


def compute_mep_peak_to_peak_uV(
    trace_uV: np.ndarray,
    times_ms: np.ndarray,
    window_ms: tuple[float, float] = DEFAULT_MEP_WINDOW_MS,
) -> tuple[float, np.ndarray]:
    trace = np.asarray(trace_uV, dtype=float)
    mask = mep_measurement_mask(times_ms, window_ms)
    if trace.size == 0 or mask.size != trace.size:
        return np.nan, mask

    window_values = trace[mask]
    finite_values = window_values[np.isfinite(window_values)]
    if finite_values.size == 0:
        return np.nan, mask
    return float(np.max(finite_values) - np.min(finite_values)), mask


def validate_realtime_info(info: mne.Info) -> mne.Info:
    if info is None:
        raise ValueError("Real-time info requires an MNE Info object.")
    if not isinstance(info, mne.Info):
        raise TypeError("Real-time info must be an mne.Info object.")
    if len(info["ch_names"]) == 0:
        raise ValueError("Real-time info must define at least one channel.")

    eeg_picks = mne.pick_types(info, eeg=True, exclude=())
    if len(eeg_picks) == 0:
        raise ValueError("Real-time info must include at least one EEG channel.")
    return info


def load_realtime_info(source_path: Path) -> mne.Info:
    source_path = Path(source_path)
    raw = None
    try:
        info = mne.io.read_info(source_path, verbose="error")
        return validate_realtime_info(info)
    except Exception:
        pass

    try:
        raw = mne.io.read_raw(source_path, preload=False, verbose="error")
        return validate_realtime_info(raw.info.copy())
    except Exception as exc:
        raise ValueError(f"Could not load an MNE Info object from {source_path.name}.") from exc
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()

def realtime_info_has_montage(info: mne.Info | None) -> bool:
    if info is None or info.get_montage() is None:
        return False

    eeg_picks = mne.pick_types(info, eeg=True, exclude=())
    if len(eeg_picks) == 0:
        return False

    positions = []
    for pick in eeg_picks:
        loc = info["chs"][pick].get("loc", np.zeros(12))
        if loc is not None and len(loc) >= 3:
            positions.append(loc[:3])
    if not positions:
        return False

    positions = np.asarray(positions, dtype=float)
    valid_positions = np.isfinite(positions).all(axis=1) & np.any(np.abs(positions) > 1e-9, axis=1)
    return bool(np.any(valid_positions))


def live_channel_type_from_info(channel_type: str) -> str:
    normalized = str(channel_type or "").strip().lower()
    if normalized in {"eeg", "emg", "stim"}:
        return normalized
    return "bad"


def channel_settings_from_info(info: mne.Info) -> list[dict]:
    validated = validate_realtime_info(info)
    bads = set(validated.get("bads", []))
    settings: list[dict] = []
    for name, channel_type in zip(
        validated["ch_names"],
        validated.get_channel_types(),
        strict=False,
    ):
        live_type = live_channel_type_from_info(channel_type)
        settings.append(
            {
                "name": name,
                "enabled": name not in bads and live_type != "bad",
                "type": live_type,
            }
        )
    return settings


def apply_realtime_info_to_stream(stream, info: mne.Info):
    realtime_info = validate_realtime_info(info)
    if len(stream.ch_names) != len(realtime_info["ch_names"]):
        raise ValueError(
            f"Stream has {len(stream.ch_names)} channels, "
            f"but real-time info defines {len(realtime_info['ch_names'])} channels."
        )
    if not np.isclose(float(stream.info["sfreq"]), float(realtime_info["sfreq"])):
        raise ValueError(
            f"Stream sampling frequency {float(stream.info['sfreq']):g} Hz does not match "
            f"real-time info sampling frequency {float(realtime_info['sfreq']):g} Hz."
        )

    rename_map = {
        old_name: new_name
        for old_name, new_name in zip(stream.ch_names, realtime_info["ch_names"], strict=False)
        if old_name != new_name
    }
    if rename_map:
        stream.rename_channels(rename_map)

    channel_types = {
        channel_name: channel_type
        for channel_name, channel_type in zip(
            realtime_info["ch_names"],
            realtime_info.get_channel_types(),
            strict=False,
        )
    }
    if channel_types:
        stream.set_channel_types(channel_types)

    stream.info["bads"] = [
        channel_name
        for channel_name in realtime_info["bads"]
        if channel_name in stream.info["ch_names"]
    ]
    montage = realtime_info.get_montage()
    if montage is not None:
        try:
            stream.set_montage(
                montage,
                on_missing="ignore",
                match_case=False,
            )
        except TypeError:
            stream.set_montage(montage)
    return stream


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


@lru_cache(maxsize=4)
def build_wildcard_event_id_map(max_event_id: int = DEFAULT_LIVE_EVENT_ID_MAX) -> dict[str, int]:
    """Return a broad positive-ID mapping so regular stim channels accept any trigger code."""
    max_event_id = max(1, int(max_event_id))
    return {str(event_id): event_id for event_id in range(1, max_event_id + 1)}


def resolve_regular_stream_event_id(
    stream,
    event_id: int | dict[str, int] | None,
    event_channels: str | list[str] | None,
) -> int | dict[str, int]:
    if event_id is not None:
        return event_id

    channels = normalize_event_channels(event_channels)
    if not channels:
        raise ValueError("No stim channel was selected for epoching.")

    # mne-lsl requires a positive event_id for regularly sampled stim channels.
    # When the user does not configure one, accept any positive trigger code so
    # the stream can connect immediately and epochs appear as events arrive.
    return build_wildcard_event_id_map()


def _rgb(x, y, z):
    """Transform x, y, z values into RGB colors."""
    rgb = np.array([x, y, z], dtype=float).T
    rgb -= np.nanmin(rgb, 0)
    rgb /= np.maximum(np.nanmax(rgb, 0), 1e-16)
    overly_light = rgb.sum(axis=1) > 2.5
    rgb[overly_light] = np.clip(rgb[overly_light] - 0.3, 0.0, 1.0)
    return rgb


def _default_trace_colors(count: int) -> list[tuple[float, float, float, float]]:
    if count <= 0:
        return []

    color = mcolors.to_rgba(theme_tokens().get("accent_soft", "#2f7e8d"))
    return [(float(color[0]), float(color[1]), float(color[2]), float(color[3])) for _ in range(count)]


def build_trace_colors(
    count: int,
    positions_3d: np.ndarray | None = None,
    *,
    prefer_position_colors: bool = False,
) -> list[tuple[float, float, float, float]]:
    fallback_colors = _default_trace_colors(count)
    if count <= 0 or not prefer_position_colors or positions_3d is None:
        return fallback_colors

    xyz = np.asarray(positions_3d, dtype=float)
    if xyz.ndim != 2 or xyz.shape[0] < count or xyz.shape[1] < 3:
        return fallback_colors

    xyz = xyz[:count, :3]
    valid_positions = np.isfinite(xyz).all(axis=1) & np.any(np.abs(xyz) > 1e-9, axis=1)
    if not np.any(valid_positions):
        return fallback_colors

    rgb_colors = _rgb(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    colors: list[tuple[float, float, float, float]] = []
    for index in range(count):
        if valid_positions[index]:
            colors.append(
                (
                    float(np.clip(rgb_colors[index, 0], 0.0, 1.0)),
                    float(np.clip(rgb_colors[index, 1], 0.0, 1.0)),
                    float(np.clip(rgb_colors[index, 2], 0.0, 1.0)),
                    1.0,
                )
            )
        else:
            colors.append(fallback_colors[index])
    return colors


def _box_size(loc, w, h, padding=0.1):
    """Compute the maximum box size without overlap."""
    if w is None and h is None:
        if loc.shape[0] <= 1:
            w = h = 1.0
        else:
            dist = distance.pdist(loc)
            if np.all(dist == 0):
                w = h = 1.0
            else:
                dist = dist[dist > 0]
                min_dist = np.min(dist)
                w = h = min_dist / np.sqrt(2)
    elif w is None:
        w = h
    elif h is None:
        h = w
    return w, h


def calculate_mne_style_layout(coords_2d, radius=0.5, width=None, height=None):
    """Calculate channel positions in MNE style for a topomap."""
    if coords_2d.size == 0:
        return np.array([])

    loc2d = coords_2d.copy()

    # Scale [x, y] to be in the range [-0.5, 0.5]
    scale = np.maximum(-np.min(loc2d, axis=0), np.max(loc2d, axis=0)).max() * 2
    if scale == 0:
        scale = 1.0
    loc2d /= scale

    # If no width or height specified, calculate the maximum value possible
    if width is None or height is None:
        width, height = _box_size(loc2d, width, height, padding=0.1)

    # Scale to viewport radius
    loc2d *= 2 * radius

    # Some subplot centers will be at the figure edge. Shrink everything so it fits
    scaling = min(1 / (1.0 + width), 1 / (1.0 + height))
    loc2d *= scaling
    width *= scaling
    height *= scaling

    # Shift to center
    loc2d += 0.5

    n_channels = loc2d.shape[0]
    pos = np.c_[
        loc2d[:, 0] - 0.5 * width,
        loc2d[:, 1] - 0.5 * height,
        width * np.ones(n_channels),
        height * np.ones(n_channels),
    ]
    return pos


def channel_grid_positions(coords_2d: np.ndarray) -> list[tuple[float, float]]:
    """Return stable 2D layout anchors for channel preview placement."""
    positions = calculate_mne_style_layout(np.asarray(coords_2d, dtype=float))
    return [tuple(float(value) for value in row[:2]) for row in positions]


def apply_artifact_mask(
    epoch_batch: np.ndarray,
    times: np.ndarray,
    art_rem: tuple[float | None, float | None],
) -> np.ndarray:
    """Mask the artifact window with NaN so downstream averaging ignores it."""
    plot_data = epoch_batch.copy()
    art_rem_start, art_rem_end = art_rem
    if art_rem_start is None or art_rem_end is None or art_rem_end <= art_rem_start:
        return plot_data

    start_idx = np.searchsorted(times, art_rem_start, side="left")
    end_idx = np.searchsorted(times, art_rem_end, side="right")
    if end_idx > start_idx:
        plot_data[:, :, start_idx:end_idx] = np.nan
    return plot_data


def apply_raw_artifact_mask(
    raw_chunk: np.ndarray,
    stim_chunk: np.ndarray,
    sfreq: float,
    art_rem: tuple[float | None, float | None],
) -> np.ndarray:
    """Mask raw samples around positive stim onsets so the monitor skips artifact bursts."""
    segments = raw_artifact_segments(stim_chunk, sfreq, art_rem)
    if not segments:
        return np.array(raw_chunk, copy=True, dtype=float)
    return _mask_segments_2d(raw_chunk, segments)


def normalize_live_event_values(
    event_id: int | dict[str, int] | None,
) -> set[int] | None:
    if event_id is None:
        return None
    if isinstance(event_id, dict):
        return {int(value) for value in event_id.values() if int(value) > 0}
    value = int(event_id)
    return {value} if value > 0 else set()


def detect_stim_onsets(
    stim_chunk: np.ndarray,
    *,
    previous_value: int = 0,
    allowed_event_values: set[int] | None = None,
) -> tuple[np.ndarray, int]:
    stim_values = np.asarray(stim_chunk, dtype=int)
    if stim_values.ndim == 1:
        stim_values = stim_values[np.newaxis, :]
    if stim_values.size == 0 or stim_values.shape[-1] == 0:
        return np.array([], dtype=int), int(previous_value)

    combined = np.max(np.maximum(stim_values, 0), axis=0)
    previous = np.concatenate(([int(previous_value)], combined[:-1]))
    event_samples = np.flatnonzero((combined > 0) & (previous <= 0))
    if allowed_event_values is not None:
        event_samples = event_samples[np.isin(combined[event_samples], list(allowed_event_values))]
    last_value = int(combined[-1]) if combined.size > 0 else int(previous_value)
    return event_samples.astype(int, copy=False), last_value


def raw_artifact_segments(
    stim_chunk: np.ndarray,
    sfreq: float,
    art_rem: tuple[float | None, float | None],
) -> list[tuple[int, int]]:
    """Return sample spans affected by raw artifact removal."""
    stim_values = np.asarray(stim_chunk)
    if stim_values.ndim == 1:
        stim_values = stim_values[np.newaxis, :]
    art_rem_start, art_rem_end = art_rem
    if (
        stim_values.size == 0
        or stim_values.shape[-1] == 0
        or art_rem_start is None
        or art_rem_end is None
        or art_rem_end <= art_rem_start
    ):
        return []

    event_samples, _ = detect_stim_onsets(stim_values)
    if event_samples.size == 0:
        return []

    start_offset = int(np.floor(float(art_rem_start) * float(sfreq)))
    end_offset = int(np.ceil(float(art_rem_end) * float(sfreq))) + 1
    segments = [
        (int(event_sample + start_offset), int(event_sample + end_offset))
        for event_sample in event_samples
    ]
    return _merge_sample_segments(segments, stim_values.shape[-1])


def _merge_sample_segments(
    segments: list[tuple[int, int]],
    sample_count: int,
) -> list[tuple[int, int]]:
    if sample_count <= 0 or not segments:
        return []

    ordered = sorted(
        (
            max(0, int(start_idx)),
            min(sample_count, int(end_idx)),
        )
        for start_idx, end_idx in segments
    )
    merged: list[list[int]] = []
    for start_idx, end_idx in ordered:
        if end_idx <= start_idx:
            continue
        if not merged or start_idx > merged[-1][1]:
            merged.append([start_idx, end_idx])
            continue
        merged[-1][1] = max(merged[-1][1], end_idx)
    return [(start_idx, end_idx) for start_idx, end_idx in merged]


def _interpolate_segments_2d(data: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    filled = np.array(data, copy=True, dtype=float)
    if filled.ndim != 2 or filled.shape[-1] == 0:
        return filled

    for start_idx, end_idx in _merge_sample_segments(segments, filled.shape[-1]):
        left_idx = start_idx - 1
        right_idx = end_idx
        if left_idx < 0 and right_idx >= filled.shape[-1]:
            filled[:, start_idx:end_idx] = 0.0
            continue
        if left_idx < 0:
            filled[:, start_idx:end_idx] = filled[:, right_idx][:, np.newaxis]
            continue
        if right_idx >= filled.shape[-1]:
            filled[:, start_idx:end_idx] = filled[:, left_idx][:, np.newaxis]
            continue

        weights = np.linspace(0.0, 1.0, end_idx - start_idx + 2, dtype=float)[1:-1]
        left = filled[:, left_idx][:, np.newaxis]
        right = filled[:, right_idx][:, np.newaxis]
        filled[:, start_idx:end_idx] = left + (right - left) * weights[np.newaxis, :]
    return filled


def _mask_segments_2d(data: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    masked = np.array(data, copy=True, dtype=float)
    if masked.ndim != 2 or masked.shape[-1] == 0:
        return masked

    for start_idx, end_idx in _merge_sample_segments(segments, masked.shape[-1]):
        masked[:, start_idx:end_idx] = np.nan
    return masked


def _interpolate_segments_3d(data: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    filled = np.array(data, copy=True, dtype=float)
    if filled.ndim != 3 or filled.shape[-1] == 0:
        return filled

    for start_idx, end_idx in _merge_sample_segments(segments, filled.shape[-1]):
        left_idx = start_idx - 1
        right_idx = end_idx
        if left_idx < 0 and right_idx >= filled.shape[-1]:
            filled[:, :, start_idx:end_idx] = 0.0
            continue
        if left_idx < 0:
            filled[:, :, start_idx:end_idx] = filled[:, :, right_idx][:, :, np.newaxis]
            continue
        if right_idx >= filled.shape[-1]:
            filled[:, :, start_idx:end_idx] = filled[:, :, left_idx][:, :, np.newaxis]
            continue

        weights = np.linspace(0.0, 1.0, end_idx - start_idx + 2, dtype=float)[1:-1]
        left = filled[:, :, left_idx][:, :, np.newaxis]
        right = filled[:, :, right_idx][:, :, np.newaxis]
        filled[:, :, start_idx:end_idx] = left + (right - left) * weights[np.newaxis, np.newaxis, :]
    return filled


def fill_raw_artifact_window(
    raw_chunk: np.ndarray,
    stim_chunk: np.ndarray,
    sfreq: float,
    art_rem: tuple[float | None, float | None],
) -> np.ndarray:
    segments = raw_artifact_segments(stim_chunk, sfreq, art_rem)
    if not segments:
        return np.array(raw_chunk, copy=True, dtype=float)

    return _interpolate_segments_2d(raw_chunk, segments)


def fill_epoch_artifact_window(
    epoch_batch: np.ndarray,
    times: np.ndarray,
    art_rem: tuple[float | None, float | None],
) -> np.ndarray:
    art_rem_start, art_rem_end = art_rem
    if art_rem_start is None or art_rem_end is None or art_rem_end <= art_rem_start:
        return np.array(epoch_batch, copy=True, dtype=float)

    start_idx = np.searchsorted(times, art_rem_start, side="left")
    end_idx = np.searchsorted(times, art_rem_end, side="right")
    return _interpolate_segments_3d(epoch_batch, [(start_idx, end_idx)])


def apply_baseline_correction(
    epoch_batch: np.ndarray,
    times: np.ndarray,
    baseline: tuple[float | None, float | None] = (None, 0.0),
) -> np.ndarray:
    corrected = np.array(epoch_batch, copy=True, dtype=float)
    if corrected.ndim != 3 or corrected.shape[-1] == 0 or times.size == 0:
        return corrected

    baseline_start, baseline_end = baseline
    mask = np.ones(times.shape, dtype=bool)
    if baseline_start is not None:
        mask &= times >= float(baseline_start)
    if baseline_end is not None:
        mask &= times <= float(baseline_end)
    if not np.any(mask):
        return corrected

    baseline_values = corrected[:, :, mask]
    baseline_mean = np.nanmean(baseline_values, axis=-1, keepdims=True)
    baseline_mean = np.where(np.isfinite(baseline_mean), baseline_mean, 0.0)
    return corrected - baseline_mean


def average_reference_ignore_nan(epoch_batch: np.ndarray) -> np.ndarray:
    """Apply average reference while preserving NaN-masked samples."""
    if epoch_batch.shape[1] <= 1:
        return epoch_batch

    finite_mask = np.isfinite(epoch_batch)
    counts = np.sum(finite_mask, axis=1, keepdims=True)
    sums = np.where(finite_mask, epoch_batch, 0.0).sum(axis=1, keepdims=True)
    channel_mean = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=float),
        where=counts > 0,
    )
    return epoch_batch - channel_mean


def buffer_epoch_snapshot(
    epoch_buffer: np.ndarray,
    buffer_idx: int,
    n_valid_epochs: int,
    display_epoch_count: int = 0,
) -> np.ndarray:
    """Return a chronological snapshot of the buffered epochs."""
    if n_valid_epochs <= 0:
        return epoch_buffer[:0].copy()

    max_epochs = epoch_buffer.shape[0]
    if n_valid_epochs < max_epochs:
        ordered = epoch_buffer[:n_valid_epochs]
    else:
        ordered = np.concatenate(
            (epoch_buffer[buffer_idx:], epoch_buffer[:buffer_idx]),
            axis=0,
        )

    if display_epoch_count > 0:
        ordered = ordered[-min(display_epoch_count, ordered.shape[0]) :]

    return ordered.copy()


def append_limited_history(
    history: np.ndarray,
    chunk: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, int]:
    """Append samples while keeping only the most recent max_samples columns."""
    history = np.asarray(history, dtype=float)
    chunk = np.asarray(chunk, dtype=float)
    max_samples = max(0, int(max_samples))

    if chunk.ndim != 2:
        raise ValueError("chunk must be a 2D array")
    if history.ndim != 2:
        raise ValueError("history must be a 2D array")
    if history.shape[0] != chunk.shape[0]:
        raise ValueError("history and chunk must have the same channel count")
    if max_samples == 0:
        return np.empty((chunk.shape[0], 0), dtype=float), history.shape[1] + chunk.shape[1]
    if chunk.shape[1] == 0:
        return history[:, -max_samples:].copy(), max(0, history.shape[1] - max_samples)

    total_samples = history.shape[1] + chunk.shape[1]
    dropped_samples = max(0, total_samples - max_samples)
    if chunk.shape[1] >= max_samples:
        return chunk[:, -max_samples:].copy(), dropped_samples

    history_keep = max_samples - chunk.shape[1]
    if history.shape[1] == 0:
        return chunk.copy(), dropped_samples

    history_tail = history[:, -history_keep:]
    return np.concatenate((history_tail, chunk), axis=1), dropped_samples


def compute_epoch_mean_std(epoch_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std across epochs while ignoring NaN-masked samples."""
    if epoch_batch.size == 0:
        raise ValueError("epoch_batch must not be empty")

    finite_mask = np.isfinite(epoch_batch)
    counts = np.sum(finite_mask, axis=0)
    sums = np.where(finite_mask, epoch_batch, 0.0).sum(axis=0)
    mean_data = np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=float),
        where=counts > 0,
    )

    centered = np.where(finite_mask, epoch_batch - mean_data[np.newaxis, :, :], 0.0)
    sq_sums = np.square(centered).sum(axis=0)
    std_data = np.sqrt(
        np.divide(
            sq_sums,
            counts,
            out=np.full(mean_data.shape, np.nan, dtype=float),
            where=counts > 0,
        )
    )
    return mean_data, std_data


def build_live_filter_pipeline(
    sfreq: float,
    params: dict,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, np.ndarray]]]:
    """Build dynamic filter coefficients for live visualization."""
    nyquist = sfreq / 2.0
    bandpass_sos = None
    notch_filters: list[tuple[np.ndarray, np.ndarray]] = []

    if params.get("apply_bandpass", False):
        low, high = params.get("bandpass_range", (None, None))
        low = float(low) if low is not None else None
        high = float(high) if high is not None else None

        if low is not None and low <= 0:
            low = None
        if high is not None and high >= nyquist:
            high = None

        if low is not None and high is not None and low < high:
            bandpass_sos = signal.butter(
                4,
                [low, high],
                btype="bandpass",
                fs=sfreq,
                output="sos",
            )
        elif low is not None:
            bandpass_sos = signal.butter(
                4,
                low,
                btype="highpass",
                fs=sfreq,
                output="sos",
            )
        elif high is not None:
            bandpass_sos = signal.butter(
                4,
                high,
                btype="lowpass",
                fs=sfreq,
                output="sos",
            )

    if params.get("apply_notch", False):
        for freq in params.get("notch_freqs", []) or []:
            try:
                freq_value = float(freq)
            except (TypeError, ValueError):
                continue
            if 0 < freq_value < nyquist:
                notch_filters.append(signal.iirnotch(freq_value, Q=30.0, fs=sfreq))

    return bandpass_sos, notch_filters


def apply_frequency_filters(
    data: np.ndarray,
    bandpass_sos: np.ndarray | None,
    notch_filters: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Apply the active live filters to raw or epoched data."""
    if bandpass_sos is None and not notch_filters:
        return np.array(data, copy=True)

    filtered = np.array(data, copy=True, dtype=float)
    if bandpass_sos is not None:
        try:
            filtered = signal.sosfiltfilt(bandpass_sos, filtered, axis=-1)
        except ValueError:
            filtered = signal.sosfilt(bandpass_sos, filtered, axis=-1)

    for b, a in notch_filters:
        try:
            filtered = signal.filtfilt(b, a, filtered, axis=-1)
        except ValueError:
            filtered = signal.lfilter(b, a, filtered, axis=-1)

    return filtered


def frequency_filters_enabled(
    bandpass_sos: np.ndarray | None,
    notch_filters: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    return bandpass_sos is not None or bool(notch_filters)

class PlayerWidget(QWidget):
    """A widget to play a file as an LSL stream."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = None
        self.setWindowTitle("LSL File Player")
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setVerticalSpacing(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        
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
        self.chunk_size_input.setValue(128)
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
                name=stream_name,
                n_repeat=np.inf
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
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, stream_name):
        super().__init__()
        self.stream_name = stream_name

    def run(self):
        try:
            # Short duration, we just want the info
            stream = mne_lsl.stream.StreamLSL(1, name=self.stream_name)
            stream.connect(acquisition_delay=0.1, processing_flags="all", timeout=5)
            info = stream.info.copy()
            stream.disconnect()
            self.finished.emit(info)
        except Exception as e:
            self.error.emit(f"Could not find or connect to stream '{self.stream_name}': {e}")

class ConnectionWidget(QWidget):
    """A widget to get stream connection parameters from the user."""

    SETTINGS_PATH = "real_time/connection"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self.ch_names = []
        self.realtime_info: mne.Info | None = None
        self.info_source_path: str | None = None
        self.info_label: str | None = None
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Initializes the user interface for the connection settings."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # --- Connection Group ---
        conn_group_box = QGroupBox("Connection Settings")
        form_layout = QFormLayout(conn_group_box)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form_layout.setVerticalSpacing(8)
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
        self.info_edit = QLineEdit("No MNE info selected...")
        self.info_edit.setReadOnly(True)
        self.select_info_button = QPushButton("Select Info Source")
        self.find_channels_button = QPushButton("Find Info from Stream")
        form_layout.addRow("Stream Name:", self.stream_name_input)
        form_layout.addRow("Stream Duration (s):", self.stream_duration_input)
        form_layout.addRow("Epoching (t_start, t_end):", self.tlim_input)
        form_layout.addRow("MNE Info:", self.info_edit)
        form_layout.addRow(self.select_info_button)
        form_layout.addRow(self.find_channels_button)
        main_layout.addWidget(conn_group_box)

        info_group_box = QGroupBox("Info Summary")
        info_layout = QVBoxLayout(info_group_box)
        self.info_summary_label = QLabel("No MNE info loaded.")
        self.info_summary_label.setObjectName("mutedLabel")
        self.info_summary_label.setWordWrap(True)
        info_layout.addWidget(self.info_summary_label)
        main_layout.addWidget(info_group_box)
        
        # --- Connections ---
        self.select_info_button.clicked.connect(self.select_info_source)
        self.find_channels_button.clicked.connect(self.find_channels)

    def select_info_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select MNE Info Source",
            "",
            "MNE files (*.fif *.fif.gz);;All files (*)",
        )
        if not file_path:
            return

        try:
            info = load_realtime_info(file_path)
            self.set_realtime_info(
                info,
                label=Path(file_path).name,
                source_path=file_path,
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid MNE Info", str(exc))

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

    def on_channels_found(self, info):
        """Applies the stream info after finding a stream."""
        try:
            self.set_realtime_info(info, label="Stream info")
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Stream Info", str(exc))
        self.find_channels_button.setEnabled(True)
        self.find_channels_button.setText("Find Info from Stream")

    def on_find_error(self, error_message):
        QMessageBox.critical(self, "Error Finding Stream", error_message)
        self.find_channels_button.setEnabled(True)
        self.find_channels_button.setText("Find Info from Stream")

    def _info_channel_settings(self) -> list[dict]:
        if self.realtime_info is None:
            self.ch_names = []
            return []
        settings = channel_settings_from_info(self.realtime_info)
        self.ch_names = [channel["name"] for channel in settings]
        return settings

    def _update_info_summary(self):
        if self.realtime_info is None:
            self.info_summary_label.setText("No MNE info loaded.")
            return

        settings = self._info_channel_settings()
        eeg_count = sum(1 for channel in settings if channel["type"] == "eeg")
        emg_count = sum(1 for channel in settings if channel["type"] == "emg")
        stim_channels = [channel["name"] for channel in settings if channel["type"] == "stim"]
        bad_channels = [channel["name"] for channel in settings if not channel["enabled"]]
        stim_text = ", ".join(stim_channels) if stim_channels else "None"
        bad_text = ", ".join(bad_channels) if bad_channels else "None"
        montage_text = "Yes" if realtime_info_has_montage(self.realtime_info) else "No"
        self.info_summary_label.setText(
            f"{len(settings)} channels | EEG: {eeg_count} | EMG: {emg_count} | Stim: {stim_text} | "
            f"Excluded: {bad_text} | Montage: {montage_text}"
        )

    def set_realtime_info(
        self,
        info: mne.Info,
        *,
        label: str | None = None,
        source_path: str | None = None,
    ):
        realtime_info = validate_realtime_info(info).copy()
        self.realtime_info = realtime_info
        self.info_source_path = str(source_path) if source_path else None
        self.info_label = label or "Provided MNE Info"
        self.info_edit.setText(
            f"{self.info_label} ({len(realtime_info['ch_names'])} channels, {realtime_info['sfreq']:g} Hz)"
        )
        montage_text = "includes a montage" if realtime_info_has_montage(realtime_info) else "has no montage"
        self.info_edit.setToolTip(f"Real-time info {montage_text}.")
        self._update_info_summary()

    def get_settings(self) -> dict:
        base_settings = {
            "stream_name": self.stream_name_input.text(),
            "stream_duration": self.stream_duration_input.value(),
            "tlim": self.tlim_input.value(),
        }
        if self.realtime_info is not None:
            base_settings["info"] = self.realtime_info.copy()
            base_settings["info_label"] = self.info_label
        channel_settings = self._info_channel_settings()
        event_id, event_channels, bads = build_epoch_stream_configuration(channel_settings)

        base_settings['event_id'] = event_id
        base_settings['event_channels'] = event_channels
        base_settings['bads'] = bads

        return base_settings

    def save_settings(self):
        """Saves the current settings to QSettings."""
        params = {
            "stream_name": self.stream_name_input.text(),
            "stream_duration": self.stream_duration_input.value(),
            "tlim": self.tlim_input.value(),
        }
        if self.info_source_path:
            params["info_source_path"] = self.info_source_path

        self.settings_store.set(self.SETTINGS_PATH, params)
        self.settings_store.sync()

    def load_settings(self):
        """Loads settings from QSettings, applying defaults for missing values."""
        default_params = {
            "stream_name": "EEGStream",
            "stream_duration": 5,
            "tlim": (-0.1, 0.3),
            "info_source_path": "",
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

        info_source_path = params.get("info_source_path")
        if info_source_path:
            try:
                self.set_realtime_info(
                    load_realtime_info(info_source_path),
                    label=Path(info_source_path).name,
                    source_path=info_source_path,
                )
                return
            except (TypeError, ValueError):
                self.info_edit.setText("Saved MNE info source could not be loaded.")


class ConnectionManager:

    """Handles the connection to an LSL stream."""

    def __init__(self, params):
        self.params = params
        self.raw = None
        self.last_error = ""

    def connect_to_stream(self):
        """Establishes a connection to the LSL stream."""
        try:
            stream_duration = self.params.get("stream_duration")
            stream_name = self.params.get("stream_name")
            event_id = self.params.get("event_id")
            event_channels = self.params.get("event_channels")
            event_channel_list = normalize_event_channels(event_channels)
            realtime_info = self.params.get("info")
            if realtime_info is None:
                raise ValueError("Real-time visualization requires an mne.Info object.")

            self.raw = mne_lsl.stream.StreamLSL(stream_duration, name=stream_name)
            self.raw.connect(
                acquisition_delay=0.1, processing_flags="all", timeout=5
            )

            apply_realtime_info_to_stream(self.raw, realtime_info)

            bads = self.params.get('bads', [])
            if bads:
                self.raw.info['bads'] = bads

            missing_event_channels = [
                channel
                for channel in event_channel_list
                if channel not in self.raw.info["ch_names"]
            ]
            if missing_event_channels:
                raise ValueError(
                    "Event channel(s) are not present in the real-time info: "
                    + ", ".join(missing_event_channels)
                )

            event_id = resolve_regular_stream_event_id(
                self.raw,
                event_id,
                event_channel_list,
            )
            self.params["event_id"] = event_id
            print("Connection successful.")
            return self.raw
        except Exception as e:
            self.last_error = str(e)
            print(f"Failed to connect to stream: {e}")
            return None

    def disconnect(self):
        if self.raw is not None:
            try:
                self.raw.disconnect()
            except Exception:
                pass

class ConnectionWorker(QObject):
    """Worker thread for handling the LSL stream connection."""

    finished = Signal(object)  # Emits the connected stream on success
    error = Signal(str)  # Emits error message on failure

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.conn_manager = None
        self._stop_requested = False

    def run(self):
        """Tries to connect to the stream."""
        try:
            if self._stop_requested:
                return
            self.conn_manager = ConnectionManager(self.params)
            stream = self.conn_manager.connect_to_stream()
            if self._stop_requested:
                if self.conn_manager is not None:
                    self.conn_manager.disconnect()
                return
            if stream:
                self.finished.emit(stream)
            else:
                self.error.emit(
                    self.conn_manager.last_error
                    or "Failed to connect to the LSL stream. Please check the stream name and ensure it is available."
                )
        except Exception as e:
            self.error.emit(f"An error occurred during connection: {e}")
        finally:
            if self._stop_requested and self.conn_manager is not None:
                self.conn_manager.disconnect()

    @Slot()
    def stop(self):
        self._stop_requested = True
        if self.conn_manager is not None:
            self.conn_manager.disconnect()

class RealTimeSettingsWidget(QWidget):
    """A widget to configure real-time plotting parameters."""

    SETTINGS_PATH = "real_time/plotting"
    settingsChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_store = get_settings_store()
        self._suppress_settings_changed = False
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Initializes the user interface for the plot settings."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        group_box = QGroupBox("Real-Time Plot Settings")
        group_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        main_layout.addWidget(group_box)

        form_layout = QFormLayout(group_box)
        form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form_layout.setVerticalSpacing(8)

        self.refresh_rate_input = QSpinBox()
        self.refresh_rate_input.setSuffix(" Hz")
        self.refresh_rate_input.setRange(1, 60)
        form_layout.addRow("Refresh Rate:", self.refresh_rate_input)

        form_layout.addRow(QLabel("<br><b>Butterfly SNR</b>"))

        self.snr_baseline_input = OptionalRangeWidget(
            ("Start:", "End:"),
            suffix=" ms",
            required=(True, True),
            range=(None, None),
            scale=1e-3,
        )
        self.snr_response_input = OptionalRangeWidget(
            ("Start:", "End:"),
            suffix=" ms",
            required=(True, True),
            range=(None, None),
            scale=1e-3,
        )
        for range_widget in (self.snr_baseline_input, self.snr_response_input):
            for spinbox in (range_widget.low_input, range_widget.high_input):
                spinbox.setRange(-5000.0, 5000.0)
                spinbox.setDecimals(1)
            range_widget.adjust_spinbox_width()
        form_layout.addRow("Baseline RMS:", self.snr_baseline_input)
        form_layout.addRow("Response RMS:", self.snr_response_input)

        form_layout.addRow(QLabel("<br><b>Real-Time Pre-processing</b>"))

        self.art_rem_input = OptionalRangeWidget(
            ("Start:", "End:"),
            range=(None, None),
            suffix=" ms",
            required=(True, True),
            scale=1e-3,
        )
        form_layout.addRow("Artifact Removal:", self.art_rem_input)

        self.amplitude_scale_combo = QComboBox()
        for label, scale in REALTIME_AMPLITUDE_SCALE_OPTIONS:
            self.amplitude_scale_combo.addItem(label, scale)
        form_layout.addRow("Input Amplitude:", self.amplitude_scale_combo)

        self.average_reference_checkbox = QCheckBox("Apply Average Reference")
        form_layout.addRow(self.average_reference_checkbox)

        self.apply_bandpass = QCheckBox("Apply Bandpass Filter")
        self.bandpass_input = OptionalRangeWidget(
            labels=("Low:", "High:"), suffix=" Hz", range=(0.1, 200)
        )
        form_layout.addRow(self.apply_bandpass, self.bandpass_input)

        self.apply_notch = QCheckBox("Apply Notch Filter")
        self.notch_input = QLineEdit()
        self.notch_input.setPlaceholderText("50, 100")
        form_layout.addRow(self.apply_notch, self.notch_input)

        self.apply_bandpass.toggled.connect(self.bandpass_input.setEnabled)
        self.apply_notch.toggled.connect(self.notch_input.setEnabled)
        self.refresh_rate_input.valueChanged.connect(self._emit_settings_changed)
        self.art_rem_input.valueChanged.connect(self._emit_settings_changed)
        self.amplitude_scale_combo.currentIndexChanged.connect(self._emit_settings_changed)
        self.average_reference_checkbox.toggled.connect(self._emit_settings_changed)
        self.apply_bandpass.toggled.connect(self._emit_settings_changed)
        self.bandpass_input.valueChanged.connect(self._emit_settings_changed)
        self.apply_notch.toggled.connect(self._emit_settings_changed)
        self.notch_input.textChanged.connect(self._emit_settings_changed)
        self.snr_baseline_input.valueChanged.connect(self._emit_settings_changed)
        self.snr_response_input.valueChanged.connect(self._emit_settings_changed)
        main_layout.addStretch(1)

    def _set_amplitude_scale(self, value):
        scale = normalize_realtime_amplitude_scale(value)
        for index in range(self.amplitude_scale_combo.count()):
            if np.isclose(float(self.amplitude_scale_combo.itemData(index)), scale):
                self.amplitude_scale_combo.setCurrentIndex(index)
                return
        self.amplitude_scale_combo.setCurrentIndex(0)

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
            "snr_baseline": self.snr_baseline_input.value(),
            "snr_response": self.snr_response_input.value(),
            "art_rem": self.art_rem_input.value(),
            "amplitude_scale": normalize_realtime_amplitude_scale(self.amplitude_scale_combo.currentData()),
            "reference": "average" if self.average_reference_checkbox.isChecked() else "none",
            "apply_bandpass": self.apply_bandpass.isChecked(),
            "bandpass_range": self.bandpass_input.value(),
            "apply_notch": self.apply_notch.isChecked(),
            "notch_freqs": notch_freqs,
        }

    def set_settings(self, params: dict):
        """Sets the UI components from a settings dictionary."""
        self._suppress_settings_changed = True
        try:
            self.refresh_rate_input.setValue(params.get("refresh_rate"))
            self._set_amplitude_scale(params.get("amplitude_scale", 1.0))
            self.snr_baseline_input.setValue(
                params.get("snr_baseline", DEFAULT_SNR_BASELINE_WINDOW)
                or DEFAULT_SNR_BASELINE_WINDOW
            )
            self.snr_response_input.setValue(
                params.get("snr_response", DEFAULT_SNR_RESPONSE_WINDOW)
                or DEFAULT_SNR_RESPONSE_WINDOW
            )
            self.average_reference_checkbox.setChecked(params.get("reference", "average") == "average")
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
        finally:
            self._suppress_settings_changed = False

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
            "snr_baseline": DEFAULT_SNR_BASELINE_WINDOW,
            "snr_response": DEFAULT_SNR_RESPONSE_WINDOW,
            "art_rem": (-0.005, 0.005),
            "amplitude_scale": 1.0,
            "reference": "average",
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

    def _emit_settings_changed(self, *_):
        if self._suppress_settings_changed:
            return
        self.settingsChanged.emit(self.get_settings())

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
    FILTER_PARAM_KEYS = frozenset({"apply_bandpass", "bandpass_range", "apply_notch", "notch_freqs"})
    EPOCH_PROCESSING_PARAM_KEYS = FILTER_PARAM_KEYS | frozenset({"amplitude_scale", "art_rem", "reference"})
    EPOCH_REPROCESS_BATCH_SIZE = 16

    def __init__(self, stream, params, parent=None):
        super().__init__(parent)
        self.stream = stream
        self.params = dict(params)
        self.is_running = True
        self.max_epochs = int(self.params.get("max_epochs", 200) or 200)
        self.display_epoch_count = int(self.params.get("display_epoch_count", 0) or 0)
        self.raw_picks = np.asarray(mne.pick_types(self.stream.info, eeg=True, exclude=()), dtype=int)
        self.emg_picks = np.asarray(mne.pick_types(self.stream.info, emg=True, exclude=()), dtype=int)
        self.emg_names = [self.stream.info["ch_names"][pick] for pick in self.emg_picks]
        self.raw_event_channels = normalize_event_channels(self.params.get("event_channels"))
        self.raw_event_picks = np.asarray(
            mne.pick_channels(self.stream.info["ch_names"], include=self.raw_event_channels, ordered=True),
            dtype=int,
        )
        used_signal_picks = set(self.raw_picks) | set(self.emg_picks)
        self.raw_monitor_picks = np.asarray(
            list(self.raw_picks)
            + list(self.emg_picks)
            + [pick for pick in self.raw_event_picks if pick not in used_signal_picks],
            dtype=int,
        )

        self.ch_names = list(self.params.get("ch_names", []))
        if not self.ch_names:
            if self.raw_picks.size > 0:
                self.ch_names = [self.stream.info["ch_names"][pick] for pick in self.raw_picks]
            else:
                self.ch_names = list(self.stream.info["ch_names"])
        self.bads = list(self.params.get("bads", []))
        self.epoch_channel_count = len(self.ch_names)

        self.original_sfreq = float(self.stream.info["sfreq"])
        tmin, tmax = self.params.get("tlim", (-0.2, 0.5))
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.epoch_start_offset = int(round(self.tmin * self.original_sfreq))
        self.epoch_end_offset = int(round(self.tmax * self.original_sfreq))
        if self.epoch_end_offset <= self.epoch_start_offset:
            self.epoch_end_offset = self.epoch_start_offset + 1

        self.original_times = (
            np.arange(self.epoch_start_offset, self.epoch_end_offset + 1, dtype=float)
            / self.original_sfreq
        )
        self.decimate = max(1, int(self.params.get("decimate", 5) or 5))
        self.times = self.original_times[:: self.decimate]

        n_times = self.original_times.size
        self.epoch_buffer = np.full((self.max_epochs, self.epoch_channel_count, n_times), np.nan)
        self.processed_epoch_buffer = np.full(
            (self.max_epochs, self.epoch_channel_count, self.times.size),
            np.nan,
        )
        self.emg_epoch_buffer = np.full((self.max_epochs, len(self.emg_picks), n_times), np.nan)
        self.processed_emg_epoch_buffer = np.full(
            (self.max_epochs, len(self.emg_picks), self.times.size),
            np.nan,
        )
        self.buffer_idx = 0
        self.n_valid_epochs = 0
        self.total_epochs_seen = 0

        self.raw_display_seconds = max(0.5, float(self.params.get("stream_duration", 5) or 5.0))
        history_seconds = max(self.raw_display_seconds + 0.25, (self.tmax - self.tmin) + 1.0, 2.0)
        self.max_history_samples = max(8, int(np.ceil(history_seconds * self.original_sfreq)))
        self.raw_display_samples = max(1, int(np.ceil(self.raw_display_seconds * self.original_sfreq)))
        self.raw_history = np.empty((len(self.raw_picks), 0), dtype=float)
        self.emg_history = np.empty((len(self.emg_picks), 0), dtype=float)
        self.stim_history = np.empty((len(self.raw_event_picks), 0), dtype=float)
        self.history_start_sample = 0
        self.total_samples_seen = 0
        self.pending_events: deque[int] = deque()
        self.allowed_event_values = normalize_live_event_values(self.params.get("event_id"))
        self._last_stim_value = 0
        self._force_epoch_emit = False
        self._force_raw_emit = False

        self.bandpass_sos = None
        self.notch_filters: list[tuple[np.ndarray, np.ndarray]] = []
        self._rebuild_filter_pipeline()
        self._update_epoch_good_mask()

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

        epoch_updated = False
        new_samples = int(getattr(self.stream, "n_new_samples", 0) or 0)
        if new_samples > 0:
            try:
                raw_chunk_all, _ = self.stream.get_data(
                    winsize=new_samples / self.original_sfreq,
                    picks=self.raw_monitor_picks,
                    exclude=(),
                )
                if raw_chunk_all is not None and raw_chunk_all.ndim == 2 and raw_chunk_all.shape[1] > 0:
                    eeg_count = len(self.raw_picks)
                    emg_count = len(self.emg_picks)
                    raw_chunk = np.asarray(raw_chunk_all[:eeg_count], dtype=float)
                    emg_chunk = np.asarray(raw_chunk_all[eeg_count : eeg_count + emg_count], dtype=float)
                    stim_chunk = (
                        np.asarray(raw_chunk_all[eeg_count + emg_count :], dtype=float)
                        if raw_chunk_all.shape[0] > eeg_count + emg_count
                        else np.empty((0, raw_chunk_all.shape[1]), dtype=float)
                    )
                    self._append_history(raw_chunk, emg_chunk, stim_chunk)
                    self._queue_pending_events(stim_chunk)
                    epoch_updated = self._consume_pending_events()
            except Exception:
                pass

        emit_dict = self._build_snapshot(
            include_raw=(new_samples > 0 or self._force_raw_emit),
            include_epoch=(epoch_updated or self._force_epoch_emit),
        )
        self._force_raw_emit = False
        self._force_epoch_emit = False
        if emit_dict:
            self.data_ready.emit(emit_dict)

    def _process_epoch_batch(self, epoch_batch: np.ndarray) -> np.ndarray:
        plot_data = np.asarray(epoch_batch, dtype=float) * self._input_amplitude_scale()
        plot_data = apply_baseline_correction(plot_data, self.original_times, baseline=(None, 0.0))
        plot_data = fill_epoch_artifact_window(
            plot_data,
            self.original_times,
            self.params.get("art_rem", (0, 0)),
        )
        if self._frequency_filters_enabled():
            plot_data = self._apply_live_filters(plot_data)
        plot_data = apply_artifact_mask(
            plot_data,
            self.original_times,
            self.params.get("art_rem", (0, 0)),
        )

        if self.params.get("reference", "average") == "average" and plot_data.shape[1] > 1:
            plot_data = average_reference_ignore_nan(plot_data)

        return plot_data

    def _process_epoch_batch_for_display(self, epoch_batch: np.ndarray) -> np.ndarray:
        return self._process_epoch_batch(epoch_batch)[:, :, :: self.decimate]

    def _process_emg_epoch_batch(self, epoch_batch: np.ndarray) -> np.ndarray:
        plot_data = np.asarray(epoch_batch, dtype=float) * self._input_amplitude_scale()
        plot_data = apply_baseline_correction(plot_data, self.original_times, baseline=(None, 0.0))
        plot_data = fill_epoch_artifact_window(
            plot_data,
            self.original_times,
            self.params.get("art_rem", (0, 0)),
        )
        if self._frequency_filters_enabled():
            plot_data = self._apply_live_filters(plot_data)
        plot_data = apply_artifact_mask(
            plot_data,
            self.original_times,
            self.params.get("art_rem", (0, 0)),
        )
        return plot_data

    def _process_emg_epoch_batch_for_display(self, epoch_batch: np.ndarray) -> np.ndarray:
        return self._process_emg_epoch_batch(epoch_batch)[:, :, :: self.decimate]

    @Slot(dict)
    def update_params(self, new_params):
        """Update processing parameters."""
        changed_keys = set(new_params)
        self.params.update(new_params)
        if "ch_names" in new_params:
            self.ch_names = list(new_params["ch_names"])
            self.epoch_channel_count = len(self.ch_names)
        if "bads" in new_params:
            self.bads = list(new_params["bads"])
        if {"ch_names", "bads"} & changed_keys:
            self._update_epoch_good_mask()
        if "display_epoch_count" in new_params:
            self.display_epoch_count = int(new_params["display_epoch_count"] or 0)
        if "event_id" in new_params:
            self.allowed_event_values = normalize_live_event_values(new_params["event_id"])
        if changed_keys & self.FILTER_PARAM_KEYS:
            self._rebuild_filter_pipeline()
        if changed_keys & self.EPOCH_PROCESSING_PARAM_KEYS:
            self._reprocess_epoch_buffer()
        if hasattr(self, "timer"):
            self.timer.setInterval(max(1, int(round(1000 / self.params.get("refresh_rate", 24)))))
        self._force_raw_emit = True
        self._force_epoch_emit = True
        emit_dict = self._build_snapshot(include_raw=True, include_epoch=True)
        if emit_dict:
            self.data_ready.emit(emit_dict)

    @Slot()
    def clear_epochs(self):
        """Reset the circular epoch buffer and counters."""
        self.epoch_buffer.fill(np.nan)
        self.processed_epoch_buffer.fill(np.nan)
        self.emg_epoch_buffer.fill(np.nan)
        self.processed_emg_epoch_buffer.fill(np.nan)
        self.buffer_idx = 0
        self.n_valid_epochs = 0
        self.total_epochs_seen = 0

    def _rebuild_filter_pipeline(self):
        self.bandpass_sos, self.notch_filters = build_live_filter_pipeline(
            self.original_sfreq,
            self.params,
        )

    def _frequency_filters_enabled(self) -> bool:
        return frequency_filters_enabled(self.bandpass_sos, self.notch_filters)

    def _apply_live_filters(self, data: np.ndarray) -> np.ndarray:
        return apply_frequency_filters(data, self.bandpass_sos, self.notch_filters)

    def _input_amplitude_scale(self) -> float:
        return normalize_realtime_amplitude_scale(self.params.get("amplitude_scale", 1.0))

    def _update_epoch_good_mask(self):
        bads = set(self.bads)
        self._epoch_good_mask = np.asarray(
            [channel_name not in bads for channel_name in self.ch_names],
            dtype=bool,
        )

    def _reprocess_epoch_buffer(self):
        self.processed_epoch_buffer.fill(np.nan)
        if self.n_valid_epochs <= 0:
            return

        valid_count = self.max_epochs if self.n_valid_epochs >= self.max_epochs else self.n_valid_epochs
        for start_idx in range(0, valid_count, self.EPOCH_REPROCESS_BATCH_SIZE):
            end_idx = min(valid_count, start_idx + self.EPOCH_REPROCESS_BATCH_SIZE)
            self.processed_epoch_buffer[start_idx:end_idx] = self._process_epoch_batch_for_display(
                self.epoch_buffer[start_idx:end_idx]
            )
            if self.emg_picks.size > 0:
                self.processed_emg_epoch_buffer[start_idx:end_idx] = self._process_emg_epoch_batch_for_display(
                    self.emg_epoch_buffer[start_idx:end_idx]
                )

    def _append_history(self, raw_chunk: np.ndarray, emg_chunk: np.ndarray, stim_chunk: np.ndarray):
        if raw_chunk.ndim != 2 or raw_chunk.shape[1] == 0:
            return

        raw_chunk = np.asarray(raw_chunk, dtype=float)
        emg_chunk = np.asarray(emg_chunk, dtype=float)
        stim_chunk = np.asarray(stim_chunk, dtype=float)
        self.raw_history, dropped_samples = append_limited_history(
            self.raw_history,
            raw_chunk,
            self.max_history_samples,
        )

        if self.emg_picks.size > 0:
            self.emg_history, _ = append_limited_history(
                self.emg_history,
                emg_chunk,
                self.max_history_samples,
            )

        if self.raw_event_picks.size > 0:
            if stim_chunk.size == 0:
                stim_chunk = np.zeros((len(self.raw_event_picks), raw_chunk.shape[1]), dtype=float)
            self.stim_history, _ = append_limited_history(
                self.stim_history,
                stim_chunk,
                self.max_history_samples,
            )

        self.total_samples_seen += raw_chunk.shape[1]
        self.history_start_sample += dropped_samples

    def _queue_pending_events(self, stim_chunk: np.ndarray):
        stim_chunk = np.asarray(stim_chunk, dtype=float)
        if stim_chunk.ndim != 2 or stim_chunk.shape[1] == 0:
            return

        event_samples, self._last_stim_value = detect_stim_onsets(
            stim_chunk,
            previous_value=self._last_stim_value,
            allowed_event_values=self.allowed_event_values,
        )
        if event_samples.size == 0:
            return

        chunk_start_sample = self.total_samples_seen - stim_chunk.shape[1]
        for event_sample in event_samples:
            self.pending_events.append(int(chunk_start_sample + event_sample))

    def _consume_pending_events(self) -> bool:
        if self.raw_history.size == 0 and self.emg_history.size == 0:
            return False

        updated = False
        history_sample_count = max(self.raw_history.shape[1], self.emg_history.shape[1])
        history_end_sample = self.history_start_sample + history_sample_count
        while self.pending_events:
            event_sample = self.pending_events[0]
            epoch_start_sample = event_sample + self.epoch_start_offset
            epoch_end_sample = event_sample + self.epoch_end_offset + 1
            if epoch_end_sample > history_end_sample:
                break

            self.pending_events.popleft()
            if epoch_start_sample < self.history_start_sample:
                continue

            history_start_idx = epoch_start_sample - self.history_start_sample
            history_end_idx = epoch_end_sample - self.history_start_sample
            epoch = self.raw_history[:, history_start_idx:history_end_idx]
            if epoch.shape != (self.epoch_channel_count, self.original_times.size):
                continue
            emg_epoch = None
            if self.emg_picks.size > 0 and self.emg_history.size > 0:
                candidate = self.emg_history[:, history_start_idx:history_end_idx]
                if candidate.shape == (len(self.emg_picks), self.original_times.size):
                    emg_epoch = candidate
            self._store_epoch_batch(epoch[np.newaxis, :, :], None if emg_epoch is None else emg_epoch[np.newaxis, :, :])
            updated = True
        return updated

    def _store_epoch_batch(self, epoch_batch: np.ndarray, emg_epoch_batch: np.ndarray | None = None):
        n_new = int(epoch_batch.shape[0])
        if n_new <= 0:
            return
        self.total_epochs_seen += n_new

        if n_new >= self.max_epochs:
            epoch_batch = epoch_batch[-self.max_epochs :]
            processed_batch = self._process_epoch_batch_for_display(epoch_batch)
            self.epoch_buffer[:] = epoch_batch
            self.processed_epoch_buffer[:] = processed_batch
            if emg_epoch_batch is not None and self.emg_picks.size > 0:
                emg_epoch_batch = emg_epoch_batch[-self.max_epochs :]
                self.emg_epoch_buffer[:] = emg_epoch_batch
                self.processed_emg_epoch_buffer[:] = self._process_emg_epoch_batch_for_display(emg_epoch_batch)
            self.buffer_idx = 0
            self.n_valid_epochs = self.max_epochs
            return

        processed_batch = self._process_epoch_batch_for_display(epoch_batch)
        processed_emg_batch = None
        if emg_epoch_batch is not None and self.emg_picks.size > 0:
            processed_emg_batch = self._process_emg_epoch_batch_for_display(emg_epoch_batch)
        start_idx = self.buffer_idx
        end_idx = start_idx + n_new
        if end_idx <= self.max_epochs:
            self.epoch_buffer[start_idx:end_idx] = epoch_batch
            self.processed_epoch_buffer[start_idx:end_idx] = processed_batch
            if emg_epoch_batch is not None and processed_emg_batch is not None:
                self.emg_epoch_buffer[start_idx:end_idx] = emg_epoch_batch
                self.processed_emg_epoch_buffer[start_idx:end_idx] = processed_emg_batch
        else:
            part1_n = self.max_epochs - start_idx
            self.epoch_buffer[start_idx:] = epoch_batch[:part1_n]
            self.epoch_buffer[: n_new - part1_n] = epoch_batch[part1_n:]
            self.processed_epoch_buffer[start_idx:] = processed_batch[:part1_n]
            self.processed_epoch_buffer[: n_new - part1_n] = processed_batch[part1_n:]
            if emg_epoch_batch is not None and processed_emg_batch is not None:
                self.emg_epoch_buffer[start_idx:] = emg_epoch_batch[:part1_n]
                self.emg_epoch_buffer[: n_new - part1_n] = emg_epoch_batch[part1_n:]
                self.processed_emg_epoch_buffer[start_idx:] = processed_emg_batch[:part1_n]
                self.processed_emg_epoch_buffer[: n_new - part1_n] = processed_emg_batch[part1_n:]

        self.buffer_idx = end_idx % self.max_epochs
        self.n_valid_epochs = min(self.max_epochs, self.n_valid_epochs + n_new)

    def _build_snapshot(self, *, include_raw: bool, include_epoch: bool) -> dict:
        emit_dict: dict[str, dict] = {}

        if include_raw and self.raw_history.size > 0:
            sample_count = min(self.raw_display_samples, self.raw_history.shape[1])
            raw_chunk = self.raw_history[:, -sample_count:] * self._input_amplitude_scale()
            stim_chunk = (
                self.stim_history[:, -sample_count:]
                if self.stim_history.size > 0
                else np.empty((0, sample_count), dtype=float)
            )
            artifact_segments = raw_artifact_segments(
                stim_chunk,
                self.original_sfreq,
                self.params.get("art_rem", (0, 0)),
            )
            if artifact_segments:
                raw_chunk = _interpolate_segments_2d(raw_chunk, artifact_segments)
            if self._frequency_filters_enabled():
                raw_chunk = self._apply_live_filters(raw_chunk)
            if artifact_segments:
                raw_chunk = _mask_segments_2d(raw_chunk, artifact_segments)
            time_axis, scaled_chunk = self._scale_raw_chunk(raw_chunk)
            emit_dict["raw_data"] = {
                "time_axis": time_axis,
                "scaled_data": scaled_chunk,
            }

        if include_epoch:
            plot_data = buffer_epoch_snapshot(
                self.processed_epoch_buffer,
                self.buffer_idx,
                self.n_valid_epochs,
                self.display_epoch_count,
            )
            displayed_epochs = plot_data.shape[0]

            if displayed_epochs > 0:
                mean_data, std_data = compute_epoch_mean_std(plot_data)
                good_mask = self._epoch_good_mask
                if good_mask.shape[0] != mean_data.shape[0]:
                    bads = set(self.bads)
                    good_mask = np.ones(mean_data.shape[0], dtype=bool)
                    for index, channel_name in enumerate(self.ch_names[: mean_data.shape[0]]):
                        good_mask[index] = channel_name not in bads
                valid_data = mean_data[good_mask] if np.any(good_mask) else np.empty((0, mean_data.shape[1]))
                global_min, global_max = -10.0, 10.0
                if valid_data.size > 0 and np.any(np.isfinite(valid_data)):
                    global_min = np.nanmin(valid_data) * 1e6
                    global_max = np.nanmax(valid_data) * 1e6
                if global_min >= global_max:
                    global_min -= 1.0
                    global_max += 1.0
            else:
                mean_data = np.full((self.epoch_buffer.shape[1], len(self.times)), np.nan)
                std_data = np.full_like(mean_data, np.nan)
                global_min, global_max = -10.0, 10.0

            emit_dict["epoch_data"] = {
                "mean_data": mean_data,
                "std_data": std_data,
                "n_epochs": displayed_epochs,
                "buffered_epochs": self.n_valid_epochs,
                "total_epochs": self.total_epochs_seen,
                "global_min": global_min,
                "global_max": global_max,
            }

            if self.emg_picks.size > 0:
                emg_plot_data = buffer_epoch_snapshot(
                    self.processed_emg_epoch_buffer,
                    self.buffer_idx,
                    self.n_valid_epochs,
                    self.display_epoch_count,
                )
                if emg_plot_data.shape[0] > 0:
                    emg_mean_data, emg_std_data = compute_epoch_mean_std(emg_plot_data)
                    mep_epochs = emg_plot_data.shape[0]
                else:
                    emg_mean_data = np.full((len(self.emg_names), len(self.times)), np.nan)
                    emg_std_data = np.full_like(emg_mean_data, np.nan)
                    mep_epochs = 0

                emit_dict["mep_data"] = {
                    "mean_data": emg_mean_data,
                    "std_data": emg_std_data,
                    "emg_names": self.emg_names,
                    "n_epochs": mep_epochs,
                }

        return emit_dict

    def _scale_raw_chunk(self, raw_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_samples = raw_chunk.shape[1]
        time_axis = (np.arange(n_samples, dtype=float) - (n_samples - 1)) / self.original_sfreq

        finite_mask = np.isfinite(raw_chunk)
        counts = np.sum(finite_mask, axis=1, keepdims=True)
        sums = np.where(finite_mask, raw_chunk, 0.0).sum(axis=1, keepdims=True)
        means = np.divide(
            sums,
            counts,
            out=np.zeros((raw_chunk.shape[0], 1), dtype=float),
            where=counts > 0,
        )
        centered_chunk = raw_chunk - means

        scale_mode = self.params.get("scale_mode", "Global Auto-Scale")
        if scale_mode == "Global Auto-Scale":
            global_std = float(np.nanstd(centered_chunk))
            scale = 1.0 / (global_std * 6.0) if np.isfinite(global_std) and global_std > 0 else 1.0
            scaled_chunk = centered_chunk * scale
        else:
            local_stds = np.nanstd(centered_chunk, axis=1, keepdims=True)
            local_stds[np.isnan(local_stds) | (local_stds <= 0)] = 1.0
            scaled_chunk = centered_chunk * (1.0 / (local_stds * 6.0))

        return time_axis, scaled_chunk

class SingleChannelPlot(QDialog):
    """A dialog for plotting data from a single EEG channel using Matplotlib."""
    closed = Signal()
    DISPLAY_POINTS_PER_PIXEL = 1.5
    DISPLAY_MIN_POINTS = 320
    DISPLAY_MAX_POINTS = 6000

    def __init__(self, ch_name, times, parent):
        super().__init__(parent)
        self.ch_name = ch_name
        self.times = times * 1e3 # Convert to ms
        self.parent = parent
        self.latest_evoked_uV = np.full_like(self.times, np.nan, dtype=float)
        self.latest_std_uV = np.full_like(self.times, np.nan, dtype=float)
        self.latest_n_epochs = 0
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
        position_text = self.parent.channel_position_text(self.ch_name)
        self.position_label = QLabel(position_text)
        self.position_label.setObjectName("mutedLabel")
        self.position_label.setWordWrap(True)
        self.position_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.position_label.setVisible(bool(position_text))
        layout.addWidget(self.position_label)
        self.cursor_label = QLabel()
        self.cursor_label.setObjectName("mutedLabel")
        self.cursor_label.setWordWrap(True)
        self.cursor_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._set_cursor_label(message="hover over the plot")
        layout.addWidget(self.cursor_label)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        self.mean_curve, = self.ax.plot([], [], linewidth=2.0, label="Mean")
        self.upper_bound_curve, = self.ax.plot([], [], linewidth=0, alpha=0)
        self.lower_bound_curve, = self.ax.plot([], [], linewidth=0, alpha=0)
        self.std_fill = None
        self.v_line = self.ax.axvline(0.0, linestyle="--", linewidth=1.0)

        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Potential (uV)")
        self.ax.set_xlim(float(self.times[0]), float(self.times[-1]))
        self.ax.grid(True)
        self.legend = self.ax.legend(loc="upper right", frameon=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_moved)
        self.refresh_theme()

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

    @staticmethod
    def _format_cursor_value(value: float, *, decimals: int) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{decimals}f}"

    def _set_cursor_label(
        self,
        *,
        time_ms: float | None = None,
        amp_uV: float = np.nan,
        std_uV: float = np.nan,
        message: str | None = None,
    ):
        if message is not None:
            self.cursor_label.setText(f"<b>Cursor</b>: {message}")
            return

        time_text = "n/a" if time_ms is None or not np.isfinite(time_ms) else f"{time_ms:.1f} ms"
        amp_text = self._format_cursor_value(amp_uV, decimals=2)
        std_text = self._format_cursor_value(std_uV, decimals=2)
        self.cursor_label.setText(
            f"<b>Time</b>: {time_text} | "
            f"<b>Amplitude</b>: {amp_text} uV | "
            f"<b>SD</b>: {std_text} uV"
        )

    def refresh_theme(self):
        tokens = theme_tokens()
        self.figure.patch.set_facecolor(tokens["panel"])
        self.ax.set_facecolor(tokens["plot_background"])
        self.ax.tick_params(colors=tokens["text"])
        self.ax.xaxis.label.set_color(tokens["text"])
        self.ax.yaxis.label.set_color(tokens["text"])
        self.ax.title.set_color(tokens["text"])
        for spine in self.ax.spines.values():
            spine.set_color(tokens["border"])
        self.ax.grid(True, color=tokens["grid"], alpha=0.28, linewidth=0.7)

        self.mean_curve.set_color(tokens["accent_soft"])
        self.v_line.set_color(tokens["accent"])
        self.v_line.set_linestyle("--")

        if self.std_fill is not None:
            self.std_fill.set_facecolor(mcolors.to_rgba(tokens["accent_fill"], alpha=0.34))
            self.std_fill.set_edgecolor("none")

        if self.legend is not None:
            frame = self.legend.get_frame()
            frame.set_facecolor(tokens["panel"])
            frame.set_edgecolor(tokens["border"])
            for text in self.legend.get_texts():
                text.set_color(tokens["text"])

        self.canvas.draw_idle()

    def update_plot(self, evoked_uV, std_uV, n_epochs):
        """Updates the plot with new data."""
        self.latest_evoked_uV = np.array(evoked_uV, copy=True)
        self.latest_std_uV = np.array(std_uV, copy=True)
        self.latest_n_epochs = int(n_epochs)
        upper = evoked_uV + std_uV
        lower = evoked_uV - std_uV
        indices = self._display_indices(self.times.size)
        display_times = self.times[indices]
        display_evoked = np.asarray(evoked_uV)[indices]
        display_upper = np.asarray(upper)[indices]
        display_lower = np.asarray(lower)[indices]
        self.mean_curve.set_data(display_times, display_evoked)
        self.upper_bound_curve.set_data(display_times, display_upper)
        self.lower_bound_curve.set_data(display_times, display_lower)

        if self.std_fill is not None:
            self.std_fill.remove()
        self.std_fill = self.ax.fill_between(
            display_times,
            display_lower,
            display_upper,
            color=theme_tokens()["accent_fill"],
            alpha=0.34,
            linewidth=0,
        )

        finite_envelope = np.concatenate(
            [
                np.asarray(upper)[np.isfinite(upper)],
                np.asarray(lower)[np.isfinite(lower)],
            ]
        )
        if finite_envelope.size > 0:
            y_min = float(np.min(finite_envelope))
            y_max = float(np.max(finite_envelope))
            padding = max(1.0, (y_max - y_min) * 0.12)
            if y_min >= y_max:
                y_min -= padding
                y_max += padding
            else:
                y_min -= padding
                y_max += padding
            self.ax.set_ylim(y_min, y_max)
        else:
            self.ax.set_ylim(-10.0, 10.0)

        if n_epochs <= 0:
            self.ax.set_title(f"{self.ch_name} (no epochs)")
            self._set_cursor_label(message="no data")
            self.refresh_theme()
            self.canvas.draw_idle()
            return

        t0_idx = int(np.argmin(abs(self.times)))
        rms_std = np.nanmean(std_uV[t0_idx:])
        if np.isnan(rms_std):
            self.ax.set_title(f"{self.ch_name} (n={n_epochs})")
        else:
            self.ax.set_title(
                rf"{self.ch_name} (n={n_epochs} | $R_{{std}}$={rms_std:.1f} uV)"
            )
        self.refresh_theme()
        self.canvas.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.latest_evoked_uV.size == 0 or self.latest_std_uV.size == 0:
            return
        self.update_plot(self.latest_evoked_uV, self.latest_std_uV, self.latest_n_epochs)

    def _on_mouse_moved(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.times.size == 0:
            return

        idx = int(np.clip(np.searchsorted(self.times, event.xdata), 0, len(self.times) - 1))
        if idx > 0 and abs(self.times[idx - 1] - event.xdata) <= abs(self.times[idx] - event.xdata):
            idx -= 1

        time_ms = float(self.times[idx])
        amp_uV = float(self.latest_evoked_uV[idx]) if idx < len(self.latest_evoked_uV) else np.nan
        std_uV = float(self.latest_std_uV[idx]) if idx < len(self.latest_std_uV) else np.nan
        self.v_line.set_xdata([time_ms, time_ms])
        self._set_cursor_label(time_ms=time_ms, amp_uV=amp_uV, std_uV=std_uV)
        self.canvas.draw_idle()

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
        self.resize(760, 620)
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
        self.canvas.draw_idle()

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

class RealTimeERP(QMainWindow):
    params_changed = Signal(dict)
    clear_epochs_requested = Signal()
    closed = Signal()
    """Main window for real-time ERP visualization."""

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.stream = None
        self.epochs = None
        self.info = None
        self.sfreq = None
        self.mean_data = None
        self.std_data = None
        self.n_epochs = 0
        self.buffered_epochs = 0
        self.total_epochs = 0
        self.ch_names = []
        self.emg_names = []
        self.channel_indices = {}
        self.coords_3d = np.empty((0, 3), dtype=float)
        self.coords_2d = np.empty((0, 2), dtype=float)
        self.channel_coords = {}
        self.colors = []
        self.times = np.array([], dtype=float)
        self.opened_single_channels = {}
        self.bads = list(self.params.get("bads", []))
        self.max_buffered_epochs = int(self.params.get("max_epochs", 200) or 200)
        self.params["max_epochs"] = self.max_buffered_epochs
        self.params["display_epoch_count"] = int(self.params.get("display_epoch_count", 0) or 0)
        self.conn_thread = None
        self.conn_worker = None
        self.data_thread = None
        self.data_worker = None
        self.progress_dialog = None
        self.topomap_dialog = None
        self.pending_raw_payload = None
        self.latest_raw_payload = None
        self.pending_epoch_payload = None
        self.latest_epoch_payload = None
        self.pending_mep_payload = None
        self.latest_mep_payload = None
        self.raw_render_timer = QTimer(self)
        self.raw_render_timer.setSingleShot(True)
        self.raw_render_timer.setInterval(0)
        self.raw_render_timer.timeout.connect(self._flush_raw_render_cycle)
        self.epoch_render_timer = QTimer(self)
        self.epoch_render_timer.setSingleShot(True)
        self.epoch_render_timer.setInterval(0)
        self.epoch_render_timer.timeout.connect(self._flush_epoch_render_cycle)
        self.topomap_refresh_timer = QTimer(self)
        self.topomap_refresh_timer.setSingleShot(True)
        self.topomap_refresh_timer.setInterval(140)
        self.topomap_refresh_timer.timeout.connect(self._flush_topomap_update)
        self.active_montage_name = None
        self.has_visual_montage = False
        self.theme_name = current_theme_name()
        self.tokens = theme_tokens(self.theme_name)
        self._shutting_down = False

        self.setup_ui()
        self.setWindowTitle("Real-Time TEP Visualization")
        self.resize(1480, 940)

    def setup_ui(self):
        self.setObjectName("RealTimeERPWindow")
        self.setAnimated(False)
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.GroupedDragging
        )

        self.raw_dock = RawMonitorDock(self)
        self.topo_dock = ChannelLayoutDock(self)
        self.evoked_dock = EvokedButterflyDock(self)
        self.mep_dock = MEPDock(self)
        self.top_splitter = self.raw_dock
        self.raw_widget = self.raw_dock
        self.topo_widget = self.topo_dock
        self.plot_docks = {
            "raw": self.raw_dock,
            "topo": self.topo_dock,
            "evoked": self.evoked_dock,
            "mep": self.mep_dock,
        }
        self.topo_dock.channel_left_clicked.connect(self.on_topo_pick)
        self.topo_dock.channel_right_clicked.connect(self._toggle_bad_channel)
        self.evoked_dock.roi_changed.connect(self.on_roi_changed)
        self.evoked_dock.channel_left_clicked.connect(self.on_topo_pick)
        self.evoked_dock.channel_right_clicked.connect(self._toggle_bad_channel)
        for dock_name, dock in self.plot_docks.items():
            dock.visibilityChanged.connect(
                lambda visible, name=dock_name: self._on_plot_dock_visibility_changed(name, visible)
            )

        self.toolbar = self._create_toolbar()
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.controls_toolbar = self._create_controls_toolbar()
        self.addToolBar(Qt.TopToolBarArea, self.controls_toolbar)
        self._update_montage_dependent_ui()
        self._update_mep_dependent_ui()
        self._update_topbar_minimum_width()
        self._restore_default_dock_layout()
        self.refresh_theme()

    def refresh_theme(self):
        self.theme_name = current_theme_name()
        self.tokens = theme_tokens(self.theme_name)

        if self.ch_names:
            self.colors = build_trace_colors(
                len(self.ch_names),
                self.coords_3d,
                prefer_position_colors=self._has_visual_montage(),
            )

        self.raw_dock.refresh_theme(self.tokens, self.colors)
        self.evoked_dock.refresh_theme(self.tokens, self.colors)
        self.mep_dock.refresh_theme(self.tokens)
        self.topo_dock.refresh_theme(self.tokens, self.colors, self.bads)

        if self.topomap_dialog is not None:
            self.topomap_dialog.refresh_theme()
        for dialog in self.opened_single_channels.values():
            dialog.refresh_theme()

    def _has_visual_montage(self) -> bool:
        return bool(self.has_visual_montage)

    def _update_montage_dependent_ui(self):
        has_montage = self._has_visual_montage()
        if has_montage:
            if self.dockWidgetArea(self.topo_dock) == Qt.DockWidgetArea.NoDockWidgetArea:
                self.addDockWidget(Qt.LeftDockWidgetArea, self.topo_dock)
                self.splitDockWidget(
                    self.raw_dock,
                    self.topo_dock,
                    Qt.Orientation.Horizontal,
                )
                self.resizeDocks(
                    [self.raw_dock, self.topo_dock],
                    [640, 640],
                    Qt.Orientation.Horizontal,
                )
            self.topo_dock.show()
        else:
            self.topo_dock.hide()
        if hasattr(self, "topo_toggle_action"):
            self.topo_toggle_action.setEnabled(has_montage)
            self.topo_toggle_action.setVisible(has_montage)
        if not has_montage and self.topomap_dialog is not None:
            self.topomap_dialog.close()
        self._schedule_topbar_minimum_width_update()

    def _has_mep_channels(self) -> bool:
        return bool(self.emg_names)

    def _update_mep_dependent_ui(self):
        has_mep = self._has_mep_channels()
        if has_mep:
            if self.dockWidgetArea(self.mep_dock) == Qt.DockWidgetArea.NoDockWidgetArea:
                self.addDockWidget(Qt.BottomDockWidgetArea, self.mep_dock)
                self.splitDockWidget(
                    self.evoked_dock,
                    self.mep_dock,
                    Qt.Orientation.Horizontal,
                )
                self.resizeDocks(
                    [self.evoked_dock, self.mep_dock],
                    [760, 420],
                    Qt.Orientation.Horizontal,
                )
            self.mep_dock.show()
        else:
            self.mep_dock.hide()
        if hasattr(self, "mep_toggle_action"):
            self.mep_toggle_action.setEnabled(has_mep)
            self.mep_toggle_action.setVisible(has_mep)
        for action in getattr(self, "mep_toolbar_actions", ()):
            action.setVisible(has_mep)
        for attr_name in ("mep_active_label", "mep_active_combo", "mep_reference_label", "mep_reference_combo"):
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                widget.setEnabled(has_mep)
                widget.setVisible(has_mep)
        self._schedule_topbar_minimum_width_update()

    def _schedule_topbar_minimum_width_update(self):
        if hasattr(self, "toolbar") and hasattr(self, "controls_toolbar"):
            QTimer.singleShot(0, self._update_topbar_minimum_width)

    def _update_topbar_minimum_width(self):
        toolbars = [
            toolbar
            for toolbar in (
                getattr(self, "toolbar", None),
                getattr(self, "controls_toolbar", None),
            )
            if toolbar is not None
        ]
        if not toolbars:
            return
        for toolbar in toolbars:
            toolbar.updateGeometry()
        required_width = max(toolbar.sizeHint().width() for toolbar in toolbars) + 24
        self.setMinimumWidth(max(900, required_width))

    def _restore_default_dock_layout(self):
        for dock in (self.raw_dock, self.evoked_dock):
            dock.show()
            if dock.isFloating():
                dock.setFloating(False)
        if self.topo_dock.isFloating():
            self.topo_dock.setFloating(False)
        if self.mep_dock.isFloating():
            self.mep_dock.setFloating(False)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.raw_dock)
        if self._has_visual_montage():
            self.topo_dock.show()
            self.addDockWidget(Qt.LeftDockWidgetArea, self.topo_dock)
            self.splitDockWidget(self.raw_dock, self.topo_dock, Qt.Orientation.Horizontal)
        else:
            self.topo_dock.hide()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.evoked_dock)
        if self._has_mep_channels():
            self.mep_dock.show()
            self.addDockWidget(Qt.BottomDockWidgetArea, self.mep_dock)
            self.splitDockWidget(self.evoked_dock, self.mep_dock, Qt.Orientation.Horizontal)
        else:
            self.mep_dock.hide()
        if self._has_visual_montage():
            self.resizeDocks([self.raw_dock, self.topo_dock], [640, 640], Qt.Orientation.Horizontal)
        if self._has_mep_channels():
            self.resizeDocks([self.evoked_dock, self.mep_dock], [760, 420], Qt.Orientation.Horizontal)
        self.resizeDocks([self.raw_dock, self.evoked_dock], [620, 320], Qt.Orientation.Vertical)
        self._update_montage_dependent_ui()
        self._update_mep_dependent_ui()
        self._render_latest_visible_docks(force=True)

    def _on_plot_dock_visibility_changed(self, dock_name: str, visible: bool):
        if not visible:
            return
        if dock_name == "raw":
            self._render_raw_payload(self.latest_raw_payload, force=True)
            return
        if dock_name == "topo" and not self._has_visual_montage():
            return
        if dock_name == "mep":
            self._render_mep_payload(self.latest_mep_payload, force=True)
            return
        self._render_epoch_payload(self.latest_epoch_payload, force=True)

    def _render_latest_visible_docks(self, *, force: bool):
        self._render_epoch_payload(self.latest_epoch_payload, force=force)
        self._render_mep_payload(self.latest_mep_payload, force=force)
        self._render_raw_payload(self.latest_raw_payload, force=force)

    def _create_toolbar(self):
        toolbar = QToolBar(self)
        toolbar.setObjectName("RealTimeToolBar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

        self.raw_toggle_action = self.raw_dock.toggleViewAction()
        self.raw_toggle_action.setText("Raw")
        toolbar.addAction(self.raw_toggle_action)

        self.topo_toggle_action = self.topo_dock.toggleViewAction()
        self.topo_toggle_action.setText("Channel Layout")
        toolbar.addAction(self.topo_toggle_action)

        self.evoked_toggle_action = self.evoked_dock.toggleViewAction()
        self.evoked_toggle_action.setText("Butterfly")
        toolbar.addAction(self.evoked_toggle_action)

        self.mep_toggle_action = self.mep_dock.toggleViewAction()
        self.mep_toggle_action.setText("MEP")
        toolbar.addAction(self.mep_toggle_action)

        toolbar.addSeparator()
        toolbar.addAction("Reset Layout").triggered.connect(self._restore_default_dock_layout)

        toolbar.addSeparator()

        self.clear_epochs_action = toolbar.addAction("Clear Epochs")
        self.clear_epochs_action.triggered.connect(self.clear_epochs)

        self.epoch_count_label = QLabel("Epochs: 0")
        self.epoch_count_label.setObjectName("mutedLabel")
        toolbar.addWidget(self.epoch_count_label)

        toolbar.addSeparator()
        toolbar.addAction("Settings").triggered.connect(self.update_settings)
        return toolbar

    def _create_controls_toolbar(self):
        toolbar = QToolBar(self)
        toolbar.setObjectName("RealTimeControlsToolBar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(["Global Auto-Scale", "Local Auto-Scale"])
        self.scale_mode_combo.currentTextChanged.connect(self._on_scale_mode_changed)
        toolbar.addWidget(QLabel(" Scale: "))
        toolbar.addWidget(self.scale_mode_combo)

        self.n_chan_spinbox = QSpinBox()
        self.n_chan_spinbox.setRange(1, 256)
        self.n_chan_spinbox.setValue(32)
        self.n_chan_spinbox.valueChanged.connect(self._update_n_channels_shown)
        toolbar.addWidget(QLabel(" View Chans: "))
        toolbar.addWidget(self.n_chan_spinbox)

        toolbar.addSeparator()
        self.display_epoch_spinbox = QSpinBox()
        self.display_epoch_spinbox.setRange(0, self.max_buffered_epochs)
        self.display_epoch_spinbox.setSpecialValueText("All")
        self.display_epoch_spinbox.setToolTip("0 shows all buffered epochs; any positive value shows the last N epochs.")
        self.display_epoch_spinbox.setValue(self.params["display_epoch_count"])
        self.display_epoch_spinbox.valueChanged.connect(self._on_display_epoch_count_changed)
        toolbar.addWidget(QLabel(" View Epochs: "))
        toolbar.addWidget(self.display_epoch_spinbox)

        self.mep_toolbar_actions = []
        self.mep_toolbar_actions.append(toolbar.addSeparator())
        self.mep_active_label = QLabel(" MEP Active: ")
        self.mep_active_combo = QComboBox()
        self.mep_active_combo.setMinimumWidth(120)
        self.mep_active_combo.currentTextChanged.connect(self._on_mep_channel_changed)
        self.mep_reference_label = QLabel(" Ref: ")
        self.mep_reference_combo = QComboBox()
        self.mep_reference_combo.setMinimumWidth(120)
        self.mep_reference_combo.currentTextChanged.connect(self._on_mep_channel_changed)
        self.mep_toolbar_actions.extend(
            (
                toolbar.addWidget(self.mep_active_label),
                toolbar.addWidget(self.mep_active_combo),
                toolbar.addWidget(self.mep_reference_label),
                toolbar.addWidget(self.mep_reference_combo),
            )
        )
        return toolbar

    def _update_epoch_count_label(self):
        self.epoch_count_label.setText(f"Epochs: {self.n_epochs}")

    def _on_display_epoch_count_changed(self, value: int):
        self.params["display_epoch_count"] = int(value)
        self._update_epoch_count_label()
        if self._qt_object_alive(self.data_worker):
            self.params_changed.emit({"display_epoch_count": int(value)})

    def _create_raw_widget(self):
        group = QGroupBox("Raw Data Monitor")
        group.setObjectName("RawDockWidget")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(2, 8, 2, 2)
        self.raw_figure = Figure()
        self.raw_figure.subplots_adjust(left=0.16, right=0.99, bottom=0.11, top=0.98)
        self.raw_canvas = FigureCanvas(self.raw_figure)
        self.raw_ax = self.raw_figure.add_subplot(111)
        layout.addWidget(self.raw_canvas)
        return group

    def _create_topo_widget(self):
        group = QGroupBox("Channel Layout")
        group.setObjectName("TopoDockWidget")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(2, 8, 2, 2)
        self.topo_figure = Figure()
        self.topo_figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        self.topo_canvas = FigureCanvas(self.topo_figure)
        self.topo_host_ax = self.topo_figure.add_subplot(111)
        self.topo_canvas.mpl_connect("button_press_event", self._handle_topo_canvas_click)
        layout.addWidget(self.topo_canvas)
        return group

    def _create_evoked_widget(self):
        group = QGroupBox("Evoked Potentials")
        group.setObjectName("EvokedDockWidget")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(2, 8, 2, 2)
        self.evoked_figure = Figure()
        self.evoked_figure.subplots_adjust(left=0.1, right=0.99, bottom=0.12, top=0.98)
        self.evoked_canvas = FigureCanvas(self.evoked_figure)
        self.evoked_ax = self.evoked_figure.add_subplot(111)
        self.roi_selector = None
        self.roi_patch = None
        self.roi_region = None
        layout.addWidget(self.evoked_canvas)
        return group

    def _apply_axis_theme(self, ax, *, show_x_grid: bool, show_y_grid: bool):
        ax.set_facecolor(self.tokens["plot_background"])
        ax.tick_params(colors=self.tokens["text"])
        ax.xaxis.label.set_color(self.tokens["text"])
        ax.yaxis.label.set_color(self.tokens["text"])
        ax.title.set_color(self.tokens["text"])
        for spine in ax.spines.values():
            spine.set_color(self.tokens["border"])
        ax.grid(False)
        if show_x_grid:
            ax.grid(True, axis="x", color=self.tokens["grid"], alpha=0.28, linewidth=0.7)
        if show_y_grid:
            ax.grid(True, axis="y", color=self.tokens["grid"], alpha=0.28, linewidth=0.7)

    def _default_roi_region(self) -> tuple[float, float]:
        if len(self.times) == 0:
            return (0.0, 0.0)

        time_axis_ms = self.times * 1e3
        min_x = float(time_axis_ms[0])
        max_x = float(time_axis_ms[-1])
        span = max(10.0, min(30.0, (max_x - min_x) * 0.08))

        if min_x <= 0.0 <= max_x:
            start_ms = 0.0
        else:
            start_ms = min_x
        end_ms = min(max_x, start_ms + span)
        if end_ms <= start_ms:
            end_ms = min(max_x, start_ms + 1.0)
        return (start_ms, end_ms)

    def _update_roi_patch(self):
        if getattr(self, "roi_patch", None) is not None:
            self.roi_patch.remove()
            self.roi_patch = None

        if self.roi_region is None or not hasattr(self, "evoked_ax"):
            return

        start_ms, end_ms = self.roi_region
        if end_ms <= start_ms:
            return

        self.roi_patch = self.evoked_ax.axvspan(
            start_ms,
            end_ms,
            facecolor=self.tokens["roi"],
            alpha=0.22,
            edgecolor="none",
            zorder=0.2,
        )

    def _on_roi_span_selected(self, start_ms: float, end_ms: float):
        if len(self.times) == 0:
            return

        min_x = float(self.times[0] * 1e3)
        max_x = float(self.times[-1] * 1e3)
        start_ms, end_ms = sorted((float(start_ms), float(end_ms)))
        start_ms = float(np.clip(start_ms, min_x, max_x))
        end_ms = float(np.clip(end_ms, min_x, max_x))
        if end_ms <= start_ms:
            return

        self.roi_region = (start_ms, end_ms)
        self._update_roi_patch()
        self.on_roi_changed()
        self.evoked_canvas.draw_idle()

    def _setup_evoked_plot(self):
        if self.roi_selector is not None:
            self.roi_selector.disconnect_events()

        self.evoked_ax.clear()
        time_axis_ms = self.times * 1e3
        self.evoked_lines = [
            self.evoked_ax.plot(
                time_axis_ms,
                np.full_like(time_axis_ms, np.nan, dtype=float),
                linewidth=1.3,
                color=self.colors[i],
            )[0]
            for i in range(len(self.ch_names))
        ]
        self.evoked_ax.set_xlim(float(time_axis_ms[0]), float(time_axis_ms[-1]))
        self.evoked_ax.set_ylim(-10.0, 10.0)
        self.evoked_ax.set_xlabel("Time (ms)")
        self.evoked_ax.set_ylabel("Potential (uV)")
        self.roi_region = self._default_roi_region()
        self._update_roi_patch()
        self.roi_selector = SpanSelector(
            self.evoked_ax,
            self._on_roi_span_selected,
            "horizontal",
            minspan=0.25,
            props={"facecolor": self.tokens["roi"], "alpha": 0.22},
            button=[MouseButton.LEFT],
            interactive=False,
            drag_from_anywhere=False,
            ignore_event_outside=True,
        )

    def _setup_topo_plot(self):
        self.topo_figure.clear()
        self.topo_host_ax = self.topo_figure.add_subplot(111)
        self.topo_host_ax.set_aspect("equal")
        self.topo_host_ax.set_xlim(0.0, 1.0)
        self.topo_host_ax.set_ylim(0.0, 1.0)
        self.topo_host_ax.set_xticks([])
        self.topo_host_ax.set_yticks([])
        for spine in self.topo_host_ax.spines.values():
            spine.set_visible(False)

        self.topo_head_outline = Circle((0.5, 0.5), 0.47, fill=False, linewidth=1.2)
        self.topo_host_ax.add_patch(self.topo_head_outline)

        self.topo_channel_axes = []
        self.topo_plot_lines = []
        self.topo_axis_channel_map = {}

        positions = calculate_mne_style_layout(self.coords_2d)
        time_axis_ms = self.times * 1e3
        for i, ch_name in enumerate(self.ch_names):
            if i >= positions.shape[0]:
                break
            left, bottom, width, height = positions[i]
            inset_ax = self.topo_host_ax.inset_axes([left, bottom, width, height])
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_title(ch_name, fontsize=7, pad=1)
            line, = inset_ax.plot(
                time_axis_ms,
                np.full_like(time_axis_ms, np.nan, dtype=float),
                linewidth=1.2,
                color=self.colors[i],
            )
            self.topo_channel_axes.append(inset_ax)
            self.topo_plot_lines.append(line)
            self.topo_axis_channel_map[inset_ax] = ch_name

        self._update_topo_axis_limits(-10.0, 10.0)

    def _update_topo_axis_limits(self, global_min: float, global_max: float):
        if len(self.times) == 0:
            return

        y_padding = max(1.0, (global_max - global_min) * 0.12)
        y_min = global_min - y_padding
        y_max = global_max + y_padding
        time_axis_ms = self.times * 1e3

        for axis in getattr(self, "topo_channel_axes", []):
            axis.set_xlim(float(time_axis_ms[0]), float(time_axis_ms[-1]))
            axis.set_ylim(y_min, y_max)

    @staticmethod
    def _qt_object_alive(obj: QObject | None) -> bool:
        return obj is not None and isValid(obj)

    def _clear_qt_attr(self, attr_name: str):
        setattr(self, attr_name, None)

    def _stop_qthread(self, attr_name: str, timeout_ms: int = 1500):
        thread = getattr(self, attr_name, None)
        if not self._qt_object_alive(thread):
            setattr(self, attr_name, None)
            return

        try:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(timeout_ms):
                    thread.terminate()
                    thread.wait(timeout_ms)
        except RuntimeError:
            pass

        try:
            still_running = self._qt_object_alive(thread) and thread.isRunning()
        except RuntimeError:
            still_running = False
        if not still_running:
            setattr(self, attr_name, None)

    def start_visualization(self):
        self._shutting_down = False
        self.conn_thread = QThread(self)
        self.conn_worker = ConnectionWorker(self.params)
        self.conn_worker.moveToThread(self.conn_thread)
        self.conn_thread.destroyed.connect(lambda *_: self._clear_qt_attr("conn_thread"))
        self.conn_worker.destroyed.connect(lambda *_: self._clear_qt_attr("conn_worker"))

        self.progress_dialog = QProgressDialog("Connecting to LSL stream...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.destroyed.connect(lambda *_: self._clear_qt_attr("progress_dialog"))

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

    def _on_connection_success(self, stream):
        if self._shutting_down:
            try:
                stream.disconnect()
            except Exception:
                pass
            return
        if self._qt_object_alive(self.progress_dialog):
            self.progress_dialog.accept()
        self.stream = stream

        self._apply_visual_montage()
        self._update_montage_dependent_ui()

        eeg_picks = mne.pick_types(self.stream.info, eeg=True, exclude=())
        self.info = mne.pick_info(self.stream.info, eeg_picks)
        self.ch_names = self.info["ch_names"]
        self.channel_indices = {name: index for index, name in enumerate(self.ch_names)}
        self.info["bads"] = [name for name in self.bads if name in self.ch_names]
        self.bads = list(self.info["bads"])
        self.sfreq = float(self.info["sfreq"])
        coords_3d = []
        for ch in self.info["chs"]:
            loc = ch.get("loc", np.zeros(12))
            if loc is not None and len(loc) >= 3 and not np.all(loc[:3] == 0):
                coords_3d.append(loc[:3])
            else:
                coords_3d.append([0.0, 0.0, 0.0])
        coords_3d = np.array(coords_3d)
        self.coords_3d = coords_3d
        self.coords_2d = self.project_electrodes_to_2d(coords_3d)
        self.channel_coords = {
            channel_name: tuple(float(value) for value in self.coords_2d[index])
            for index, channel_name in enumerate(self.ch_names)
        }
        self.colors = build_trace_colors(
            len(self.ch_names),
            self.coords_3d,
            prefer_position_colors=self._has_visual_montage(),
        )

        self.n_chan_spinbox.setRange(1, max(1, len(self.ch_names)))
        self.n_chan_spinbox.setValue(len(self.ch_names))

        self.params["ch_names"] = self.ch_names
        self.params["bads"] = self.info["bads"]
        self.params["scale_mode"] = self.scale_mode_combo.currentText()
        self.params["display_epoch_count"] = self.display_epoch_spinbox.value()
        self._update_epoch_count_label()

        # Start data processing worker thread
        self.data_thread = QThread(self)
        self.data_worker = DataProcessingWorker(self.stream, self.params)
        self.data_worker.moveToThread(self.data_thread)
        self.data_thread.destroyed.connect(lambda *_: self._clear_qt_attr("data_thread"))
        self.data_worker.destroyed.connect(lambda *_: self._clear_qt_attr("data_worker"))

        self.times = np.asarray(self.data_worker.times, dtype=float)
        self.emg_names = list(self.data_worker.emg_names)
        self.params["emg_names"] = self.emg_names

        self.setup_plots()
        self._configure_mep_controls()

        # Connections for data worker
        self.data_worker.data_ready.connect(self._on_data_ready)
        self.data_worker.finished.connect(self.data_thread.quit)
        self.data_thread.started.connect(self.data_worker.run)
        self.data_thread.finished.connect(self.data_worker.deleteLater)
        self.data_thread.finished.connect(self.data_thread.deleteLater)
        self.params_changed.connect(self.data_worker.update_params)
        self.clear_epochs_requested.connect(self.data_worker.clear_epochs)
        self.data_thread.start()

    def _on_connection_failure(self, error_message):
        if self._qt_object_alive(self.progress_dialog):
            self.progress_dialog.reject()
        QMessageBox.critical(self, "Connection Failed", error_message)
        self.close()

    def setup_plots(self):
        if not self.ch_names or self.times.size == 0:
            return

        time_axis_ms = self.times * 1e3
        stream_duration = float(self.params.get("stream_duration", 5) or 5.0)

        self.raw_dock.configure(self.ch_names, self.colors, stream_duration)
        self.evoked_dock.configure(time_axis_ms, self.ch_names, self.colors)
        self._apply_snr_windows_to_plot()
        self.mep_dock.configure(time_axis_ms)
        if self._has_visual_montage():
            positions = calculate_mne_style_layout(self.coords_2d)
            self.topo_dock.configure(time_axis_ms, self.ch_names, self.colors, positions)
        self._update_montage_dependent_ui()
        self._update_mep_dependent_ui()
        self._update_n_channels_shown()
        self.refresh_theme()
        self._render_latest_visible_docks(force=True)

    def _setup_raw_plot(self):
        self.raw_ax.clear()
        self.raw_curves = []
        n_chans = len(self.ch_names)

        self.raw_offsets = np.arange(n_chans)[::-1]
        stream_duration = self.params.get("stream_duration", 5)
        self.raw_ax.set_xlim(-stream_duration, 0.0)
        self.raw_ax.set_ylim(-0.5, n_chans - 0.5)
        self.raw_ax.set_xlabel("Time (s)")
        self.raw_ax.set_yticks(self.raw_offsets)
        self.raw_ax.set_yticklabels(self.ch_names)

        for i in range(n_chans):
            curve, = self.raw_ax.plot([], [], linewidth=1.0, color=self.colors[i])
            self.raw_curves.append(curve)

        self._update_n_channels_shown()

    def _update_n_channels_shown(self):
        if not self.ch_names:
            return
        self.raw_dock.set_visible_channels(self.n_chan_spinbox.value())

    def _configure_mep_controls(self):
        for combo in (self.mep_active_combo, self.mep_reference_combo):
            combo.blockSignals(True)
            combo.clear()

        try:
            self.mep_active_combo.addItems(self.emg_names)
            self.mep_reference_combo.addItem("None")
            self.mep_reference_combo.addItems(self.emg_names)

            if self.emg_names:
                self.mep_active_combo.setCurrentText(self.emg_names[0])
            if len(self.emg_names) > 1:
                self.mep_reference_combo.setCurrentText(self.emg_names[1])
            else:
                self.mep_reference_combo.setCurrentText("None")
        finally:
            for combo in (self.mep_active_combo, self.mep_reference_combo):
                combo.blockSignals(False)

        self._update_mep_dependent_ui()
        self._render_mep_payload(self.latest_mep_payload, force=True)

    def _current_mep_selection(self) -> tuple[str | None, str | None]:
        if not self.emg_names:
            return None, None

        active = self.mep_active_combo.currentText()
        if active not in self.emg_names:
            active = self.emg_names[0]

        reference = self.mep_reference_combo.currentText()
        if reference == "None" or reference == active or reference not in self.emg_names:
            reference = None
        return active, reference

    def _on_mep_channel_changed(self, *_):
        self._render_mep_payload(self.latest_mep_payload, force=True)

    def _on_scale_mode_changed(self, text: str):
        if self._qt_object_alive(self.data_worker):
            self.params_changed.emit({"scale_mode": text})
        self.params["scale_mode"] = text
        self._render_epoch_payload(self.latest_epoch_payload, force=True)
        self._render_raw_payload(self.latest_raw_payload, force=True)

    @staticmethod
    def to_rgb(positions_3d):
        xyz = positions_3d.copy()
        if xyz.shape[0] > 1:
            xyz -= xyz.min(axis=0)
            max_vals = xyz.max(axis=0)
            max_vals[max_vals == 0] = 1 # Avoid division by zero
            xyz /= max_vals
        return xyz * 255

    def _flush_epoch_render_cycle(self):
        epoch_payload = self.pending_epoch_payload
        mep_payload = self.pending_mep_payload
        self.pending_epoch_payload = None
        self.pending_mep_payload = None
        if epoch_payload is not None:
            self._render_epoch_payload(epoch_payload)
        if mep_payload is not None:
            self._render_mep_payload(mep_payload)

    def _flush_raw_render_cycle(self):
        raw_payload = self.pending_raw_payload
        self.pending_raw_payload = None
        if raw_payload is not None:
            self._render_raw_payload(raw_payload)

    def _render_epoch_payload(self, payload, *, force: bool = False):
        if payload is None or not self.ch_names:
            return

        mean_data = payload.mean_data if isinstance(payload, EpochRenderPayload) else payload["mean_data"]
        std_data = payload.std_data if isinstance(payload, EpochRenderPayload) else payload["std_data"]
        n_epochs = payload.n_epochs if isinstance(payload, EpochRenderPayload) else int(payload["n_epochs"])
        buffered_epochs = (
            payload.buffered_epochs
            if isinstance(payload, EpochRenderPayload)
            else int(payload.get("buffered_epochs", n_epochs))
        )
        total_epochs = (
            payload.total_epochs
            if isinstance(payload, EpochRenderPayload)
            else int(payload.get("total_epochs", buffered_epochs))
        )
        global_min = payload.global_min if isinstance(payload, EpochRenderPayload) else float(payload["global_min"])
        global_max = payload.global_max if isinstance(payload, EpochRenderPayload) else float(payload["global_max"])

        if mean_data.shape[0] != len(self.ch_names) or std_data.shape[0] != len(self.ch_names):
            return

        self.mean_data = mean_data
        self.std_data = std_data
        self.n_epochs = n_epochs
        self.buffered_epochs = buffered_epochs
        self.total_epochs = total_epochs

        self._update_epoch_count_label()
        self.setWindowTitle(f"Real-Time TEP - Epochs: {self.n_epochs}")

        mean_data_uV = self.mean_data * 1e6
        if force or self.evoked_dock.isVisible():
            self.evoked_dock.update_data(mean_data_uV, self.bads, global_min, global_max)
        if self._has_visual_montage() and (force or self.topo_dock.isVisible()):
            self.topo_dock.update_data(
                mean_data_uV,
                self.bads,
                global_min,
                global_max,
                self.params.get("scale_mode", "Global Auto-Scale"),
            )

        for ch_name, plot_dialog in list(self.opened_single_channels.items()):
            ch_idx = self.channel_indices.get(ch_name)
            if ch_idx is None:
                continue
            plot_dialog.update_plot(self.mean_data[ch_idx, :] * 1e6, self.std_data[ch_idx, :] * 1e6, self.n_epochs)

        if self._has_visual_montage() and self.topomap_dialog is not None:
            self._schedule_topomap_update()

    def _render_mep_payload(self, payload, *, force: bool = False):
        if payload is None or not self._has_mep_channels() or self.times.size == 0:
            return
        if not force and not self.mep_dock.isVisible():
            return

        mean_data = payload.mean_data if isinstance(payload, MEPRenderPayload) else payload["mean_data"]
        emg_names = payload.emg_names if isinstance(payload, MEPRenderPayload) else list(payload["emg_names"])
        n_epochs = payload.n_epochs if isinstance(payload, MEPRenderPayload) else int(payload["n_epochs"])
        if np.ndim(mean_data) != 2 or mean_data.shape[0] != len(emg_names):
            return

        active_channel, reference_channel = self._current_mep_selection()
        if active_channel is None:
            return

        trace_uV = build_mep_trace_uV(
            mean_data,
            emg_names,
            active_channel,
            reference_channel,
        )
        if trace_uV.size != self.times.size:
            return

        times_ms = self.times * 1e3
        p2p_uV, window_mask = compute_mep_peak_to_peak_uV(trace_uV, times_ms)
        self.mep_dock.update_data(
            trace_uV,
            window_mask,
            p2p_uV=p2p_uV,
            threshold_uV=MEP_THRESHOLD_UV,
            active_channel=active_channel,
            reference_channel=reference_channel,
            n_epochs=n_epochs,
        )

    def _render_raw_payload(self, payload, *, force: bool = False):
        if payload is None or not self.ch_names:
            return

        time_axis = payload.time_axis if isinstance(payload, RawRenderPayload) else payload["time_axis"]
        scaled_data = payload.scaled_data if isinstance(payload, RawRenderPayload) else payload["scaled_data"]
        if time_axis is None or scaled_data is None:
            return
        if np.ndim(scaled_data) != 2 or scaled_data.shape[0] != len(self.ch_names):
            return
        if np.size(time_axis) == 0 or scaled_data.shape[1] == 0:
            return

        if force or self.raw_dock.isVisible():
            self.raw_dock.update_data(np.asarray(time_axis), np.asarray(scaled_data))

    def _on_data_ready(self, data_dict):
        if "epoch_data" in data_dict:
            epoch_data = data_dict["epoch_data"]
            self.pending_epoch_payload = EpochRenderPayload(
                mean_data=np.asarray(epoch_data["mean_data"]),
                std_data=np.asarray(epoch_data["std_data"]),
                n_epochs=int(epoch_data["n_epochs"]),
                buffered_epochs=int(epoch_data.get("buffered_epochs", epoch_data["n_epochs"])),
                total_epochs=int(epoch_data.get("total_epochs", epoch_data.get("buffered_epochs", epoch_data["n_epochs"]))),
                global_min=float(epoch_data["global_min"]),
                global_max=float(epoch_data["global_max"]),
            )
            self.latest_epoch_payload = self.pending_epoch_payload
            if not self.epoch_render_timer.isActive():
                self.epoch_render_timer.start()

        if "mep_data" in data_dict:
            mep_data = data_dict["mep_data"]
            self.pending_mep_payload = MEPRenderPayload(
                mean_data=np.asarray(mep_data["mean_data"]),
                std_data=np.asarray(mep_data["std_data"]),
                emg_names=list(mep_data["emg_names"]),
                n_epochs=int(mep_data["n_epochs"]),
            )
            self.latest_mep_payload = self.pending_mep_payload
            if not self.epoch_render_timer.isActive():
                self.epoch_render_timer.start()

        if "raw_data" in data_dict:
            raw_data = data_dict["raw_data"]
            time_axis = raw_data.get("time_axis")
            scaled_data = raw_data.get("scaled_data")
            if time_axis is not None and scaled_data is not None:
                self.pending_raw_payload = RawRenderPayload(
                    time_axis=np.asarray(time_axis),
                    scaled_data=np.asarray(scaled_data),
                )
                self.latest_raw_payload = self.pending_raw_payload
                if not self.raw_render_timer.isActive():
                    self.raw_render_timer.start()

    def clear_epochs(self):
        self.n_epochs = 0
        self.buffered_epochs = 0
        self.total_epochs = 0
        self.topomap_refresh_timer.stop()
        if self.ch_names and self.times.size > 0:
            empty = np.full((len(self.ch_names), len(self.times)), np.nan)
            cleared_payload = EpochRenderPayload(
                mean_data=empty.copy(),
                std_data=empty.copy(),
                n_epochs=0,
                buffered_epochs=0,
                total_epochs=0,
                global_min=-10.0,
                global_max=10.0,
            )
            self.latest_epoch_payload = cleared_payload
            self.pending_epoch_payload = None
            self._render_epoch_payload(cleared_payload, force=True)
            if self.emg_names:
                empty_mep = np.full((len(self.emg_names), len(self.times)), np.nan)
                cleared_mep_payload = MEPRenderPayload(
                    mean_data=empty_mep.copy(),
                    std_data=empty_mep.copy(),
                    emg_names=self.emg_names.copy(),
                    n_epochs=0,
                )
                self.latest_mep_payload = cleared_mep_payload
                self.pending_mep_payload = None
                self._render_mep_payload(cleared_mep_payload, force=True)
        else:
            self.mean_data = None
            self.std_data = None
            self._update_epoch_count_label()
        if self.topomap_dialog is not None:
            self.topomap_dialog.close()
        if self._qt_object_alive(self.data_worker):
            self.clear_epochs_requested.emit()

    def _current_live_plot_settings(self) -> dict:
        return {
            "refresh_rate": int(self.params.get("refresh_rate", 24) or 24),
            "snr_baseline": self.params.get(
                "snr_baseline",
                DEFAULT_SNR_BASELINE_WINDOW,
            ),
            "snr_response": self.params.get(
                "snr_response",
                DEFAULT_SNR_RESPONSE_WINDOW,
            ),
            "art_rem": self.params.get("art_rem", (-0.005, 0.005)),
            "amplitude_scale": normalize_realtime_amplitude_scale(self.params.get("amplitude_scale", 1.0)),
            "reference": "average" if self.params.get("reference", "average") == "average" else "none",
            "apply_bandpass": bool(self.params.get("apply_bandpass", False)),
            "bandpass_range": self.params.get("bandpass_range", (8.0, 80.0)),
            "apply_notch": bool(self.params.get("apply_notch", False)),
            "notch_freqs": list(self.params.get("notch_freqs", [50.0]) or []),
        }

    @staticmethod
    def _snr_window_ms(value, default_window) -> tuple[float, float]:
        window = value or default_window
        try:
            start, end = float(window[0]), float(window[1])
        except (IndexError, TypeError, ValueError):
            start, end = default_window
        if not np.isfinite(start) or not np.isfinite(end) or start == end:
            start, end = default_window
        return tuple(sorted((start * 1e3, end * 1e3)))

    def _apply_snr_windows_to_plot(self):
        self.evoked_dock.set_snr_windows(
            self._snr_window_ms(
                self.params.get("snr_baseline"),
                DEFAULT_SNR_BASELINE_WINDOW,
            ),
            self._snr_window_ms(
                self.params.get("snr_response"),
                DEFAULT_SNR_RESPONSE_WINDOW,
            ),
        )

    @Slot(dict)
    def apply_live_settings(self, new_params: dict):
        if not new_params:
            return
        self.params.update(new_params)
        if {"snr_baseline", "snr_response"} & set(new_params):
            self._apply_snr_windows_to_plot()
        if self._qt_object_alive(self.data_worker):
            self.params_changed.emit(new_params)

    def update_settings(self):
        """Open the settings dialog and apply changes."""
        previous_params = self._current_live_plot_settings()
        dialog = RealTimeSettings(self)
        dialog.settings_widget.set_settings(previous_params)
        dialog.settings_widget.settingsChanged.connect(self.apply_live_settings)
        accepted = bool(dialog.exec())
        try:
            dialog.settings_widget.settingsChanged.disconnect(self.apply_live_settings)
        except (RuntimeError, TypeError):
            pass

        if accepted:
            new_params = dialog.get_settings()
            dialog.settings_widget.save_settings()
            self.apply_live_settings(new_params)
        else:
            self.apply_live_settings(previous_params)

    def on_roi_changed(self, *_):
        """Handle ROI changes on the evoked dock."""
        self._schedule_topomap_update()

    def _schedule_topomap_update(self, immediate: bool = False):
        if not self._has_visual_montage():
            if self.topomap_dialog is not None:
                self.topomap_dialog.close()
            return
        if self.mean_data is None:
            return
        if immediate:
            self.topomap_refresh_timer.stop()
            self._flush_topomap_update()
            return
        if not self.topomap_refresh_timer.isActive():
            self.topomap_refresh_timer.start()

    def _flush_topomap_update(self):
        payload = self._build_topomap_payload()
        if payload is None:
            if self.topomap_dialog is not None:
                self.topomap_dialog.close()
            return

        topo_data, info_good, title = payload
        if self.topomap_dialog is None:
            self.topomap_dialog = TopomapPlot(self)
            self.topomap_dialog.closed.connect(self._on_topomap_closed)

        self.topomap_dialog.update_plot(
            topo_data,
            info_good,
            title,
            names=info_good['ch_names'],
            cmap="RdBu_r",
        )

    def _build_topomap_payload(self):
        roi_region = getattr(self.evoked_dock, "roi_region", None)
        if (
            not self._has_visual_montage()
            or self.mean_data is None
            or roi_region is None
            or self.info is None
            or self.times.size == 0
        ):
            return None
        start_ms, end_ms = roi_region

        time_axis_ms = self.times * 1e3
        start_idx = np.searchsorted(time_axis_ms, start_ms, side="left")
        end_idx = np.searchsorted(time_axis_ms, end_ms, side="right")

        if start_idx >= end_idx:
            return None

        good_indices = np.asarray(mne.pick_types(self.info, eeg=True, exclude="bads"))
        if len(good_indices) == 0:
            return None

        positions = []
        for pick in good_indices:
            loc = self.info["chs"][pick].get("loc", np.zeros(12))
            positions.append(loc[:3] if loc is not None and len(loc) >= 3 else np.zeros(3))
        positions = np.asarray(positions, dtype=float)
        valid_position_mask = (
            np.isfinite(positions).all(axis=1)
            & np.any(np.abs(positions) > 1e-9, axis=1)
        )
        good_indices = good_indices[valid_position_mask]
        if len(good_indices) < 2:
            return None
        
        topo_window = self.mean_data[good_indices, start_idx:end_idx]
        finite_mask = np.isfinite(topo_window)
        counts = np.sum(finite_mask, axis=1)
        sums = np.where(finite_mask, topo_window, 0.0).sum(axis=1)
        topo_data = np.divide(
            sums,
            counts,
            out=np.full(good_indices.shape, np.nan, dtype=float),
            where=counts > 0,
        )
        valid_mask = np.isfinite(topo_data)
        if not np.any(valid_mask):
            return None

        good_indices = good_indices[valid_mask]
        topo_data = topo_data[valid_mask]

        info_good = mne.pick_info(self.info, good_indices)
            
        title = f"Topomap from {start_ms:.1f} to {end_ms:.1f} ms"
        return topo_data, info_good, title

    def _on_topomap_closed(self):
        self.topomap_refresh_timer.stop()
        self.topomap_dialog = None

    def _handle_topo_canvas_click(self, event):
        # Legacy inline-topography handler kept for compatibility with the old plot path.
        return

    def on_topo_pick(self, ch_name):
        """Open a single-channel plot when a layout trace is clicked."""
        if ch_name in self.opened_single_channels:
            self.opened_single_channels[ch_name].raise_()
            self.opened_single_channels[ch_name].activateWindow()
            return

        dialog = SingleChannelPlot(ch_name, self.times, self)
        dialog.closed.connect(lambda ch=ch_name: self.opened_single_channels.pop(ch, None))
        self.opened_single_channels[ch_name] = dialog
        
        if self.mean_data is not None and self.std_data is not None:
            ch_idx = self.channel_indices.get(ch_name)
            if ch_idx is not None:
                evoked_uV = self.mean_data[ch_idx, :] * 1e6
                std_uV = self.std_data[ch_idx, :] * 1e6
                dialog.update_plot(evoked_uV, std_uV, self.n_epochs)

    def _toggle_bad_channel(self, ch_name: str):
        next_bads = [bad for bad in self.bads if bad != ch_name]
        if ch_name not in self.bads:
            next_bads.append(ch_name)
        self._set_bad_channels(next_bads)

    def _set_bad_channels(self, bads: list[str]):
        ordered_bads = [name for name in self.ch_names if name in set(bads)]
        self.bads = ordered_bads
        self.params["bads"] = ordered_bads
        if self.info is not None:
            self.info["bads"] = ordered_bads.copy()
        if self._qt_object_alive(self.data_worker):
            self.params_changed.emit({"bads": ordered_bads})
        self.refresh_theme()
        self._render_epoch_payload(self.latest_epoch_payload, force=True)
        if self._has_visual_montage() and self.topomap_dialog is not None and self.mean_data is not None:
            self._schedule_topomap_update(immediate=True)

    def _shutdown_threads(self):
        self._shutting_down = True
        self.raw_render_timer.stop()
        self.epoch_render_timer.stop()
        self.topomap_refresh_timer.stop()

        if self._qt_object_alive(self.data_worker):
            try:
                self.data_worker.stop()
            except (Exception, RuntimeError):
                pass

        self._stop_qthread("data_thread")

        if self._qt_object_alive(self.conn_worker):
            try:
                self.conn_worker.stop()
            except (Exception, RuntimeError):
                pass

        if self._qt_object_alive(self.progress_dialog):
            try:
                self.progress_dialog.reject()
            except (Exception, RuntimeError):
                pass

        if self.stream is not None:
            try:
                self.stream.disconnect()
            except Exception:
                pass
            self.stream = None

        self._stop_qthread("conn_thread")
        self.epochs = None
        self.pending_raw_payload = None
        self.latest_raw_payload = None
        self.pending_epoch_payload = None
        self.latest_epoch_payload = None

    def closeEvent(self, event):
        self._shutdown_threads()
        if self.topomap_dialog is not None:
            self.topomap_dialog.close()
        for dialog in list(self.opened_single_channels.values()):
            dialog.close()
        self.closed.emit()
        super().closeEvent(event)

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
        if self.stream is None:
            return

        self.has_visual_montage = realtime_info_has_montage(self.stream.info)
        if self.has_visual_montage:
            self.active_montage_name = self.params.get("info_label") or "MNE info montage"
            return

        self.active_montage_name = None

    def channel_position_text(self, ch_name: str) -> str:
        coords = self.channel_coords.get(ch_name)
        if coords is None or not self._has_visual_montage():
            return ""
        if all(abs(value) < 1e-9 for value in coords):
            return ""

        return f"Layout: x={coords[0]:.3f}, y={coords[1]:.3f}"

class RealTimeMainWidget(QWidget):
    """The main entry point widget for the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rt_erp_widget = None
        self.setup_ui()
        self.setWindowTitle("LSL Stream Visualizer")

    def setup_ui(self):
        """Sets up the UI for the main widget."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.setMinimumWidth(0)

        self.main_container = QWidget()
        self.main_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self.main_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.connection_widget = ConnectionWidget()
        self.plot_settings_widget = RealTimeSettingsWidget()
        self.plot_settings_widget.settingsChanged.connect(self._forward_plot_settings)
        # self.plot_settings_widget.setMaximumWidth(420)
        self.player_widget = PlayerWidget()

        live_card = self._build_card("Connect to Live Stream")
        live_layout = live_card.layout()
        live_help = QLabel(
            "Select an MNE info source, then launch the live visualizer."
        )
        live_help.setObjectName("mutedLabel")
        live_help.setWordWrap(True)
        live_layout.addWidget(live_help)

        self.config_row = QHBoxLayout()
        self.config_row.setContentsMargins(0, 0, 0, 0)
        self.config_row.setSpacing(12)
        self.config_row.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.config_row.addWidget(self.connection_widget, 2)
        self.config_row.addWidget(
            self.plot_settings_widget,
            1,
            Qt.AlignmentFlag.AlignTop,
        )
        live_layout.addLayout(self.config_row)

        self.launch_button = QPushButton("Launch Live Visualizer")
        self.launch_button.setDefault(True)
        self.launch_button.setMinimumWidth(260)
        self.launch_button.clicked.connect(self.launch_visualizer)

        launch_row = QHBoxLayout()
        launch_row.setContentsMargins(0, 4, 0, 0)
        launch_row.addStretch(1)
        launch_row.addWidget(self.launch_button)
        launch_row.addStretch(1)
        live_layout.addLayout(launch_row)
        layout.addWidget(live_card)

        file_card = self._build_card("Play from File")
        file_card.layout().addWidget(self.player_widget, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(file_card)
        layout.addStretch()
        main_layout.addWidget(self.main_container, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_compact_layout(event.size().width() < 980)

    def set_compact_layout(self, compact: bool):
        if not hasattr(self, "config_row"):
            return
        direction = (
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        if self.config_row.direction() != direction:
            self.config_row.setDirection(direction)
            self.config_row.setStretch(0, 2 if not compact else 0)
            self.config_row.setStretch(1, 1 if not compact else 0)
            self.plot_settings_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding
                if compact
                else QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Maximum,
            )

    def _build_card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("pipelineCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        card_layout.addWidget(title_label)
        return card

    def refresh_theme(self):
        if self.rt_erp_widget is not None:
            self.rt_erp_widget.refresh_theme()

    @Slot(dict)
    def _forward_plot_settings(self, params: dict):
        if self.rt_erp_widget is not None:
            self.rt_erp_widget.apply_live_settings(params)

    @Slot()
    def _on_visualizer_closed(self):
        self.rt_erp_widget = None

    def launch_visualizer(self):
        """Launches the real-time ERP visualizer."""
        params = self.connection_widget.get_settings()
        if params.get("info") is None:
            QMessageBox.warning(
                self,
                "No MNE Info",
                "Select an MNE info source before launching the live visualizer.",
            )
            return
        if not params.get("event_channels"):
            QMessageBox.warning(
                self,
                "No Event Channel",
                "The selected MNE info must include at least one stim channel so epochs can be built from incoming events.",
            )
            return

        self.connection_widget.save_settings()
        self.plot_settings_widget.save_settings()
        params.update(self.plot_settings_widget.get_settings())
        if self.rt_erp_widget:
            self.rt_erp_widget.close()
            self.rt_erp_widget.deleteLater()

        self.rt_erp_widget = RealTimeERP(params, self)
        self.rt_erp_widget.closed.connect(self._on_visualizer_closed)
        self.rt_erp_widget.show()
        self.rt_erp_widget.start_visualization()

    def closeEvent(self, event):
        self.player_widget.close()
        if self.rt_erp_widget:
            self.rt_erp_widget.close()
            self.rt_erp_widget = None
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(current_theme_name())
    app.setStyle("Fusion")
    main_window = RealTimeMainWidget()
    main_window.show()
    sys.exit(app.exec())
