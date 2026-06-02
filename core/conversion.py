from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from core.channel_info import (
    ChannelInfo,
    MontageValidationError,
    identify_montage_issues,
)

try:
    import mat73
except ImportError:  # pragma: no cover - optional dependency
    mat73 = None


MAT_DATA_KEYS = ("y", "data")
MAT_SFREQ_KEYS = ("SR", "sfreq", "Fs")
MAT_VOLT_CHANNEL_TYPES = {"eeg", "eog", "ecg", "emg", "seeg", "ecog", "dbs"}


@dataclass(frozen=True)
class ConversionEntry:
    source_path: Path
    output_path: Path


def normalize_output_filename(name: str) -> str:
    cleaned_name = Path(name.strip()).name
    if not cleaned_name:
        raise ValueError("Output file name cannot be empty.")

    suffixes = "".join(Path(cleaned_name).suffixes)
    if suffixes:
        cleaned_name = cleaned_name[: -len(suffixes)]

    if cleaned_name.endswith("_raw"):
        cleaned_name = cleaned_name[:-4]

    if not cleaned_name:
        raise ValueError("Output file name cannot be empty.")

    return f"{cleaned_name}_raw.fif"


def default_output_filename(source_path: Path) -> str:
    return normalize_output_filename(source_path.name)


def discover_convertible_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")

    supported_files: list[Path] = []
    for file_path in sorted(path for path in folder.rglob("*") if path.is_file()):
        if file_path.name.endswith("_raw.fif"):
            continue
        if file_path.suffix.lower() == ".mat":
            supported_files.append(file_path)
            continue
        try:
            raw = mne.io.read_raw(file_path, preload=False, verbose="error")
        except Exception:
            continue
        else:
            supported_files.append(file_path)
            if hasattr(raw, "close"):
                raw.close()

    return supported_files


def validate_conversion_info(info: mne.Info) -> mne.Info:
    if info is None:
        raise ValueError("An MNE Info object is required before converting files.")
    if not isinstance(info, mne.Info):
        raise TypeError("Conversion info must be an mne.Info object.")
    if len(info["ch_names"]) == 0:
        raise ValueError("Conversion info must define at least one channel.")

    montage = info.get_montage()
    if montage is None:
        raise ValueError("Conversion info must include a montage.")

    channel_info = _channel_info_from_info(info)
    issues = identify_montage_issues(channel_info, montage)
    if issues:
        raise MontageValidationError(
            "Conversion info montage does not cover all required channels.",
            channel_info=channel_info,
            issues=issues,
        )

    return info


def load_conversion_info(source_path: Path) -> mne.Info:
    source_path = Path(source_path)
    raw = None
    try:
        raw = mne.io.read_raw(source_path, preload=False, verbose="error")
        return validate_conversion_info(raw.info.copy())
    except Exception as exc:
        raise ValueError(
            f"Could not load an MNE Info object with montage from {source_path.name}."
        ) from exc
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()


def apply_info_to_raw(raw: mne.io.BaseRaw, info: mne.Info) -> mne.io.BaseRaw:
    conversion_info = validate_conversion_info(info)
    _validate_channel_count(
        actual_count=len(raw.ch_names),
        expected_count=len(conversion_info["ch_names"]),
        source_label="recording",
    )
    _validate_sfreq(
        actual=float(raw.info["sfreq"]),
        expected=float(conversion_info["sfreq"]),
    )

    desired_names = list(conversion_info["ch_names"])
    rename_map = {
        old_name: new_name
        for old_name, new_name in zip(raw.ch_names, desired_names, strict=False)
        if old_name != new_name
    }
    if rename_map:
        raw.rename_channels(rename_map)

    desired_types = conversion_info.get_channel_types()
    current_types = raw.get_channel_types()
    channel_type_map = {
        channel_name: desired_type
        for channel_name, current_type, desired_type in zip(
            raw.ch_names,
            current_types,
            desired_types,
            strict=False,
        )
        if current_type != desired_type
    }
    if channel_type_map:
        raw.set_channel_types(channel_type_map)

    raw.info["bads"] = [
        channel_name
        for channel_name in conversion_info["bads"]
        if channel_name in raw.ch_names
    ]
    raw.set_montage(
        conversion_info.get_montage(),
        on_missing="raise",
        match_case=False,
    )
    return raw


def load_raw_from_source(
    source_path: Path,
    *,
    info: mne.Info,
    preload: bool = True,
) -> mne.io.BaseRaw:
    source_path = Path(source_path)
    conversion_info = validate_conversion_info(info)
    if source_path.suffix.lower() == ".mat":
        return load_raw_from_mat(source_path, info=conversion_info)
    raw = mne.io.read_raw(source_path, preload=preload, verbose="error")
    return apply_info_to_raw(raw, conversion_info)


def load_raw_from_mat(
    mat_path: Path,
    *,
    info: mne.Info,
) -> mne.io.RawArray:
    conversion_info = validate_conversion_info(info)
    data = _load_mat_file(mat_path)
    mat_sfreq = _extract_optional_scalar(data, MAT_SFREQ_KEYS)
    if mat_sfreq is not None:
        _validate_sfreq(
            actual=mat_sfreq,
            expected=float(conversion_info["sfreq"]),
        )

    raw_array = _extract_array(data, MAT_DATA_KEYS, "data matrix")
    ch_names = list(conversion_info["ch_names"])
    ch_types = conversion_info.get_channel_types()

    if raw_array.ndim != 2:
        raise ValueError(f"MAT file must contain a 2D array: {mat_path.name}")
    if raw_array.shape[0] != len(ch_names) and raw_array.shape[1] == len(ch_names):
        raw_array = raw_array.T
    if raw_array.shape[0] != len(ch_names):
        _validate_channel_count(
            actual_count=int(raw_array.shape[0]),
            expected_count=len(ch_names),
            source_label=f"MAT data matrix {raw_array.shape}",
        )

    raw_array = raw_array.astype(float, copy=True)
    volt_indices = [
        idx for idx, channel_type in enumerate(ch_types) if channel_type in MAT_VOLT_CHANNEL_TYPES
    ]
    if volt_indices:
        raw_array[volt_indices, :] *= 1e-6

    raw = mne.io.RawArray(raw_array, conversion_info.copy(), verbose=False)
    _maybe_add_event_annotations(raw)
    return raw


def convert_files(
    entries: list[ConversionEntry],
    *,
    info: mne.Info,
) -> dict[str, list[Path]]:
    if not entries:
        raise ValueError("No files were selected for conversion.")

    conversion_info = validate_conversion_info(info)
    converted_paths: list[Path] = []
    managed_raws: list[mne.io.BaseRaw] = []

    try:
        for entry in entries:
            raw = load_raw_from_source(
                entry.source_path,
                info=conversion_info,
                preload=True,
            )
            managed_raws.append(raw)

            entry.output_path.parent.mkdir(parents=True, exist_ok=True)
            raw.save(entry.output_path, overwrite=True, verbose="error")
            converted_paths.append(entry.output_path)

        return {"converted": converted_paths}
    finally:
        for raw in managed_raws:
            if hasattr(raw, "close"):
                raw.close()


def validate_source_for_info(
    source_path: Path,
    *,
    info: mne.Info,
) -> None:
    raw = load_raw_from_source(
        source_path,
        info=info,
        preload=False,
    )
    try:
        validate_conversion_info(raw.info)
    finally:
        if hasattr(raw, "close"):
            raw.close()


def _load_mat_file(mat_path: Path) -> dict:
    try:
        return loadmat(mat_path)
    except NotImplementedError:
        if mat73 is None:
            raise
    except ValueError:
        if mat73 is None:
            raise
    except Exception:
        if mat73 is None:
            raise

    return mat73.loadmat(mat_path)


def _extract_optional_scalar(data: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in data:
            continue
        value = np.asarray(data[key]).squeeze()
        if value.size != 1:
            continue
        return float(value)
    return None


def _extract_array(data: dict, keys: tuple[str, ...], label: str) -> np.ndarray:
    for key in keys:
        if key not in data:
            continue
        array = np.asarray(data[key])
        if array.size > 0:
            return array
    raise ValueError(f"MAT file is missing a valid {label}.")


def _maybe_add_event_annotations(raw: mne.io.BaseRaw) -> None:
    if "stim" not in raw.get_channel_types():
        return
    events = mne.find_events(raw, verbose=False)
    if len(events) == 0:
        return
    annotations = mne.annotations_from_events(
        events=events,
        sfreq=raw.info["sfreq"],
        event_desc=lambda event_id: f"STIM {event_id}",
    )
    raw.set_annotations(annotations)


def _channel_info_from_info(info: mne.Info) -> ChannelInfo:
    return ChannelInfo(
        ch_names=[str(name) for name in info["ch_names"]],
        ch_types=[str(kind).strip().lower() for kind in info.get_channel_types()],
    )


def _validate_channel_count(
    *,
    actual_count: int,
    expected_count: int,
    source_label: str,
) -> None:
    if actual_count == expected_count:
        return
    raise ValueError(
        f"{source_label} contains {actual_count} channels, "
        f"but conversion info defines {expected_count} channels."
    )


def _validate_sfreq(*, actual: float, expected: float) -> None:
    if np.isclose(actual, expected):
        return
    raise ValueError(
        f"Source sampling frequency {actual:g} Hz does not match "
        f"conversion info sampling frequency {expected:g} Hz."
    )
