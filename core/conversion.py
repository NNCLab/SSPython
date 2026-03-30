from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

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
    merge: bool = False


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


def load_channel_info(channel_info_path: Path) -> tuple[list[str], list[str]]:
    with Path(channel_info_path).open("r", encoding="utf-8") as stream:
        channel_info = json.load(stream)

    ch_names = channel_info.get("ch_names")
    ch_types = channel_info.get("ch_types")
    if not isinstance(ch_names, list) or not isinstance(ch_types, list):
        raise ValueError(
            "Channel info JSON must include 'ch_names' and 'ch_types' lists."
        )
    if len(ch_names) != len(ch_types):
        raise ValueError("'ch_names' and 'ch_types' must have the same length.")
    return [str(name) for name in ch_names], [str(kind) for kind in ch_types]


def load_montage(*, builtin_name: str | None = None, custom_path: Path | None = None):
    if builtin_name:
        return mne.channels.make_standard_montage(builtin_name)
    if custom_path is not None:
        custom_path = Path(custom_path)
        if custom_path.suffix.lower() == ".fif":
            return mne.channels.read_dig_fif(custom_path)
        return mne.channels.read_custom_montage(custom_path)
    raise ValueError("A montage must be selected before converting files.")


def apply_montage(
    raw: mne.io.BaseRaw,
    *,
    builtin_montage: str | None = None,
    custom_montage_path: Path | None = None,
) -> mne.io.BaseRaw:
    montage = load_montage(
        builtin_name=builtin_montage,
        custom_path=custom_montage_path,
    )
    raw.set_montage(montage, on_missing="raise")
    return raw


def load_raw_from_source(
    source_path: Path,
    *,
    mat_channel_info_path: Path | None = None,
    preload: bool = True,
) -> mne.io.BaseRaw:
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".mat":
        return load_raw_from_mat(source_path, channel_info_path=mat_channel_info_path)
    return mne.io.read_raw(source_path, preload=preload, verbose="error")


def load_raw_from_mat(
    mat_path: Path,
    *,
    channel_info_path: Path | None,
) -> mne.io.RawArray:
    if channel_info_path is None:
        raise ValueError(
            f"MAT conversion requires a channel info JSON file: {mat_path.name}"
        )

    data = _load_mat_file(mat_path)
    sfreq = _extract_scalar(data, MAT_SFREQ_KEYS, "sampling frequency")
    raw_array = _extract_array(data, MAT_DATA_KEYS, "data matrix")
    ch_names, ch_types = load_channel_info(channel_info_path)

    if raw_array.ndim != 2:
        raise ValueError(f"MAT file must contain a 2D array: {mat_path.name}")
    if raw_array.shape[0] != len(ch_names) and raw_array.shape[1] == len(ch_names):
        raw_array = raw_array.T
    if raw_array.shape[0] != len(ch_names):
        raise ValueError(
            f"MAT data shape {raw_array.shape} does not match "
            f"{len(ch_names)} channels in {channel_info_path}."
        )

    raw_array = raw_array.astype(float, copy=True)
    volt_indices = [
        idx for idx, channel_type in enumerate(ch_types) if channel_type in MAT_VOLT_CHANNEL_TYPES
    ]
    if volt_indices:
        raw_array[volt_indices, :] *= 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(raw_array, info, verbose=False)
    _maybe_add_event_annotations(raw)
    return raw


def convert_files(
    entries: list[ConversionEntry],
    *,
    builtin_montage: str | None = None,
    custom_montage_path: Path | None = None,
    mat_channel_info_path: Path | None = None,
    merged_output_path: Path | None = None,
) -> dict[str, list[Path] | Path | None]:
    if not entries:
        raise ValueError("No files were selected for conversion.")

    converted_paths: list[Path] = []
    merge_raws: list[mne.io.BaseRaw] = []
    managed_raws: list[mne.io.BaseRaw] = []

    try:
        for entry in entries:
            raw = load_raw_from_source(
                entry.source_path,
                mat_channel_info_path=mat_channel_info_path,
                preload=True,
            )
            managed_raws.append(raw)
            apply_montage(
                raw,
                builtin_montage=builtin_montage,
                custom_montage_path=custom_montage_path,
            )

            entry.output_path.parent.mkdir(parents=True, exist_ok=True)
            raw.save(entry.output_path, overwrite=True, verbose="error")
            converted_paths.append(entry.output_path)

            if entry.merge:
                merge_raws.append(raw)

        merged_path = None
        if merged_output_path is not None:
            if len(merge_raws) < 2:
                raise ValueError("Select at least two files to create a merged output.")
            merged_output_path = Path(merged_output_path)
            merged_output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_raw = mne.concatenate_raws(merge_raws, verbose="error")
            merged_raw.save(merged_output_path, overwrite=True, verbose="error")
            merged_path = merged_output_path

        return {"converted": converted_paths, "merged": merged_path}
    finally:
        for raw in managed_raws:
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


def _extract_scalar(data: dict, keys: tuple[str, ...], label: str) -> float:
    for key in keys:
        if key not in data:
            continue
        value = np.asarray(data[key]).squeeze()
        if value.size != 1:
            continue
        return float(value)
    raise ValueError(f"MAT file is missing a valid {label}.")


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
