from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np


@dataclass(frozen=True)
class MergeEntry:
    source_path: Path
    annotation_renames: dict[str, str] | None = None


def read_annotation_descriptions(source_path: Path) -> list[str]:
    raw = _load_raw_fif(source_path, preload=False)
    try:
        return _unique_descriptions(raw.annotations.description)
    finally:
        if hasattr(raw, "close"):
            raw.close()


def validate_merge_entries(entries: list[MergeEntry]) -> list[MergeEntry]:
    if len(entries) < 2:
        raise ValueError("Select at least two FIF files to merge.")

    finalized_labels: dict[str, Path] = {}
    for entry in entries:
        if Path(entry.source_path).suffix.lower() != ".fif":
            raise ValueError("Only .fif files can be merged.")

        descriptions = read_annotation_descriptions(entry.source_path)
        rename_map = {
            str(original): str(updated).strip()
            for original, updated in (entry.annotation_renames or {}).items()
        }
        mapped_labels = [rename_map.get(description, description).strip() for description in descriptions]

        if any(not label for label in mapped_labels):
            raise ValueError(
                f"Annotation names cannot be empty: {Path(entry.source_path).name}"
            )
        if len(set(mapped_labels)) != len(mapped_labels):
            raise ValueError(
                f"Annotation names must remain unique within {Path(entry.source_path).name}."
            )

        for label in mapped_labels:
            previous_source = finalized_labels.get(label)
            if previous_source is not None and previous_source != entry.source_path:
                raise ValueError(
                    "Annotation names would collide across merged files. "
                    f"Rename '{label}' before merging."
                )
            finalized_labels[label] = Path(entry.source_path)

    return entries


def merge_fif_files(entries: list[MergeEntry], output_path: Path) -> Path:
    validate_merge_entries(entries)
    managed_raws: list[mne.io.BaseRaw] = []

    try:
        for entry in entries:
            raw = _load_raw_fif(entry.source_path, preload=True)
            managed_raws.append(raw)
            _apply_annotation_renames(raw, entry.annotation_renames or {})

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_raw = mne.concatenate_raws(managed_raws, verbose="error")
        merged_raw.save(output_path, overwrite=True, verbose="error")
        return output_path
    finally:
        for raw in managed_raws:
            if hasattr(raw, "close"):
                raw.close()


def _load_raw_fif(source_path: Path, *, preload: bool) -> mne.io.BaseRaw:
    source_path = Path(source_path)
    if source_path.suffix.lower() != ".fif":
        raise ValueError("Only .fif files can be merged.")
    try:
        return mne.io.read_raw_fif(source_path, preload=preload, verbose="error")
    except Exception as exc:
        raise ValueError(
            f"Only raw FIF files can be merged: {source_path.name}"
        ) from exc


def _apply_annotation_renames(
    raw: mne.io.BaseRaw,
    annotation_renames: dict[str, str],
) -> None:
    if not raw.annotations:
        return

    descriptions = np.asarray(raw.annotations.description, dtype=object)
    renamed = np.array(
        [str(annotation_renames.get(description, description)) for description in descriptions],
        dtype=str,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=raw.annotations.onset.copy(),
            duration=raw.annotations.duration.copy(),
            description=renamed,
            orig_time=raw.annotations.orig_time,
            ch_names=raw.annotations.ch_names,
        )
    )


def _unique_descriptions(descriptions: np.ndarray) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for description in descriptions:
        label = str(description)
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered
