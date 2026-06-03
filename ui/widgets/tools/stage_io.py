from __future__ import annotations

from pathlib import Path

import mne


def stage_kind_for_id(stage_id: str) -> str | None:
    if stage_id in {"raw", "filtered_raw"}:
        return "raw"
    if stage_id in {"epochs", "preprocessed"}:
        return "epochs"
    if stage_id in {"continuous_ica", "epochs_ica"}:
        return "ica"
    return None


def load_stage_object(stage_id: str, file_path: Path, data_kind: str | None = None):
    kind = data_kind or stage_kind_for_id(stage_id)
    if kind == "raw":
        return mne.io.read_raw_fif(file_path, preload=False)
    if kind == "epochs":
        return mne.read_epochs(file_path, preload=False)
    if kind == "ica":
        return mne.preprocessing.read_ica(file_path)
    raise ValueError(f"Unsupported stage: {stage_id}")
