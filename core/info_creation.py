from __future__ import annotations

import csv
import math
from pathlib import Path

import mne
import numpy as np

from core.channel_info import (
    ChannelInfo,
    MONTAGE_REQUIRED_CHANNEL_TYPES,
    MontageValidationError,
    identify_montage_issues,
    validate_channel_info,
)


def read_electrodes_tsv(
    path: Path | str, *, unit: str = "m"
) -> mne.channels.DigMontage:
    """Read name/x/y/z electrode positions already in MNE head coordinates."""
    scales = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    if unit not in scales:
        raise ValueError("Electrode coordinate units must be m, cm, or mm.")

    positions: dict[str, np.ndarray] = {}
    seen_names: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if reader.fieldnames is None:
            raise ValueError("The electrodes TSV file is empty.")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        if not {"name", "x", "y", "z"}.issubset(reader.fieldnames):
            raise ValueError("Electrodes TSV must contain name, x, y, and z columns.")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Electrodes TSV column names must be unique.")

        for row in reader:
            line_number = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Invalid number of fields on TSV line {line_number}.")
            name = row["name"].strip()
            if not name or name.casefold() == "n/a":
                raise ValueError(f"Missing electrode name on TSV line {line_number}.")
            if name.casefold() in seen_names:
                raise ValueError(f"Duplicate electrode name '{name}'.")
            seen_names.add(name.casefold())
            coordinates = [row[axis].strip() for axis in ("x", "y", "z")]
            missing = [value.casefold() in {"", "n/a"} for value in coordinates]
            # BIDS permits electrodes whose position is entirely unavailable.
            # Leave these out so they cannot satisfy montage validation.
            if all(missing):
                continue
            if any(missing):
                raise ValueError(f"Incomplete coordinates for electrode '{name}'.")
            try:
                xyz = np.array([float(value) for value in coordinates])
            except ValueError as exc:
                raise ValueError(f"Invalid coordinates for electrode '{name}'.") from exc
            if not np.isfinite(xyz).all():
                raise ValueError(f"Coordinates for electrode '{name}' must be finite.")
            positions[name] = xyz * scales[unit]

    if not positions:
        raise ValueError("Electrodes TSV contains no available electrode positions.")
    return mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head")


def create_info(
    channel_info: ChannelInfo,
    sfreq: float,
    montage: str | mne.channels.DigMontage,
) -> mne.Info:
    """Create an Info with validated channel metadata and montage coverage."""
    channel_info = ChannelInfo(
        ch_names=[str(name).strip() for name in channel_info.ch_names],
        ch_types=[str(kind).strip().lower() for kind in channel_info.ch_types],
    )
    if not channel_info.ch_names:
        raise ValueError("Add at least one channel before creating INFO.")
    validate_channel_info(channel_info)
    try:
        sfreq = float(sfreq)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sampling frequency must be a positive finite number.") from exc
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be a positive finite number.")

    if isinstance(montage, str):
        montage = mne.channels.make_standard_montage(montage)
    if not isinstance(montage, mne.channels.DigMontage):
        raise ValueError("Choose a built-in montage or an electrodes TSV file.")
    issues = identify_montage_issues(channel_info, montage)
    if issues:
        names = ", ".join(issue.name for issue in issues)
        raise MontageValidationError(
            f"The selected montage has no position for: {names}. "
            "Check the channel names and types or choose another montage.",
            channel_info=channel_info,
            issues=issues,
        )

    info = mne.create_info(channel_info.ch_names, sfreq, channel_info.ch_types)
    info.set_montage(montage, match_case=False, on_missing="raise")
    for channel, kind in zip(info["chs"], channel_info.ch_types, strict=True):
        if kind in MONTAGE_REQUIRED_CHANNEL_TYPES and not np.isfinite(channel["loc"][:3]).all():
            raise ValueError(f"No finite electrode position for '{channel['ch_name']}'.")
    return info
