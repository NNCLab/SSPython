from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mne


COMMON_CHANNEL_TYPES = [
    "eeg",
    "eog",
    "ecg",
    "emg",
    "stim",
    "misc",
    "seeg",
    "ecog",
    "dbs",
    "bio",
    "resp",
    "temperature",
    "gsr",
    "csd",
    "hbo",
    "hbr",
    "fnirs_cw_amplitude",
    "fnirs_od",
    "eyegaze",
    "pupil",
]
MONTAGE_REQUIRED_CHANNEL_TYPES = {"eeg", "seeg", "ecog", "dbs"}


@dataclass(frozen=True)
class ChannelInfo:
    ch_names: list[str]
    ch_types: list[str]


@dataclass(frozen=True)
class ChannelIssue:
    index: int
    name: str
    channel_type: str
    reason: str


class MontageValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        channel_info: ChannelInfo,
        issues: list[ChannelIssue],
    ):
        super().__init__(message)
        self.channel_info = channel_info
        self.issues = issues


def channel_info_from_payload(payload: object) -> ChannelInfo:
    if isinstance(payload, dict):
        ch_names = payload.get("ch_names")
        ch_types = payload.get("ch_types")
    elif isinstance(payload, list):
        ch_names = []
        ch_types = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Channel info entry at index {index} must be an object."
                )
            ch_names.append(item.get("name"))
            ch_types.append(item.get("type", "eeg"))
    else:
        raise ValueError(
            "Channel info JSON must be either an object with 'ch_names'/'ch_types' "
            "or a list of {'name', 'type'} objects."
        )

    if not isinstance(ch_names, list) or not isinstance(ch_types, list):
        raise ValueError(
            "Channel info JSON must include 'ch_names' and 'ch_types' lists."
        )
    if len(ch_names) != len(ch_types):
        raise ValueError("'ch_names' and 'ch_types' must have the same length.")

    return ChannelInfo(
        ch_names=[str(name or "").strip() for name in ch_names],
        ch_types=[str(kind or "").strip().lower() for kind in ch_types],
    )


def load_channel_info(channel_info_path: Path) -> ChannelInfo:
    with Path(channel_info_path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return validate_channel_info(channel_info_from_payload(payload))


def save_channel_info(channel_info: ChannelInfo, output_path: Path) -> Path:
    validated = validate_channel_info(channel_info)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "ch_names": validated.ch_names,
                "ch_types": validated.ch_types,
            },
            stream,
            indent=4,
        )
    return output_path


def resolve_channel_info(
    channel_info: ChannelInfo | Path | str | None,
) -> ChannelInfo | None:
    if channel_info is None:
        return None
    if isinstance(channel_info, ChannelInfo):
        return validate_channel_info(channel_info)
    return load_channel_info(Path(channel_info))


def channel_info_from_raw(raw: mne.io.BaseRaw) -> ChannelInfo:
    return ChannelInfo(
        ch_names=[str(name) for name in raw.ch_names],
        ch_types=[str(kind).strip().lower() for kind in raw.get_channel_types()],
    )


def validate_channel_info(channel_info: ChannelInfo) -> ChannelInfo:
    issues = identify_channel_info_issues(channel_info)
    if issues:
        unique_reasons: list[str] = []
        for issue in issues:
            if issue.reason not in unique_reasons:
                unique_reasons.append(issue.reason)
        raise ValueError(" ".join(unique_reasons))
    return channel_info


def identify_channel_info_issues(channel_info: ChannelInfo) -> list[ChannelIssue]:
    issues_by_index: dict[int, list[str]] = {}

    if len(channel_info.ch_names) != len(channel_info.ch_types):
        raise ValueError("'ch_names' and 'ch_types' must have the same length.")

    normalized_name_counts = Counter(
        name.casefold().strip()
        for name in channel_info.ch_names
        if str(name).strip()
    )

    invalid_type_indices: set[int] = set()
    if channel_info.ch_types:
        try:
            dummy_names = [f"CH_{index + 1}" for index in range(len(channel_info.ch_types))]
            mne.create_info(
                ch_names=dummy_names,
                sfreq=1.0,
                ch_types=channel_info.ch_types,
            )
        except Exception:
            for index, channel_type in enumerate(channel_info.ch_types):
                try:
                    mne.create_info(
                        ch_names=[f"CH_{index + 1}"],
                        sfreq=1.0,
                        ch_types=[channel_type],
                    )
                except Exception:
                    invalid_type_indices.add(index)

    for index, (name, channel_type) in enumerate(
        zip(channel_info.ch_names, channel_info.ch_types, strict=False)
    ):
        normalized_name = name.strip()
        normalized_key = normalized_name.casefold()
        reasons: list[str] = []
        if not normalized_name:
            reasons.append("Channel name cannot be empty.")
        if normalized_name and normalized_name_counts[normalized_key] > 1:
            reasons.append("Channel names must be unique.")
        if not channel_type:
            reasons.append("Channel type cannot be empty.")
        elif index in invalid_type_indices:
            reasons.append(f"'{channel_type}' is not a valid MNE channel type.")
        if reasons:
            issues_by_index[index] = reasons

    return _flatten_issues(channel_info, issues_by_index)


def identify_montage_issues(
    channel_info: ChannelInfo,
    montage: mne.channels.DigMontage,
) -> list[ChannelIssue]:
    issues_by_index: dict[int, list[str]] = {}
    for issue in identify_channel_info_issues(channel_info):
        issues_by_index.setdefault(issue.index, []).append(issue.reason)

    montage_names = {name.casefold().strip() for name in montage.ch_names}
    for index, (name, channel_type) in enumerate(
        zip(channel_info.ch_names, channel_info.ch_types, strict=False)
    ):
        normalized_name = name.strip()
        normalized_type = channel_type.strip().lower()
        if (
            normalized_name
            and normalized_type in MONTAGE_REQUIRED_CHANNEL_TYPES
            and normalized_name.casefold() not in montage_names
        ):
            issues_by_index.setdefault(index, []).append(
                "Channel name is not present in the selected montage."
            )

    return _flatten_issues(channel_info, issues_by_index)


def apply_channel_info(
    raw: mne.io.BaseRaw,
    channel_info: ChannelInfo | Path | str | None,
) -> mne.io.BaseRaw:
    resolved = resolve_channel_info(channel_info)
    if resolved is None:
        return raw

    if len(raw.ch_names) != len(resolved.ch_names):
        raise ValueError(
            f"Channel info defines {len(resolved.ch_names)} channels, "
            f"but the recording contains {len(raw.ch_names)}."
        )

    rename_map = {
        old_name: new_name
        for old_name, new_name in zip(raw.ch_names, resolved.ch_names, strict=False)
        if old_name != new_name
    }
    if rename_map:
        raw.rename_channels(rename_map)

    current_types = raw.get_channel_types()
    channel_type_map = {
        channel_name: desired_type
        for channel_name, current_type, desired_type in zip(
            raw.ch_names,
            current_types,
            resolved.ch_types,
            strict=False,
        )
        if current_type != desired_type
    }
    if channel_type_map:
        raw.set_channel_types(channel_type_map)

    return raw


def _flatten_issues(
    channel_info: ChannelInfo,
    issues_by_index: dict[int, list[str]],
) -> list[ChannelIssue]:
    flattened: list[ChannelIssue] = []
    for index in sorted(issues_by_index):
        reasons = list(dict.fromkeys(issues_by_index[index]))
        flattened.append(
            ChannelIssue(
                index=index,
                name=channel_info.ch_names[index],
                channel_type=channel_info.ch_types[index],
                reason=" ".join(reasons),
            )
        )
    return flattened
