"""Exact event selection and event identity for analysis derivatives."""

import json

import numpy as np


TFR_EVENT_PREFIX = "sspython:event:"


def event_options(epochs):
    """Return one option per event code actually present, including aliases."""
    names_by_code = {}
    for name, code in (epochs.event_id or {}).items():
        names_by_code.setdefault(int(code), []).append(str(name))
    codes, counts = np.unique(epochs.events[:, 2], return_counts=True)
    return [
        (int(code), " / ".join(names_by_code.get(int(code), [])) or f"Event {code}", int(count))
        for code, count in zip(codes, counts)
    ]


def event_label(epochs, event_code):
    if event_code is None:
        return "All events"
    for code, name, _ in event_options(epochs):
        if code == int(event_code):
            return f"{name} [event_id={code}]"
    return f"Event {event_code}"


def select_event_epochs(epochs, event_code=None):
    """Select by exact code; a string names a single event_id entry, not a tag."""
    if event_code is None:
        return epochs
    if isinstance(event_code, str):
        event_code = epochs.event_id[event_code]
    indices = np.flatnonzero(epochs.events[:, 2] == int(event_code))
    if not len(indices):
        raise ValueError(f"No epochs match event_id={event_code}.")
    return epochs[indices]


def tfr_event_metadata(tfr):
    """Read event identity from the comment, which MNE persists in TFR files."""
    comment = getattr(tfr, "comment", None)
    if not isinstance(comment, str) or not comment.startswith(TFR_EVENT_PREFIX):
        return None
    try:
        metadata = json.loads(comment[len(TFR_EVENT_PREFIX):])
    except (ValueError, TypeError):
        return None
    if not isinstance(metadata, dict) or "event_id" not in metadata:
        return None
    code = metadata["event_id"]
    if code is not None and not isinstance(code, int):
        return None
    return metadata


def compute_event_tfrs(epochs, *, event_code=None, **params):
    """Preserve condition averages alongside the pooled average and ITC."""
    selected = select_event_epochs(epochs, event_code)
    codes = [event_code]
    if (params.get("average", False) or params.get("method") == "stockwell") and event_code is None:
        options = event_options(selected)
        if len(options) > 1:
            codes.extend(code for code, _, _ in options)
    powers, coherences = [], []
    for code in codes:
        result = select_event_epochs(selected, code).copy().compute_tfr(**params)
        power, itc = result if isinstance(result, tuple) else (result, None)
        for derivative in (power, itc):
            if derivative is not None:
                derivative.comment = TFR_EVENT_PREFIX + json.dumps({"event_id": code})
        powers.append(power)
        if itc is not None:
            coherences.append(itc)
    power = powers[0] if len(powers) == 1 else powers
    itc = (coherences[0] if len(coherences) == 1 else coherences) if coherences else None
    return (power, itc) if itc is not None else power
