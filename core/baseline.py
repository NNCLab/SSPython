"""Baseline defaults and validation shared by processing and displays."""

import json

import numpy as np


def processing_history(inst):
    description = inst.info.get("description")
    try:
        history = json.loads(description) if description else []
    except (TypeError, ValueError):
        return []
    return [entry for entry in history if isinstance(entry, dict)] if isinstance(history, list) else []


def event_baseline_default(inst):
    """Suggest a pre-event interval only for explicitly event-based data."""
    mode = next(
        (entry["epoching"].get("mode") for entry in reversed(processing_history(inst))
         if isinstance(entry.get("epoching"), dict)),
        None,
    )
    times = np.asarray(inst.times)
    if mode != "event" or len(times) < 2 or times[0] >= 0 or times[-1] < 0:
        return None
    baseline = (max(float(times[0]), -0.2), 0.0)
    try:
        return validate_baseline(times, baseline)
    except ValueError:
        return None


def voltage_baseline_default(inst):
    """Preserve the baseline choice made during preprocessing or import."""
    if inst is None or getattr(inst, "baseline", None) is not None:
        return None
    if any("filter_epochs" in entry for entry in processing_history(inst)):
        return None
    return event_baseline_default(inst)


def validate_baseline(times, baseline):
    """Resolve endpoints and reject invalid or insufficiently sampled intervals.

    None disables correction; (None, None) explicitly selects the entire epoch.
    Half a sample of tolerance allows boundaries after resampling.
    """
    if baseline is None:
        return None
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("A baseline requires at least two time samples.")
    if len(baseline) != 2:
        raise ValueError("A baseline requires a start and a stop time.")
    start, stop = baseline
    start = float(times[0]) if start is None else float(start)
    stop = float(times[-1]) if stop is None else float(stop)
    if not np.isfinite([start, stop]).all() or start >= stop:
        raise ValueError("Baseline start must be finite and earlier than its stop.")
    tolerance = float(np.min(np.diff(times))) / 2 + 1e-12
    if start < times[0] - tolerance or stop > times[-1] + tolerance:
        raise ValueError(
            f"Baseline [{start:g}, {stop:g}] s is outside the data interval "
            f"[{times[0]:g}, {times[-1]:g}] s."
        )
    start, stop = max(start, float(times[0])), min(stop, float(times[-1]))
    mask = (times >= start - 1e-12) & (times <= stop + 1e-12)
    if np.count_nonzero(mask) < 2:
        raise ValueError("The selected baseline must contain at least two samples.")
    return start, stop
