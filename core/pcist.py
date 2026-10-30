"""PCIst analysis of condition-averaged EEG, with explicit millisecond units."""

import mne
import numpy as np

from core.analysis_events import event_label, select_event_epochs
from core.external import pci_st


PCIST_DEFAULTS = {
    "baseline_window": (-400.0, -50.0),
    "response_window": (0.0, 300.0),
    "k": 1.2,
    "min_snr": 1.1,
    "max_var": 99.0,
    "n_components": None,
    "embed": False,
    "L": 2,
    "tau": 1,
    "n_steps": 100,
    "max_thr_p": 1.0,
    "baseline_corr": False,
    "resample": None,
    "avgref": False,
}


def _number(name, value, *, minimum=None, maximum=None, integer=False, strict=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(number) or (integer and number != int(number)):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}.")
    if minimum is not None and (number <= minimum if strict else number < minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return int(number) if integer else number


def _parameters(parameters):
    unknown = set(parameters) - PCIST_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown PCIst parameter(s): {', '.join(sorted(unknown))}.")
    result = {**PCIST_DEFAULTS, **parameters}
    for name in ("baseline_window", "response_window"):
        value = result[name]
        try:
            if len(value) != 2:
                raise ValueError
            window = tuple(_number(name, point) for point in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain two finite times in milliseconds.") from exc
        if window[0] >= window[1]:
            raise ValueError(f"{name} start must be before its end.")
        result[name] = window
    if result["baseline_window"][1] > result["response_window"][0]:
        raise ValueError("The baseline window must end at or before the response window starts.")
    for name in ("embed", "baseline_corr", "avgref"):
        if not isinstance(result[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be True or False.")
        result[name] = bool(result[name])
    result["k"] = _number("k", result["k"], minimum=1, strict=True)
    result["min_snr"] = _number("min_snr", result["min_snr"], minimum=0)
    result["max_var"] = _number("max_var", result["max_var"], minimum=0,
                                maximum=100, strict=True)
    result["max_thr_p"] = _number("max_thr_p", result["max_thr_p"], minimum=0, strict=True)
    result["n_steps"] = _number("n_steps", result["n_steps"], minimum=2, integer=True)
    for name in ("L", "tau"):
        result[name] = _number(name, result[name], minimum=1, integer=True)
    if result["n_components"] is not None:
        result["n_components"] = _number("n_components", result["n_components"],
                                           minimum=1, integer=True)
    if result["resample"] is not None:
        result["resample"] = _number("resample", result["resample"], minimum=0, strict=True)
    return result


def _window_times(times, window):
    return times[(times >= window[0]) & (times < window[1])]


def calculate_pcist(epochs, *, event_code=None, picks=None, **parameters):
    """Calculate PCIst and all diagnostics from an averaged event response.

    Windows and returned times are in milliseconds. EEG is converted from MNE's
    volts to microvolts before calling the bundled algorithm. ``picks`` contains
    EEG channel names; omitted picks select all good EEG channels. No input data
    is modified. As in the bundled implementation, ``baseline_corr=True``
    subtracts the mean of *all original samples before -50 ms*, independently of
    ``baseline_window``. Window ends are exclusive. Embedding delays are samples
    at the sampling rate used after optional resampling.

    ``NST_*`` are the algorithm's transition densities; multiplying them by
    ``len(response_times)`` puts their differences on the same scale as each
    component's final ``dNST`` contribution. ``component_indices`` gives the
    original zero-based SVD component indices after variance and SNR selection.
    """
    params = _parameters(parameters)
    if isinstance(event_code, str):
        try:
            event_code = epochs.event_id[event_code]
        except KeyError as exc:
            raise ValueError(f"Unknown PCIst event: {event_code}.") from exc
    selected = select_event_epochs(epochs, event_code)
    good_indices = mne.pick_types(selected.info, eeg=True, exclude="bads")
    good_names = [selected.ch_names[index] for index in good_indices]
    if picks is None:
        ch_names = good_names
    elif isinstance(picks, str):
        ch_names = [picks]
    else:
        try:
            ch_names = list(picks)
        except TypeError as exc:
            raise ValueError("Select EEG channels by name.") from exc
    if not ch_names:
        raise ValueError("Select at least one good EEG channel for PCIst.")
    if any(not isinstance(name, str) or name not in good_names for name in ch_names):
        raise ValueError("PCIst channels must be existing EEG channels not marked bad.")
    if len(set(ch_names)) != len(ch_names):
        raise ValueError("PCIst channel selection must not contain duplicates.")
    data = selected.get_data(picks=ch_names)
    if data.shape[0] == 0:
        raise ValueError("No epochs are available for PCIst.")
    if not np.all(np.isfinite(data)):
        raise ValueError("PCIst input contains NaN or infinite values.")
    times = np.asarray(selected.times, dtype=float) * 1000.0
    if (times.ndim != 1 or len(times) < 2 or data.shape[-1] != len(times)
            or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0)):
        raise ValueError("PCIst needs at least two ordered, finite sample times.")
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-6, atol=1e-9):
        raise ValueError("PCIst input must be sampled at a uniform rate.")
    for name in ("baseline_window", "response_window"):
        if params[name][0] < times[0] - 1e-7 or params[name][1] > times[-1] + 1e-7:
            raise ValueError(f"{name} must lie within the loaded epoch ({times[0]:g} to {times[-1]:g} ms).")
        if len(_window_times(times, params[name])) < 2:
            raise ValueError(f"{name} needs at least two samples (the end is exclusive).")
    if params["baseline_corr"] and not np.any(times < -50):
        raise ValueError("Baseline correction requires original samples before -50 ms.")
    evoked = np.mean(data, axis=0) * 1e6
    if not np.all(np.isfinite(evoked)):
        raise ValueError("The averaged PCIst input must contain finite values.")

    # Validate the exact timeline produced by the upstream crop/resampling path
    # before its SVD or any quadratic distance matrix allocations occur.
    crop = (params["baseline_window"][0], params["response_window"][1])
    cropped_times = _window_times(times, crop)
    if params["resample"] is not None:
        n_samples = int((cropped_times[-1] - cropped_times[0]) / 1000 * params["resample"])
        if n_samples < 2:
            raise ValueError("Resampling leaves fewer than two samples.")
    _, processed_times = pci_st.preprocess_signal(evoked[:1], times, crop, **params)
    if params["n_components"] is not None:
        available = min(len(ch_names), len(_window_times(processed_times, params["response_window"])))
        if params["n_components"] > available:
            raise ValueError(f"n_components must be at most {available} for this channel/response selection.")
    effective_times = processed_times
    if params["embed"]:
        cut = (params["L"] - 1) * params["tau"]
        effective_times = effective_times[cut:]
    baseline_times = _window_times(effective_times, params["baseline_window"])
    response_times = _window_times(effective_times, params["response_window"])
    if len(baseline_times) < 2 or len(response_times) < 2:
        raise ValueError("The baseline and response each need at least two samples after resampling and embedding.")
    result = pci_st.calc_PCIst(evoked, times, full_return=True, **params)
    return {
        **result,
        "PCI": float(result["PCI"]),
        "params": params,
        "ch_names": ch_names,
        "n_epochs": int(data.shape[0]),
        "event_label": event_label(selected, event_code),
        "baseline_times": baseline_times,
        "response_times": response_times,
    }
