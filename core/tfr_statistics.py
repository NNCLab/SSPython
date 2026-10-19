"""Shared baseline bootstrap and event-specific power masks.

The tail probabilities and FDR threshold intentionally reproduce the historical
analysis script. Statistics operate on raw, single-trial power, independently
of display normalization.
"""

import json
import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from core.analysis_events import (
    TFR_EVENT_PREFIX, event_options, select_event_epochs, tfr_event_metadata,
)
from core.baseline import validate_baseline

logger = logging.getLogger(__name__)


def bootstrap_power(data, times, *, baseline=(None, -0.1), alpha=0.05,
                    n_permutations=500, fdr_correction=True, random_state=None,
                    verbose=True):
    """Resample baseline times with replacement; return (channel, freq, time).

    Keep singleton axes, and use the historical NumPy RNG so a seed reproduces
    the original script. ``None`` respects ``np.random.seed`` as before.
    """
    data = np.asarray(data)
    times = np.asarray(times)
    if data.ndim != 4 or not all(data.shape) or data.shape[-1] != len(times):
        raise ValueError("Statistics require trial × channel × frequency × time power.")
    if np.iscomplexobj(data) or not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("Statistics require finite, nonnegative power values.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
        raise ValueError("The number of permutations must be a positive integer.")
    interval = validate_baseline(times, baseline)
    if interval is None:
        raise ValueError("Statistics require a baseline interval.")
    samples = np.flatnonzero((times >= interval[0]) & (times <= interval[1]))
    if len(samples) < 2:
        raise ValueError("The statistics baseline must contain at least two samples.")
    rng = np.random if random_state is None else np.random.RandomState(random_state)
    n_trials, n_channels, n_freqs, n_times = data.shape
    mask = np.zeros((n_channels, n_freqs, n_times), dtype=bool)
    for ch in tqdm(range(n_channels), desc="Bootstrapping", disable=not verbose):
        ch_data = data[:, ch]
        average = ch_data.mean(axis=0)
        boots = np.empty((n_permutations, n_freqs, 2))
        resampled = np.empty((n_trials, n_freqs, len(samples)))
        for permutation in range(n_permutations):
            for trial in range(n_trials):
                indices = rng.choice(samples, len(samples), replace=True)
                resampled[trial] = ch_data[trial][:, indices]
            boot_mean = resampled.mean(axis=0)
            boots[permutation, :, 0] = boot_mean.max(axis=1)
            boots[permutation, :, 1] = boot_mean.min(axis=1)
        # Sorted extrema give exactly the same counts as the original loops,
        # without allocating permutations × frequencies × times comparisons.
        pvals = np.empty_like(average, dtype=float)
        for freq in range(n_freqs):
            high = np.sort(boots[:, freq, 0])
            low = np.sort(boots[:, freq, 1])
            p_high = (n_permutations - np.searchsorted(high, average[freq], side="left")) / n_permutations
            p_low = np.searchsorted(low, average[freq], side="right") / n_permutations
            pvals[freq] = np.minimum(p_high, p_low)
        if fdr_correction:
            _, corrected = fdrcorrection(pvals.ravel(), alpha=alpha / 2)
            pvals = corrected.reshape(pvals.shape)
        mask[ch] = pvals < alpha
    return mask


def compute_tfr_statistics(epochs, power, *, event_code=None, compute_params, settings):
    """Build a pooled mask and a mask for each selected event, even for averages.

    Stockwell averages internally: computing each trial separately recovers the
    trial powers needed by the same bootstrap, using MNE's public API.
    """
    if compute_params.get("output", "power") != "power":
        raise ValueError("Bootstrap statistics are available only for power output.")
    selected = select_event_epochs(epochs, event_code)
    derivatives = power if isinstance(power, list) else [power]
    first = derivatives[0]
    params = dict(compute_params, average=False, return_itc=False, output="power")
    if first.data.ndim == 4:
        trial_data, times, freqs, names = first.data, first.times, first.freqs, first.ch_names
    elif params["method"] == "stockwell":
        params["average"] = True  # Each call contains exactly one trial.
        trials = []
        for index in tqdm(range(len(selected)), desc="Stockwell trial power", disable=not settings.get("verbose", True)):
            trial = selected[index:index + 1].copy().compute_tfr(**params)
            trials.append(trial.data)
        trial_data = np.stack(trials)
        times, freqs, names = trial.times, trial.freqs, trial.ch_names
    else:
        trials = selected.copy().compute_tfr(**params)
        trial_data, times, freqs, names = trials.data, trials.times, trials.freqs, trials.ch_names

    codes = [event_code]
    if event_code is None:
        codes.extend(code for code, _, _ in event_options(selected))
    records = []
    for code in codes:
        event_data = trial_data if code is None else trial_data[selected.events[:, 2] == code]
        records.append({
            "event_id": code,
            "n_epochs": len(event_data),
            "mask": bootstrap_power(event_data, times, **settings),
        })
    source_id = uuid4().hex
    for derivative in derivatives:
        metadata = dict(tfr_event_metadata(derivative) or {"event_id": event_code})
        metadata["statistics_id"] = source_id
        derivative.comment = TFR_EVENT_PREFIX + json.dumps(metadata)
    return {
        "version": 1, "source_id": source_id,
        "method": "baseline_resampling_with_replacement",
        "settings": dict(settings), "ch_names": list(names),
        "times": np.asarray(times).copy(), "freqs": np.asarray(freqs).copy(),
        "records": records,
    }


def matching_tfr_mask(statistics, tfr, event_code=None):
    """Return a mask aligned to this derivative, or None when it cannot match."""
    if not statistics or tfr is None:
        return None
    metadata = tfr_event_metadata(tfr)
    if (metadata is None or not statistics.get("source_id")
            or metadata.get("statistics_id") != statistics["source_id"]):
        return None
    if event_code is None:
        event_code = metadata["event_id"]
    if (not np.array_equal(statistics["times"], tfr.times)
            or not np.array_equal(statistics["freqs"], tfr.freqs)):
        return None
    if any(name not in statistics["ch_names"] for name in tfr.ch_names):
        return None
    for record in statistics["records"]:
        if record["event_id"] == event_code:
            mask = np.asarray(record["mask"], dtype=bool)
            expected = (len(statistics["ch_names"]), len(tfr.freqs), len(tfr.times))
            if mask.shape != expected:
                return None
            indices = [statistics["ch_names"].index(name) for name in tfr.ch_names]
            return mask[indices]
    return None


def save_tfr_statistics(path, statistics):
    """Persist masks and provenance without pickle; None removes stale masks."""
    path = Path(path)
    if statistics is None:
        path.unlink(missing_ok=True)
        return
    metadata = {key: value for key, value in statistics.items()
                if key not in ("times", "freqs", "records")}
    metadata["records"] = [{key: value for key, value in record.items() if key != "mask"}
                           for record in statistics["records"]]
    arrays = {f"mask_{index}": record["mask"]
              for index, record in enumerate(statistics["records"])}
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata),
                                times=statistics["times"], freqs=statistics["freqs"], **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_tfr_statistics(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = json.loads(str(archive["metadata"].item()))
            if result["version"] != 1:
                raise ValueError("Unsupported TFR statistics version.")
            result["times"] = archive["times"].copy()
            result["freqs"] = archive["freqs"].copy()
            if (not isinstance(result["source_id"], str)
                    or not isinstance(result["settings"], dict)
                    or not isinstance(result["ch_names"], list)
                    or result["times"].ndim != 1 or result["freqs"].ndim != 1):
                raise ValueError("Invalid TFR statistics metadata.")
            expected = (len(result["ch_names"]), len(result["freqs"]), len(result["times"]))
            for index, record in enumerate(result["records"]):
                record["mask"] = archive[f"mask_{index}"].copy()
                if (record["mask"].shape != expected or record["mask"].dtype != bool
                        or (record["event_id"] is not None and not isinstance(record["event_id"], int))):
                    raise ValueError("Invalid TFR statistics mask.")
        return result
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Could not load TFR statistics from %s", path, exc_info=True)
        return None
