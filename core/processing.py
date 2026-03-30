# %% Imports
import os
from pathlib import Path
import mne
import numpy as np
from typing import Literal, Optional
from statsmodels.stats.multitest import fdrcorrection
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import core.external.pci_st as pci_st
import logging
from datetime import datetime
import json

from core.pipelines import (
    build_analysis_paths,
    build_processing_paths,
    source_stem_from_derivative,
    source_stem_from_raw,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    A class to handle the preprocessing of a single subject's TMS-EEG data.

    This class takes a raw file, applies standard preprocessing steps like
    filtering, epoching, and ICA, and saves the cleaned data.
    """

    def __init__(self, raw_filepath: Path, output_dir: Path, verbose=True):
        """
        Initializes the ERPpreprocessor.

        Args:
            raw_filepath (str): The full path to the raw FIF file.
            output_dir (str): The directory where intermediate and final files will be saved.
        """
        if not os.path.exists(raw_filepath):
            raise FileNotFoundError(f"Raw file not found at: {raw_filepath}")

        self.raw_filepath = Path(raw_filepath)
        self.label = source_stem_from_raw(self.raw_filepath)
        self.output_dir = Path(output_dir)
        self.paths = self._get_paths()
        self.has = lambda x: self.paths.get(x, False) and self.paths[x].exists()
        self.processing_order = [
            "raw",
            "filtered_raw",
            "continuous_ica",
            "epochs",
            "epochs_ica",
            "preprocessed",
        ]

        # Ensure output directory exists
        logger.info(f"Preprocessor initialized for {self.label}.")
        logger.info(f"Output will be saved in: {self.output_dir}")

    @property
    def raw(self):
        if not hasattr(self, "_raw"):
            self._raw = mne.io.read_raw_fif(self.paths["raw"], preload=False)
        return self._raw

    @property
    def filtered_raw(self):
        if not hasattr(self, "_filtered_raw"):
            self._filtered_raw = mne.io.read_raw_fif(
                self.paths["filtered_raw"], preload=False
            )
        return self._filtered_raw

    @property
    def continuous_ica(self):
        if not hasattr(self, "_continuous_ica"):
            self._continuous_ica = mne.preprocessing.read_ica(
                self.paths["continuous_ica"]
            )
        return self._continuous_ica

    @property
    def epochs(self):
        if not hasattr(self, "_epochs"):
            if not self.has("epochs"):
                raise FileNotFoundError(
                    "Epochs file does not exist. Please run epoching first."
                )
            self._epochs = mne.read_epochs(self.paths["epochs"], preload=False)
        return self._epochs

    @property
    def epochs_ica(self):
        if not hasattr(self, "_epochs_ica"):
            self._epochs_ica = mne.preprocessing.read_ica(self.paths["epochs_ica"])
        return self._epochs_ica

    @property
    def preprocessed(self):
        if not hasattr(self, "_preprocessed"):
            if not self.has("preprocessed"):
                raise FileNotFoundError(
                    "Preprocessed file does not exist. Please run preprocessing first."
                )
            self._preprocessed = mne.read_epochs(
                self.paths["preprocessed"], preload=False
            )
        return self._preprocessed

    @property
    def epoching_mode(self) -> Literal["event", "fixed", "unknown"]:
        """
        Checks the epoching mode ('event' or 'fixed') by reading the
        description from the epochs file info.
        """
        if not self.has("epochs"):
            return "unknown"

        description_str = self.epochs.info.get("description")
        if not description_str:
            return "unknown"

        try:
            log_list = json.loads(description_str)
            return next(
                (log["epoching"]["mode"] for log in log_list if "epoching" in log),
                "unknown",
            )
        except (json.JSONDecodeError, KeyError, StopIteration):
            return "unknown"

    def __str__(self):
        msg = f"ERPpreprocessor for {self.label} \n >> "
        msg += " | ".join(
            [
                (
                    "✅ " + attr.capitalize()
                    if getattr(self, attr) is not None
                    else "🚫 " + attr.capitalize()
                )
                for attr in self.processing_order
            ]
        )
        # show output directory
        msg += f"\nOutput directory: {self.output_dir}"
        return msg

    def _get_paths(self):
        """Helper method to define file paths for all stages."""
        return build_processing_paths(self.raw_filepath, self.output_dir)

    def _get_last_continuous(self):
        if self.has("filtered_raw"):
            return self.filtered_raw
        else:
            return self.raw

    def _clear_downstream_files(self, current_step: str, verbose: bool = True):
        """
        Deletes all files and memory caches downstream from the current processing step.
        """
        try:
            start_index = self.processing_order.index(current_step)
        except ValueError:
            logger.info(
                f"Warning: Step '{current_step}' not found in processing order."
            )
            return

        # Iterate through all steps that come AFTER the current one
        for i in range(start_index + 1, len(self.processing_order)):
            step_to_delete = self.processing_order[i]
            if step_to_delete == "raw":
                continue

            # --- 2. Delete Physical File ---
            if self.has(step_to_delete):
                file_path = self.paths[step_to_delete]
                if verbose:
                    logger.info(
                        f"Overwriting '{step_to_delete}'. Deleting downstream file: {file_path.name}"
                    )

                # Construct the private attribute name (e.g., 'epochs' -> '_epochs')
                private_attr = f"_{step_to_delete}"
                if hasattr(self, private_attr):
                    # If the object is in memory, load its data to release the file lock before deleting.
                    obj = getattr(self, private_attr)
                    if hasattr(obj, "load_data"):
                        obj.load_data()
                    delattr(self, private_attr)
                    if verbose:
                        logger.info(f"Cleared in-memory object: {private_attr}")
                try:
                    os.remove(file_path)
                except PermissionError:
                    # Fallback for Windows if .close() didn't work or wasn't available
                    logger.warning(
                        f"File lock detected on {file_path.name}. Attempting to force release..."
                    )
                    try:
                        # Re-access property to force load, then gc might help, but usually .close() is enough
                        getattr(self, step_to_delete).load_data()
                    except Exception as e:
                        logger.error(f"Could not delete file {file_path.name}: {e}")

    @staticmethod
    def add_description(obj, description: dict):
        current_info = obj.info.get("description")
        if not current_info:
            log_list = []
        else:
            try:
                parsed = json.loads(current_info)
            except (json.JSONDecodeError, TypeError):
                parsed = [{"description": str(current_info)}]

            if isinstance(parsed, list):
                log_list = parsed
            elif isinstance(parsed, dict):
                log_list = [parsed]
            else:
                log_list = [{"description": str(parsed)}]
        log_list.append(description)
        obj.info["description"] = json.dumps(log_list, indent=4)

    @staticmethod
    def interpolate_tms_pulse(
        raw: mne.io.Raw,
        window: tuple[float, float] = (-2e-3, 6e-3),
        smoothing: tuple[float, float] = (-2e-3, 2e-3),
        span: int = 5,
        event_id: list[int] | None = None,
        verbose=True,
    ) -> mne.io.Raw:

        logger.info(f"Interpolating TMS pulse artifact in window {window}s...")

        sfreq = raw.info["sfreq"]
        n_samples = raw.n_times

        # --- STEP 1: Read onsets ---
        events, _ = mne.events_from_annotations(raw, verbose=False)

        if event_id:
            mask = np.isin(events[:, 2], event_id)
            events = events[mask]
        onsets = events[:, 0]

        # Convert time in seconds to integer samples, rounding to the nearest sample
        window_s = np.array([np.round(w * sfreq) for w in window], dtype=int)
        smoothing_s = np.array([np.round(sw * sfreq) for sw in smoothing], dtype=int)
        window_len = window_s[1] - window_s[0]
        span_s = (span // 2, span // 2)

        # --- STEP 2: Filter onsets to prevent out-of-bounds indexing ---
        min_rel_idx = window_s[0] - window_len

        # Calculate the latest sample needed relative to an onset
        # The max index is from the end smoothing: (cut1 + smoothing_s[1]) + span_s[1]
        max_rel_idx = window_s[1] + smoothing_s[1] + span_s[1]

        # Determine the required margin at the start and end of the recording
        required_start_padding = abs(min_rel_idx)
        required_end_padding = max_rel_idx

        # Filter out any onsets that are too close to the edges
        original_n_onsets = len(onsets)
        valid_onsets_mask = (onsets >= required_start_padding) & (
            onsets < n_samples - required_end_padding
        )
        onsets = onsets[valid_onsets_mask]

        if len(onsets) < original_n_onsets:
            removed_count = original_n_onsets - len(onsets)
            logger.info(
                f"Removed {removed_count} event(s) too close to the recording's boundaries."
            )

        if len(onsets) == 0:
            logger.warning(
                "No valid onsets found after filtering. Returning original data."
            )
            return raw

        # OLD
        # def tms_pulse_removal(y):
        #     for onset in onsets:
        #         cut0 = int(onset+window[0])
        #         cut1 = int(onset+window[1])

        #         y[cut0:cut1] = y[cut0-window_len:cut1-window_len][::-1]

        #         y[cut0+smoothing[0]:cut0+smoothing[1]] = np.array([np.mean(y[samp-span[0]:samp+span[1]]) for samp in range(cut0+smoothing[0], cut0+smoothing[1])])
        #         y[cut1+smoothing[0]:cut1+smoothing[1]] = np.array([np.mean(y[samp-span[0]:samp+span[1]]) for samp in range(cut1+smoothing[0], cut1+smoothing[1])])
        #     return y

        def tms_pulse_removal(y):
            # This function is applied to each channel independently
            for onset in onsets:
                cut0 = onset + window_s[0]
                cut1 = onset + window_s[1]

                # Replace artifact with a reversed copy of the preceding window
                y[cut0:cut1] = y[cut0 - window_len : cut1 - window_len][::-1]

                # Smooth the transition points using a moving average
                # Note: A list comprehension calculates all means before assigning,
                # which is important for the logic to be correct.

                # Smooth at the start of the replaced window
                start_smooth_range = range(cut0 + smoothing_s[0], cut0 + smoothing_s[1])
                y[start_smooth_range.start : start_smooth_range.stop] = np.array(
                    [
                        np.mean(y[samp - span_s[0] : samp + span_s[1] + 1])
                        for samp in start_smooth_range
                    ]
                )

                # Smooth at the end of the replaced window
                end_smooth_range = range(cut1 + smoothing_s[0], cut1 + smoothing_s[1])
                y[end_smooth_range.start : end_smooth_range.stop] = np.array(
                    [
                        np.mean(y[samp - span_s[0] : samp + span_s[1] + 1])
                        for samp in end_smooth_range
                    ]
                )
            return y

        # raw.apply_function(tms_pulse_removal, picks='eeg', verbose=verbose)
        ch_types = raw.get_channel_types()
        eeg_ch_idx = {
            ch: idx
            for idx, ch in enumerate(raw.info["ch_names"])
            if ch_types[idx] == "eeg"
        }
        pbar = tqdm(eeg_ch_idx.items(), desc="Removing artifacts...")
        for ch, idx in pbar:
            pbar.set_description(f"Removing artifact from {ch}...")
            raw._data[idx, :] = tms_pulse_removal(raw._data[idx, :])
        return raw

    def run_artifact_removal(
        self,
        window: tuple[float, float] | None = (-2e-3, 6e-3),
        smoothing: tuple[float, float] | None = (-2e-3, 2e-3),
        span: int = 5,
        event_id: list[int] | None = None,
        verbose=True,
    ):
        """Applies the TMS pulse interpolation to the raw data."""
        self._clear_downstream_files("raw")  # Clear everything after the raw input

        raw = self.raw.load_data()

        interpolated_raw = self.interpolate_tms_pulse(
            raw,
            window=window,
            smoothing=smoothing,
            span=span,
            event_id=event_id,
            verbose=verbose,
        )
        self.add_description(
            interpolated_raw,
            {
                "artifact_removal": {
                    "window": window,
                    "smoothing": smoothing,
                    "span": span,
                    "event_id": event_id,
                    "date": datetime.now().isoformat(),
                }
            },
        )

        interpolated_raw.save(self.paths["filtered_raw"], overwrite=True)
        self._filtered_raw = interpolated_raw

    def filter_continuous(
        self,
        notch: int | None = None,
        bandpass: tuple[float, float] | None = None,
        resample: int | None = None,
    ):
        self._clear_downstream_files("filtered_raw")

        raw = self._get_last_continuous().load_data()

        if notch is not None:
            notch_freqs = np.arange(
                notch, self.raw.info["sfreq"] / 2, notch
            )  # Harmonics up until sfreq / 2 (Nyquist)
            logger.info(f"Applying notch filter {notch}")
            raw.notch_filter(notch_freqs)

        if bandpass is not None and bandpass != (None, None):
            logger.info(
                f"Applying filter {bandpass[0] if bandpass[0] is not None else '-∞'} - {bandpass[1] if bandpass[1] is not None else '∞'} Hz..."
            )
            raw.filter(
                l_freq=bandpass[0],
                h_freq=bandpass[1],
                method="iir",
                iir_params=dict(order=3, ftype="butter", phase="zero-double"),
                verbose=False,
            )

        if resample is not None:
            logger.info(f"Resampling to {resample} Hz...")
            raw.resample(resample, npad="auto")

        self.add_description(
            raw,
            {
                "filter_continuous": {
                    "notch": notch,
                    "bandpass": bandpass,
                    "resample": resample,
                    "date": datetime.now().isoformat(),
                }
            },
        )

        raw.save(self.paths["filtered_raw"], overwrite=True, verbose=False)
        self._filtered_raw = raw

    def run_continuous_ica(
        self,
        n_components=None,
        max_iter: Literal["auto"] | int = "auto",
        random_state: int | None = None,
        method: Literal[
            "fastica", "infomax", "picard", "infomax (extended)", "picard (extended)"
        ] = "fastica",
        verbose: bool = True,
    ):
        """Fits and applies Independent Component Analysis (ICA) for artifact removal."""
        self._clear_downstream_files("continuous_ica")

        raw = self._get_last_continuous().load_data()
        logger.info("Fitting ICA...")
        fit_params = None
        if "(extended)" in method:
            method = method.replace(" (extended)", "").strip()
            fit_params = dict(extended=True)

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method=method,
            fit_params=fit_params,
            max_iter=max_iter,
            verbose=verbose,
        )

        ica.fit(raw, verbose=verbose)
        self.add_description(
            ica,
            {
                "continuous_ica": {
                    "n_components": n_components,
                    "max_iter": max_iter,
                    "random_state": random_state,
                    "method": method,
                    "extended": bool(fit_params and fit_params.get("extended")),
                    "input_stage": "filtered_raw" if self.has("filtered_raw") else "raw",
                    "date": datetime.now().isoformat(),
                }
            },
        )
        ica.save(self.paths["continuous_ica"], overwrite=True, verbose=False)
        self._continuous_ica = ica

    def run_epoching(
        self,
        mode: str = "event",  # New parameter: 'event' or 'fixed'
        fixed_duration: float | None = None,  # New parameter
        fixed_overlap: float = 0.0,  # New parameter
        event_id: dict | list[int] | int | None = None,
        detrend: (
            int | None
        ) = None,  # Changed type to int per your GUI map (0 or 1) or None
        tlim: tuple[float, float] = (-0.8, 0.8),
        resample: int | None = None,
        pick_eeg_only: bool = True,
        bandpass: tuple[float, float] | None = None,
        notch: int | None = None,
        reference: list[str] | str | None = None,
        verbose: bool = True,
    ):
        """
        Segments the data into epochs.

        Supports both Event-based (TMS/ERP) and Fixed-length (Resting State) epoching.

        Args:
            mode (str, optional): The epoching mode. Can be 'event' or 'fixed'. Defaults to "event".
            fixed_duration (float | None, optional): The duration of the epochs in seconds for fixed-length epoching. Defaults to None.
            fixed_overlap (float, optional): The overlap between epochs in seconds for fixed-length epoching. Defaults to 0.0.
            event_id (dict | None, optional): The event ID to use for epoching. Defaults to None.
            detrend (int | None, optional): The detrending method. Can be 0 for constant (DC) detrending, 1 for linear detrending, or None for no detrending. Defaults to None.
            tlim (tuple[float, float], optional): The time limits for the epochs in seconds. Defaults to (-0.8, 0.8).
            resample (int | None, optional): The resampling frequency. Defaults to None.
            pick_eeg_only (bool, optional): Whether to pick only EEG channels. Defaults to True.
            bandpass (tuple[float, float] | None, optional): The bandpass filter frequencies. Defaults to None.
            notch (int | None, optional): The notch filter frequency. Defaults to None.
            reference (list[str] | str | None, optional): The reference channels. Defaults to None.
            verbose (bool, optional): The verbosity level. Defaults to True.
        """
        self._clear_downstream_files("epochs")

        # 1. Load Data
        raw = self._get_last_continuous().load_data()

        # 2. Common Pre-processing (Reference, Notch, Bandpass, ICA)
        #    These apply to the continuous data regardless of epoching mode.

        if reference is not None:
            logger.info(f"Setting reference ({reference})...")
            if reference == "average":
                raw.set_eeg_reference(reference, projection=True)
                raw.apply_proj()
            else:
                raw.set_eeg_reference(reference)

        if notch is not None:
            # Harmonics up until sfreq / 2 (Nyquist)
            notch_freqs = np.arange(notch, raw.info["sfreq"] / 2, notch)
            logger.info(f"Applying notch filter {notch} Hz...")
            raw.notch_filter(notch_freqs)

        if bandpass is not None and bandpass != (None, None):
            l_freq = bandpass[0] if bandpass[0] is not None else 0
            h_freq = bandpass[1] if bandpass[1] is not None else raw.info["sfreq"] / 2
            logger.info(f"Applying filter {l_freq} - {h_freq} Hz...")

            raw.filter(
                l_freq=bandpass[0],
                h_freq=bandpass[1],
                method="iir",
                iir_params=dict(order=3, ftype="butter", phase="zero-double"),
                verbose=False,
            )

        if self.has("continuous_ica"):
            logger.info(
                f"Applying continuous ICA ({len(self.continuous_ica.exclude)} components excluded)..."
            )
            raw = self.continuous_ica.apply(raw)

        # 3. Branching Logic: Event vs Fixed
        logger.info(f"Epoching mode: {mode}")

        if mode == "fixed":
            # --- Mode A: Fixed Length (Resting State) ---
            if not fixed_duration:
                raise ValueError(
                    "Fixed duration must be specified for resting state epoching."
                )

            # Handle channel picking manually for fixed length
            if pick_eeg_only:
                raw.pick_types(eeg=True, meg=False, eog=False, stim=False)

            logger.info(
                f"Creating fixed epochs: {fixed_duration}s duration, {fixed_overlap}s overlap"
            )
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=fixed_duration,
                overlap=fixed_overlap,
                preload=True,
                verbose=verbose,
            )
            # Note: make_fixed_length_epochs does not accept 'detrend' or 'baseline' arguments.
            # If detrending is required, it is usually handled by the high-pass filter above.

        else:
            # --- Mode B: Event Based (ERP/TMS) ---
            if not raw.annotations:
                raise ValueError(
                    "No annotations found in Raw data for event-based epoching."
                )

            events, annotation_event_map = mne.events_from_annotations(raw)
            resolved_event_id = self._resolve_annotation_event_id(
                annotation_event_map,
                event_id,
            )

            logger.info(f"Creating event-based epochs: {tlim}")
            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=resolved_event_id,
                tmin=tlim[0],
                tmax=tlim[1],
                picks="eeg" if pick_eeg_only else None,
                baseline=None,
                preload=True,
                detrend=detrend,
                verbose=verbose,
                reject=None, 
            )

        # 4. Common Post-processing (Resample, Save)
        if resample is not None:
            logger.info(f"Resampling to {resample} Hz...")
            epochs.resample(resample, npad="auto")

        # Update metadata description
        self.add_description(
            epochs,
            {
                "epoching": {
                    "mode": mode,
                    "continuous_ica": self.has("continuous_ica"),
                    "event_id": (
                        resolved_event_id if mode == "event" else "N/A"
                    ),
                    "tlim": tlim if mode == "event" else "N/A",
                    "fixed_duration": fixed_duration if mode == "fixed" else "N/A",
                    "fixed_overlap": fixed_overlap if mode == "fixed" else "N/A",
                    "resample": resample,
                    "pick_eeg_only": pick_eeg_only,
                    "bandpass": bandpass,
                    "notch": notch,
                    "reference": reference,
                    "date": datetime.now().isoformat(),
                }
            },
        )

        logger.info("Saving epochs...")
        epochs.save(self.paths["epochs"], overwrite=True, verbose=verbose)
        self._epochs = epochs

    @staticmethod
    def _resolve_annotation_event_id(
        annotation_event_map: dict[str, int],
        selected_event_id: dict[str, int] | list[int] | int | None,
    ) -> dict[str, int] | int | None:
        if selected_event_id is None:
            return annotation_event_map or None

        if isinstance(selected_event_id, dict):
            resolved_map = {}
            for label, code in selected_event_id.items():
                label = str(label)
                resolved_map[label] = int(annotation_event_map.get(label, code))
            return resolved_map

        if isinstance(selected_event_id, (list, tuple, set, np.ndarray)):
            selected_codes = {
                int(code) for code in selected_event_id if code is not None
            }
        else:
            selected_codes = {int(selected_event_id)}

        resolved_map = {
            label: int(code)
            for label, code in annotation_event_map.items()
            if int(code) in selected_codes
        }
        if resolved_map:
            return resolved_map
        if len(selected_codes) == 1:
            return next(iter(selected_codes))
        return {str(code): int(code) for code in sorted(selected_codes)}

    def update_epochs(self, epochs: mne.Epochs):
        self._clear_downstream_files("epochs")
        self.add_description(
            epochs,
            {
                "epoch_review": {
                    "remaining_epochs": len(epochs),
                    "total_epochs": len(epochs.drop_log),
                    "bad_channels": epochs.info.get("bads", []),
                    "date": datetime.now().isoformat(),
                }
            },
        )
        epochs.save(self.paths["epochs"], overwrite=True, verbose=False)
        self._epochs = epochs

    def run_rereferencing(
        self,
        reference: list[str] | str | None = None,
    ):
        self._clear_downstream_files("epochs")
        if reference is not None:
            logger.info(f"Setting reference {reference}...")
            self.epochs.set_eeg_reference(reference, projection=True)
            self.epochs.apply_proj()
            self.add_description(
                self.epochs,
                {
                    "rereference": {
                        "reference": reference,
                        "projection_applied": True,
                        "date": datetime.now().isoformat(),
                    }
                },
            )
            logger.info("Saving changes...")
            self.epochs.save(self.paths["epochs"], overwrite=True)

    def run_epochs_ica(
        self,
        n_components=None,
        max_iter: Literal["auto"] | int = "auto",
        random_state: int | None = None,
        reference: list[str] | str | None = "average",
        method: Literal[
            "fastica", "infomax", "picard", "infomax (extended)", "picard (extended)"
        ] = "fastica",
        verbose: bool = True,
    ):
        """Fits and applies Independent Component Analysis (ICA) for artifact removal."""

        self._clear_downstream_files("epochs_ica")

        logger.info("Fitting ICA...")

        fit_params = None
        if "(extended)" in method:
            method = method.replace(" (extended)", "").strip()
            fit_params = dict(extended=True)
        pbar = tqdm(total=0, desc="Fitting ICA...", disable=not verbose)
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method=method,
            fit_params=fit_params,
            max_iter=max_iter,
            verbose=verbose,
        )

        epochs_for_ica = self.epochs.copy()

        # Conditionally apply baseline correction only if the data has a projection
        # and the baseline period is valid for the data's time range.
        if self.epochs.proj:
            baseline = (None, 0)
            tmin, tmax = epochs_for_ica.times.min(), epochs_for_ica.times.max()
            baseline_tmin = baseline[0] if baseline[0] is not None else tmin
            baseline_tmax = baseline[1] if baseline[1] is not None else tmax

            if (
                baseline_tmin < tmin
                or baseline_tmax > tmax
                or baseline_tmin >= baseline_tmax
            ):
                logger.warning(
                    f"Baseline {baseline} is outside of data time range [{tmin:.3f}, {tmax:.3f}]. Skipping baseline correction for ICA fitting."
                )
            else:
                logger.info(
                    f"Applying baseline correction {baseline} before fitting ICA..."
                )
                epochs_for_ica.apply_baseline(baseline)

        ica.fit(epochs_for_ica, verbose=verbose)
        self.add_description(
            ica,
            {
                "epochs_ica": {
                    "n_components": n_components,
                    "max_iter": max_iter,
                    "random_state": random_state,
                    "reference": reference,
                    "method": method,
                    "extended": bool(fit_params and fit_params.get("extended")),
                    "projection_present": bool(self.epochs.proj),
                    "date": datetime.now().isoformat(),
                }
            },
        )
        ica.save(self.paths["epochs_ica"], overwrite=True, verbose=False)

    def filter_epochs(
        self,
        bads: list[str] | None = None,
        bandpass: tuple[float, float] = (8.0, 45.0),
        resample: int | None = None,
        interpolate_bad_channels=True,
        reference: list[str] | str | None = None,
        baseline=(None, 0),
        verbose=True,
    ):

        logger.info("Loading epoched data...")
        epochs = self.epochs.load_data().copy()

        logger.info("Applying ICA...")

        if self.has("epochs_ica"):
            self.epochs_ica.apply(epochs, verbose=verbose)

        if bads is not None and len(bads) > 0:
            epochs.info["bads"] = list(set(epochs.info["bads"]).union(set(bads)))

        if bandpass[0] is not None or bandpass[1] is not None:
            logger.info(
                f"Applying filter {bandpass[0] if bandpass[0] is not None else '-∞'}-{bandpass[1] if bandpass[1] is not None else '∞'} Hz..."
            )
            epochs.filter(
                l_freq=bandpass[0],
                h_freq=bandpass[1],
                method="iir",
                iir_params=dict(order=3, ftype="butter", phase="zero-double"),
                verbose=False,
            )

        if interpolate_bad_channels:
            logger.info("Interpolating bad channels...")
            epochs.interpolate_bads(reset_bads=True, verbose=verbose)

        if reference is not None:
            logger.info(f"Setting reference ({reference})...")
            if reference == "average":
                epochs.set_eeg_reference(reference, projection=True)
                epochs.apply_proj()
            else:
                epochs.set_eeg_reference(reference)

        if resample:
            logger.info(f"Resampling to {resample} Hz...")
            epochs.resample(resample, npad="auto")

        # Only apply baseline correction for event-related data
        if self.epoching_mode == "event":
            if baseline is not None and baseline != (None, None):
                tmin, tmax = epochs.times.min(), epochs.times.max()
                baseline_tmin = baseline[0] if baseline[0] is not None else tmin
                baseline_tmax = baseline[1] if baseline[1] is not None else tmax

                if (
                    baseline_tmin < tmin
                    or baseline_tmax > tmax
                    or baseline_tmin >= baseline_tmax
                ):
                    logger.warning(
                        f"Baseline {baseline} is outside of data time range [{tmin:.3f}, {tmax:.3f}]. Skipping baseline correction."
                    )
                else:
                    logger.info(
                        f"Applying baseline correction {baseline} for event-related epochs..."
                    )
                    epochs.apply_baseline(baseline=baseline)
        else:
            logger.info("Skipping baseline correction for fixed-length epochs.")
            baseline = "skipped"  # For logging purposes

        self.add_description(
            epochs,
            {
                "filter_epochs": {
                    "bads": bads,
                    "bandpass": bandpass,
                    "resample": resample,
                    "interpolate_bad_channels": interpolate_bad_channels,
                    "reference": reference,
                    "baseline": baseline,
                    "date": datetime.now().isoformat(),
                }
            },
        )

        logger.info("Saving preprocessed data...")
        epochs.save(self.paths["preprocessed"], overwrite=True, verbose=verbose)
        self._preprocessed = epochs


class TMSEEGAnalysis:
    """
    A class to run outcome analyses on preprocessed TMS-EEG data.
    """

    def __init__(
        self, data_input: Path | str, output_dir: Path | str, preload=True, verbose=True
    ):
        """
        Initializes the TMSEEGAnalysis.

        Args:
            data_input (Union[str, mne.BaseEpochs]): Path to a preprocessed -epo.fif file
                                                     or an MNE Epochs object directly.
        """
        self.epochs = mne.read_epochs(data_input, preload=preload)
        self.evoked = self.epochs.average()
        self.times = self.epochs.times
        self.info = self.epochs.info
        self.data_input = Path(data_input)
        self.label = source_stem_from_derivative(self.data_input)
        self.output_dir = Path(output_dir)
        self.paths = self._get_paths(self.output_dir)
        self.has = lambda x: self.paths.get(x, False) and self.paths[x].exists()
        self.preload = preload
        if self.preload:
            self.derivatives = self._get_derivatives()

        # control verbose
        logger.info(f"Analysis initialized for {self.label}.")

    def _get_indexes(self, tlim: tuple[Optional[float], Optional[float]]) -> np.ndarray:
        """Gets indices for a given time interval."""
        return np.where(
            (self.times >= (tlim[0] or -np.inf)) & (self.times <= (tlim[1] or np.inf))
        )[0]

    def _get_paths(self, output_dir: Path | str = None):
        """Helper method to define file paths for all needed files."""
        return build_analysis_paths(self.data_input, Path(output_dir))

    def _get_derivatives(self):
        """Helper method to define file paths for all needed derivative files."""
        all_derivatives = {
            "tfr": (
                mne.time_frequency.read_tfrs(self.paths["tfr"], verbose=False)
                if self.has("tfr")
                else None
            ),
            "tfr_mask": (
                np.load(self.paths["tfr_mask"], allow_pickle=True)
                if self.has("tfr_mask")
                else None
            ),
            "itc": (
                mne.time_frequency.read_tfrs(self.paths["itc"], verbose=False)
                if self.has("itc")
                else None
            ),
            "itc_mask": (
                np.load(self.paths["itc_mask"], allow_pickle=True)
                if self.has("itc_mask")
                else None
            ),
            "masked_ave": (
                mne.read_evokeds(self.paths["masked_ave"], verbose=False)[0]
                if self.has("masked_ave")
                else None
            ),
        }
        return all_derivatives

    # ==== Statistics ====
    @staticmethod
    def _evoked_cluster_stat(
        data: np.ndarray,
        surrogates: np.ndarray,
        adjacency: scipy.sparse._csr.csr_array,
        threshold=2.0,
        tail=0,
        alpha=0.05,
        verbose=True,
    ):
        """Performs cluster-based statistics on evoked data."""
        n_ch, n_t = data.shape
        n_surr = surrogates.shape[-1]

        # estatística ponto-a-ponto
        mu = surrogates.mean(axis=-1)
        sigma = surrogates.std(axis=-1, ddof=1)
        T_obs = (data - mu) / sigma  # shape (ch, t)

        # clusters no dado real
        T_flat = T_obs.ravel()
        clus_idx, _ = mne.stats.cluster_level._find_clusters(
            T_flat, threshold, tail, adjacency
        )
        clus_mass = np.array([np.abs(T_flat[c]).sum() for c in clus_idx])

        # distribuição nula
        pbar = tqdm(
            range(n_surr),
            total=n_surr,
            desc="Creating Null Distribution",
            disable=not verbose,
        )
        H0 = np.empty(n_surr)
        for k in pbar:
            T_k = (surrogates[..., k] - mu) / sigma
            idx_k, _ = mne.stats.cluster_level._find_clusters(
                T_k.ravel(), threshold, tail, adjacency
            )
            H0[k] = max([np.abs(T_k.ravel()[c]).sum() for c in idx_k], default=0.0)

        # p-values corrigidos
        pvals = np.array([(H0 >= m).mean() for m in clus_mass])

        # voltar clusters para máscara 2-D se quiser plotar
        cluster_masks = []
        for idx in clus_idx:  # idx é 1-D
            mask = np.zeros(T_flat.size, dtype=bool)
            mask[idx] = True  # marca só os pontos do cluster
            cluster_masks.append(mask.reshape(n_ch, n_t))

        # juntando as mascaras
        if tail == 0:
            alpha = alpha / 2
        sig_mask = np.zeros_like(cluster_masks[0], dtype=bool)
        for mask, p in zip(cluster_masks, pvals):
            if p <= alpha:
                sig_mask |= mask

        return sig_mask, T_obs, cluster_masks, pvals, H0

    def circshift_test_evoked(
        self, n_perm=1000, alpha=0.05, verbose=True, overwrite=False
    ):
        """Perform cluster-based comparison of post-stimulus activity against surrogates generated by circularly shifting trials."""
        if not overwrite and self.derivatives["masked_ave"] is not None:
            logger.info(
                "Masked evoked data already exists. Use overwrite=True to replace it."
            )
            return self.derivatives["masked_ave"]

        data = np.transpose(self.epochs.get_data(), (1, 2, 0))
        n_channels, n_times, n_trials = data.shape

        # Making Surrogates
        surrogates = np.zeros((n_channels, n_times, n_perm))

        pbar = tqdm(
            range(n_perm),
            total=n_perm,
            desc="Circ-shift Surrogates",
            disable=not verbose,
        )
        for n in pbar:
            surrogate = np.zeros((n_channels, n_times))
            for j in range(n_trials):
                shift = np.random.randint(1, n_times)
                trial_shifted = np.roll(data[:, :, j], shift=shift, axis=1)
                surrogate += trial_shifted
            surrogates[:, :, n] = surrogate / n_trials  # Trials avg (surrogate)

        evoked = np.mean(data, axis=2)  # Trials avg (real Signal)
        n_chs, n_times = evoked.shape

        # Adjacency matrix (space + time)
        spatial_adj, _ = mne.channels.find_ch_adjacency(self.info, "eeg")
        full_adjacency = mne.stats.combine_adjacency(spatial_adj, n_times)

        # Clusterstat
        sig_mask, T_obs, cluster_masks, pvals, H0 = self._evoked_cluster_stat(
            data=evoked,
            surrogates=surrogates,
            adjacency=full_adjacency,
            alpha=alpha,
            verbose=verbose,
        )

        stats = {
            "sig_mask": sig_mask,
            "T_obs": T_obs,
            "cluster_masks": cluster_masks,
            "pvals": pvals,
            "H0": H0,
        }

        # Apply mask
        masked_ave = self.evoked.copy()
        masked_ave.data = sig_mask * self.evoked.data

        self.derivatives["masked_ave"] = masked_ave
        mne.write_evokeds(self.paths["masked_ave"], [masked_ave], overwrite=overwrite)

        return masked_ave, stats

    @staticmethod
    def bootstrap_tfr(
        tfr: mne.time_frequency.EpochsTFR,
        baseline: tuple[float, float] = (None, -0.1),
        alpha: float = 0.05,
        n_permutations: int = 500,
        fdr_correction: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """Performs a permutation test on Time-Frequency Representation (TFR) data."""
        data = tfr.data
        times = tfr.times
        baseline_samples = np.where(
            (times >= (baseline[0] or -np.inf)) & (times <= (baseline[1] or np.inf))
        )[0]

        if data.ndim == 3:
            data = data[np.newaxis, :, :, :]
        _, n_channels, n_freqs, n_times = data.shape

        mask = np.zeros((n_channels, n_freqs, n_times), dtype=bool)

        pbar = tqdm(
            range(n_channels),
            total=n_channels,
            desc="Bootstrapping",
            leave=False,
            disable=not verbose,
        )
        for ch in pbar:
            ch_data = data[:, ch, :, :]
            avg_data = np.mean(ch_data, axis=0)

            # Bootstrap distribution
            n_trials = ch_data.shape[0]
            boots = np.zeros((n_permutations, n_freqs, 2))
            for n in range(n_permutations):
                resampled_data = np.zeros((n_trials, n_freqs, len(baseline_samples)))
                for t in range(n_trials):
                    rand_idx = np.random.choice(
                        baseline_samples, len(baseline_samples), replace=True
                    )
                    resampled_data[t, :, :] = ch_data[t, :, rand_idx].T

                boot_mean = np.mean(resampled_data, axis=0)
                boots[n, :, 0] = np.max(boot_mean, axis=1)
                boots[n, :, 1] = np.min(boot_mean, axis=1)

            # P-values
            pvals = np.zeros_like(avg_data)
            for d in range(n_freqs):
                for t in range(n_times):
                    p1 = (np.sum(boots[:, d, 0] >= avg_data[d, t])) / n_permutations
                    p2 = (np.sum(boots[:, d, 1] <= avg_data[d, t])) / n_permutations
                    pvals[d, t] = min(p1, p2)

            # Correction
            if fdr_correction:
                _, pvals_corrected = fdrcorrection(pvals.ravel(), alpha=alpha / 2)
                pvals = pvals_corrected.reshape(pvals.shape)

            mask[ch, :, :] = pvals < alpha

        return np.squeeze(mask)

    # ==== Field Power ====
    @staticmethod
    def compute_field_power(epochs: mne.Epochs) -> np.ndarray:
        """Computes the Global Field Power (GFP)."""
        return np.sqrt(np.mean(epochs.average().data ** 2, axis=0)) * 1e6  # μV

    def get_field_power(self, tlim: tuple[float, float] = (0.01, 0.12), picks=None):
        """Extract the AUC of the GFP for a given time interval."""
        idx = self._get_indexes(tlim)
        return np.trapezoid(
            self.compute_field_power(self.epochs.copy().pick(picks))[idx],
            self.times[idx],
        )

    def plot_field_power(
        self,
        tlim: tuple[float, float] = (0.01, 0.12),
        xlim: tuple[float, float] = (-0.1, 0.3),
        picks=None,
        axis=None,
    ):
        """Plot the GFP for a given time interval."""
        idx = self._get_indexes(tlim)
        auc = self.get_field_power(tlim, picks)
        gfp = self.compute_field_power(self.epochs.copy().pick(picks))

        g_or_l = "Global" if picks is None else "Local"

        # --- 5. Plot the results (optional) ---
        if axis is None:
            fig, axis = plt.subplots()
        axis.plot(
            self.times,
            gfp,
            label=f"{g_or_l} Field Power {'('+', '.join(picks)+')' if picks is not None else ''}",
            color="k",
        )
        axis.fill_between(self.times[idx], gfp[idx], alpha=0.5, label=f"AUC: {auc:.3f}")
        axis.set_title(
            f"{g_or_l} Field Power ({g_or_l[0]}FP) and Area Under the Curve (AUC)"
        )
        axis.set_xlabel("Time (s)")
        axis.set_xlim(xlim)
        axis.set_ylabel(rf"{'G' if picks is None else 'L'}FP ($\mu V$)")
        axis.legend()
        axis.grid(True)

    # ==== Signal to Noise Ratio ====
    def compute_snr(
        self, baseline: tuple = (None, 0), response: tuple = (0.01, 0.12)
    ) -> float:
        """
        Computes the Signal-to-Noise Ratio (SNR).

        Args:
            baseline (tuple): The time interval for the baseline.
            response (tuple): The time interval for the response.

        Returns:
            float: The computed SNR value.
        """
        baseline_idx = self._get_indexes(baseline)
        response_idx = self._get_indexes(response)
        baseline_power = np.mean(self.evoked.data[:, baseline_idx] ** 2)
        response_power = np.mean(self.evoked.data[:, response_idx] ** 2)
        snr = np.sqrt(response_power / baseline_power)
        return snr

    # ==== Phase Locking Factor ====
    @staticmethod
    def compute_plf(epochs: mne.Epochs, baseline: tuple = (None, 0), alpha=0.05):
        """
        Computes a Rayleigh-based statistical comparison of PLF values pre vs post stimulus.

        Parameters:
            epochs   : mne.Epochs object (assumed real-valued, already epochs)
            baseline : tuple (start, end) time window in seconds for baseline
            alpha    : float, significance level for Rayleigh test

        Returns:
        """

        def raylcdf(x, b):
            """Rayleigh cumulative distribution function."""
            return 1 - np.exp(-(x**2) / (2 * b**2))

        epochs = epochs.copy()
        X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        X = np.transpose(X, (1, 2, 0))  # → (n_channels, n_times, n_trials)

        n_channels, n_times, _ = X.shape
        base_mask = np.logical_and(
            epochs.times >= (baseline[0] or -np.inf),
            epochs.times <= (baseline[1] or np.inf),
        )
        xaxe = np.arange(0, 1.0001, 0.0001)
        evk = np.zeros((n_channels, n_times))
        evk_thresh = np.zeros_like(evk)
        binary_mask = np.zeros_like(evk)
        thresholds = np.zeros(n_channels)
        alpha_corr = alpha / n_channels  # Bonferroni correction

        for ch in range(n_channels):
            Signal = X[ch, :, :]  # shape: (n_times, n_trials)
            analytic = scipy.Signal.hilbert(Signal, axis=0)
            phases = analytic / np.abs(analytic)
            plf = np.abs(np.mean(phases, axis=1))  # PLF over trials
            evk[ch, :] = plf

            mean_base = np.mean(plf[base_mask])
            b = mean_base / np.sqrt(np.pi / 2)
            cdf_vals = raylcdf(xaxe, b)
            idx = np.where(cdf_vals <= 1 - alpha_corr)[0]

            if idx.size > 0:
                threshold = xaxe[idx[-1]]
                thresholds[ch] = threshold
                sig_mask = plf >= threshold
                evk_thresh[ch, sig_mask] = plf[sig_mask]
                binary_mask[ch, sig_mask] = 1

        plf = evk
        plf = mne.EvokedArray(evk, epochs.info)
        plf_masked = plf.copy()
        plf_masked._data = evk_thresh

        return plf, plf_masked, binary_mask, thresholds

    def get_plf(
        self,
        tlim: tuple[float, float] = (0.01, 0.12),
        picks=None,
        apply_mask=True,
        mask_params: dict = dict(baseline=(None, 0), alpha=0.05),
    ):
        """Extract the AUC for the PLF for a given time interval."""
        epochs = self.epochs.copy().pick(picks)
        plf, plf_masked, _, _ = self.compute_plf(epochs, **mask_params)
        idx = self._get_indexes(tlim)
        plf = plf_masked if apply_mask else plf
        plf = plf.get_data()
        plf = np.mean(plf**2, axis=0)
        auc = np.trapezoid(plf, epochs.times)
        return auc

    def plot_plf(
        self,
        tlim: tuple[float, float] = (0.01, 0.12),
        xlim: tuple[float, float] = (-0.1, 0.3),
        picks=None,
        apply_mask=False,
        mask_params: dict = dict(baseline=(None, 0), alpha=0.05),
        axis=None,
    ):
        """Plot the PLF for a given time interval."""
        epochs = self.epochs.copy().pick(picks)
        n_chs = len(epochs.ch_names)

        idx = self._get_indexes(tlim)
        auc = self.get_plf(tlim, picks, apply_mask, mask_params)
        plf, _, mask, _ = self.compute_plf(self.epochs.copy().pick(picks))
        mask[mask == 0] = 0.5
        if axis is None:
            fig, axis = plt.subplots()
        plf_data = plf.get_data()
        axis.imshow(
            plf_data,
            alpha=mask if apply_mask else 1,
            aspect="auto",
            cmap="turbo",
            origin="lower",
            interpolation="none",
            vmin=0,
            vmax=1,
            extent=[epochs.times[0], epochs.times[-1], -0.5, n_chs - 0.5],
        )
        axis.axvline(0, color="k", linestyle="--")
        cbar = axis.figure.colorbar(plt.cm.ScalarMappable(cmap="turbo"), ax=axis)
        cbar.ax.set_title("PLF", fontsize=10)
        axis.set_title(f"Phase Locking Factor (AUC: {auc:.3f})")
        axis.set_xlabel("Time (s)")
        axis.set_xlim(xlim)
        axis.set_yticks(np.arange(n_chs))
        axis.set_yticklabels(epochs.ch_names, fontsize=6 if picks is None else 10)
        axis.set_ylabel("Channels")

    # ==== Time-Frequency ====
    def compute_tfr_and_itc(
        self,
        freqs: np.ndarray = np.arange(8, 46),
        n_cycles=3.5,
        picks=None,
        verbose=True,
    ):
        logger.info("Computing TFR...")
        tfr = self.epochs.compute_tfr(
            "morlet",
            freqs,
            average=False,
            n_cycles=n_cycles,
            verbose=verbose,
            picks=picks,
        )
        logger.info("Computing ITC...")
        _, itc = self.epochs.compute_tfr(
            "morlet",
            freqs,
            return_itc=True,
            average=True,
            n_cycles=n_cycles,
            picks=picks,
            verbose=verbose,
        )
        self.derivatives["tfr"] = tfr
        self.derivatives["itc"] = itc
        return tfr, itc

    def save_tfr_and_itc(self, overwrite=False, verbose=True):
        """Saves the TFR and ITC data to files."""

        if self.derivatives["tfr"] is None and self.derivatives["itc"] is None:
            raise ValueError(
                "TFR and ITC not computed. Please run compute_tfr_and_itc() first."
            )
        if not overwrite and (self.has("tfr") and self.has("itc")):
            raise ValueError(
                "TFR and ITC already exist. Set overwrite=True to replace them."
            )

        logger.info("Saving TFR and ITC data...")
        if not self.has("tfr") or overwrite:
            self.derivatives["tfr"].save(
                self.paths["tfr"], overwrite=True, verbose=False
            )
        if not self.has("itc") or overwrite:
            self.derivatives["itc"].save(
                self.paths["itc"], overwrite=True, verbose=False
            )
        logger.info("TFR and ITC saved successfully.")

    def compute_tfr_and_itc_mask(
        self,
        baseline: tuple[float, float] = (None, -0.1),
        alpha: float = 0.05,
        n_permutations: int = 500,
        fdr_correction: bool = True,
        picks=None,
    ):
        """Computes the TFR and ITC masks based on bootstrap statistics."""
        if not self.derivatives["tfr"] or not self.derivatives["itc"]:
            raise ValueError(
                "TFR and ITC not computed. Please run compute_tfr_and_itc() first."
            )

        logger.info("Computing TFR and ITC masks...")
        tfr_mask = self.bootstrap_tfr(
            self.derivatives["tfr"].pick(picks),
            baseline=baseline,
            alpha=alpha,
            n_permutations=n_permutations,
            fdr_correction=fdr_correction,
        )
        itc_mask = self.bootstrap_tfr(
            self.derivatives["itc"].pick(picks),
            baseline=baseline,
            alpha=alpha,
            n_permutations=n_permutations,
            fdr_correction=fdr_correction,
        )
        self.derivatives["tfr_mask"] = tfr_mask
        self.derivatives["itc_mask"] = itc_mask

    def save_tfr_and_itc_mask(self, overwrite=False, verbose=True):
        if (
            self.derivatives["tfr_mask"] is None
            and self.derivatives["itc_mask"] is None
        ):
            raise ValueError(
                "TFR and ITC masks not computed. Please run compute_tfr_and_itc_mask() first."
            )
        if not overwrite and (self.has("tfr_mask") or self.has("itc_mask")):
            raise ValueError(
                "TFR and/or ITC masks already exist. Set overwrite=True to replace them."
            )

        logger.info("Saving TFR and ITC masks...")
        np.save(self.paths["tfr_mask"], self.derivatives["tfr_mask"], allow_pickle=True)
        np.save(self.paths["itc_mask"], self.derivatives["itc_mask"], allow_pickle=True)
        logger.info("TFR and ITC masks saved successfully.")

    def run_tfr_analysis(
        self,
        overwrite=False,
        verbose=True,
        tfr_kwargs: dict = dict(),
        tfr_mask_kwargs=dict(),
    ):
        """Runs the TFR and ITC computation and saves the results."""
        # check if TFR and ITC already exist
        if self.has("tfr") and self.has("itc") and not overwrite:
            raise ValueError(
                "TFR and ITC already exist. Set overwrite=True to replace them."
            )
        self.compute_tfr_and_itc(**tfr_kwargs)
        self.save_tfr_and_itc(overwrite=overwrite, verbose=verbose)
        self.compute_tfr_and_itc_mask(**tfr_mask_kwargs)
        self.save_tfr_and_itc_mask(overwrite=overwrite, verbose=verbose)

    def get_tfr(
        self,
        picks=None,
        tlim=(0.015, 0.12),
        flim=(None, None),
        apply_mask=False,
        agg_func=np.mean,
    ):
        if not self.derivatives["tfr"]:
            raise ValueError(
                "TFR not computed. Please run compute_tfr_and_itc() first."
            )
        if apply_mask and self.derivatives["tfr_mask"] is None:
            raise ValueError(
                "TFR mask not computed. Please run compute_tfr_and_itc_mask() first."
            )

        data = (
            self.derivatives["tfr"]
            .average()
            .get_data(
                picks=picks, tmin=tlim[0], tmax=tlim[1], fmin=flim[0], fmax=flim[1]
            )
        )
        if apply_mask:
            picks = picks if picks else self.derivatives["tfr"].ch_names
            ch_idx = [
                idx
                for idx, ch in enumerate(self.derivatives["tfr"].ch_names)
                if ch in picks
            ]
            times = self.derivatives["tfr"].times
            freqs = self.derivatives["tfr"].freqs
            t_idx = np.where(
                (times >= (tlim[0] or -np.inf)) & (times <= (tlim[1] or np.inf))
            )[0]
            f_idx = np.where(
                (freqs >= (flim[0] or -np.inf)) & (freqs <= (flim[1] or np.inf))
            )[0]
            ix = np.ix_(ch_idx, f_idx, t_idx)
            mask = self.derivatives["tfr_mask"][ix]
            data = data * mask
        return agg_func(data)

    def get_itc(
        self,
        picks=None,
        tlim=(0.015, 0.12),
        flim=(None, None),
        apply_mask=False,
        agg_func=np.mean,
    ):
        if not self.derivatives["itc"]:
            raise ValueError(
                "ITC not computed. Please run compute_tfr_and_itc() first."
            )
        if apply_mask and self.derivatives["itc_mask"] is None:
            raise ValueError(
                "ITC mask not computed. Please run compute_tfr_and_itc_mask() first."
            )
        data = self.derivatives["itc"].get_data(
            picks=picks, tmin=tlim[0], tmax=tlim[1], fmin=flim[0], fmax=flim[1]
        )

        if apply_mask:
            picks = picks if picks else self.derivatives["itc"].ch_names
            ch_idx = [
                idx
                for idx, ch in enumerate(self.derivatives["itc"].ch_names)
                if ch in picks
            ]
            times = self.derivatives["itc"].times
            freqs = self.derivatives["itc"].freqs
            t_idx = np.where(
                (times >= (tlim[0] or -np.inf)) & (times <= (tlim[1] or np.inf))
            )[0]
            f_idx = np.where(
                (freqs >= (flim[0] or -np.inf)) & (freqs <= (flim[1] or np.inf))
            )[0]
            ix = np.ix_(ch_idx, f_idx, t_idx)
            mask = self.derivatives["itc_mask"][ix]
            data = data * mask
        return agg_func(data)

    def get_natural_frequency(
        self,
        picks=None,
        tlim=(0.015, 0.12),
        apply_mask=True,
        use_itc=False,
        agg_func=np.sum,
        full_return=False,
    ):
        """Computes the natural frequency"""
        if not self.derivatives["tfr"] or not self.derivatives["itc"]:
            raise ValueError(
                "TFR or ITC not computed. Please run compute_tfr_and_itc() first."
            )
        if apply_mask and (
            self.derivatives["tfr_mask"] is None or self.derivatives["itc_mask"] is None
        ):
            raise ValueError(
                "TFR or ITC mask not computed. Please run compute_tfr_and_itc_mask() first."
            )

        tfr = self.derivatives["tfr"] if not use_itc else self.derivatives["itc"]
        times = tfr.times
        ch_names = tfr.ch_names
        freqs = tfr.freqs
        data = tfr.average().get_data(picks=picks, tmin=tlim[0], tmax=tlim[1])

        if apply_mask:
            mask = (
                self.derivatives["tfr_mask"]
                if not use_itc
                else self.derivatives["itc_mask"]
            )
            picks_idx = (
                [ch_names.index(ch) for ch in picks] if picks else range(len(ch_names))
            )
            t_idx = np.where(
                (times >= (tlim[0] or -np.inf)) & (times <= (tlim[1] or np.inf))
            )[0]
            ix = np.ix_(picks_idx, range(len(freqs)), t_idx)
            mask = mask[ix]
            data = data * mask

        data = data.mean(0)  # Channel average
        agg_freq = agg_func(data, axis=-1)  # Over time -> only freqs left
        nf = freqs[np.argmax(agg_freq)]
        return nf if not full_return else (nf, agg_freq)

    def plot_natural_frequency(
        self,
        tlim=(0.015, 0.12),
        xlim=(-0.1, 0.4),
        apply_mask=True,
        use_itc=False,
        picks=None,
        axis=None,
    ):
        """Plots the natural frequency."""
        if not self.derivatives["tfr"] or not self.derivatives["itc"]:
            raise ValueError(
                "TFR or ITC not computed. Please run compute_tfr_and_itc() first."
            )
        else:
            self.compute_tfr_and_itc(picks=picks)
            self.compute_tfr_and_itc_mask(picks=picks)

        if apply_mask and not (
            (self.derivatives["tfr_mask"] is not None)
            or (self.derivatives["itc_mask"] is not None)
        ):
            raise ValueError(
                "TFR or ITC mask not computed. Please run compute_tfr_and_itc_mask() first."
            )

        tfr = self.derivatives["tfr"] if not use_itc else self.derivatives["itc"]
        mask = (
            self.derivatives["tfr_mask"]
            if not use_itc
            else self.derivatives["itc_mask"]
        )
        nf, agg_freq = self.get_natural_frequency(
            picks=picks,
            tlim=tlim,
            apply_mask=apply_mask,
            use_itc=use_itc,
            full_return=True,
        )

        data = (
            tfr.average().get_data(picks=picks, tmin=xlim[0], tmax=xlim[1]).mean(0)
        )  # Average Over Channels
        picks_idx = (
            [tfr.ch_names.index(ch) for ch in picks]
            if picks
            else range(len(tfr.ch_names))
        )
        t_idx = np.where(
            (tfr.times >= (xlim[0] or -np.inf)) & (tfr.times <= (xlim[1] or np.inf))
        )[0]
        f_idx = range(len(tfr.freqs))
        ix = np.ix_(picks_idx, f_idx, t_idx)
        mask = mask[ix].sum(0) if apply_mask else np.ones_like(data)
        # Normalize so that values above 0 are between .5 and 1
        max_count = np.max(mask)
        alpha = mask / max_count if max_count > 0 else mask
        alpha = 0.5 + 0.5 * alpha
        alpha[alpha == 0] = 0.25

        if axis is None:
            fig, axis = plt.subplots(figsize=(8, 8))
            fig.suptitle("Natural Frequency")

        im = axis.imshow(
            data,
            alpha=alpha,
            cmap="turbo",
            origin="lower",
            aspect="auto",
            extent=[xlim[0], xlim[1], tfr.freqs[0] - 0.5, tfr.freqs[-1] - 0.5],
        )
        axis.grid()
        axis.axvline(0, color="k", linestyle="--")
        axis.axhline(nf, color="r", linestyle="--")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(axis)

        # Evoked
        eax = divider.append_axes("top", size="25%", pad=0.05)
        # remove spines
        for spine in eax.spines.values():
            spine.set_visible(False)
        eax.spines["left"].set_visible(True)
        eax.set_xticklabels([])
        eax.grid()
        evk_data = self.evoked.data * 1e6
        eax.plot(self.evoked.times, evk_data.T, color="k")
        eax.set_xlim(xlim)
        for ch in picks:
            ch_idx = self.evoked.ch_names.index(ch)
            eax.plot(self.evoked.times, evk_data[ch_idx].T, label=ch, lw=2)
        eax.legend(ncols=len(picks) // 2, loc="lower right")
        eax.set_ylabel("Amplitude (µV)")
        eax.axvline(0, color="k", linestyle="--")

        # Freq
        fax = divider.append_axes("right", size="25%", pad=0.05)
        fax.grid("horizontal")
        for spine in fax.spines.values():
            spine.set_visible(False)
        fax.set_xticklabels([])
        fax.set_yticklabels([])
        fax.plot(agg_freq, tfr.freqs, color="k")
        fax.set_ylim([tfr.freqs[0] - 0.5, tfr.freqs[-1] - 0.5])
        fax.axhline(nf, color="r", linestyle="--")
        fax.text(0, nf, f"{nf:.1f} Hz", color="r", ha="left", va="bottom")

        # add colorbar on another axis
        cax = divider.append_axes("right", size="5%", pad=0.25)
        plt.colorbar(im, cax=cax)
        cax.set_ylabel("Power")
        return fig

    # ==== Complexity ====
    def compute_pcist(
        self,
        response: tuple = (0, 0.3),
        baseline: tuple = (None, 0),
        full_return: bool = False,
        params: dict = {
            "k": 1.2,
            "min_snr": 1.1,
            "max_var": 99,
            "embed": False,
            "n_steps": 100,
            "full_return": True,
        },
    ):
        _response = (
            response[0] if response[0] is not None else self.times[0],
            response[1] if response[1] is not None else self.times[-1],
        )
        _baseline = (
            baseline[0] if baseline[0] is not None else self.times[0],
            baseline[1] if baseline[1] is not None else self.times[-1],
        )
        params["response_window"] = _response
        params["baseline_window"] = _baseline
        pci_dict = pci_st.calc_PCIst(self.evoked.data, self.evoked.times, **params)
        return pci_dict if full_return else pci_dict["PCI"]

    # === General ===
    def ensure_prerequisites(
        self,
        run_tfr: bool = True,
        run_pci: bool = True,
        run_circshift: bool = True,
        tfr_kwargs: dict = None,
        tfr_mask_kwargs: dict = None,
        pci_kwargs: dict = None,
        circshift_kwargs: dict = None,
        verbose: bool = True,
    ):
        """
        Checks for and computes all necessary prerequisite data (derivatives).

        This method intelligently runs time-consuming computations only if the
        results are not already loaded or present in the derivatives dictionary.
        """
        if verbose:
            logger.info(f"\n--- Ensuring prerequisites for {self.label} ---")

        # Set default kwargs if None are provided
        tfr_kwargs = tfr_kwargs if tfr_kwargs is not None else {}
        tfr_mask_kwargs = tfr_mask_kwargs if tfr_mask_kwargs is not None else {}
        pci_kwargs = pci_kwargs if pci_kwargs is not None else {}
        circshift_kwargs = circshift_kwargs if circshift_kwargs is not None else {}

        # --- 1. TFR and ITC ---
        if run_tfr and self.derivatives.get("tfr") is None:
            if verbose:
                logger.info("TFR/ITC not found. Computing now...")
            self.compute_tfr_and_itc(**tfr_kwargs)
            self.save_tfr_and_itc(overwrite=True)  # Save the result

            # Masks must be computed after TFR
            if self.derivatives.get("tfr_mask") is None:
                if verbose:
                    logger.info("Computing TFR/ITC masks...")
                self.compute_tfr_and_itc_mask(**tfr_mask_kwargs)
                self.save_tfr_and_itc_mask(overwrite=True)  # Save the result
        elif run_tfr and verbose:
            logger.info("TFR/ITC and masks already loaded.")

        # --- 2. Circular Shift Test (for masked evoked) ---
        if run_circshift and self.derivatives.get("masked_ave") is None:
            if verbose:
                logger.info("Masked evoked not found. Running circ-shift test...")
            self.circshift_test_evoked(**circshift_kwargs)
        elif run_circshift and verbose:
            logger.info("Masked evoked already loaded.")

        # --- 3. PCIst ---
        if run_pci and "pci" not in self.derivatives:
            if verbose:
                logger.info("PCI not found. Computing now...")
            # We store the full dictionary and the scalar value separately
            pci_dict = self.compute_pcist(**pci_kwargs, full_return=True)
            self.derivatives["pci"] = pci_dict.get("PCI", None)
            self.derivatives["pci_full"] = pci_dict
        elif run_pci and verbose:
            logger.info("PCI already computed.")

        if verbose:
            logger.info("--- All prerequisites are met ---")

    def get_all_outcomes(
        self, tlim_dict: dict, picks: list | None = None, prereq_kwargs: dict = None
    ) -> dict:
        """
        Ensures all computations are run and then extracts all scalar outcome measures.

        Args:
            tlim_dict (dict): Dictionary defining time limits for different analyses.
            picks (list | None): List of channels for regional analysis.
            prereq_kwargs (dict): Kwargs to pass to the ensure_prerequisites method.

        Returns:
            dict: A dictionary containing all computed outcome measures.
        """
        prereq_kwargs = prereq_kwargs if prereq_kwargs is not None else {}

        # This is the key step: run the gatekeeper function first!
        self.ensure_prerequisites(**prereq_kwargs)

        outcomes = {}

        # Field Power
        outcomes["gfp_auc"] = self.get_field_power(
            tlim=tlim_dict.get("gfp", (None, None)), picks=picks
        )

        # Signal-to-Noise Ratio
        outcomes["snr"] = self.compute_snr(
            baseline=(-0.2, 0), response=tlim_dict.get("gfp", (0.01, 0.12))
        )

        # Phase Locking Factor
        outcomes["plf_auc"] = self.get_plf(
            tlim=tlim_dict.get("gfp", (None, None)), picks=picks, apply_mask=True
        )

        # Natural Frequency (now guaranteed to have TFR data)
        outcomes["natural_freq_hz"] = self.get_natural_frequency(
            tlim=tlim_dict.get("tfr", (None, None)), picks=picks, apply_mask=True
        )

        # Example: Average power in the Beta band
        outcomes["tfr_beta_power"] = self.get_tfr(
            tlim=tlim_dict.get("tfr", (None, None)),
            flim=(13, 30),
            picks=picks,
            apply_mask=True,
        )

        # PCI (now guaranteed to be computed)
        outcomes["pci_st"] = self.derivatives.get("pci", None)

        return outcomes

        # Phase Locking Factor
        outcomes["plf_auc"] = self.get_plf(
            tlim=tlim_dict.get("gfp", (None, None)), picks=picks, apply_mask=True
        )

        # Natural Frequency (now guaranteed to have TFR data)
        outcomes["natural_freq_hz"] = self.get_natural_frequency(
            tlim=tlim_dict.get("tfr", (None, None)), picks=picks, apply_mask=True
        )

        # Example: Average power in the Beta band
        outcomes["tfr_beta_power"] = self.get_tfr(
            tlim=tlim_dict.get("tfr", (None, None)),
            flim=(13, 30),
            picks=picks,
            apply_mask=True,
        )

        # PCI (now guaranteed to be computed)
        outcomes["pci_st"] = self.derivatives.get("pci", None)

        return outcomes
