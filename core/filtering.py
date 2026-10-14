"""Validated filter designs shared by the preprocessing steps."""

from dataclasses import asdict, dataclass, fields
from math import isfinite

import numpy as np


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive number.")


@dataclass(frozen=True)
class FilterConfig:
    method: str = "iir"
    phase: str = "zero"
    iir_order: int = 3
    iir_ftype: str = "butter"
    iir_rp: float = 1.0
    iir_rs: float = 40.0
    filter_length: str = "auto"
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    l_trans_bandwidth: float | str = "auto"
    h_trans_bandwidth: float | str = "auto"

    @classmethod
    def from_dict(cls, value=None, **defaults):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls(**defaults)
        if not isinstance(value, dict):
            raise ValueError("Filter settings must be a configuration object.")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown filter settings: {', '.join(sorted(unknown))}")
        return cls(**(defaults | value))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.method not in {"fir", "iir"}:
            raise ValueError("Filter method must be FIR or IIR.")
        phases = {"zero", "zero-double", "forward"} if self.method == "iir" else {
            "zero", "zero-double", "minimum"
        }
        if self.phase not in phases:
            raise ValueError(f"Phase '{self.phase}' is not supported for {self.method.upper()}.")
        if self.method == "iir":
            if type(self.iir_order) is not int or not 1 <= self.iir_order <= 20:
                raise ValueError("IIR order must be an integer from 1 to 20.")
            if self.iir_ftype not in {"butter", "bessel", "cheby1", "cheby2", "ellip"}:
                raise ValueError("Unsupported IIR filter family.")
            if self.iir_ftype in {"cheby1", "ellip"}:
                _positive(self.iir_rp, "Passband ripple")
            if self.iir_ftype in {"cheby2", "ellip"}:
                _positive(self.iir_rs, "Stopband attenuation")
        else:
            if self.fir_window not in {"hamming", "hann", "blackman"}:
                raise ValueError("Unsupported FIR window.")
            if self.fir_design not in {"firwin", "firwin2"}:
                raise ValueError("Unsupported FIR design.")
            if self.filter_length != "auto":
                if not isinstance(self.filter_length, str):
                    raise ValueError("FIR length must be 'auto' or a duration such as '2s'.")
                length = self.filter_length
                try:
                    if length.endswith("ms"):
                        duration = float(length[:-2]) / 1000
                    elif length.endswith("s"):
                        duration = float(length[:-1])
                    else:
                        raise ValueError
                    _positive(duration, "FIR length")
                except ValueError:
                    raise ValueError("FIR length must be 'auto' or a duration such as '2s'.") from None
            for value in (self.l_trans_bandwidth, self.h_trans_bandwidth):
                if value != "auto":
                    _positive(value, "Transition width")

    def kwargs(self, *, notch=False):
        self.validate()
        params = {"method": self.method, "phase": self.phase}
        if self.method == "iir":
            iir = {"order": self.iir_order, "ftype": self.iir_ftype, "output": "sos"}
            if self.iir_ftype in {"cheby1", "ellip"}:
                iir["rp"] = self.iir_rp
            if self.iir_ftype in {"cheby2", "ellip"}:
                iir["rs"] = self.iir_rs
            params["iir_params"] = iir
        else:
            params.update(filter_length=self.filter_length, fir_window=self.fir_window,
                          fir_design=self.fir_design)
            if not notch:
                params.update(l_trans_bandwidth=self.l_trans_bandwidth,
                              h_trans_bandwidth=self.h_trans_bandwidth)
        return params


@dataclass(frozen=True)
class NotchConfig:
    design: FilterConfig = FilterConfig(method="fir", iir_order=4)
    width: float | None = None
    trans_bandwidth: float = 1.0

    @classmethod
    def from_dict(cls, value=None):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - {"design", "width", "trans_bandwidth"}:
            raise ValueError("Invalid notch settings.")
        return cls(
            design=FilterConfig.from_dict(value.get("design"), method="fir", iir_order=4),
            width=value.get("width"),
            trans_bandwidth=value.get("trans_bandwidth", 1.0),
        )

    def to_dict(self):
        return asdict(self)

    def validate(self):
        self.design.validate()
        if self.width is not None:
            _positive(self.width, "Notch width")
        _positive(self.trans_bandwidth, "Notch transition width")


def validate_bandpass(bandpass, sfreq=None):
    if bandpass is None:
        return
    if not isinstance(bandpass, (list, tuple)) or len(bandpass) != 2:
        raise ValueError("Specify a high-pass and low-pass cutoff, or disable both.")
    low, high = bandpass
    for value in bandpass:
        if value is not None:
            _positive(value, "Filter cutoff")
            if sfreq is not None and value >= sfreq / 2:
                raise ValueError(f"Filter cutoffs must be below Nyquist ({sfreq / 2:g} Hz).")
    if low is not None and high is not None and low >= high:
        raise ValueError("The high-pass cutoff must be lower than the low-pass cutoff.")


def validate_resample(resample):
    if resample is not None:
        _positive(resample, "Resampling frequency")


def apply_bandpass(inst, bandpass, config=None):
    config = FilterConfig.from_dict(config)
    config.validate()
    validate_bandpass(bandpass)
    if bandpass is not None and any(value is not None for value in bandpass):
        validate_bandpass(bandpass, inst.info["sfreq"])
        inst.filter(l_freq=bandpass[0], h_freq=bandpass[1], **config.kwargs(), verbose=False)
    return config.to_dict()


def apply_notch(inst, notch, config=None):
    config = NotchConfig.from_dict(config)
    config.validate()
    if notch is None:
        return config.to_dict() | {"frequencies": []}
    _positive(notch, "Notch frequency")
    nyquist = inst.info["sfreq"] / 2
    if notch >= nyquist:
        raise ValueError(f"Notch frequency must be below Nyquist ({nyquist:g} Hz).")
    frequencies = np.arange(notch, nyquist, notch)
    widths = frequencies / 200 if config.width is None else config.width
    if np.any(frequencies - widths / 2 - config.trans_bandwidth / 2 <= 0) or np.any(
        frequencies + widths / 2 + config.trans_bandwidth / 2 >= nyquist
    ):
        raise ValueError("Notch bands and their transitions must fit between 0 Hz and Nyquist.")
    kwargs = config.design.kwargs(notch=True)
    # MNE supports multiple simultaneous stop bands only for FIR filters.
    groups = [frequencies] if config.design.method == "fir" else [[freq] for freq in frequencies]
    for group in groups:
        inst.notch_filter(group, notch_widths=config.width,
                          trans_bandwidth=config.trans_bandwidth, **kwargs, verbose=False)
    return config.to_dict() | {"frequencies": frequencies.tolist()}
