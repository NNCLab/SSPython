from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox, QWidget,
)

from core.filtering import FilterConfig, NotchConfig
from .advanced_settings import AdvancedSettings


def _combo(items):
    widget = QComboBox()
    for label, value in items:
        widget.addItem(label, value)
    return widget


class FilterDesignWidget(QWidget):
    """One editor for the design parameters accepted by Raw and Epochs filters."""

    def __init__(self, *, notch=False, parent=None):
        super().__init__(parent)
        self.notch = notch
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.method_input = _combo([("IIR", "iir"), ("FIR", "fir")])
        self.phase_input = QComboBox()
        self.form.addRow("Method:", self.method_input)
        self.form.addRow("Phase:", self.phase_input)
        self.order_input = QSpinBox()
        self.order_input.setRange(1, 20)
        self.family_input = _combo([
            ("Butterworth", "butter"), ("Bessel", "bessel"),
            ("Chebyshev I", "cheby1"), ("Chebyshev II", "cheby2"), ("Elliptic", "ellip"),
        ])
        self.ripple_input = QDoubleSpinBox()
        self.attenuation_input = QDoubleSpinBox()
        for widget in (self.ripple_input, self.attenuation_input):
            widget.setRange(0.01, 200)
            widget.setSuffix(" dB")
        self.form.addRow("IIR family:", self.family_input)
        self.form.addRow("IIR order:", self.order_input)
        self.form.addRow("Passband ripple:", self.ripple_input)
        self.form.addRow("Stopband attenuation:", self.attenuation_input)
        self.length_input = QLineEdit()
        self.length_input.setPlaceholderText("auto, 2s, or 500ms")
        self.window_input = _combo([(name.title(), name) for name in ("hamming", "hann", "blackman")])
        self.design_input = _combo([("firwin", "firwin"), ("firwin2", "firwin2")])
        self.low_transition_input = QLineEdit()
        self.high_transition_input = QLineEdit()
        self.form.addRow("FIR length:", self.length_input)
        self.form.addRow("FIR window:", self.window_input)
        self.form.addRow("FIR design:", self.design_input)
        if not notch:
            for label, widget in (("High-pass transition (Hz):", self.low_transition_input),
                                  ("Low-pass transition (Hz):", self.high_transition_input)):
                widget.setPlaceholderText("auto or Hz")
                self.form.addRow(label, widget)
        self.method_input.currentIndexChanged.connect(self._update_method)
        self.family_input.currentIndexChanged.connect(self._update_rows)
        self.set_config(FilterConfig(method="fir", iir_order=4) if notch else FilterConfig())

    def _update_method(self):
        phase = self.phase_input.currentData()
        self.phase_input.clear()
        phases = [("Zero phase", "zero"), ("Zero phase (double)", "zero-double")]
        phases.append(("Forward (causal)", "forward") if self.method_input.currentData() == "iir"
                      else ("Minimum phase", "minimum"))
        for label, value in phases:
            self.phase_input.addItem(label, value)
        self.phase_input.setCurrentIndex(max(0, self.phase_input.findData(phase)))
        self._update_rows()

    def _update_rows(self):
        iir = self.method_input.currentData() == "iir"
        family = self.family_input.currentData()
        for widget in (self.family_input, self.order_input):
            self.form.setRowVisible(widget, iir)
        self.form.setRowVisible(self.ripple_input, iir and family in {"cheby1", "ellip"})
        self.form.setRowVisible(self.attenuation_input, iir and family in {"cheby2", "ellip"})
        for widget in (self.length_input, self.window_input, self.design_input):
            self.form.setRowVisible(widget, not iir)
        if not self.notch:
            for widget in (self.low_transition_input, self.high_transition_input):
                self.form.setRowVisible(widget, not iir)

    def set_config(self, config):
        config = FilterConfig.from_dict(config)
        self.method_input.setCurrentIndex(self.method_input.findData(config.method))
        self._update_method()
        self.phase_input.setCurrentIndex(self.phase_input.findData(config.phase))
        self.family_input.setCurrentIndex(self.family_input.findData(config.iir_ftype))
        self.order_input.setValue(config.iir_order)
        self.ripple_input.setValue(config.iir_rp)
        self.attenuation_input.setValue(config.iir_rs)
        self.length_input.setText(str(config.filter_length))
        self.window_input.setCurrentIndex(self.window_input.findData(config.fir_window))
        self.design_input.setCurrentIndex(self.design_input.findData(config.fir_design))
        self.low_transition_input.setText(str(config.l_trans_bandwidth))
        self.high_transition_input.setText(str(config.h_trans_bandwidth))
        self._update_rows()

    def get_config(self):
        def transition(widget, *, active):
            text = widget.text().strip().lower()
            if text == "auto":
                return text
            try:
                return float(text)
            except ValueError:
                if not active:
                    return "auto"
                raise ValueError("Transition width must be 'auto' or a positive number in Hz.") from None

        fir = self.method_input.currentData() == "fir"
        config = FilterConfig(
            method=self.method_input.currentData(), phase=self.phase_input.currentData(),
            iir_order=self.order_input.value(), iir_ftype=self.family_input.currentData(),
            iir_rp=self.ripple_input.value(), iir_rs=self.attenuation_input.value(),
            filter_length=self.length_input.text().strip().lower(),
            fir_window=self.window_input.currentData(), fir_design=self.design_input.currentData(),
            l_trans_bandwidth=transition(self.low_transition_input, active=fir) if not self.notch else "auto",
            h_trans_bandwidth=transition(self.high_transition_input, active=fir) if not self.notch else "auto",
        )
        config.validate()
        return config


class FilterSettingsWidget(AdvancedSettings):
    def __init__(self, *, notch=False, parent=None):
        super().__init__("Advanced filter settings", parent)
        self.bandpass_design = FilterDesignWidget()
        group = QGroupBox("High-pass / low-pass design")
        layout = QFormLayout(group)
        layout.addRow(self.bandpass_design)
        self.form.addRow(group)
        self.notch_design = None
        if notch:
            self.notch_design = FilterDesignWidget(notch=True)
            group = QGroupBox("Notch design")
            layout = QFormLayout(group)
            layout.addRow(self.notch_design)
            self.notch_width_input = QLineEdit()
            self.notch_width_input.setPlaceholderText("auto or Hz")
            self.notch_transition_input = QDoubleSpinBox()
            self.notch_transition_input.setRange(0.01, 1000)
            self.notch_transition_input.setSuffix(" Hz")
            layout.addRow("Notch width (Hz):", self.notch_width_input)
            layout.addRow("Transition width:", self.notch_transition_input)
            self.form.addRow(group)
        self.set_params({})

    def set_params(self, params):
        self.bandpass_design.set_config(FilterConfig.from_dict(params.get("filter_config")))
        if self.notch_design is not None:
            config = NotchConfig.from_dict(params.get("notch_config"))
            self.notch_design.set_config(config.design)
            self.notch_width_input.setText("auto" if config.width is None else str(config.width))
            self.notch_transition_input.setValue(config.trans_bandwidth)

    def get_params(self):
        params = {"filter_config": self.bandpass_design.get_config().to_dict()}
        if self.notch_design is not None:
            text = self.notch_width_input.text().strip().lower()
            try:
                width = None if text in {"", "auto"} else float(text)
            except ValueError:
                raise ValueError("Notch width must be 'auto' or a positive number in Hz.") from None
            config = NotchConfig(self.notch_design.get_config(), width,
                                 self.notch_transition_input.value())
            config.validate()
            params["notch_config"] = config.to_dict()
        return params
