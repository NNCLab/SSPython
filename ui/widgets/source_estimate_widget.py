from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import apply_inverse, make_inverse_operator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.tools.optional_range_widget import OptionalRangeWidget
from utils import Worker

os.environ.setdefault("MNE_QT_BACKEND", "PySide6")


@dataclass(frozen=True)
class STCSourceConfig:
    mode: str
    subject: str = "fsaverage"
    subjects_dir: Path | None = None
    src: Path | None = None
    bem: Path | None = None
    trans: Path | str | None = "fsaverage"


@dataclass(frozen=True)
class STCSourceModel:
    mode: str
    subject: str
    subjects_dir: Path
    src: Path
    bem: Path
    trans: Path | str

    def to_metadata(self) -> dict[str, str | bool]:
        return {
            "mode": self.mode,
            "using_fsaverage": self.mode == "fsaverage",
            "subject": self.subject,
            "subjects_dir": str(self.subjects_dir),
            "src": str(self.src),
            "bem": str(self.bem),
            "trans": str(self.trans),
        }


RANK_VALUES: dict[str, str | None] = {
    "Default": None,
    "info": "info",
}

PICK_ORI_VALUES: dict[str, str | None] = {
    "None": None,
    "normal": "normal",
    "vector": "vector",
}


def parse_covariance_methods(text: str) -> str | list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Noise covariance method is required.")
    methods = [
        part.strip()
        for part in cleaned.replace(";", ",").split(",")
        if part.strip()
    ]
    if not methods:
        raise ValueError("Noise covariance method is required.")
    return methods[0] if len(methods) == 1 else methods


def metadata_to_source_config(metadata: dict[str, Any] | None) -> STCSourceConfig | None:
    if not metadata or not isinstance(metadata, dict):
        return None
    source = metadata.get("source", metadata)
    if not isinstance(source, dict):
        return None
    mode = str(source.get("mode") or "fsaverage")
    subject = str(source.get("subject") or "fsaverage")
    subjects_dir = _optional_path(source.get("subjects_dir"))
    src = _optional_path(source.get("src"))
    bem = _optional_path(source.get("bem"))
    trans_value = source.get("trans")
    trans = _optional_path(trans_value) if mode != "fsaverage" else "fsaverage"
    return STCSourceConfig(
        mode=mode,
        subject=subject,
        subjects_dir=subjects_dir,
        src=src,
        bem=bem,
        trans=trans,
    )


def prepare_stc_source_model(config: STCSourceConfig) -> STCSourceModel:
    if config.mode == "fsaverage":
        fs_dir = Path(fetch_fsaverage(verbose=True))
        subjects_dir = fs_dir.parent
        src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
        bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
        _ensure_file(src, "fsaverage source space")
        _ensure_file(bem, "fsaverage BEM solution")
        return STCSourceModel(
            mode="fsaverage",
            subject="fsaverage",
            subjects_dir=subjects_dir,
            src=src,
            bem=bem,
            trans="fsaverage",
        )

    subjects_dir = _ensure_directory(config.subjects_dir, "subjects directory")
    src = _ensure_file(config.src, "source space")
    bem = _ensure_file(config.bem, "BEM solution")
    if isinstance(config.trans, str) and config.trans == "fsaverage":
        trans: Path | str = "fsaverage"
    else:
        trans = _ensure_file(_optional_path(config.trans), "MRI transform")
    subject = (config.subject or "").strip()
    if not subject:
        raise ValueError("Subject name is required for custom source files.")
    return STCSourceModel(
        mode="custom",
        subject=subject,
        subjects_dir=subjects_dir,
        src=src,
        bem=bem,
        trans=trans,
    )


def compute_stc(
    epochs: mne.BaseEpochs,
    *,
    source_config: STCSourceConfig,
    condition: str | None,
    pick_eeg: bool,
    set_eeg_reference: bool,
    noise_tmin: float | None,
    noise_tmax: float | None,
    noise_method: str | list[str],
    rank: str | None,
    mindist: float,
    n_jobs: int | None,
    loose: float,
    depth: float,
    inverse_method: str,
    snr: float,
    pick_ori: str | None,
    verbose: bool = True,
) -> tuple[mne.SourceEstimate, STCSourceModel]:
    source_model = prepare_stc_source_model(source_config)
    analysis_epochs = epochs[condition].copy() if condition else epochs.copy()
    if len(analysis_epochs) == 0:
        raise ValueError("No epochs are available for the selected condition.")

    if pick_eeg:
        analysis_epochs.pick("eeg")
    if len(analysis_epochs.ch_names) == 0:
        raise ValueError("No channels are available after applying channel picks.")

    if set_eeg_reference and "eeg" in set(analysis_epochs.get_channel_types()):
        analysis_epochs.set_eeg_reference(projection=True, verbose=verbose)

    noise_cov = mne.compute_covariance(
        analysis_epochs,
        tmin=noise_tmin,
        tmax=noise_tmax,
        method=noise_method,
        rank=rank,
        verbose=verbose,
    )
    evoked = analysis_epochs.average()
    forward = mne.make_forward_solution(
        analysis_epochs.info,
        trans=source_model.trans,
        src=source_model.src,
        bem=source_model.bem,
        meg=False,
        eeg=True,
        mindist=mindist,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    inverse_operator = make_inverse_operator(
        analysis_epochs.info,
        forward,
        noise_cov,
        loose=loose,
        depth=depth,
        rank=rank,
        verbose=verbose,
    )
    stc = apply_inverse(
        evoked,
        inverse_operator,
        1.0 / snr**2,
        method=inverse_method,
        pick_ori=pick_ori,
        verbose=verbose,
    )
    stc.subject = source_model.subject
    return stc, source_model


class ComputeSTCSettingsDialog(QDialog):
    """Collects parameters for source time course computation."""

    def __init__(self, epochs: mne.BaseEpochs, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.setWindowTitle("Compute Source Time Course")
        self.setModal(False)
        self.setMinimumWidth(720)

        layout = QVBoxLayout(self)
        layout.addWidget(self._create_source_group())
        layout.addWidget(self._create_data_group())
        layout.addWidget(self._create_forward_inverse_group())

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Compute")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.fsaverage_radio.toggled.connect(self._sync_source_mode)
        self.subjects_dir_button.clicked.connect(self._choose_subjects_dir)
        self.src_button.clicked.connect(
            lambda: self._choose_file(self.src_input, "Select Source Space")
        )
        self.bem_button.clicked.connect(
            lambda: self._choose_file(self.bem_input, "Select BEM Solution")
        )
        self.trans_button.clicked.connect(
            lambda: self._choose_file(self.trans_input, "Select MRI Transform")
        )
        self.alignment_button.clicked.connect(self.plot_coregistration_alignment)
        self.n_jobs_checkbox.toggled.connect(self.n_jobs_input.setEnabled)
        self._sync_source_mode()

    def _create_source_group(self) -> QGroupBox:
        group = QGroupBox("Source Model")
        layout = QVBoxLayout(group)

        self.fsaverage_radio = QRadioButton("Use provided fsaverage")
        self.custom_radio = QRadioButton("Use custom source files")
        self.fsaverage_radio.setChecked(True)
        layout.addWidget(self.fsaverage_radio)
        layout.addWidget(self.custom_radio)

        form = QFormLayout()
        self.subject_input = QLineEdit("fsaverage")
        self.subjects_dir_row, self.subjects_dir_input, self.subjects_dir_button = self._path_row("Select")
        self.src_row, self.src_input, self.src_button = self._path_row("Select")
        self.bem_row, self.bem_input, self.bem_button = self._path_row("Select")
        self.trans_row, self.trans_input, self.trans_button = self._path_row("Select")
        self.trans_input.setPlaceholderText("Transform FIF file")

        form.addRow("subject:", self.subject_input)
        form.addRow("subjects_dir:", self.subjects_dir_row)
        form.addRow("src:", self.src_row)
        form.addRow("bem:", self.bem_row)
        form.addRow("trans:", self.trans_row)
        layout.addLayout(form)

        self.alignment_button = QPushButton("Plot Coregistration Alignment")
        layout.addWidget(self.alignment_button)
        return group

    def _create_data_group(self) -> QGroupBox:
        group = QGroupBox("Data and Noise Covariance")
        form = QFormLayout(group)

        self.condition_combo = QComboBox()
        self.condition_combo.addItem("All epochs", None)
        for event_name in self.epochs.event_id or {}:
            self.condition_combo.addItem(event_name, event_name)

        self.pick_eeg_checkbox = QCheckBox("Use EEG channels only")
        self.pick_eeg_checkbox.setChecked(True)
        self.reference_checkbox = QCheckBox("Set average EEG reference projection")
        self.reference_checkbox.setChecked(True)

        times = self.epochs.times
        self.noise_window_input = OptionalRangeWidget(
            labels=("tmin:", "tmax:"),
            suffix=" s",
            range=(float(times[0]), float(times[-1])) if len(times) else (None, None),
            parent=self,
        )
        self.noise_window_input.setValue((None, 0.0))
        for spinbox in (
            self.noise_window_input.low_input,
            self.noise_window_input.high_input,
        ):
            spinbox.setDecimals(4)
            spinbox.setSingleStep(0.01)
            if len(times):
                spinbox.setRange(float(times[0]), float(times[-1]))
        self.noise_window_input.adjust_spinbox_width()

        self.noise_method_input = QLineEdit("shrunk, empirical")
        self.noise_method_input.setPlaceholderText("e.g. shrunk, empirical")
        self.rank_combo = QComboBox()
        self.rank_combo.addItems(list(RANK_VALUES.keys()))

        form.addRow("condition:", self.condition_combo)
        form.addRow(self.pick_eeg_checkbox)
        form.addRow(self.reference_checkbox)
        form.addRow("noise window:", self.noise_window_input)
        form.addRow("noise method:", self.noise_method_input)
        form.addRow("rank:", self.rank_combo)
        return group

    def _create_forward_inverse_group(self) -> QGroupBox:
        group = QGroupBox("Forward and Inverse Solution")
        form = QFormLayout(group)

        self.mindist_input = QDoubleSpinBox()
        self.mindist_input.setRange(0.0, 100.0)
        self.mindist_input.setDecimals(2)
        self.mindist_input.setValue(5.0)
        self.mindist_input.setSuffix(" mm")

        self.n_jobs_checkbox = QCheckBox("Set")
        self.n_jobs_input = QSpinBox()
        self.n_jobs_input.setRange(-1, 256)
        self.n_jobs_input.setValue(-1)
        self.n_jobs_input.setEnabled(False)
        n_jobs_row = QWidget()
        n_jobs_layout = QHBoxLayout(n_jobs_row)
        n_jobs_layout.setContentsMargins(0, 0, 0, 0)
        n_jobs_layout.addWidget(self.n_jobs_checkbox)
        n_jobs_layout.addWidget(self.n_jobs_input)
        n_jobs_layout.addStretch()

        self.loose_input = QDoubleSpinBox()
        self.loose_input.setRange(0.0, 1.0)
        self.loose_input.setDecimals(2)
        self.loose_input.setSingleStep(0.05)
        self.loose_input.setValue(0.2)

        self.depth_input = QDoubleSpinBox()
        self.depth_input.setRange(0.0, 1.0)
        self.depth_input.setDecimals(2)
        self.depth_input.setSingleStep(0.05)
        self.depth_input.setValue(0.8)

        self.inverse_method_combo = QComboBox()
        self.inverse_method_combo.addItems(["dSPM", "MNE", "sLORETA", "eLORETA"])

        self.snr_input = QDoubleSpinBox()
        self.snr_input.setRange(0.01, 1000.0)
        self.snr_input.setDecimals(2)
        self.snr_input.setValue(3.0)

        self.pick_ori_combo = QComboBox()
        self.pick_ori_combo.addItems(list(PICK_ORI_VALUES.keys()))

        form.addRow("mindist:", self.mindist_input)
        form.addRow("n_jobs:", n_jobs_row)
        form.addRow("loose:", self.loose_input)
        form.addRow("depth:", self.depth_input)
        form.addRow("method:", self.inverse_method_combo)
        form.addRow("SNR:", self.snr_input)
        form.addRow("pick_ori:", self.pick_ori_combo)
        return group

    def _path_row(self, button_text: str) -> tuple[QWidget, QLineEdit, QPushButton]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit()
        button = QPushButton(button_text)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row, line_edit, button

    def _choose_subjects_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select subjects_dir")
        if folder:
            self.subjects_dir_input.setText(folder)

    def _choose_file(self, line_edit: QLineEdit, title: str):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "FIF files (*.fif *.fif.gz);;All files (*)",
        )
        if file_path:
            line_edit.setText(file_path)

    def _sync_source_mode(self):
        use_custom = self.custom_radio.isChecked()
        self.subject_input.setEnabled(use_custom)
        for widget in (
            self.subjects_dir_input,
            self.subjects_dir_button,
            self.src_input,
            self.src_button,
            self.bem_input,
            self.bem_button,
            self.trans_input,
            self.trans_button,
        ):
            widget.setEnabled(use_custom)
        if not use_custom:
            self.subject_input.setText("fsaverage")

    def get_source_config(self) -> STCSourceConfig:
        if self.fsaverage_radio.isChecked():
            return STCSourceConfig(mode="fsaverage")
        return STCSourceConfig(
            mode="custom",
            subject=self.subject_input.text().strip(),
            subjects_dir=_required_path_text(self.subjects_dir_input.text(), "subjects_dir"),
            src=_required_path_text(self.src_input.text(), "source space"),
            bem=_required_path_text(self.bem_input.text(), "BEM solution"),
            trans=_required_path_text(self.trans_input.text(), "MRI transform"),
        )

    def get_compute_params(self) -> dict[str, Any]:
        noise_tmin, noise_tmax = self.noise_window_input.value()
        n_jobs = int(self.n_jobs_input.value()) if self.n_jobs_checkbox.isChecked() else None
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0.")
        return {
            "source_config": self.get_source_config(),
            "condition": self.condition_combo.currentData(),
            "pick_eeg": self.pick_eeg_checkbox.isChecked(),
            "set_eeg_reference": self.reference_checkbox.isChecked(),
            "noise_tmin": noise_tmin,
            "noise_tmax": noise_tmax,
            "noise_method": parse_covariance_methods(self.noise_method_input.text()),
            "rank": RANK_VALUES[self.rank_combo.currentText()],
            "mindist": float(self.mindist_input.value()),
            "n_jobs": n_jobs,
            "loose": float(self.loose_input.value()),
            "depth": float(self.depth_input.value()),
            "inverse_method": self.inverse_method_combo.currentText(),
            "snr": float(self.snr_input.value()),
            "pick_ori": PICK_ORI_VALUES[self.pick_ori_combo.currentText()],
        }

    def _validation_error(self) -> str | None:
        try:
            params = self.get_compute_params()
            if params["snr"] <= 0:
                return "SNR must be greater than 0."
            source_config = params["source_config"]
            if source_config.mode == "custom":
                prepare_stc_source_model(source_config)
        except ValueError as exc:
            return str(exc)
        return None

    def accept(self):
        error = self._validation_error()
        if error:
            QMessageBox.warning(self, "Invalid STC Parameters", error)
            return
        super().accept()

    def plot_coregistration_alignment(self):
        try:
            source_config = self.get_source_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Source Model", str(exc))
            return

        worker = Worker(
            lambda: prepare_stc_source_model(source_config),
            parent=self,
            add_loggers="mne",
        )
        source_model = worker.exec_with_dialog(
            "Please wait",
            "Preparing source model files...",
        )
        if source_model is None:
            return

        try:
            mne.viz.plot_alignment(
                self.epochs.info,
                trans=source_model.trans,
                subject=source_model.subject,
                subjects_dir=source_model.subjects_dir,
                src=source_model.src,
                bem=source_model.bem,
                meg=False,
                eeg=["original", "projected"],
                dig="fiducials",
                mri_fiducials=True,
                show_axes=True,
                verbose=True,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Coregistration Plot Failed",
                str(exc),
            )


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _required_path_text(text: str, label: str) -> Path:
    path = _optional_path(text)
    if path is None:
        raise ValueError(f"{label} is required.")
    return path


def _ensure_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required.")
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _ensure_directory(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required.")
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{label} does not exist: {path}")
    return path
