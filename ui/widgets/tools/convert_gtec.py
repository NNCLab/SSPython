import sys
import json
from pathlib import Path
from scipy.io import loadmat
import mat73
import mne
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from utils import Worker
import logging

logger = logging.getLogger(__file__)


class GtecConverter(QDialog):
    """
    A PySide6 application to convert gTEC EEG data (.mat) to .fif format,
    adding a channel montage from a -dig.fif file.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("gTEC to FIF Converter")
        self.setMinimumWidth(500)
        self.setModal(True)

        # File path storage
        self.mat_path = ""
        self.montage_path = ""
        self.fif_path = ""

        # --- UI Elements ---
        # Layouts
        layout = QVBoxLayout()
        h_layout1 = QVBoxLayout()
        h_layout2 = QVBoxLayout()
        h_layout3 = QVBoxLayout()

        # Input file selection
        self.mat_label = QLineEdit("No .mat file selected...")
        self.mat_label.setReadOnly(True)
        self.mat_label.setAlignment(Qt.AlignCenter)
        self.mat_button = QPushButton("Select EEG (.mat)")

        # chlocs file selection
        self.chlocs_label = QLineEdit("No chlocs .json file selected...")
        self.chlocs_label.setReadOnly(True)
        self.chlocs_label.setAlignment(Qt.AlignCenter)
        self.chlocs_button = QPushButton("Select ch locs (*.json))")

        # Output file selection
        self.fif_label = QLineEdit("No output .fif file selected...")
        self.fif_label.setReadOnly(True)
        self.fif_label.setAlignment(Qt.AlignCenter)
        self.fif_button = QPushButton("Set Output (.fif)")

        # Convert button
        self.convert_button = QPushButton("Convert to .fif")
        self.convert_button.setStyleSheet("font-weight: bold;")
        self.convert_button.setDefault(True)

        # --- Layout Setup ---
        h_layout1.addWidget(self.mat_label)
        h_layout1.addWidget(self.mat_button)
        layout.addLayout(h_layout1)

        h_layout2.addWidget(self.chlocs_label)
        h_layout2.addWidget(self.chlocs_button)
        layout.addLayout(h_layout2)

        h_layout3.addWidget(self.fif_label)
        h_layout3.addWidget(self.fif_button)
        layout.addLayout(h_layout3)

        layout.addSpacing(20)
        layout.addWidget(self.convert_button)

        # Watermark
        watermark = QLabel("Couto B.A.N. (2025)")
        watermark.setAlignment(Qt.AlignRight)
        watermark.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(watermark)

        self.setLayout(layout)

        # --- Connections ---
        self.mat_button.clicked.connect(self.select_mat_file)
        self.chlocs_button.clicked.connect(self.select_chlocs_file)
        self.fif_button.clicked.connect(self.select_fif_file)
        self.convert_button.clicked.connect(self.run_conversion)

    def select_mat_file(self):
        """Opens a file dialog to select a BrainAmp .mat file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select gTEC Save File", "", "Matlab File (*.mat)"
        )
        if path:
            self.mat_path = Path(path)
            self.mat_label.setText(self.mat_path.stem)
            self.mat_label.setToolTip(str(self.mat_path))

    def select_chlocs_file(self):
        """Opens a file dialog to select a .chlocs chlocs file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select chlocs File", "", "chlocs file (*.json)"
        )
        if path:
            self.chlocs_path = Path(path)
            self.chlocs_label.setText(self.chlocs_path.stem)
            self.chlocs_label.setToolTip(str(self.chlocs_path))

    def select_fif_file(self):
        """Opens a file dialog to set the output .fif file path."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FIF File", "", "FIF file (*.fif)"
        )
        if path:
            path = Path(path)
            self.fif_path = path.parent / (path.stem + "_raw.fif")
            self.fif_label.setText(self.fif_path.stem)
            self.fif_label.setToolTip(str(self.fif_path))

    @staticmethod
    def _run_conversion(mat_path, chlocs_path, fif_path, parent=None):
        logger.info("Reading .mat file...")
        try:
            data = loadmat(mat_path)
        except:
            try:
                data = mat73.loadmat(mat_path)
            except Exception as e:
                QMessageBox.critical(
                    parent, "Warning", "Could not load file. Please check file format."
                )

        sfreq = data["SR"]
        raw_array = data["y"]

        logger.info("Reading chlocs file...")
        with open(chlocs_path, "r") as f:
            chlocs = json.load(f)
        ch_names = chlocs.get("ch_names", None)
        ch_types = chlocs.get("ch_types", None)
        if ch_names is None or ch_types is None:
            raise ValueError(
                "Invalid chlocs file format. Missing 'ch_names' or 'ch_types'."
            )
        eeg_idx = [i for i, ch_type in enumerate(ch_types) if ch_type == "eeg"]
        raw_array[eeg_idx, :] = raw_array[eeg_idx, :] * 1e-6

        logger.info("Creating .fif file...")
        montage = mne.channels.make_standard_montage("easycap-M1")
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(raw_array, info)
        raw.set_montage(montage, on_missing="warn")
        events = mne.find_events(raw)
        if len(events) > 0:
            event_desc = lambda x: f"STIM {x}"
            annot_from_events = mne.annotations_from_events(
                events=events, sfreq=raw.info["sfreq"], event_desc=event_desc
            )
            raw.set_annotations(annot_from_events)
        raw.save(fif_path, overwrite=True)
        return raw

    def run_conversion(self):
        """Executes the conversion process using the selected file paths."""
        if not all([self.mat_path, self.chlocs_path, self.fif_path]):
            QMessageBox.critical(None, "Error", "Please, select all needed files...")
            return
        worker = Worker(
            lambda: self._run_conversion(
                self.mat_path, self.chlocs_path, self.fif_path, self
            ),
            parent=self,
        )
        result = worker.exec_with_dialog(
            "Loading", "Converting to .fif... please wait..."
        )
        if result:
            QMessageBox.information(
                None, "Success", "Conversion completed successfully!"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = GtecConverter()
    dialog.exec()
    sys.exit(app.exec())
