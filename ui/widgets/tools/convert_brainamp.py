import sys
from pathlib import Path
import mne
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt
from utils import Worker


class BrainampConverter(QDialog):
    """
    A PySide6 application to convert BrainAmp EEG data (.vhdr) to .fif format,
    applying a standard 'easycap-M1' montage and allowing channel renaming.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BrainAmp Converter")
        self.setMinimumSize(500, 600)

        # File path storage
        self.vhdr_path = ""
        self.fif_path = ""

        # --- UI Elements ---
        layout = QVBoxLayout()
        h_layout1 = QHBoxLayout()
        h_layout3 = QHBoxLayout()

        # Input file selection
        self.vhdr_label = QLineEdit("No .vhdr file selected...")
        self.vhdr_label.setReadOnly(True)
        self.vhdr_button = QPushButton("Select EEG (.vhdr)")

        # Output file selection
        self.fif_label = QLineEdit("No output .fif file selected...")
        self.fif_label.setReadOnly(True)
        self.fif_button = QPushButton("Set Output (.fif)")

        # Channel Renaming Table
        self.rename_label = QLabel(
            "Rename Channels (edit names in the 'New Name' column):"
        )
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(2)
        self.channel_table.setHorizontalHeaderLabels(["Old Name", "New Name"])
        self.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Convert button
        self.convert_button = QPushButton("Convert to .fif")
        self.convert_button.setStyleSheet("font-weight: bold;")
        self.convert_button.setDefault(True)

        # --- Layout Setup ---
        h_layout1.addWidget(self.vhdr_label)
        h_layout1.addWidget(self.vhdr_button)
        layout.addLayout(h_layout1)

        h_layout3.addWidget(self.fif_label)
        h_layout3.addWidget(self.fif_button)
        layout.addLayout(h_layout3)

        # Add renaming widget to layout
        layout.addSpacing(15)
        layout.addWidget(self.rename_label)
        layout.addWidget(self.channel_table)

        layout.addSpacing(20)
        layout.addWidget(self.convert_button)

        # Watermark
        watermark = QLabel("Couto B.A.N. (2025)")
        watermark.setAlignment(Qt.AlignRight)
        watermark.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(watermark)

        self.setLayout(layout)

        # --- Connections ---
        self.vhdr_button.clicked.connect(self.select_vhdr_file)
        self.fif_button.clicked.connect(self.select_fif_file)
        self.convert_button.clicked.connect(self.run_conversion)

    def select_vhdr_file(self):
        """Opens a file dialog to select a .vhdr file and populates the channel table."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select BrainAmp Header File", "", "BrainVision VHDR (*.vhdr)"
        )
        if not path:
            return

        self.vhdr_path = Path(path)
        self.vhdr_label.setText(self.vhdr_path.name)
        self.vhdr_label.setToolTip(str(self.vhdr_path))

        try:
            info = mne.io.read_raw_brainvision(self.vhdr_path, preload=False).info
            ch_names = info["ch_names"]

            self.channel_table.setRowCount(0)
            self.channel_table.setRowCount(len(ch_names))

            for i, ch_name in enumerate(ch_names):
                old_name_item = QTableWidgetItem(ch_name)
                old_name_item.setFlags(old_name_item.flags() & ~Qt.ItemIsEditable)
                new_name_item = QTableWidgetItem("")

                self.channel_table.setItem(i, 0, old_name_item)
                self.channel_table.setItem(i, 1, new_name_item)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Reading File",
                f"Could not read channel info from the selected file.\n\nError: {e}",
            )
            self.channel_table.setRowCount(0)

    def select_fif_file(self):
        """Opens a file dialog to set the output .fif file path."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FIF File", "", "FIF file (*.fif)"
        )
        if path:
            path = Path(path)
            if not path.name.endswith("_raw.fif"):
                path = path.with_name(path.stem + "_raw.fif")
            self.fif_path = path
            self.fif_label.setText(self.fif_path.name)
            self.fif_label.setToolTip(str(self.fif_path))

    def run_conversion(self):
        """Validates input and executes the conversion process."""
        if not all([self.vhdr_path, self.fif_path]):
            QMessageBox.critical(
                self,
                "Error",
                "Please select an input EEG file and an output file path.",
            )
            return

        rename_map = {}
        for row in range(self.channel_table.rowCount()):
            old_name = self.channel_table.item(row, 0).text()
            new_name_item = self.channel_table.item(row, 1)

            if new_name_item and new_name_item.text().strip():
                new_name = new_name_item.text().strip()
                rename_map[old_name] = new_name

        worker = Worker(
            lambda: self._run_conversion(self.vhdr_path, self.fif_path, rename_map),
            parent=self,
        )
        result = worker.exec_with_dialog(
            "Processing...", "Converting to .fif, please wait..."
        )

        if result:
            success, message = result
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Conversion Failed", message)

    @staticmethod
    def _run_conversion(vhdr_path, fif_path, rename_map):
        """
        Performs data conversion, renaming, and applies the standard montage.
        """
        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

            if rename_map:
                raw.rename_channels(rename_map)

            for ch in raw.ch_names:
                if "EOG" in ch:
                    raw.set_channel_types({ch: "eog"})

            # --- New: Apply standard montage with specific error handling ---
            try:
                montage = mne.channels.make_standard_montage("easycap-M1")
                # Use on_missing='raise' to catch mismatches
                raw.set_montage(montage, on_missing="raise")
            except ValueError as e:
                # This error occurs if channels in `raw` don't match the montage
                error_msg = (
                    f"Failed to set 'easycap-M1' montage.\n\n"
                    f"Error: {e}\n\n"
                    "Please use the renaming tool to match your channel names to the "
                    "standard montage (e.g., 'Fp1', 'Fz', 'P7', etc.)."
                )
                return (False, error_msg)

            raw.save(fif_path, overwrite=True)
            return (True, f"Conversion successful!\n\nFile saved to:\n{fif_path}")
        except Exception as e:
            return (False, f"An unexpected error occurred:\n\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrainampConverter()
    window.show()
    sys.exit(app.exec())
