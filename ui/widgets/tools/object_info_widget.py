import sys
import mne
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QScrollArea,
    QGroupBox,
    QPushButton,
    QHBoxLayout,
)
from PySide6.QtCore import Qt
from typing import Union, Optional


class ObjectInfoWidget(QWidget):
    """
    A custom widget to dynamically display information from MNE objects.

    This widget can be updated at any time with a new MNE object
    (mne.io.Raw, mne.Epochs, mne.preprocessing.ICA) and an optional
    custom title.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the widget structure. The widget starts empty.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.mne_object = None
        self.object_type = "Unknown"
        self._init_ui()

    def _init_ui(self):
        """Initializes the static user interface elements of the widget."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main container group box, title will be set dynamically
        self.main_info_group = QGroupBox("No object loaded")
        main_layout.addWidget(self.main_info_group)

        # A container layout inside the main group box to hold everything
        container_layout = QVBoxLayout(self.main_info_group)

        # **CORRECTION**: Create a dedicated layout for basic info that we can clear
        self.basic_info_layout = QGridLayout()
        container_layout.addLayout(self.basic_info_layout)

        # The details group box is a permanent part of the structure
        self.details_group_box = QGroupBox("Detailed Information")
        self.details_group_box.setCheckable(True)
        self.details_group_box.setChecked(False)
        # **CORRECTION**: Add it to the permanent container layout
        container_layout.addWidget(self.details_group_box)

        # Scroll area for the details (unchanged)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(200)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        details_layout = QVBoxLayout(self.details_group_box)
        details_layout.setContentsMargins(2, 5, 2, 2)
        details_layout.addWidget(self.scroll_area)

        # Connect toggle Signal and set initial visibility (unchanged)
        self.details_group_box.toggled.connect(self.scroll_area.setVisible)
        self.scroll_area.setVisible(False)

        # Start with the entire widget hidden until an object is loaded
        self.main_info_group.setVisible(False)

    def update_info(
        self,
        mne_object: Optional[Union[mne.io.Raw, mne.Epochs, mne.preprocessing.ICA]],
        title: Optional[str] = None,
    ):
        """
        Updates the widget to display information for a new MNE object.
        """
        if mne_object is None:
            self.clear_info()
            return

        self.mne_object = mne_object
        self._determine_object_type()

        # **CORRECTION**: Clear only the basic info layout
        self._clear_layout(self.basic_info_layout)

        display_title = title if title is not None else f"{self.object_type} Info:"
        self.main_info_group.setTitle(display_title)

        self._populate_content()
        self.main_info_group.setVisible(True)

    def clear_info(self):
        """Clears all information and hides the widget."""
        self.mne_object = None
        self.object_type = "Unknown"
        self._clear_layout(self.basic_info_layout)  # Clear the basic info
        self.scroll_area.setWidget(None)
        self.main_info_group.setTitle("No object loaded")
        self.main_info_group.setVisible(False)

    def _populate_content(self):
        """Populates both basic and detailed info sections based on the object type."""
        scroll_content = QWidget()  # Default empty widget

        # **CORRECTION**: Populate the dedicated basic_info_layout
        if self.object_type == "Raw":
            self._populate_raw_basics(self.basic_info_layout)
            scroll_content = self._create_dict_widget(self.mne_object.info)
        elif self.object_type == "Epochs":
            self._populate_epochs_basics(self.basic_info_layout)
            scroll_content = self._create_dict_widget(self.mne_object.info)
        elif self.object_type == "ICA":
            self._populate_ica_basics(self.basic_info_layout)
            ica_attrs = {
                "noise_cov": self.mne_object.noise_cov,
                "ch_names": self.mne_object.ch_names,
                "n_pca_components": self.mne_object.n_pca_components,
                "exclusions": self.mne_object.exclude,
            }
            scroll_content = self._create_dict_widget(ica_attrs)
        else:
            self._add_info_row(
                self.basic_info_layout, 0, "Error", "Unsupported object type"
            )

        # **CORRECTION**: We no longer add details_group_box here. It's already in the layout.
        # We just need to set the content for the scroll area.
        self.scroll_area.setWidget(scroll_content)

    def _clear_layout(self, layout: QGridLayout):
        """Removes all widgets from a given layout."""
        if layout is None:
            return
        # Use a reversed loop to safely remove items
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def _determine_object_type(self):
        """Determines the type of the current MNE object."""
        if isinstance(self.mne_object, (mne.io.Raw, mne.io.RawArray)):
            self.object_type = "Raw"
        elif isinstance(
            self.mne_object, (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF)
        ):
            self.object_type = "Epochs"
        elif isinstance(self.mne_object, mne.preprocessing.ICA):
            self.object_type = "ICA"
        else:
            self.object_type = "Unknown"

    def _add_info_row(self, layout, row, key, value):
        """Helper to add a key-value pair to the layout."""
        layout.addWidget(QLabel(f"<b>{key}:</b>"), row, 0)
        layout.addWidget(QLabel(str(value)), row, 1)

    def _populate_raw_basics(self, layout: QGridLayout):
        """Populates basic info for a Raw object."""
        self._add_info_row(layout, 0, "Channels", len(self.mne_object.ch_names))
        self._add_info_row(layout, 1, "Time Points", self.mne_object.n_times)
        self._add_info_row(
            layout, 2, "Sampling Freq (Hz)", f"{self.mne_object.info['sfreq']:.2f}"
        )
        self._add_info_row(
            layout, 3, "Duration (s)", f"{self.mne_object.times[-1]:.2f}"
        )
        self._add_info_row(
            layout,
            4,
            "Filters",
            f"{self.mne_object.info['highpass']} to {self.mne_object.info['lowpass']}",
        )

    def _populate_epochs_basics(self, layout: QGridLayout):
        """Populates basic info for an Epochs object."""
        n_kept = len(self.mne_object)
        n_total = len(self.mne_object.drop_log)
        self._add_info_row(layout, 0, "Epochs (Kept/Total)", f"{n_kept}/{n_total}")
        self._add_info_row(
            layout,
            1,
            "Channels (Good/Total)",
            f"{len(self.mne_object.ch_names)-len(self.mne_object.info['bads'])}/{len(self.mne_object.ch_names)}",
        )
        self._add_info_row(
            layout, 2, "Sampling Freq (Hz)", f"{self.mne_object.info['sfreq']:.2f}"
        )
        self._add_info_row(
            layout,
            3,
            "Time Window (ms)",
            f"{self.mne_object.tmin*1e3:.2f} to {self.mne_object.tmax*1e3:.2f}",
        )
        self._add_info_row(
            layout,
            4,
            "Baseline (ms)",
            (
                f"{self.mne_object.baseline[0]*1e3:.2f} to {self.mne_object.baseline[1]*1e3:.2f}"
                if self.mne_object.baseline
                else "None"
            ),
        )
        self._add_info_row(
            layout,
            5,
            "Filters",
            f"{self.mne_object.info['highpass']} to {self.mne_object.info['lowpass']}",
        )

    def _populate_ica_basics(self, layout: QGridLayout):
        """Populates basic info for an ICA object."""
        n_components = (
            self.mne_object.n_components_ if self.mne_object.n_components_ else 0
        )
        n_excluded = len(self.mne_object.exclude)
        self._add_info_row(
            layout,
            0,
            "Components (Kept/Total)",
            f"{n_components - n_excluded}/{n_components}",
        )
        self._add_info_row(
            layout,
            1,
            "Samples",
            self.mne_object.n_samples_ if self.mne_object.n_samples_ else "N/A",
        )
        self._add_info_row(
            layout,
            2,
            "Fit Status",
            "Fitted" if self.mne_object.pre_whitener_ is not None else "Not Fitted",
        )
        self._add_info_row(layout, 3, "Method", self.mne_object.method)

    def _create_dict_widget(self, info_dict: dict) -> QWidget:
        """Creates a widget containing a grid layout populated from a dictionary."""
        content_widget = QWidget()
        layout = QGridLayout(content_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        for row, (key, value) in enumerate(info_dict.items()):
            key_label = QLabel(f"<b>{key}:</b>")
            key_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )

            value_label = QLabel(str(value))
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )

            layout.addWidget(key_label, row, 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(value_label, row, 1)
        return content_widget


class MainWindow(QMainWindow):
    """Main application window for demonstration."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNE Info Viewer - Dynamic Demo")
        self.setGeometry(100, 100, 500, 600)

        # --- Create Dummy MNE Objects ---
        self._create_mne_objects()

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Create Buttons for Control ---
        button_layout = QHBoxLayout()
        raw_button = QPushButton("Load Raw")
        filtered_raw_button = QPushButton("Load Filtered Raw")
        epochs_button = QPushButton("Load Epochs")
        clear_button = QPushButton("Clear")

        button_layout.addWidget(raw_button)
        button_layout.addWidget(filtered_raw_button)
        button_layout.addWidget(epochs_button)
        button_layout.addWidget(clear_button)

        main_layout.addLayout(button_layout)

        # --- Instantiate the single, reusable info widget ---
        self.info_widget = ObjectInfoWidget()
        main_layout.addWidget(self.info_widget)
        main_layout.addStretch()

        # --- Connect Buttons to Slots ---
        raw_button.clicked.connect(self.display_raw)
        filtered_raw_button.clicked.connect(self.display_filtered_raw)
        epochs_button.clicked.connect(self.display_epochs)
        clear_button.clicked.connect(self.info_widget.clear_info)

        # --- Display initial data ---
        self.display_raw()

    def _create_mne_objects(self):
        """Creates MNE objects and stores them as instance attributes."""
        ch_names = [f"EEG {i:03}" for i in range(20)] + ["EOG 061"]
        ch_types = ["eeg"] * 20 + ["eog"]
        sfreq = 200
        n_times = sfreq * 30
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(len(ch_names), n_times)

        self.raw = mne.io.RawArray(data, info)
        self.raw.info["description"] = "Original Raw object for demo."
        self.raw.info["bads"] = ["EEG 008"]

        # Create a "filtered" version for the demo
        self.filtered_raw = self.raw.copy().filter(l_freq=1.0, h_freq=40.0)
        self.filtered_raw.info["description"] = "Filtered Raw object."
        self.filtered_raw.info["bads"].append("EEG 012")

        events = mne.make_fixed_length_events(self.raw, duration=1.0)
        self.epochs = mne.Epochs(self.raw, events, tmin=-0.2, tmax=0.5, preload=True)

    def display_raw(self):
        self.info_widget.update_info(self.raw, title="Raw Info:")

    def display_filtered_raw(self):
        # Here we demonstrate the custom title feature!
        self.info_widget.update_info(self.filtered_raw, title="Filtered Raw Info:")

    def display_epochs(self):
        self.info_widget.update_info(self.epochs, title="Epochs Info:")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
