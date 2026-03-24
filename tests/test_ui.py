import unittest
import mne
import numpy as np
from PySide6.QtWidgets import QApplication
from main import MainWindow
from ui.widgets.tools.object_info_widget import ObjectInfoWidget, extract_event_counts
from ui.widgets.real_time_widget import (
    build_epoch_stream_configuration,
    channel_grid_positions,
    detect_regular_stream_event_ids,
)

class TestUI(unittest.TestCase):

    def test_main_window_initialization(self):
        """Test if the MainWindow can be initialized without errors."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        window = MainWindow()
        self.assertIsInstance(window, MainWindow)
        window.close()

    def test_object_info_widget_event_counts(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(ch_names=["EEG 001", "EEG 002"], sfreq=1000, ch_types="eeg")
        raw = mne.io.RawArray(np.random.randn(2, 2000), info)
        raw.set_annotations(
            mne.Annotations(
                onset=[0.2, 0.6, 1.0],
                duration=[0.0, 0.0, 0.0],
                description=["TMS", "Sham", "TMS"],
            )
        )

        self.assertEqual(extract_event_counts(raw), [("Sham", 1), ("TMS", 2)])

        widget = ObjectInfoWidget()
        widget.update_info(raw, "Recording summary")
        self.assertTrue(widget.events_frame.isVisible())
        widget.close()

    def test_extract_event_counts_from_epochs_uses_event_ids(self):
        info = mne.create_info(ch_names=["EEG 001", "EEG 002"], sfreq=1000, ch_types="eeg")
        data = np.random.randn(3, 2, 200)
        events = np.array([[0, 0, 1], [220, 0, 2], [440, 0, 1]])
        epochs = mne.EpochsArray(
            data,
            info,
            events=events,
            event_id={"Pulse": 1, "Sham": 2},
            tmin=-0.1,
            verbose=False,
        )

        self.assertEqual(extract_event_counts(epochs), [("Pulse", 2), ("Sham", 1)])

    def test_build_epoch_stream_configuration_uses_enabled_stim_channels(self):
        channel_settings = [
            {"name": "C3", "enabled": True, "type": "eeg", "event_id": 1, "event_name": ""},
            {"name": "TRIGGER", "enabled": True, "type": "stim", "event_id": 7, "event_name": "TMS"},
            {"name": "MARKER", "enabled": False, "type": "stim", "event_id": 9, "event_name": "Ignore"},
            {"name": "NOISY", "enabled": True, "type": "bad", "event_id": 1, "event_name": ""},
        ]

        event_id, event_channels, bads = build_epoch_stream_configuration(channel_settings)
        self.assertEqual(event_id, {"TMS": 7})
        self.assertEqual(event_channels, "TRIGGER")
        self.assertEqual(bads, ["MARKER", "NOISY"])

    def test_build_epoch_stream_configuration_leaves_event_id_auto_when_unset(self):
        channel_settings = [
            {"name": "C3", "enabled": True, "type": "eeg", "event_id": 0, "event_name": ""},
            {"name": "STI 014", "enabled": True, "type": "stim", "event_id": 0, "event_name": ""},
        ]

        event_id, event_channels, bads = build_epoch_stream_configuration(channel_settings)
        self.assertIsNone(event_id)
        self.assertEqual(event_channels, "STI 014")
        self.assertEqual(bads, [])

    def test_detect_regular_stream_event_ids_reads_stim_codes(self):
        stim_data = np.array([[0, 0, 5, 5, 0, 0, 9, 9, 0, 0]], dtype=float)
        detected = detect_regular_stream_event_ids(stim_data, "STI 014", 1000.0)
        self.assertEqual(detected, [5, 9])

    def test_channel_grid_positions_returns_unique_slots(self):
        coords = np.array([
            [-0.5, 0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.5, -0.5],
        ])
        positions = channel_grid_positions(coords)
        self.assertEqual(len(positions), 4)
        self.assertEqual(len(set(positions)), 4)

if __name__ == '__main__':
    unittest.main()
