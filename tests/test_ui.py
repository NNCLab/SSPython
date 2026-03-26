import unittest
import mne
import numpy as np
from PySide6.QtWidgets import QApplication
from main import MainWindow
from ui.widgets.tools.object_info_widget import ObjectInfoWidget, extract_event_counts
from ui.widgets.real_time_widget import (
    ConnectionWidget,
    RealTimeERP,
    apply_artifact_mask,
    apply_frequency_filters,
    buffer_epoch_snapshot,
    build_live_filter_pipeline,
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

    def test_build_epoch_stream_configuration_excludes_unsupported_live_types(self):
        channel_settings = [
            {"name": "VEOG", "enabled": True, "type": "eog", "event_id": 0, "event_name": ""},
            {"name": "TIME", "enabled": True, "type": "times", "event_id": 0, "event_name": ""},
            {"name": "STI 014", "enabled": True, "type": "stim", "event_id": 1, "event_name": ""},
        ]

        event_id, event_channels, bads = build_epoch_stream_configuration(channel_settings)
        self.assertEqual(event_id, 1)
        self.assertEqual(event_channels, "STI 014")
        self.assertEqual(bads, ["VEOG", "TIME"])

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

    def test_apply_artifact_mask_replaces_window_with_nan(self):
        epoch_batch = np.arange(20, dtype=float).reshape(1, 2, 10)
        times = np.linspace(-0.1, 0.08, 10)

        masked = apply_artifact_mask(epoch_batch, times, (-0.02, 0.02))
        start_idx = np.searchsorted(times, -0.02, side="left")
        end_idx = np.searchsorted(times, 0.02, side="right")

        self.assertTrue(np.isnan(masked[:, :, start_idx:end_idx]).all())
        self.assertTrue(np.isfinite(masked[:, :, :start_idx]).all())
        self.assertTrue(np.isfinite(masked[:, :, end_idx:]).all())

    def test_buffer_epoch_snapshot_keeps_chronological_order_and_last_n(self):
        epoch_buffer = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
        snapshot = buffer_epoch_snapshot(epoch_buffer, buffer_idx=2, n_valid_epochs=5, display_epoch_count=2)

        expected = np.concatenate((epoch_buffer[2:], epoch_buffer[:2]), axis=0)[-2:]
        np.testing.assert_array_equal(snapshot, expected)

    def test_apply_frequency_filters_preserves_shape(self):
        params = {
            "apply_bandpass": True,
            "bandpass_range": (1.0, 40.0),
            "apply_notch": True,
            "notch_freqs": [50.0],
        }
        bandpass_sos, notch_filters = build_live_filter_pipeline(200.0, params)
        data = np.random.randn(2, 400)

        filtered = apply_frequency_filters(data, bandpass_sos, notch_filters)

        self.assertEqual(filtered.shape, data.shape)
        self.assertTrue(np.isfinite(filtered).all())

    def test_real_time_erp_keeps_top_panels_available_on_init(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = RealTimeERP({})
        self.assertFalse(widget.top_splitter.isHidden())
        self.assertFalse(widget.raw_widget.isHidden())
        self.assertFalse(widget.topo_widget.isHidden())
        widget.close()

    def test_connection_widget_uses_compact_channel_columns(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = ConnectionWidget()
        headers = [
            widget.channel_table.horizontalHeaderItem(i).text()
            for i in range(widget.channel_table.columnCount())
        ]
        self.assertEqual(headers, ["Enabled", "Name", "Type"])
        widget.close()

if __name__ == '__main__':
    unittest.main()
