import unittest
import mne
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem
from main import MainWindow
from ui.widgets.evoked_plot import EvokedPlotWidget
from ui.widgets.preprocessing_widgets import (
    EpochingSettingsWidget,
    PreprocessingSettingsWidget,
)
from ui.widgets.tools.conversion_tool import ConvertToolDialog, MergeToolDialog
from ui.widgets.tools.object_info_widget import ObjectInfoWidget, extract_event_counts
from ui.widgets.real_time_widget import (
    ConnectionWidget,
    RealTimeERP,
    apply_realtime_info_to_stream,
    apply_artifact_mask,
    apply_frequency_filters,
    buffer_epoch_snapshot,
    build_live_filter_pipeline,
    build_epoch_stream_configuration,
    channel_settings_from_info,
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

    def test_connection_widget_uses_info_summary_instead_of_channel_table(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = ConnectionWidget()
        self.assertFalse(hasattr(widget, "channel_table"))
        self.assertTrue(hasattr(widget, "info_summary_label"))
        self.assertEqual(widget.info_summary_label.text(), "No MNE info loaded.")
        widget.close()

    def test_real_time_channel_settings_come_from_info_not_names(self):
        montage = mne.channels.make_standard_montage("standard_1020")
        info = mne.create_info(
            ["Fz", "TRIGGER", "STI 014"],
            sfreq=200,
            ch_types=["eeg", "stim", "misc"],
        )
        info.set_montage(montage, on_missing="ignore")

        settings = channel_settings_from_info(info)

        self.assertEqual(
            settings,
            [
                {"name": "Fz", "enabled": True, "type": "eeg"},
                {"name": "TRIGGER", "enabled": True, "type": "stim"},
                {"name": "STI 014", "enabled": False, "type": "bad"},
            ],
        )

    def test_connection_widget_passes_montaged_info_to_settings(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        montage = mne.channels.make_standard_montage("standard_1020")
        info = mne.create_info(
            ["Fz", "TRIGGER", "STI 014"],
            sfreq=200,
            ch_types=["eeg", "stim", "misc"],
        )
        info.set_montage(montage, on_missing="ignore")

        widget = ConnectionWidget()
        widget.set_realtime_info(info, label="Live Info")
        settings = widget.get_settings()

        self.assertEqual(settings["info"]["ch_names"], ["Fz", "TRIGGER", "STI 014"])
        self.assertEqual(settings["event_channels"], "TRIGGER")
        self.assertEqual(settings["bads"], ["STI 014"])
        self.assertIn("Live Info", widget.info_edit.text())
        self.assertIn("Stim: TRIGGER", widget.info_summary_label.text())
        widget.close()

    def test_apply_realtime_info_to_stream_applies_names_types_and_montage(self):
        montage = mne.channels.make_standard_montage("standard_1020")
        realtime_info = mne.create_info(
            ["Fz", "TRIGGER", "STI 014"],
            sfreq=200,
            ch_types=["eeg", "stim", "misc"],
        )
        realtime_info.set_montage(montage, on_missing="ignore")

        stream_info = mne.create_info(
            ["A", "B", "C"],
            sfreq=200,
            ch_types=["eeg", "eeg", "misc"],
        )
        stream = mne.io.RawArray(np.zeros((3, 10)), stream_info, verbose=False)

        apply_realtime_info_to_stream(stream, realtime_info)

        self.assertEqual(stream.ch_names, ["Fz", "TRIGGER", "STI 014"])
        self.assertEqual(stream.get_channel_types(), ["eeg", "stim", "misc"])
        self.assertIsNotNone(stream.get_montage())

    def test_conversion_and_merge_tools_are_separate_actions(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        montage = mne.channels.make_standard_montage("standard_1020")
        info = mne.create_info(["Fz", "Cz"], sfreq=200, ch_types="eeg")
        info.set_montage(montage)

        convert_dialog = ConvertToolDialog(info=info, info_label="Test Info")
        merge_dialog = MergeToolDialog()
        window = MainWindow()

        self.assertEqual(convert_dialog.windowTitle(), "Convert EEG Data")
        self.assertIn("Test Info", convert_dialog.conversion_widget.info_edit.text())
        self.assertEqual(merge_dialog.windowTitle(), "Merge FIF Data")

        action_texts = [action.text() for action in window.conversion_menu.actions()]
        self.assertEqual(action_texts, ["Convert EEG data", "Merge FIF data"])

        convert_dialog.close()
        merge_dialog.close()
        window.close()

    def test_epoching_settings_preserve_selected_event_mapping(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = EpochingSettingsWidget(show_events_list=True)
        widget.event_id_map = {"Pulse": 7, "Sham": 42}
        widget.events_list.clear()
        for label in ("Pulse", "Sham"):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            widget.events_list.addItem(item)
        widget.show()
        app.processEvents()
        widget.events_list.item(1).setCheckState(Qt.CheckState.Unchecked)
        self.assertEqual(widget.get_event_selection(), {"Pulse": 7})
        widget.close()

    def test_evoked_plot_widget_rebuilds_axes_cleanly_on_update(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_names = montage.ch_names[:8]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
        info.set_montage(montage)
        evoked = mne.EvokedArray(
            np.random.randn(len(ch_names), 300) * 1e-6,
            info,
            tmin=-0.1,
            verbose=False,
        )

        widget = EvokedPlotWidget()
        widget.update_plot(evoked, label="Test")
        first_axes_count = len(widget.canvas.figure.axes)
        widget.update_plot(evoked, label="Test")
        second_axes_count = len(widget.canvas.figure.axes)

        self.assertEqual(second_axes_count, first_axes_count)
        widget.close()

    def test_evoked_plot_widget_exposes_event_selector_for_epochs(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_names = montage.ch_names[:8]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
        info.set_montage(montage)
        data = np.random.randn(6, len(ch_names), 300) * 1e-6
        events = np.array(
            [
                [0, 0, 1],
                [400, 0, 2],
                [800, 0, 1],
                [1200, 0, 2],
                [1600, 0, 1],
                [2000, 0, 2],
            ]
        )
        epochs = mne.EpochsArray(
            data,
            info,
            events=events,
            event_id={"Pulse": 1, "Sham": 2},
            tmin=-0.1,
            verbose=False,
        )

        widget = EvokedPlotWidget()
        widget.update_plot(epochs, label="Test")
        self.assertEqual(widget.event_selector.count(), 3)
        self.assertEqual(widget.event_selector.currentData()["code"], None)

        widget.event_selector.setCurrentIndex(2)
        app.processEvents()

        self.assertEqual(widget.selected_event_code, 2)
        self.assertEqual(widget.selected_event_label, "Sham")
        self.assertIn("Sham", widget.title_label.text())
        widget.close()

    def test_preprocessing_settings_widget_supports_average_reference(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = PreprocessingSettingsWidget()
        widget.set_params(widget.get_defaults() | {"reference": "average"})

        self.assertTrue(widget.average_ref_radio.isChecked())
        self.assertEqual(widget.get_params()["reference"], "average")
        widget.close()

if __name__ == '__main__':
    unittest.main()
