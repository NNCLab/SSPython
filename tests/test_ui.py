import unittest
import tempfile
from pathlib import Path
import mne
import numpy as np
from matplotlib.backend_bases import MouseButton
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem
from main import MainWindow
from core.pipelines import DatasetRecord, build_processing_paths, get_pipeline
from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.widgets.evoked_plot import EvokedPlotWidget
from ui.widgets.psd_plot import PSDPlotSettingsWidget
from ui.widgets.source_estimate_widget import (
    ComputeSTCSettingsDialog,
    STCSourceConfig,
    metadata_to_source_config,
    parse_covariance_methods,
)
from ui.widgets.time_frequency_widget import (
    ComputeTFRSettingsDialog,
    POWER_DB_MODE,
    RAW_POWER_MODE,
    TimeFrequencyDialog,
    TimeFrequencyWidget,
    apply_time_frequency_transform,
    build_time_frequency_display,
    parse_channel_selection,
    parse_tfr_freqs,
)
from ui.widgets.preprocessing_widgets import (
    EpochingSettingsWidget,
    PreprocessingSettingsWidget,
)
from ui.widgets.workspace_panel import DatasetInspectorPanel
from ui.widgets.tools.conversion_tool import ConvertToolDialog, MergeToolDialog
from ui.widgets.tools.object_info_widget import ObjectInfoWidget, extract_event_counts
from ui.widgets.tools.optional_range_widget import OptionalRangeWidget
from ui.widgets.real_time_widget import (
    ConnectionWidget,
    DataProcessingWorker,
    RealTimeERP,
    RealTimeSettingsWidget,
    apply_realtime_info_to_stream,
    apply_artifact_mask,
    apply_frequency_filters,
    append_limited_history,
    buffer_epoch_snapshot,
    build_trace_colors,
    build_mep_trace_uV,
    build_live_filter_pipeline,
    build_epoch_stream_configuration,
    channel_settings_from_info,
    channel_grid_positions,
    compute_mep_peak_to_peak_uV,
    detect_regular_stream_event_ids,
    raw_artifact_segments,
    realtime_info_has_montage,
)
from ui.widgets.real_time_plot_docks import RawMonitorDock

class TestUI(unittest.TestCase):

    def test_main_window_initialization(self):
        """Test if the MainWindow can be initialized without errors."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        window = MainWindow()
        self.assertIsInstance(window, MainWindow)
        window.close()

    def test_main_window_navigation_uses_analysis_without_continuous_analysis(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        window = MainWindow()

        nav_labels = [
            window.nav_list.item(index).text()
            for index in range(window.nav_list.count())
            if window.nav_list.item(index).text()
        ]

        self.assertIn("Analysis", nav_labels)
        self.assertNotIn("Continuous Analysis", nav_labels)
        self.assertNotIn("ERP Analysis", nav_labels)
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

    def test_append_limited_history_keeps_recent_samples(self):
        history = np.arange(2 * 4, dtype=float).reshape(2, 4)
        chunk = np.arange(100, 110, dtype=float).reshape(2, 5)

        limited, dropped = append_limited_history(history, chunk, max_samples=6)

        expected = np.concatenate((history[:, -1:], chunk), axis=1)
        np.testing.assert_array_equal(limited, expected)
        self.assertEqual(dropped, 3)

    def test_raw_artifact_segments_are_merged_and_clipped(self):
        stim_data = np.array([[0, 0, 1, 1, 0, 2]], dtype=float)

        segments = raw_artifact_segments(stim_data, 1000.0, (0.0, 0.001))

        self.assertEqual(segments, [(2, 4), (5, 6)])

    def test_data_processing_worker_reuses_processed_epoch_cache(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(["Cz"], sfreq=1000.0, ch_types=["eeg"])

        class FakeStream:
            def __init__(self, stream_info):
                self.info = stream_info
                self.n_new_samples = 0

        worker = DataProcessingWorker(
            FakeStream(info),
            {
                "tlim": (-0.01, 0.01),
                "decimate": 1,
                "max_epochs": 4,
                "art_rem": (None, None),
                "reference": "none",
            },
        )
        process_calls = []
        original_process_epoch_batch = worker._process_epoch_batch

        def counting_process(epoch_batch):
            process_calls.append(epoch_batch.shape[0])
            return original_process_epoch_batch(epoch_batch)

        worker._process_epoch_batch = counting_process
        epoch_batch = np.ones((2, 1, worker.original_times.size), dtype=float)

        worker._store_epoch_batch(epoch_batch)
        first_snapshot = worker._build_snapshot(include_raw=False, include_epoch=True)
        second_snapshot = worker._build_snapshot(include_raw=False, include_epoch=True)

        self.assertEqual(process_calls, [2])
        self.assertEqual(first_snapshot["epoch_data"]["n_epochs"], 2)
        self.assertEqual(second_snapshot["epoch_data"]["n_epochs"], 2)

        worker.update_params({"bads": ["Cz"]})
        self.assertEqual(process_calls, [2])

        worker.update_params({"apply_notch": True, "notch_freqs": [50.0]})
        self.assertEqual(process_calls, [2, 2])

    def test_data_processing_worker_emits_mep_data_for_emg_channels(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(
            ["Cz", "EMG Active", "EMG Ref"],
            sfreq=1000.0,
            ch_types=["eeg", "emg", "emg"],
        )

        class FakeStream:
            def __init__(self, stream_info):
                self.info = stream_info
                self.n_new_samples = 0

        worker = DataProcessingWorker(
            FakeStream(info),
            {
                "tlim": (-0.01, 0.06),
                "decimate": 1,
                "max_epochs": 4,
                "art_rem": (None, None),
                "reference": "none",
            },
        )
        eeg_batch = np.ones((2, 1, worker.original_times.size), dtype=float)
        emg_batch = np.zeros((2, 2, worker.original_times.size), dtype=float)
        emg_batch[:, 0, :] = 40e-6
        emg_batch[:, 1, :] = 5e-6

        worker._store_epoch_batch(eeg_batch, emg_batch)
        snapshot = worker._build_snapshot(include_raw=False, include_epoch=True)

        self.assertIn("mep_data", snapshot)
        self.assertEqual(snapshot["mep_data"]["emg_names"], ["EMG Active", "EMG Ref"])
        self.assertEqual(snapshot["mep_data"]["mean_data"].shape, (2, worker.times.size))
        self.assertEqual(snapshot["mep_data"]["n_epochs"], 2)

    def test_data_processing_worker_reprocesses_when_average_reference_toggles(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(["Cz", "Pz"], sfreq=1000.0, ch_types=["eeg", "eeg"])

        class FakeStream:
            def __init__(self, stream_info):
                self.info = stream_info
                self.n_new_samples = 0

        worker = DataProcessingWorker(
            FakeStream(info),
            {
                "tlim": (0.001, 0.003),
                "decimate": 1,
                "max_epochs": 4,
                "art_rem": (None, None),
                "reference": "average",
            },
        )
        epoch_batch = np.zeros((1, 2, worker.original_times.size), dtype=float)
        epoch_batch[:, 0, :] = 1.0
        epoch_batch[:, 1, :] = 3.0

        worker._store_epoch_batch(epoch_batch)

        np.testing.assert_allclose(worker.processed_epoch_buffer[0, 0], -1.0)
        np.testing.assert_allclose(worker.processed_epoch_buffer[0, 1], 1.0)

        worker.update_params({"reference": "none"})

        np.testing.assert_allclose(worker.processed_epoch_buffer[0, 0], 1.0)
        np.testing.assert_allclose(worker.processed_epoch_buffer[0, 1], 3.0)

    def test_data_processing_worker_applies_input_amplitude_scale(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(
            ["Cz", "EMG"],
            sfreq=1000.0,
            ch_types=["eeg", "emg"],
        )

        class FakeStream:
            def __init__(self, stream_info):
                self.info = stream_info
                self.n_new_samples = 0

        worker = DataProcessingWorker(
            FakeStream(info),
            {
                "tlim": (0.001, 0.003),
                "decimate": 1,
                "max_epochs": 4,
                "art_rem": (None, None),
                "reference": "none",
                "amplitude_scale": 1e-6,
            },
        )
        eeg_batch = np.full((1, 1, worker.original_times.size), 100.0, dtype=float)
        emg_batch = np.full((1, 1, worker.original_times.size), 50.0, dtype=float)

        worker._store_epoch_batch(eeg_batch, emg_batch)

        np.testing.assert_allclose(worker.processed_epoch_buffer[0, 0], 100e-6)
        np.testing.assert_allclose(worker.processed_emg_epoch_buffer[0, 0], 50e-6)

        worker.update_params({"amplitude_scale": 1.0})

        np.testing.assert_allclose(worker.processed_epoch_buffer[0, 0], 100.0)
        np.testing.assert_allclose(worker.processed_emg_epoch_buffer[0, 0], 50.0)

    def test_mep_trace_and_peak_to_peak_threshold_helpers(self):
        times_ms = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        mean_emg_data = np.array(
            [
                [0.0, 0.0, 10e-6, 60e-6, -5e-6, 0.0, 0.0],
                [0.0, 0.0, 5e-6, 5e-6, 5e-6, 0.0, 0.0],
            ]
        )

        trace_uV = build_mep_trace_uV(mean_emg_data, ["Active", "Ref"], "Active", "Ref")
        p2p_uV, mask = compute_mep_peak_to_peak_uV(trace_uV, times_ms)

        self.assertTrue(mask[2:6].all())
        self.assertGreaterEqual(p2p_uV, 50.0)

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

    def test_real_time_erp_hides_topology_without_montage_on_init(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = RealTimeERP({})
        self.assertFalse(widget.top_splitter.isHidden())
        self.assertFalse(widget.raw_widget.isHidden())
        self.assertTrue(widget.topo_widget.isHidden())
        self.assertFalse(widget.topo_toggle_action.isEnabled())
        widget.close()

    def test_realtime_info_montage_detection_requires_channel_positions(self):
        info = mne.create_info(["Cz", "Pz"], sfreq=1000.0, ch_types="eeg")
        self.assertFalse(realtime_info_has_montage(info))

        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
        self.assertTrue(realtime_info_has_montage(info))

    def test_trace_colors_only_vary_when_montage_positions_are_used(self):
        no_montage_colors = build_trace_colors(4, np.zeros((4, 3)), prefer_position_colors=False)
        self.assertEqual(len(set(no_montage_colors)), 1)

        positions = np.array(
            [
                [-0.04, 0.02, 0.08],
                [0.04, 0.02, 0.08],
                [-0.03, -0.03, 0.08],
                [0.03, -0.03, 0.08],
            ]
        )
        montage_colors = build_trace_colors(4, positions, prefer_position_colors=True)
        self.assertGreater(len(set(montage_colors)), 1)

    def test_topomap_payload_requires_montage(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = RealTimeERP({})
        info = mne.create_info(["Cz", "Pz"], sfreq=1000.0, ch_types="eeg")
        widget.info = info
        widget.ch_names = info["ch_names"]
        widget.times = np.linspace(-0.01, 0.02, 20)
        widget.mean_data = np.ones((2, widget.times.size), dtype=float)
        widget.evoked_dock.roi_region = (0.0, 10.0)
        widget.has_visual_montage = False

        self.assertIsNone(widget._build_topomap_payload())
        widget.close()

    def test_raw_monitor_downsamples_to_visible_pixels(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        dock = RawMonitorDock()
        dock.resize(320, 180)
        dock.configure(["Cz", "Pz"], [(0.1, 0.2, 0.3, 1.0), (0.1, 0.2, 0.3, 1.0)], 5.0)
        dock.set_visible_channels(1)

        time_axis = np.linspace(-5.0, 0.0, 20000)
        scaled_data = np.vstack((np.sin(time_axis * 80.0), np.cos(time_axis * 80.0)))
        dock.update_data(time_axis, scaled_data)
        app.processEvents()

        self.assertLessEqual(len(dock.curves[0].xData), dock.DISPLAY_MAX_POINTS)
        hidden_x = dock.curves[1].xData
        self.assertEqual(0 if hidden_x is None else len(hidden_x), 0)
        dock.close()

    def test_real_time_erp_enables_mep_controls_for_emg_channels(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = RealTimeERP({})
        widget.emg_names = ["EMG Active", "EMG Ref"]
        widget._configure_mep_controls()

        self.assertTrue(widget.mep_toggle_action.isEnabled())
        self.assertFalse(widget.mep_active_combo.isHidden())
        self.assertFalse(widget.mep_reference_combo.isHidden())
        self.assertTrue(widget.mep_active_combo.isEnabled())
        self.assertTrue(widget.mep_reference_combo.isEnabled())
        self.assertEqual(widget.mep_active_combo.currentText(), "EMG Active")
        self.assertEqual(widget.mep_reference_combo.currentText(), "EMG Ref")
        widget.close()

    def test_real_time_erp_hides_mep_controls_without_emg_channels(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = RealTimeERP({})
        widget.emg_names = []
        widget._configure_mep_controls()

        self.assertFalse(widget.mep_toggle_action.isEnabled())
        self.assertFalse(widget.mep_toggle_action.isVisible())
        self.assertTrue(widget.mep_active_combo.isHidden())
        self.assertTrue(widget.mep_reference_combo.isHidden())
        self.assertFalse(widget.mep_active_combo.isEnabled())
        self.assertFalse(widget.mep_reference_combo.isEnabled())
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

    def test_connection_widget_accepts_info_without_montage(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(
            ["Cz", "TRIGGER"],
            sfreq=1000.0,
            ch_types=["eeg", "stim"],
        )

        widget = ConnectionWidget()
        widget.set_realtime_info(info, label="No Montage Info")
        settings = widget.get_settings()

        self.assertEqual(settings["info"]["ch_names"], ["Cz", "TRIGGER"])
        self.assertEqual(settings["event_channels"], "TRIGGER")
        self.assertIn("Montage: No", widget.info_summary_label.text())
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

    def test_real_time_channel_settings_keep_emg_channels_enabled(self):
        info = mne.create_info(
            ["Cz", "EMG Active", "TRIGGER"],
            sfreq=1000,
            ch_types=["eeg", "emg", "stim"],
        )

        settings = channel_settings_from_info(info)

        self.assertEqual(settings[1], {"name": "EMG Active", "enabled": True, "type": "emg"})

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

    def test_dataset_inspector_deletes_selected_and_subsequent_derivatives(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            raw_path = workspace_root / "sub-01" / "eeg" / "sub-01_raw.fif"
            raw_path.parent.mkdir(parents=True)
            raw_path.touch()

            pipeline = get_pipeline("standard")
            derivative_root = workspace_root / "derivatives" / "standard"
            paths = build_processing_paths(raw_path, derivative_root, pipeline=pipeline)
            derivative_stage_ids = [
                "filtered_raw",
                "continuous_ica",
                "epochs",
                "epochs_ica",
                "preprocessed",
            ]
            for stage_id in derivative_stage_ids:
                paths[stage_id].touch()

            dataset = DatasetRecord(
                raw_path=raw_path,
                pipeline=pipeline,
                derivative_root=derivative_root,
                paths=paths,
            )

            panel = DatasetInspectorPanel()
            panel.set_dataset(dataset)

            delete_paths = panel._existing_derivative_paths_from_stage("epochs")
            self.assertEqual(
                delete_paths,
                [paths["epochs"], paths["epochs_ica"], paths["preprocessed"]],
            )

            errors = panel._delete_derivative_paths(delete_paths)

            self.assertEqual(errors, [])
            self.assertTrue(paths["raw"].exists())
            self.assertTrue(paths["filtered_raw"].exists())
            self.assertTrue(paths["continuous_ica"].exists())
            self.assertFalse(paths["epochs"].exists())
            self.assertFalse(paths["epochs_ica"].exists())
            self.assertFalse(paths["preprocessed"].exists())
            self.assertEqual(panel._existing_derivative_paths_from_stage("raw"), [])
            panel.close()

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

    def test_optional_range_widget_keeps_required_bounds_enabled_without_value(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = OptionalRangeWidget(required=(True, True))
        widget.setValue(None)

        self.assertTrue(widget.low_check.isChecked())
        self.assertTrue(widget.high_check.isChecked())
        self.assertTrue(widget.low_input.isEnabled())
        self.assertTrue(widget.high_input.isEnabled())
        widget.close()

    def test_epoching_settings_restore_time_limits_after_fixed_mode(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = EpochingSettingsWidget(show_events_list=True)
        params = widget.get_defaults()
        params.update(
            {
                "mode": "fixed",
                "tlim": (None, None),
                "fixed_duration": 3.0,
                "fixed_overlap": 0.0,
            }
        )
        widget.set_params(params)

        widget.mode_combobox.setCurrentText("Event-based")
        app.processEvents()

        self.assertTrue(widget.tlim_edit.low_check.isChecked())
        self.assertTrue(widget.tlim_edit.high_check.isChecked())
        self.assertTrue(widget.tlim_edit.low_input.isEnabled())
        self.assertTrue(widget.tlim_edit.high_input.isEnabled())
        self.assertEqual(widget.get_params()["tlim"], (-0.8, 0.8))
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

    def test_evoked_plot_widget_defaults_to_full_epoch_xlim(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_names = montage.ch_names[:8]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
        info.set_montage(montage)
        evoked = mne.EvokedArray(
            np.random.randn(len(ch_names), 1000) * 1e-6,
            info,
            tmin=-0.25,
            verbose=False,
        )

        widget = EvokedPlotWidget()
        widget.update_plot(evoked, label="Test")
        x_min, x_max = widget.canvas.axes.get_xlim()

        self.assertAlmostEqual(x_min, evoked.times[0] * 1000)
        self.assertAlmostEqual(x_max, evoked.times[-1] * 1000)
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
        app.processEvents()
        self.assertFalse(widget.event_selector.isHidden())
        self.assertEqual(widget.event_selector.count(), 3)
        self.assertEqual(widget.event_selector.currentData()["code"], None)

        widget.event_selector.setCurrentIndex(2)
        app.processEvents()

        self.assertEqual(widget.selected_event_code, 2)
        self.assertEqual(widget.selected_event_label, "Sham")
        self.assertIn("Sham", widget.title_label.text())
        widget.close()

    def test_evoked_plot_widget_hides_event_selector_for_single_event(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_names = montage.ch_names[:8]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
        info.set_montage(montage)
        data = np.random.randn(3, len(ch_names), 300) * 1e-6
        events = np.array([[0, 0, 1], [400, 0, 1], [800, 0, 1]])
        epochs = mne.EpochsArray(
            data,
            info,
            events=events,
            event_id={"Pulse": 1},
            tmin=-0.1,
            verbose=False,
        )

        widget = EvokedPlotWidget()
        widget.update_plot(epochs, label="Test")

        self.assertTrue(widget.event_selector.isHidden())
        self.assertEqual(widget.event_selector.count(), 0)
        self.assertIsNone(widget.selected_event_code)
        widget.close()

    def test_psd_plot_settings_widget_builds_epoch_plot_params(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(["Cz", "Pz"], sfreq=1000, ch_types="eeg")
        epochs = mne.EpochsArray(
            np.random.randn(3, 2, 300) * 1e-6,
            info,
            tmin=-0.1,
            verbose=False,
        )

        widget = PSDPlotSettingsWidget(epochs)
        widget.frequency_range_input.setValue((2.0, 40.0))
        widget.time_range_input.setValue((-0.05, 0.19))
        widget.picks_input.setText("Cz, Pz")
        widget.method_input.setCurrentText("welch")
        widget.average_input.setChecked(True)
        widget.db_input.setChecked(False)
        widget.xscale_input.setCurrentText("log")
        widget.exclude_bads_input.setChecked(False)

        params = widget.get_plot_params()

        self.assertEqual(params["fmin"], 2.0)
        self.assertEqual(params["fmax"], 40.0)
        self.assertEqual(params["tmin"], -0.05)
        self.assertEqual(params["tmax"], 0.19)
        self.assertEqual(params["picks"], ["Cz", "Pz"])
        self.assertEqual(params["method"], "welch")
        self.assertTrue(params["average"])
        self.assertFalse(params["dB"])
        self.assertEqual(params["xscale"], "log")
        self.assertEqual(params["exclude"], ())
        self.assertIsNone(widget.validation_error())
        widget.close()

    def test_time_frequency_transform_applies_db_baseline(self):
        power = np.array(
            [
                [1.0, 3.0, 6.0],
                [2.0, 2.0, 8.0],
            ]
        )
        times_s = np.array([-0.1, 0.0, 0.1])

        transformed, units = apply_time_frequency_transform(
            power,
            times_s,
            mode=POWER_DB_MODE,
            baseline_s=(-0.1, 0.0),
            apply_baseline=True,
        )

        expected = 10.0 * np.log10(power / np.array([[2.0], [2.0]]))
        np.testing.assert_allclose(transformed, expected)
        self.assertEqual(units, "Power (dB)")

    def test_time_frequency_display_filters_channel_and_event(self):
        info = mne.create_info(["Cz", "Pz"], sfreq=1000, ch_types="eeg")
        events = np.array([[0, 0, 1], [200, 0, 2], [400, 0, 1]])
        epochs = mne.EpochsArray(
            np.random.randn(3, 2, 4) * 1e-6,
            info,
            events=events,
            event_id={"Pulse": 1, "Sham": 2},
            tmin=-0.001,
            verbose=False,
        )
        freqs = np.array([8.0, 12.0])
        tfr_data = np.ones((3, 2, 2, 4), dtype=float)
        tfr_data[:, 0, :, :] = 2.0
        tfr_data[:, 1, :, :] = 4.0
        tfr_data[1, 1, :, :] = 8.0
        tfr = mne.time_frequency.EpochsTFRArray(
            info,
            tfr_data,
            epochs.times,
            freqs,
            events=events,
            event_id={"Pulse": 1, "Sham": 2},
        )

        display = build_time_frequency_display(
            epochs=epochs,
            tfr=tfr,
            channel_name="Pz",
            event_code=2,
            transform_mode=RAW_POWER_MODE,
            apply_baseline=False,
        )

        np.testing.assert_allclose(display.values, np.full((2, 4), 8.0e12))
        self.assertEqual(display.channel_label, "Pz")
        self.assertEqual(display.event_label, "Sham")
        self.assertEqual(display.n_epochs, 1)

    def test_compute_tfr_settings_dialog_builds_mne_params(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(["Cz", "Pz"], sfreq=1000, ch_types="eeg")
        epochs = mne.EpochsArray(
            np.random.randn(2, 2, 20) * 1e-6,
            info,
            tmin=-0.01,
            verbose=False,
        )

        dialog = ComputeTFRSettingsDialog(epochs)
        dialog.method_combo.setCurrentText("multitaper")
        dialog.freqs_input.setText("8:12:2")
        dialog.time_range_input.setValue((-0.005, 0.009))
        dialog.picks_input.setText("Cz, Pz")
        dialog.proj_checkbox.setChecked(True)
        dialog.output_combo.setCurrentText("power")
        dialog.average_checkbox.setChecked(True)
        dialog.return_itc_checkbox.setChecked(True)
        dialog.decim_input.setValue(2)
        dialog.n_jobs_checkbox.setChecked(True)
        dialog.n_jobs_input.setValue(-1)
        dialog.verbose_combo.setCurrentText("INFO")
        dialog.n_cycles_input.setText("3.5")
        dialog.time_bandwidth_input.setValue(5.0)

        params = dialog.get_compute_params()

        self.assertEqual(params["method"], "multitaper")
        np.testing.assert_allclose(params["freqs"], np.array([8.0, 10.0, 12.0]))
        self.assertEqual(params["tmin"], -0.005)
        self.assertEqual(params["tmax"], 0.009)
        self.assertEqual(params["picks"], ["Cz", "Pz"])
        self.assertTrue(params["proj"])
        self.assertEqual(params["output"], "power")
        self.assertTrue(params["average"])
        self.assertTrue(params["return_itc"])
        self.assertEqual(params["decim"], 2)
        self.assertEqual(params["n_jobs"], -1)
        self.assertEqual(params["verbose"], "INFO")
        self.assertEqual(params["n_cycles"], 3.5)
        self.assertTrue(params["use_fft"])
        self.assertTrue(params["zero_mean"])
        self.assertEqual(params["time_bandwidth"], 5.0)
        dialog.close()

    def test_compute_stc_settings_dialog_builds_mne_params(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(["Cz", "Pz"], sfreq=1000, ch_types="eeg")
        events = np.array([[0, 0, 1], [100, 0, 2]])
        epochs = mne.EpochsArray(
            np.random.randn(2, 2, 20) * 1e-6,
            info,
            events=events,
            event_id={"Pulse": 1, "Sham": 2},
            tmin=-0.01,
            verbose=False,
        )

        dialog = ComputeSTCSettingsDialog(epochs)
        self.assertFalse(dialog.isModal())
        dialog.condition_combo.setCurrentText("Pulse")
        dialog.noise_window_input.setValue((-0.01, 0.0))
        dialog.noise_method_input.setText("empirical")
        dialog.rank_combo.setCurrentText("info")
        dialog.mindist_input.setValue(7.5)
        dialog.n_jobs_checkbox.setChecked(True)
        dialog.n_jobs_input.setValue(2)
        dialog.loose_input.setValue(0.4)
        dialog.depth_input.setValue(0.6)
        dialog.inverse_method_combo.setCurrentText("sLORETA")
        dialog.snr_input.setValue(4.0)
        dialog.pick_ori_combo.setCurrentText("normal")

        params = dialog.get_compute_params()

        self.assertIsInstance(params["source_config"], STCSourceConfig)
        self.assertEqual(params["source_config"].mode, "fsaverage")
        self.assertEqual(params["condition"], "Pulse")
        self.assertTrue(params["pick_eeg"])
        self.assertTrue(params["set_eeg_reference"])
        self.assertEqual(params["noise_tmin"], -0.01)
        self.assertEqual(params["noise_tmax"], 0.0)
        self.assertEqual(params["noise_method"], "empirical")
        self.assertEqual(params["rank"], "info")
        self.assertEqual(params["mindist"], 7.5)
        self.assertEqual(params["n_jobs"], 2)
        self.assertEqual(params["loose"], 0.4)
        self.assertEqual(params["depth"], 0.6)
        self.assertEqual(params["inverse_method"], "sLORETA")
        self.assertEqual(params["snr"], 4.0)
        self.assertEqual(params["pick_ori"], "normal")
        self.assertEqual(parse_covariance_methods("shrunk, empirical"), ["shrunk", "empirical"])
        dialog.close()

    def test_stc_metadata_maps_to_source_config(self):
        metadata = {
            "source": {
                "mode": "custom",
                "subject": "sub-01",
                "subjects_dir": "subjects",
                "src": "sub-01-src.fif",
                "bem": "sub-01-bem-sol.fif",
                "trans": "sub-01-trans.fif",
            }
        }

        config = metadata_to_source_config(metadata)

        self.assertEqual(config.mode, "custom")
        self.assertEqual(config.subject, "sub-01")
        self.assertEqual(config.subjects_dir, Path("subjects"))
        self.assertEqual(config.src, Path("sub-01-src.fif"))
        self.assertEqual(config.bem, Path("sub-01-bem-sol.fif"))
        self.assertEqual(config.trans, Path("sub-01-trans.fif"))

    def test_tfr_frequency_parser_supports_stockwell_auto(self):
        self.assertEqual(parse_tfr_freqs("auto", method="stockwell"), "auto")
        np.testing.assert_allclose(
            parse_tfr_freqs("8, 45", method="stockwell"),
            np.array([8.0, 45.0]),
        )
        with self.assertRaises(ValueError):
            parse_tfr_freqs("auto", method="morlet")

    def test_channel_selection_parser_accepts_typed_names(self):
        available = ["Cz", "Pz", "EEG 001"]
        self.assertEqual(parse_channel_selection("Cz, Pz", available), ["Cz", "Pz"])
        self.assertEqual(parse_channel_selection("cz", available), ["Cz"])
        self.assertEqual(parse_channel_selection("EEG 001", available), ["EEG 001"])
        self.assertIsNone(parse_channel_selection("", available))
        with self.assertRaises(ValueError):
            parse_channel_selection("Missing", available)

    def test_time_frequency_widget_is_non_modal_and_updates_roi_marginals(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        info = mne.create_info(["Cz", "Pz", "Fz"], sfreq=1000, ch_types="eeg")
        events = np.array([[0, 0, 1], [200, 0, 1]])
        epochs = mne.EpochsArray(
            np.random.randn(2, 3, 20) * 1e-6,
            info,
            events=events,
            event_id={"Pulse": 1},
            tmin=-0.01,
            verbose=False,
        )
        freqs = np.array([6.0, 10.0, 20.0])
        tfr = mne.time_frequency.EpochsTFRArray(
            info,
            np.abs(np.random.randn(2, 3, 3, 20)) + 1.0,
            epochs.times,
            freqs,
            events=events,
            event_id={"Pulse": 1},
        )
        itc = mne.time_frequency.AverageTFRArray(
            info,
            np.full((3, 3, 20), 0.5),
            epochs.times,
            freqs,
        )

        widget = TimeFrequencyWidget(
            epochs,
            label="Test TF",
            tfr_derivatives={
                "Power (TFR)": tfr,
                "Inter-trial coherence (ITC)": itc,
            },
        )
        self.assertIsNotNone(widget.image_artist)
        self.assertFalse(hasattr(widget, "dock_button"))
        self.assertFalse(hasattr(widget, "close_button"))
        self.assertFalse(hasattr(widget, "band_checkboxes"))
        self.assertFalse(hasattr(widget, "link_time_checkbox"))
        self.assertFalse(hasattr(widget, "compute_button"))
        self.assertEqual(widget.derivative_combo.count(), 2)
        self.assertTrue(hasattr(widget, "channels_input"))
        self.assertTrue(hasattr(widget, "channel_picker_button"))
        self.assertEqual(len(widget.erp_channel_lines), 3)
        self.assertIsNone(widget.erp_average_line)
        widget._set_selected_channels(["Cz", "Pz"])
        self.assertEqual(widget._selected_channels(), ["Cz", "Pz"])
        self.assertEqual(len(widget.erp_channel_lines), 2)
        self.assertIsNotNone(widget.erp_average_line)
        self.assertEqual(widget.erp_average_line.get_color(), "black")

        widget.canvas.draw()
        self.assertGreater(
            widget.spectrum_ax.get_position().x0,
            widget.main_ax.get_position().x1,
        )
        self.assertGreater(
            widget.erp_ax.get_position().y0,
            widget.main_ax.get_position().y1,
        )
        self.assertTrue(
            widget.spectrum_ax.get_shared_y_axes().joined(
                widget.spectrum_ax,
                widget.main_ax,
            )
        )
        self.assertLess(
            widget.time_power_ax.get_position().y1,
            widget.main_ax.get_position().y0,
        )
        self.assertTrue(
            widget.time_power_ax.get_shared_x_axes().joined(
                widget.time_power_ax,
                widget.main_ax,
            )
        )
        self.assertGreater(
            widget.slider_ax.get_position().x0 - widget.colorbar_ax.get_position().x1,
            0.02,
        )
        self.assertFalse(widget.settings_panel.isHidden())
        widget.controls_toggle_button.setChecked(False)
        app.processEvents()
        self.assertTrue(widget.settings_panel.isHidden())
        widget.controls_toggle_button.setChecked(True)
        app.processEvents()
        self.assertFalse(widget.settings_panel.isHidden())

        widget.derivative_combo.setCurrentIndex(1)
        app.processEvents()
        self.assertEqual(widget.current_derivative_key, "Inter-trial coherence (ITC)")

        widget._set_roi_from_values(0.0, 8.0, 8.0, 12.0, update_controls=True, draw=True)
        self.assertAlmostEqual(widget.roi_time_start_input.value(), 0.0)
        self.assertAlmostEqual(widget.roi_time_stop_input.value(), 8.0)
        self.assertAlmostEqual(widget.roi_freq_low_input.value(), 8.0)
        self.assertAlmostEqual(widget.roi_freq_high_input.value(), 12.0)
        self.assertIn("Mean:", widget.roi_mean_label.text())
        self.assertEqual(len(widget.time_power_line.get_xdata()), len(epochs.times))

        class PlotEvent:
            def __init__(self, xdata, ydata, axis):
                self.inaxes = axis
                self.button = MouseButton.LEFT
                self.xdata = xdata
                self.ydata = ydata
                self.x, self.y = axis.transData.transform((xdata, ydata))

        widget._on_button_press(PlotEvent(0.0, 8.0, widget.main_ax))
        widget._on_button_release(PlotEvent(6.0, 12.0, widget.main_ax))
        self.assertAlmostEqual(widget.roi_time_start_input.value(), 0.0)
        self.assertAlmostEqual(widget.roi_time_stop_input.value(), 6.0)
        self.assertAlmostEqual(widget.roi_freq_low_input.value(), 8.0)
        self.assertAlmostEqual(widget.roi_freq_high_input.value(), 12.0)

        class HoverEvent:
            def __init__(self, xdata, ydata, axis):
                self.inaxes = axis
                self.xdata = xdata
                self.ydata = ydata
                self.x, self.y = axis.transData.transform((xdata, ydata))

        widget._on_mouse_move(HoverEvent(0.0, 0.0, widget.erp_ax))
        self.assertTrue(widget.erp_tooltip.get_visible())
        self.assertTrue(widget.erp_hover_vline.get_visible())
        self.assertTrue(widget.erp_hover_hline.get_visible())
        self.assertIn("Time:", widget.erp_tooltip.get_text())
        widget._on_mouse_move(HoverEvent(0.0, 0.0, widget.time_power_ax))
        self.assertTrue(widget.time_power_tooltip.get_visible())
        self.assertTrue(widget.time_power_hover_vline.get_visible())
        self.assertTrue(widget.time_power_hover_hline.get_visible())
        self.assertIn("Avg power:", widget.time_power_tooltip.get_text())
        widget._on_mouse_move(HoverEvent(0.0, 10.0, widget.spectrum_ax))
        self.assertTrue(widget.spectrum_tooltip.get_visible())
        self.assertTrue(widget.spectrum_hover_vline.get_visible())
        self.assertTrue(widget.spectrum_hover_hline.get_visible())
        self.assertIn("Freq:", widget.spectrum_tooltip.get_text())

        dialog = TimeFrequencyDialog(widget=widget)
        self.assertFalse(dialog.isModal())
        self.assertGreaterEqual(dialog.minimumWidth(), 900)
        dialog.close()

    def test_analysis_page_time_frequency_button_requires_derivatives(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        page = ErpAnalysisPage()

        class FakeAnalysis:
            derivatives = {}

        page.tep = FakeAnalysis()
        self.assertFalse(page._has_time_frequency_derivatives())
        self.assertFalse(page._has_source_estimate_derivative())

        page.tep.derivatives = {"tfr": object(), "itc": None}
        self.assertTrue(page._has_time_frequency_derivatives())
        self.assertEqual(
            list(page._available_time_frequency_derivatives().keys()),
            ["Power (TFR)"],
        )
        self.assertEqual(page.tf_button.text(), "Compute TFR")
        self.assertEqual(page.time_frequency_button.text(), "Plot TFR")
        self.assertIs(page.tf_button.parentWidget(), page.analysis_group)
        self.assertIs(page.time_frequency_button.parentWidget(), page.analysis_group)

        page.tep.derivatives = {"stc": object(), "stc_metadata": None}
        self.assertTrue(page._has_source_estimate_derivative())
        self.assertEqual(page.stc_button.text(), "Compute STC")
        self.assertEqual(page.source_estimate_button.text(), "Plot STC")
        self.assertIs(page.stc_button.parentWidget(), page.analysis_group)
        self.assertIs(page.source_estimate_button.parentWidget(), page.analysis_group)
        page.close()

    def test_preprocessing_settings_widget_supports_average_reference(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = PreprocessingSettingsWidget()
        widget.set_params(widget.get_defaults() | {"reference": "average"})

        self.assertTrue(widget.average_ref_radio.isChecked())
        self.assertEqual(widget.get_params()["reference"], "average")
        widget.close()

    def test_real_time_settings_widget_supports_average_reference(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = RealTimeSettingsWidget()
        widget.set_settings(
            {
                "refresh_rate": 24,
                "art_rem": (-0.005, 0.005),
                "reference": "average",
                "apply_bandpass": False,
                "bandpass_range": (8.0, 80.0),
                "apply_notch": False,
                "notch_freqs": [50.0],
            }
        )

        self.assertTrue(widget.average_reference_checkbox.isChecked())
        self.assertEqual(widget.get_settings()["reference"], "average")

        widget.average_reference_checkbox.setChecked(False)

        self.assertEqual(widget.get_settings()["reference"], "none")
        widget.close()

    def test_real_time_settings_widget_supports_input_amplitude_scale(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = RealTimeSettingsWidget()
        widget.set_settings(
            {
                "refresh_rate": 24,
                "art_rem": (-0.005, 0.005),
                "amplitude_scale": 1e-6,
                "reference": "average",
                "apply_bandpass": False,
                "bandpass_range": (8.0, 80.0),
                "apply_notch": False,
                "notch_freqs": [50.0],
            }
        )

        self.assertEqual(widget.amplitude_scale_combo.currentText(), "Microvolts (uV)")
        self.assertAlmostEqual(widget.get_settings()["amplitude_scale"], 1e-6)

        widget.amplitude_scale_combo.setCurrentText("Volts (V)")

        self.assertEqual(widget.get_settings()["amplitude_scale"], 1.0)
        widget.close()

if __name__ == '__main__':
    unittest.main()
