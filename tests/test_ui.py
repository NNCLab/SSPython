import unittest
import tempfile
from pathlib import Path
import mne
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem
from main import MainWindow
from core.pipelines import DatasetRecord, build_processing_paths, get_pipeline
from ui.widgets.evoked_plot import EvokedPlotWidget
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
