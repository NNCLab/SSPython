import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
from PySide6.QtWidgets import QApplication

from core.processing import Preprocessor
from ui.widgets.preprocessing_widgets import EpochingDialog, EpochingSettingsWidget


def resting_raw(boundaries=(), *, first_samp=0):
    raw = mne.io.RawArray(
        np.zeros((1, 1200)),
        mne.create_info(["Cz"], sfreq=100, ch_types="eeg"),
        first_samp=first_samp,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            boundaries,
            [0.0] * len(boundaries),
            ["EDGE boundary"] * len(boundaries),
        )
    )
    return raw


class TestRestingStateEpoching(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def preprocessor(self, raw):
        source_path = self.root / "rest_raw.fif"
        raw.save(source_path, overwrite=True, verbose=False)
        preprocessor = Preprocessor(source_path, self.root / "derivatives")
        self.addCleanup(
            preprocessor.release_cached_stages,
            ["raw", "filtered_raw", "epochs"],
            False,
        )
        return preprocessor

    def saved_epochs(self, preprocessor):
        return mne.read_epochs(preprocessor.paths["epochs"], preload=True, verbose=False)

    def test_no_boundaries_keep_single_condition_with_other_annotations(self):
        raw = resting_raw()
        raw.set_annotations(mne.Annotations([1.0], [0.0], ["Trigger"]))
        preprocessor = self.preprocessor(raw)

        preprocessor.run_epoching(mode="fixed", fixed_duration=2.0, verbose=False)

        epochs = self.saved_epochs(preprocessor)
        self.assertEqual(epochs.event_id, {"1": 1})
        np.testing.assert_array_equal(epochs.events[:, 2], [1] * 6)

    def test_one_boundary_defaults_to_pre_post_on_boundary_sample(self):
        preprocessor = self.preprocessor(resting_raw([4.0], first_samp=500))

        preprocessor.run_epoching(mode="fixed", fixed_duration=2.0, verbose=False)

        epochs = self.saved_epochs(preprocessor)
        self.assertEqual(epochs.event_id, {"pre": 1, "post": 2})
        np.testing.assert_array_equal(
            epochs.events[:, 0], [500, 700, 900, 1100, 1300, 1500]
        )
        np.testing.assert_array_equal(epochs.events[:, 2], [1, 1, 2, 2, 2, 2])

    def test_multiple_boundaries_use_numbered_sections_and_deduplicate(self):
        preprocessor = self.preprocessor(resting_raw([4.0, 4.0, 8.0]))

        preprocessor.run_epoching(mode="fixed", fixed_duration=2.0, verbose=False)

        epochs = self.saved_epochs(preprocessor)
        self.assertEqual(epochs.event_id, {"1": 1, "2": 2, "3": 3})
        np.testing.assert_array_equal(epochs.events[:, 2], [1, 1, 2, 2, 3, 3])

    def test_custom_labels_survive_save_and_are_logged(self):
        preprocessor = self.preprocessor(resting_raw([4.0, 8.0]))

        preprocessor.run_epoching(
            mode="fixed",
            fixed_duration=2.0,
            fixed_overlap=1.0,
            section_labels=[" pre ", "task", "post"],
            verbose=False,
        )

        epochs = self.saved_epochs(preprocessor)
        self.assertEqual(epochs.event_id, {"pre": 1, "task": 2, "post": 3})
        np.testing.assert_array_equal(epochs.events[:, 2], [1] * 4 + [2] * 4 + [3] * 3)
        self.assertEqual(len(epochs["task"]), 4)
        log = json.loads(epochs.info["description"])[-1]["epoching"]
        self.assertEqual(log["event_id"], epochs.event_id)
        self.assertEqual(log["section_labels"], ["pre", "task", "post"])
        self.assertEqual(preprocessor.epoching_mode, "fixed")

    def test_resampling_uses_boundary_samples_at_new_sampling_rate(self):
        preprocessor = self.preprocessor(resting_raw([4.0, 8.0], first_samp=500))

        preprocessor.run_epoching(
            mode="fixed", fixed_duration=2.0, resample=50, verbose=False
        )

        epochs = self.saved_epochs(preprocessor)
        self.assertEqual(epochs.info["sfreq"], 50)
        np.testing.assert_array_equal(
            epochs.events[:, 0], [250, 350, 450, 550, 650, 750]
        )
        np.testing.assert_array_equal(epochs.events[:, 2], [1, 1, 2, 2, 3, 3])

    def test_concatenated_raw_preserves_bad_boundary_rejection(self):
        parts = [resting_raw().crop(tmax=3.99) for _ in range(3)]
        raw = mne.concatenate_raws(parts, verbose=False)
        preprocessor = self.preprocessor(raw)

        preprocessor.run_epoching(mode="fixed", fixed_duration=3.0, verbose=False)

        epochs = self.saved_epochs(preprocessor)
        # The epochs starting at 3 s and 6 s cross BAD boundaries and are rejected.
        np.testing.assert_array_equal(epochs.events[:, 0], [0, 900])
        np.testing.assert_array_equal(epochs.events[:, 2], [1, 3])
        self.assertEqual(preprocessor.epochs.event_id, {"1": 1, "2": 2, "3": 3})

    def test_invalid_labels_do_not_clear_existing_derivatives(self):
        preprocessor = self.preprocessor(resting_raw([4.0]))
        with patch.object(preprocessor, "_clear_downstream_files") as clear:
            for labels in (
                ["pre"],
                ["pre", "post", "extra"],
                ["pre", " "],
                ["pre", " pre "],
                "pre,post",
            ):
                with self.subTest(labels=labels), self.assertRaises(ValueError):
                    preprocessor.run_epoching(
                        mode="fixed",
                        fixed_duration=2.0,
                        section_labels=labels,
                        verbose=False,
                    )
            clear.assert_not_called()

    def test_event_based_epoching_preserves_trigger_conditions(self):
        raw = resting_raw([4.0])
        raw.annotations.append([1.0, 5.0], [0.0, 0.0], ["Pulse", "Sham"])
        preprocessor = self.preprocessor(raw)

        preprocessor.run_epoching(tlim=(-0.1, 0.1), verbose=False)

        self.assertEqual(preprocessor.epochs.event_id, {"Pulse": 1, "Sham": 2})
        np.testing.assert_array_equal(preprocessor.epochs.events[:, 2], [1, 2])


class TestRestingStateEpochingUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.settings_store = Mock()
        self.settings_store.current_pipeline_id.return_value = "standard"
        self.settings_store.get.return_value = {}
        settings_patch = patch(
            "ui.widgets.preprocessing_widgets.get_settings_store",
            return_value=self.settings_store,
        )
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.widget = EpochingSettingsWidget(show_events_list=True)
        self.addCleanup(self.widget.close)

    def test_no_boundary_hides_section_label_fields(self):
        self.widget.populate_events_list(resting_raw())

        self.assertTrue(self.widget.sections_group.isHidden())
        self.assertEqual(self.widget.section_label_edits, [])
        self.assertIsNone(self.widget.get_params()["section_labels"])

    def test_one_boundary_informs_user_and_offers_pre_post(self):
        self.widget.populate_events_list(resting_raw([4.0]))

        self.assertEqual(
            self.widget.mode_combobox.currentText(), "Fixed-length (Resting State)"
        )
        self.assertFalse(self.widget.sections_group.isHidden())
        self.assertIn("1 EDGE boundary", self.widget.sections_notice.text())
        self.assertIn("2 recording sections", self.widget.sections_notice.text())
        self.assertEqual(self.widget.get_params()["section_labels"], ["pre", "post"])

    def test_multiple_boundaries_offer_numbered_defaults_and_custom_names(self):
        self.widget.populate_events_list(resting_raw([4.0, 8.0]))
        self.assertEqual(self.widget.get_params()["section_labels"], ["1", "2", "3"])

        for edit, label in zip(self.widget.section_label_edits, ["pre", "task", "post"]):
            edit.setText(label)

        self.assertEqual(self.widget.get_params()["section_labels"], ["pre", "task", "post"])
        self.widget.save_settings()
        saved_params = self.settings_store.set.call_args.args[1]
        self.assertNotIn("section_labels", saved_params)

    def test_section_fields_only_appear_in_fixed_mode(self):
        raw = resting_raw([4.0])
        raw.annotations.append(1.0, 0.0, "Pulse")
        self.widget.populate_events_list(raw)
        self.assertTrue(self.widget.sections_group.isHidden())
        self.assertIsNone(self.widget.get_params()["section_labels"])

        self.widget.mode_combobox.setCurrentText("Fixed-length (Resting State)")
        self.assertFalse(self.widget.sections_group.isHidden())
        self.assertEqual(self.widget.get_params()["section_labels"], ["pre", "post"])

    def test_invalid_names_are_rejected_before_processing(self):
        self.widget.populate_events_list(resting_raw([4.0]))
        self.widget.section_label_edits[1].setText("pre")
        with self.assertRaisesRegex(ValueError, "unique"):
            self.widget.get_params()
        self.widget.section_label_edits[1].clear()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.widget.get_params()

    def test_reset_keeps_boundary_only_recording_in_fixed_mode(self):
        self.widget.populate_events_list(resting_raw([4.0]))
        self.widget.section_label_edits[0].setText("before")

        self.widget.load_settings()

        self.assertEqual(self.widget.get_params()["mode"], "fixed")
        self.assertEqual(self.widget.get_params()["section_labels"], ["pre", "post"])
        self.assertFalse(self.widget.sections_group.isHidden())

    def test_dialog_passes_custom_labels_and_uses_latest_raw(self):
        preprocessor = Mock()
        preprocessor.has.side_effect = lambda stage: stage == "raw"
        preprocessor.raw = resting_raw()
        preprocessor._get_last_continuous.return_value = resting_raw([4.0])
        dialog = EpochingDialog(preprocessor)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.settings_widget.boundary_count, 1)
        for edit, label in zip(
            dialog.settings_widget.section_label_edits, ["before", "after"]
        ):
            edit.setText(label)

        with patch("ui.widgets.preprocessing_widgets.Worker") as worker:
            dialog.run_process()
            worker.call_args.args[0]()

        self.assertEqual(
            preprocessor.run_epoching.call_args.kwargs["section_labels"],
            ["before", "after"],
        )


if __name__ == "__main__":
    unittest.main()
