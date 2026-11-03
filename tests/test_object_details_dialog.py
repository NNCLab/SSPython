import gc
import json
import os
import unittest
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPlainTextEdit, QTreeWidget

from ui.widgets.tools.object_info_widget import ObjectDetailsDialog


class TestObjectDetailsDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, obj, **kwargs):
        dialog = ObjectDetailsDialog("Recording.fif", obj, **kwargs)
        self.addCleanup(dialog.close)
        dialog.show()
        self.app.processEvents()
        return dialog

    def text(self, dialog, name):
        return [label.text() for label in dialog.findChildren(QLabel, name)]

    def test_flat_details_keep_events_filters_and_channels_without_repeating_summary(self):
        info = mne.create_info(["C3", "C4"], 100, "eeg")
        info["bads"] = ["C4"]
        raw = mne.io.RawArray(np.zeros((2, 100)), info, verbose=False)
        raw.set_annotations(mne.Annotations([0.1, 0.2], [0, 0], ["<target>", "<target>"]))
        dialog = self.dialog(raw)

        self.assertEqual(len(dialog.summary_widget.metric_labels), 4)
        self.assertTrue(dialog.summary_widget.events_label.isHidden())
        self.assertEqual(self.text(dialog, "fileInfoDetailKey"), ["Bad Channels", "High-pass", "Low-pass", "<target>"])
        self.assertEqual(self.text(dialog, "fileInfoDetailValue"), ["C4", "0.00", "50.00", "2", "C3, C4"])
        self.assertEqual(dialog.findChildren(QPlainTextEdit), [])
        for label in dialog.findChildren(QLabel, "fileInfoDetailValue"):
            self.assertEqual(label.textFormat(), Qt.TextFormat.PlainText)
            self.assertTrue(label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse)

    def test_epoch_details_keep_baseline_and_bad_channels(self):
        info = mne.create_info(["C3", "C4"], 100, "eeg")
        info["bads"] = ["C4"]
        epochs = mne.EpochsArray(
            np.zeros((2, 2, 21)), info, tmin=-0.1, baseline=(-0.1, 0), verbose=False
        )
        dialog = self.dialog(epochs)

        self.assertEqual(self.text(dialog, "fileInfoDetailKey")[:2], ["Bad Channels", "Baseline"])
        self.assertIn("-100.00 ms to 0.00 ms", self.text(dialog, "fileInfoDetailValue"))
        self.assertNotIn("Window", self.text(dialog, "fileInfoDetailKey"))

    def test_ica_keeps_excluded_indices_pca_and_channel_names(self):
        ica = mne.preprocessing.ICA(n_components=3, method="fastica")
        ica.n_components_ = 3
        ica.n_samples_ = 1000
        ica.exclude = [0, 2]
        ica.n_pca_components = 4
        ica.ch_names = ["C3", "C4", "Cz", "Pz"]
        dialog = self.dialog(ica)

        self.assertEqual(self.text(dialog, "fileInfoDetailKey"), ["Excluded components", "PCA Components", "Channels"])
        self.assertEqual(self.text(dialog, "fileInfoDetailValue"), ["0, 2", "4", "C3, C4, Cz, Pz"])
        self.assertLess(dialog.height(), 540)

    def test_file_without_data_keeps_header_and_footer_compact(self):
        dialog = self.dialog(None, file_path="C:/study/recording.fif", status_text="Saved", description="Recording details")

        self.assertLess(dialog.height(), 200)
        title = dialog.findChild(QLabel, "fileInfoTitle")
        description = dialog.findChild(QLabel, "fileInfoDescription")
        self.assertLess(description.y() - title.geometry().bottom(), 12)

    def test_history_starts_collapsed_and_retains_nested_settings(self):
        info = mne.create_info(["C3"], 100, "eeg")
        info["description"] = json.dumps([
            {"filter": {"date": "2026-11-03", "notch": [50, 100], "options": {"phase": "zero"}}},
            {"review": {"date": "2026-11-03", "rejected_epochs": [1, 3]}},
        ])
        raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose=False)
        dialog = self.dialog(raw)
        tree = dialog.findChild(QTreeWidget, "fileInfoHistoryTree")

        self.assertEqual(tree.topLevelItemCount(), 2)
        operation = tree.topLevelItem(0)
        self.assertFalse(operation.isExpanded())
        self.assertEqual([operation.text(0), operation.text(1)], ["Filter", "2026-11-03"])
        self.assertEqual(operation.child(0).text(1), "50, 100")
        self.assertEqual(operation.child(1).child(0).text(1), "zero")
        operation.setExpanded(True)
        self.assertTrue(operation.isExpanded())

    def test_long_filename_and_path_fit_narrow_dialog_with_history_below(self):
        info = mne.create_info(["C3"], 100, "eeg")
        info["description"] = '[{"filter": {"date": "2026-11-03", "low": 1}}]'
        raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose=False)
        path = "C:/study/" + "long-subject-name_" * 20 + ".fif"
        dialog = self.dialog(raw, file_path=path, allow_delete=True)
        dialog.findChild(QLabel, "fileInfoTitle").setText("long-subject-name_" * 20 + ".fif")
        dialog.resize(680, 660)
        self.app.processEvents()

        self.assertEqual(dialog.width(), 680)
        self.assertEqual(dialog.details_splitter.orientation(), Qt.Orientation.Vertical)
        self.assertEqual(dialog.path_field.text(), path)
        dialog.path_field.selectAll()
        self.assertEqual(dialog.path_field.selectedText(), path)
        self.assertGreater(dialog.processing_log_section.y(), dialog.details_scroll.y())
        self.assertTrue(dialog.rect().contains(dialog.delete_button.mapTo(dialog, dialog.delete_button.rect().bottomRight())))

    def test_dialog_does_not_hold_an_open_data_object(self):
        raw = mne.io.RawArray(np.zeros((1, 100)), mne.create_info(["C3"], 100, "eeg"), verbose=False)
        reference = weakref.ref(raw)
        dialog = self.dialog(raw)
        del raw
        gc.collect()

        self.assertIsNone(reference())
        dialog.release_object()
        self.assertEqual(dialog.summary_widget.metric_labels, [])


if __name__ == "__main__":
    unittest.main()
