import gc
import os
import unittest
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QApplication

from ui.widgets.tools.compact_object_info import CompactObjectInfoWidget


class TestCompactObjectInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widget = CompactObjectInfoWidget()
        self.addCleanup(self.widget.close)

    def metrics(self):
        texts = []
        for label in self.widget.metric_labels:
            document = QTextDocument()
            document.setHtml(label.text())
            texts.append(document.toPlainText())
        return texts

    def test_raw_summary_preserves_metrics_and_annotation_counts(self):
        info = mne.create_info(["C3", "C4"], 100, "eeg")
        raw = mne.io.RawArray(np.zeros((2, 100)), info, verbose=False)
        raw.set_annotations(mne.Annotations([0.1, 0.2, 0.3], [0, 0, 0], ["Go", "Go", "<target>"]))

        self.widget.update_info(raw)

        self.assertEqual(
            self.metrics(),
            ["2 channels", "100 time points", "100.00 Hz sampling", "0.99 s duration"],
        )
        self.assertEqual(self.widget.events_label.text(), "Events · <target>: 1 · Go: 2")
        self.assertFalse(self.widget.isHidden())

    def test_epochs_summary_preserves_retention_bad_channels_and_events(self):
        info = mne.create_info(["C3", "C4"], 100, "eeg")
        info["bads"] = ["C4"]
        epochs = mne.EpochsArray(
            np.zeros((3, 2, 21)),
            info,
            events=np.array([[100, 0, 1], [200, 0, 1], [300, 0, 2]]),
            event_id={"target <A>": 1, "other": 2},
            tmin=-0.1,
            verbose=False,
        )
        epochs.drop([1], verbose=False)

        self.widget.update_info(epochs)

        self.assertEqual(
            self.metrics(),
            ["2/3 epochs", "1/2 channels", "100.00 Hz sampling", "-100.00 ms to 100.00 ms window"],
        )
        self.assertEqual(self.widget.events_label.text(), "Events · target <A>: 1 · other: 1")

    def test_ica_summary_preserves_components_exclusions_and_method(self):
        ica = mne.preprocessing.ICA(n_components=2, method="fastica")
        ica.n_components_ = 2
        ica.exclude = [1]
        ica.n_samples_ = 200

        self.widget.update_info(ica)

        self.assertEqual(
            self.metrics(),
            ["1/2 components", "1 excluded", "200 samples", "FASTICA method"],
        )
        self.assertTrue(self.widget.events_label.isHidden())

    def test_clearing_removes_old_details_and_does_not_retain_data(self):
        raw = mne.io.RawArray(
            np.zeros((1, 10)), mne.create_info(["C3"], 100, "eeg"), verbose=False
        )
        reference = weakref.ref(raw)
        self.widget.update_info(raw)
        del raw
        gc.collect()
        self.assertIsNone(reference())

        self.widget.update_info(None)

        self.assertTrue(self.widget.isHidden())
        self.assertEqual(self.widget.metrics_layout.count(), 0)
        self.assertEqual(self.widget.events_label.text(), "")

    def test_metrics_wrap_and_expand_without_overlap(self):
        raw = mne.io.RawArray(
            np.zeros((2, 100)), mne.create_info(["C3", "C4"], 100, "eeg"), verbose=False
        )
        self.widget.update_info(raw)
        self.widget.resize(900, 100)
        self.widget.show()
        self.app.processEvents()
        wide_height = self.widget.metrics_layout.heightForWidth(900)

        self.widget.resize(220, 180)
        self.app.processEvents()

        self.assertGreater(self.widget.metrics_layout.heightForWidth(220), wide_height)
        for index, label in enumerate(self.widget.metric_labels):
            self.assertTrue(self.widget.metrics_widget.rect().contains(label.geometry()))
            for other in self.widget.metric_labels[index + 1:]:
                self.assertFalse(label.geometry().intersects(other.geometry()))

    def test_wide_epoch_window_stays_on_one_line_and_wraps_when_narrow(self):
        epochs = mne.EpochsArray(
            np.zeros((2, 2, 151)),
            mne.create_info(["C3", "C4"], 200, "eeg"),
            tmin=-0.2,
            verbose=False,
        )
        self.widget.update_info(epochs)
        self.widget.resize(1100, 160)
        self.widget.show()
        self.app.processEvents()
        window = self.widget.metric_labels[-1]

        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(window.font())
        document.setHtml(window.text())
        natural_height = document.size().height()
        self.assertGreaterEqual(window.width(), document.idealWidth())
        self.assertLessEqual(window.height(), natural_height + 1)

        self.widget.resize(180, 300)
        self.app.processEvents()

        self.assertLess(window.width(), document.idealWidth())
        self.assertGreater(window.height(), natural_height + 1)
        self.assertTrue(self.widget.metrics_widget.rect().contains(window.geometry()))


if __name__ == "__main__":
    unittest.main()
