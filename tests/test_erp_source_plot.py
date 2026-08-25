import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from ui.pages.erp_analysis_page import ErpAnalysisPage
from ui.widgets.source_estimate_widget import STCSourceModel


class _ImmediateWorker:
    def __init__(self, function, **kwargs):
        self.function = function

    def exec_with_dialog(self, *args, **kwargs):
        return self.function()


class _FakeSourceEstimate:
    subject = "fsaverage"
    vertices = [np.array([1], dtype=int), np.array([], dtype=int)]

    def __init__(self):
        self.plot_kwargs = None
        self.brain = object()

    def get_peak(self, hemi):
        return 1, 0.1

    def plot(self, **kwargs):
        self.plot_kwargs = kwargs
        return self.brain


class TestErpSourcePlot(unittest.TestCase):
    def test_plot_stc_passes_source_spaces_and_requires_pyvistaqt(self):
        app = QApplication.instance() or QApplication([])
        page = ErpAnalysisPage()
        stc = _FakeSourceEstimate()
        page.tep = SimpleNamespace(
            label="Test dataset",
            derivatives={"stc": stc, "stc_metadata": None},
        )
        source_model = STCSourceModel(
            mode="fsaverage",
            subject="fsaverage",
            subjects_dir=Path("subjects"),
            src=Path("fsaverage-ico-5-src.fif"),
            bem=Path("fsaverage-bem-sol.fif"),
            trans="fsaverage",
        )
        source_spaces = object()

        with (
            patch(
                "ui.pages.erp_analysis_page.prepare_stc_source_model",
                return_value=source_model,
            ),
            patch(
                "ui.pages.erp_analysis_page.mne.read_source_spaces",
                return_value=source_spaces,
            ) as read_source_spaces,
            patch("ui.pages.erp_analysis_page.Worker", _ImmediateWorker),
            patch(
                "ui.pages.erp_analysis_page.install_mne_brain_vertex_picker_compat"
            ),
            patch(
                "ui.pages.erp_analysis_page.install_mne_brain_empty_label_compat"
            ),
        ):
            page.plot_source_estimate()

        read_source_spaces.assert_called_once_with(source_model.src, verbose=False)
        self.assertIs(stc.plot_kwargs["src"], source_spaces)
        self.assertEqual(stc.plot_kwargs["backend"], "pyvistaqt")
        self.assertIs(page.source_estimate_brain, stc.brain)
        page.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
