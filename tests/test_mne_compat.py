import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.mne_compat import (
    install_mne_brain_empty_label_compat,
    install_mne_brain_vertex_picker_compat,
)


class _FakePicker:
    def __init__(self, picked_dataset, mapper_dataset):
        self._picked_dataset = picked_dataset
        self._mapper = SimpleNamespace(dataset=mapper_dataset)

    def GetDataSet(self):
        return self._picked_dataset

    def GetMapper(self):
        return self._mapper

    def GetCellId(self):
        return 42


class TestMneCompat(unittest.TestCase):
    def test_vertex_picker_uses_mapper_dataset_and_remains_idempotent(self):
        original_surface = object()
        picked_copy = object()

        class AffectedBrain:
            def __init__(self):
                self._layered_meshes = {
                    "lh": SimpleNamespace(_polydata=original_surface)
                }
                self.received_dataset = None
                self.received_cell_id = None

            def _on_pick(self, picker, event):
                self.received_dataset = picker.GetDataSet()
                self.received_cell_id = picker.GetCellId()

        with patch("core.mne_compat.mne.viz.Brain", AffectedBrain):
            self.assertTrue(install_mne_brain_vertex_picker_compat())
            self.assertFalse(install_mne_brain_vertex_picker_compat())

            brain = AffectedBrain()
            picker = _FakePicker(picked_copy, original_surface)
            brain._on_pick(picker, None)

        self.assertIs(brain.received_dataset, original_surface)
        self.assertEqual(brain.received_cell_id, 42)
        self.assertIs(picker.GetDataSet(), picked_copy)

    def test_vertex_picker_does_not_patch_an_upstream_fixed_callback(self):
        class FixedBrain:
            def _on_pick(self, picker, event):
                mapper_dataset = getattr(picker.GetMapper(), "dataset", None)
                return mapper_dataset

        original_callback = FixedBrain._on_pick
        with patch("core.mne_compat.mne.viz.Brain", FixedBrain):
            self.assertFalse(install_mne_brain_vertex_picker_compat())

        self.assertIs(FixedBrain._on_pick, original_callback)

    def test_empty_annotation_label_is_ignored(self):
        class AffectedBrain:
            def _add_label_glyph(self, hemi, mesh, vertex_id):
                raise ValueError(
                    "source space does not contain any vertices for 1 label"
                )

        with patch("core.mne_compat.mne.viz.Brain", AffectedBrain):
            self.assertTrue(install_mne_brain_empty_label_compat())
            self.assertFalse(install_mne_brain_empty_label_compat())
            self.assertIsNone(AffectedBrain()._add_label_glyph("lh", object(), 1))

    def test_unrelated_label_error_is_not_suppressed(self):
        class AffectedBrain:
            def _add_label_glyph(self, hemi, mesh, vertex_id):
                raise ValueError("unrelated label failure")

        with patch("core.mne_compat.mne.viz.Brain", AffectedBrain):
            install_mne_brain_empty_label_compat()
            with self.assertRaisesRegex(ValueError, "unrelated label failure"):
                AffectedBrain()._add_label_glyph("lh", object(), 1)


if __name__ == "__main__":
    unittest.main()
