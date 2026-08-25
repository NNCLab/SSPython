"""Compatibility fixes for supported MNE dependency combinations."""

from __future__ import annotations

import inspect
import logging
from functools import wraps

import mne


logger = logging.getLogger(__name__)

_PICKER_COMPAT_MARKER = "_sspython_mapper_dataset_compat"
_EMPTY_LABEL_COMPAT_MARKER = "_sspython_empty_label_compat"


class _PickerDatasetProxy:
    """Delegate to a VTK picker while returning a canonical dataset."""

    def __init__(self, picker, dataset):
        self._picker = picker
        self._dataset = dataset

    def GetDataSet(self):
        return self._dataset

    def __getattr__(self, name):
        return getattr(self._picker, name)


def _brain_callback_has_mapper_dataset_fix(callback) -> bool:
    """Return whether an MNE callback includes the upstream PyVista fix."""
    try:
        return "mapper_dataset" in inspect.getsource(callback)
    except (OSError, TypeError):
        # PyInstaller preserves this MNE source file. If another distribution
        # does not, limit the fallback to the affected MNE release series.
        return not mne.__version__.startswith("1.11.")


def install_mne_brain_vertex_picker_compat() -> bool:
    """Backport MNE PR #14202 when the installed MNE lacks its fix.

    Recent PyVista versions can return an internal mesh copy from
    ``vtkCellPicker.GetDataSet()`` while the mapper still references MNE's
    original cortical mesh. MNE 1.11 then treats a surface click as a volume
    click and silently ignores it. This wrapper substitutes the mapper's
    canonical mesh before delegating to MNE's original callback.

    Returns
    -------
    bool
        True when the compatibility wrapper was installed, otherwise False.
    """
    brain_class = mne.viz.Brain
    original_on_pick = brain_class._on_pick

    if getattr(original_on_pick, _PICKER_COMPAT_MARKER, False):
        return False
    if _brain_callback_has_mapper_dataset_fix(original_on_pick):
        return False

    @wraps(original_on_pick)
    def _on_pick_with_mapper_dataset(self, picker, event):
        mapper = picker.GetMapper()
        mapper_dataset = getattr(mapper, "dataset", None)
        layered_meshes = getattr(self, "_layered_meshes", None)
        if layered_meshes is None:
            layered_meshes = getattr(self, "layered_meshes", {})

        for layered_mesh in layered_meshes.values():
            if layered_mesh._polydata is mapper_dataset:
                picker = _PickerDatasetProxy(picker, mapper_dataset)
                break

        return original_on_pick(self, picker, event)

    setattr(_on_pick_with_mapper_dataset, _PICKER_COMPAT_MARKER, True)
    brain_class._on_pick = _on_pick_with_mapper_dataset
    logger.info("Installed MNE Brain vertex-picking compatibility fix")
    return True


def install_mne_brain_empty_label_compat() -> bool:
    """Ignore annotation labels that contain no sampled source vertices.

    MNE maps clicks using the full-resolution cortical annotation, while an
    STC commonly uses a sparse source space. Some annotation regions (most
    often ``unknown``) can therefore contain no STC samples. MNE 1.11 raises
    from its Qt callback in that case; an empty region should simply add no
    trace.

    Returns
    -------
    bool
        True when the compatibility wrapper was installed, otherwise False.
    """
    brain_class = mne.viz.Brain
    original_add_label_glyph = brain_class._add_label_glyph
    if getattr(original_add_label_glyph, _EMPTY_LABEL_COMPAT_MARKER, False):
        return False

    @wraps(original_add_label_glyph)
    def _add_nonempty_label_glyph(self, hemi, mesh, vertex_id):
        try:
            return original_add_label_glyph(self, hemi, mesh, vertex_id)
        except ValueError as exc:
            if "source space does not contain any vertices" not in str(exc):
                raise
            logger.info("Ignoring empty source-space label selected in MNE Brain")
            return None

    setattr(_add_nonempty_label_glyph, _EMPTY_LABEL_COMPAT_MARKER, True)
    brain_class._add_label_glyph = _add_nonempty_label_glyph
    logger.info("Installed MNE Brain empty-label compatibility fix")
    return True
