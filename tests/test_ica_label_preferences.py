import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog

from core.app_settings import SettingsStore
from ui.widgets.auto_label import (
    EXCLUDABLE_LABELS,
    ICALabelingDialog,
    ICALabelingSettingsWidget,
)


@pytest.fixture
def settings_store(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    store = SettingsStore(QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat,
    ))
    monkeypatch.setattr("ui.widgets.auto_label.get_settings_store", lambda: store)
    yield store
    app.processEvents()


def snapshot(store):
    return {key: store.get(key) for key in store.settings.allKeys()}


def test_expert_label_suggestions_save_using_existing_keys(settings_store):
    settings_store.set("ica_label/exclude/eye blink", False)
    widget = ICALabelingSettingsWidget()
    assert not widget.label_checkboxes["eye blink"].isChecked()
    assert widget.label_checkboxes["muscle artifact"].isChecked()
    selected = {"eye blink", "line noise"}
    for label, checkbox in widget.label_checkboxes.items():
        checkbox.setChecked(label in selected)
    widget.save_settings()

    assert widget.get_settings() == {
        label: label in selected for label in EXCLUDABLE_LABELS
    }
    for label in EXCLUDABLE_LABELS:
        assert settings_store.get(f"ica_label/exclude/{label}") == (label in selected)
    reloaded = ICALabelingSettingsWidget()
    assert reloaded.get_settings() == widget.get_settings()
    widget.close()
    reloaded.close()


def test_label_dialog_accept_keeps_run_edits_out_of_preferences(settings_store):
    suggestions = {"muscle artifact", "eye blink"}
    expert = ICALabelingSettingsWidget()
    for label, checkbox in expert.label_checkboxes.items():
        checkbox.setChecked(label in suggestions)
    expert.save_settings()
    saved = snapshot(settings_store)

    with patch.object(ICALabelingDialog, "run_and_populate", lambda self: None):
        dialog = ICALabelingDialog(None, None)
        assert dialog.get_selected_labels_to_exclude() == suggestions
        dialog.label_checkboxes["eye blink"].setChecked(False)
        dialog.label_checkboxes["line noise"].setChecked(True)
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.get_selected_labels_to_exclude() == {"muscle artifact", "line noise"}
        assert snapshot(settings_store) == saved
        reopened = ICALabelingDialog(None, None)
        assert reopened.get_selected_labels_to_exclude() == suggestions
        reopened.reject()

    assert snapshot(settings_store) == saved
    expert.close()
    dialog.close()
    reopened.close()
