import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_artifact_suggestions_keep_application_store_after_mne_browser_init(tmp_path):
    """MNE changes Qt's global identity; saved SSPython suggestions must stay put."""
    # A subprocess keeps Qt's global settings paths and application identity out of
    # other tests, while all settings remain in temporary INI files.
    script = textwrap.dedent("""
        import json
        from pathlib import Path
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication

        from core.app_settings import SettingsStore, get_settings_store
        from ui.pages.preferences_page import PreferencesPage
        from ui.widgets.preprocessing_widgets import ArtifactRemovalDialog
        from mne.viz.backends._utils import _init_mne_qtapp

        root = Path(sys.argv[1]).resolve()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root))
        QSettings.setPath(
            QSettings.Format.IniFormat, QSettings.Scope.SystemScope, str(root / 'system'),
        )
        app = QApplication([])
        app.setOrganizationName('SSPython')
        app.setApplicationName('SSPython')

        # Guard against a factory accidentally selecting registry-backed settings:
        # inspect its path before letting migration write anything.
        with patch.object(SettingsStore, 'migrate', lambda self: None):
            store = get_settings_store()
        assert Path(store.settings.fileName()).resolve().is_relative_to(root)
        store.migrate()
        key = 'pipelines/standard/preprocessing/artifact_removal'
        store.set(key, {'window': (-0.002, 0.006)})
        store.sync()

        foreign = QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, 'MNE', 'MNE-Python',
        )
        foreign.setValue(key, {'window': (-0.002, 0.010)})
        foreign.sync()

        with patch('ui.pages.preferences_page.QMessageBox.information'), patch(
            'ui.pages.preferences_page.apply_theme', lambda: None,
        ):
            page = PreferencesPage(store)
            page.artifact_widget.window_edit.setValue((-0.002, 0.006))
            page.save_all_settings()

        saved = {name: store.get(name) for name in store.settings.allKeys()}
        preference_path = page.artifact_widget.settings_store.settings.fileName()
        _init_mne_qtapp(enable_icon=False)
        assert (app.organizationName(), app.applicationName()) == ('MNE', 'MNE-Python')

        runs = []
        processor = SimpleNamespace(
            has=lambda stage: False, paths={},
            run_artifact_removal=lambda **params: runs.append(params),
        )
        dialog = ArtifactRemovalDialog(processor)
        dialog_path = dialog.settings_widget.settings_store.settings.fileName()
        opening_window = dialog.settings_widget.window_edit.value()
        independent_store = get_settings_store()

        class ImmediateWorker:
            def __init__(self, operation, **kwargs):
                self.operation = operation
                self._error = None

            def exec_with_dialog(self, *args, **kwargs):
                return self.operation()

        dialog.settings_widget.window_edit.setValue((-0.002, 0.008))
        with patch(
            'ui.widgets.preprocessing_widgets.confirm_processing_update',
            lambda *args, **kwargs: True,
        ), patch('ui.widgets.preprocessing_widgets.Worker', ImmediateWorker):
            dialog.run_process()
        reopened = ArtifactRemovalDialog(processor)
        result = {
            'preferences_window': page.artifact_widget.window_edit.value(),
            'dialog_window': opening_window,
            'reopened_window': reopened.settings_widget.window_edit.value(),
            'run_window': runs[0]['window'],
            'same_path': preference_path == dialog_path == independent_store.settings.fileName(),
            'independent_instances': independent_store is not store,
            'canonical_unchanged': saved == {
                name: store.get(name) for name in store.settings.allKeys()
            },
            'foreign_window': foreign.value(key)['window'],
        }
        reopened.close()
        dialog.close()
        page.close()
        print(json.dumps(result))
    """)
    environment = os.environ.copy()
    environment.update({
        "QT_QPA_PLATFORM": "offscreen",
        "_MNE_FAKE_HOME_DIR": str(tmp_path / "mne-home"),
        "MPLCONFIGDIR": str(tmp_path / "mpl-config"),
    })
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "settings")],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["same_path"]
    assert observed["independent_instances"]
    assert observed["preferences_window"] == [-0.002, 0.006]
    assert observed["dialog_window"] == [-0.002, 0.006]
    assert observed["reopened_window"] == [-0.002, 0.006]
    assert observed["run_window"] == [-0.002, 0.008]
    assert observed["canonical_unchanged"]
    assert observed["foreign_window"] == [-0.002, 0.010]
