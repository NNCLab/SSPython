from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QSettings

SETTINGS_SCHEMA_VERSION = 2

DEFAULT_THEME = "dark"
DEFAULT_OUTPUT_ROOT = "derivatives"
DEFAULT_PIPELINE_ID = "tms_eeg"


class SettingsStore:
    """A thin, schema-aware wrapper around ``QSettings``."""

    LEGACY_MAPPINGS: dict[str, tuple[str, ...]] = {
        "appearance/theme": ("theme",),
        "workspace/current_folder": ("current_folder",),
        "workspace/output_root": ("global_settings/output_dir",),
        "appearance/plots/global": ("plot_settings/plot_params",),
        "appearance/plots/psd": ("plot_settings/psd_plot_params",),
        "pipelines/tms_eeg/preprocessing/artifact_removal": (
            "erp_preprocessing/artifact_removal",
        ),
        "pipelines/tms_eeg/preprocessing/continuous_filter": (
            "erp_preprocessing/continuous_filter",
        ),
        "pipelines/tms_eeg/preprocessing/epoching": ("erp_preprocessing/epoching",),
        "pipelines/tms_eeg/preprocessing/re_reference": (
            "erp_preprocessing/re_reference",
        ),
        "pipelines/tms_eeg/preprocessing/apply": ("erp_preprocessing/apply_params",),
        "pipelines/tms_eeg/ica/continuous": ("ica/continuous",),
        "pipelines/tms_eeg/ica/epochs": ("ica/epochs",),
        "pipelines/tms_eeg/ica/plotting": ("ica/plotting",),
        "real_time/connection": ("real_time/connection",),
        "real_time/plotting": ("real_time/plot_settings",),
    }

    def __init__(self, settings: QSettings | None = None):
        self.settings = settings or QSettings()
        self.migrate()

    def migrate(self):
        version = self.settings.value("app/schema_version", 0, type=int)
        if version >= SETTINGS_SCHEMA_VERSION:
            return

        for new_key, legacy_keys in self.LEGACY_MAPPINGS.items():
            if self.settings.contains(new_key):
                continue
            for legacy_key in legacy_keys:
                if self.settings.contains(legacy_key):
                    self.settings.setValue(new_key, self.settings.value(legacy_key))
                    break

        if not self.settings.contains("workspace/output_root"):
            self.settings.setValue("workspace/output_root", DEFAULT_OUTPUT_ROOT)
        if not self.settings.contains("workspace/current_pipeline"):
            self.settings.setValue("workspace/current_pipeline", DEFAULT_PIPELINE_ID)
        if not self.settings.contains("appearance/theme"):
            self.settings.setValue("appearance/theme", DEFAULT_THEME)

        self.settings.setValue("app/schema_version", SETTINGS_SCHEMA_VERSION)
        self.settings.sync()

    def get(
        self,
        key: str,
        default: Any = None,
        *,
        legacy_keys: Iterable[str] = (),
        value_type: type | None = None,
    ) -> Any:
        if self.settings.contains(key):
            if value_type is None:
                return self.settings.value(key, default)
            return self.settings.value(key, default, type=value_type)

        for legacy_key in legacy_keys:
            if self.settings.contains(legacy_key):
                if value_type is None:
                    value = self.settings.value(legacy_key, default)
                else:
                    value = self.settings.value(legacy_key, default, type=value_type)
                self.settings.setValue(key, value)
                return value

        mapped_legacy = self.LEGACY_MAPPINGS.get(key, ())
        for legacy_key in mapped_legacy:
            if self.settings.contains(legacy_key):
                if value_type is None:
                    value = self.settings.value(legacy_key, default)
                else:
                    value = self.settings.value(legacy_key, default, type=value_type)
                self.settings.setValue(key, value)
                return value

        return default

    def set(self, key: str, value: Any):
        self.settings.setValue(key, value)

    def contains(self, key: str) -> bool:
        return self.settings.contains(key)

    def remove(self, key: str):
        self.settings.remove(key)

    def sync(self):
        self.settings.sync()

    def clear(self):
        self.settings.clear()
        self.settings.setValue("app/schema_version", SETTINGS_SCHEMA_VERSION)
        self.settings.setValue("appearance/theme", DEFAULT_THEME)
        self.settings.setValue("workspace/output_root", DEFAULT_OUTPUT_ROOT)
        self.settings.setValue("workspace/current_pipeline", DEFAULT_PIPELINE_ID)
        self.settings.sync()

    def get_path(self, key: str, *, legacy_keys: Iterable[str] = ()) -> Path | None:
        raw_value = self.get(key, None, legacy_keys=legacy_keys)
        if not raw_value:
            return None
        return Path(raw_value)

    def output_root(self) -> str:
        return self.get("workspace/output_root", DEFAULT_OUTPUT_ROOT)

    def current_pipeline_id(self) -> str:
        return self.get("workspace/current_pipeline", DEFAULT_PIPELINE_ID)

    def set_current_pipeline_id(self, pipeline_id: str):
        self.set("workspace/current_pipeline", pipeline_id)

    def current_folder(self) -> Path | None:
        return self.get_path(
            "workspace/current_folder",
            legacy_keys=("current_folder",),
        )

    def set_current_folder(self, folder_path: str | Path | None):
        if folder_path is None:
            self.remove("workspace/current_folder")
            return
        self.set("workspace/current_folder", str(folder_path))

    def pipeline_path(self, section: str, pipeline_id: str | None = None) -> str:
        active_pipeline_id = pipeline_id or self.current_pipeline_id()
        return f"pipelines/{active_pipeline_id}/{section}"


def get_settings_store(settings: QSettings | None = None) -> SettingsStore:
    return SettingsStore(settings)
