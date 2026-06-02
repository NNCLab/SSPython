from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QSettings

SETTINGS_SCHEMA_VERSION = 3

DEFAULT_THEME = "dark"
DEFAULT_OUTPUT_ROOT = "derivatives"
DEFAULT_PIPELINE_ID = "standard"

LEGACY_PIPELINE_ID_ALIASES = {
    "tms_eeg": DEFAULT_PIPELINE_ID,
    "continuous_eeg": DEFAULT_PIPELINE_ID,
}

PIPELINE_SETTING_LEGACY_KEYS: dict[str, tuple[str, ...]] = {
    "preprocessing/artifact_removal": ("erp_preprocessing/artifact_removal",),
    "preprocessing/continuous_filter": ("erp_preprocessing/continuous_filter",),
    "preprocessing/epoching": ("erp_preprocessing/epoching",),
    "preprocessing/re_reference": ("erp_preprocessing/re_reference",),
    "preprocessing/apply": ("erp_preprocessing/apply_params",),
    "ica/continuous": ("ica/continuous",),
    "ica/epochs": ("ica/epochs",),
    "ica/plotting": ("ica/plotting",),
}


class SettingsStore:
    """A thin, schema-aware wrapper around ``QSettings``."""

    LEGACY_MAPPINGS: dict[str, tuple[str, ...]] = {
        "appearance/theme": ("theme",),
        "workspace/current_folder": ("current_folder",),
        "workspace/output_root": ("global_settings/output_dir",),
        "appearance/plots/global": ("plot_settings/plot_params",),
        "appearance/plots/psd": ("plot_settings/psd_plot_params",),
        "pipelines/standard/preprocessing/artifact_removal": (
            "pipelines/tms_eeg/preprocessing/artifact_removal",
            "pipelines/continuous_eeg/preprocessing/artifact_removal",
            "erp_preprocessing/artifact_removal",
        ),
        "pipelines/standard/preprocessing/continuous_filter": (
            "pipelines/tms_eeg/preprocessing/continuous_filter",
            "pipelines/continuous_eeg/preprocessing/continuous_filter",
            "erp_preprocessing/continuous_filter",
        ),
        "pipelines/standard/preprocessing/epoching": (
            "pipelines/tms_eeg/preprocessing/epoching",
            "pipelines/continuous_eeg/preprocessing/epoching",
            "erp_preprocessing/epoching",
        ),
        "pipelines/standard/preprocessing/re_reference": (
            "pipelines/tms_eeg/preprocessing/re_reference",
            "pipelines/continuous_eeg/preprocessing/re_reference",
            "erp_preprocessing/re_reference",
        ),
        "pipelines/standard/preprocessing/apply": (
            "pipelines/tms_eeg/preprocessing/apply",
            "pipelines/continuous_eeg/preprocessing/apply",
            "erp_preprocessing/apply_params",
        ),
        "pipelines/standard/ica/continuous": (
            "pipelines/tms_eeg/ica/continuous",
            "pipelines/continuous_eeg/ica/continuous",
            "ica/continuous",
        ),
        "pipelines/standard/ica/epochs": (
            "pipelines/tms_eeg/ica/epochs",
            "pipelines/continuous_eeg/ica/epochs",
            "ica/epochs",
        ),
        "pipelines/standard/ica/plotting": (
            "pipelines/tms_eeg/ica/plotting",
            "pipelines/continuous_eeg/ica/plotting",
            "ica/plotting",
        ),
        "real_time/connection": ("real_time/connection",),
        "real_time/plotting": ("real_time/plot_settings",),
    }

    def __init__(self, settings: QSettings | None = None):
        self.settings = settings or QSettings()
        self.migrate()

    def migrate(self):
        version = self.settings.value("app/schema_version", 0, type=int)
        if version >= SETTINGS_SCHEMA_VERSION:
            self._normalize_current_pipeline_setting()
            return

        self._migrate_standard_pipeline_settings()

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
        else:
            self._normalize_current_pipeline_setting()
        if not self.settings.contains("appearance/theme"):
            self.settings.setValue("appearance/theme", DEFAULT_THEME)

        self.settings.setValue("app/schema_version", SETTINGS_SCHEMA_VERSION)
        self.settings.sync()

    def _legacy_pipeline_ids_for_migration(self) -> tuple[str, ...]:
        current_pipeline = self.settings.value("workspace/current_pipeline", "")
        ordered_ids: list[str] = []
        if current_pipeline in LEGACY_PIPELINE_ID_ALIASES:
            ordered_ids.append(current_pipeline)
        for pipeline_id in LEGACY_PIPELINE_ID_ALIASES:
            if pipeline_id not in ordered_ids:
                ordered_ids.append(pipeline_id)
        return tuple(ordered_ids)

    def _migrate_standard_pipeline_settings(self):
        legacy_pipeline_ids = self._legacy_pipeline_ids_for_migration()
        for section, legacy_keys in PIPELINE_SETTING_LEGACY_KEYS.items():
            standard_key = f"pipelines/{DEFAULT_PIPELINE_ID}/{section}"
            if self.settings.contains(standard_key):
                continue

            candidate_keys = tuple(
                f"pipelines/{pipeline_id}/{section}" for pipeline_id in legacy_pipeline_ids
            ) + legacy_keys
            for candidate_key in candidate_keys:
                if self.settings.contains(candidate_key):
                    self.settings.setValue(standard_key, self.settings.value(candidate_key))
                    break

    def _normalize_pipeline_id(self, pipeline_id: str | None) -> str:
        if not pipeline_id:
            return DEFAULT_PIPELINE_ID
        return LEGACY_PIPELINE_ID_ALIASES.get(pipeline_id, pipeline_id)

    def _normalize_current_pipeline_setting(self):
        current_pipeline = self.settings.value("workspace/current_pipeline", DEFAULT_PIPELINE_ID)
        normalized_pipeline = self._normalize_pipeline_id(str(current_pipeline) if current_pipeline else None)
        if current_pipeline != normalized_pipeline:
            self.settings.setValue("workspace/current_pipeline", normalized_pipeline)

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
        pipeline_id = self.get("workspace/current_pipeline", DEFAULT_PIPELINE_ID)
        normalized_pipeline_id = self._normalize_pipeline_id(str(pipeline_id) if pipeline_id else None)
        if pipeline_id != normalized_pipeline_id:
            self.set("workspace/current_pipeline", normalized_pipeline_id)
        return normalized_pipeline_id

    def set_current_pipeline_id(self, pipeline_id: str):
        self.set("workspace/current_pipeline", self._normalize_pipeline_id(pipeline_id))

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
        active_pipeline_id = self._normalize_pipeline_id(pipeline_id) if pipeline_id else self.current_pipeline_id()
        return f"pipelines/{active_pipeline_id}/{section}"


def get_settings_store(settings: QSettings | None = None) -> SettingsStore:
    return SettingsStore(settings)
