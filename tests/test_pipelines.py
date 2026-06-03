import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QSettings

from core.app_settings import SettingsStore
from core.pipelines import (
    PipelineDefinition,
    PipelineStage,
    all_pipelines,
    discover_datasets,
    get_pipeline,
)


class TestPipelines(unittest.TestCase):
    def test_catalog_exposes_standard_pipeline(self):
        pipelines = all_pipelines()

        self.assertEqual([pipeline.id for pipeline in pipelines], ["standard"])
        self.assertEqual(pipelines[0].name, "Standard")
        self.assertTrue(pipelines[0].workflow_sections)

    def test_legacy_pipeline_ids_resolve_to_standard(self):
        self.assertEqual(get_pipeline("tms_eeg").id, "standard")
        self.assertEqual(get_pipeline("continuous_eeg").id, "standard")

    def test_standard_pipeline_uses_legacy_derivatives_when_they_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            raw_dir = workspace_root / "sub-01" / "eeg"
            raw_dir.mkdir(parents=True)
            raw_path = raw_dir / "sub-01_task-tms_raw.fif"
            raw_path.touch()

            legacy_derivative = (
                workspace_root
                / "derivatives"
                / "tms_eeg"
                / "sub-01"
                / "eeg"
                / "sub-01_task-tms_desc-epoched_epo.fif"
            )
            legacy_derivative.parent.mkdir(parents=True)
            legacy_derivative.touch()

            datasets = discover_datasets(workspace_root, get_pipeline("standard"), "derivatives")

            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0].derivative_root, workspace_root / "derivatives" / "tms_eeg")
            self.assertEqual(datasets[0].paths["epochs"], legacy_derivative)

    def test_custom_pipeline_stage_can_define_derivative_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            raw_dir = workspace_root / "sub-01" / "eeg"
            raw_dir.mkdir(parents=True)
            raw_path = raw_dir / "sub-01_task-rest_raw.fif"
            raw_path.touch()

            pipeline = PipelineDefinition(
                id="custom",
                name="Custom",
                summary="Custom workflow",
                accent_color="#123456",
                stages=(
                    PipelineStage("raw", "Raw", "Raw", "Source", data_kind="raw"),
                    PipelineStage(
                        "clean_epochs",
                        "Clean Epochs",
                        "Clean",
                        "Custom cleaned epochs.",
                        data_kind="epochs",
                        derivative_desc="clean",
                        derivative_suffix="epo",
                    ),
                ),
                workflow_sections=(),
                analysis_ready_stage="clean_epochs",
            )

            datasets = discover_datasets(workspace_root, pipeline, "derivatives")

            self.assertEqual(len(datasets), 1)
            self.assertEqual(
                datasets[0].paths["clean_epochs"],
                workspace_root
                / "derivatives"
                / "custom"
                / "sub-01"
                / "eeg"
                / "sub-01_task-rest_desc-clean_epo.fif",
            )

    def test_custom_pipeline_can_override_standard_stage_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            raw_dir = workspace_root / "sub-01" / "eeg"
            raw_dir.mkdir(parents=True)
            raw_path = raw_dir / "sub-01_task-rest_raw.fif"
            raw_path.touch()

            pipeline = PipelineDefinition(
                id="custom",
                name="Custom",
                summary="Custom workflow",
                accent_color="#123456",
                stages=(
                    PipelineStage("raw", "Raw", "Raw", "Source", data_kind="raw"),
                    PipelineStage(
                        "epochs",
                        "Custom Epochs",
                        "Epochs",
                        "Custom epoch output.",
                        data_kind="epochs",
                        derivative_desc="custom",
                        derivative_suffix="epo",
                    ),
                ),
                workflow_sections=(),
                analysis_ready_stage="epochs",
            )

            datasets = discover_datasets(workspace_root, pipeline, "derivatives")

            self.assertEqual(
                datasets[0].paths["epochs"],
                workspace_root
                / "derivatives"
                / "custom"
                / "sub-01"
                / "eeg"
                / "sub-01_task-rest_desc-custom_epo.fif",
            )

    def test_settings_migrate_active_legacy_pipeline_to_standard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = str(Path(tmp_dir) / "settings.ini")
            settings = QSettings(settings_path, QSettings.Format.IniFormat)
            settings.clear()
            settings.setValue("workspace/current_pipeline", "continuous_eeg")
            settings.setValue(
                "pipelines/continuous_eeg/preprocessing/epoching",
                {"mode": "fixed", "fixed_duration": 3.0},
            )
            settings.sync()

            store = SettingsStore(settings)

            self.assertEqual(store.current_pipeline_id(), "standard")
            self.assertEqual(
                store.get("pipelines/standard/preprocessing/epoching")["mode"],
                "fixed",
            )


if __name__ == "__main__":
    unittest.main()
