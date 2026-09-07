import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QSettings

from core.app_settings import SettingsStore
from core.pipelines import (
    PipelineDefinition,
    PipelineStage,
    all_pipelines,
    build_analysis_paths,
    discover_analysis_datasets,
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

    def test_discovers_preprocessed_epochs_without_raw_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            preprocessed_path = (
                workspace_root
                / "imported"
                / "sub-01_task-tms_desc-preprocessed_epo.fif"
            )
            preprocessed_path.parent.mkdir(parents=True)
            preprocessed_path.touch()

            datasets = discover_analysis_datasets(
                workspace_root, get_pipeline("standard"), "derivatives"
            )

            self.assertEqual(len(datasets), 1)
            self.assertTrue(datasets[0].is_analysis_only)
            self.assertIsNone(datasets[0].raw_path)
            self.assertEqual(datasets[0].source_path, preprocessed_path)
            self.assertTrue(datasets[0].stage_exists("preprocessed"))
            self.assertEqual(datasets[0].status_text(), "Ready")
            self.assertEqual(datasets[0].progress_fraction(), 1.0)
            self.assertEqual(
                datasets[0].completed_stage_count(),
                len(datasets[0].pipeline.stages),
            )

    def test_preprocessed_discovery_accepts_hyphenated_and_compressed_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            paths = [
                workspace_root / "alpha-preprocessed-epo.fif",
                workspace_root / "beta_PREPROCESSED_epo.fif.gz",
            ]
            for path in paths:
                path.touch()

            datasets = discover_analysis_datasets(
                workspace_root, get_pipeline("standard"), "derivatives"
            )

            self.assertEqual([dataset.source_path for dataset in datasets], paths)

    def test_raw_backed_preprocessed_file_is_not_listed_twice(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            raw_path = workspace_root / "sub-01" / "eeg" / "sub-01_task-tms_raw.fif"
            raw_path.parent.mkdir(parents=True)
            raw_path.touch()
            pipeline = get_pipeline("standard")
            derivative_root = pipeline.derivative_root(workspace_root, "derivatives")
            preprocessed_path = (
                derivative_root
                / "sub-01"
                / "eeg"
                / "sub-01_task-tms_desc-preprocessed_epo.fif"
            )
            preprocessed_path.parent.mkdir(parents=True)
            preprocessed_path.touch()

            datasets = discover_analysis_datasets(
                workspace_root, pipeline, "derivatives"
            )

            self.assertEqual(len(datasets), 1)
            self.assertFalse(datasets[0].is_analysis_only)
            self.assertEqual(datasets[0].paths["preprocessed"], preprocessed_path)

    def test_normal_discovery_does_not_include_analysis_only_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            preprocessed_path = workspace_root / "orphan_desc-preprocessed_epo.fif"
            preprocessed_path.touch()

            datasets = discover_datasets(
                workspace_root, get_pipeline("standard"), "derivatives"
            )

            self.assertEqual(datasets, [])

    def test_normal_discovery_marks_matching_external_preprocessed_file_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            raw_path = (
                workspace_root
                / "sub-01"
                / "eeg"
                / "sub-01_ses-baseline_task-ACC_raw.fif"
            )
            raw_path.parent.mkdir(parents=True)
            raw_path.touch()
            preprocessed_path = (
                workspace_root
                / "imports"
                / "sub-01_ses-baseline_task-ACC_desc-preprocessed_epo.fif"
            )
            preprocessed_path.parent.mkdir(parents=True)
            preprocessed_path.touch()

            datasets = discover_datasets(
                workspace_root, get_pipeline("standard"), "derivatives"
            )

            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0].paths["preprocessed"], preprocessed_path)
            self.assertTrue(datasets[0].stage_exists("preprocessed"))
            self.assertEqual(datasets[0].status_text(), "Ready")
            self.assertEqual(datasets[0].progress_fraction(), 1.0)
            self.assertEqual(datasets[0].remaining_stage_count(), 0)

            analysis_datasets = discover_analysis_datasets(
                workspace_root, get_pipeline("standard"), "derivatives"
            )
            self.assertEqual(len(analysis_datasets), 1)
            self.assertEqual(
                analysis_datasets[0].paths["preprocessed"], preprocessed_path
            )

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

    def test_analysis_paths_include_source_estimate_derivatives(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            derivative_root = Path(tmp_dir) / "derivatives" / "standard"
            preprocessed_path = (
                derivative_root
                / "sub-01"
                / "eeg"
                / "sub-01_task-tms_desc-preprocessed_epo.fif"
            )

            paths = build_analysis_paths(preprocessed_path, derivative_root)

            self.assertEqual(
                paths["stc"],
                derivative_root
                / "sub-01"
                / "eeg"
                / "sub-01_task-tms_desc-stc_stc.h5",
            )
            self.assertEqual(
                paths["stc_metadata"],
                derivative_root
                / "sub-01"
                / "eeg"
                / "sub-01_task-tms_desc-stc_metadata.json",
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
