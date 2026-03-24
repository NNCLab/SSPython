from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .app_settings import DEFAULT_PIPELINE_ID


@dataclass(frozen=True)
class PipelineStage:
    id: str
    label: str
    short_label: str
    description: str
    optional: bool = False


@dataclass(frozen=True)
class PipelineDefinition:
    id: str
    name: str
    summary: str
    accent_color: str
    stages: tuple[PipelineStage, ...]
    analysis_ready_stage: str = "preprocessed"

    def derivative_root(self, workspace_root: Path, output_root: str) -> Path:
        return workspace_root / output_root / self.id

    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.id for stage in self.stages)

    def stage_index(self, stage_id: str) -> int:
        return self.stage_ids().index(stage_id)


PROCESSING_STAGES = (
    PipelineStage("raw", "Raw Import", "Raw", "Source recording available."),
    PipelineStage(
        "filtered_raw",
        "Continuous Cleanup",
        "Filter",
        "Artifact interpolation, filtering, or resampling saved.",
        optional=True,
    ),
    PipelineStage(
        "continuous_ica",
        "Continuous ICA",
        "ICA-C",
        "ICA fitted on the continuous recording.",
        optional=True,
    ),
    PipelineStage(
        "epochs",
        "Epoch Extraction",
        "Epochs",
        "Epoched data available for review.",
    ),
    PipelineStage(
        "epochs_ica",
        "Epoch ICA",
        "ICA-E",
        "ICA fitted on epoched data.",
        optional=True,
    ),
    PipelineStage(
        "preprocessed",
        "Preprocessed Output",
        "Ready",
        "Final preprocessed epochs are available.",
    ),
)


PIPELINES: dict[str, PipelineDefinition] = {
    "tms_eeg": PipelineDefinition(
        id="tms_eeg",
        name="TMS-EEG",
        summary="Artifact-aware preprocessing and ERP/TEP analysis.",
        accent_color="#0f7a82",
        stages=PROCESSING_STAGES,
    ),
    "continuous_eeg": PipelineDefinition(
        id="continuous_eeg",
        name="Continuous EEG",
        summary="Continuous inspection, filtering, PSD, and fixed-length epochs.",
        accent_color="#8a5a2b",
        stages=PROCESSING_STAGES,
    ),
}


def get_pipeline(pipeline_id: str | None) -> PipelineDefinition:
    if pipeline_id and pipeline_id in PIPELINES:
        return PIPELINES[pipeline_id]
    return PIPELINES[DEFAULT_PIPELINE_ID]


def all_pipelines() -> list[PipelineDefinition]:
    return list(PIPELINES.values())


def strip_extensions(file_path: Path) -> str:
    suffixes = "".join(file_path.suffixes)
    if suffixes:
        return file_path.name[: -len(suffixes)]
    return file_path.stem


def source_stem_from_raw(raw_path: Path) -> str:
    stem = strip_extensions(raw_path)
    return stem[:-4] if stem.endswith("_raw") else stem


def source_stem_from_derivative(derivative_path: Path) -> str:
    stem = strip_extensions(derivative_path)
    return re.sub(r"_desc-[^_]+_[^_]+$", "", stem)


def infer_relative_derivative_dir(file_path: Path) -> Path:
    source_stem = source_stem_from_raw(file_path)
    tokens = source_stem.split("_")
    parts = []

    subject = next((token for token in tokens if token.startswith("sub-")), None)
    session = next((token for token in tokens if token.startswith("ses-")), None)

    if subject:
        parts.append(subject)
    if session:
        parts.append(session)

    modality = file_path.parent.name if file_path.parent.name in {"eeg", "ieeg", "meg"} else "eeg"
    parts.append(modality)
    return Path(*parts)


def derivative_filename(source_stem: str, desc: str, suffix: str, extension: str) -> str:
    return f"{source_stem}_desc-{desc}_{suffix}{extension}"


def build_processing_paths(
    raw_path: Path,
    derivative_root: Path,
    *,
    create_dirs: bool = True,
) -> dict[str, Path]:
    source_stem = source_stem_from_raw(raw_path)
    relative_dir = infer_relative_derivative_dir(raw_path)
    base_dir = derivative_root / relative_dir
    paths = {
        "raw": raw_path,
        "filtered_raw": base_dir / derivative_filename(source_stem, "filtered", "raw", ".fif"),
        "continuous_ica": base_dir / derivative_filename(source_stem, "continuousica", "ica", ".fif"),
        "epochs": base_dir / derivative_filename(source_stem, "epoched", "epo", ".fif"),
        "epochs_ica": base_dir / derivative_filename(source_stem, "epochsica", "ica", ".fif"),
        "preprocessed": base_dir / derivative_filename(source_stem, "preprocessed", "epo", ".fif"),
    }
    if create_dirs:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def build_analysis_paths(
    preprocessed_path: Path,
    derivative_root: Path,
    *,
    create_dirs: bool = True,
) -> dict[str, Path]:
    try:
        relative_dir = preprocessed_path.parent.relative_to(derivative_root)
    except ValueError:
        relative_dir = infer_relative_derivative_dir(preprocessed_path)

    source_stem = source_stem_from_derivative(preprocessed_path)
    base_dir = derivative_root / relative_dir
    paths = {
        "tfr": base_dir / derivative_filename(source_stem, "tfr", "tfr", ".h5"),
        "tfr_mask": base_dir / derivative_filename(source_stem, "tfrmask", "mask", ".npy"),
        "itc": base_dir / derivative_filename(source_stem, "itc", "itc", ".h5"),
        "itc_mask": base_dir / derivative_filename(source_stem, "itcmask", "mask", ".npy"),
        "masked_ave": base_dir / derivative_filename(source_stem, "maskedave", "ave", ".fif"),
    }
    if create_dirs:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    return paths


@dataclass(frozen=True)
class DatasetRecord:
    raw_path: Path
    pipeline: PipelineDefinition
    derivative_root: Path
    paths: dict[str, Path]

    @property
    def source_stem(self) -> str:
        return source_stem_from_raw(self.raw_path)

    @property
    def display_name(self) -> str:
        return self.source_stem

    @property
    def relative_dir(self) -> Path:
        return infer_relative_derivative_dir(self.raw_path)

    def stage_exists(self, stage_id: str) -> bool:
        path = self.paths.get(stage_id)
        return bool(path and path.exists())

    def stage_definition(self, stage_id: str) -> PipelineStage:
        return next(stage for stage in self.pipeline.stages if stage.id == stage_id)

    def stage_status(self, stage_id: str) -> str:
        if self.stage_exists(stage_id):
            return "complete"

        stage = self.stage_definition(stage_id)
        stage_index = self.pipeline.stage_index(stage_id)
        later_stages = self.pipeline.stages[stage_index + 1 :]
        later_progress_exists = any(self.stage_exists(later_stage.id) for later_stage in later_stages)

        if stage.optional and later_progress_exists:
            return "skipped"
        return "pending"

    def display_file_name(self, stage_id: str) -> str:
        path = self.paths.get(stage_id)
        if path is None:
            return ""
        suffixes = "".join(path.suffixes)
        if path.is_file() and suffixes:
            return path.name[: -len(suffixes)]
        return path.name

    def completed_stage_count(self) -> int:
        return sum(
            1
            for stage in self.pipeline.stages
            if self.stage_status(stage.id) in {"complete", "skipped"}
        )

    def remaining_stage_count(self) -> int:
        return max(len(self.pipeline.stages) - self.completed_stage_count(), 0)

    def progress_fraction(self) -> float:
        if not self.pipeline.stages:
            return 0.0
        return self.completed_stage_count() / len(self.pipeline.stages)

    def status_text(self) -> str:
        if self.stage_exists(self.pipeline.analysis_ready_stage):
            return "Ready"
        remaining = self.remaining_stage_count()
        return f"{remaining} step{'s' if remaining != 1 else ''} left"

    def latest_completed_stage_id(self) -> str:
        latest_stage = "raw"
        for stage in self.pipeline.stages:
            if self.stage_status(stage.id) in {"complete", "skipped"}:
                latest_stage = stage.id
        return latest_stage


def discover_datasets(
    workspace_root: Path | None,
    pipeline: PipelineDefinition,
    output_root: str,
) -> list[DatasetRecord]:
    if workspace_root is None or not workspace_root.exists():
        return []

    workspace_root = Path(workspace_root)
    output_root_path = workspace_root / output_root
    derivative_root = pipeline.derivative_root(workspace_root, output_root)

    datasets: list[DatasetRecord] = []
    for raw_path in sorted(workspace_root.rglob("*_raw.fif")):
        try:
            raw_path.relative_to(output_root_path)
            continue
        except ValueError:
            pass

        paths = build_processing_paths(raw_path, derivative_root, create_dirs=False)
        datasets.append(
            DatasetRecord(
                raw_path=raw_path,
                pipeline=pipeline,
                derivative_root=derivative_root,
                paths=paths,
            )
        )

    return datasets
