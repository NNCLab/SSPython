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
    data_kind: str | None = None
    derivative_desc: str | None = None
    derivative_suffix: str | None = None
    derivative_extension: str = ".fif"


@dataclass(frozen=True)
class PipelineAction:
    id: str
    label: str
    stage_id: str
    role: str
    handler: str
    requires_stage_id: str | None = None


@dataclass(frozen=True)
class PipelinePanel:
    id: str
    info_stage_id: str | None
    actions: tuple[PipelineAction, ...] = ()
    plot_stage_id: str | None = None


@dataclass(frozen=True)
class PipelineSection:
    id: str
    title: str
    panels: tuple[PipelinePanel, ...]


@dataclass(frozen=True)
class PipelineDefinition:
    id: str
    name: str
    summary: str
    accent_color: str
    stages: tuple[PipelineStage, ...]
    workflow_sections: tuple[PipelineSection, ...]
    analysis_ready_stage: str = "preprocessed"
    legacy_ids: tuple[str, ...] = ()

    def derivative_root(self, workspace_root: Path, output_root: str) -> Path:
        return workspace_root / output_root / self.id

    def derivative_roots(self, workspace_root: Path, output_root: str) -> tuple[Path, ...]:
        return (self.derivative_root(workspace_root, output_root),) + tuple(
            workspace_root / output_root / legacy_id for legacy_id in self.legacy_ids
        )

    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.id for stage in self.stages)

    def stage_index(self, stage_id: str) -> int:
        return self.stage_ids().index(stage_id)


PROCESSING_STAGES = (
    PipelineStage("raw", "Raw Import", "Raw", "Source recording available.", data_kind="raw"),
    PipelineStage(
        "filtered_raw",
        "Continuous Cleanup",
        "Filter",
        "Artifact interpolation, filtering, or resampling saved.",
        optional=True,
        data_kind="raw",
        derivative_desc="filtered",
        derivative_suffix="raw",
    ),
    PipelineStage(
        "continuous_ica",
        "Continuous ICA",
        "ICA-C",
        "ICA fitted on the continuous recording.",
        optional=True,
        data_kind="ica",
        derivative_desc="continuousica",
        derivative_suffix="ica",
    ),
    PipelineStage(
        "epochs",
        "Epoch Extraction",
        "Epochs",
        "Epoched data available for review.",
        data_kind="epochs",
        derivative_desc="epoched",
        derivative_suffix="epo",
    ),
    PipelineStage(
        "epochs_ica",
        "Epoch ICA",
        "ICA-E",
        "ICA fitted on epoched data.",
        optional=True,
        data_kind="ica",
        derivative_desc="epochsica",
        derivative_suffix="ica",
    ),
    PipelineStage(
        "preprocessed",
        "Preprocessed Output",
        "Ready",
        "Final preprocessed epochs are available.",
        data_kind="epochs",
        derivative_desc="preprocessed",
        derivative_suffix="epo",
    ),
)


STANDARD_WORKFLOW_SECTIONS = (
    PipelineSection(
        id="continuous_processing",
        title="Continuous Processing",
        panels=(
            PipelinePanel(
                id="recording",
                info_stage_id="raw",
                actions=(
                    PipelineAction(
                        id="inspect_raw",
                        label="Inspect Raw Data",
                        stage_id="raw",
                        role="inspect",
                        handler="inspect_raw_data",
                        requires_stage_id="raw",
                    ),
                    PipelineAction(
                        id="artifact_removal",
                        label="Remove Stimulation Artifact",
                        stage_id="filtered_raw",
                        role="process",
                        handler="artifact_removal",
                        requires_stage_id="raw",
                    ),
                    PipelineAction(
                        id="filter_continuous",
                        label="Filter and Resample",
                        stage_id="filtered_raw",
                        role="process",
                        handler="filter_continuous",
                        requires_stage_id="raw",
                    ),
                ),
            ),
            PipelinePanel(
                id="continuous_ica",
                info_stage_id="continuous_ica",
                actions=(
                    PipelineAction(
                        id="run_continuous_ica",
                        label="Run Continuous ICA",
                        stage_id="continuous_ica",
                        role="process",
                        handler="run_continuous_ica",
                        requires_stage_id="raw",
                    ),
                    PipelineAction(
                        id="inspect_continuous_ica",
                        label="Inspect Continuous ICA",
                        stage_id="continuous_ica",
                        role="inspect",
                        handler="inspect_continuous_ica",
                        requires_stage_id="continuous_ica",
                    ),
                ),
            ),
        ),
    ),
    PipelineSection(
        id="epoching_and_ica",
        title="Epoching and ICA",
        panels=(
            PipelinePanel(
                id="epochs",
                info_stage_id="epochs",
                actions=(
                    PipelineAction(
                        id="segment",
                        label="Segment into Epochs",
                        stage_id="epochs",
                        role="process",
                        handler="segment_epochs",
                        requires_stage_id="raw",
                    ),
                    PipelineAction(
                        id="plot_raw_evoked",
                        label="Plot Raw Evoked",
                        stage_id="epochs",
                        role="inspect",
                        handler="plot_raw_evoked",
                        requires_stage_id="epochs",
                    ),
                    PipelineAction(
                        id="inspect_epochs",
                        label="Inspect Epochs",
                        stage_id="epochs",
                        role="inspect",
                        handler="inspect_epochs",
                        requires_stage_id="epochs",
                    ),
                    PipelineAction(
                        id="rereference_epochs",
                        label="Re-reference Epochs",
                        stage_id="epochs",
                        role="process",
                        handler="rereference_epochs",
                        requires_stage_id="epochs",
                    ),
                ),
            ),
            PipelinePanel(
                id="epochs_ica",
                info_stage_id="epochs_ica",
                actions=(
                    PipelineAction(
                        id="run_epochs_ica",
                        label="Run Epoch ICA",
                        stage_id="epochs_ica",
                        role="process",
                        handler="run_epochs_ica",
                        requires_stage_id="epochs",
                    ),
                    PipelineAction(
                        id="inspect_epochs_ica",
                        label="Inspect Epoch ICA",
                        stage_id="epochs_ica",
                        role="inspect",
                        handler="inspect_epochs_ica",
                        requires_stage_id="epochs_ica",
                    ),
                ),
            ),
        ),
    ),
    PipelineSection(
        id="preprocessed_output",
        title="Preprocessed Output",
        panels=(
            PipelinePanel(
                id="preprocessed",
                info_stage_id="preprocessed",
                actions=(
                    PipelineAction(
                        id="apply_filters",
                        label="Apply ICA and Final Filters",
                        stage_id="preprocessed",
                        role="process",
                        handler="apply_filters",
                        requires_stage_id="epochs",
                    ),
                ),
                plot_stage_id="preprocessed",
            ),
        ),
    ),
)


PIPELINES: dict[str, PipelineDefinition] = {
    "standard": PipelineDefinition(
        id="standard",
        name="Standard",
        summary=(
            "Shared preprocessing workflow with event-based and fixed-length "
            "epoch segmentation options."
        ),
        accent_color="#0f7a82",
        stages=PROCESSING_STAGES,
        workflow_sections=STANDARD_WORKFLOW_SECTIONS,
        legacy_ids=("tms_eeg", "continuous_eeg"),
    ),
}

PIPELINE_ID_ALIASES = {
    "tms_eeg": DEFAULT_PIPELINE_ID,
    "continuous_eeg": DEFAULT_PIPELINE_ID,
}


def normalize_pipeline_id(pipeline_id: str | None) -> str:
    if not pipeline_id:
        return DEFAULT_PIPELINE_ID
    return PIPELINE_ID_ALIASES.get(pipeline_id, pipeline_id)


def get_pipeline(pipeline_id: str | None) -> PipelineDefinition:
    normalized_id = normalize_pipeline_id(pipeline_id)
    if normalized_id in PIPELINES:
        return PIPELINES[normalized_id]
    return PIPELINES[DEFAULT_PIPELINE_ID]


def all_pipelines() -> list[PipelineDefinition]:
    return list(PIPELINES.values())


def register_pipeline(pipeline: PipelineDefinition):
    if pipeline.id in PIPELINES or pipeline.id in PIPELINE_ID_ALIASES:
        raise ValueError(f"Pipeline id already registered: {pipeline.id}")
    PIPELINES[pipeline.id] = pipeline


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
    source_stem = re.sub(
        r"_desc-[^_]+_[^_]+$", "", stem, flags=re.IGNORECASE
    )
    return re.sub(
        r"[-_]preprocessed[-_]epo$", "", source_stem, flags=re.IGNORECASE
    )


def is_preprocessed_epochs_path(file_path: Path) -> bool:
    """Return whether a filename identifies a preprocessed Epochs FIF file."""
    name = file_path.name.casefold()
    return "preprocessed" in name and name.endswith(
        ("_epo.fif", "-epo.fif", "_epo.fif.gz", "-epo.fif.gz")
    )


def discover_preprocessed_epochs_paths(workspace_root: Path) -> list[Path]:
    return sorted(
        path
        for path in workspace_root.rglob("*")
        if path.is_file() and is_preprocessed_epochs_path(path)
    )


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
    pipeline: PipelineDefinition | None = None,
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
    if pipeline is not None:
        for stage in pipeline.stages:
            if stage.id == "raw":
                paths["raw"] = raw_path
            elif stage.derivative_desc and stage.derivative_suffix:
                paths[stage.id] = base_dir / derivative_filename(
                    source_stem,
                    stage.derivative_desc,
                    stage.derivative_suffix,
                    stage.derivative_extension,
                )
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
        "stc": base_dir / derivative_filename(source_stem, "stc", "stc", ".h5"),
        "stc_metadata": base_dir / derivative_filename(
            source_stem, "stc", "metadata", ".json"
        ),
    }
    if create_dirs:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    return paths


@dataclass(frozen=True)
class DatasetRecord:
    raw_path: Path | None
    pipeline: PipelineDefinition
    derivative_root: Path
    paths: dict[str, Path]

    @property
    def is_analysis_only(self) -> bool:
        return self.raw_path is None

    @property
    def source_path(self) -> Path:
        if self.raw_path is not None:
            return self.raw_path
        return self.paths[self.pipeline.analysis_ready_stage]

    @property
    def source_stem(self) -> str:
        if self.raw_path is not None:
            return source_stem_from_raw(self.raw_path)
        return source_stem_from_derivative(self.source_path)

    @property
    def display_name(self) -> str:
        return self.source_stem

    @property
    def relative_dir(self) -> Path:
        if self.raw_path is not None:
            return infer_relative_derivative_dir(self.raw_path)
        try:
            return self.source_path.parent.relative_to(self.derivative_root)
        except ValueError:
            return infer_relative_derivative_dir(self.source_path)

    @property
    def analysis_paths(self) -> dict[str, Path]:
        analysis_input = self.paths.get(self.pipeline.analysis_ready_stage)
        if analysis_input is None:
            return {}
        return build_analysis_paths(
            analysis_input,
            self.derivative_root,
            create_dirs=False,
        )

    def stage_exists(self, stage_id: str) -> bool:
        path = self.paths.get(stage_id)
        return bool(path and path.exists())

    def stage_definition(self, stage_id: str) -> PipelineStage:
        return next(stage for stage in self.pipeline.stages if stage.id == stage_id)

    def stage_status(self, stage_id: str) -> str:
        if self.stage_exists(stage_id):
            return "complete"

        if self.is_analysis_only:
            return "skipped"

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
        # A saved analysis-ready output is the terminal pipeline result. Earlier
        # intermediate files may legitimately have been removed or imported
        # from another workspace, but that must not make a ready dataset appear
        # partially complete in the workspace progress bar.
        if self.stage_exists(self.pipeline.analysis_ready_stage):
            return len(self.pipeline.stages)
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
    *,
    preprocessed_paths: list[Path] | None = None,
) -> list[DatasetRecord]:
    if workspace_root is None or not workspace_root.exists():
        return []

    workspace_root = Path(workspace_root)
    output_root_path = workspace_root / output_root
    if preprocessed_paths is None:
        preprocessed_paths = discover_preprocessed_epochs_paths(workspace_root)
    preprocessed_by_stem: dict[str, list[Path]] = {}
    for preprocessed_path in preprocessed_paths:
        source_stem = source_stem_from_derivative(preprocessed_path).casefold()
        preprocessed_by_stem.setdefault(source_stem, []).append(preprocessed_path)

    datasets: list[DatasetRecord] = []
    for raw_path in sorted(workspace_root.rglob("*_raw.fif")):
        try:
            raw_path.relative_to(output_root_path)
            continue
        except ValueError:
            pass

        derivative_root, paths = select_processing_paths(
            raw_path,
            pipeline,
            workspace_root,
            output_root,
        )
        analysis_ready_path = paths.get(pipeline.analysis_ready_stage)
        if analysis_ready_path is not None and not analysis_ready_path.exists():
            matching_paths = preprocessed_by_stem.get(
                source_stem_from_raw(raw_path).casefold(), []
            )
            if len(matching_paths) == 1:
                paths = dict(paths)
                paths[pipeline.analysis_ready_stage] = matching_paths[0]
        dataset = DatasetRecord(
            raw_path=raw_path,
            pipeline=pipeline,
            derivative_root=derivative_root,
            paths=paths,
        )
        datasets.append(dataset)

    return sorted(datasets, key=lambda dataset: str(dataset.source_path).casefold())


def discover_analysis_datasets(
    workspace_root: Path | None,
    pipeline: PipelineDefinition,
    output_root: str,
) -> list[DatasetRecord]:
    """Discover each analysis-ready Epochs file in a workspace exactly once."""
    if workspace_root is None or not workspace_root.exists():
        return []

    workspace_root = Path(workspace_root)
    preprocessed_paths = discover_preprocessed_epochs_paths(workspace_root)
    raw_datasets = discover_datasets(
        workspace_root,
        pipeline,
        output_root,
        preprocessed_paths=preprocessed_paths,
    )
    datasets_by_preprocessed_path = {
        path.resolve(): dataset
        for dataset in raw_datasets
        if (path := dataset.paths.get(pipeline.analysis_ready_stage)) is not None
        and path.exists()
    }

    for preprocessed_path in preprocessed_paths:
        resolved_path = preprocessed_path.resolve()
        if resolved_path in datasets_by_preprocessed_path:
            continue

        derivative_root = pipeline.derivative_root(workspace_root, output_root)
        for candidate_root in pipeline.derivative_roots(workspace_root, output_root):
            try:
                preprocessed_path.relative_to(candidate_root)
            except ValueError:
                continue
            derivative_root = candidate_root
            break

        datasets_by_preprocessed_path[resolved_path] = DatasetRecord(
            raw_path=None,
            pipeline=pipeline,
            derivative_root=derivative_root,
            paths={pipeline.analysis_ready_stage: preprocessed_path},
        )

    return sorted(
        datasets_by_preprocessed_path.values(),
        key=lambda dataset: str(
            dataset.paths[pipeline.analysis_ready_stage]
        ).casefold(),
    )


def derivative_progress_count(
    paths: dict[str, Path],
    pipeline: PipelineDefinition,
) -> int:
    return sum(
        1
        for stage in pipeline.stages
        if stage.id != "raw" and paths.get(stage.id) and paths[stage.id].exists()
    )


def select_processing_paths(
    raw_path: Path,
    pipeline: PipelineDefinition,
    workspace_root: Path,
    output_root: str,
) -> tuple[Path, dict[str, Path]]:
    candidates: list[tuple[int, Path, dict[str, Path]]] = []
    for derivative_root in pipeline.derivative_roots(workspace_root, output_root):
        paths = build_processing_paths(
            raw_path,
            derivative_root,
            pipeline=pipeline,
            create_dirs=False,
        )
        progress_count = derivative_progress_count(paths, pipeline)
        candidates.append((progress_count, derivative_root, paths))

    best_progress, best_root, best_paths = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    if best_progress == 0:
        return pipeline.derivative_root(workspace_root, output_root), build_processing_paths(
            raw_path,
            pipeline.derivative_root(workspace_root, output_root),
            pipeline=pipeline,
            create_dirs=False,
        )
    return best_root, best_paths
