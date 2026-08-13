from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import mne


logger = logging.getLogger(__name__)


def evoked_filename(preprocessed_path: Path) -> str:
    """Return an MNE-compatible evoked filename for a preprocessed epochs file."""
    filename = preprocessed_path.name
    if filename.endswith("_epo.fif"):
        return f"{filename[:-len('_epo.fif')]}_ave.fif"
    if filename.endswith("-epo.fif"):
        return f"{filename[:-len('-epo.fif')]}-ave.fif"
    return f"{preprocessed_path.stem}_ave.fif"


def preprocessed_export_targets(
    source_paths: Iterable[Path], destination: Path
) -> list[tuple[Path, Path]]:
    return _build_export_targets(source_paths, destination, lambda path: path.name)


def evoked_export_targets(
    source_paths: Iterable[Path], destination: Path
) -> list[tuple[Path, Path]]:
    return _build_export_targets(source_paths, destination, evoked_filename)


def export_preprocessed_files(
    source_paths: Iterable[Path],
    destination: Path,
    *,
    overwrite: bool = False,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Path]:
    """Copy preprocessed epoch files into a single destination folder."""
    targets = preprocessed_export_targets(source_paths, destination)
    _prepare_destination(destination, targets, overwrite=overwrite)

    exported_paths: list[Path] = []
    for source, target in targets:
        _check_cancelled(is_cancelled)
        logger.info("Exporting %s", source.name)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        exported_paths.append(target)
    return exported_paths


def export_evoked_files(
    source_paths: Iterable[Path],
    destination: Path,
    *,
    overwrite: bool = False,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Path]:
    """Average preprocessed epoch files and save their Evoked objects."""
    targets = evoked_export_targets(source_paths, destination)
    _prepare_destination(destination, targets, overwrite=overwrite)

    exported_paths: list[Path] = []
    for source, target in targets:
        _check_cancelled(is_cancelled)
        logger.info("Averaging %s", source.name)
        epochs = mne.read_epochs(source, preload=False, verbose="error")
        evoked = epochs.average()
        _check_cancelled(is_cancelled)
        evoked.save(target, overwrite=overwrite, verbose="error")
        exported_paths.append(target)
    return exported_paths


def _build_export_targets(
    source_paths: Iterable[Path],
    destination: Path,
    filename_for_source: Callable[[Path], str],
) -> list[tuple[Path, Path]]:
    destination = Path(destination)
    targets: list[tuple[Path, Path]] = []
    target_sources: dict[str, Path] = {}

    for source_path in source_paths:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Preprocessed epochs file not found: {source}")

        target = destination / filename_for_source(source)
        target_key = target.name.casefold()
        previous_source = target_sources.get(target_key)
        if previous_source is not None:
            raise ValueError(
                "The selected datasets produce the same export filename: "
                f"{previous_source} and {source}"
            )
        target_sources[target_key] = source
        targets.append((source, target))

    return targets


def _prepare_destination(
    destination: Path,
    targets: list[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    destination = Path(destination)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Export destination is not a folder: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    if overwrite:
        return
    existing_targets = [target for _, target in targets if target.exists()]
    if existing_targets:
        names = ", ".join(target.name for target in existing_targets)
        raise FileExistsError(f"Export files already exist: {names}")


def _check_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None:
        is_cancelled()
