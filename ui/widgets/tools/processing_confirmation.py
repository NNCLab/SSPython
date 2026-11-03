"""Ask before a processing action replaces or deletes saved results."""

from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from core.processing import Preprocessor


def _unique_paths(paths) -> list[Path]:
    unique = {}
    for path in paths:
        path = Path(path)
        unique.setdefault(path.resolve(), path)
    return list(unique.values())


def confirm_file_changes(
    parent,
    *,
    title: str,
    message: str,
    deleted_paths=(),
    replaced_paths=(),
    always_ask: bool = False,
) -> bool:
    """Confirm a caller's list of existing files before any destructive work."""
    deleted = _unique_paths(deleted_paths)
    deleted_keys = {path.resolve() for path in deleted}
    replaced = [
        path for path in _unique_paths(replaced_paths)
        if path.resolve() not in deleted_keys
    ]
    if not deleted and not replaced and not always_ask:
        return True

    parts = [message]
    if replaced:
        parts.append(
            "These saved files will be replaced:\n"
            + "\n".join(f" - {path}" for path in replaced)
        )
    if deleted:
        parts.append(
            "These dependent files will be deleted:\n"
            + "\n".join(f" - {path}" for path in deleted)
        )
        parts.append("Deleting these files cannot be undone.")
    parts.append("Continue?")
    reply = QMessageBox.question(
        parent,
        title,
        "\n\n".join(parts),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes


def confirm_processing_update(
    parent,
    preprocessor: Preprocessor,
    stage_id: str,
    *,
    title: str,
    message: str,
    confirm_overwrite: bool = True,
    extra_deleted_paths=(),
    extra_replaced_paths=(),
    always_ask: bool = False,
) -> bool:
    """Check downstream files even when this processing stage was skipped."""
    deleted = [path for _, path in preprocessor.existing_downstream_files(stage_id)]
    deleted.extend(Path(path) for path in extra_deleted_paths if Path(path).is_file())
    has_output = confirm_overwrite and bool(preprocessor.has(stage_id))
    output = preprocessor.paths.get(stage_id) if has_output else None
    replaced = [output] if output is not None else []
    replaced.extend(Path(path) for path in extra_replaced_paths if Path(path).is_file())
    return confirm_file_changes(
        parent,
        title=title,
        message=message,
        deleted_paths=deleted,
        replaced_paths=replaced,
        always_ask=always_ask or has_output,
    )
