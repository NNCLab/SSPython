"""Inspect saved dataset files and manage their dependent derivatives."""

from __future__ import annotations

import gc
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox

from core.pipelines import DatasetRecord
from ui.widgets.layout_helpers import dataset_label
from ui.widgets.tools.object_info_widget import ObjectDetailsDialog
from ui.widgets.tools.stage_io import load_stage_object
from utils import Worker


class DatasetFileController(QObject):
    derivatives_about_to_be_deleted = Signal(object)
    derivatives_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_dataset: DatasetRecord | None = None

    def set_dataset(self, dataset):
        self.current_dataset = dataset

    def inspect_stage(self, stage_id):
        dataset = self.current_dataset
        if dataset is None or dataset.stage_status(stage_id) != "complete":
            return

        stage = dataset.stage_definition(stage_id)
        stage_path = dataset.paths[stage_id]
        worker = Worker(
            lambda: load_stage_object(stage_id, stage_path, stage.data_kind),
            parent=self.parent(),
            add_loggers="mne",
        )
        try:
            mne_object = worker.exec_with_dialog(
                "Loading file information", f"Loading {stage.label.lower()}…",
            )
        except Exception as exc:
            QMessageBox.critical(self.parent(), "Unable to load file information", str(exc))
            worker.deleteLater()
            return
        if mne_object is None:
            worker.deleteLater()
            return

        delete_requested = False
        dialog = ObjectDetailsDialog(
            dataset_label(dataset, path=stage_path),
            mne_object,
            file_path=str(stage_path),
            status_text="Saved",
            description=f"{stage.label} · {stage.description}",
            allow_delete=bool(self._existing_derivative_paths_from_stage(stage_id)),
            parent=self.parent(),
        )

        def request_delete():
            nonlocal delete_requested
            delete_requested = True
            dialog.accept()

        dialog.open_location_requested.connect(lambda: self._open_file_location(stage_path))
        dialog.delete_requested.connect(request_delete)
        try:
            dialog.exec()
        finally:
            # Release the inspection reader before asking pages to release their
            # own caches, so Windows can delete the files after confirmation.
            dialog.release_object()
            dialog.deleteLater()
            worker._result = None
            worker.deleteLater()
            close = getattr(mne_object, "close", None)
            if callable(close):
                close()
            del mne_object
            gc.collect()

        if delete_requested:
            self._confirm_and_delete_derivatives(stage_id)

    def _open_file_location(self, file_path: Path):
        location = file_path.parent if file_path.is_file() else file_path
        if not location.exists():
            QMessageBox.warning(
                self.parent(),
                "Location not found",
                f"The file location does not exist:\n{location}",
            )
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(location)))
        if not opened:
            QMessageBox.warning(
                self.parent(),
                "Unable to open location",
                f"Could not open the file location:\n{location}",
            )

    def _existing_derivative_paths_from_stage(self, stage_id: str) -> list[Path]:
        if self.current_dataset is None or stage_id == "raw":
            return []
        if self.current_dataset.is_analysis_only:
            return []

        try:
            start_index = self.current_dataset.pipeline.stage_index(stage_id)
        except ValueError:
            return []

        paths: list[Path] = []
        for stage in self.current_dataset.pipeline.stages[start_index:]:
            if stage.id == "raw":
                continue
            path = self.current_dataset.paths.get(stage.id)
            if path is not None and path.exists() and path.is_file():
                paths.append(path)
            if stage.id == "filtered_raw":
                checkpoint = self.current_dataset.paths.get("filter_input_raw")
                if checkpoint is not None and checkpoint.is_file():
                    paths.append(checkpoint)

        try:
            analysis_ready_index = self.current_dataset.pipeline.stage_index(
                self.current_dataset.pipeline.analysis_ready_stage
            )
        except ValueError:
            analysis_ready_index = -1
        if start_index <= analysis_ready_index:
            paths.extend(
                path
                for path in self.current_dataset.analysis_paths.values()
                if path.exists() and path.is_file()
            )
        return paths

    def _confirm_and_delete_derivatives(self, stage_id: str):
        delete_paths = self._existing_derivative_paths_from_stage(stage_id)
        if not delete_paths:
            return

        stage = self.current_dataset.stage_definition(stage_id) if self.current_dataset else None
        stage_label = stage.label if stage else "selected stage"
        file_word = "file" if len(delete_paths) == 1 else "files"
        preview = "\n".join(path.name for path in delete_paths[:5])
        if len(delete_paths) > 5:
            preview += f"\n...and {len(delete_paths) - 5} more"

        message_box = QMessageBox(self.parent())
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setWindowTitle("Delete derivatives?")
        message_box.setText(
            f"Delete {len(delete_paths)} derivative {file_word} from {stage_label} onward?"
        )
        message_box.setInformativeText(
            f"{preview}\n\nThis cannot be undone."
        )
        message_box.setDetailedText("\n".join(str(path) for path in delete_paths))
        delete_button = message_box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        message_box.setDefaultButton(cancel_button)
        message_box.exec()

        if message_box.clickedButton() != delete_button:
            return

        errors = self._delete_derivative_paths(delete_paths)
        self.derivatives_changed.emit()

        if errors:
            error_text = "\n".join(f"{path}: {error}" for path, error in errors)
            QMessageBox.warning(
                self.parent(),
                "Unable to delete some derivatives",
                error_text,
            )

    def _delete_derivative_paths(self, delete_paths: list[Path]) -> list[tuple[Path, str]]:
        # This direct signal gives pages a chance to release cached MNE readers
        # before Windows enforces its mandatory file locks.
        self.derivatives_about_to_be_deleted.emit(tuple(delete_paths))
        gc.collect()

        errors: list[tuple[Path, str]] = []
        for path in delete_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                errors.append((path, str(exc)))
        return errors
