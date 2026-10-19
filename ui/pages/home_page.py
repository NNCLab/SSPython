from PySide6.QtCore import Qt, Slot
from pathlib import Path
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.pipelines import PipelineDefinition, all_pipelines
from ui.widgets.layout_helpers import FilenameLabel, FlowLayout, dataset_label

from .base_page import BasePage


class PipelineCard(QFrame):
    def __init__(self, pipeline: PipelineDefinition, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.setObjectName("pipelineCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.badge = QLabel("Pipeline")
        self.badge.setObjectName("pipelineBadge")
        layout.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignLeft)

        self.title = QLabel(pipeline.name)
        self.title.setObjectName("pipelineCardTitle")
        layout.addWidget(self.title)

        self.summary = QLabel(pipeline.summary)
        self.summary.setObjectName("mutedLabel")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.button = QPushButton("Set Active Pipeline")
        layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignLeft)

    def set_active(self, active: bool):
        self.setProperty("active", active)
        self.badge.setText("Active pipeline" if active else "Pipeline")
        self.button.setText("Current Pipeline" if active else "Set Active Pipeline")
        self.button.setEnabled(not active)
        self.style().unpolish(self)
        self.style().polish(self)


class HomePage(BasePage):
    def __init__(self, parent=None):
        super().__init__("Home", parent)
        self.main_window = parent
        self.cards: dict[str, PipelineCard] = {}
        self.set_content_maximum_width(1120)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 24, 24, 24)
        hero_layout.setSpacing(12)

        self.hero_title = QLabel("Start an EEG workflow")
        self.hero_title.setObjectName("heroTitle")
        hero_layout.addWidget(self.hero_title)

        self.hero_text = FilenameLabel()
        self.hero_text.setWordWrap(True)
        self.hero_text.setObjectName("mutedLabel")
        hero_layout.addWidget(self.hero_text)

        actions = FlowLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(10)

        self.continue_button = QPushButton("Open workspace")
        self.continue_button.clicked.connect(self._continue_workflow)
        actions.addWidget(self.continue_button)
        self.open_workspace_button = QPushButton("Open another workspace")
        self.open_workspace_button.setObjectName("secondaryButton")
        self.open_workspace_button.clicked.connect(lambda: self.main_window.workspace_panel.select_folder())
        actions.addWidget(self.open_workspace_button)

        preferences_button = QPushButton("Preferences")
        preferences_button.setObjectName("secondaryButton")
        preferences_button.clicked.connect(lambda: self.main_window.navigate_to_page("Preferences"))
        actions.addWidget(preferences_button)

        hero_layout.addLayout(actions)
        self.add_content(hero)

        self.pipeline_label = QLabel()
        self.pipeline_label.setObjectName("mutedLabel")
        self.add_content(self.pipeline_label)

        section_title = QLabel("Recent workspaces")
        section_title.setObjectName("sectionTitle")
        self.add_content(section_title)
        self.recent_container = QWidget()
        self.recent_layout = QVBoxLayout(self.recent_container)
        self.recent_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_layout.setSpacing(8)
        self.add_content(self.recent_container)

        cards_container = QWidget()
        cards_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        cards_layout = QGridLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setHorizontalSpacing(16)
        cards_layout.setVerticalSpacing(16)

        pipelines = all_pipelines()
        for index, pipeline in enumerate(pipelines if len(pipelines) > 1 else []):
            card = PipelineCard(pipeline)
            card.button.clicked.connect(
                lambda checked=False, pipeline_id=pipeline.id: self.main_window.set_current_pipeline(pipeline_id)
            )
            self.cards[pipeline.id] = card
            cards_layout.addWidget(card, index // 2, index % 2)

        self.add_content(cards_container)

        note = QLabel(
            "SSPython is currently in a pre-release stage. "
            "Please validate all outputs before using them for publication or clinical interpretation. "
            "We welcome your feedback to help us improve!"
        )
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        self.add_content(note)

        if self.main_window:
            self.main_window.current_pipeline_changed.connect(self.on_pipeline_changed)
            self.main_window.current_folder_changed.connect(lambda _: self.update_ui_state())
            self.main_window.current_dataset_changed.connect(lambda _: self.update_ui_state())
            self.on_pipeline_changed(self.main_window.current_pipeline)
        self.update_ui_state()

    def _continue_workflow(self):
        if not self.main_window:
            return
        dataset = self.main_window.current_dataset
        if dataset:
            self.main_window.navigate_to_page("Analysis" if dataset.stage_exists(dataset.pipeline.analysis_ready_stage) else "Preprocessing")
        elif self.main_window.current_folder:
            self.main_window.navigate_to_page("Preprocessing")
            self.main_window.focus_workspace()
        else:
            self.main_window.workspace_panel.select_folder()

    def _open_recent(self, path):
        if path.is_dir():
            self.main_window.set_current_folder(str(path))
            self._continue_workflow()
        else:
            self.hero_text.setText(f"Workspace is no longer available: {path}")

    def update_ui_state(self):
        if not self.main_window:
            return
        folder = self.main_window.current_folder
        dataset = self.main_window.current_dataset
        self.hero_title.setText("Continue your EEG workflow" if folder else "Start an EEG workflow")
        self.hero_text.setText(dataset_label(dataset) if dataset else "Open a workspace containing EEG recordings, or convert source files from the File menu.")
        ready = bool(dataset and dataset.stage_exists(dataset.pipeline.analysis_ready_stage))
        self.continue_button.setText("Open Analysis" if ready else "Continue preprocessing" if dataset else "Choose dataset" if folder else "Open workspace")
        self.open_workspace_button.setVisible(bool(folder))
        self.set_page_subtitle(folder.name if folder else "Your recordings, processing stages, and analysis results in one workspace.")
        self.subtitle_label.setToolTip(str(folder) if folder else "")
        while self.recent_layout.count():
            item = self.recent_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        recent = self.main_window.settings_store.get("workspace/recent_folders", [])
        recent = recent if isinstance(recent, list) else []
        for entry in recent:
            path = Path(entry)
            button = QPushButton(f"{path.name}   ·   {path.parent}")
            button.setObjectName("recentWorkspaceButton")
            button.setToolTip(str(path))
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            button.clicked.connect(lambda checked=False, selected=path: self._open_recent(selected))
            self.recent_layout.addWidget(button)
        if not recent:
            label = QLabel("Opened workspaces will appear here.")
            label.setObjectName("mutedLabel")
            self.recent_layout.addWidget(label)

    @Slot(object)
    def on_pipeline_changed(self, pipeline):
        self.pipeline_label.setText(f"Active pipeline: {pipeline.name}")
        for pipeline_id, card in self.cards.items():
            card.set_active(pipeline_id == pipeline.id)
