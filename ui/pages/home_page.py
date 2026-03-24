from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.pipelines import PipelineDefinition, all_pipelines

from .base_page import BasePage


class PipelineCard(QFrame):
    def __init__(self, pipeline: PipelineDefinition, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.setObjectName("pipelineCard")

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
        layout.addStretch()

    def set_active(self, active: bool):
        self.setProperty("active", active)
        self.badge.setText("Active pipeline" if active else "Pipeline")
        self.button.setText("Current Pipeline" if active else "Set Active Pipeline")
        self.button.setEnabled(not active)
        self.style().unpolish(self)
        self.style().polish(self)


class HomePage(BasePage):
    def __init__(self, parent=None):
        super().__init__("SSPython", parent)
        self.main_window = parent
        self.cards: dict[str, PipelineCard] = {}

        self.set_page_subtitle(
            "Get started by selecting a pipeline below. Your choice will tailor the workspace and available analyses."
        )

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 24, 24, 24)
        hero_layout.setSpacing(12)

        hero_title = QLabel("Welcome to SSPython!")
        hero_title.setObjectName("heroTitle")
        hero_layout.addWidget(hero_title)

        hero_text = QLabel(
            "Your open-source EEG analysis toolkit. Streamline your workflow from raw data to insightful results with our powerful and intuitive pipelines."
        )
        hero_text.setWordWrap(True)
        hero_text.setObjectName("mutedLabel")
        hero_layout.addWidget(hero_text)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(10)

        open_workspace_button = QPushButton("Open Workspace")
        open_workspace_button.clicked.connect(lambda: self.main_window.workspace_panel.select_folder())
        actions.addWidget(open_workspace_button)

        preferences_button = QPushButton("Preferences")
        preferences_button.setObjectName("secondaryButton")
        preferences_button.clicked.connect(lambda: self.main_window.navigate_to_page("Preferences"))
        actions.addWidget(preferences_button)
        actions.addStretch()

        hero_layout.addLayout(actions)
        self.add_content(hero)

        section_title = QLabel("<h2>Pipelines</h2>")
        self.add_content(section_title)

        cards_container = QWidget()
        cards_layout = QGridLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setHorizontalSpacing(16)
        cards_layout.setVerticalSpacing(16)

        for index, pipeline in enumerate(all_pipelines()):
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
            self.on_pipeline_changed(self.main_window.current_pipeline)

    @Slot(object)
    def on_pipeline_changed(self, pipeline):
        for pipeline_id, card in self.cards.items():
            card.set_active(pipeline_id == pipeline.id)
