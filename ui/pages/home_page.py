import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
    QWidget, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QColor
from utils import get_path
from .base_page import BasePage

class FeaturesSection(QWidget):
    """The section detailing the key features with improved styling."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        # Correctly display the icon using a QPixmap in a QLabel
        icon_path = get_path('assets/logo.png')
        if os.path.exists(icon_path):
            icon_pixmap = QIcon(icon_path).pixmap(QSize(256, 256))
            icon_label = QLabel()
            icon_label.setPixmap(icon_pixmap)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            layout.addWidget(icon_label)

        # Use unique variable names to avoid overwriting
        subtitle1 = QLabel("<h3>A Toolkit Designed for EEG Data Analysis</h3>")
        subtitle1.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        subtitle2 = QLabel("<h4>Developed by Couto, B.A.N.</h4>")
        subtitle2.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        subtitle3 = QLabel("<h2>Please, fill the <a href=\"https://redcap.aau.dk/surveys/?s=M7HXDRMC7P4NJJK8\">Survey</a> to help us improve the software!</h2>")
        subtitle3.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        subtitle3.setOpenExternalLinks(True)        

        # Add widgets to the layout
        layout.addSpacing(10)
        layout.addWidget(subtitle3)
        layout.addSpacing(10)

        # Warning frame
        warning_frame = QFrame()
        warning_frame.setObjectName("warningFrame")
        warning_frame.setFrameShape(QFrame.Shape.StyledPanel)
        warning_layout = QVBoxLayout(warning_frame)
        warning_layout.setContentsMargins(20, 15, 20, 15) # Add some padding

        message = """
            <b>⚠️ Software Status: Pre-Release</b>
            <p>Thank you for trying out SSPy!</p>
            <p>This is a pre-release version intended for <b>evaluation and feedback purposes only</b>. We are working hard to build a reliable tool, but at this stage, please be aware of the following:</p>
            <ul>
                <li><b>Results may be inaccurate:</b> The algorithms and data processing pipelines are still being validated. Do not rely on the output for research publications or clinical assessments.</li>
                <li><b>Features may change:</b> Functionality might be added, removed, or modified in future updates.</li>
                <li><b>Bugs are expected:</b> You may encounter errors or unexpected behavior.</li>
            </ul>
            <p>👍 <b>We value your input!</b> If you find a bug or have a suggestion, please report it at <a href=\"https://github.com/Boutoo/SSPython\">SSPython Github Page</a>.</p>
        """
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        message_label.setTextFormat(Qt.TextFormat.RichText) # Ensure HTML tags are rendered
        message_label.setOpenExternalLinks(True) # Make links clickable

        warning_layout.addWidget(message_label)
        layout.addWidget(warning_frame) # Add the styled frame to the main layout

        # Using a QHBoxLayout for features
        features_layout = QHBoxLayout()
        features_layout.setSpacing(30)
        features_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        features = [
            ("TMS-EEG Preprocessing", "Specialized pipelines to handle the unique artifacts of TMS-EEG data."),
            ("Continuous EEG Analysis", "Robust tools for filtering, epoching, and analyzing continuous EEG data."),
            ("Preprocessing Made Easy", "An intuitive, parameter-driven workflow reduces complexity."),
            ("Interactive Visualization", "Generate publication-quality plots and inspect your data at every step.")
        ]

        for title_text, desc_text in features:
            card = self.create_feature_card(title_text, desc_text)
            features_layout.addWidget(card)

        # layout.addSpacing(40) # Add some space before the feature cards
        layout.addLayout(features_layout)
        
        layout.addWidget(subtitle1)
        layout.addWidget(subtitle2)
    
    def create_feature_card(self, title_text, desc_text):
        """Creates a modern-looking feature card with a drop shadow."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setFrameShadow(QFrame.Shadow.Plain)  # Use Plain shadow as base for custom styling
        card.setMinimumSize(QSize(220, 180))  # Prevent clipping

        # Add a stylesheet for rounded corners and a theme-aware background
        card.setStyleSheet("""
            QFrame {
                background-color: palette(window);
                border-radius: 10px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)  # Add padding
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Use a larger font for the title
        title = QLabel(f"<h3>{title_text}</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)

        desc = QLabel(desc_text)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)  # Ensure description text wraps properly

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch()

        # Add a modern drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 50))
        card.setGraphicsEffect(shadow)

        return card

class HomePage(BasePage):
    """A simple, welcoming homepage for the application."""
    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setObjectName("HomePage")

        features_section = FeaturesSection()

        # Center the features section horizontally
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addStretch()
        main_layout.addWidget(features_section)
        main_layout.addStretch()

        self.content_layout.addLayout(main_layout)
        self.content_layout.addStretch()  # Pushes content up from the bottom

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    assets_dir = get_path('assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    window = QMainWindow()
    window.setWindowTitle("SSPython Toolkit")
    window.resize(1280, 800)
    
    home_page = HomePage()
    window.setCentralWidget(home_page)
    
    window.show()
    sys.exit(app.exec())
