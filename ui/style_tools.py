from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import re
import logging
from utils import get_path

logger = logging.getLogger(__file__)

QSS_TEMPLATE = """
/* ----- Main Window & General ----- */
QMainWindow, QDialog, QWidget {
    background-color: {background_1};
    color: {text_1};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12pt;
}
#warningFrame {
    border: 2px solid {danger_color};
    border-radius: 8px;
}

QWidget:disabled {
    color: {text_2};
}

/* ----- Labels ----- */
QLabel {
    background-color: transparent;
    color: {text_1};
}

/* ----- Push Buttons ----- */
QPushButton {
    background-color: {background_2};
    color: {text_1};
    border: 1px solid {accent_1};
    border-radius: 8px;
    padding: 8px 16px;
    min-width: 60px;
}

QPushButton:hover {
    background-color: {background_3};
    border-color: {accent_2};
}

QPushButton:pressed {
    background-color: {background_4};
    border-color: {accent_3};
}

/* This is the key for the default button (e.g., the 'OK' button) */
QPushButton:default {
    background-color: {accent_1};
    color: {selection_text}; /* Use a contrasting text color */
    border: 2px solid {accent_2};
    font-weight: bold;
}

QPushButton:default:hover {
    background-color: {accent_2};
}

QPushButton:disabled {
    background-color: {background_2};
    color: {text_2};
    border-color: {background_3};
}

/* ----- Input Fields ----- */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: {background_2};
    color: {text_1};
    border: 1px solid {background_3};
    border-radius: 8px;
    padding: 8px;
    selection-background-color: {accent_1};
    selection-color: {selection_text};
}

QSpinBox, QDoubleSpinBox {
    background-color: {background_2};
    border-color: {background_3};
    padding: 4px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid {accent_1};
}

/* ----- ComboBox ----- */
QComboBox {
    background-color: {background_2};
    border: 1px solid {background_3};
    border-radius: 8px;
    padding: 8px;
}

QComboBox:hover {
    border-color: {accent_1};
}

/* Style of the dropdown list itself */
QComboBox QAbstractItemView {
    background-color: {background_3};
    color: {text_1};
    border: 1px solid {accent_1};
    selection-background-color: {accent_1};
    selection-color: {selection_text};
    outline: 0px; /* Remove dotted border on focus */
}


/* ----- CheckBox & RadioButton ----- */

/* ----- ScrollBars ----- */
QScrollBar:vertical, QScrollBar:horizontal {
    border: none;
    background-color: {background_1};
    width: 12px;
    margin: 0px;
}

QScrollBar::handle {
    background-color: {background_3};
    min-height: 20px;
    border-radius: 6px;
}

QScrollBar::handle:hover {
    background-color: {background_4};
}

QScrollBar::handle:pressed {
    background-color: {accent_1};
}

/* Hide the arrows for a modern look */
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0px;
    width: 0px;
}


/* ----- ProgressBar ----- */
QProgressBar {
    border: 1px solid {background_3};
    border-radius: 8px;
    text-align: center;
    color: {text_1};
}

QProgressBar::chunk {
    background-color: {accent_1};
    border-radius: 7px;
    margin: 1px;
}


/* ----- Sliders ----- */
QSlider::groove:horizontal {
    border: 1px solid {background_3};
    height: 4px;
    background: {background_2};
    margin: 2px 0;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: {accent_1};
    border: 1px solid {accent_1};
    width: 16px;
    height: 16px;
    margin: -7px 0; /* Center the handle on the groove */
    border-radius: 9px;
}

QSlider::groove:vertical {
    border: 1px solid {background_3};
    height: 4px;
    background: {background_2};
    margin: 0 2;
    border-radius: 2px;
}

QSlider::handle:vertical {
    background: {accent_1};
    border: 1px solid {accent_1};
    width: 16px;
    height: 16px;
    margin: 0 -7px; /* Center the handle on the groove */
    border-radius: 9px;
}

/* ----- Menus & ToolTips ----- */
QMenuBar {
    background-color: {background_2};
}

QMenuBar::item {
    background: transparent;
    padding: 4px 8px;
}

QMenuBar::item:selected {
    background-color: {accent_1};
    color: {selection_text};
}

QMenu {
    background-color: {background_3};
    border: 1px solid {accent_1};
}

QMenu::item:selected {
    background-color: {accent_1};
    color: {selection_text};
}

QToolTip {
    background-color: {background_4};
    color: {text_1};
    border: 1px solid {background_3};
    padding: 5px;
    border-radius: 4px;
}
/* ----- ListWidget, GroupBox, ToolBox ----- */

/* ----- QListWidget ----- */
QListWidget {
    background-color: {background_2};
    border: 1px solid {background_3};
    border-radius: 8px;
    padding: 4px; /* Add some inner spacing */
    outline: 0px; /* Remove dotted focus border */
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 6px; /* Rounded items */
    background-color: transparent;
}

QListWidget::item:hover {
    background-color: {background_3};
}

QListWidget::item:selected {
    background-color: {accent_1};
    color: {selection_text};
}

/* Style for selected item when widget is not in focus */
QListWidget::item:selected:!active {
    background-color: {background_4};
}


/* ----- QGroupBox ----- */
QGroupBox {
    background-color: {background_2};
    border: 1px solid {background_3};
    border-radius: 8px;
    margin-top: 10px; /* Make space for the title */
    padding: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    border-radius: 4px 4px 0 0;
    left: 10px; /* Position title inside the border */
    background-color: {background_2};
    color: {text_1};
    font-weight: bold;
}


/* ----- QToolBox ----- */
QToolBox::tab {
    background-color: {background_2};
    color: {text_1};
    border: 1px solid {background_3};
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QToolBox::tab:hover {
    background-color: {background_3};
}

QToolBox::tab:selected {
    background-color: {accent_1};
    color: {selection_text};
    border-bottom-color: {accent_1}; /* Make tab look attached */
}

/* The container for the widgets in the toolbox */
QToolBox QWidget {
    background-color: {background_1};
}
"""

NO_COLOR = """
/* ----- Main Window & General ----- */
QMainWindow, QDialog, QWidget {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12pt;
}
#warningFrame {
    border-radius: 8px;
}

/* ----- Labels ----- */

/* ----- Push Buttons ----- */
QPushButton {
    border-radius: 8px;
    padding: 8px 16px;
    min-width: 60px;
}

/* This is the key for the default button (e.g., the 'OK' button) */
QPushButton:default {
    font-weight: bold;
}

/* ----- Input Fields ----- */
QLineEdit, QTextEdit, QPlainTextEdit {
    border-radius: 8px;
    padding: 8px;
}

QSpinBox, QDoubleSpinBox {
    padding: 4px;
}

/* ----- ComboBox ----- */
QComboBox {
    border-radius: 8px;
    padding: 8px;
}

/* Style of the dropdown list itself */
QComboBox QAbstractItemView {
    outline: 0px; /* Remove dotted border on focus */
}


/* ----- CheckBox & RadioButton ----- */

/* ----- ScrollBars ----- */
QScrollBar:vertical, QScrollBar:horizontal {
    width: 12px;
    margin: 0px;
}

QScrollBar::handle {
    min-height: 20px;
    border-radius: 6px;
}

/* Hide the arrows for a modern look */
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0px;
    width: 0px;
}


/* ----- ProgressBar ----- */
QProgressBar {
    border-radius: 8px;
    text-align: center;
}

QProgressBar::chunk {
    border-radius: 7px;
    margin: 1px;
}


/* ----- Sliders ----- */
QSlider::groove:horizontal {
    height: 4px;
    margin: 2px 0;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -7px 0; /* Center the handle on the groove */
    border-radius: 9px;
}

QSlider::groove:vertical {
    height: 4px;
    margin: 0 2;
    border-radius: 2px;
}

QSlider::handle:vertical {
    width: 16px;
    height: 16px;
    margin: 0 -7px; /* Center the handle on the groove */
    border-radius: 9px;
}

/* ----- Menus & ToolTips ----- */
QMenuBar::item {
    padding: 4px 8px;
}

QToolTip {
    padding: 5px;
    border-radius: 4px;
}
/* ----- ListWidget, GroupBox, ToolBox ----- */

/* ----- QListWidget ----- */
QListWidget {
    border-radius: 8px;
    padding: 4px; /* Add some inner spacing */
    outline: 0px; /* Remove dotted focus border */
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 6px; /* Rounded items */
}

/* ----- QGroupBox ----- */
QGroupBox {
    border-radius: 8px;
    margin-top: 10px; /* Make space for the title */
    padding: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    border-radius: 4px 4px 0 0;
    left: 10px; /* Position title inside the border */
    font-weight: bold;
}


/* ----- QToolBox ----- */
QToolBox::tab {
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
"""

# FIXES: QCheckbox

DARK_THEME = {
    # Backgrounds (shades of dark teal/charcoal)
    "background_1": "#1f2b2a",  # Very dark, desaturated teal
    "background_2": "#2a3a38",  # Slightly lighter for inputs/buttons
    "background_3": "#354946",  # Hover state
    "background_4": "#405854",  # Pressed state
    # Text (light colors for contrast)
    "text_1": "#f0f0f0",  # Primary text (off-white)
    "text_2": "#a0a0a0",  # Secondary/disabled text (light grey)
    "selection_text": "#ffffff",  # Text on top of selection (pure white)
    # Accents (based on your rusty orange)
    "accent_1": "#ba552d",  # Main accent color
    "accent_2": "#c96a46",  # Lighter for hover
    "accent_3": "#d87f60",  # Even lighter for pressed
    # Status Colors (slightly softer for a modern feel)
    "danger_color": "#e57373",
    "success_color": "#81c784",
}
NORDIC_DARK = {
    "background_1": "#2E3440",  # Very dark grey-blue
    "background_2": "#3B4252",  # Dark grey-blue
    "background_3": "#434C5E",  # Grey-blue
    "background_4": "#4C566A",  # Lighter grey-blue
    "text_1": "#ECEFF4",  # Light grey/off-white
    "text_2": "#81A1C1",  # Muted blue for disabled text
    "accent_1": "#88C0D0",  # Frosty light blue
    "accent_2": "#8FBCBB",  # Frosty teal
    "accent_3": "#5E81AC",  # Darker blue
    "danger_color": "#BF616A",  # Nordic red
    "selection_text": "#2E3440",  # Text color for on-accent backgrounds
}

LIGHT_THEME = {
    # Backgrounds (clean and light)
    "background_1": "#f5f5f5",  # Main background (light grey)
    "background_2": "#ffffff",  # Inputs/buttons (pure white)
    "background_3": "#e0e0e0",  # Hover state / borders
    "background_4": "#d0d0d0",  # Pressed state
    # Text (based on your dark teal)
    "text_1": "#12403c",  # Primary text color
    "text_2": "#758a88",  # Secondary/disabled (muted teal)
    "selection_text": "#ffffff",  # Text on top of selection (pure white)
    # Accents (based on your rusty orange)
    "accent_1": "#ba552d",  # Main accent color
    "accent_2": "#c96a46",  # Lighter for hover
    "accent_3": "#d87f60",  # Even lighter for pressed
    # Status Colors (strong and clear)
    "danger_color": "#d32f2f",
    "success_color": "#388e3c",
}

ICONS = {
    "checkmark": get_path("assets/icons/checkmark.svg"),
    "down_arrow": get_path("assets/icons/down_arrow.svg"),
}


def apply_theme(template: str = QSS_TEMPLATE, colors: dict = DARK_THEME) -> str:
    """
    Processes the QSS template by replacing color placeholders
    and formatting it into a valid QSS string.
    """
    # is_dark_mode = QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark
    # if is_dark_mode: colors = NORDIC_DARK
    # else: colors = LIGHT_THEME
    # logger.info(f"Applying current theme")
    # # First, replace the {color_name} placeholders with actual hex codes
    # stylesheet = template

    # for color_name, color_value in colors.items():
    #     stylesheet = stylesheet.replace(f"{{{color_name}}}", color_value)
    # for icon_name, icon_path in ICONS.items():
    #     stylesheet = stylesheet.replace(f"{{{icon_name}}}", icon_path)
    pass

    # QApplication.instance().setStyleSheet(stylesheet)
