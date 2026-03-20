
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QToolBox,
    QScrollArea, QGroupBox, QPushButton, QCheckBox, QRadioButton, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QDial, QDateEdit,
    QTimeEdit, QDateTimeEdit, QFontComboBox, QCommandLinkButton, QLabel,
    QLCDNumber, QProgressBar, QListView, QTableView, QTreeView, QTabWidget,
    QStackedWidget, QFrame, QHBoxLayout, QCalendarWidget, QFormLayout,
    QDialog
)
from PySide6.QtCore import Slot, Qt, QStringListModel, QDateTime, QTimer
from PySide6.QtGui import QStandardItemModel, QStandardItem

class QSSEditor(QWidget):
    """
    A widget that provides a QSS editor and a preview of all available widgets.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QSS Editor")

        main_layout = QHBoxLayout(self)

        # QSS Editor
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Enter your QSS here...")
        self.editor.textChanged.connect(self.apply_qss)
        main_layout.addWidget(self.editor)

        # Collapsible widget section
        self.toolbox = QToolBox()
        main_layout.addWidget(self.toolbox)

        # --- Add all widgets in groups ---

        # Input Widgets
        input_widget = QWidget()
        input_layout = QFormLayout(input_widget)
        input_layout.addRow("QPushButton", QPushButton("QPushButton"))
        input_layout.addRow("QCheckBox", QCheckBox("QCheckBox"))
        input_layout.addRow("QRadioButton", QRadioButton("QRadioButton"))
        input_layout.addRow("QCommandLinkButton", QCommandLinkButton("QCommandLinkButton", "Description"))
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("QLineEdit")
        input_layout.addRow("QLineEdit", line_edit)
        combo = QComboBox()
        combo.addItems(["Item 1", "Item 2", "Item 3"])
        input_layout.addRow("QComboBox", combo)
        input_layout.addRow("QFontComboBox", QFontComboBox())
        self.add_widget_group(input_widget, "Input Widgets")

        # Spin Boxes
        spin_widget = QWidget()
        spin_layout = QFormLayout(spin_widget)
        spin_box = QSpinBox()
        spin_box.setValue(10)
        spin_layout.addRow("QSpinBox", spin_box)
        double_spin_box = QDoubleSpinBox()
        double_spin_box.setValue(10.5)
        spin_layout.addRow("QDoubleSpinBox", double_spin_box)
        self.add_widget_group(spin_widget, "Spin Boxes")

        # Sliders and Dials
        slider_widget = QWidget()
        slider_layout = QFormLayout(slider_widget)
        slider = QSlider(Qt.Horizontal)
        slider.setValue(40)
        slider_layout.addRow("QSlider", slider)
        dial = QDial()
        dial.setValue(60)
        slider_layout.addRow("QDial", dial)
        self.add_widget_group(slider_widget, "Sliders and Dials")

        # Date and Time
        datetime_widget = QWidget()
        datetime_layout = QFormLayout(datetime_widget)
        date_edit = QDateEdit()
        date_edit.setDateTime(QDateTime.currentDateTime())
        datetime_layout.addRow("QDateEdit", date_edit)
        time_edit = QTimeEdit()
        time_edit.setDateTime(QDateTime.currentDateTime())
        datetime_layout.addRow("QTimeEdit", time_edit)
        datetime_edit = QDateTimeEdit()
        datetime_edit.setDateTime(QDateTime.currentDateTime())
        datetime_layout.addRow("QDateTimeEdit", datetime_edit)
        self.add_widget_group(datetime_widget, "Date and Time")

        # Display Widgets
        display_widget = QWidget()
        display_layout = QFormLayout(display_widget)
        display_layout.addRow("QLabel", QLabel("QLabel"))
        lcd = QLCDNumber(8)
        timer = QTimer(self)
        timer.timeout.connect(lambda: lcd.display(QDateTime.currentDateTime().toString("hh:mm:ss")))
        timer.start(1000)
        display_layout.addRow("QLCDNumber", lcd)
        progress = QProgressBar()
        progress.setRange(0, 0)
        display_layout.addRow("QProgressBar", progress)
        self.add_widget_group(display_widget, "Display Widgets")
        
        # Calendar
        calendar_widget = QWidget()
        calendar_layout = QVBoxLayout(calendar_widget)
        calendar_layout.addWidget(QLabel("QCalendarWidget"))
        calendar_layout.addWidget(QCalendarWidget())
        self.add_widget_group(calendar_widget, "Calendar")


        # View Widgets
        view_widget = QWidget()
        view_layout = QVBoxLayout(view_widget)
        
        view_layout.addWidget(QLabel("QListView"))
        list_view = QListView()
        list_model = QStringListModel(["Apple", "Banana", "Cherry"])
        list_view.setModel(list_model)
        view_layout.addWidget(list_view)
        
        view_layout.addWidget(QLabel("QTableView"))
        table_view = QTableView()
        table_model = QStandardItemModel(3, 2)
        table_model.setHorizontalHeaderLabels(["Column 1", "Column 2"])
        table_model.setItem(0, 0, QStandardItem("Row 1, Col 1"))
        table_model.setItem(0, 1, QStandardItem("Row 1, Col 2"))
        table_model.setItem(1, 0, QStandardItem("Row 2, Col 1"))
        table_model.setItem(1, 1, QStandardItem("Row 2, Col 2"))
        table_model.setItem(2, 0, QStandardItem("Row 3, Col 1"))
        table_model.setItem(2, 1, QStandardItem("Row 3, Col 2"))
        table_view.setModel(table_model)
        view_layout.addWidget(table_view)
        
        view_layout.addWidget(QLabel("QTreeView"))
        tree_view = QTreeView()
        tree_model = QStandardItemModel()
        root_item = tree_model.invisibleRootItem()
        branch1 = QStandardItem("Branch 1")
        branch1.appendRow(QStandardItem("Leaf 1.1"))
        branch1.appendRow(QStandardItem("Leaf 1.2"))
        root_item.appendRow(branch1)
        branch2 = QStandardItem("Branch 2")
        branch2.appendRow(QStandardItem("Leaf 2.1"))
        root_item.appendRow(branch2)
        tree_view.setModel(tree_model)
        view_layout.addWidget(tree_view)
        self.add_widget_group(view_widget, "View Widgets")

        # Container Widgets
        container_widget = QWidget()
        container_layout = QVBoxLayout(container_widget)
        
        container_layout.addWidget(QLabel("QTabWidget"))
        tab_widget = QTabWidget()
        tab_widget.addTab(QLabel("Content for Tab 1"), "Tab 1")
        tab_widget.addTab(QLabel("Content for Tab 2"), "Tab 2")
        container_layout.addWidget(tab_widget)
        
        stacked_group = QGroupBox("QStackedWidget")
        stacked_layout = QVBoxLayout(stacked_group)
        stacked_widget = QStackedWidget()
        stacked_widget.addWidget(QLabel("This is Page 1"))
        stacked_widget.addWidget(QLabel("This is Page 2"))
        combo_stack = QComboBox()
        combo_stack.addItems(["Page 1", "Page 2"])
        combo_stack.currentIndexChanged.connect(stacked_widget.setCurrentIndex)
        stacked_layout.addWidget(combo_stack)
        stacked_layout.addWidget(stacked_widget)
        container_layout.addWidget(stacked_group)

        container_layout.addWidget(QLabel("QFrame"))
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setLineWidth(2)
        frame_layout = QHBoxLayout(frame)
        frame_layout.addWidget(QLabel("QFrame Content"))
        container_layout.addWidget(frame)
        self.add_widget_group(container_widget, "Container Widgets")

    def add_widget_group(self, widget, title):
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)
        self.toolbox.addItem(scroll_area, title)

    @Slot()
    def apply_qss(self):
        """
        Applies the QSS from the editor to the entire application.
        """
        qss = self.editor.toPlainText()
        QApplication.instance().setStyleSheet(qss)

class QSSEditorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QSS Editor")
        widget = QSSEditor(self)
        layout = QVBoxLayout(self)
        layout.addWidget(widget)
        self.setLayout(layout)

def main():
    """
    Main function to run the application.
    """
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    
    qss_editor_widget = QSSEditor()
    
    main_window.setCentralWidget(qss_editor_widget)
    main_window.setWindowTitle("QSS Editor Application")
    main_window.resize(1000, 800)
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
