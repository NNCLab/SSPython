import unittest
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from main import MainWindow

class TestUI(unittest.TestCase):

    def test_main_window_initialization(self):
        """Test if the MainWindow can be initialized without errors."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        window = MainWindow()
        self.assertIsInstance(window, MainWindow)
        window.close()

if __name__ == '__main__':
    unittest.main()
