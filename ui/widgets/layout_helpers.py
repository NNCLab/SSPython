"""Small layout primitives shared by the workspace pages."""

from math import ceil
from PySide6.QtCore import QEvent, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QTextLayout, QTextOption
from PySide6.QtWidgets import QLabel, QLayout, QSizePolicy, QHBoxLayout, QToolButton, QWidget
from utils import themed_svg_icon


def dataset_label(dataset, *, path=None) -> str:
    """Keep the input filename intact while omitting its file extension."""
    input_path = path or dataset.source_path
    if input_path.name.lower().endswith(".fif.gz"):
        return input_path.name[:-len(".fif.gz")]
    return input_path.stem


class FilenameLabel(QLabel):
    """Keep filenames intact and wrap long, unbroken names at the available width."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setMinimumWidth(0)

    def minimumSizeHint(self):
        return QSize(0, self.fontMetrics().height())

    def sizeHint(self):
        width = min(360, self.fontMetrics().horizontalAdvance(self.text()))
        return QSize(width, self.heightForWidth(width))

    def heightForWidth(self, width):
        return max(self.fontMetrics().height(), ceil(self._text_layout(width).boundingRect().height()))

    def _text_layout(self, width):
        # Add display-only break opportunities without changing the filename
        # exposed through QLabel.text(), accessibility, or the tooltip.
        layout = QTextLayout(self.text().replace("_", "_\u200b"), self.font())
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.beginLayout()
        y = 0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(1, width))
            line.setPosition(QPointF(0, y))
            y += line.height()
        layout.endLayout()
        return layout

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        self._text_layout(self.contentsRect().width()).draw(painter, QPointF(self.contentsRect().topLeft()))


class InspectableLabel(FilenameLabel):
    """A wrapping name that can open information by mouse or keyboard."""

    activated = Signal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.inspectable = False
        self._pressed = False

    def set_inspectable(self, available):
        self.inspectable = available
        self.setProperty("inspectable", available)
        self.setCursor(Qt.CursorShape.PointingHandCursor if available else Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus if available else Qt.FocusPolicy.NoFocus)
        self.setAccessibleDescription("Open file information" if available else "")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        self._pressed = self.inspectable and event.button() == Qt.MouseButton.LeftButton
        if self._pressed:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pressed and event.button() == Qt.MouseButton.LeftButton:
            self._pressed = False
            if self.rect().contains(event.position().toPoint()):
                self.activated.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.inspectable and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.hasFocus() and self.inspectable:
            painter = QPainter(self)
            pen = painter.pen()
            pen.setColor(self.palette().color(self.foregroundRole()))
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class FileInfoButton(QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fileInfoButton")
        self.setIconSize(QSize(18, 18))
        self._refresh_icon()

    def _refresh_icon(self):
        self.setIcon(themed_svg_icon("assets/icons/info.svg", color=self.palette().color(self.foregroundRole()), size=18))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self._refresh_icon()


class InspectionLink(QWidget):
    """Keep a filename or stage name beside a discoverable information button."""

    activated = Signal()

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setObjectName("fileInfoLink")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.label = InspectableLabel(text)
        self.info_button = FileInfoButton()
        self.info_button.setAutoRaise(True)
        self.info_button.setFixedSize(24, 24)
        self.info_button.setToolTip("Show file information")
        self.info_button.setAccessibleName("Show file information")
        self.info_button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.info_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.label, 1)
        self.label.activated.connect(self.activated)
        self.info_button.clicked.connect(self.activated)
        self.set_available(False)

    def set_available(self, available):
        self.label.set_inspectable(available)
        self.info_button.setVisible(available)
        self.info_button.setAccessibleName(f"Show file information for {self.label.text()}")


class FlowLayout(QLayout):
    """Wrap controls and chips instead of imposing a horizontal minimum width."""

    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), test_only=True)

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect)

    def _layout(self, rect, test_only=False):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, row_height = area.x(), area.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            width = min(hint.width(), max(1, area.width()))
            height = item.heightForWidth(width) if item.hasHeightForWidth() else hint.height()
            if x > area.x() and x + width > area.right() + 1:
                x, y = area.x(), y + row_height + self.spacing()
                row_height = 0
            if not test_only:
                item.setGeometry(QRect(x, y, width, height))
            x += width + self.spacing()
            row_height = max(row_height, height)
        return y + row_height - rect.y() + margins.bottom()
