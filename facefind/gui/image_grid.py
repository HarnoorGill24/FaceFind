"""Widget displaying a grid of image thumbnails."""

from pathlib import Path  # For handling file paths
from typing import List

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class ImageGrid(QWidget):
    """Simple thumbnail grid with multi-select."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setIconSize(QSize(128, 128))
        self.list.setSpacing(8)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list)

    def load_dir(self, directory: Path, exts=(".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        self.list.clear()
        directory = Path(directory)
        if not directory.exists():
            return
        for p in sorted(directory.glob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                item = QListWidgetItem()
                item.setText(p.name)
                pix = QPixmap(str(p))
                if not pix.isNull():
                    item.setIcon(QIcon(pix.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                self.list.addItem(item)

    def selected_paths(self) -> List[Path]:
        return [Path(i.data(Qt.ItemDataRole.UserRole)) for i in self.list.selectedItems()]
