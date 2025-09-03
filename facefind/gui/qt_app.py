#!/usr/bin/env python3
"""Bootstraps the FaceFind Qt application."""
import sys
import traceback
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def install_excepthook():
    def _hook(exctype, value, tb):
        # Print to stderr instead of aborting the Qt app
        traceback.print_exception(exctype, value, tb)

    sys.excepthook = _hook


def main():
    install_excepthook()
    app = QApplication(sys.argv)
    app.setApplicationName("FaceFind")
    win = MainWindow(repo_root=Path(__file__).resolve().parents[1])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
