"""Desktop application entry point."""

from __future__ import annotations
from .themes import LIGHT_APP_STYLE, DARK_APP_STYLE
from .settings_memory import load_theme

import os
import sys

os.environ.setdefault("QT_API", "pyside6")


def _ensure_standard_streams() -> None:
    """Provide writable streams for console-oriented libraries in windowed builds."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


_ensure_standard_streams()

import matplotlib

matplotlib.use("QtAgg")

from ..qt_compat import QApplication, run_application
from .views import ChromatoPyMainWindow

def main() -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("chromatoPy")
    theme = load_theme()
    app.setStyleSheet(DARK_APP_STYLE if theme == "dark" else LIGHT_APP_STYLE)

    window = ChromatoPyMainWindow()
    window.show()

    if owns_app:
        return run_application(app)
    return 0
