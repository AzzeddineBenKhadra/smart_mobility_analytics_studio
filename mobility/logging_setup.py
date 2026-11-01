"""Central logging + friendly error messages."""

import logging
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def get_logger(name="mobility"):
    """Return a logger; configure console + rotating file handlers once."""
    global _CONFIGURED
    root = logging.getLogger("mobility")

    if not _CONFIGURED:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

        try:  # the file handler is optional (read-only file systems, etc.)
            LOG_DIR.mkdir(exist_ok=True)
            file_handler = RotatingFileHandler(
                LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=2,
                encoding="utf-8")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError:
            pass

        root.propagate = False
        _CONFIGURED = True

    return root if name == "mobility" else logging.getLogger(name)


def friendly_message(exc, context=""):
    """Turn an exception into a short, actionable message for the UI."""
    prefix = f"{context}: " if context else ""
    text = str(exc)

    if isinstance(exc, MemoryError):
        hint = ("The dataset is too large for this step. Filter the data, "
                "sample it, or use Parquet.")
    elif isinstance(exc, KeyError):
        hint = (f"A column named {text} was not found. If you changed the "
                "dataset or the column mapping, re-select the column.")
    elif isinstance(exc, (ValueError, TypeError)) and "could not convert" in text:
        hint = ("Some values could not be converted to numbers. "
                "Use 'Convert column to numeric' in Data Cleaning first.")
    elif isinstance(exc, ValueError):
        hint = text
    elif isinstance(exc, FileNotFoundError):
        hint = "A required file was not found."
    elif isinstance(exc, PermissionError):
        hint = "The app is not allowed to access that file."
    elif isinstance(exc, ImportError):
        hint = (f"An optional package is missing ({text}). "
                "Install it with pip and restart the app.")
    else:
        hint = f"{type(exc).__name__}: {text}"

    return prefix + hint


def format_traceback(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
