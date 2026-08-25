import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

# --------------------------------------------------
# Log configuration constants
# --------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # number of rotated log files to keep

# --------------------------------------------------
# Resolve project root
# --------------------------------------------------
# This file lives at <project_root>/src/logger/__init__.py, so the
# project root is three directory levels up from this file's absolute
# path: __init__.py -> logger/ -> src/ -> <project_root>.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # .../src/logger
PACKAGE_PARENT_DIR = os.path.dirname(THIS_DIR)               # .../src
ROOT_DIR = os.path.dirname(PACKAGE_PARENT_DIR)                # .../<project_root>

LOG_DIR_PATH = os.path.join(ROOT_DIR, LOG_DIR)
os.makedirs(LOG_DIR_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR_PATH, LOG_FILE)


def configure_logger():
    """
    Configures the root logger with a rotating file handler (INFO+) and a
    console handler (INFO+). Safe to call multiple times — skips
    re-attaching handlers if the root logger is already configured, so
    importing this module from more than one place in the same run
    doesn't duplicate every log line.
    """
    logger = logging.getLogger()

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


configure_logger()