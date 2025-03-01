import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict

from integration.data.config import DEBUG_LOG, USER_LOG
from integration.util.misc_util import touch

# --- Default Logging Utility Names ---
DEFAULT_DEBUG_UTILITY_NAME = "debug"
DEFAULT_USER_UTILITY_NAME = "user"


class CustomFormatter(logging.Formatter):
    """
    Custom formatter to include thread and process information.
    """

    def format(self, record):
        record.thread_id = threading.get_ident()
        record.thread_name = threading.current_thread().name
        record.process_id = os.getpid()
        return super().format(record)


class LoggingUtility:
    """
    Ultimate logging utility with process and thread-safe writing to a specified file.
    """

    def __init__(self, name: str, log_file: Optional[str] = None, log_level=logging.INFO, max_bytes: int = 10 * 1024 * 1024,
                 backup_count: int = 5):
        """
        Initialize the logging utility.

        Args:
            name (str): Name of the logging utility.
            log_file (Optional[str]): Path to the log file. If None, logs to console.
            log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
            max_bytes (int): Maximum size of the log file in bytes before rotation.
            backup_count (int): Number of backup log files to keep.
        """
        self.name = name
        self.logger = logging.getLogger(f"utility_{name}")

        # Clear any existing handlers to avoid duplication
        if self.logger.handlers:
            self.logger.handlers.clear()

        self.logger.setLevel(log_level)
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        self.formatter = CustomFormatter(
            "%(asctime)s - %(process_id)d - %(thread_id)d - %(thread_name)s - %(levelname)s - %(message)s"
        )

        self._add_console_handler()
        if log_file:
            self._add_file_handler(log_file, max_bytes, backup_count)

    def _add_console_handler(self):
        """
        Add a console handler to the logger.
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self, log_file: str, max_bytes: int, backup_count: int):
        """
        Add a file handler to the logger.

        Args:
            log_file (str): Path to the log file.
            max_bytes (int): Maximum size of the log file in bytes.
            backup_count (int): Number of backup files to keep.
        """
        # Make sure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        touch(log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            encoding="utf-8",
            maxBytes=max_bytes,
            backupCount=backup_count,
            delay=True  # Delay file opening until first log
        )
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger


# --- Individual module-specific locks instead of class-level locks ---
_utilities_lock = threading.RLock()  # Use RLock to allow recursive acquisition
_utilities: Dict[str, LoggingUtility] = {}


def create_logging_utility(name: str, log_file: Optional[str] = None, log_level=logging.INFO,
                           max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> LoggingUtility:
    """
    Creates and registers a new LoggingUtility.

    Args:
        name (str): Name of the logging utility.
        log_file (Optional[str]): Path to the log file. If None, logs to console.
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        max_bytes (int): Maximum size of the log file in bytes before rotation.
        backup_count (int): Number of backup log files to keep.

    Returns:
        LoggingUtility: The created or existing logging utility.
    """
    # Check without lock first for performance
    if name in _utilities:
        return _utilities[name]

    # Use lock for creation and registration
    with _utilities_lock:
        # Double-check inside lock
        if name in _utilities:
            return _utilities[name]

        utility = LoggingUtility(name, log_file, log_level, max_bytes, backup_count)
        _utilities[name] = utility
        return utility

def get_utility_logger(utility_name: str) -> logging.Logger:
    """
    Gets the logger of the utility itself.

    Args:
        utility_name (str): Name of the logging utility.

    Returns:
        logging.Logger: The utility logger.

    Raises:
        ValueError: If the specified logging utility does not exist.
    """
    if utility_name not in _utilities:
        # Try to create default utility if it doesn't exist
        if utility_name == DEFAULT_DEBUG_UTILITY_NAME:
            create_logging_utility(DEFAULT_DEBUG_UTILITY_NAME, DEBUG_LOG, logging.DEBUG, 5 * 1024 * 1024, 3)
        elif utility_name == DEFAULT_USER_UTILITY_NAME:
            create_logging_utility(DEFAULT_USER_UTILITY_NAME, USER_LOG, logging.INFO, 5 * 1024 * 1024, 3)
        else:
            raise ValueError(f"Logging utility '{utility_name}' not found and is not a default utility.")

    return _utilities[utility_name].get_logger()


# --- Initialize default utilities lazily ---
# Default loggers will be created on first use
DEBUG_LOGGER = None
USER_LOGGER = None


# Function to get default loggers (lazy initialization)
def get_debug_logger():
    global DEBUG_LOGGER
    if DEBUG_LOGGER is None:
        DEBUG_LOGGER = get_utility_logger(DEFAULT_DEBUG_UTILITY_NAME)
    return DEBUG_LOGGER


def get_user_logger():
    global USER_LOGGER
    if USER_LOGGER is None:
        USER_LOGGER = get_utility_logger(DEFAULT_USER_UTILITY_NAME)
    return USER_LOGGER


get_debug_logger()
get_user_logger()
