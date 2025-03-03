import locale
import os
import sys

from integration.util.logging_util import USER_LOGGER, DEBUG_LOGGER


def sanitize_message(message):
    """
    Sanitize message for logging by encoding/decoding based on the system's terminal encoding.

    On Windows systems where the terminal encoding starts with 'cp' (e.g. cp1252, cp437),
    non-representable characters are replaced with their backslash escape sequences.
    On Linux systems (typically UTF-8), the message is returned unchanged.
    """
    # Use sys.stdout.encoding if available, otherwise fallback to locale preferred encoding.
    encoding = sys.stdout.encoding or locale.getpreferredencoding()
    if encoding.lower().startswith('cp'):
        return message.encode(encoding, 'backslashreplace').decode(encoding)
    return message


class LoggingService:
    """Logging service for centralizing all logging operations."""
    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = LoggingService()
        return cls._instance

    def __init__(self):
        """Initialize the logging service."""
        if LoggingService._instance is not None:
            raise Exception("LoggingService is a singleton. Use get_instance() to access it.")
        self.user_logger = USER_LOGGER
        self.debug_logger = DEBUG_LOGGER

    def log_user(self, message):
        """Log message to user console."""
        self.user_logger.info(sanitize_message(message))

    def log_debug(self, message):
        """Log debug message."""
        self.debug_logger.info(sanitize_message(message))

    def log_error(self, message, exc_info=False):
        """Log error message to both user and debug logs."""
        self.user_logger.error(sanitize_message(message))
        self.debug_logger.error(sanitize_message(message), exc_info=exc_info)

    def log_warning(self, message):
        """Log warning message to both user and debug logs."""
        self.user_logger.warning(sanitize_message(message))
        self.debug_logger.warning(sanitize_message(message))

    def read_debug_log(self, log_path, max_lines=100):
        """Read the last N lines from the debug log file."""
        if not os.path.exists(log_path):
            return "No log file found."

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return ''.join(lines[-max_lines:])
        except Exception as e:
            return f"Error reading log file: {str(e)}"
