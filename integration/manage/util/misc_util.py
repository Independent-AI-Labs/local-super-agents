import importlib.util
import os
import zipfile
from datetime import datetime

from integration.data.config import ALL_LOGS


def rotate_logs(log_dir="logs"):
    """
    Compress all existing .log files in `log_dir` into a timestamped ZIP,
    then truncate (empty) each .log file so fresh logs can start.
    If the directory or log files don't exist, create them.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create any missing log files
    for log_file in ALL_LOGS:
        log_path = os.path.join(log_dir, log_file)
        if not os.path.exists(log_path):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("")  # Create an empty log file

    # Create a zip filename with date + time
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"log_{now_str}.zip"
    zip_path = os.path.join(log_dir, zip_filename)

    # Collect all .log files in the log directory
    log_files = [
        f for f in os.listdir(log_dir)
        if f.endswith(".log") and os.path.isfile(os.path.join(log_dir, f))
    ]

    # Create ZIP and add each log file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for log_file in log_files:
            full_path = os.path.join(log_dir, log_file)
            zipf.write(full_path, arcname=log_file)

    # Truncate each log file (create empty new logs)
    for log_file in log_files:
        full_path = os.path.join(log_dir, log_file)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("")  # Just ensure the file is empty


def get_module_path(module):
    """
    Get the file path of a module by its name.
    """
    spec = importlib.util.find_spec(module.__name__)
    if spec is None or not spec.origin:
        raise ImportError(f"Module '{module.__name__}' could not be found.")
    return spec.origin
