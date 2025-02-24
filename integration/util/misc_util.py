import os
import sys
import tkinter as tk
from tkinter import filedialog

import chardet


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def concatenate_py_files(repo_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    try:
                        # Detect the encoding of the file
                        encoding = detect_encoding(file_path)
                        with open(file_path, 'r', encoding=encoding) as infile:
                            outfile.write(f"# File: {file_path}\n")
                            outfile.write(infile.read())
                            outfile.write("\n\n" + "#" * 79 + "\n\n")
                    except (UnicodeDecodeError, FileNotFoundError) as e:
                        print(f"Skipping file {file_path} due to encoding error: {e}")
                        # Try with utf-8
                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                outfile.write(f"# File: {file_path}\n")
                                outfile.write(infile.read())
                                outfile.write("\n\n" + "#" * 79 + "\n\n")
                        except UnicodeDecodeError as e:
                            print(f"Failed to read file {file_path} with utf-8 encoding: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {file_path}: {e}")


def format_size(size_bytes):
    """Formats file size in a human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


ENV_EXCLUSION = []  # Define ENV_EXCLUSION as an empty list
MAX_LIST_SIZE = 256  # Define maximum number of items to display in directory content


def get_folder_path(folder_path: str = "") -> tuple[str, list[list[str]]]:  # Modified return type
    """
     Opens a folder dialog to select a folder and lists its content with file sizes.
     Returns the selected folder path and a list of lists, each containing filename and size.
     Limits the list size to MAX_LIST_SIZE.
     """
    # Validate parameter type
    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")

    dir_content_with_sizes = []
    try:
        # Check for environment variable conditions
        if any(var in os.environ for var in ENV_EXCLUSION) or sys.platform == "darwin":
            if os.path.isdir(folder_path):
                items = list(os.scandir(folder_path))[:MAX_LIST_SIZE]  # Limit list size
                for entry in items:
                    size = entry.stat().st_size if entry.is_file() else 0  # Get size for files only
                    formatted_size = format_size(size)
                    dir_content_with_sizes.append([entry.name, formatted_size])  # Modified to list of lists
            return folder_path or "", dir_content_with_sizes

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory(initialdir=folder_path or ".")
        root.destroy()
        if selected_folder:
            items = list(os.scandir(selected_folder))[:MAX_LIST_SIZE]  # Limit list size
            for entry in items:
                size = entry.stat().st_size if entry.is_file() else 0  # Get size for files only
                formatted_size = format_size(size)
                dir_content_with_sizes.append([entry.name, formatted_size])  # Modified to list of lists
        return selected_folder or folder_path, dir_content_with_sizes
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}") from e
