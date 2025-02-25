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


def select_directory(dir_path: str = "") -> str:
    """
    Opens a folder dialog to select a directory or uses the provided path based on conditions.
    Returns the selected directory path or the provided path if conditions are met or dialog is cancelled.
    """
    if not isinstance(dir_path, str):
        raise TypeError("dir_path must be a string")

    if any(var in os.environ for var in ENV_EXCLUSION) or sys.platform == "darwin":
        return dir_path or ""  # Return provided path directly if conditions are met

    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory(initialdir=dir_path or ".")
        root.destroy()
        return selected_folder or dir_path  # Return selected folder or original dir_path if dialog cancelled
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}") from e


def list_directory_contents(dir_path: str, exclude_hidden: bool = True) -> list[list[str]]:
    """
    Lists the contents of the given directory with file sizes.
    Returns a list of lists, each containing filename and size.
    Limits the list size to MAX_LIST_SIZE and excludes hidden files based on exclude_hidden flag.
    """
    if not isinstance(dir_path, str):
        raise TypeError("dir_path must be a string")
    if not isinstance(exclude_hidden, bool):
        raise TypeError("exclude_hidden must be a boolean")

    dir_content_with_sizes = []
    if os.path.isdir(dir_path):
        try:
            items = list(os.scandir(dir_path))[:MAX_LIST_SIZE]  # Limit list size
            for entry in items:
                if not exclude_hidden or not entry.name.startswith("."):  # Apply hidden file exclusion based on flag
                    size = entry.stat().st_size if entry.is_file() else 0  # Get size for files only
                    formatted_size = format_size(size)
                    dir_content_with_sizes.append([entry.name, formatted_size])
        except Exception as e:
            raise RuntimeError(f"Error reading directory contents: {e}") from e
    return dir_content_with_sizes


def select_and_list_directory_contents(dir_path: str = "", exclude_hidden: bool = True) -> tuple[str, list[list[str]]]:
    """
    Wrapper function to select a directory and list its contents.
    Returns the selected folder path and a list of lists, each containing filename and size.
    """
    if not isinstance(dir_path, str):
        raise TypeError("dir_path must be a string")
    if not isinstance(exclude_hidden, bool):
        raise TypeError("exclude_hidden must be a boolean")

    selected_folder = select_directory(dir_path)
    if not os.path.isdir(selected_folder):  # Handle case where select_directory might return non-dir path if conditions are met or dialog is cancelled.
        dir_contents = list_directory_contents(dir_path, exclude_hidden)  # Fallback to original dir_path if selection fails or conditions are met.
        return dir_path, dir_contents
    dir_contents = list_directory_contents(selected_folder, exclude_hidden)
    return selected_folder, dir_contents
