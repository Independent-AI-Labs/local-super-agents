"""
File utilities for VibeCheck.

This module provides utility functions for file operations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Set, Tuple


def ensure_directory(directory_path: str) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path (str): The path to the directory

    Returns:
        bool: True if the directory exists or was created successfully, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def list_files(directory_path: str, include_extensions: Optional[List[str]] = None,
              exclude_dirs: Optional[List[str]] = None) -> List[str]:
    """
    List files in a directory, optionally filtered by extension.

    Args:
        directory_path (str): The path to the directory
        include_extensions (Optional[List[str]], optional): File extensions to include. Defaults to None.
        exclude_dirs (Optional[List[str]], optional): Directories to exclude. Defaults to None.

    Returns:
        List[str]: A list of file paths relative to the directory_path
    """
    file_list = []
    
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        if exclude_dirs:
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            # Check if the file has an included extension
            if include_extensions and not any(file.endswith(ext) for ext in include_extensions):
                continue
                
            # Get the relative path
            rel_path = os.path.relpath(os.path.join(root, file), directory_path)
            file_list.append(rel_path)
    
    return file_list


def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.

    Args:
        file_path (str): The path to the file

    Returns:
        str: The file extension, or an empty string if there is no extension
    """
    return os.path.splitext(file_path)[1].lower()


def read_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    Read a text file.

    Args:
        file_path (str): The path to the file
        encoding (str, optional): The file encoding. Defaults to 'utf-8'.

    Returns:
        Optional[str]: The file contents, or None if the file could not be read
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Write to a text file.

    Args:
        file_path (str): The path to the file
        content (str): The content to write
        encoding (str, optional): The file encoding. Defaults to 'utf-8'.

    Returns:
        bool: True if the file was written successfully, False otherwise
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        return False


def copy_file(src_path: str, dst_path: str) -> bool:
    """
    Copy a file.

    Args:
        src_path (str): The source path
        dst_path (str): The destination path

    Returns:
        bool: True if the file was copied successfully, False otherwise
    """
    try:
        # Make sure the destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"Error copying file from {src_path} to {dst_path}: {e}")
        return False


def delete_file(file_path: str) -> bool:
    """
    Delete a file.

    Args:
        file_path (str): The path to the file

    Returns:
        bool: True if the file was deleted successfully, False otherwise
    """
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """
    Get the size of a file in bytes.

    Args:
        file_path (str): The path to the file

    Returns:
        Optional[int]: The file size in bytes, or None if the file does not exist
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        print(f"Error getting size of file {file_path}: {e}")
        return None


def is_file_empty(file_path: str) -> bool:
    """
    Check if a file is empty.

    Args:
        file_path (str): The path to the file

    Returns:
        bool: True if the file is empty, False otherwise
    """
    size = get_file_size(file_path)
    return size == 0 if size is not None else True


def find_files_with_pattern(directory_path: str, pattern: str) -> List[str]:
    """
    Find files matching a glob pattern.

    Args:
        directory_path (str): The path to the directory
        pattern (str): The glob pattern to match

    Returns:
        List[str]: A list of file paths relative to the directory_path
    """
    try:
        paths = list(Path(directory_path).glob(pattern))
        return [str(p.relative_to(directory_path)) for p in paths if p.is_file()]
    except Exception as e:
        print(f"Error finding files with pattern {pattern} in {directory_path}: {e}")
        return []


def find_directories(directory_path: str, exclude_hidden: bool = True) -> List[str]:
    """
    Find all directories within a directory.

    Args:
        directory_path (str): The path to the directory
        exclude_hidden (bool, optional): Whether to exclude hidden directories. Defaults to True.

    Returns:
        List[str]: A list of directory paths relative to the directory_path
    """
    try:
        result = []
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                if exclude_hidden and item.startswith('.'):
                    continue
                result.append(item)
        return result
    except Exception as e:
        print(f"Error finding directories in {directory_path}: {e}")
        return []


def detect_project_type(directory_path: str) -> List[str]:
    """
    Detect the project type based on files in the directory.

    Args:
        directory_path (str): The path to the directory

    Returns:
        List[str]: A list of detected project types (e.g. ['python', 'flask'])
    """
    project_types = []
    
    # Check for Python project
    if os.path.exists(os.path.join(directory_path, 'setup.py')) or \
       os.path.exists(os.path.join(directory_path, 'pyproject.toml')) or \
       list(Path(directory_path).glob('**/*.py')):
        project_types.append('python')
        
        # Check for specific Python frameworks
        if os.path.exists(os.path.join(directory_path, 'manage.py')):
            project_types.append('django')
        elif list(Path(directory_path).glob('**/app.py')) or \
             list(Path(directory_path).glob('**/flask_app.py')):
            project_types.append('flask')
    
    # Check for JavaScript/Node.js project
    if os.path.exists(os.path.join(directory_path, 'package.json')):
        project_types.append('javascript')
        project_types.append('nodejs')
        
        # Check for specific JS frameworks
        with open(os.path.join(directory_path, 'package.json'), 'r') as f:
            try:
                import json
                package_json = json.load(f)
                dependencies = {**package_json.get('dependencies', {}), **package_json.get('devDependencies', {})}
                
                if 'react' in dependencies:
                    project_types.append('react')
                if 'vue' in dependencies:
                    project_types.append('vue')
                if 'angular' in dependencies or '@angular/core' in dependencies:
                    project_types.append('angular')
            except json.JSONDecodeError:
                pass
    
    # Check for Java project
    if list(Path(directory_path).glob('**/*.java')) or \
       os.path.exists(os.path.join(directory_path, 'pom.xml')) or \
       os.path.exists(os.path.join(directory_path, 'build.gradle')):
        project_types.append('java')
        
        # Check for specific Java frameworks
        if os.path.exists(os.path.join(directory_path, 'pom.xml')):
            try:
                with open(os.path.join(directory_path, 'pom.xml'), 'r') as f:
                    content = f.read()
                    if 'springframework' in content:
                        project_types.append('spring')
            except Exception:
                pass
    
    return project_types
