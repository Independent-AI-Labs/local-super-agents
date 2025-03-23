"""
Git integration for VibeCheck.

This module provides functions for integrating with Git to track changes,
get file history, and manage source code versioning.
"""

import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def is_git_repository(path: str) -> bool:
    """
    Check if a directory is a Git repository.

    Args:
        path (str): The path to check

    Returns:
        bool: True if the directory is a Git repository, False otherwise
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking if directory is a Git repository: {e}")
        return False


def initialize_git_repository(path: str) -> bool:
    """
    Initialize a new Git repository.

    Args:
        path (str): The path to initialize the repository in

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    if is_git_repository(path):
        return True
    
    try:
        result = subprocess.run(
            ['git', 'init'],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error initializing Git repository: {e}")
        return False


def get_file_status(path: str) -> Dict[str, str]:
    """
    Get the status of files in a Git repository.

    Args:
        path (str): The path to the repository

    Returns:
        Dict[str, str]: A dictionary mapping file paths to their status
            (modified, added, deleted, unchanged, untracked)
    """
    if not is_git_repository(path):
        return {}
    
    statuses: Dict[str, str] = {}
    
    try:
        # Get status of tracked files with changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        for line in result.stdout.splitlines():
            if not line or len(line) < 3:
                continue
                
            status_code = line[:2].strip()
            file_path = line[3:].strip()
            
            status_mapping = {
                'M': 'modified',
                'A': 'added',
                'D': 'deleted',
                'R': 'renamed',
                'C': 'copied',
                'U': 'updated but unmerged',
                '??': 'untracked'
            }
            
            status = status_mapping.get(status_code, 'unknown')
            statuses[file_path] = status
        
        # Get list of all tracked files
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        for line in result.stdout.splitlines():
            file_path = line.strip()
            if file_path and file_path not in statuses:
                statuses[file_path] = 'unchanged'
        
        return statuses
    
    except subprocess.SubprocessError as e:
        print(f"Error getting Git file status: {e}")
        return {}


def get_file_diff(path: str, file_path: str) -> Optional[str]:
    """
    Get the diff for a file.

    Args:
        path (str): The path to the repository
        file_path (str): The path to the file, relative to the repository root

    Returns:
        Optional[str]: The diff, or None if there is no diff or an error occurred
    """
    if not is_git_repository(path):
        return None
    
    try:
        result = subprocess.run(
            ['git', 'diff', '--', file_path],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        return result.stdout if result.stdout else None
    except subprocess.SubprocessError as e:
        print(f"Error getting Git diff: {e}")
        return None


def get_file_history(path: str, file_path: str, limit: int = 10) -> List[Dict]:
    """
    Get the commit history for a file.

    Args:
        path (str): The path to the repository
        file_path (str): The path to the file, relative to the repository root
        limit (int, optional): The maximum number of commits to return. Defaults to 10.

    Returns:
        List[Dict]: A list of dictionaries with commit information (hash, author, date, message)
    """
    if not is_git_repository(path):
        return []
    
    try:
        result = subprocess.run(
            ['git', 'log', f'-{limit}', '--date=iso', '--pretty=format:%H|%an|%ad|%s', '--', file_path],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        commits = []
        for line in result.stdout.splitlines():
            if not line:
                continue
                
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
                
            commit_hash, author, date_str, message = parts
            
            try:
                # Parse ISO date format
                date = datetime.fromisoformat(date_str.replace(' ', 'T').replace(' -', '-'))
            except ValueError:
                date = None
            
            commits.append({
                'hash': commit_hash,
                'author': author,
                'date': date,
                'message': message
            })
        
        return commits
    
    except subprocess.SubprocessError as e:
        print(f"Error getting Git file history: {e}")
        return []


def stage_file(path: str, file_path: str) -> bool:
    """
    Stage a file for commit.

    Args:
        path (str): The path to the repository
        file_path (str): The path to the file, relative to the repository root

    Returns:
        bool: True if the file was staged successfully, False otherwise
    """
    if not is_git_repository(path):
        return False
    
    try:
        result = subprocess.run(
            ['git', 'add', file_path],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error staging file: {e}")
        return False


def commit_changes(path: str, message: str) -> bool:
    """
    Commit staged changes.

    Args:
        path (str): The path to the repository
        message (str): The commit message

    Returns:
        bool: True if the commit was successful, False otherwise
    """
    if not is_git_repository(path):
        return False
    
    try:
        result = subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error committing changes: {e}")
        return False


def get_commit_diff(path: str, commit_hash: str) -> Optional[str]:
    """
    Get the diff for a commit.

    Args:
        path (str): The path to the repository
        commit_hash (str): The hash of the commit

    Returns:
        Optional[str]: The diff, or None if there is no diff or an error occurred
    """
    if not is_git_repository(path):
        return None
    
    try:
        result = subprocess.run(
            ['git', 'show', commit_hash],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        return result.stdout if result.stdout else None
    except subprocess.SubprocessError as e:
        print(f"Error getting commit diff: {e}")
        return None
