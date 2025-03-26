"""Functions for loading and parsing ARC task data."""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator

import numpy as np

from config import TRAINING_DATA_DIR

logger = logging.getLogger(__name__)


def find_task_files(directory: Path = TRAINING_DATA_DIR) -> List[Path]:
    """
    Find all JSON task files in the specified directory.
    
    Args:
        directory: Directory to search for task files
        
    Returns:
        List of Path objects to the JSON task files
    """
    # List all .json files in the specified directory
    task_files = list(directory.glob("*.json"))
    logger.info(f"Found {len(task_files)} task files in {directory}")
    return task_files


def load_task(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse a single ARC task file.
    
    Args:
        file_path: Path to the JSON task file
        
    Returns:
        Dictionary representing the parsed task data or None if an error occurs
    """
    try:
        with open(file_path, 'r') as f:
            task_data = json.load(f)
        
        # Validate structure
        if not _validate_task_structure(task_data):
            logger.warning(f"Invalid task structure in {file_path}")
            return None
        
        return task_data
    
    except Exception as e:
        logger.error(f"Error loading task file {file_path}: {e}")
        return None


def _validate_task_structure(task_data: Dict[str, Any]) -> bool:
    """
    Validate the structure of a loaded task.
    
    Args:
        task_data: Dictionary of task data
        
    Returns:
        True if structure is valid, False otherwise
    """
    # Check for required keys
    if 'train' not in task_data or 'test' not in task_data:
        return False
    
    # Validate training pairs
    if not isinstance(task_data['train'], list) or not task_data['train']:
        return False
    
    for pair in task_data['train']:
        if 'input' not in pair or 'output' not in pair:
            return False
    
    # Validate test pairs
    if not isinstance(task_data['test'], list) or not task_data['test']:
        return False
    
    for pair in task_data['test']:
        if 'input' not in pair:
            return False
        
    return True


def convert_to_numpy(grid_data: List[List[int]]) -> np.ndarray:
    """
    Convert a grid from JSON list format to NumPy array.
    
    Args:
        grid_data: Grid data as a list of lists of integers
        
    Returns:
        NumPy array representation of the grid
    """
    return np.array(grid_data, dtype=np.int32)


def iter_tasks(directory: Path = TRAINING_DATA_DIR) -> Iterator[tuple[str, Dict[str, Any]]]:
    """
    Iterate over all task files in the specified directory.
    
    Args:
        directory: Directory containing task files
        
    Yields:
        Tuples of (task_id, task_data)
    """
    task_files = find_task_files(directory)
    
    for file_path in task_files:
        task_id = file_path.stem
        task_data = load_task(file_path)
        
        if task_data:
            yield task_id, task_data
