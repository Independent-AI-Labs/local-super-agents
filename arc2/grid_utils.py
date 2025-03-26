"""Enhanced utility functions for grid manipulation and formatting."""

import numpy as np
from colorama import init, Fore, Back, Style
from typing import List, Tuple

# Initialize colorama
init(autoreset=True)

# Color mapping for grid cells
COLOR_MAP = {
    0: Style.DIM + Fore.WHITE + '█',   # Background/empty
    1: Back.RED + ' ',                 # Red
    2: Back.GREEN + ' ',               # Green
    3: Back.BLUE + ' ',                # Blue
    4: Back.YELLOW + ' ',              # Yellow
    5: Back.MAGENTA + ' ',             # Magenta
    6: Back.CYAN + ' ',                # Cyan
    7: Back.WHITE + Fore.BLACK + '█',  # White
    8: Back.BLACK + ' ',               # Black
    9: Back.LIGHTRED_EX + ' '          # Light Red
}

def get_grid_dimensions(grid: np.ndarray) -> Tuple[int, int]:
    """
    Get the dimensions (height, width) of a grid.
    
    Args:
        grid: NumPy array representing the grid
        
    Returns:
        Tuple of (height, width)
    """
    return grid.shape


def get_bottom_left_position(grid_height: int, top_left: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert top-left grid coordinate to bottom-left coordinate.

    In ARC grids, the origin (0,0) is at the top-left. This function
    converts to a coordinate system where (0,0) is at the bottom-left,
    which is more intuitive for some shape operations.

    Args:
        grid_height: Height of the grid
        top_left: (row, column) of the top-left point

    Returns:
        (x, y) coordinates where x=column and y=grid_height-1-row
    """
    row, col = top_left
    return col, grid_height - 1 - row


def calculate_bounding_box(coordinates: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    Calculate the bounding box for a set of coordinates.

    Args:
        coordinates: List of (row, col) coordinates

    Returns:
        Tuple of (top_row, left_col, height, width)
    """
    if not coordinates:
        return (0, 0, 0, 0)

    rows = [r for r, _ in coordinates]
    cols = [c for _, c in coordinates]

    top_row = min(rows)
    left_col = min(cols)
    bottom_row = max(rows)
    right_col = max(cols)

    height = bottom_row - top_row + 1
    width = right_col - left_col + 1

    return (top_row, left_col, height, width)


def get_subgrid(grid: np.ndarray, top_row: int, left_col: int, height: int, width: int) -> np.ndarray:
    """
    Extract a subgrid from the given grid.

    Args:
        grid: The source grid
        top_row: Top row of the subgrid
        left_col: Leftmost column of the subgrid
        height: Height of the subgrid
        width: Width of the subgrid

    Returns:
        NumPy array containing the subgrid
    """
    return grid[top_row:top_row + height, left_col:left_col + width].copy()


def format_grid_ascii(grid: np.ndarray, show_numbers: bool = True) -> str:
    """
    Create a detailed text representation of the grid with numbers.

    Args:
        grid: NumPy array representing the grid
        show_numbers: Whether to show cell numbers in the detailed view

    Returns:
        Detailed text representation of the grid
    """
    detailed_grid = []
    for row in grid:
        if show_numbers:
            detailed_grid.append(' '.join(str(cell).rjust(2) for cell in row))
        else:
            detailed_grid.append(' '.join(str(cell) for cell in row))
    return '\n'.join(detailed_grid)


def print_grid_colored(grid: np.ndarray):
    """
    Print a grid using colored blocks.

    Args:
        grid: NumPy array representing the grid
    """
    for row in grid:
        # Use list comprehension to map each cell to its color
        colored_row = [COLOR_MAP.get(cell, Fore.WHITE + '?') for cell in row]
        print(''.join(colored_row) + Style.RESET_ALL)


def crop_grid_to_shape(grid: np.ndarray) -> np.ndarray:
    """
    Crop a grid to the minimal bounding box containing non-zero elements.

    Args:
        grid: Input numpy grid

    Returns:
        Cropped grid with minimal non-zero region
    """
    # Find rows and columns with non-zero elements
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)

    # If no non-zero elements, return original grid
    if not np.any(rows) or not np.any(cols):
        return grid

    # Find range of non-zero rows and columns
    row_range = np.where(rows)[0]
    col_range = np.where(cols)[0]

    # Crop the grid
    min_row, max_row = row_range[0], row_range[-1] + 1
    min_col, max_col = col_range[0], col_range[-1] + 1

    return grid[min_row:max_row, min_col:max_col]


def grid_to_string(grid: np.ndarray) -> str:
    """
    Convert a grid to a compact string representation.

    Args:
        grid: NumPy array representing the grid

    Returns:
        Compact string representation of the grid
    """
    return '\n'.join(''.join(str(cell) for cell in row) for row in grid)


def grid_to_text_prompt(grid: np.ndarray) -> str:
    """
    Convert a grid to a text prompt-friendly representation.

    Args:
        grid: NumPy array representing the grid

    Returns:
        Text representation of the grid
    """
    return format_grid_ascii(grid)


def parse_grid_from_string(grid_str: str) -> np.ndarray:
    """
    Parse a grid from a string representation.

    Args:
        grid_str: String representation of the grid

    Returns:
        NumPy array grid
    """
    # Remove any whitespace and split into lines
    lines = [line.strip() for line in grid_str.split('\n') if line.strip()]

    # Convert to 2D list of integers
    grid = np.array([list(map(int, list(line))) for line in lines])

    return grid