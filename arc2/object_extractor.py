"""Enhanced functions for extracting shapes from ARC grids."""

import numpy as np
from typing import List, Tuple, Set
import scipy.ndimage as ndi

from config import MIN_OBJECT_SIZE
from data_structures import Shape
from grid_utils import (
    get_grid_dimensions, 
    get_bottom_left_position, 
    calculate_bounding_box, 
    get_subgrid,
    format_grid_ascii,
    print_grid_colored
)


def extract_shapes(grid: np.ndarray, background_value: int = 0) -> List[Shape]:
    """
    Extract distinct shapes from a grid, preserving all unique values.
    
    Args:
        grid: NumPy array representing the grid
        background_value: Value to treat as background (default: 0)
        
    Returns:
        List of Shape objects
    """
    # Get grid dimensions
    grid_height, grid_width = get_grid_dimensions(grid)
    
    # Find unique non-background values
    unique_values = np.unique(grid)
    unique_values = unique_values[unique_values != background_value]
    
    shapes = []
    
    # Process each unique value
    for value in unique_values:
        # Create a binary mask for this value
        value_mask = (grid == value)
        
        # Use connected components to identify distinct objects of this value
        structure = np.ones((3, 3), dtype=np.int32)
        labeled_array, num_features = ndi.label(value_mask, structure=structure)
        
        # Process each connected component
        for label in range(1, num_features + 1):
            # Get coordinates of cells in this component
            component_mask = (labeled_array == label)
            cell_coordinates = list(zip(*np.where(component_mask)))
            
            # Skip if object is too small
            if len(cell_coordinates) < MIN_OBJECT_SIZE:
                continue
            
            # Calculate bounding box
            top_row, left_col, height, width = calculate_bounding_box(cell_coordinates)
            
            # Extract subgrid for this shape
            shape_subgrid = get_subgrid(grid, top_row, left_col, height, width)
            
            # Calculate bottom-left position
            bottom_left = get_bottom_left_position(grid_height, (top_row, left_col))
            
            # Create Shape object
            shape = Shape(
                subgrid=shape_subgrid,
                coordinates_in_grid=cell_coordinates,
                bounding_box=(top_row, left_col, height, width),
                bottom_left_position=bottom_left,
                colors={value}  # Use the unique color/value
            )
            
            shapes.append(shape)
    
    return shapes


def print_grid_details(grid: np.ndarray):
    """
    Print detailed information about a grid for debugging.
    
    Args:
        grid: NumPy array representing the grid
    """
    print("Grid Details:")
    print(f"Shape: {grid.shape}")
    print(f"Unique values: {np.unique(grid)}")
    
    print("\nDetailed Grid View (Numbers):")
    print(format_grid_ascii(grid))
    
    print("\nColorful Grid Representation:")
    print_grid_colored(grid)


def extract_and_print_shapes(grid: np.ndarray):
    """
    Extract and print details about shapes in a grid.
    
    Args:
        grid: NumPy array representing the grid
        
    Returns:
        List of extracted Shape objects
    """
    print("\n--- Grid Shape Extraction ---")
    print_grid_details(grid)
    
    shapes = extract_shapes(grid)
    
    print(f"\nFound {len(shapes)} distinct shapes:")
    for i, shape in enumerate(shapes, 1):
        print(f"\nShape {i}:")
        print(f"Bottom-left position: {shape.bottom_left_position}")
        print(f"Bounding box: {shape.bounding_box}")
        print(f"Colors: {shape.colors}")
        print("Subgrid:")
        print(format_grid_ascii(shape.subgrid))
    
    return shapes
