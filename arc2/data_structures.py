"""Data structures for representing ARC task components."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Any, Optional
import numpy as np


@dataclass
class Shape:
    """Represents a single shape extracted from a grid."""
    
    # Grid representation of the shape (sub-grid)
    subgrid: np.ndarray
    
    # Coordinates of shape cells in the original grid [(row, col), ...]
    coordinates_in_grid: List[Tuple[int, int]]
    
    # Bounding box: (top_row, left_col, height, width)
    bounding_box: Tuple[int, int, int, int]
    
    # Bottom-left position (x, y) coordinates
    bottom_left_position: Tuple[int, int] = None
    
    # Set of unique colors (integers) in the shape
    colors: Set[int] = field(default_factory=set)
    
    # Textual description of the shape (provided by Gemini)
    description: str = None
    
    # Optional unique identifier for the shape (for matching)
    shape_id: Optional[str] = None


@dataclass
class ProcessedPair:
    """Represents a processed input/output pair from an ARC task."""
    
    # Original grids
    input_grid: np.ndarray
    output_grid: np.ndarray
    
    # Extracted shapes
    input_shapes: List[Shape] = field(default_factory=list)
    output_shapes: List[Shape] = field(default_factory=list)


@dataclass
class ProcessedTask:
    """Represents a fully processed ARC task."""
    
    # Task identifier
    task_id: str
    
    # Processed training pairs
    train_pairs: List[ProcessedPair] = field(default_factory=list)
    
    # Processed test pairs
    test_pairs: List[ProcessedPair] = field(default_factory=list)


@dataclass
class PredictedShape:
    """Represents a shape predicted by the solver."""
    
    # Description of the shape
    description: str
    
    # Bottom-left position (x, y)
    position: Tuple[int, int]
    
    # Optional predicted dimensions of the output grid
    grid_dimensions: Optional[Tuple[int, int]] = None
