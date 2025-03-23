"""
Architecture-related models for VibeKiller.
"""

from datetime import datetime
from typing import Dict

from pydantic import BaseModel


class ArchitecturalDocument(BaseModel):
    """Represents an architectural document for a project."""
    content: str
    last_modified: datetime


class ArchitecturalDiagram(BaseModel):
    """Represents a diagram generated from an architectural document."""
    diagram_type: str  # "dataflow", "security", "module", etc.
    content: str  # SVG or other serializable format
    generated_at: datetime


class ArchitectureData(BaseModel):
    """Container for all architecture-related data."""
    document: ArchitecturalDocument
    diagrams: Dict[str, ArchitecturalDiagram]
