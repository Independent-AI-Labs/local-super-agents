"""
Implementation tracking models for VibeKiller.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class FileStatus(BaseModel):
    """Status information for a single file in the project."""
    path: str
    status: str  # "modified", "added", "deleted", "unchanged"
    diff: Optional[str] = None
    implementation_percentage: float
    last_analyzed: datetime


class ModuleStatus(BaseModel):
    """Status information for a module (group of related files)."""
    name: str
    files: List[FileStatus]
    implementation_percentage: float


class ImplementationData(BaseModel):
    """Container for all implementation tracking data."""
    modules: Dict[str, ModuleStatus]
    overall_percentage: float
