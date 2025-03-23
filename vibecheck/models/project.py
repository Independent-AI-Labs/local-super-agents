"""
Project-related models for VibeCheck.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from vibecheck.models.architecture import ArchitectureData
from vibecheck.models.implementation import ImplementationData
from vibecheck.models.environment import EnvironmentData
from vibecheck.models.tests import TestData


class ProjectMetadata(BaseModel):
    """Metadata for a VibeCheck project."""
    name: str
    path: str
    created_at: datetime
    last_modified: datetime
    description: Optional[str] = None


class VibeCheckProject(BaseModel):
    """Main project model that contains all project data."""
    metadata: ProjectMetadata
    architecture: ArchitectureData
    implementation: ImplementationData
    environment: EnvironmentData
    tests: TestData
