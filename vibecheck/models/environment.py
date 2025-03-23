"""
Environment management models for VibeKiller.
"""

from typing import List, Optional

from pydantic import BaseModel


class Dependency(BaseModel):
    """Represents a project dependency."""
    name: str
    version: str
    is_dev_dependency: bool = False


class VirtualEnvironment(BaseModel):
    """Represents a virtual environment configuration."""
    name: str
    type: str  # "conda", "venv", "poetry", etc.
    path: str
    python_version: str


class Compiler(BaseModel):
    """Represents a compiler configuration."""
    name: str
    version: str
    path: str


class EnvironmentVariable(BaseModel):
    """Represents an environment variable."""
    name: str
    value: str
    description: Optional[str] = None
    is_secret: bool = False


class EnvironmentData(BaseModel):
    """Container for all environment-related data."""
    dependencies: List[Dependency]
    virtual_environments: List[VirtualEnvironment]
    compilers: List[Compiler]
    environment_variables: List[EnvironmentVariable] = []
    active_environment: Optional[str] = None
