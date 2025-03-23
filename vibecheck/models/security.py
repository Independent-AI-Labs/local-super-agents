"""
Security analysis models for VibeKiller.
"""

from datetime import datetime
from typing import List

from pydantic import BaseModel


class SecurityVulnerability(BaseModel):
    """Represents a single security vulnerability found in the code."""
    severity: str  # "critical", "high", "medium", "low", "info"
    description: str
    location: str  # File and line number
    recommendation: str


class SecurityAnalysis(BaseModel):
    """Results of a security analysis for a specific file or module."""
    path: str
    vulnerabilities: List[SecurityVulnerability]
    analysis_date: datetime
    llm_insights: str
