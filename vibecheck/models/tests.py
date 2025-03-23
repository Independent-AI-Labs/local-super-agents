"""
Test management models for VibeKiller.
"""

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel


class TestResult(BaseModel):
    """Results of an individual test."""
    name: str
    status: str  # "passed", "failed", "error", "skipped"
    duration: float
    output: str


class TestSuiteResult(BaseModel):
    """Results of a test suite (collection of tests)."""
    name: str
    tests: List[TestResult]
    passed_count: int
    failed_count: int
    execution_date: datetime


class TestData(BaseModel):
    """Container for all test-related data."""
    test_suites: Dict[str, TestSuiteResult]
