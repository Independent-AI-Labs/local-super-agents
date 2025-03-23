"""
User interface views for VibeCheck.

This package provides Gradio-based UI components for the VibeCheck application.
"""

from vibecheck.views.welcome import create_welcome_tab
from vibecheck.views.architecture import create_architecture_tab
from vibecheck.views.implementation import create_implementation_tab
from vibecheck.views.environment import create_environment_tab
from vibecheck.views.build_test import create_build_test_tab

__all__ = [
    "create_welcome_tab",
    "create_architecture_tab",
    "create_implementation_tab",
    "create_environment_tab",
    "create_build_test_tab"
]
