"""
Main application entry point for the improved VibeCheck app.
"""

import os
import gradio as gr

from vibecheck.views.welcome import create_welcome_tab
from vibecheck.views.architecture import create_architecture_tab
from vibecheck.views.implementation import create_implementation_tab
from vibecheck.views.environment import create_environment_tab
from vibecheck.views.build_test import create_build_test_tab


def create_app() -> gr.Blocks:
    """
    Create the main VibeCheck application with improved UI.

    Returns:
        gr.Blocks: The Gradio application
    """

    # Create the application with improved UI
    with gr.Blocks(title="VibeCheck: Software Engineering + LLMs", theme=gr.themes.Soft())as app:
        # Create the main tabs with the welcome tab directly inside
        with gr.Tabs() as tabs:
            # Create welcome tab and get the state
            welcome_tab, state = create_welcome_tab()

            # Create and add other tabs
            architecture_tab = create_architecture_tab(state)
            implementation_tab = create_implementation_tab(state)
            environment_tab = create_environment_tab(state)
            build_test_tab = create_build_test_tab(state)

        # Add a footer with app information
        gr.Markdown(
            """
            <div style="text-align: center; margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee;">
                <p>✨ VibeCheck - Put an end to "vibe coding" and start building proper software. ✨</p>
                <p style="font-size: 0.8em; color: #666;">© 2025 VibeCheck</p>
            </div>
            """
        )

    return app


def main():
    # Create and launch the app
    app = create_app()
    app.launch(share=False)


if __name__ == "__main__":
    main()
