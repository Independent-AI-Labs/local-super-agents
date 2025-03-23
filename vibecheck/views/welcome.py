"""
Minimalist welcome screen for the VibeCheck app with streamlined project creation.
"""

import os
from typing import Dict, List, Tuple

import gradio as gr

from vibecheck.controllers.project_controller import ProjectController
from vibecheck.utils.file_utils import detect_project_type, ensure_directory
from integration.util.misc_util import select_and_list_directory_contents, list_directory_contents


def create_welcome_tab() -> Tuple[gr.Tab, Dict]:
    """
    Create a minimalist welcome tab for the VibeCheck app.
    - Requires directory selection first
    - Checks for .git directory
    - Creates .vibecheck directory if not exists
    - Only shows project details after directory selection

    Returns:
        Tuple containing the tab and a dictionary of important components
    """
    # State variables to be used across components
    state = {
        "current_project": None,
        "current_path": os.getcwd(),
    }

    # Create the welcome tab
    with gr.Tab("🏠 Welcome") as welcome_tab:
        # Create a more appealing header with emojis and better typography
        gr.Markdown(
            """
            # ✨ VibeCheck ✨

            VibeCheck helps you manage your software projects by enforcing proper software
            engineering practices, even when using Large Language Models (LLMs) to generate code.
            """
        )

        # Project selection and creation UI
        with gr.Column():
            # Large centered select directory button
            with gr.Row(elem_id="directory-selection-row"):
                with gr.Column(scale=1):
                    pass  # Spacer for centering
                
                with gr.Column(scale=2, elem_id="select-directory-container"):
                    gr.Markdown("### Select a Project Directory")
                    select_dir_btn = gr.Button("📂 Select Directory", size="lg", variant="primary")
                    
                    # Current path display
                    current_dir = gr.Textbox(
                        label="Current Directory",
                        value=os.getcwd(),
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    pass  # Spacer for centering
            
            # Project details section - initially hidden
            with gr.Row(visible=False) as project_details:
                # Left column for project configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Project Configuration")
                    
                    # Project details with emojis
                    project_name = gr.Textbox(
                        label="📝 Project Name",
                        placeholder="Enter project name",
                        value=""
                    )

                    project_description = gr.Textbox(
                        label="📋 Project Description",
                        placeholder="Briefly describe your project (optional)",
                        lines=3
                    )
                    
                    # Save button
                    save_project_btn = gr.Button("💾 Save Project", variant="primary", size="lg")
                    
                    # Status message for feedback
                    status_msg = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=False
                    )
                
                # Right column for project type and next steps
                with gr.Column(scale=1):
                    # Project type tags
                    gr.Markdown("### 📦 Project Type")
                    project_type_tags = gr.Dataframe(
                        headers=["📦 Project Type"],
                        col_count=(1, "fixed"),
                        interactive=False,
                        value=[]
                    )
                    
                    # Next steps
                    gr.Markdown(
                        """
                        ### 🚀 Next Steps
                        
                        After saving your project:
                        
                        - Click on the **Architecture** tab to design your software architecture
                        - Use the **Environment** tab to configure your development environment
                        - Track implementation progress in the **Implementation** tab
                        - Run tests in the **Build & Test** tab
                        """
                    )

    # Function to select directory and check for git/vibecheck
    def select_directory() -> Tuple[str, gr.update, str, str, List[List[str]]]:
        current_path = state.get("current_path", os.getcwd())
        selected_path, _ = select_and_list_directory_contents(current_path)
        state["current_path"] = selected_path
        
        # Check if it's a git repository
        git_dir = os.path.join(selected_path, ".git")
        if not os.path.isdir(git_dir):
            # Return empty project types and show warning
            return selected_path, gr.update(visible=False), "", "", []
        
        # Create .vibecheck directory if it doesn't exist
        vibecheck_dir = os.path.join(selected_path, ".vibecheck")
        ensure_directory(vibecheck_dir)
        
        # Detect project types regardless of whether a VibeCheck project exists
        project_types = detect_project_type(selected_path)
        project_type_list = [[pt] for pt in project_types]
        
        # Check if a VibeCheck project already exists
        project = ProjectController.load_project(selected_path)
        if project:
            state["current_project"] = project
            
            # Update state and return values
            return (
                selected_path, 
                gr.update(visible=True),
                project.metadata.name,
                project.metadata.description or "",
                project_type_list
            )
        else:
            # Extract project name from directory
            suggested_name = os.path.basename(selected_path)
            return selected_path, gr.update(visible=True), suggested_name, "", project_type_list

    # Function to save/create a project
    def save_project(path: str, name: str, description: str) -> str:
        if not name:
            return "⚠️ Project name is required."

        try:
            # Check if project already exists
            existing_project = ProjectController.load_project(path)
            
            if existing_project:
                # Update existing project
                existing_project.metadata.name = name
                existing_project.metadata.description = description
                ProjectController.save_project(existing_project)
                project = existing_project
            else:
                # Create new project
                project = ProjectController.create_project(name, path, description)
            
            state["current_project"] = project
            return f"✅ Project saved successfully."
            
        except Exception as e:
            return f"❌ Error saving project: {str(e)}"

    # Connect event handlers
    select_dir_btn.click(
        select_directory,
        inputs=None,
        outputs=[current_dir, project_details, project_name, project_description, project_type_tags]
    )
    
    save_project_btn.click(
        save_project,
        inputs=[current_dir, project_name, project_description],
        outputs=[status_msg]
    ).then(
        lambda: gr.update(visible=True),
        None,
        [status_msg]
    )
    
    # Return the welcome tab and the state
    return welcome_tab, state
