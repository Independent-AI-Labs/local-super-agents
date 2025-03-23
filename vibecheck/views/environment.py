"""
Improved environment management view for the VibeCheck app.
"""

from typing import Dict, List, Optional, Tuple

import gradio as gr

from ..controllers.environment_controller import EnvironmentController
from ..models.environment import Dependency, VirtualEnvironment, Compiler, EnvironmentVariable


def create_environment_tab(state: Dict) -> gr.Tab:
    """
    Create the environment tab for the VibeCheck app with improved visual design.

    Args:
        state: Application state dictionary

    Returns:
        The environment tab component
    """
    with gr.Tab("🔧 Environment") as environment_tab:
        gr.Markdown(
            """
            # 🔧 Environment Manager
            
            Manage your development environment, dependencies, and configuration
            to ensure consistent builds and reproducible results.
            """
        )
        
        with gr.Tabs() as env_tabs:
            # Dependencies tab with improved styling
            with gr.Tab("📦 Dependencies") as dependencies_tab:
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 Project Dependencies")
                        
                        dependencies_table = gr.Dataframe(
                            headers=["📦 Name", "🏷️ Version", "🔄 Type"],
                            col_count=(3, "fixed"),
                            interactive=False,
                            elem_id="dependencies-table"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Add Dependency")
                        
                        dep_name = gr.Textbox(
                            label="📦 Package Name",
                            placeholder="e.g., numpy"
                        )
                        
                        dep_version = gr.Textbox(
                            label="🏷️ Version",
                            placeholder="e.g., 1.22.0"
                        )
                        
                        dep_type = gr.Radio(
                            choices=[("Production", "Production"), ("Development", "Development")],
                            label="🔄 Dependency Type",
                            value="Production"
                        )
                        
                        with gr.Row():
                            add_dep_btn = gr.Button("➕ Add", size="sm")
                            remove_dep_btn = gr.Button("🗑️ Remove Selected", size="sm")
                        
                        dep_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            visible=False
                        )
                
                with gr.Row():
                    gr.Markdown("### 📄 Generate Requirements Files")
                    generate_req_btn = gr.Button("📄 Generate requirements.txt", size="sm")
                    generate_req_dev_btn = gr.Button("📄 Generate requirements.txt (with dev)", size="sm")
            
            # Virtual Environments tab with improved styling
            with gr.Tab("🧪 Virtual Environments") as venv_tab:
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 Virtual Environments")
                        
                        venv_table = gr.Dataframe(
                            headers=["🧪 Name", "🔄 Type", "🐍 Python Version", "📂 Path", "✅ Active"],
                            col_count=(5, "fixed"),
                            interactive=False,
                            elem_id="venv-table"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configure Environment")
                        
                        venv_name = gr.Textbox(
                            label="🧪 Environment Name",
                            placeholder="e.g., myenv"
                        )
                        
                        venv_type = gr.Radio(
                            choices=[
                                ("Python venv", "venv"),
                                ("Conda Environment", "conda"),
                                ("Poetry Environment", "poetry")
                            ],
                            label="🔄 Environment Type",
                            value="venv"
                        )
                        
                        python_version = gr.Textbox(
                            label="🐍 Python Version",
                            placeholder="e.g., 3.9",
                            value="3.9"
                        )
                        
                        with gr.Row():
                            configure_venv_btn = gr.Button("⚙️ Configure", size="sm")
                            activate_venv_btn = gr.Button("✅ Activate Selected", size="sm")
                        
                        venv_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            visible=False
                        )
            
            # Compilers tab with improved styling
            with gr.Tab("🔨 Compilers") as compilers_tab:
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 Compiler Configurations")
                        
                        compiler_table = gr.Dataframe(
                            headers=["🔨 Name", "🏷️ Version", "📂 Path"],
                            col_count=(3, "fixed"),
                            interactive=False,
                            elem_id="compiler-table"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Add Compiler")
                        
                        compiler_name = gr.Textbox(
                            label="🔨 Compiler Name",
                            placeholder="e.g., gcc"
                        )
                        
                        compiler_version = gr.Textbox(
                            label="🏷️ Version",
                            placeholder="e.g., 11.2.0"
                        )
                        
                        compiler_path = gr.Textbox(
                            label="📂 Path",
                            placeholder="e.g., /usr/bin/gcc"
                        )
                        
                        add_compiler_btn = gr.Button("➕ Add Compiler", size="sm")
                        
                        compiler_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            visible=False
                        )
            
            # Environment Variables tab with improved styling
            with gr.Tab("🔑 Environment Variables") as env_vars_tab:
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 Environment Variables")
                        
                        env_var_table = gr.Dataframe(
                            headers=["🔑 Name", "📝 Value", "ℹ️ Description", "🔒 Secret"],
                            col_count=(4, "fixed"),
                            interactive=False,
                            elem_id="env-var-table"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Add Environment Variable")
                        
                        env_var_name = gr.Textbox(
                            label="🔑 Variable Name",
                            placeholder="e.g., API_KEY"
                        )
                        
                        env_var_value = gr.Textbox(
                            label="📝 Value",
                            placeholder="e.g., your-api-key-here"
                        )
                        
                        env_var_desc = gr.Textbox(
                            label="ℹ️ Description",
                            placeholder="Optional description"
                        )
                        
                        is_secret = gr.Checkbox(
                            label="🔒 Is Secret",
                            value=False
                        )
                        
                        with gr.Row():
                            add_env_var_btn = gr.Button("➕ Add Variable", size="sm")
                            remove_env_var_btn = gr.Button("🗑️ Remove Selected", size="sm")
                        
                        env_var_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            visible=False
                        )
        
        # Function to load dependencies
        def load_dependencies() -> List[List[str]]:
            if not state.get("current_project"):
                return []
            
            project_path = state["current_project"].metadata.path
            env_data = EnvironmentController.load_environment_data(project_path)
            
            if not env_data:
                return []
            
            dependencies = []
            for dep in env_data.dependencies:
                dependencies.append([
                    dep.name,
                    dep.version,
                    "Development" if dep.is_dev_dependency else "Production"
                ])
            
            return dependencies
        
        # Function to load virtual environments
        def load_virtual_environments() -> List[List[str]]:
            if not state.get("current_project"):
                return []
            
            project_path = state["current_project"].metadata.path
            env_data = EnvironmentController.load_environment_data(project_path)
            
            if not env_data:
                return []
            
            active_env = env_data.active_environment
            environments = []
            
            for env in env_data.virtual_environments:
                environments.append([
                    env.name,
                    env.type,
                    env.python_version,
                    env.path,
                    "✅" if env.name == active_env else ""
                ])
            
            return environments
        
        # Function to load compilers
        def load_compilers() -> List[List[str]]:
            if not state.get("current_project"):
                return []
            
            project_path = state["current_project"].metadata.path
            env_data = EnvironmentController.load_environment_data(project_path)
            
            if not env_data:
                return []
            
            compilers = []
            for compiler in env_data.compilers:
                compilers.append([
                    compiler.name,
                    compiler.version,
                    compiler.path
                ])
            
            return compilers
        
        # Function to load environment variables
        def load_environment_variables() -> List[List[str]]:
            if not state.get("current_project"):
                return []
            
            project_path = state["current_project"].metadata.path
            env_data = EnvironmentController.load_environment_data(project_path)
            
            if not env_data:
                return []
            
            env_vars = []
            for var in env_data.environment_variables:
                # Mask secret values
                display_value = "********" if var.is_secret else var.value
                
                env_vars.append([
                    var.name,
                    display_value,
                    var.description or "",
                    "🔒 Yes" if var.is_secret else "No"
                ])
            
            return env_vars
        
        # Function to add a dependency
        def add_dependency(name: str, version: str, dep_type: str) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            if not name or not version:
                return "⚠️ Package name and version are required."
            
            project_path = state["current_project"].metadata.path
            is_dev = dep_type == "Development"
            
            success = EnvironmentController.add_dependency(project_path, name, version, is_dev)
            if success:
                return f"✅ Added {dep_type.lower()} dependency: {name}=={version}"
            else:
                return f"❌ Failed to add dependency: {name}"
        
        # Function to remove a dependency
        def remove_dependency(evt: gr.SelectData) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            selected_row = evt.index[0]
            selected_name = evt.value[0]
            
            project_path = state["current_project"].metadata.path
            success = EnvironmentController.remove_dependency(project_path, selected_name)
            
            if success:
                return f"✅ Removed dependency: {selected_name}"
            else:
                return f"❌ Failed to remove dependency: {selected_name}"
        
        # Function to configure a virtual environment
        def configure_virtual_environment(name: str, env_type: str, py_version: str) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            if not name:
                return "⚠️ Environment name is required."
            
            project_path = state["current_project"].metadata.path
            env = EnvironmentController.configure_virtual_env(project_path, name, env_type, py_version)
            
            if env:
                return f"✅ Configured {env_type} environment: {name} with Python {py_version}"
            else:
                return f"❌ Failed to configure environment: {name}"
        
        # Function to activate a virtual environment
        def activate_environment(evt: gr.SelectData) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            selected_row = evt.index[0]
            selected_env = evt.value[0]
            
            project_path = state["current_project"].metadata.path
            success = EnvironmentController.set_active_environment(project_path, selected_env)
            
            if success:
                return f"✅ Activated environment: {selected_env}"
            else:
                return f"❌ Failed to activate environment: {selected_env}"
        
        # Function to add a compiler
        def add_compiler(name: str, version: str, path: str) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            if not name or not path:
                return "⚠️ Compiler name and path are required."
            
            project_path = state["current_project"].metadata.path
            success = EnvironmentController.add_compiler(project_path, name, version, path)
            
            if success:
                return f"✅ Added compiler: {name} {version}"
            else:
                return f"❌ Failed to add compiler: {name}"
        
        # Function to add an environment variable
        def add_environment_variable(name: str, value: str, description: str, secret: bool) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            if not name:
                return "⚠️ Variable name is required."
            
            project_path = state["current_project"].metadata.path
            success = EnvironmentController.add_environment_variable(
                project_path, name, value, description, secret
            )
            
            if success:
                return f"✅ Added environment variable: {name}"
            else:
                return f"❌ Failed to add environment variable: {name}"
        
        # Function to remove an environment variable
        def remove_environment_variable(evt: gr.SelectData) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            selected_row = evt.index[0]
            selected_name = evt.value[0]
            
            project_path = state["current_project"].metadata.path
            success = EnvironmentController.remove_environment_variable(project_path, selected_name)
            
            if success:
                return f"✅ Removed environment variable: {selected_name}"
            else:
                return f"❌ Failed to remove environment variable: {selected_name}"
        
        # Function to generate requirements.txt
        def generate_requirements(include_dev: bool) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open."
            
            project_path = state["current_project"].metadata.path
            success, file_path = EnvironmentController.generate_requirements_file(project_path, include_dev)
            
            if success:
                return f"✅ Generated requirements.txt at {file_path}"
            else:
                return "❌ Failed to generate requirements.txt"
        
        # Connect event handlers for dependencies
        add_dep_btn.click(
            add_dependency,
            inputs=[dep_name, dep_version, dep_type],
            outputs=[dep_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [dep_status]
        ).then(
            load_dependencies,
            None,
            [dependencies_table]
        )
        
        remove_dep_btn.click(
            remove_dependency,
            inputs=None,
            outputs=[dep_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [dep_status]
        ).then(
            load_dependencies,
            None,
            [dependencies_table]
        )
        
        generate_req_btn.click(
            lambda: generate_requirements(False),
            inputs=None,
            outputs=[dep_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [dep_status]
        )
        
        generate_req_dev_btn.click(
            lambda: generate_requirements(True),
            inputs=None,
            outputs=[dep_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [dep_status]
        )
        
        # Connect event handlers for virtual environments
        configure_venv_btn.click(
            configure_virtual_environment,
            inputs=[venv_name, venv_type, python_version],
            outputs=[venv_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [venv_status]
        ).then(
            load_virtual_environments,
            None,
            [venv_table]
        )
        
        activate_venv_btn.click(
            activate_environment,
            inputs=None,
            outputs=[venv_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [venv_status]
        ).then(
            load_virtual_environments,
            None,
            [venv_table]
        )
        
        # Connect event handlers for compilers
        add_compiler_btn.click(
            add_compiler,
            inputs=[compiler_name, compiler_version, compiler_path],
            outputs=[compiler_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [compiler_status]
        ).then(
            load_compilers,
            None,
            [compiler_table]
        )
        
        # Connect event handlers for environment variables
        add_env_var_btn.click(
            add_environment_variable,
            inputs=[env_var_name, env_var_value, env_var_desc, is_secret],
            outputs=[env_var_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [env_var_status]
        ).then(
            load_environment_variables,
            None,
            [env_var_table]
        )
        
        remove_env_var_btn.click(
            remove_environment_variable,
            inputs=None,
            outputs=[env_var_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [env_var_status]
        ).then(
            load_environment_variables,
            None,
            [env_var_table]
        )
        
        # When the Environment tab is clicked, load all the data
        environment_tab.select(
            load_dependencies,
            None,
            [dependencies_table]
        ).then(
            load_virtual_environments,
            None,
            [venv_table]
        ).then(
            load_compilers,
            None,
            [compiler_table]
        ).then(
            load_environment_variables,
            None,
            [env_var_table]
        )
    
    return environment_tab
