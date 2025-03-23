"""
Improved implementation tracking view for the VibeCheck app.
"""

from typing import Dict, List, Optional, Tuple

import gradio as gr

from ..controllers.implementation_controller import ImplementationController
from ..controllers.security_controller import SecurityController
from ..controllers.architecture_controller import ArchitectureController
from ..models.implementation import FileStatus, ModuleStatus, ImplementationData
from ..models.security import SecurityVulnerability, SecurityAnalysis


def create_implementation_tab(state: Dict) -> gr.Tab:
    """
    Create the implementation tab for the VibeCheck app with improved visual design.

    Args:
        state: Application state dictionary

    Returns:
        The implementation tab component
    """
    with gr.Tab("💻 Implementation") as implementation_tab:
        gr.Markdown(
            """
            # 💻 Implementation Tracker
            
            Track your implementation progress against your architectural design.
            This tab shows the status of your code, implementation percentages,
            and security analysis results.
            """
        )
        
        with gr.Row():
            # Source tree navigator with improved styling
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Source Tree")
                
                file_tree = gr.Dataframe(
                    headers=["📁 Path", "🔄 Status", "📊 Implementation %"],
                    col_count=(3, "fixed"),
                    interactive=False,
                    height=400,
                    elem_id="source-tree"
                )
                
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            # Right panel with tabs
            with gr.Column(scale=2):
                with gr.Tabs() as implementation_tabs:
                    # Status/Progress tab with improved styling
                    with gr.Tab("📊 Status & Progress") as status_tab:
                        with gr.Row():
                            # Progress visualization
                            with gr.Column(scale=3):
                                progress_bar = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    label="Overall Implementation Progress",
                                    interactive=False,
                                    elem_id="progress-bar"
                                )
                            
                            with gr.Column(scale=1):
                                progress_percentage = gr.Textbox(
                                    value="0%",
                                    label="Percentage",
                                    interactive=False,
                                    elem_id="progress-percentage"
                                )
                        
                        module_progress = gr.Dataframe(
                            headers=["📦 Module", "📄 Files", "📊 Implementation %"],
                            col_count=(3, "fixed"),
                            interactive=False,
                            elem_id="module-progress"
                        )
                        
                        last_analyzed = gr.Textbox(
                            label="🕒 Last Analyzed",
                            interactive=False
                        )
                    
                    # Security tab with improved styling
                    with gr.Tab("🔒 Security") as security_tab:
                        # Security vulnerabilities table
                        gr.Markdown("### 🛡️ Security Vulnerabilities")
                        vulnerabilities = gr.Dataframe(
                            headers=["⚠️ Severity", "📝 Description", "📍 Location"],
                            col_count=(3, "fixed"),
                            interactive=False,
                            elem_id="vulnerabilities"
                        )
                        
                        # LLM insights with improved styling
                        gr.Markdown("### 💡 Security Insights")
                        llm_insights = gr.Markdown(
                            elem_id="security-insights"
                        )
                        
                        analyze_security_btn = gr.Button("🔍 Analyze Security", size="sm")
                    
                    # Source tab with improved styling
                    with gr.Tab("📝 Source") as source_tab:
                        selected_file = gr.Textbox(
                            label="📄 Selected File",
                            interactive=False
                        )
                        
                        file_content = gr.TextArea(
                            lines=25,
                            label="💻 Source Code",
                            interactive=True,
                            elem_id="source-editor"
                        )
                        
                        with gr.Row():
                            save_file_btn = gr.Button("💾 Save File", size="sm")
                            file_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                visible=False
                            )
        
        # Function to load the implementation data
        def load_implementation_data() -> Tuple[List[List[str]], float, str, List[List[str]], str]:
            if not state.get("current_project"):
                return [], 0, "0%", [], "No project is currently open"
            
            project_path = state["current_project"].metadata.path
            
            # Get architectural diagrams for implementation percentage calculation
            diagrams = ArchitectureController.get_all_diagrams(project_path)
            
            # Analyze implementation
            impl_data = ImplementationController.analyze_implementation(project_path, diagrams)
            
            # Prepare file tree data
            file_tree_data = []
            for module_name, module in impl_data.modules.items():
                for file in module.files:
                    # Add emoji based on status
                    status_emoji = "🆕" if file.status == "added" else "🔄" if file.status == "modified" else "✅"
                    status_display = f"{status_emoji} {file.status}"
                    
                    file_tree_data.append([
                        file.path,
                        status_display,
                        f"{file.implementation_percentage:.1f}%"
                    ])
            
            # Sort by path
            file_tree_data.sort(key=lambda x: x[0])
            
            # Prepare module progress data
            module_progress_data = []
            for module_name, module in impl_data.modules.items():
                module_progress_data.append([
                    module_name,
                    len(module.files),
                    f"{module.implementation_percentage:.1f}%"
                ])
            
            # Sort by module name
            module_progress_data.sort(key=lambda x: x[0])
            
            # Format last analyzed time
            last_analyzed_time = impl_data.analyzed_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(impl_data, 'analyzed_at') else "Unknown"
            
            return (
                file_tree_data,
                impl_data.overall_percentage,
                f"{impl_data.overall_percentage:.1f}%",
                module_progress_data,
                f"Last analyzed: {last_analyzed_time}"
            )
        
        # Function to handle file selection
        def handle_file_selection(evt: gr.SelectData) -> Tuple[str, str]:
            if not state.get("current_project"):
                return "", ""
            
            project_path = state["current_project"].metadata.path
            selected_row = evt.index[0]
            selected_path = evt.value[0]
            
            try:
                file_path = os.path.join(project_path, selected_path)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return selected_path, content
            except Exception as e:
                return selected_path, f"Error reading file: {str(e)}"
        
        # Function to save a file
        def save_file(selected_file: str, content: str) -> str:
            if not state.get("current_project") or not selected_file:
                return "⚠️ No file selected or no project open."
            
            project_path = state["current_project"].metadata.path
            file_path = os.path.join(project_path, selected_file)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"✅ File {selected_file} saved successfully."
            except Exception as e:
                return f"❌ Error saving file: {str(e)}"
        
        # Function to analyze security
        def analyze_security(selected_file: str) -> Tuple[List[List[str]], str]:
            if not state.get("current_project"):
                return [], "⚠️ No project is currently open."
            
            project_path = state["current_project"].metadata.path
            target_path = selected_file if selected_file else None
            
            security_results = SecurityController.run_security_analysis(project_path, target_path)
            
            # Combine all vulnerabilities for display
            all_vulnerabilities = []
            llm_insights_text = ""
            
            for result in security_results:
                for vuln in result.vulnerabilities:
                    # Add emoji based on severity
                    severity_emoji = "🔴" if vuln.severity == "critical" else "🟠" if vuln.severity == "high" else "🟡" if vuln.severity == "medium" else "🟢"
                    severity_display = f"{severity_emoji} {vuln.severity}"
                    
                    all_vulnerabilities.append([
                        severity_display,
                        vuln.description,
                        vuln.location
                    ])
                
                # Use the insights from the most relevant result
                if target_path and target_path in result.path:
                    llm_insights_text = result.llm_insights
                elif not target_path and not llm_insights_text:
                    llm_insights_text = result.llm_insights
            
            # Sort by severity (critical, high, medium, low, info)
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            all_vulnerabilities.sort(key=lambda x: severity_order.get(x[0].lower().split()[-1], 99))
            
            # Add header to insights if vulnerabilities were found
            if all_vulnerabilities:
                final_insights = f"## 🔍 Security Analysis Results\n\n{llm_insights_text}"
            else:
                final_insights = "## ✅ No security vulnerabilities detected\n\nThe code appears to be secure based on the current analysis."
            
            return all_vulnerabilities, final_insights
        
        # Connect event handlers
        refresh_btn.click(
            load_implementation_data,
            inputs=None,
            outputs=[file_tree, progress_bar, progress_percentage, module_progress, last_analyzed]
        )
        
        file_tree.select(
            handle_file_selection,
            inputs=None,
            outputs=[selected_file, file_content]
        )
        
        save_file_btn.click(
            save_file,
            inputs=[selected_file, file_content],
            outputs=[file_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [file_status]
        )
        
        analyze_security_btn.click(
            analyze_security,
            inputs=[selected_file],
            outputs=[vulnerabilities, llm_insights]
        )
        
        # When the Implementation tab is clicked, load the implementation data
        implementation_tab.select(
            load_implementation_data,
            inputs=None,
            outputs=[file_tree, progress_bar, progress_percentage, module_progress, last_analyzed]
        )
    
    return implementation_tab


# Add missing import
import os
