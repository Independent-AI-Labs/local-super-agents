"""
Improved architecture designer view for the VibeCheck app.
"""

from typing import Dict, Optional, Tuple, List

import gradio as gr

from vibecheck.controllers.architecture_controller import ArchitectureController
from vibecheck.models.architecture import ArchitecturalDocument, ArchitecturalDiagram


def create_architecture_tab(state: Dict) -> gr.Tab:
    """
    Create the architecture tab for the VibeCheck app with improved visual design.

    Args:
        state: Application state dictionary

    Returns:
        The architecture tab component
    """
    with gr.Tab("🏗️ Architecture") as architecture_tab:
        gr.Markdown(
            """
            # 🏗️ Architecture Designer
            
            Define your software architecture before implementation. This helps enforce
            proper software engineering practices and provides a clear blueprint for 
            your development.
            """
        )
        
        with gr.Row():
            # Architecture document editor with improved styling
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Architectural Document")
                
                doc_editor = gr.TextArea(
                    lines=25,
                    label="Document Content",
                    placeholder="# Your Architecture Document\n\n## System Overview\n\nDescribe your system here...",
                    interactive=True,
                    elem_id="architecture-doc-editor"
                )
                
                with gr.Row():
                    save_doc_btn = gr.Button("💾 Save Document", size="sm")
                    generate_diagrams_btn = gr.Button("📊 Generate Diagrams", size="sm")
                    analyze_doc_btn = gr.Button("🔍 Analyze Document", size="sm")
                
                doc_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False
                )
            
            # Architecture diagram viewer with improved styling
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Architectural Diagrams")
                
                diagram_type = gr.Radio(
                    choices=[
                        ("Module Diagram", "module"),
                        ("Data Flow Diagram", "dataflow"),
                        ("Security Diagram", "security"),
                        ("Mermaid Diagram", "mermaid")
                    ],
                    label="📊 Diagram Type",
                    value="module",
                    elem_id="diagram-type-selector"
                )
                
                diagram_viewer = gr.HTML(
                    label="Diagram",
                    elem_id="diagram-viewer"
                )
        
        # Analysis panel with improved styling
        with gr.Accordion("🔍 Architecture Analysis", open=False, visible=False) as analysis_header:
            with gr.Row() as analysis_row:
                analysis_result = gr.Markdown(
                    value="Click 'Analyze Document' to get insights about your architecture."
                )
            
            with gr.Row() as components_row:    
                # Tables for components and relationships
                with gr.Column(scale=1):
                    components_table = gr.Dataframe(
                        headers=["📦 Component", "📝 Description"],
                        col_count=(2, "fixed"),
                        label="Components",
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    relationships_table = gr.Dataframe(
                        headers=["📦 Source", "🔄 Relationship", "📦 Target"],
                        col_count=(3, "fixed"),
                        label="Relationships",
                        interactive=False
                    )
        
        # Function to save the architecture document
        def save_document(content: str) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open. Please open or create a project first."
            
            project_path = state["current_project"].metadata.path
            
            success = ArchitectureController.save_document(project_path, content)
            if success:
                return "✅ Architecture document saved successfully."
            else:
                return "❌ Failed to save architecture document."
        
        # Function to generate diagrams from the document
        def generate_diagrams(content: str) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open. Please open or create a project first."
            
            project_path = state["current_project"].metadata.path
            
            diagrams = ArchitectureController.generate_diagrams(project_path, content)
            if diagrams:
                return f"✅ Generated {len(diagrams)} diagrams successfully."
            else:
                return "❌ Failed to generate diagrams."
        
        # Function to display a diagram based on its type
        def display_diagram(diagram_type: str) -> str:
            if not state.get("current_project"):
                return "<p>⚠️ No project is currently open. Please open or create a project first.</p>"
            
            project_path = state["current_project"].metadata.path
            
            if diagram_type == "mermaid":
                # Get Mermaid diagram
                mermaid_code = ArchitectureController.get_mermaid_diagram(project_path)
                return f"""
                <div class="mermaid">
                {mermaid_code}
                </div>
                <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                <script>mermaid.initialize({{startOnLoad:true}});</script>
                """
            
            # Get regular SVG diagram
            diagram = ArchitectureController.get_diagram(project_path, diagram_type)
            if diagram:
                return f'<div style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: white;">{diagram.content}</div>'
            else:
                return '<p>⚠️ No diagram available. Please generate diagrams first.</p>'
        
        # Function to load the architecture document from the project
        def load_document() -> str:
            if not state.get("current_project"):
                return "# 📂 No Project Open\n\nPlease open or create a project first."
            
            project_path = state["current_project"].metadata.path
            
            document = ArchitectureController.load_document(project_path)
            if document:
                return document.content
            else:
                return "# 🏗️ New Architecture Document\n\n## System Overview\n\nDescribe your system here..."
        
        # Function to analyze the architecture document
        def analyze_document(content: str) -> str:
            if not state.get("current_project"):
                return "⚠️ No project is currently open. Please open or create a project first."
            
            if not content.strip():
                return "⚠️ The architecture document is empty. Please add content before analyzing."
            
            project_path = state["current_project"].metadata.path
            
            # Save the document first to ensure we're analyzing the latest version
            ArchitectureController.save_document(project_path, content)
            
            # Get the analysis
            return ArchitectureController.analyze_architecture_document(project_path, content)
            
        # Function to get components and relationships
        def get_components_and_relationships() -> Tuple[List[List[str]], List[List[str]]]:
            if not state.get("current_project"):
                return [], []
            
            project_path = state["current_project"].metadata.path
            
            # Get components and relationships
            data = ArchitectureController.get_components_and_relationships(project_path)
            components = data["components"]
            relationships = data["relationships"]
            
            # Format for the UI
            components_data = [[c["name"], c["description"]] for c in components]
            relationships_data = [[r["source"], r["type"].replace("_", " "), r["target"]] for r in relationships]
            
            return components_data, relationships_data
            
        # Function to show analysis UI elements
        def show_analysis_ui() -> Tuple[gr.update, gr.update, gr.update]:
            return gr.update(visible=True, open=True), gr.update(visible=True), gr.update(visible=True)
            
        # Connect event handlers
        save_doc_btn.click(
            save_document,
            inputs=[doc_editor],
            outputs=[doc_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [doc_status]
        )
        
        generate_diagrams_btn.click(
            generate_diagrams,
            inputs=[doc_editor],
            outputs=[doc_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [doc_status]
        ).then(
            display_diagram,
            inputs=[diagram_type],
            outputs=[diagram_viewer]
        )
        
        analyze_doc_btn.click(
            analyze_document,
            inputs=[doc_editor],
            outputs=[analysis_result]
        ).then(
            show_analysis_ui,
            None,
            [analysis_header, analysis_row, components_row]
        ).then(
            get_components_and_relationships,
            None,
            [components_table, relationships_table]
        )
        
        diagram_type.change(
            display_diagram,
            inputs=[diagram_type],
            outputs=[diagram_viewer]
        )
        
        # When the Architecture tab is clicked, load the document and diagrams
        architecture_tab.select(
            load_document,
            inputs=None,
            outputs=[doc_editor]
        ).then(
            display_diagram,
            inputs=[diagram_type],
            outputs=[diagram_viewer]
        )
    
    return architecture_tab
