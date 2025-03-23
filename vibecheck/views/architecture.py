"""
Complete fixed architecture.py with all fixes implemented and all event handlers properly configured.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import json
from datetime import datetime

import gradio as gr

from vibecheck.controllers.architecture_controller import ArchitectureController
from vibecheck.models.architecture import ArchitecturalDocument, ArchitecturalDiagram
from vibecheck.utils.architecture_utils import ArchitectureDocumentManager, FileChangeTracker
from vibecheck.utils.cache_utils import AnalysisCache
from vibecheck.utils.file_utils import ensure_directory, read_file, write_file
from vibecheck.utils.architecture_ui_utils import (
    get_document_list, get_document_content, get_document_content_safe,
    save_document_content, update_scope, has_document_changed, display_diagram,
    create_default_document_content, clean_document_selection, format_scope_text
)
from vibecheck import config
from vibecheck.constants.architecture_constants import *


def create_architecture_tab(state: Dict) -> gr.Tab:
    """
    Create the architecture tab for the VibeCheck app with improved visual design.

    Args:
        state: Application state dictionary

    Returns:
        The architecture tab component
    """
    # Initialize document state
    document_state = {
        "current_document": None,
        "editing": False,
        "scope": []
    }
    
    with gr.Tab(TAB_TITLE) as architecture_tab:
        gr.Markdown(TAB_DESCRIPTION)
        
        # Scope display at the top
        scope_display = gr.Markdown(
            NO_SCOPE, 
            elem_id="scope-display"
        )
        
        with gr.Row():
            # Left panel: Document list and actions
            with gr.Column(scale=1):
                gr.Markdown(DOCUMENTS_HEADER)
                
                # Document list with checkboxes
                document_list = gr.CheckboxGroup(
                    choices=[],
                    value=[],  # Initialize with empty selection
                    label=DOCUMENT_LABEL,
                    info=DOCUMENT_INFO,
                    elem_id="document-list",
                    interactive=True,  # Ensure checkboxes are interactive
                    visible=True,      # Explicitly make visible
                    container=True,    # Ensure container is rendered
                    scale=1,           # Full width
                    min_width=200      # Minimum width to ensure visibility
                )
                
                # Document details
                document_details = gr.Dataframe(
                    headers=DOCUMENT_HEADERS,
                    col_count=(3, "fixed"),
                    interactive=False,
                    elem_id="document-details",
                    label="Document Details",
                    height=300,
                    value=[]  # Initialize with empty dataframe
                )
                
                with gr.Row():
                    # Document actions
                    new_doc_btn = gr.Button(NEW_DOC_BTN, size="sm")
                    # Change to use UploadButton component instead of file_upload
                    upload_doc_btn = gr.UploadButton(UPLOAD_DOC_BTN, size="sm", file_types=[".json", ".yaml", ".xml", ".txt", ".md"])
                
                with gr.Row():
                    delete_selected_btn = gr.Button(DELETE_BTN, size="sm", variant="secondary")
                
                # Upload info
                upload_info = gr.Markdown(
                    UPLOAD_INFO,
                    elem_id="upload-info"
                )
                
                # Hidden status message (invisible but still available for event handling)
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                    value=""  # Initialize with empty string
                )
                
                # Current document name (hidden)
                current_doc_name = gr.Textbox(
                    label="Current Document",
                    interactive=False,
                    visible=False,
                    value=""  # Initialize with empty string
                )
            
            # Right panel: Document content, diagrams, and analysis
            with gr.Column(scale=2):
                # Tabbed interface for document content, diagrams, and analysis
                with gr.Tabs() as document_tabs:
                    # Document content tab
                    with gr.Tab(DOCUMENT_TAB) as content_tab:
                        with gr.Row():
                            edit_btn = gr.Button(EDIT_BTN, size="sm", visible=False)
                            save_btn = gr.Button(SAVE_BTN, size="sm", visible=False)
                            cancel_btn = gr.Button(CANCEL_BTN, size="sm", visible=False)
                            analyze_btn = gr.Button(ANALYZE_BTN, size="sm", visible=False)
                            
                        document_heading = gr.Markdown(
                            NO_DOCUMENT_SELECTED,
                            elem_id="document-heading"
                        )
                        
                        # Warning for unsaved changes
                        change_warning = gr.Markdown(
                            DOCUMENT_CHANGED_WARNING,
                            visible=False,
                            elem_id="change-warning"
                        )
                        
                        # Document display/edit area - INCREASED HEIGHT BY 50%
                        document_display = gr.Markdown(
                            elem_id="document-display",
                            value=NO_DOCUMENT_CONTENT,
                            height=600  # Increased from 400 to 600 (50% taller)
                        )
                        
                        document_editor = gr.TextArea(
                            lines=30,  # Increased from 20 to 30 (50% more lines)
                            max_lines=38,  # Increased from 25 to 38 (50% more max lines)
                            label="Edit Document",
                            elem_id="document-editor",
                            visible=False,
                            value=""  # Initialize with empty string
                        )
                    
                    # Diagrams tab
                    with gr.Tab(DIAGRAMS_TAB) as diagrams_tab:
                        with gr.Row():
                            # Move the button to the top
                            generate_diagrams_btn = gr.Button(GENERATE_DIAGRAMS_BTN, size="sm", visible=False)
                        
                        # Diagram type selection
                        diagram_type = gr.Radio(
                            choices=DIAGRAM_TYPES,
                            label=DIAGRAM_TYPE_LABEL,
                            value="module",
                            elem_id="diagram-type-selector"
                        )
                        
                        # Progress/status indicator for diagrams - hidden but kept for event handling
                        diagram_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            visible=False,
                            value=""  # Initialize with empty string
                        )
                        
                        # Diagram display
                        diagram_viewer = gr.HTML(
                            label="Diagram",
                            elem_id="diagram-viewer",
                            value=NO_DIAGRAM_SELECTED
                        )
                    
                    # Analysis tab
                    with gr.Tab(ANALYSIS_TAB) as analysis_tab:
                        with gr.Row():
                            # Add Analyze button to analysis tab
                            analyze_tab_btn = gr.Button(RUN_CRITICAL_ANALYSIS_BTN, size="sm", visible=False)
                        
                        # Analysis results - INCREASED HEIGHT BY 50%
                        analysis_result = gr.Markdown(
                            NO_ANALYSIS_SELECTED,
                            elem_id="analysis-result",
                            height=450  # Increased from 300 to 450 (50% taller)
                        )
                        
                        with gr.Row():
                            # Tables for components and relationships
                            with gr.Column(scale=1):
                                components_table = gr.Dataframe(
                                    headers=COMPONENTS_TABLE_HEADERS,
                                    col_count=(2, "fixed"),
                                    label="Components",
                                    interactive=False,
                                    value=[],  # Initialize with empty dataframe
                                    height=450  # Increased from 300 to 450 (50% taller)
                                )
                            
                            with gr.Column(scale=1):
                                relationships_table = gr.Dataframe(
                                    headers=RELATIONSHIPS_TABLE_HEADERS,
                                    col_count=(3, "fixed"),
                                    label="Relationships",
                                    interactive=False,
                                    value=[],  # Initialize with empty dataframe
                                    height=450  # Increased from 300 to 450 (50% taller)
                                )
        
        # ----- MAIN FUNCTIONS -----
        
        def load_documents():
            """Load document list, details and scope."""
            if not state.get("current_project"):
                return [], [], [], NO_SCOPE
            
            try:
                project_path = state["current_project"].metadata.path
                
                # Ensure architecture directories exist
                docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
                ensure_directory(docs_dir)
                
                # Get document list and scope
                doc_names, doc_rows, scope = get_document_list(project_path)
                
                # Store scope in state
                document_state["scope"] = scope
                
                # Make sure scope is a list of strings
                if not isinstance(scope, list):
                    scope = []
                
                # Ensure scope only contains document names as strings
                clean_scope = clean_document_selection(scope)
                
                # Update scope text
                scope_text = format_scope_text(clean_scope)
                
                # Print for debugging
                print(f"Loaded {len(doc_names)} documents: {doc_names}")
                print(f"Scope: {clean_scope}")
                
                # Return document names for the checkbox group, cleaned scope, and rows for the details table
                return doc_names, clean_scope, doc_rows, scope_text
            except Exception as e:
                print(ERROR_LOADING_DOCUMENTS.format(error=e))
                return [], [], [], SCOPE_ERROR
        
        def ensure_scope_ui_update():
            """Force reload the document list and scope to ensure UI is updated."""
            if not state.get("current_project"):
                return [], [], [], NO_SCOPE
            
            project_path = state["current_project"].metadata.path
            
            # Get fresh document list and scope
            doc_names, doc_rows, scope = get_document_list(project_path)
            
            # Clean scope to ensure proper format
            clean_scope = clean_document_selection(scope)
            
            # Update scope text
            scope_text = format_scope_text(clean_scope)
            
            # Return values for UI update
            return doc_names, clean_scope, doc_rows, scope_text
        
        def handle_document_selection(evt: gr.SelectData):
            """Handle document selection from the list."""
            if not state.get("current_project"):
                return "", NO_DOCUMENT_SELECTED, NO_DOCUMENT_CONTENT, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            # Get the selected document name
            doc_name = evt.value
            if not doc_name:
                return "", NO_DOCUMENT_SELECTED, NO_DOCUMENT_CONTENT, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            project_path = state["current_project"].metadata.path
            
            # Update current document in state
            document_state["current_document"] = doc_name
            document_state["editing"] = False
            
            # Get document content
            content = get_document_content(project_path, doc_name)
            
            # Check if document has changed
            has_changed = has_document_changed(project_path, doc_name)
            
            # Update heading
            heading = f"### 📝 {doc_name}"
            
            return (
                doc_name,                         # Current document name
                heading,                          # Document heading
                content,                          # Document content
                gr.update(visible=has_changed),   # Show change warning
                gr.update(visible=True),          # Show edit button
                gr.update(visible=False),         # Hide save button
                gr.update(visible=False),         # Hide cancel button
                gr.update(visible=True)           # Show analyze button
            )
        
        def update_scope_selection(selected_docs):
            """Update scope when selection changes."""
            if not state.get("current_project"):
                return NO_SCOPE
            
            # Clean the selection
            clean_selected_docs = clean_document_selection(selected_docs)
            
            project_path = state["current_project"].metadata.path
            
            # Update state and save to file
            document_state["scope"] = clean_selected_docs
            update_scope(project_path, clean_selected_docs)
            
            # Update scope display
            scope_text = format_scope_text(clean_selected_docs)
            
            return scope_text
        
        def enable_editing():
            """Enable document editing mode."""
            return (
                gr.update(visible=False),  # Hide document display
                gr.update(visible=True),   # Show document editor
                gr.update(visible=False),  # Hide edit button
                gr.update(visible=True),   # Show save button
                gr.update(visible=True),   # Show cancel button
                gr.update(visible=False)   # Hide analyze button
            )
        
        def save_document(doc_name, content):
            """Save document and update UI."""
            if not state.get("current_project") or not doc_name:
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            project_path = state["current_project"].metadata.path
            
            # Save the document
            success = save_document_content(project_path, doc_name, content)
            
            # Update state
            document_state["editing"] = False
            
            # Document has definitely changed since we just saved it
            return (
                gr.update(visible=False),    # Hide status
                gr.update(visible=True),     # Show document display
                gr.update(visible=False),    # Hide document editor
                gr.update(visible=True),     # Show edit button
                gr.update(visible=False),    # Hide save button
                gr.update(visible=False),    # Hide cancel button
                gr.update(visible=True),     # Show change warning
                gr.update(visible=True)      # Show analyze button
            )
        
        def cancel_editing(doc_name):
            """Cancel editing and restore original content."""
            document_state["editing"] = False
            
            if not state.get("current_project") or not doc_name:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=True)
            
            project_path = state["current_project"].metadata.path
            
            # Reload document content
            content = get_document_content(project_path, doc_name)
            
            return (
                gr.update(visible=True),   # Show document display
                gr.update(visible=False),  # Hide document editor
                gr.update(visible=True),   # Show edit button
                gr.update(visible=False),  # Hide save button
                gr.update(visible=False),  # Hide cancel button
                content,                   # Original content for the editor
                gr.update(visible=True)    # Show analyze button
            )

        def create_new_document():
            """Create a new document with unique name and update scope selection properly."""
            if not state.get("current_project"):
                return [], [], []

            project_path = state["current_project"].metadata.path

            # Generate a unique document name
            base_name = "new_architecture"
            doc_name = base_name
            count = 1

            # Get existing document names
            doc_names, _, _ = get_document_list(project_path)

            while doc_name in doc_names:
                doc_name = f"{base_name}_{count}"
                count += 1

            # Create default content
            content = create_default_document_content(doc_name)

            # Save the document
            success = save_document_content(project_path, doc_name, content)

            # Get updated document list
            doc_names, doc_rows, scope = get_document_list(project_path)

            # Add the new document to the scope
            if success and doc_name not in scope:
                scope.append(doc_name)
                update_scope(project_path, scope)

            return doc_names, scope, doc_rows
        
        def process_uploaded_file(file_info):
            """Process uploaded document file."""
            if not state.get("current_project"):
                return [], [], []
                
            if file_info is None:
                return [], [], []
            
            # For UploadButton, file_info is always a file object
            file_path = file_info.name if hasattr(file_info, 'name') else None
                
            if not file_path or not os.path.exists(file_path):
                return [], [], []
            
            project_path = state["current_project"].metadata.path
            
            # Extract filename without extension
            filename = os.path.basename(file_path)
            doc_name = os.path.splitext(filename)[0]
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(ERROR_READING_FILE.format(error=e))
                return [], [], []
            
            # Save as document
            save_document_content(project_path, doc_name, content)
            
            # Get updated document list
            doc_names, doc_rows, scope = get_document_list(project_path)
            
            return doc_names, scope, doc_rows

        def delete_selected_documents(selected_docs):
            """Delete selected documents in a non-blocking way."""
            if not state.get("current_project"):
                return [], [], []

            # Clean the selection
            clean_selected_docs = clean_document_selection(selected_docs)

            if not clean_selected_docs:
                return [], [], []

            project_path = state["current_project"].metadata.path

            # Get current scope before deletion
            scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
            current_scope = []
            if os.path.exists(scope_path):
                try:
                    with open(scope_path, 'r') as f:
                        scope_data = json.load(f)
                        current_scope = scope_data.get('documents', [])
                except Exception as e:
                    print(f"Error loading scope: {e}")

            # Simplified deletion process to prevent blocking
            for doc_name in clean_selected_docs:
                try:
                    # Delete the document files without complex error handling to reduce blocking
                    json_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.json")
                    md_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.md")

                    # Simple file deletion with minimal error handling
                    if os.path.exists(json_path):
                        os.remove(json_path)

                    if os.path.exists(md_path):
                        os.remove(md_path)

                    # Delete any associated diagram files (simplified)
                    diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
                    if os.path.exists(diagrams_dir):
                        for filename in os.listdir(diagrams_dir):
                            if filename.startswith(f"{doc_name}_"):
                                try:
                                    os.remove(os.path.join(diagrams_dir, filename))
                                except:
                                    pass

                    # Delete analysis file (simplified)
                    analysis_path = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR, f"{doc_name}.json")
                    if os.path.exists(analysis_path):
                        try:
                            os.remove(analysis_path)
                        except:
                            pass
                except Exception as e:
                    print(ERROR_DELETING_DOCUMENT.format(path=doc_name, error=e))

            # Get updated document list
            doc_names, doc_rows, _ = get_document_list(project_path)

            # Update scope to remove deleted documents
            updated_scope = [doc for doc in current_scope if doc in doc_names]
            update_scope(project_path, updated_scope)

            return doc_names, updated_scope, doc_rows

        def generate_diagrams(doc_name):
            """Generate diagrams for a document, ensuring re-generation on repeated clicks."""
            if not state.get("current_project") or not doc_name:
                return gr.update(visible=False)

            project_path = state["current_project"].metadata.path

            # Get document content
            content = get_document_content(project_path, doc_name)
            if not content:
                return gr.update(visible=False)

            # Force diagram regeneration by clearing cache first
            cache_key = f"architecture_{doc_name}"
            AnalysisCache.invalidate_cache(project_path, cache_key)

            # Generate diagrams via controller
            try:
                diagrams = ArchitectureController.generate_diagrams(project_path, doc_name, content)

                if diagrams:
                    print(f"Successfully generated {len(diagrams)} diagrams for {doc_name}")
                    for diagram_type, diagram in diagrams.items():
                        print(f"Generated {diagram_type} diagram of length: {len(diagram.content) if diagram and hasattr(diagram, 'content') else 0}")

                return gr.update(visible=False)
            except Exception as e:
                print(ERROR_GENERATING_DIAGRAMS.format(error=e))
                import traceback
                traceback.print_exc()
                return gr.update(visible=False)

        def analyze_document(doc_name):
            """Analyze architecture document with enhanced critical focus, without generating diagrams."""
            if not state.get("current_project") or not doc_name:
                # Set loading state
                return EMPTY_DOCUMENT_ERROR, gr.update(visible=False), [], []
            
            project_path = state["current_project"].metadata.path
            
            # Get document content
            content = get_document_content(project_path, doc_name)
            if not content or content.strip() == "":
                return EMPTY_DOCUMENT_ERROR, gr.update(visible=False), [], []
            
            # Force re-analysis by clearing cache first
            cache_key = f"architecture_{doc_name}"
            AnalysisCache.invalidate_cache(project_path, f"{cache_key}_analysis")
            
            # Analyze document via controller
            try:
                # Add critical analysis focus by appending to content
                enhanced_content = content + ANALYSIS_FOCUS_APPENDIX
                
                analysis = ArchitectureController.analyze_architecture_document(project_path, doc_name, enhanced_content)
                if not analysis:
                    return ANALYSIS_FAILED, gr.update(visible=False), [], []
            except Exception as e:
                return ERROR_ANALYZING_DOCUMENT.format(error=e), gr.update(visible=False), [], []
            
            # Get components and relationships
            try:
                data = ArchitectureController.get_components_and_relationships(project_path, doc_name)
                
                components = data.get("components", [])
                relationships = data.get("relationships", [])
                
                # Format for the UI
                components_data = [[c.get("name", ""), c.get("description", "")] for c in components]
                relationships_data = [[r.get("source", ""), r.get("type", "").replace("_", " "), r.get("target", "")] for r in relationships]
            except Exception as e:
                print(f"Error getting components and relationships: {e}")
                components_data = []
                relationships_data = []
            
            # Make sure we return a string, not a boolean
            if isinstance(analysis, bool):
                analysis = "Analysis completed" if analysis else "Analysis failed"
                
            return analysis, gr.update(visible=False), components_data, relationships_data
        
        def reset_document_view():
            """Reset the document view when no document is selected."""
            return (
                "",                          # Clear current document
                NO_DOCUMENT_SELECTED,        # Reset heading
                NO_DOCUMENT_CONTENT,         # Reset content
                gr.update(visible=False),    # Hide change warning
                gr.update(visible=False),    # Hide edit button
                gr.update(visible=False),    # Hide save button
                gr.update(visible=False),    # Hide cancel button
                gr.update(visible=False),    # Hide analyze button
                gr.update(visible=False),    # Hide generate diagrams button
                gr.update(visible=False)     # Hide analyze button in analysis tab
            )
        
        def update_diagram_view(doc_name, diagram_type):
            """Update diagram view with selected document and diagram type."""
            if not doc_name:
                return NO_DOCUMENT_FIRST
                
            if not state.get("current_project"):
                return NO_PROJECT_OPEN
                
            project_path = state["current_project"].metadata.path
            return display_diagram(project_path, doc_name, diagram_type)
            
        # ----- EVENT HANDLERS -----
        
        # Document list selection for scope
        document_list.change(
            update_scope_selection,
            inputs=[document_list],
            outputs=[scope_display]
        )
        
        # Document selection from dataframe
        def on_document_select(evt: gr.SelectData):
            """Handle document selection from the details dataframe, refreshing all tabs."""
            if not state.get("current_project"):
                return reset_document_view() + (NO_DOCUMENT_FIRST, NO_ANALYSIS_SELECTED, [], [])

            # Get the selected document name directly from the row data
            try:
                # Try to get document name from the row clicked in dataframe
                doc_name = evt.row_value[0]

                print(f"Selected document: {doc_name}")

                if not doc_name:
                    return reset_document_view() + (NO_DOCUMENT_FIRST, NO_ANALYSIS_SELECTED, [], [])

                project_path = state["current_project"].metadata.path

                # Update current document in state
                document_state["current_document"] = doc_name
                document_state["editing"] = False

                # Get document content
                content = get_document_content(project_path, doc_name)

                print(f"Document content length: {len(content) if content else 0}")

                # Check if document has changed
                has_changed = has_document_changed(project_path, doc_name)

                # Update heading
                heading = f"### 📝 {doc_name}"

                # Generate diagram HTML for selected diagram type
                diagram_html = display_diagram(project_path, doc_name, diagram_type.value)

                # Get analysis if it exists
                analysis_content = NO_ANALYSIS_SELECTED
                components_data = []
                relationships_data = []

                # Load existing analysis if available
                analysis_path = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR, f"{doc_name}.json")
                if os.path.exists(analysis_path):
                    try:
                        with open(analysis_path, 'r') as f:
                            analysis_data = json.load(f)
                            if 'content' in analysis_data:
                                analysis_content = analysis_data['content']

                        # Also try to load components and relationships
                        data = ArchitectureController.get_components_and_relationships(project_path, doc_name)
                        components = data.get("components", [])
                        relationships = data.get("relationships", [])

                        # Format for the UI
                        components_data = [[c.get("name", ""), c.get("description", "")] for c in components]
                        relationships_data = [[r.get("source", ""), r.get("type", "").replace("_", " "), r.get("target", "")] for r in relationships]
                    except Exception as e:
                        print(ERROR_LOADING_ANALYSIS.format(error=e))

                return (
                    doc_name,                       # Current document name
                    heading,                        # Document heading
                    content,                        # Document content
                    gr.update(visible=has_changed), # Show change warning based on status
                    gr.update(visible=True),        # Show edit button
                    gr.update(visible=False),       # Hide save button
                    gr.update(visible=False),       # Hide cancel button
                    gr.update(visible=True),        # Show analyze button
                    gr.update(visible=True),        # Show generate diagrams button
                    diagram_html,                   # Update diagram viewer
                    gr.update(visible=True),        # Show analyze button in analysis tab
                    analysis_content,               # Analysis content
                    components_data,                # Components table data
                    relationships_data              # Relationships table data
                )
            except Exception as e:
                print(ERROR_DOCUMENT_SELECTION.format(error=e))
                import traceback
                traceback.print_exc()
                return reset_document_view() + (NO_DOCUMENT_FIRST, NO_ANALYSIS_SELECTED, [], [])
            
        # Use proper select event handling
        document_details.select(
            on_document_select,  # Function that receives the event 
            outputs=[
                current_doc_name,
                document_heading,
                document_display,
                change_warning,
                edit_btn,
                save_btn,
                cancel_btn,
                analyze_btn,
                generate_diagrams_btn,
                diagram_viewer,
                analyze_tab_btn,
                analysis_result,       # Added for tab synchronization
                components_table,      # Added for tab synchronization
                relationships_table    # Added for tab synchronization
            ]
        )
        
        # Edit button
        edit_btn.click(
            enable_editing,
            inputs=None,
            outputs=[
                document_display,
                document_editor,
                edit_btn,
                save_btn,
                cancel_btn,
                analyze_btn  # Add this to hide analyze button
            ]
        ).then(
            lambda doc_name: get_document_content_safe(state, doc_name),  # Use safe function
            inputs=[current_doc_name],
            outputs=[document_editor]
        )
        
        # Save button
        save_btn.click(
            save_document,
            inputs=[current_doc_name, document_editor],
            outputs=[
                status_msg,         # visibility
                document_display,
                document_editor,
                edit_btn,
                save_btn,
                cancel_btn,
                change_warning,
                analyze_btn         # Add this to show analyze button again
            ]
        ).then(
            lambda doc_name: get_document_content_safe(state, doc_name),  # Use safe function
            inputs=[current_doc_name],
            outputs=[document_display]
        ).then(
            load_documents,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        )
        
        # Cancel button
        cancel_btn.click(
            cancel_editing,
            inputs=[current_doc_name],
            outputs=[
                document_display,
                document_editor,
                edit_btn,
                save_btn,
                cancel_btn,
                document_editor,
                analyze_btn  # Add this to show analyze button again
            ]
        )
        
        # New document button - Fixed scope update
        new_doc_btn.click(
            create_new_document,
            inputs=None,
            outputs=[document_list, document_list, document_details]
        ).then(
            # Explicitly update scope display
            ensure_scope_ui_update,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        )
        
        # Upload document button - Fixed scope update
        upload_doc_btn.upload(
            process_uploaded_file,
            inputs=[upload_doc_btn],
            outputs=[document_list, document_list, document_details]
        ).then(
            # Explicitly update scope display
            ensure_scope_ui_update,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        )
        
        # Delete selected button - Fixed scope update and UI freezing
        delete_selected_btn.click(
            delete_selected_documents,
            inputs=[document_list],
            outputs=[document_list, document_list, document_details]
        ).then(
            # Use simple function for document view reset
            lambda: (
                "",                          # Clear current document name
                NO_DOCUMENT_SELECTED,        # Reset heading
                NO_DOCUMENT_CONTENT,         # Reset content
                gr.update(visible=False),    # Hide change warning
                gr.update(visible=False),    # Hide edit button
                gr.update(visible=False),    # Hide save button
                gr.update(visible=False),    # Hide cancel button
                gr.update(visible=False),    # Hide analyze button
                gr.update(visible=False)     # Hide generate diagrams button
            ),
            inputs=None,
            outputs=[
                current_doc_name,
                document_heading,
                document_display,
                change_warning,
                edit_btn,
                save_btn,
                cancel_btn,
                analyze_btn,
                generate_diagrams_btn
            ]
        ).then(
            # Use simple function for diagram and analysis reset
            lambda: (
                NO_DIAGRAM_SELECTED,         # Reset diagram viewer
                gr.update(visible=False),    # Hide analyze button in analysis tab
                NO_ANALYSIS_SELECTED,        # Reset analysis
                [],                          # Clear components table
                []                           # Clear relationships table
            ),
            inputs=None,
            outputs=[
                diagram_viewer,
                analyze_tab_btn,
                analysis_result,
                components_table,
                relationships_table
            ]
        ).then(
            # Explicitly update scope display
            ensure_scope_ui_update,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        )
        
        # Diagram type change
        diagram_type.change(
            update_diagram_view,
            inputs=[current_doc_name, diagram_type],
            outputs=[diagram_viewer]
        )
        
        # Generate diagrams button - Added loading state
        generate_diagrams_btn.click(
            # First show loading state
            lambda: GENERATING_DIAGRAMS,
            inputs=None,
            outputs=[diagram_viewer]
        ).then(
            generate_diagrams,
            inputs=[current_doc_name],
            outputs=[diagram_status]
        ).then(
            update_diagram_view,
            inputs=[current_doc_name, diagram_type],
            outputs=[diagram_viewer]
        ).then(
            load_documents,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        )
        
        # Analyze button in document tab - Added loading state, removed diagram generation
        analyze_btn.click(
            # First show loading state
            lambda: (ANALYSIS_LOADING, gr.update(visible=False), [], []),
            inputs=None,
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        ).then(
            analyze_document,
            inputs=[current_doc_name],
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        ).then(
            lambda: 2,  # Switch to analysis tab (index 2)
            inputs=None,
            outputs=document_tabs
        ).then(
            load_documents,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        )
        
        # Analyze button in analysis tab - Added loading state, removed diagram generation
        analyze_tab_btn.click(
            # First show loading state
            lambda: (ANALYSIS_LOADING, gr.update(visible=False), [], []),
            inputs=None,
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        ).then(
            analyze_document,
            inputs=[current_doc_name],
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        ).then(
            load_documents,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        )
        
        # When the Architecture tab is clicked - Explicit UI refresh
        architecture_tab.select(
            # Explicit call to reload everything
            ensure_scope_ui_update,
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        ).then(
            # Reset document view
            lambda: (
                "",                          # Clear current document name
                NO_DOCUMENT_SELECTED,        # Reset heading
                NO_DOCUMENT_CONTENT,         # Reset content
                gr.update(visible=False),    # Hide change warning
                gr.update(visible=False),    # Hide edit button
                gr.update(visible=False),    # Hide save button
                gr.update(visible=False),    # Hide cancel button
                gr.update(visible=False),    # Hide analyze button
                gr.update(visible=False),    # Hide generate diagrams button
                NO_DIAGRAM_SELECTED,         # Reset diagram viewer
                gr.update(visible=False),    # Hide analyze button in analysis tab
                NO_ANALYSIS_SELECTED,        # Reset analysis
                [],                          # Clear components table
                []                           # Clear relationships table
            ),
            inputs=None,
            outputs=[
                current_doc_name,
                document_heading,
                document_display,
                change_warning,
                edit_btn,
                save_btn,
                cancel_btn,
                analyze_btn,
                generate_diagrams_btn,
                diagram_viewer,
                analyze_tab_btn,
                analysis_result,
                components_table,
                relationships_table
            ]
        )
        
        # Initialize data when tab is loaded
        def init_tab_data():
            # Perform initial document loading
            load_documents()
        
        # Call init function when the tab is created
        init_tab_data()
        
    # Return the tab to be added to the main UI
    return architecture_tab