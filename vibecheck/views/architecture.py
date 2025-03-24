"""
Architecture tab module for the VibeCheck app.

This module contains the main tab creation function and UI components.
Event handling and business logic are in architecture_handlers.py.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os

import gradio as gr

from vibecheck.utils.file_utils import ensure_directory
from vibecheck.utils.architecture_ui_utils import format_scope_text
from vibecheck import config
from vibecheck.constants.architecture_constants import *
from vibecheck.handlers.architecture_handlers import ArchitectureHandlers


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
                    min_width=200,      # Minimum width to ensure visibility
                    show_label=False
                )
                
                # Document details
                document_details = gr.Dataframe(
                    headers=DOCUMENT_HEADERS,
                    col_count=(2, "fixed"),
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
                            elem_id="diagram-type-selector",
                            interactive=False
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
        
        # Create a dictionary of UI components to be used by the handlers
        components = {
            "document_list": document_list,
            "document_details": document_details,
            "current_doc_name": current_doc_name,
            "document_heading": document_heading,
            "document_display": document_display,
            "document_editor": document_editor,
            "change_warning": change_warning,
            "scope_display": scope_display,
            "edit_btn": edit_btn,
            "save_btn": save_btn,
            "cancel_btn": cancel_btn,
            "analyze_btn": analyze_btn,
            "generate_diagrams_btn": generate_diagrams_btn,
            "diagram_type": diagram_type,
            "diagram_status": diagram_status,
            "diagram_viewer": diagram_viewer,
            "analyze_tab_btn": analyze_tab_btn,
            "analysis_result": analysis_result,
            "components_table": components_table,
            "relationships_table": relationships_table,
            "status_msg": status_msg,
            "document_tabs": document_tabs
        }
        
        # Initialize the handlers
        handlers = ArchitectureHandlers(state, document_state, components)
        
        # ----- EVENT HANDLERS -----
        
        # Run initialization with a slight delay to ensure UI is ready
        def delayed_init():
            import time
            print("Starting delayed initialization")
            time.sleep(0.5)  # Short delay to let the UI render
            handlers.init_tab_data()
            print("Delayed initialization complete")
            
        import threading
        threading.Thread(target=delayed_init).start()
        
        # Document list selection for scope
        document_list.change(
            handlers.update_scope_selection,
            inputs=[document_list],
            outputs=[scope_display, document_list]
        )
        
        # Document selection from dataframe
        # Document selection from dataframe
        document_details.select(
            handlers.on_document_select,
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
                relationships_table,
                diagram_type  # Add this line
            ]
        )
        
        # Edit button
        edit_btn.click(
            handlers.enable_editing,
            inputs=None,
            outputs=[
                document_display,
                document_editor,
                edit_btn,
                save_btn,
                cancel_btn,
                analyze_btn  
            ]
        ).then(
            handlers.get_document_content_for_edit,
            inputs=[current_doc_name],
            outputs=[document_editor]
        )
        
        # Save button
        save_btn.click(
            handlers.save_document,
            inputs=[current_doc_name, document_editor],
            outputs=[
                status_msg,         
                document_display,
                document_editor,
                edit_btn,
                save_btn,
                cancel_btn,
                change_warning,
                analyze_btn         
            ]
        ).then(
            handlers.get_document_content_for_display,
            inputs=[current_doc_name],
            outputs=[document_display]
        )
        
        # Cancel button
        cancel_btn.click(
            handlers.cancel_editing,
            inputs=[current_doc_name],
            outputs=[
                document_display,
                document_editor,
                edit_btn,
                save_btn,
                cancel_btn,
                document_editor,
                analyze_btn  
            ]
        )
        
        # New document button
        new_doc_btn.click(
            handlers.create_new_document,
            inputs=None,
            outputs=[document_list, document_list, document_details]
        ).then(
            handlers.update_document_list_ui,
            inputs=[document_list, document_list, document_details],
            outputs=[document_list, scope_display]
        )
        
        # Upload document button
        upload_doc_btn.upload(
            handlers.process_uploaded_file,
            inputs=[upload_doc_btn],
            outputs=[document_list, document_list, document_details]
        ).then(
            handlers.update_document_list_ui,
            inputs=[document_list, document_list, document_details],
            outputs=[document_list, scope_display]
        )

        def get_selected_documents(handler_instance):
            """Helper function to get only the selected documents."""
            selected_docs = handler_instance.components["document_list"].value
            if not isinstance(selected_docs, list):
                selected_docs = [selected_docs] if selected_docs else []
            return selected_docs

        # Update the delete_selected_btn.click to use this helper
        delete_selected_btn.click(
            # First capture the current selection
            lambda: gr.update(interactive=False),  # Temporarily disable the button during deletion
            inputs=None,
            outputs=delete_selected_btn
        ).then(
            # This function will directly handle deletion
            lambda selected_docs, project_state: handlers.direct_delete_documents(selected_docs, project_state),
            # Pass the current selection and project state
            inputs=[document_list, gr.State(lambda: state)],
            # Return the updated document list, scope, and details
            outputs=[document_list, document_list, document_details, scope_display]
        ).then(
            # Reset document view if needed
            handlers.reset_document_view,
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
            # Reset diagram and analysis views
            handlers.reset_diagram_analysis_view,
            inputs=None,
            outputs=[
                diagram_viewer,
                analyze_tab_btn,
                analysis_result,
                components_table,
                relationships_table
            ]
        ).then(
            # Re-enable the delete button
            lambda: gr.update(interactive=True),
            inputs=None,
            outputs=delete_selected_btn
        )

        # Diagram type change
        diagram_type.change(
            handlers.update_diagram_view,
            inputs=[current_doc_name, diagram_type],
            outputs=[diagram_viewer]
        )
        
        # Generate diagrams button
        generate_diagrams_btn.click(
            handlers.show_generating_diagrams,
            inputs=None,
            outputs=[diagram_viewer]
        ).then(
            handlers.generate_diagrams,
            inputs=[current_doc_name],
            outputs=[diagram_status, diagram_type]  # Add diagram_type to outputs
        ).then(
            handlers.update_diagram_view,
            inputs=[current_doc_name, diagram_type],
            outputs=[diagram_viewer]
        )
        
        # Analyze button in document tab
        analyze_btn.click(
            handlers.show_analysis_loading,
            inputs=None,
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        ).then(
            handlers.analyze_document,
            inputs=[current_doc_name],
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        ).then(
            handlers.switch_to_analysis_tab,
            inputs=None,
            outputs=[document_tabs]
        )
        
        # Analyze button in analysis tab
        analyze_tab_btn.click(
            handlers.show_analysis_loading,
            inputs=None,
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        ).then(
            handlers.analyze_document,
            inputs=[current_doc_name],
            outputs=[
                analysis_result,
                change_warning,
                components_table,
                relationships_table
            ]
        )
        
        # Tab selection event
        architecture_tab.select(
            handlers.load_documents,  # This now preserves selection and returns both doc_names and preserved selection
            inputs=None,
            outputs=[document_list, document_list, document_details, scope_display]
        ).then(
            # Reset document view
            handlers.reset_document_view,
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
            # Reset diagram and analysis view
            handlers.reset_diagram_analysis_view,
            inputs=None,
            outputs=[
                diagram_viewer,
                analyze_tab_btn,
                analysis_result,
                components_table,
                relationships_table
            ]
        )
        
        # Tab data will be initialized by the delayed initialization thread
        
    # Return the tab to be added to the main UI
    return architecture_tab
