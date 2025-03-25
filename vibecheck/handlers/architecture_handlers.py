"""
Architecture handlers module for the VibeCheck app.

This module contains the event handlers and business logic for the architecture tab.
It is separated from the UI component definitions for maintainability.
"""

import json
import os
from datetime import datetime
from typing import Dict

import gradio as gr

from vibecheck import config
from vibecheck.constants.architecture_constants import *
from vibecheck.constants.architecture_templates import (
    GENERATING_DIAGRAMS_TEMPLATE,
    NO_DIAGRAM_SELECTED_TEMPLATE,
    NO_DOCUMENT_FIRST_TEMPLATE
)
from vibecheck.controllers.architecture_controller import ArchitectureController
from vibecheck.utils.architecture_ui_utils import (
    get_document_list, get_document_content, get_document_content_safe,
    save_document_content, update_scope, has_document_changed, display_diagram,
    create_default_document_content, clean_document_selection, format_scope_text
)
from vibecheck.utils.cache_utils import AnalysisCache
from vibecheck.utils.file_utils import ensure_directory


class ArchitectureHandlers:
    """
    Handlers for architecture tab events and business logic.
    
    This class encapsulates all the event handlers and business logic for
    the architecture tab, keeping them separate from the UI component definitions.
    """

    def __init__(self, state: Dict, document_state: Dict, components: Dict):
        """
        Initialize the handlers with application state and UI components.
        
        Args:
            state: Application state dictionary
            document_state: Document state dictionary
            components: Dictionary of UI components
        """
        self.state = state
        self.document_state = document_state
        self.components = components

    def load_documents(self):
        """
        Load document list using the existing get_document_list utility.
        Ensures current UI selection is preserved.

        Returns:
            Tuple of (all document names, preserved selection, document rows, scope text)
        """
        if not self.state.get("current_project"):
            print("No current project found")
            return [], [], [], NO_SCOPE

        try:
            project_path = self.state["current_project"].metadata.path

            # Use the existing utility function to get all documents
            doc_names, doc_rows, scope_from_file = get_document_list(project_path)

            # CRITICAL: Get the current UI selection
            current_selection = self.components["document_list"].value
            if not isinstance(current_selection, list):
                current_selection = [current_selection] if current_selection else []

            # Clean up the current selection
            valid_current_selection = [doc for doc in current_selection if doc in doc_names]

            # If no valid current selection, use scope from file
            preserved_selection = valid_current_selection if valid_current_selection else scope_from_file

            # Update scope in document state
            self.document_state["scope"] = preserved_selection

            # Format scope text
            scope_text = format_scope_text(preserved_selection)

            print(f"Loaded documents: {doc_names}")
            print(f"Current UI selection: {current_selection}")
            print(f"Scope from file: {scope_from_file}")
            print(f"Preserved selection: {preserved_selection}")

            # ALSO update the scope file to match the preserved selection
            try:
                scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
                scope_data = {
                    'documents': preserved_selection,
                    'last_updated': datetime.now().isoformat()
                }
                ensure_directory(os.path.dirname(scope_path))
                with open(scope_path, 'w') as f:
                    json.dump(scope_data, f, indent=2)
            except Exception as e:
                print(f"Error updating scope file: {e}")

            # Return full document list, preserved selection, rows, and scope text
            return doc_names, preserved_selection, doc_rows, scope_text

        except Exception as e:
            import traceback
            print(f"Error in load_documents: {e}")
            traceback.print_exc()
            return [], [], [], SCOPE_ERROR

    def ensure_scope_ui_update(self):
        """Force reload the document list and scope to ensure UI is updated."""
        if not self.state.get("current_project"):
            return [], [], [], NO_SCOPE

        project_path = self.state["current_project"].metadata.path

        # Get fresh document list and scope
        doc_names, doc_rows, scope = get_document_list(project_path)

        # Clean scope to ensure proper format
        clean_scope = clean_document_selection(scope)

        # Update scope text
        scope_text = format_scope_text(clean_scope)

        # Return values for UI update
        return doc_names, clean_scope, doc_rows, scope_text

    def update_scope_selection(self, selected_docs):
        """
        Update scope when selection changes in the UI.
        Preserves ALL document choices while updating the selection.

        Args:
            selected_docs: Currently selected documents in the UI

        Returns:
            Tuple of (scope display text, updated document list)
        """
        if not self.state.get("current_project"):
            print("No current project in update_scope_selection")
            return NO_SCOPE, gr.update(choices=[], value=[])

        # Get project path for path operations
        project_path = self.state["current_project"].metadata.path

        # Create a clean list of selected documents
        clean_selected_docs = clean_document_selection(selected_docs)
        print(f"Updating scope with selection: {clean_selected_docs}")

        # Update document state
        self.document_state["scope"] = clean_selected_docs

        # Get the complete list of all documents
        docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
        all_doc_names = []
        if os.path.exists(docs_dir):
            for filename in os.listdir(docs_dir):
                if filename.endswith(('.json', '.md', '.txt', '.yaml', '.xml')):
                    doc_name = os.path.splitext(filename)[0]
                    if doc_name not in all_doc_names:
                        all_doc_names.append(doc_name)

        # Save to scope file
        scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
        try:
            # Ensure directory exists
            ensure_directory(os.path.dirname(scope_path))

            # Save scope data
            scope_data = {
                'documents': clean_selected_docs,
                'last_updated': datetime.now().isoformat()
            }

            with open(scope_path, 'w') as f:
                json.dump(scope_data, f, indent=2)

            print(f"Saved scope with {len(clean_selected_docs)} documents")
        except Exception as e:
            print(f"Error saving scope: {e}")

        # Update scope display
        scope_text = format_scope_text(clean_selected_docs)

        # CRITICAL: Keep all document choices and just update value
        return scope_text, gr.update(
            choices=all_doc_names,  # Keep ALL document choices
            value=clean_selected_docs  # Update the selection
        )

    def update_document_list_ui(self, doc_names, scope, doc_rows):
        """
        Update document list UI with consistent selection and scope handling.

        Ensures that:
        1. ALL available documents remain in the choices
        2. Current selection is preserved
        3. If no selection, use the scope
        """
        # Validate inputs
        if not isinstance(doc_names, list):
            doc_names = []
        if not isinstance(scope, list):
            scope = []

        # Get the current selection from the UI component
        current_selection = self.components["document_list"].value
        if not isinstance(current_selection, list):
            current_selection = [current_selection] if current_selection else []

        # Determine the valid selection
        # Priority 1: Current UI selection that exists in doc_names
        # Priority 2: Scope documents that exist in doc_names
        valid_selection = [doc for doc in current_selection if doc in doc_names]

        # If no valid current selection, fall back to scope
        if not valid_selection:
            valid_selection = [doc for doc in scope if doc in doc_names]

        # Format scope text based on selection
        scope_text = format_scope_text(valid_selection)

        print(f"Updating Document List:")
        print(f"  All Documents: {doc_names}")
        print(f"  Current Selection: {current_selection}")
        print(f"  Valid Selection: {valid_selection}")
        print(f"  Scope from file: {scope}")

        # Return update preserving ALL documents and CURRENT selection
        return (
            gr.update(
                choices=doc_names,  # ALL documents remain
                value=valid_selection  # Preserve selection or use scope
            ),
            scope_text
        )

    def on_document_select(self, evt: gr.SelectData):
        """Handle document selection from the details dataframe, refreshing all tabs."""
        if not self.state.get("current_project"):
            return self.reset_document_view() + (NO_DOCUMENT_FIRST, gr.update(visible=False), NO_ANALYSIS_SELECTED, [], [])

        # Get the selected document name directly from the row data
        try:
            # Extract the full document name from the row value
            if hasattr(evt, 'row_value') and isinstance(evt.row_value, list) and len(evt.row_value) > 0:
                doc_name = str(evt.row_value[0])
            elif hasattr(evt, 'value') and isinstance(evt.value, list) and len(evt.value) > 0:
                doc_name = str(evt.value[0])
            else:
                doc_name = None

            print(f"Raw selection data: {evt}")
            print(f"Selected document: {doc_name}")

            if not doc_name:
                return self.reset_document_view() + (NO_DOCUMENT_FIRST, gr.update(visible=False), NO_ANALYSIS_SELECTED, [], [])

            project_path = self.state["current_project"].metadata.path

            # Update current document in state
            self.document_state["current_document"] = doc_name
            self.document_state["editing"] = False

            # Get document content
            content = get_document_content(project_path, doc_name)

            print(f"Document content length: {len(content) if content else 0}")

            # Check if document has changed
            has_changed = has_document_changed(project_path, doc_name)

            # Update heading
            heading = f"### 📝 {doc_name}"

            # Check if any diagrams exist for this document
            diagrams_exist = False
            diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
            if os.path.exists(diagrams_dir):
                for diagram_type in ["module", "dataflow", "security"]:
                    json_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.json")
                    mmd_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.mmd")
                    if os.path.exists(json_path) or os.path.exists(mmd_path):
                        diagrams_exist = True
                        break

            # Generate diagram HTML for selected diagram type
            diagram_html = display_diagram(project_path, doc_name, self.components["diagram_type"].value)

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

            # Make sure to return values in the EXACT order expected by the outputs in architecture.py
            return (
                doc_name,  # Current document name
                heading,  # Document heading
                content,  # Document content
                gr.update(visible=has_changed),  # Show change warning based on status
                gr.update(visible=True),  # Show edit button
                gr.update(visible=False),  # Hide save button
                gr.update(visible=False),  # Hide cancel button
                gr.update(visible=True),  # Show analyze button
                gr.update(visible=True),  # Show generate diagrams button
                diagram_html,  # Update diagram viewer
                gr.update(visible=True),  # Show analyze button in analysis tab
                analysis_content,  # Analysis content
                components_data,  # Components table data
                relationships_data,  # Relationships table data
                gr.update(interactive=diagrams_exist)  # Enable diagram selector only if diagrams exist
            )
        except Exception as e:
            import traceback
            print(ERROR_DOCUMENT_SELECTION.format(error=e))
            traceback.print_exc()
            return self.reset_document_view() + (NO_DOCUMENT_FIRST, gr.update(visible=False), NO_ANALYSIS_SELECTED, [], [])

    def enable_editing(self):
        """Enable document editing mode."""
        return (
            gr.update(visible=False),  # Hide document display
            gr.update(visible=True),  # Show document editor
            gr.update(visible=False),  # Hide edit button
            gr.update(visible=True),  # Show save button
            gr.update(visible=True),  # Show cancel button
            gr.update(visible=False)  # Hide analyze button
        )

    def get_document_content_for_edit(self, doc_name):
        """Get document content for editing."""
        return get_document_content_safe(self.state, doc_name)

    def get_document_content_for_display(self, doc_name):
        """Get document content for display."""
        return get_document_content_safe(self.state, doc_name)

    def save_document(self, doc_name, content):
        """Save document and update UI."""
        if not self.state.get("current_project") or not doc_name:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(
                visible=False), gr.update(visible=False), gr.update(visible=False)

        project_path = self.state["current_project"].metadata.path

        # Save the document
        success = save_document_content(project_path, doc_name, content)

        # Update state
        self.document_state["editing"] = False

        # Document has definitely changed since we just saved it
        return (
            gr.update(visible=False),  # Hide status
            gr.update(visible=True),  # Show document display
            gr.update(visible=False),  # Hide document editor
            gr.update(visible=True),  # Show edit button
            gr.update(visible=False),  # Hide save button
            gr.update(visible=False),  # Hide cancel button
            gr.update(visible=True),  # Show change warning
            gr.update(visible=True)  # Show analyze button
        )

    def cancel_editing(self, doc_name):
        """Cancel editing and restore original content."""
        self.document_state["editing"] = False

        if not self.state.get("current_project") or not doc_name:
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(
                visible=False), "", gr.update(visible=True)

        project_path = self.state["current_project"].metadata.path

        # Reload document content
        content = get_document_content(project_path, doc_name)

        return (
            gr.update(visible=True),  # Show document display
            gr.update(visible=False),  # Hide document editor
            gr.update(visible=True),  # Show edit button
            gr.update(visible=False),  # Hide save button
            gr.update(visible=False),  # Hide cancel button
            content,  # Original content for the editor
            gr.update(visible=True)  # Show analyze button
        )

    def process_uploaded_file(self, file_info):
        """Process uploaded document file."""
        if not self.state.get("current_project"):
            print("No current project when uploading file")
            return [], [], []

        if file_info is None:
            print("No file info provided")
            return [], [], []

        # For UploadButton, file_info is always a file object
        file_path = file_info.name if hasattr(file_info, 'name') else None

        if not file_path or not os.path.exists(file_path):
            print(f"File path invalid or does not exist: {file_path}")
            return [], [], []

        try:
            project_path = self.state["current_project"].metadata.path
            print(f"Processing uploaded file: {file_path} for project: {project_path}")

            # Ensure all required directories exist
            docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
            ensure_directory(docs_dir)

            # Ensure the scope directory exists
            scope_dir = os.path.dirname(os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE))
            ensure_directory(scope_dir)

            # Extract filename without extension
            filename = os.path.basename(file_path)
            doc_name = os.path.splitext(filename)[0]

            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"Successfully read file content, length: {len(content)}")
            except Exception as e:
                print(ERROR_READING_FILE.format(error=e))
                return [], [], []

            # Save as document
            success = save_document_content(project_path, doc_name, content)

            # Get current scope
            scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
            current_scope = []

            if os.path.exists(scope_path):
                try:
                    with open(scope_path, 'r') as f:
                        scope_data = json.load(f)
                        current_scope = scope_data.get('documents', [])
                except Exception as e:
                    print(f"Error reading scope file: {e}")

            # Add the new document to the scope if not already there
            if success and doc_name not in current_scope:
                current_scope.append(doc_name)

                # Save updated scope
                try:
                    scope_data = {
                        'documents': current_scope,
                        'last_updated': datetime.now().isoformat()
                    }

                    with open(scope_path, 'w') as f:
                        json.dump(scope_data, f, indent=2)

                    print(f"Updated scope with uploaded document: {current_scope}")
                except Exception as e:
                    print(f"Error updating scope file: {e}")

            # Get updated document list
            try:
                doc_names, doc_rows, scope = get_document_list(project_path)
                print(f"After upload - doc_names: {len(doc_names)}, scope: {len(scope)}")
            except Exception as e:
                print(f"Error getting updated document list: {e}")
                # Fallback to manual list
                doc_names = []
                doc_rows = []
                if os.path.exists(docs_dir):
                    for filename in os.listdir(docs_dir):
                        if filename.endswith(('.json', '.md')):
                            name = os.path.splitext(filename)[0]
                            if name not in doc_names:
                                doc_names.append(name)
                                doc_rows.append([name, "Just uploaded", "Not analyzed"])
                scope = current_scope

            return doc_names, scope, doc_rows

        except Exception as e:
            import traceback
            print(f"Unexpected error in process_uploaded_file: {e}")
            traceback.print_exc()
            return [], [], []

    def delete_selected_documents(self, selected_docs):
        """
        Delete only the specifically selected documents.

        Args:
            selected_docs: Documents specifically selected for deletion

        Returns:
            Updated document_list, scope, and document_details
        """
        if not self.state.get("current_project"):
            print("No current project when deleting documents")
            return [], [], []

        # Clean the selection to ensure we have proper string values
        clean_selected_docs = clean_document_selection(selected_docs)

        print(f"Documents selected for deletion: {clean_selected_docs}")

        if not clean_selected_docs:
            print("No documents selected for deletion")
            # Return the current document list if no documents selected for deletion
            try:
                project_path = self.state["current_project"].metadata.path
                doc_names, doc_rows, scope = get_document_list(project_path)
                return doc_names, scope, doc_rows
            except Exception as e:
                print(f"Error getting document list: {e}")
                return [], [], []

        try:
            project_path = self.state["current_project"].metadata.path

            # Get the current list of all documents before deletion
            try:
                all_doc_names, doc_rows, current_scope = get_document_list(project_path)
                print(f"All documents before deletion: {all_doc_names}")
                print(f"Current scope before deletion: {current_scope}")
            except Exception as e:
                print(f"Error getting complete document list: {e}")
                return [], [], []

            # Verify the docs to delete actually exist
            docs_to_delete = [doc for doc in clean_selected_docs if doc in all_doc_names]
            print(f"Verified documents to delete: {docs_to_delete}")

            if not docs_to_delete:
                print("None of the selected documents exist for deletion")
                return all_doc_names, current_scope, doc_rows

            # Track successfully deleted documents
            deleted_docs = []

            # Delete each selected document
            for doc_name in docs_to_delete:
                success = True
                try:
                    # 1. Delete document files
                    docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
                    for ext in ['.json', '.md', '.txt', '.yaml', '.xml']:
                        file_path = os.path.join(docs_dir, f"{doc_name}{ext}")
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                print(f"Deleted file: {file_path}")
                            except PermissionError:
                                print(f"Permission denied when deleting {file_path}")
                                success = False
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}")
                                success = False

                    # 2. Delete all diagram files
                    diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
                    if os.path.exists(diagrams_dir):
                        for filename in os.listdir(diagrams_dir):
                            if filename.startswith(f"{doc_name}_"):
                                try:
                                    diagram_path = os.path.join(diagrams_dir, filename)
                                    os.remove(diagram_path)
                                    print(f"Deleted diagram: {diagram_path}")
                                except Exception as e:
                                    print(f"Error deleting diagram {filename}: {e}")

                    # 3. Delete analysis files
                    analysis_path = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR, f"{doc_name}.json")
                    if os.path.exists(analysis_path):
                        try:
                            os.remove(analysis_path)
                            print(f"Deleted analysis: {analysis_path}")
                        except Exception as e:
                            print(f"Error deleting analysis {analysis_path}: {e}")

                    # 4. Delete any cache files
                    cache_dir = os.path.join(project_path, config.CACHE_DIR)
                    if os.path.exists(cache_dir):
                        for filename in os.listdir(cache_dir):
                            if f"architecture_{doc_name}" in filename:
                                try:
                                    cache_path = os.path.join(cache_dir, filename)
                                    os.remove(cache_path)
                                    print(f"Deleted cache: {cache_path}")
                                except Exception as e:
                                    print(f"Error deleting cache {filename}: {e}")

                    if success:
                        deleted_docs.append(doc_name)
                except Exception as e:
                    print(f"Error processing deletion for {doc_name}: {e}")
                    import traceback
                    traceback.print_exc()

            # Get the updated list of documents after deletion
            try:
                new_doc_names, new_doc_rows, _ = get_document_list(project_path)
                print(f"Documents after deletion: {new_doc_names}")
            except Exception as e:
                print(f"Error getting document list after deletion: {e}")
                # Calculate new document list by removing deleted docs
                new_doc_names = [doc for doc in all_doc_names if doc not in deleted_docs]
                new_doc_rows = [row for row in doc_rows if row[0] not in deleted_docs]

            # Update scope by removing only the deleted documents
            updated_scope = [doc for doc in current_scope if doc not in deleted_docs]
            print(f"Updated scope after deletion: {updated_scope}")

            # Update the scope file
            try:
                scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
                scope_data = {
                    'documents': updated_scope,
                    'last_updated': datetime.now().isoformat()
                }

                # Ensure the directory exists
                scope_dir = os.path.dirname(scope_path)
                ensure_directory(scope_dir)

                with open(scope_path, 'w') as f:
                    json.dump(scope_data, f, indent=2)

                print(f"Saved updated scope: {updated_scope}")
            except Exception as e:
                print(f"Error updating scope file after deletion: {e}")

            print(f"Returning {len(new_doc_names)} documents and {len(updated_scope)} in scope")
            return new_doc_names, updated_scope, new_doc_rows

        except Exception as e:
            import traceback
            print(f"Unexpected error in delete_selected_documents: {e}")
            traceback.print_exc()

            # Try to get current document list as fallback
            try:
                doc_names, doc_rows, scope = get_document_list(self.state["current_project"].metadata.path)
                return doc_names, scope, doc_rows
            except Exception:
                # Return empty lists only as last resort
                return [], [], []

    def reset_document_view(self):
        """Reset the document view when no document is selected."""
        return (
            "",  # Clear current document
            NO_DOCUMENT_SELECTED,  # Reset heading
            NO_DOCUMENT_CONTENT,  # Reset content
            gr.update(visible=False),  # Hide change warning
            gr.update(visible=False),  # Hide edit button
            gr.update(visible=False),  # Hide save button
            gr.update(visible=False),  # Hide cancel button
            gr.update(visible=False),  # Hide analyze button
            gr.update(visible=False)  # Hide generate diagrams button
        )

    def reset_diagram_analysis_view(self):
        """Reset the diagram and analysis views."""
        return (
            NO_DIAGRAM_SELECTED_TEMPLATE,  # Reset diagram viewer
            gr.update(visible=False),  # Hide analyze button in analysis tab
            NO_ANALYSIS_SELECTED,  # Reset analysis
            [],  # Clear components table
            []  # Clear relationships table
        )

    def update_diagram_view(self, doc_name, diagram_type):
        """Update diagram view with selected document and diagram type."""
        if not doc_name:
            return NO_DOCUMENT_FIRST_TEMPLATE

        if not self.state.get("current_project"):
            return NO_PROJECT_OPEN

        project_path = self.state["current_project"].metadata.path
        return display_diagram(project_path, doc_name, diagram_type)

    def generate_diagrams(self, doc_name):
        """Generate diagrams for a document, ensuring forced re-generation on every request."""
        if not self.state.get("current_project") or not doc_name:
            return gr.update(visible=False), gr.update(interactive=False)

        project_path = self.state["current_project"].metadata.path

        # Get document content
        content = get_document_content(project_path, doc_name)
        if not content:
            print(f"No content found for document: {doc_name}")
            return gr.update(visible=False), gr.update(interactive=False)

        # IMPORTANT: Force diagram regeneration by clearing cache
        cache_key = f"architecture_{doc_name}"
        AnalysisCache.invalidate_cache(project_path, cache_key)

        # Delete any existing diagram files to force regeneration
        diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
        if os.path.exists(diagrams_dir):
            for diagram_type in ["module", "dataflow", "security"]:
                json_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.json")
                mmd_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.mmd")

                if os.path.exists(json_path):
                    try:
                        os.remove(json_path)
                        print(f"Deleted existing diagram for regeneration: {json_path}")
                    except Exception as e:
                        print(f"Error deleting existing diagram: {e}")

                if os.path.exists(mmd_path):
                    try:
                        os.remove(mmd_path)
                        print(f"Deleted existing Mermaid file for regeneration: {mmd_path}")
                    except Exception as e:
                        print(f"Error deleting existing Mermaid file: {e}")

        # Generate diagrams via controller
        try:
            diagrams = ArchitectureController.generate_diagrams(project_path, doc_name, content)

            if diagrams:
                print(f"Successfully generated {len(diagrams)} diagrams for {doc_name}")
                for diagram_type, diagram in diagrams.items():
                    print(f"Generated {diagram_type} diagram of length: {len(diagram.content) if diagram and hasattr(diagram, 'content') else 0}")

                # Enable the diagram type selector since diagrams are now available
                return gr.update(visible=False), gr.update(interactive=True)
            else:
                return gr.update(visible=False), gr.update(interactive=False)
        except Exception as e:
            import traceback
            print(ERROR_GENERATING_DIAGRAMS.format(error=e))
            traceback.print_exc()
            return gr.update(visible=False), gr.update(interactive=False)

    def show_generating_diagrams(self):
        """Show loading state for diagram generation."""
        return GENERATING_DIAGRAMS_TEMPLATE

    def analyze_document(self, doc_name):
        """Analyze architecture document with enhanced critical focus and force regeneration."""
        if not self.state.get("current_project") or not doc_name:
            print(f"No project or document name for analysis: {doc_name}")
            return EMPTY_DOCUMENT_ERROR, gr.update(visible=False), [], []

        project_path = self.state["current_project"].metadata.path

        # Get document content
        content = get_document_content(project_path, doc_name)
        if not content or content.strip() == "":
            print(f"Empty document content for: {doc_name}")
            return EMPTY_DOCUMENT_ERROR, gr.update(visible=False), [], []

        # IMPORTANT: Force re-analysis by clearing cache and existing analysis file
        cache_key = f"architecture_{doc_name}"
        AnalysisCache.invalidate_cache(project_path, cache_key)

        # Delete any existing analysis file to force regeneration
        analysis_path = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR, f"{doc_name}.json")
        if os.path.exists(analysis_path):
            try:
                os.remove(analysis_path)
                print(f"Deleted existing analysis for regeneration: {analysis_path}")
            except Exception as e:
                print(f"Error deleting existing analysis: {e}")

        # Analyze document via controller
        try:
            analysis = ArchitectureController.analyze_architecture_document(project_path, doc_name, content)
            if not analysis:
                print(f"Analysis failed for document: {doc_name}")
                return ANALYSIS_FAILED, gr.update(visible=False), [], []

            print(f"Analysis completed successfully for: {doc_name}")
        except Exception as e:
            import traceback
            print(ERROR_ANALYZING_DOCUMENT.format(error=e))
            traceback.print_exc()
            return ERROR_ANALYZING_DOCUMENT.format(error=e), gr.update(visible=False), [], []

        # Get components and relationships
        try:
            # Force regeneration of components and relationships
            data = ArchitectureController.get_components_and_relationships(project_path, doc_name)

            components = data.get("components", [])
            relationships = data.get("relationships", [])

            # Format for the UI
            components_data = [[c.get("name", ""), c.get("description", "")] for c in components]
            relationships_data = [[r.get("source", ""), r.get("type", "").replace("_", " "), r.get("target", "")] for r in relationships]

            print(f"Found {len(components_data)} components and {len(relationships_data)} relationships")
        except Exception as e:
            import traceback
            print(f"Error getting components and relationships: {e}")
            traceback.print_exc()
            components_data = []
            relationships_data = []

        # Make sure we return a string, not a boolean
        if isinstance(analysis, bool):
            analysis = "Analysis completed" if analysis else "Analysis failed"

        return analysis, gr.update(visible=False), components_data, relationships_data

    def show_analysis_loading(self):
        """Show loading state for analysis."""
        return ANALYSIS_LOADING, gr.update(visible=False), [], []

    def switch_to_analysis_tab(self):
        """Switch to the analysis tab."""
        return 2  # Index of the analysis tab

    def init_tab_data(self):
        """Initialize the tab data with proper scope handling and UI updates"""
        if not self.state.get("current_project"):
            print("No current project in init_tab_data")
            return

        try:
            project_path = self.state["current_project"].metadata.path
            print(f"Initializing tab data for project: {project_path}")

            # Ensure all required directories exist
            docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
            ensure_directory(docs_dir)
            ensure_directory(os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR))
            ensure_directory(os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR))

            # Manually list all documents
            manual_doc_names = []
            if os.path.exists(docs_dir):
                for filename in os.listdir(docs_dir):
                    if filename.endswith(('.json', '.md', '.txt', '.yaml', '.xml')):
                        doc_name = os.path.splitext(filename)[0]
                        if doc_name not in manual_doc_names:
                            manual_doc_names.append(doc_name)

            # Try to load scope
            try:
                scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
                with open(scope_path, 'r') as f:
                    scope_data = json.load(f)
                    existing_scope = scope_data.get('documents', [])
            except (FileNotFoundError, json.JSONDecodeError):
                existing_scope = []

            # Validate scope against existing documents
            clean_scope = [doc for doc in existing_scope if doc in manual_doc_names]

            # Create document rows
            doc_rows = [[name, "Unknown", "Not analyzed"] for name in manual_doc_names]

            # Scope text
            scope_text = format_scope_text(clean_scope)

            print(f"All Documents: {manual_doc_names}")
            print(f"Current Scope: {clean_scope}")

            # Force direct component updates
            if hasattr(self.components["document_list"], "update"):
                self.components["document_list"].update(
                    choices=manual_doc_names,  # ALL documents
                    value=clean_scope  # Selected documents from scope
                )

            if hasattr(self.components["document_details"], "update"):
                self.components["document_details"].update(value=doc_rows)

            if hasattr(self.components["scope_display"], "update"):
                self.components["scope_display"].update(value=scope_text)

            print("Tab data initialization complete")
        except Exception as e:
            import traceback
            print(f"Error in init_tab_data: {e}")
            traceback.print_exc()

    def create_new_document(self):
        """Create a new document with unique name and update scope selection properly."""
        if not self.state.get("current_project"):
            print("No current project when creating new document")
            return [], [], []

        try:
            project_path = self.state["current_project"].metadata.path
            print(f"Creating new document in project: {project_path}")

            # Ensure all required directories exist
            docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
            ensure_directory(docs_dir)

            # Ensure the scope directory exists
            scope_dir = os.path.dirname(os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE))
            ensure_directory(scope_dir)

            # Generate a unique document name
            base_name = "Specification "
            doc_name = base_name
            count = 1

            # Get existing document names
            try:
                doc_names, _, _ = get_document_list(project_path)
                print(f"Existing documents: {doc_names}")
            except Exception as e:
                print(f"Error getting document list: {e}")
                doc_names = []

            while doc_name in doc_names:
                doc_name = f"{base_name} {count}"
                count += 1

            print(f"Generated unique name: {doc_name}")

            # Create default content
            content = create_default_document_content(doc_name)

            # Save the document directly to ensure it's created
            json_path = os.path.join(docs_dir, f"{doc_name}.json")
            md_path = os.path.join(docs_dir, f"{doc_name}.md")

            try:
                # Save as JSON
                document_data = {
                    'content': content,
                    'last_modified': datetime.now().isoformat()
                }

                with open(json_path, 'w') as f:
                    json.dump(document_data, f, indent=2)

                # Also save as markdown
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"Successfully created document files: {json_path} and {md_path}")
                success = True
            except Exception as e:
                print(f"Error saving document files: {e}")
                success = False

            # Get current scope
            scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
            current_scope = []

            if os.path.exists(scope_path):
                try:
                    with open(scope_path, 'r') as f:
                        scope_data = json.load(f)
                        current_scope = scope_data.get('documents', [])
                except Exception as e:
                    print(f"Error reading scope file: {e}")

            # Add the new document to the scope
            if success and doc_name not in current_scope:
                current_scope.append(doc_name)

                # Save updated scope
                try:
                    scope_data = {
                        'documents': current_scope,
                        'last_updated': datetime.now().isoformat()
                    }

                    with open(scope_path, 'w') as f:
                        json.dump(scope_data, f, indent=2)

                    print(f"Updated scope with new document: {current_scope}")
                except Exception as e:
                    print(f"Error updating scope file: {e}")

            # Get updated document list and scope - use direct file check if get_document_list fails
            try:
                doc_names, doc_rows, scope = get_document_list(project_path)
            except Exception as e:
                print(f"Error getting updated document list: {e}")
                # Fallback to manual list
                doc_names = []
                doc_rows = []
                if os.path.exists(docs_dir):
                    for filename in os.listdir(docs_dir):
                        if filename.endswith(('.json', '.md')):
                            name = os.path.splitext(filename)[0]
                            if name not in doc_names:
                                doc_names.append(name)
                                doc_rows.append([name, "Just created", "Not analyzed"])
                scope = current_scope

            print(f"Returning updated lists: {len(doc_names)} docs, {len(scope)} in scope")
            return doc_names, scope, doc_rows

        except Exception as e:
            import traceback
            print(f"Unexpected error in create_new_document: {e}")
            traceback.print_exc()
            return [], [], []

    def direct_delete_documents(self, selected_docs, app_state):
        """
        Directly delete selected documents, with improved UI refresh handling.

        Args:
            selected_docs: Documents currently checked in UI
            app_state: Application state

        Returns:
            Updated document list, scope, details, and scope display text
        """
        print(f"direct_delete_documents called")
        print(f"User selected documents for deletion: {selected_docs}")

        if not app_state or not app_state.get("current_project"):
            print("No current project")
            return [], [], [], "No current project"

        project_path = app_state["current_project"].metadata.path
        print(f"Project path: {project_path}")

        # Ensure we have a list of strings
        if not isinstance(selected_docs, list):
            print(f"Selected docs is not a list: {type(selected_docs)}")
            selected_docs = [selected_docs] if selected_docs else []

        # Handle empty selection
        if not selected_docs:
            print("No documents selected for deletion")
            try:
                all_docs, all_rows, current_scope = get_document_list(project_path)
                scope_text = format_scope_text(current_scope)
                return all_docs, current_scope, all_rows, scope_text
            except Exception as e:
                print(f"Error getting document list: {e}")
                return [], [], [], "Error loading documents"

        print(f"Documents selected for deletion: {selected_docs}")

        # Get initial document state
        try:
            all_docs, all_rows, current_scope = get_document_list(project_path)
            print(f"Initial docs: {all_docs}")
            print(f"Initial scope: {current_scope}")
        except Exception as e:
            print(f"Error getting initial document state: {e}")
            return [], [], [], f"Error: {str(e)}"

        # Delete the selected documents
        deleted_docs = []
        for doc_name in selected_docs:
            try:
                print(f"Processing deletion for: {doc_name}")
                docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)

                # 1. Delete document files
                deleted_any_files = False
                for ext in ['.json', '.md', '.txt', '.yaml', '.xml']:
                    file_path = os.path.join(docs_dir, f"{doc_name}{ext}")
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                            deleted_any_files = True
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")

                # 2. Delete diagram files
                diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
                if os.path.exists(diagrams_dir):
                    for filename in os.listdir(diagrams_dir):
                        if filename.startswith(f"{doc_name}_"):
                            diagram_path = os.path.join(diagrams_dir, filename)
                            try:
                                os.remove(diagram_path)
                                print(f"Deleted diagram: {diagram_path}")
                            except Exception as e:
                                print(f"Error deleting diagram {filename}: {e}")

                # 3. Delete analysis file
                analysis_path = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR, f"{doc_name}.json")
                if os.path.exists(analysis_path):
                    try:
                        os.remove(analysis_path)
                        print(f"Deleted analysis: {analysis_path}")
                    except Exception as e:
                        print(f"Error deleting analysis {analysis_path}: {e}")

                # Consider the document deleted if we deleted any files
                if deleted_any_files:
                    deleted_docs.append(doc_name)
                    print(f"Successfully deleted document: {doc_name}")
                else:
                    print(f"No files found for document: {doc_name}")

            except Exception as e:
                print(f"Error deleting document {doc_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"Successfully deleted documents: {deleted_docs}")

        # IMPORTANT: Directly refresh document list from filesystem after deletion
        try:
            # Use a short delay to ensure filesystem changes are registered
            import time
            time.sleep(0.1)

            # Get fresh document list from disk
            remaining_docs, remaining_rows, _ = get_document_list(project_path)
            print(f"Fresh document list from disk: {remaining_docs}")
        except Exception as e:
            print(f"Error getting fresh document list: {e}")
            # Calculate remaining docs as fallback
            remaining_docs = [doc for doc in all_docs if doc not in deleted_docs]
            remaining_rows = [row for row in all_rows if row[0] not in deleted_docs]

        # Update scope based on remaining documents
        remaining_scope = [doc for doc in current_scope if doc in remaining_docs]
        print(f"Updated scope: {remaining_scope}")

        # Update the scope file
        try:
            scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
            scope_data = {
                'documents': remaining_scope,
                'last_updated': datetime.now().isoformat()
            }

            # Ensure the directory exists
            ensure_directory(os.path.dirname(scope_path))

            with open(scope_path, 'w') as f:
                json.dump(scope_data, f, indent=2)
            print(f"Updated scope file with: {remaining_scope}")
        except Exception as e:
            print(f"Error updating scope file: {e}")

        # Update the scope display text
        scope_text = format_scope_text(remaining_scope)

        # Clear current document state if it was deleted
        current_doc = self.document_state.get("current_document")
        if current_doc in deleted_docs:
            self.document_state["current_document"] = None
            self.document_state["editing"] = False

        print(f"Final return values:")
        print(f"  - remaining_docs: {remaining_docs}")
        print(f"  - remaining_scope: {remaining_scope}")
        print(f"  - remaining_rows count: {len(remaining_rows)}")

        # Apply Gradio update to ensure UI refreshes
        return gr.update(choices=remaining_docs), gr.update(value=remaining_scope), remaining_rows, scope_text