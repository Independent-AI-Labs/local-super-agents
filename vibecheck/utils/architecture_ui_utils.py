"""
UI utility functions for the architecture module in VibeCheck.

This module provides utility functions specifically for the architecture UI,
handling document content management, visualization, and UI state management.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any

from vibecheck import config
from vibecheck.constants.architecture_templates import NO_DOCUMENT_FIRST_TEMPLATE, DEFAULT_DOCUMENT_TEMPLATE
from vibecheck.controllers.architecture_controller import ArchitectureController
from vibecheck.utils.file_utils import ensure_directory
from vibecheck.constants.architecture_constants import (
    NO_DOCUMENT_FIRST,
    NO_PROJECT_OPEN,
    NO_SCOPE,
    SCOPE_ERROR,
    NO_DIAGRAM_SELECTED
)


# ----- Document management utilities -----

def get_document_list(project_path: str) -> Tuple[List[str], List[List[Any]], List[str]]:
    """
    Get unique list of ALL documents with their metadata.

    Args:
        project_path: Path to the project

    Returns:
        Tuple of (doc_names, doc_rows, scope)
    """
    # Ensure architecture documents directory exists
    docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
    ensure_directory(docs_dir)

    # Find ALL documents in the directory
    all_doc_names = []
    doc_rows = []

    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(('.json', '.md', '.txt', '.yaml', '.xml')):
                doc_name = os.path.splitext(filename)[0]
                if doc_name not in all_doc_names:
                    all_doc_names.append(doc_name)

                    # Check document status
                    file_path = os.path.join(docs_dir, filename)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    status = f"{mod_time.strftime('%Y-%m-%d %H:%M')}"

                    # Check if the document has been analyzed
                    analysis_path = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR, f"{doc_name}.json")
                    if os.path.exists(analysis_path):
                        analysis_time = datetime.fromtimestamp(os.path.getmtime(analysis_path))
                        analysis_status = f"Analyzed {analysis_time.strftime('%Y-%m-%d %H:%M')}"

                        # Check if document was modified after analysis
                        if os.path.getmtime(file_path) > os.path.getmtime(analysis_path):
                            analysis_status += " ⚠️"
                    else:
                        analysis_status = "Not Analyzed"

                    doc_rows.append([doc_name, analysis_status])

    # Try to load existing scope
    scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
    try:
        with open(scope_path, 'r') as f:
            scope_data = json.load(f)
            existing_scope = scope_data.get('documents', [])
    except (FileNotFoundError, json.JSONDecodeError):
        existing_scope = []

    # Validate scope against existing documents
    valid_scope = [doc for doc in existing_scope if doc in all_doc_names]

    # Update scope file to match current state
    try:
        ensure_directory(os.path.dirname(scope_path))
        with open(scope_path, 'w') as f:
            json.dump({
                'documents': valid_scope,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        print(f"Error updating scope file: {e}")

    return all_doc_names, doc_rows, valid_scope


def get_document_content(project_path: str, doc_name: str) -> str:
    """
    Get the content of a document.
    
    Args:
        project_path: Path to the project
        doc_name: Name of the document
        
    Returns:
        Document content as string
    """
    if not project_path or not doc_name:
        return ""

    # Try loading from JSON first
    json_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'content' in data:
                    return data['content']
        except Exception as e:
            print(f"Error loading JSON document {doc_name}: {e}")

    # Fall back to markdown
    md_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.md")
    if os.path.exists(md_path):
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading MD document {doc_name}: {e}")

    # Try other supported formats
    for ext in ['.txt', '.yaml', '.xml']:
        file_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}{ext}")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error loading document {doc_name}{ext}: {e}")

    return f"# {doc_name}\n\nNo content available or unable to load content."


def get_document_content_safe(state: Dict, doc_name: str) -> str:
    """
    Safely get document content with proper state handling.
    
    Args:
        state: Application state dictionary
        doc_name: Name of the document
        
    Returns:
        Document content as string
    """
    if not state.get("current_project"):
        return ""

    try:
        project_path = state["current_project"].metadata.path
        return get_document_content(project_path, doc_name)
    except Exception as e:
        print(f"Error getting document content: {e}")
        return ""


def save_document_content(project_path: str, doc_name: str, content: str) -> bool:
    """
    Save document content to both JSON and MD formats.
    
    Args:
        project_path: Path to the project
        doc_name: Name of the document
        content: Document content
        
    Returns:
        True if successfully saved, False otherwise
    """
    if not project_path or not doc_name or not isinstance(content, str):
        return False

    docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
    ensure_directory(docs_dir)

    # Save as JSON
    json_path = os.path.join(docs_dir, f"{doc_name}.json")
    try:
        document_data = {
            'content': content,
            'last_modified': datetime.now().isoformat()
        }

        with open(json_path, 'w') as f:
            json.dump(document_data, f, indent=2)

        # Also save as markdown
        md_path = os.path.join(docs_dir, f"{doc_name}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error saving document {doc_name}: {e}")
        return False


def update_scope(project_path: str, scope: List[str]) -> bool:
    """
    Update the project scope.
    
    Args:
        project_path: Path to the project
        scope: List of document names in scope
        
    Returns:
        True if successfully updated, False otherwise
    """
    if not project_path:
        return False

    # Ensure scope is a list of strings only
    if not isinstance(scope, list):
        scope = []

    # Clean the scope to ensure it only contains string document names
    clean_scope = []
    for item in scope:
        if isinstance(item, list) and len(item) > 0:
            clean_scope.append(str(item[0]))
        elif item:
            clean_scope.append(str(item))

    scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)

    try:
        scope_data = {
            'documents': clean_scope,
            'last_updated': datetime.now().isoformat()
        }

        ensure_directory(os.path.dirname(scope_path))

        with open(scope_path, 'w') as f:
            json.dump(scope_data, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving scope: {e}")
        return False


def has_document_changed(project_path: str, doc_name: str) -> bool:
    """
    Check if document has changed since last analysis.
    
    Args:
        project_path: Path to the project
        doc_name: Name of the document
        
    Returns:
        True if document has changed since last analysis, False otherwise
    """
    if not project_path or not doc_name:
        return False

    # Get document path
    json_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.json")
    md_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.md")

    # Get document modified time
    doc_mtime = 0
    if os.path.exists(json_path):
        doc_mtime = os.path.getmtime(json_path)
    elif os.path.exists(md_path):
        doc_mtime = os.path.getmtime(md_path)

    # Get analysis path
    analysis_path = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR, f"{doc_name}.json")

    # If no analysis exists, document needs analysis
    if not os.path.exists(analysis_path):
        return True

    # Check if document modified after analysis
    analysis_mtime = os.path.getmtime(analysis_path)
    return doc_mtime > analysis_mtime


def display_diagram(project_path: str, doc_name: str, diagram_type: str) -> str:
    """
    Display a diagram based on type, using Mermaid for all diagram types.

    Args:
        project_path: Path to the project
        doc_name: Name of the document
        diagram_type: Type of diagram to display

    Returns:
        HTML content for diagram display
    """
    if not project_path or not doc_name:
        return NO_DOCUMENT_FIRST_TEMPLATE

    # Get Mermaid diagram HTML by calling controller
    try:
        # Use the controller method to get Mermaid HTML
        html = ArchitectureController.get_mermaid_diagram_html(project_path, doc_name, diagram_type)
        if not html or html.strip() == "":
            return "<p>⚠️ No diagram available. Please generate diagrams first.</p>"
        
        return html
    except Exception as e:
        print(f"Error displaying diagram: {e}")
        import traceback
        traceback.print_exc()
        from vibecheck.constants.architecture_templates import DIAGRAM_ERROR_TEMPLATE
        return DIAGRAM_ERROR_TEMPLATE.format(
            diagram_type=diagram_type,
            error_message=str(e)
        )


# ----- UI state management utilities -----

def create_default_document_content(doc_name: str) -> str:
    """
    Create default content for a new document.
    
    Args:
        doc_name: Name of the document
        
    Returns:
        Default document content
    """
    return DEFAULT_DOCUMENT_TEMPLATE.format(doc_name=doc_name)


def clean_document_selection(selected_docs):
    """
    Clean document selection to ensure proper format.

    Args:
        selected_docs: List of selected documents (potentially mixed formats)

    Returns:
        Cleaned list of document names as strings
    """
    # Handle None or empty input
    if not selected_docs:
        return []

    # If not a list, convert to a list with the single item
    if not isinstance(selected_docs, list):
        if selected_docs:
            return [str(selected_docs)]
        else:
            return []

    # Process each item in the list
    clean_selected_docs = []
    for doc in selected_docs:
        # Skip empty entries
        if not doc:
            continue

        # Handle lists (like row values from dataframe)
        if isinstance(doc, list):
            if doc and doc[0]:  # Check if list has at least one non-empty item
                clean_selected_docs.append(str(doc[0]))
        else:
            # Convert to string for consistency
            clean_selected_docs.append(str(doc))

    # Remove any duplicates while maintaining order
    unique_docs = []
    for doc in clean_selected_docs:
        if doc not in unique_docs:
            unique_docs.append(doc)

    return unique_docs


def format_scope_text(scope: List[str]) -> str:
    """
    Format the scope text for display.
    
    Args:
        scope: List of document names in scope
        
    Returns:
        Formatted scope text
    """
    if not scope:
        return NO_SCOPE

    return f"🔍 **Current Scope:** {', '.join(scope)}"
