"""
Constants module for the VibeCheck architecture module.
Contains all strings, prompts, and messages used in the architecture UI.
"""

# UI Labels and Titles
TAB_TITLE = "🏗️ Architecture & Scope"
TAB_DESCRIPTION = """
# 🏗️ Software Architecture & Scope

Define your software architecture before implementation and select which 
architecture documents are part of your current scope. This helps enforce
proper software engineering practices and provides a clear blueprint for 
your development.
"""

DOCUMENTS_HEADER = "### 📚 Architecture Documents"
DOCUMENT_LABEL = "Documents"
DOCUMENT_INFO = "Select documents to include in project scope"
DOCUMENT_HEADERS = ["📋 Document", "📅 Modified", "🔍 Status"]
UPLOAD_INFO = "*Supports .json, .yaml, .xml, .txt, .md*"

# Button Labels
NEW_DOC_BTN = "📄 New Document"
UPLOAD_DOC_BTN = "📤 Upload Document"
DELETE_BTN = "🗑️ Delete Selected"
EDIT_BTN = "✏️ Edit"
SAVE_BTN = "💾 Save"
CANCEL_BTN = "❌ Cancel"
ANALYZE_BTN = "🔍 Analyze & Generate Diagrams"
RUN_CRITICAL_ANALYSIS_BTN = "🔍 Run Critical Analysis"
GENERATE_DIAGRAMS_BTN = "📊 Generate Diagrams"

# Tab Labels
DOCUMENT_TAB = "📝 Document"
DIAGRAMS_TAB = "📊 Diagrams"
ANALYSIS_TAB = "🔍 Analysis"

# Diagram Types
DIAGRAM_TYPES = [
    ("Module Diagram", "module"),
    ("Data Flow Diagram", "dataflow"),
    ("Security Diagram", "security"),
    ("Mermaid Diagram", "mermaid")
]
DIAGRAM_TYPE_LABEL = "📊 Diagram Type"

# Table Headers
COMPONENTS_TABLE_HEADERS = ["📦 Component", "📝 Description"]
RELATIONSHIPS_TABLE_HEADERS = ["📦 Source", "🔄 Relationship", "📦 Target"]

# Messages and Prompts
NO_DOCUMENT_SELECTED = "### Select a document from the list"
NO_DOCUMENT_CONTENT = "Select a document from the list to view its content."
NO_DIAGRAM_SELECTED = "<p>Select a document and diagram type to view.</p>"
NO_ANALYSIS_SELECTED = "Select a document and click 'Run Critical Analysis' to get insights."
NO_PROJECT_OPEN = "<p>No project is currently open</p>"
NO_DOCUMENT_FIRST = "<p>Select a document first</p>"
EMPTY_DOCUMENT_ERROR = "⚠️ Document is empty or could not be loaded."
ANALYSIS_FAILED = "❌ Analysis failed. Try again or check document content."
ANALYSIS_LOADING = "Performing critical analysis..."
GENERATING_DIAGRAMS = "<p>Generating diagrams... please wait</p>"
NO_SCOPE = "🔍 **Current Scope:** No documents selected"
SCOPE_ERROR = "🔍 **Current Scope:** Error loading documents"

# Warning Messages
DOCUMENT_CHANGED_WARNING = """
⚠️ **Document has been modified since last analysis.**  
You should analyze it again to update diagrams and analysis.
"""

# Analysis Prompt Enhancement
ANALYSIS_FOCUS_APPENDIX = "\n\n## Analysis Focus: Code Quality\nPlease provide a critical analysis focused on improving code quality."

# Default Document Content
DEFAULT_DOCUMENT_TEMPLATE = """# {doc_name}

## System Overview

Describe your system here...

## Components

- Component 1: Description of component 1
- Component 2: Description of component 2

## Relationships

- Component 1 communicates with Component 2
"""

# Error Messages
ERROR_LOADING_DOCUMENTS = "Error loading documents: {error}"
ERROR_READING_FILE = "Error reading file: {error}"
ERROR_DELETING_DOCUMENT = "Error deleting {path}: {error}"
ERROR_DOCUMENT_SELECTION = "Error in document selection: {error}"
ERROR_GENERATING_DIAGRAMS = "Error generating diagrams: {error}"
ERROR_ANALYZING_DOCUMENT = "Error analyzing document: {error}"
ERROR_LOADING_ANALYSIS = "Error loading existing analysis: {error}"
