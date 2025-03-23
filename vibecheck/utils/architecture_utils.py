"""
Architecture utilities for VibeCheck.

This module provides utility functions for architecture document management,
file monitoring, and scope management.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from vibecheck.utils.file_utils import ensure_directory, list_files, read_file, write_file
from vibecheck import config


class ArchitectureDocumentManager:
    """Manages architecture documents for the project."""
    
    @staticmethod
    def list_documents(project_path: str) -> List[Dict[str, any]]:
        """
        List all architecture documents in the project.
        
        Args:
            project_path: Path to the project
            
        Returns:
            List of documents with metadata
        """
        docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
        
        if not os.path.exists(docs_dir):
            return []
        
        documents = []
        
        for filename in os.listdir(docs_dir):
            if filename.endswith(('.json', '.md', '.txt', '.yaml', '.xml')):
                file_path = os.path.join(docs_dir, filename)
                
                # Check if it's a properly formatted JSON document
                is_json_doc = False
                if filename.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if 'content' in data:
                                is_json_doc = True
                    except:
                        pass
                
                # Get document metadata
                doc_name = os.path.splitext(filename)[0]
                mod_time = os.path.getmtime(file_path)
                mod_date = datetime.fromtimestamp(mod_time)
                
                # Check if the document has been analyzed
                has_been_analyzed = ArchitectureDocumentManager._has_been_analyzed(project_path, doc_name)
                
                # Check if the document has changed since last analysis
                has_changed = ArchitectureDocumentManager._has_changed_since_analysis(project_path, doc_name)
                
                documents.append({
                    'name': doc_name,
                    'filename': filename,
                    'last_modified': mod_date,
                    'is_json_doc': is_json_doc,
                    'has_been_analyzed': has_been_analyzed,
                    'has_changed': has_changed
                })
        
        # Sort by last modified time (newest first)
        documents.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return documents
    
    @staticmethod
    def create_document(project_path: str, name: str, content: str = "") -> bool:
        """
        Create a new architecture document.
        
        Args:
            project_path: Path to the project
            name: Name of the document
            content: Optional initial content
            
        Returns:
            True if successful, False otherwise
        """
        docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
        ensure_directory(docs_dir)
        
        # Create default content if none provided
        if not content:
            content = f"""# {name} Architecture Document

## System Overview

Describe your system here...

## Components

- Component 1: Description of component 1
- Component 2: Description of component 2

## Relationships

- Component 1 communicates with Component 2
"""
        
        # Create JSON document
        json_path = os.path.join(docs_dir, f"{name}.json")
        
        # Create the document model
        document_data = {
            'content': content,
            'last_modified': datetime.now().isoformat()
        }
        
        # Save as JSON
        try:
            with open(json_path, 'w') as f:
                json.dump(document_data, f, indent=2)
            
            # Also save as markdown for easier viewing
            md_path = os.path.join(docs_dir, f"{name}.md")
            write_file(md_path, content)
            
            return True
        except Exception as e:
            print(f"Error creating architecture document: {e}")
            return False
    
    @staticmethod
    def delete_documents(project_path: str, doc_names: List[str]) -> bool:
        """
        Delete architecture documents.
        
        Args:
            project_path: Path to the project
            doc_names: Names of documents to delete
            
        Returns:
            True if all deletions were successful, False otherwise
        """
        docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
        
        if not os.path.exists(docs_dir):
            return False
        
        success = True
        
        for doc_name in doc_names:
            # Delete both JSON and markdown versions if they exist
            json_path = os.path.join(docs_dir, f"{doc_name}.json")
            md_path = os.path.join(docs_dir, f"{doc_name}.md")
            
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                except Exception as e:
                    print(f"Error deleting {json_path}: {e}")
                    success = False
            
            if os.path.exists(md_path):
                try:
                    os.remove(md_path)
                except Exception as e:
                    print(f"Error deleting {md_path}: {e}")
                    success = False
            
            # Delete any associated diagrams and analysis
            ArchitectureDocumentManager._delete_associated_files(project_path, doc_name)
        
        return success
    
    @staticmethod
    def load_document_content(project_path: str, doc_name: str) -> Optional[str]:
        """
        Load the content of an architecture document.
        
        Args:
            project_path: Path to the project
            doc_name: Name of the document
            
        Returns:
            Document content or None if not found
        """
        docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
        
        # Try loading from JSON first
        json_path = os.path.join(docs_dir, f"{doc_name}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'content' in data:
                        return data['content']
            except Exception as e:
                print(f"Error loading JSON document {doc_name}: {e}")
        
        # Fall back to markdown
        md_path = os.path.join(docs_dir, f"{doc_name}.md")
        if os.path.exists(md_path):
            content = read_file(md_path)
            return content
        
        # Try other supported formats
        for ext in ['.txt', '.yaml', '.xml']:
            file_path = os.path.join(docs_dir, f"{doc_name}{ext}")
            if os.path.exists(file_path):
                content = read_file(file_path)
                return content
        
        return None
    
    @staticmethod
    def save_document_content(project_path: str, doc_name: str, content: str) -> bool:
        """
        Save the content of an architecture document.
        
        Args:
            project_path: Path to the project
            doc_name: Name of the document
            content: Document content
            
        Returns:
            True if successful, False otherwise
        """
        docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
        ensure_directory(docs_dir)
        
        # Update JSON document
        json_path = os.path.join(docs_dir, f"{doc_name}.json")
        
        document_data = {
            'content': content,
            'last_modified': datetime.now().isoformat()
        }
        
        try:
            with open(json_path, 'w') as f:
                json.dump(document_data, f, indent=2)
            
            # Also update markdown for easier viewing
            md_path = os.path.join(docs_dir, f"{doc_name}.md")
            write_file(md_path, content)
            
            # Mark the document as changed
            ArchitectureDocumentManager._mark_as_changed(project_path, doc_name)
            
            return True
        except Exception as e:
            print(f"Error saving architecture document: {e}")
            return False
    
    @staticmethod
    def upload_document(project_path: str, file_path: str) -> Optional[str]:
        """
        Upload an architecture document from a file.
        
        Args:
            project_path: Path to the project
            file_path: Path to the file to upload
            
        Returns:
            Document name if successful, None otherwise
        """
        if not os.path.exists(file_path):
            return None
        
        # Get document content
        content = read_file(file_path)
        if not content:
            return None
        
        # Extract filename without extension
        file_name = os.path.basename(file_path)
        doc_name = os.path.splitext(file_name)[0]
        
        # Create the document
        if ArchitectureDocumentManager.create_document(project_path, doc_name, content):
            return doc_name
        
        return None
    
    @staticmethod
    def get_scope(project_path: str) -> List[str]:
        """
        Get the current architecture scope (selected documents).
        
        Args:
            project_path: Path to the project
            
        Returns:
            List of document names in the scope
        """
        scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
        
        if not os.path.exists(scope_path):
            return []
        
        try:
            with open(scope_path, 'r') as f:
                data = json.load(f)
                return data.get('documents', [])
        except Exception as e:
            print(f"Error loading architecture scope: {e}")
            return []
    
    @staticmethod
    def set_scope(project_path: str, doc_names: List[str]) -> bool:
        """
        Set the architecture scope (selected documents).
        
        Args:
            project_path: Path to the project
            doc_names: Names of documents to include in the scope
            
        Returns:
            True if successful, False otherwise
        """
        scope_path = os.path.join(project_path, config.ARCHITECTURE_SCOPE_FILE)
        
        try:
            scope_data = {
                'documents': doc_names,
                'last_updated': datetime.now().isoformat()
            }
            
            ensure_directory(os.path.dirname(scope_path))
            
            with open(scope_path, 'w') as f:
                json.dump(scope_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving architecture scope: {e}")
            return False
    
    @staticmethod
    def _has_been_analyzed(project_path: str, doc_name: str) -> bool:
        """Check if a document has been analyzed."""
        analysis_path = os.path.join(
            project_path, 
            config.ARCHITECTURE_ANALYSIS_DIR, 
            f"{doc_name}.json"
        )
        return os.path.exists(analysis_path)
    
    @staticmethod
    def _has_changed_since_analysis(project_path: str, doc_name: str) -> bool:
        """Check if a document has changed since last analysis."""
        # Get document modified time
        docs_dir = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR)
        json_path = os.path.join(docs_dir, f"{doc_name}.json")
        md_path = os.path.join(docs_dir, f"{doc_name}.md")
        
        doc_mtime = 0
        if os.path.exists(json_path):
            doc_mtime = os.path.getmtime(json_path)
        elif os.path.exists(md_path):
            doc_mtime = os.path.getmtime(md_path)
        
        # Get analysis modified time
        analysis_path = os.path.join(
            project_path, 
            config.ARCHITECTURE_ANALYSIS_DIR, 
            f"{doc_name}.json"
        )
        
        if not os.path.exists(analysis_path):
            return True  # Never analyzed
        
        analysis_mtime = os.path.getmtime(analysis_path)
        
        # Check if the document has been modified since the analysis
        return doc_mtime > analysis_mtime
    
    @staticmethod
    def _mark_as_changed(project_path: str, doc_name: str) -> None:
        """Mark a document as changed (modified since last analysis)."""
        # We simply update the document modified time, which will be compared with
        # the analysis modified time in _has_changed_since_analysis
        pass
    
    @staticmethod
    def _delete_associated_files(project_path: str, doc_name: str) -> None:
        """Delete diagrams and analysis associated with a document."""
        # Delete diagrams
        diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
        if os.path.exists(diagrams_dir):
            for diagram_type in ['module', 'dataflow', 'security', 'mermaid']:
                json_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.json")
                svg_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.svg")
                
                if os.path.exists(json_path):
                    try:
                        os.remove(json_path)
                    except Exception as e:
                        print(f"Error deleting {json_path}: {e}")
                
                if os.path.exists(svg_path):
                    try:
                        os.remove(svg_path)
                    except Exception as e:
                        print(f"Error deleting {svg_path}: {e}")
        
        # Delete analysis
        analysis_dir = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR)
        if os.path.exists(analysis_dir):
            analysis_path = os.path.join(analysis_dir, f"{doc_name}.json")
            
            if os.path.exists(analysis_path):
                try:
                    os.remove(analysis_path)
                except Exception as e:
                    print(f"Error deleting {analysis_path}: {e}")


class FileChangeTracker:
    """Tracks file changes in the project."""
    
    _last_scan_time = 0
    _file_mtimes: Dict[str, float] = {}
    _callbacks: Dict[str, list] = {}
    
    @staticmethod
    def register_callback(file_path: str, callback) -> None:
        """
        Register a callback for when a file changes.
        
        Args:
            file_path: Path to the file to monitor
            callback: Function to call when the file changes
        """
        if file_path not in FileChangeTracker._callbacks:
            FileChangeTracker._callbacks[file_path] = []
        
        FileChangeTracker._callbacks[file_path].append(callback)
    
    @staticmethod
    def scan_for_changes(project_path: str) -> List[str]:
        """
        Scan the project for file changes.
        
        Args:
            project_path: Path to the project
            
        Returns:
            List of changed file paths
        """
        # Don't scan too frequently
        current_time = time.time()
        if current_time - FileChangeTracker._last_scan_time < 1.0:  # 1 second minimum between scans
            return []
        
        FileChangeTracker._last_scan_time = current_time
        
        # Get all files in the project
        all_files = list_files(
            project_path,
            exclude_dirs=config.DEFAULT_EXCLUDE_DIRS + [config.VIBECHECK_DIR]
        )
        
        changed_files = []
        
        # Check for changes in existing files
        for file_path in all_files:
            abs_path = os.path.join(project_path, file_path)
            
            if not os.path.exists(abs_path):
                continue
                
            current_mtime = os.path.getmtime(abs_path)
            
            if file_path in FileChangeTracker._file_mtimes:
                # File exists in our records, check if it's changed
                if current_mtime > FileChangeTracker._file_mtimes[file_path]:
                    changed_files.append(file_path)
                    FileChangeTracker._file_mtimes[file_path] = current_mtime
                    
                    # Trigger callbacks
                    FileChangeTracker._trigger_callbacks(file_path)
            else:
                # New file, add to our records
                FileChangeTracker._file_mtimes[file_path] = current_mtime
        
        # Check for deleted files
        for file_path in list(FileChangeTracker._file_mtimes.keys()):
            abs_path = os.path.join(project_path, file_path)
            
            if not os.path.exists(abs_path):
                # File has been deleted
                changed_files.append(file_path)
                del FileChangeTracker._file_mtimes[file_path]
                
                # Trigger callbacks
                FileChangeTracker._trigger_callbacks(file_path)
        
        return changed_files
    
    @staticmethod
    def _trigger_callbacks(file_path: str) -> None:
        """Trigger callbacks for a changed file."""
        if file_path in FileChangeTracker._callbacks:
            for callback in FileChangeTracker._callbacks[file_path]:
                try:
                    callback(file_path)
                except Exception as e:
                    print(f"Error in file change callback: {e}")


# Make sure these directories are defined in config.py
if not hasattr(config, 'ARCHITECTURE_ANALYSIS_DIR'):
    config.ARCHITECTURE_ANALYSIS_DIR = os.path.join(config.VIBECHECK_DIR, 'architecture', 'analysis')

if not hasattr(config, 'ARCHITECTURE_SCOPE_FILE'):
    config.ARCHITECTURE_SCOPE_FILE = os.path.join(config.VIBECHECK_DIR, 'architecture', 'scope.json')
