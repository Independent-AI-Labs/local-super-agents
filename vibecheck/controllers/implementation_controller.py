"""
Implementation tracking controller for VibeCheck.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vibecheck.models.implementation import FileStatus, ModuleStatus, ImplementationData
from vibecheck.models.architecture import ArchitecturalDiagram
from vibecheck.integrations.git import get_file_status, get_file_diff
from vibecheck.integrations.llm import analyze_with_llm
from vibecheck.utils.file_utils import list_files, read_file, is_file_empty, write_file
from vibecheck.utils.cache_utils import AnalysisCache, FileModificationTracker
from vibecheck import config


class ImplementationController:
    """
    Controller for implementation tracking functions including git integration
    and progress monitoring.
    """

    @staticmethod
    def analyze_implementation(project_path: str, architectural_diagrams: Optional[Dict[str, ArchitecturalDiagram]] = None) -> ImplementationData:
        """
        Analyze the implementation status of a project.

        Args:
            project_path: Path to the project
            architectural_diagrams: Optional dictionary of architectural diagrams

        Returns:
            ImplementationData with analysis results
        """
        # Check cache first
        cached_data = AnalysisCache.get_cached_analysis(project_path, "implementation", "status")
        
        if cached_data:
            try:
                return ImplementationData.parse_obj(cached_data)
            except Exception as e:
                print(f"Error parsing cached implementation data: {e}")
        
        # Get file statuses from git
        git_statuses = get_file_status(project_path)
        
        # If git is not available, fall back to scanning the directory
        if not git_statuses:
            file_paths = list_files(
                project_path,
                include_extensions=config.DEFAULT_INCLUDE_EXTENSIONS,
                exclude_dirs=config.DEFAULT_EXCLUDE_DIRS + [config.VIBECHECK_DIR]
            )
            git_statuses = {file_path: "unchanged" for file_path in file_paths}
        
        # Organize files into modules
        modules: Dict[str, List[FileStatus]] = {}
        now = datetime.now()
        
        for file_path, status in git_statuses.items():
            # Skip files in excluded directories
            if any(excluded in file_path for excluded in config.DEFAULT_EXCLUDE_DIRS):
                continue
            
            # Skip files with excluded extensions
            if not any(file_path.endswith(ext) for ext in config.DEFAULT_INCLUDE_EXTENSIONS):
                continue
                
            # Extract the module name from the file path (first directory)
            module_name = file_path.split('/')[0] if '/' in file_path else 'root'
            
            # Create module entry if it doesn't exist
            if module_name not in modules:
                modules[module_name] = []
            
            # Get file diff for modified files
            diff = None
            if status == 'modified':
                diff = get_file_diff(project_path, file_path)
            
            # Calculate implementation percentage
            impl_percentage = ImplementationController.calculate_implementation_percentage(
                os.path.join(project_path, file_path),
                architectural_diagrams
            )
            
            # Create file status
            file_status = FileStatus(
                path=file_path,
                status=status,
                diff=diff,
                implementation_percentage=impl_percentage,
                last_analyzed=now
            )
            
            modules[module_name].append(file_status)
        
        # Create module statuses
        module_statuses: Dict[str, ModuleStatus] = {}
        overall_percentage = 0.0
        total_files = 0
        
        for module_name, files in modules.items():
            # Calculate module implementation percentage (average of files)
            total_files += len(files)
            module_percentage = sum(file.implementation_percentage for file in files) / len(files) if files else 0.0
            
            module_status = ModuleStatus(
                name=module_name,
                files=files,
                implementation_percentage=module_percentage
            )
            
            module_statuses[module_name] = module_status
            overall_percentage += module_percentage * len(files)
        
        # Calculate overall implementation percentage
        if total_files > 0:
            overall_percentage /= total_files
        
        # Create implementation data
        implementation_data = ImplementationData(
            modules=module_statuses,
            overall_percentage=overall_percentage
        )
        
        # Save implementation data
        ImplementationController._save_implementation_data(project_path, implementation_data)
        
        # Cache the implementation data
        AnalysisCache.cache_analysis(
            project_path,
            "implementation",
            "status",
            json.loads(implementation_data.json()),
            ttl_seconds=3600  # 1 hour
        )
        
        return implementation_data

    @staticmethod
    def calculate_implementation_percentage(file_path: str, architectural_diagrams: Optional[Dict[str, ArchitecturalDiagram]] = None) -> float:
        """
        Calculate the implementation percentage of a file based on architectural diagrams.

        Args:
            file_path: Path to the file
            architectural_diagrams: Optional dictionary of architectural diagrams

        Returns:
            Implementation percentage (0.0 to 100.0)
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            return 0.0
        
        # If the file is empty, it's not implemented
        if is_file_empty(file_path):
            return 0.0
        
        # Read the file content
        content = read_file(file_path)
        if not content:
            return 0.0
        
        # Check cache for this file
        file_hash = FileModificationTracker.get_file_hash(file_path)
        project_path = os.path.dirname(file_path)
        file_rel_path = os.path.relpath(file_path, project_path)
        
        cached_percentage = AnalysisCache.get_cached_analysis(
            project_path, file_rel_path, "implementation_percentage"
        )
        
        if cached_percentage is not None and isinstance(cached_percentage, (int, float)):
            # Check if the file has changed since the cache was created
            if not FileModificationTracker.has_file_changed(file_path, previous_hash=file_hash):
                return float(cached_percentage)
        
        # Count lines (excluding empty lines)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        if len(lines) == 0:
            return 0.0
        
        # Count TODO/FIXME comments and other indicators of incomplete code
        todo_indicators = [
            "TODO", "FIXME", "XXX", "HACK", "BUG", 
            "to be implemented", "not implemented", "to do",
            "needs implementation", "implement this", "implement later"
        ]
        
        # Count lines with indicators (case insensitive)
        todo_count = 0
        for line in lines:
            if any(indicator.lower() in line.lower() for indicator in todo_indicators):
                todo_count += 1
        
        # Calculate percentage based on TODO/FIXME count
        todo_factor = todo_count / len(lines)
        implementation_percentage = 100.0 * (1.0 - min(todo_factor, 0.5) * 2)
        
        # Adjust based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Give higher weight to certain file types that are typically
        # more critical to the implementation
        if file_ext in ['.py', '.java', '.cpp', '.c', '.go', '.rs']:
            # Core implementation languages typically have higher weight
            pass  # default is fine
        elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
            # Frontend/script languages might have slightly lower weight
            implementation_percentage = min(95.0, implementation_percentage)
        elif file_ext in ['.html', '.css', '.scss', '.less']:
            # UI/styling files typically have lower weight
            implementation_percentage = min(90.0, implementation_percentage)
        elif file_ext in ['.md', '.txt', '.rst']:
            # Documentation files typically have lower weight
            implementation_percentage = min(80.0, implementation_percentage)
        
        # Cap at 100%
        implementation_percentage = min(100.0, max(0.0, implementation_percentage))
        
        # Cache the result
        AnalysisCache.cache_analysis(
            project_path,
            file_rel_path,
            "implementation_percentage",
            implementation_percentage,
            ttl_seconds=86400  # 24 hours
        )
        
        return implementation_percentage

    @staticmethod
    def analyze_file_content(file_path: str) -> Dict[str, any]:
        """
        Analyze a file's content for implementation insights.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(file_path):
            return {"error": "File does not exist"}
        
        # Read file content
        content = read_file(file_path)
        if not content:
            return {"error": "Failed to read file content"}
        
        # Check cache for this file
        file_hash = FileModificationTracker.get_file_hash(file_path)
        project_path = os.path.dirname(file_path)
        file_rel_path = os.path.relpath(file_path, project_path)
        
        cached_analysis = AnalysisCache.get_cached_analysis(
            project_path, file_rel_path, "content_analysis"
        )
        
        if cached_analysis is not None and isinstance(cached_analysis, dict):
            # Check if the file has changed since the cache was created
            if not FileModificationTracker.has_file_changed(file_path, previous_hash=file_hash):
                return cached_analysis
        
        # Basic analysis
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Look for TODOs, imports, functions, classes, etc.
        todos = []
        imports = []
        functions = []
        classes = []
        
        for i, line in enumerate(lines):
            line_num = i + 1
            if any(indicator in line for indicator in ["TODO", "FIXME", "XXX", "HACK", "BUG"]):
                todos.append({"line": line_num, "content": line.strip()})
            
            if "import " in line or "from " in line and " import " in line:
                imports.append({"line": line_num, "content": line.strip()})
            
            if "def " in line:
                functions.append({"line": line_num, "name": line.split("def ")[1].split("(")[0].strip()})
            
            if "class " in line:
                classes.append({"line": line_num, "name": line.split("class ")[1].split("(")[0].split(":")[0].strip()})
        
        # Calculate metrics
        loc = len(lines)
        loc_no_empty = len(non_empty_lines)
        todo_count = len(todos)
        import_count = len(imports)
        function_count = len(functions)
        class_count = len(classes)
        
        # Prepare result
        result = {
            "metrics": {
                "loc": loc,
                "loc_no_empty": loc_no_empty,
                "todo_count": todo_count,
                "import_count": import_count,
                "function_count": function_count,
                "class_count": class_count
            },
            "details": {
                "todos": todos,
                "imports": imports,
                "functions": functions,
                "classes": classes
            }
        }
        
        # Cache the result
        AnalysisCache.cache_analysis(
            project_path,
            file_rel_path,
            "content_analysis",
            result,
            ttl_seconds=86400  # 24 hours
        )
        
        return result

    @staticmethod
    def load_file_content(project_path: str, file_path: str) -> str:
        """
        Load a file's content.

        Args:
            project_path: Path to the project
            file_path: Path to the file, relative to the project root

        Returns:
            File content
        """
        abs_path = os.path.join(project_path, file_path)
        return read_file(abs_path) or ""

    @staticmethod
    def save_file_content(project_path: str, file_path: str, content: str) -> bool:
        """
        Save a file's content.

        Args:
            project_path: Path to the project
            file_path: Path to the file, relative to the project root
            content: Content to save

        Returns:
            True if successful, False otherwise
        """
        abs_path = os.path.join(project_path, file_path)
        
        # Save the file
        if write_file(abs_path, content):
            # Invalidate cache for this file
            AnalysisCache.invalidate_cache(project_path, file_path)
            return True
        
        return False

    @staticmethod
    def get_implementation_metrics(project_path: str) -> Dict[str, any]:
        """
        Get implementation metrics for the project.

        Args:
            project_path: Path to the project

        Returns:
            Dictionary with implementation metrics
        """
        # Get implementation data
        implementation_data = ImplementationController.load_implementation_data(project_path)
        if not implementation_data:
            implementation_data = ImplementationController.analyze_implementation(project_path)
        
        # Calculate metrics
        total_files = 0
        completed_files = 0
        in_progress_files = 0
        not_started_files = 0
        
        for module_name, module in implementation_data.modules.items():
            for file in module.files:
                total_files += 1
                
                if file.implementation_percentage >= 90:
                    completed_files += 1
                elif file.implementation_percentage > 0:
                    in_progress_files += 1
                else:
                    not_started_files += 1
        
        return {
            "total_files": total_files,
            "completed_files": completed_files,
            "in_progress_files": in_progress_files,
            "not_started_files": not_started_files,
            "overall_percentage": implementation_data.overall_percentage,
            "module_count": len(implementation_data.modules)
        }

    @staticmethod
    def _save_implementation_data(project_path: str, implementation_data: ImplementationData) -> None:
        """
        Save implementation data to the project.

        Args:
            project_path: Path to the project
            implementation_data: Implementation data to save
        """
        # Ensure the directory exists
        implementation_dir = os.path.join(project_path, config.IMPLEMENTATION_DIR)
        os.makedirs(implementation_dir, exist_ok=True)
        
        # Save the data
        implementation_path = os.path.join(
            project_path, 
            config.IMPLEMENTATION_PROGRESS_FILE
        )
        
        with open(implementation_path, 'w') as f:
            f.write(implementation_data.json(indent=2))

    @staticmethod
    def load_implementation_data(project_path: str) -> Optional[ImplementationData]:
        """
        Load implementation data from the project.

        Args:
            project_path: Path to the project

        Returns:
            ImplementationData or None if not found
        """
        implementation_path = os.path.join(
            project_path, 
            config.IMPLEMENTATION_PROGRESS_FILE
        )
        
        if not os.path.exists(implementation_path):
            return None
        
        try:
            with open(implementation_path, 'r') as f:
                data = json.load(f)
                return ImplementationData.parse_obj(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading implementation data: {e}")
            return None
