"""
Project management controller for VibeCheck - fixed JSON serialization.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from vibecheck.models.project import ProjectMetadata, VibeCheckProject
from vibecheck.models.architecture import ArchitectureData, ArchitecturalDocument
from vibecheck.models.implementation import ImplementationData
from vibecheck.models.security import SecurityAnalysis
from vibecheck.models.environment import EnvironmentData
from vibecheck.models.tests import TestData
from vibecheck.utils.file_utils import ensure_directory
from vibecheck.utils.cache_utils import AnalysisCache
from vibecheck import config


class ProjectController:
    """
    Controller for project management functions including creation, loading, and saving projects.
    """

    @staticmethod
    def create_project(name: str, path: str, description: Optional[str] = None) -> VibeCheckProject:
        """
        Create a new VibeCheck project.

        Args:
            name: Project name
            path: Project root directory path
            description: Optional project description

        Returns:
            A new VibeCheckProject instance
        """
        now = datetime.now()
        
        # Create project metadata
        metadata = ProjectMetadata(
            name=name,
            path=path,
            created_at=now,
            last_modified=now,
            description=description
        )
        
        # Create empty project structure
        architecture_data = ArchitectureData(
            document=ArchitecturalDocument(
                content="",
                last_modified=now
            ),
            diagrams={}
        )
        
        implementation_data = ImplementationData(
            modules={},
            overall_percentage=0.0
        )
        
        environment_data = EnvironmentData(
            dependencies=[],
            virtual_environments=[],
            compilers=[],
            environment_variables=[],
            active_environment=None
        )
        
        test_data = TestData(
            test_suites={}
        )
        
        # Create the project
        project = VibeCheckProject(
            metadata=metadata,
            architecture=architecture_data,
            implementation=implementation_data,
            environment=environment_data,
            tests=test_data
        )
        
        # Create the project directory structure
        ProjectController._create_project_directories(path)
        
        # Save the project
        ProjectController.save_project(project)
        
        return project

    @staticmethod
    def load_project(path: str) -> Optional[VibeCheckProject]:
        """
        Load a VibeCheck project from the given path.

        Args:
            path: Path to the project directory

        Returns:
            Loaded VibeCheckProject or None if not found
        """
        project_file = os.path.join(path, config.PROJECT_FILE)
        
        if not os.path.exists(project_file):
            return None
        
        try:
            with open(project_file, 'r') as f:
                project_data = json.load(f)
                
                # Convert legacy VibeKiller projects if needed
                if "VibeKillerProject" in str(project_data):
                    project_data = ProjectController._convert_legacy_project(project_data)
                
                return VibeCheckProject.parse_obj(project_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading project: {e}")
            return None

    @staticmethod
    def save_project(project: VibeCheckProject) -> bool:
        """
        Save a VibeCheck project to its specified path.

        Args:
            project: The project to save

        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Update last modified time
            project.metadata.last_modified = datetime.now()
            
            # Create the project directory structure if it doesn't exist
            ProjectController._create_project_directories(project.metadata.path)
            
            # Save the project metadata - FIXED JSON SERIALIZATION
            project_file = os.path.join(project.metadata.path, config.PROJECT_FILE)
            
            # Convert to dictionary and then to JSON manually
            project_dict = project.dict()
            
            with open(project_file, 'w') as f:
                json.dump(project_dict, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving project: {e}")
            return False

    @staticmethod
    def select_and_list_directory_contents(dir_path: str, exclude_hidden: bool = True) -> Tuple[str, List[List[str]]]:
        """
        List the contents of a directory, providing file/directory names and their types.

        Args:
            dir_path: The directory path to list
            exclude_hidden: Whether to exclude hidden files and directories

        Returns:
            Tuple containing the directory path and a list of [name, type] entries
        """
        path = Path(dir_path)
        contents = []
        
        # Ensure the path exists
        if not path.exists() or not path.is_dir():
            path = Path.cwd()
        
        # Try to list the directory contents
        try:
            for item in path.iterdir():
                # Skip hidden items if requested
                if exclude_hidden and item.name.startswith('.'):
                    continue
                    
                item_type = "directory" if item.is_dir() else "file"
                contents.append([item.name, item_type])
        except PermissionError:
            # Handle permission error by going to parent directory
            parent = path.parent
            if parent.exists() and parent != path:
                path = parent
                for item in path.iterdir():
                    if exclude_hidden and item.name.startswith('.'):
                        continue
                    item_type = "directory" if item.is_dir() else "file"
                    contents.append([item.name, item_type])
        
        # Sort directories first, then files, both alphabetically
        contents.sort(key=lambda x: (0 if x[1] == "directory" else 1, x[0].lower()))
        
        return str(path.absolute()), contents

    @staticmethod
    def get_project_info(project: VibeCheckProject) -> Dict:
        """
        Get detailed information about a project.

        Args:
            project: The project to get information for

        Returns:
            Dictionary with project information
        """
        info = {
            "name": project.metadata.name,
            "path": project.metadata.path,
            "description": project.metadata.description or "",
            "created_at": str(project.metadata.created_at),
            "last_modified": str(project.metadata.last_modified),
            "architecture": {
                "has_document": bool(project.architecture.document.content),
                "diagrams_count": len(project.architecture.diagrams)
            },
            "implementation": {
                "module_count": len(project.implementation.modules),
                "file_count": sum(len(module.files) for module in project.implementation.modules.values()),
                "overall_percentage": project.implementation.overall_percentage
            },
            "environment": {
                "dependency_count": len(project.environment.dependencies),
                "environment_count": len(project.environment.virtual_environments),
                "active_environment": project.environment.active_environment
            },
            "tests": {
                "suite_count": len(project.tests.test_suites),
                "test_count": sum(len(suite.tests) for suite in project.tests.test_suites.values())
            }
        }
        
        # Add cache information
        cache_info = AnalysisCache.get_cache_status(project.metadata.path)
        info["cache"] = cache_info
        
        return info

    @staticmethod
    def clean_project_cache(project_path: str) -> Dict:
        """
        Clean the project cache.

        Args:
            project_path: Path to the project

        Returns:
            Dictionary with cleanup results
        """
        return AnalysisCache.cleanup_cache(project_path)

    @staticmethod
    def _create_project_directories(project_path: str) -> None:
        """
        Create the directory structure for a VibeCheck project.

        Args:
            project_path: Path to the project
        """
        # Create the main .vibecheck directory
        ensure_directory(os.path.join(project_path, config.VIBECHECK_DIR))
        
        # Create subdirectories
        ensure_directory(os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR))
        ensure_directory(os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR))
        ensure_directory(os.path.join(project_path, config.IMPLEMENTATION_DIR))
        ensure_directory(os.path.join(project_path, config.IMPLEMENTATION_SECURITY_DIR))
        ensure_directory(os.path.join(project_path, config.ENVIRONMENT_DIR))
        ensure_directory(os.path.join(project_path, config.TESTS_DIR))
        ensure_directory(os.path.join(project_path, config.CACHE_DIR))
    
    @staticmethod
    def _convert_legacy_project(project_data: dict) -> dict:
        """
        Convert a legacy VibeKiller project to a VibeCheck project.

        Args:
            project_data: The legacy project data

        Returns:
            Updated project data
        """
        # Simple conversion by renaming the project type
        if isinstance(project_data, dict) and '__root__' not in project_data:
            # For newer Pydantic versions
            if 'VibeKillerProject' in str(project_data):
                project_data_str = json.dumps(project_data)
                project_data_str = project_data_str.replace('VibeKillerProject', 'VibeCheckProject')
                project_data_str = project_data_str.replace('VibeKiller', 'VibeCheck')
                project_data_str = project_data_str.replace('vibekill', 'vibecheck')
                project_data = json.loads(project_data_str)
        
        return project_data
