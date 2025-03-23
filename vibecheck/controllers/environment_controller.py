"""
Environment management controller for VibeCheck.
"""

import json
import os
import subprocess
from typing import List, Optional, Tuple, Dict

from vibecheck.models.environment import (
    Dependency,
    VirtualEnvironment,
    Compiler,
    EnvironmentVariable,
    EnvironmentData
)
from vibecheck.integrations.web import check_dependency_security
from vibecheck.utils.file_utils import ensure_directory, read_file, write_file
from vibecheck.utils.cache_utils import AnalysisCache
from vibecheck import config


class EnvironmentController:
    """
    Controller for environment management functions including dependencies,
    virtual environments, and compilers.
    """

    @staticmethod
    def add_dependency(project_path: str, name: str, version: str, is_dev: bool = False) -> bool:
        """
        Add a dependency to the project.

        Args:
            project_path: Path to the project
            name: Name of the dependency
            version: Version of the dependency
            is_dev: Whether this is a development dependency

        Returns:
            True if the dependency was added successfully, False otherwise
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            # Create a new environment data object if none exists
            env_data = EnvironmentData(
                dependencies=[],
                virtual_environments=[],
                compilers=[],
                environment_variables=[],
                active_environment=None
            )
        
        # Check if the dependency already exists
        for dep in env_data.dependencies:
            if dep.name == name:
                # Update the version if it already exists
                dep.version = version
                dep.is_dev_dependency = is_dev
                return EnvironmentController._save_environment_data(project_path, env_data)
        
        # Add the new dependency
        new_dependency = Dependency(
            name=name,
            version=version,
            is_dev_dependency=is_dev
        )
        
        env_data.dependencies.append(new_dependency)
        
        # Save the updated environment data
        return EnvironmentController._save_environment_data(project_path, env_data)

    @staticmethod
    def remove_dependency(project_path: str, name: str) -> bool:
        """
        Remove a dependency from the project.

        Args:
            project_path: Path to the project
            name: Name of the dependency to remove

        Returns:
            True if the dependency was removed successfully, False otherwise
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            return False
        
        # Remove the dependency if it exists
        original_len = len(env_data.dependencies)
        env_data.dependencies = [dep for dep in env_data.dependencies if dep.name != name]
        
        # If no dependencies were removed, return False
        if len(env_data.dependencies) == original_len:
            return False
        
        # Save the updated environment data
        return EnvironmentController._save_environment_data(project_path, env_data)

    @staticmethod
    def configure_virtual_env(project_path: str, name: str, env_type: str, python_version: str) -> Optional[VirtualEnvironment]:
        """
        Configure a virtual environment for the project.

        Args:
            project_path: Path to the project
            name: Name of the virtual environment
            env_type: Type of virtual environment ("conda", "venv", "poetry", etc.)
            python_version: Python version to use

        Returns:
            The created VirtualEnvironment or None if there was an error
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            # Create a new environment data object if none exists
            env_data = EnvironmentData(
                dependencies=[],
                virtual_environments=[],
                compilers=[],
                environment_variables=[],
                active_environment=None
            )
        
        # Check if the virtual environment already exists
        for env in env_data.virtual_environments:
            if env.name == name:
                # Update the existing environment
                env.type = env_type
                env.python_version = python_version
                
                if EnvironmentController._save_environment_data(project_path, env_data):
                    return env
                return None
        
        # Determine the path for the virtual environment
        venv_path = os.path.join(project_path, ".venv")
        if env_type == "conda":
            venv_path = os.path.join(project_path, "conda_env")
        
        # Create the new virtual environment
        new_env = VirtualEnvironment(
            name=name,
            type=env_type,
            path=venv_path,
            python_version=python_version
        )
        
        env_data.virtual_environments.append(new_env)
        
        # Set as active environment if none is set
        if not env_data.active_environment:
            env_data.active_environment = name
        
        # Save the updated environment data
        if EnvironmentController._save_environment_data(project_path, env_data):
            return new_env
        return None

    @staticmethod
    def add_compiler(project_path: str, name: str, version: str, path: str) -> bool:
        """
        Add a compiler configuration to the project.

        Args:
            project_path: Path to the project
            name: Name of the compiler
            version: Version of the compiler
            path: Path to the compiler executable

        Returns:
            True if the compiler was added successfully, False otherwise
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            # Create a new environment data object if none exists
            env_data = EnvironmentData(
                dependencies=[],
                virtual_environments=[],
                compilers=[],
                environment_variables=[],
                active_environment=None
            )
        
        # Check if the compiler already exists
        for comp in env_data.compilers:
            if comp.name == name:
                # Update the existing compiler
                comp.version = version
                comp.path = path
                return EnvironmentController._save_environment_data(project_path, env_data)
        
        # Add the new compiler
        new_compiler = Compiler(
            name=name,
            version=version,
            path=path
        )
        
        env_data.compilers.append(new_compiler)
        
        # Save the updated environment data
        return EnvironmentController._save_environment_data(project_path, env_data)

    @staticmethod
    def add_environment_variable(
        project_path: str, 
        name: str, 
        value: str, 
        description: Optional[str] = None,
        is_secret: bool = False
    ) -> bool:
        """
        Add an environment variable to the project.

        Args:
            project_path: Path to the project
            name: Name of the environment variable
            value: Value of the environment variable
            description: Optional description of the environment variable
            is_secret: Whether this is a secret environment variable

        Returns:
            True if the environment variable was added successfully, False otherwise
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            # Create a new environment data object if none exists
            env_data = EnvironmentData(
                dependencies=[],
                virtual_environments=[],
                compilers=[],
                environment_variables=[],
                active_environment=None
            )
        
        # Check if the environment variable already exists
        for env_var in env_data.environment_variables:
            if env_var.name == name:
                # Update the existing environment variable
                env_var.value = value
                env_var.description = description
                env_var.is_secret = is_secret
                return EnvironmentController._save_environment_data(project_path, env_data)
        
        # Add the new environment variable
        new_env_var = EnvironmentVariable(
            name=name,
            value=value,
            description=description,
            is_secret=is_secret
        )
        
        env_data.environment_variables.append(new_env_var)
        
        # Save the updated environment data
        return EnvironmentController._save_environment_data(project_path, env_data)

    @staticmethod
    def remove_environment_variable(project_path: str, name: str) -> bool:
        """
        Remove an environment variable from the project.

        Args:
            project_path: Path to the project
            name: Name of the environment variable to remove

        Returns:
            True if the environment variable was removed successfully, False otherwise
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            return False
        
        # Remove the environment variable if it exists
        original_len = len(env_data.environment_variables)
        env_data.environment_variables = [
            var for var in env_data.environment_variables if var.name != name
        ]
        
        # If no environment variables were removed, return False
        if len(env_data.environment_variables) == original_len:
            return False
        
        # Save the updated environment data
        return EnvironmentController._save_environment_data(project_path, env_data)

    @staticmethod
    def set_active_environment(project_path: str, env_name: str) -> bool:
        """
        Set the active virtual environment for the project.

        Args:
            project_path: Path to the project
            env_name: Name of the virtual environment to set as active

        Returns:
            True if the active environment was set successfully, False otherwise
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            return False
        
        # Check if the environment exists
        exists = any(env.name == env_name for env in env_data.virtual_environments)
        if not exists:
            return False
        
        # Set the active environment
        env_data.active_environment = env_name
        
        # Save the updated environment data
        return EnvironmentController._save_environment_data(project_path, env_data)

    @staticmethod
    def generate_requirements_file(project_path: str, dev: bool = False) -> Tuple[bool, str]:
        """
        Generate a requirements.txt file from the project dependencies.

        Args:
            project_path: Path to the project
            dev: Whether to include development dependencies

        Returns:
            Tuple of (success, file_path)
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            return False, ""
        
        # Filter dependencies
        deps = env_data.dependencies
        if not dev:
            deps = [dep for dep in deps if not dep.is_dev_dependency]
        
        if not deps:
            return False, ""
        
        # Generate requirements.txt content
        content = "\n".join(f"{dep.name}=={dep.version}" for dep in deps)
        
        # Write the file
        file_path = os.path.join(project_path, "requirements.txt")
        if write_file(file_path, content):
            return True, file_path
        else:
            return False, ""

    @staticmethod
    def check_dependencies_security(project_path: str) -> Dict:
        """
        Check security of project dependencies.

        Args:
            project_path: Path to the project

        Returns:
            Dictionary with security check results
        """
        # Load current environment data
        env_data = EnvironmentController.load_environment_data(project_path)
        if not env_data:
            return {
                "status": "error",
                "message": "No environment data found",
                "vulnerable_dependencies": []
            }
        
        # Check cache first
        cached_results = AnalysisCache.get_cached_analysis(
            project_path, "environment", "dependencies_security"
        )
        
        if cached_results and isinstance(cached_results, dict):
            return cached_results
        
        # Check security of dependencies
        vulnerable_dependencies = []
        
        for dep in env_data.dependencies:
            try:
                is_secure, vulnerabilities = check_dependency_security(dep.name, dep.version)
                
                if not is_secure:
                    vulnerable_dependencies.append({
                        "name": dep.name,
                        "version": dep.version,
                        "is_dev": dep.is_dev_dependency,
                        "vulnerabilities": vulnerabilities
                    })
            except Exception as e:
                print(f"Error checking security for {dep.name}=={dep.version}: {e}")
        
        # Prepare result
        result = {
            "status": "success",
            "message": f"Checked {len(env_data.dependencies)} dependencies",
            "vulnerable_dependencies": vulnerable_dependencies
        }
        
        # Cache the result
        AnalysisCache.cache_analysis(
            project_path,
            "environment",
            "dependencies_security",
            result,
            ttl_seconds=86400  # 24 hours
        )
        
        return result

    @staticmethod
    def load_environment_data(project_path: str) -> Optional[EnvironmentData]:
        """
        Load environment data from the project.

        Args:
            project_path: Path to the project

        Returns:
            EnvironmentData or None if not found
        """
        env_config_path = os.path.join(project_path, config.ENVIRONMENT_CONFIG_FILE)
        
        if not os.path.exists(env_config_path):
            return None
        
        try:
            with open(env_config_path, 'r') as f:
                data = json.load(f)
                return EnvironmentData.parse_obj(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading environment data: {e}")
            return None

    @staticmethod
    def _save_environment_data(project_path: str, env_data: EnvironmentData) -> bool:
        """
        Save environment data to the project.

        Args:
            project_path: Path to the project
            env_data: EnvironmentData to save

        Returns:
            True if the data was saved successfully, False otherwise
        """
        # Ensure the directory exists
        ensure_directory(os.path.join(project_path, config.ENVIRONMENT_DIR))
        
        # Save the data
        env_config_path = os.path.join(project_path, config.ENVIRONMENT_CONFIG_FILE)
        try:
            with open(env_config_path, 'w') as f:
                f.write(env_data.json(indent=2))
            return True
        except Exception as e:
            print(f"Error saving environment data: {e}")
            return False
