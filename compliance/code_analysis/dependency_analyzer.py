import ast
import importlib.util
import os
import re
import sys
import typing

import networkx as nx

# Import constants from the separate module
from constants import (
    DEFAULT_SOURCE_EXCLUDES,
    NON_SOURCE_CODE_EXCLUDES,
    BUILTIN_MODULES,
    DEFAULT_INCLUDE_PATTERN,
    ERROR_INVALID_DIRECTORY,
    ERROR_NO_DEPENDENCIES,
    REPORT_HEADER,
    REPORT_TOTAL_FILES,
    REPORT_TOTAL_DEPENDENCIES
)


class DependencyAnalyzer:
    """
    Comprehensive code dependency analyzer for Python projects.

    Supports flexible configuration and robust dependency tracking.
    """

    def __init__(
            self,
            base_path: str = None,
            include_patterns: typing.List[str] = None,
            exclude_patterns: typing.List[str] = None,
            filter_sources: bool = True,
            ignore_builtin_modules: bool = True,
            verbose: bool = False
    ):
        """
        Initialize the dependency analyzer.

        Args:
            base_path: Root directory to analyze. Defaults to current working directory.
            include_patterns: Regex patterns to include files
            exclude_patterns: Regex patterns to exclude files
            filter_sources: Filter out non-source code files
            ignore_builtin_modules: Ignore built-in Python modules
            verbose: Enable detailed logging
        """
        # Normalize and validate base path
        self.base_path = os.path.abspath(base_path or os.getcwd())
        if not os.path.isdir(self.base_path):
            raise ValueError(ERROR_INVALID_DIRECTORY.format(self.base_path))

        # Prepare exclude patterns
        default_excludes = DEFAULT_SOURCE_EXCLUDES.copy()
        if filter_sources:
            default_excludes.extend(NON_SOURCE_CODE_EXCLUDES)

        # Set include and exclude patterns
        self.include_patterns = include_patterns or DEFAULT_INCLUDE_PATTERN
        self.exclude_patterns = (exclude_patterns or []) + default_excludes

        # Configuration flags
        self.ignore_builtin_modules = ignore_builtin_modules
        self.verbose = verbose

        # Initialize dependency tracking
        self.dependency_graph = nx.DiGraph()
        self._import_cache = {}
        self._module_path_cache = {}

        # Create a mapping from module names to file paths
        # This will help resolve package imports
        self._module_file_mapping = {}

    def _log(self, message: str):
        """
        Log messages if verbose mode is enabled.

        Args:
            message: Message to log
        """
        if self.verbose:
            print(message)

    def _is_valid_file(self, filepath: str) -> bool:
        """
        Determine if a file should be included in analysis.

        Args:
            filepath: Path to the file

        Returns:
            Boolean indicating if file should be analyzed
        """
        try:
            rel_path = os.path.relpath(filepath, self.base_path)
        except ValueError:
            rel_path = filepath

        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, rel_path):
                if self.verbose:
                    self._log(f"Excluded: {rel_path} (matched {pattern})")
                return False

        # Check include patterns
        is_included = any(re.search(pattern, rel_path) for pattern in self.include_patterns)

        if self.verbose and not is_included:
            self._log(f"Not included: {rel_path}")

        return is_included

    def _build_module_mapping(self, python_files: typing.List[str]):
        """
        Build a mapping from module/package names to file paths.
        This helps resolve imports, especially for packages.

        Args:
            python_files: List of Python files to analyze
        """
        self._module_file_mapping = {}

        for filepath in python_files:
            # Get the relative path from base_path
            try:
                rel_path = os.path.relpath(filepath, self.base_path)
            except ValueError:
                # If filepath is on a different drive, use the absolute path
                rel_path = filepath

            # Skip if in excluded patterns
            if any(re.search(pattern, rel_path) for pattern in self.exclude_patterns):
                continue

            # Convert path to potential module name
            # Replace directory separators with dots
            module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')

            # Add to mapping - including all parent packages
            parts = module_path.split('.')
            for i in range(1, len(parts) + 1):
                potential_module = '.'.join(parts[:i])
                if potential_module not in self._module_file_mapping:
                    self._module_file_mapping[potential_module] = filepath

            # Handle special case for __init__.py files
            if os.path.basename(filepath) == '__init__.py':
                # The directory itself is also a module
                dir_as_module = os.path.dirname(rel_path).replace(os.path.sep, '.')
                if dir_as_module and dir_as_module not in self._module_file_mapping:
                    self._module_file_mapping[dir_as_module] = filepath

        if self.verbose:
            self._log(f"Built module mapping with {len(self._module_file_mapping)} entries")

    def _extract_imports(self, filepath: str) -> typing.List[typing.Tuple[str, str, str]]:
        """
        Extract import statements from a Python file with more context.

        Args:
            filepath: Path to the Python file

        Returns:
            List of tuples (module_name, import_type, full_import_path)
        """
        # Use cache to improve performance
        if filepath in self._import_cache:
            return self._import_cache[filepath]

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())
        except (SyntaxError, IOError) as e:
            self._log(f"Error parsing {filepath}: {e}")
            return []

        imports = []
        # Use more comprehensive import detection
        for node in ast.walk(tree):
            # Direct imports (import os)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Store the full import path, not just the first part
                    imports.append((alias.name.split('.')[0], 'import', alias.name))

            # From imports (from os import path)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Store the full module path, not just the first part
                    full_module_path = node.module
                    module_base = node.module.split('.')[0]

                    # Distinguish between absolute and relative imports
                    if node.level > 0:  # Relative import
                        # For relative imports, determine the correct base package
                        current_package = os.path.dirname(filepath).replace(os.path.sep, '.')
                        parts = current_package.split('.')

                        # Adjust for the relative level
                        if len(parts) >= node.level:
                            adjusted_base = '.'.join(parts[:-node.level] if node.level > 0 else parts)
                            if node.module:  # from .submodule import x
                                full_module_path = f"{adjusted_base}.{node.module}"
                            else:  # from . import x
                                full_module_path = adjusted_base

                        imports.append((module_base, 'from_relative', full_module_path))
                    else:
                        imports.append((module_base, 'from_absolute', full_module_path))

        # Debug output for imports
        if self.verbose:
            self._log(f"Imports found in {filepath}: {imports}")

        # Filter out built-in modules if configured
        if self.ignore_builtin_modules:
            imports = [imp for imp in imports if imp[0] not in BUILTIN_MODULES]

        # Cache and return results
        self._import_cache[filepath] = imports
        return imports

    def _resolve_module_path(self, module_name: str, full_path: str, current_file: str, import_type: str = 'import') -> typing.Optional[str]:
        """
        Resolve a module to its file path with advanced resolution strategy.

        Args:
            module_name: Base name of the module to resolve
            full_path: Full import path, including submodules
            current_file: Path of the file making the import
            import_type: Type of import (import, from_absolute, from_relative)

        Returns:
            Resolved file path or None
        """
        # Check cache first
        cache_key = (full_path, current_file, import_type)
        if cache_key in self._module_path_cache:
            return self._module_path_cache[cache_key]

        resolved_path = None

        # Try to find the module in our mapping
        if full_path in self._module_file_mapping:
            resolved_path = self._module_file_mapping[full_path]
            if self.verbose:
                self._log(f"Module {full_path} resolved from mapping to {resolved_path}")
            self._module_path_cache[cache_key] = resolved_path
            return resolved_path

        # If not in mapping, try file-based resolution
        # Potential resolution paths
        potential_paths = []
        current_dir = os.path.dirname(current_file)

        # Handle relative imports
        if import_type == 'from_relative':
            # Try to determine the package path from the current file
            package_parts = os.path.dirname(current_file).replace(os.path.sep, '.').split('.')

            # First attempt is to look for it relative to current dir
            potential_paths.extend([
                os.path.join(current_dir, full_path.replace('.', os.path.sep) + '.py'),
                os.path.join(current_dir, full_path.replace('.', os.path.sep), '__init__.py')
            ])

        # Project-level resolution
        potential_paths.extend([
            # Relative to base path
            os.path.join(self.base_path, full_path.replace('.', os.path.sep) + '.py'),
            os.path.join(self.base_path, full_path.replace('.', os.path.sep), '__init__.py'),

            # Relative to current file's directory
            os.path.join(current_dir, full_path.replace('.', os.path.sep) + '.py'),
            os.path.join(current_dir, full_path.replace('.', os.path.sep), '__init__.py')
        ])

        # Attempt to find a valid path
        for path in potential_paths:
            # Normalize the path to resolve any symlinks or redundant separators
            normalized_path = os.path.normpath(path)

            if os.path.exists(normalized_path) and self._is_valid_file(normalized_path):
                if self.verbose:
                    self._log(f"Resolved {full_path} to {normalized_path}")
                self._module_path_cache[cache_key] = normalized_path
                return normalized_path

        # Try to resolve by searching for module name patterns in all files
        # This helps with package imports like 'from compliance.code_analysis.dependency_analyzer import DependencyAnalyzer'
        for python_file in self._python_files:
            # Skip the current file
            if python_file == current_file:
                continue

            # Check if the file or directory name matches the module
            file_name = os.path.basename(python_file)
            dir_name = os.path.basename(os.path.dirname(python_file))

            # Create a pattern for the module name (ending with .py or module name matches directory)
            if (module_name + '.py' == file_name.lower() or
                    dir_name.lower() == module_name.lower() or
                    full_path.lower().endswith(file_name.lower().replace('.py', ''))):

                # Found a potential match
                resolved_path = python_file
                if self.verbose:
                    self._log(f"Found potential match for {full_path} in {resolved_path}")
                self._module_path_cache[cache_key] = resolved_path
                return resolved_path

        # Debug output for unresolved imports
        if self.verbose:
            self._log(f"Could not resolve {full_path} (base: {module_name}) in {current_file}")

        self._module_path_cache[cache_key] = None
        return None

    def analyze_dependencies(self) -> nx.DiGraph:
        """
        Recursively analyze dependencies in the project.

        Returns:
            Dependency graph of the project
        """
        # Reset graph and caches
        self.dependency_graph = nx.DiGraph()
        self._import_cache = {}
        self._module_path_cache = {}

        # Collect Python files
        self._python_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.base_path)
            for file in files
            if self._is_valid_file(os.path.join(root, file))
        ]

        self._log(f"Found {len(self._python_files)} Python files to analyze")

        # Build module mapping for better import resolution
        self._build_module_mapping(self._python_files)

        # Analyze dependencies
        for filepath in self._python_files:
            try:
                # Add node for current file
                self.dependency_graph.add_node(filepath)

                # Extract and resolve imports
                for module_name, import_type, full_path in self._extract_imports(filepath):
                    try:
                        module_path = self._resolve_module_path(module_name, full_path, filepath, import_type)

                        # Add edge if module path is valid and different from source
                        if module_path and module_path != filepath:
                            self.dependency_graph.add_edge(filepath, module_path)
                            if self.verbose:
                                self._log(f"Added dependency: {filepath} -> {module_path}")
                    except Exception as import_err:
                        self._log(f"Error resolving import {full_path} in {filepath}: {import_err}")

            except Exception as file_err:
                self._log(f"Error processing {filepath}: {file_err}")

        # Log final graph statistics
        self._log(f"Dependency graph created with {len(self.dependency_graph.nodes())} nodes")
        self._log(f"Total dependencies: {len(self.dependency_graph.edges())} edges")

        return self.dependency_graph

    def generate_dependency_report(self) -> typing.Dict[str, typing.Any]:
        """
        Generate a comprehensive report of dependency analysis.

        Returns:
            Detailed report of project dependencies
        """
        # Ensure dependencies are analyzed
        if len(self.dependency_graph) == 0:
            self.analyze_dependencies()

        # Compute centrality metrics
        degree_centrality = nx.degree_centrality(self.dependency_graph)
        in_centrality = nx.in_degree_centrality(self.dependency_graph)
        out_centrality = nx.out_degree_centrality(self.dependency_graph)

        # Prepare report
        return {
            'total_files': len(self.dependency_graph.nodes()),
            'total_dependencies': len(self.dependency_graph.edges()),
            'most_dependent_files': sorted(
                [(node, {'in_degree': self.dependency_graph.in_degree(node),
                         'in_centrality': in_centrality[node]})
                 for node in self.dependency_graph.nodes()],
                key=lambda x: x[1]['in_degree'],
                reverse=True
            )[:10],
            'most_dependencies': sorted(
                [(node, {'out_degree': self.dependency_graph.out_degree(node),
                         'out_centrality': out_centrality[node]})
                 for node in self.dependency_graph.nodes()],
                key=lambda x: x[1]['out_degree'],
                reverse=True
            )[:10]
        }