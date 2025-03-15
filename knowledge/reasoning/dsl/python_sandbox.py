import importlib
import logging
from typing import Dict, Any, Optional, Union

from knowledge.graph.kg_models import KGNode, KGEdge


class PythonSandbox:
    """
    Provides a secure execution environment for running generated Python code.

    This sandbox restricts the execution environment to a minimal set of
    safe operations, preventing access to dangerous functions or modules.
    """

    def __init__(self):
        """Initialize the sandbox with security restrictions."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Only allow these built-in functions and values
        self.allowed_builtins = {
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'len': len, 'max': max, 'min': min, 'sum': sum,
            'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'range': range,
            'True': True, 'False': False, 'None': None
        }

        # Only allow these modules
        self.allowed_modules = {
            'math': self._safe_import('math'),
            'datetime': self._safe_import('datetime'),
            'json': self._safe_import('json'),
            're': self._safe_import('re')
        }

    def _safe_import(self, module_name: str) -> Optional[Any]:
        """Safely import a module, logging any errors."""
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            self.logger.error(f"Error importing {module_name}: {str(e)}")
            return None

    def create_safe_globals(self,
                            entity: Union[KGNode, KGEdge],
                            entity_type: str) -> Dict[str, Any]:
        """
        Create a restricted global namespace for code execution.

        Args:
            entity: The node or edge to operate on
            entity_type: The type of entity ("NODE" or "LINK")

        Returns:
            Dict with safe globals for code execution
        """
        # Create a restricted globals dictionary
        safe_globals = {
            '__builtins__': self.allowed_builtins,
            'result': {},  # Container for results
        }

        # Add the entity with the appropriate name
        if entity_type == "NODE":
            safe_globals['node'] = entity
        elif entity_type == "LINK":
            safe_globals['edge'] = entity

        # Add allowed modules
        for name, module in self.allowed_modules.items():
            if module is not None:
                safe_globals[name] = module

        return safe_globals

    def execute_code(self,
                     code: str,
                     entity: Union[KGNode, KGEdge],
                     entity_type: str) -> Dict[str, Any]:
        """
        Execute the provided code in a sandbox with the given entity.

        Args:
            code: The Python code to execute
            entity: The node or edge to operate on
            entity_type: The type of entity ("NODE" or "LINK")

        Returns:
            Dict with execution results
        """
        globals_dict = self.create_safe_globals(entity, entity_type)
        locals_dict = {}

        try:
            # Execute the code
            exec(code, globals_dict, locals_dict)

            # Return the result from globals
            result = globals_dict.get('result', {})

            # If result is not a dict, wrap it
            if not isinstance(result, dict):
                result = {"value": result}

            return result
        except Exception as e:
            self.logger.error(f"Error executing sandboxed code: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
