"""
KGML Execution Context - Maintains state during KGML program execution.

This module defines the KGMLExecutionContext class which manages variables,
results, execution logs, and control flow depth during KGML interpretation.
"""

import logging
from typing import Any, Dict, List, Optional

from knowledge.graph.kg_models import KnowledgeGraph


class KGMLExecutionContext:
    """
    Context object to keep track of variables and intermediate results during KGML execution.
    
    The execution context maintains:
    - Access to the knowledge graph
    - Variable values for use in expressions and commands
    - Results from node evaluations
    - Detailed execution logs
    - Recursion depth tracking for evaluation operations
    """

    def __init__(self, kg: KnowledgeGraph):
        """
        Initialize the execution context.
        
        Args:
            kg: The Knowledge Graph instance to operate on
        """
        self.kg = kg
        self.results = {}  # Store the results of node evaluations
        self.variables = {}  # Store variables for use in conditional and loop blocks
        self.eval_depth = 0  # Track recursion depth for evaluations
        self.max_eval_depth = 10  # Maximum allowed recursion depth
        self.execution_log = []  # Log of all commands executed and their results
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_execution(self, command_type: str, details: Dict[str, Any], 
                       result: Any = None, success: bool = True) -> Dict[str, Any]:
        """
        Record the execution of a command along with its result for auditing.
        
        Args:
            command_type: Type of command executed (C, U, D, E, etc.)
            details: Details about the executed command
            result: Result of the command execution
            success: Whether the command executed successfully
            
        Returns:
            The log entry that was created
        """
        entry = {
            "command_type": command_type,
            "details": details,
            "result": result,
            "success": success,
            "variables": self.variables.copy()  # Snapshot of variables at this point
        }
        self.execution_log.append(entry)
        self.logger.debug(f"Execution: {command_type} - Success: {success}")
        return entry

    def get_variable(self, name: str, default: Any = None) -> Any:
        """
        Get the value of a variable from the context.
        
        Args:
            name: Name of the variable to retrieve
            default: Default value to return if variable not found
            
        Returns:
            The value of the variable or the default if not found
        """
        return self.variables.get(name, default)

    def set_variable(self, name: str, value: Any) -> None:
        """
        Set the value of a variable in the context.
        
        Args:
            name: Name of the variable to set
            value: Value to assign to the variable
        """
        self.variables[name] = value

    def set_result(self, node_id: str, result: Any) -> None:
        """
        Store the evaluation result for a node.
        
        Args:
            node_id: ID of the node
            result: Evaluation result
        """
        self.results[node_id] = result

    def get_result(self, node_id: str, default: Any = None) -> Any:
        """
        Get the evaluation result for a node.
        
        Args:
            node_id: ID of the node
            default: Default value to return if result not found
            
        Returns:
            The evaluation result or default if not found
        """
        return self.results.get(node_id, default)

    def increment_eval_depth(self) -> int:
        """
        Increment the evaluation depth counter.
        
        Returns:
            The new evaluation depth
            
        Raises:
            RuntimeError: If maximum evaluation depth would be exceeded
        """
        if self.eval_depth >= self.max_eval_depth:
            raise RuntimeError(f"Maximum evaluation depth exceeded: {self.max_eval_depth}")
        
        self.eval_depth += 1
        return self.eval_depth

    def decrement_eval_depth(self) -> int:
        """
        Decrement the evaluation depth counter.
        
        Returns:
            The new evaluation depth
        """
        if self.eval_depth > 0:
            self.eval_depth -= 1
        return self.eval_depth

    def clear_variables(self) -> None:
        """Clear all variables in the context."""
        self.variables.clear()

    def clear_results(self) -> None:
        """Clear all evaluation results in the context."""
        self.results.clear()

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the execution.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_commands": len(self.execution_log),
            "successful_commands": sum(1 for entry in self.execution_log if entry["success"]),
            "failed_commands": sum(1 for entry in self.execution_log if not entry["success"]),
            "command_types": {
                cmd_type: sum(1 for entry in self.execution_log if entry["command_type"] == cmd_type)
                for cmd_type in set(entry["command_type"] for entry in self.execution_log)
            }
        }
