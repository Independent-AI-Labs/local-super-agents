"""
Knowledge Graph Manipulation Language (KGML) package.

This package provides the necessary components for parsing and executing KGML programs
that manipulate a Knowledge Graph. KGML is a domain-specific language designed to 
express operations on a Knowledge Graph in a concise and readable way.

Components:
- KGMLExecutor: Main executor for KGML programs
- KGMLCommandExecutor: Executor for simple commands (C►, U►, D►, E►)
- KGMLControlExecutor: Executor for control flow statements (IF►, ELIF►, ELSE►, LOOP►)
- KGMLExecutionContext: Context for tracking execution state
- KGMLInstructionParser: Parser for natural language instructions
- PythonSandbox: Secure environment for executing Python code

Usage:
    from knowledge.reasoning.dsl.kgml_executor import KGMLExecutor
    executor = KGMLExecutor(knowledge_graph)
    result = executor.execute(kgml_code)
"""
from knowledge.reasoning.dsl.execution.kgml_command_executor import KGMLCommandExecutor
from knowledge.reasoning.dsl.execution.kgml_control_executor import KGMLControlExecutor
from knowledge.reasoning.dsl.execution.kgml_execution_context import KGMLExecutionContext
from knowledge.reasoning.dsl.execution.kgml_executor import KGMLExecutor
# Import main components for easy access
from knowledge.reasoning.dsl.kgml_instruction_parser import KGMLInstructionParser
from knowledge.reasoning.dsl.execution.python_sandbox import PythonSandbox

# Define package version
__version__ = '0.1.0'

# Define public API
__all__ = [
    'KGMLExecutor',
    'KGMLCommandExecutor',
    'KGMLControlExecutor',
    'KGMLExecutionContext',
    'KGMLInstructionParser',
    'PythonSandbox',
]
