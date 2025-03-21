"""
KGML Executor - Main module for executing KGML programs.

This module defines the KGMLExecutor class which orchestrates the execution
of KGML programs by delegating to specialized executors for different statement types.
"""

import logging
from typing import Any, Dict

from knowledge.graph.kg_models import KnowledgeGraph
from knowledge.reasoning.dsl.execution.kgml_command_executor import KGMLCommandExecutor
from knowledge.reasoning.dsl.execution.kgml_control_executor import KGMLControlExecutor
from knowledge.reasoning.dsl.execution.kgml_execution_context import KGMLExecutionContext
from knowledge.reasoning.dsl.kgml_instruction_parser import KGMLInstructionParser
from knowledge.reasoning.dsl.kgml_parser import (
    tokenize, Parser, SimpleCommand, ConditionalCommand, 
    LoopCommand, KGBlock, ASTNode
)
from knowledge.reasoning.dsl.execution.python_sandbox import PythonSandbox


class KGMLExecutor:
    """
    Main executor for KGML programs.
    
    This class coordinates the execution of KGML programs by delegating to specialized
    executors for different types of statements and maintaining the overall execution state.
    """

    def __init__(self, kg: KnowledgeGraph, model_id: str = "qwen2.5-coder:14b"):
        """
        Initialize the KGML executor.
        
        Args:
            kg: The Knowledge Graph instance
            model_id: The LLM model ID to use for instruction parsing
        """
        self.kg = kg
        self.model_id = model_id
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize execution components
        self.context = KGMLExecutionContext(kg)
        self.instruction_parser = KGMLInstructionParser()
        self.sandbox = PythonSandbox()
        
        # Initialize specialized executors
        self.command_executor = KGMLCommandExecutor(
            kg, 
            self.instruction_parser, 
            self.sandbox, 
            self.context
        )
        self.control_executor = KGMLControlExecutor(self, self.context)

    def execute(self, kgml_code: str) -> KGMLExecutionContext:
        """
        Execute a KGML program string.
        
        Args:
            kgml_code: The KGML program to execute
            
        Returns:
            The execution context with results
            
        Raises:
            SyntaxError: If there's a syntax error in the KGML code
            ValueError: If there's a semantic error in the KGML code
            RuntimeError: If execution fails for other reasons
        """
        try:
            # Parse the KGML code into an AST
            tokens = tokenize(kgml_code)
            parser = Parser(tokens)
            ast = parser.parse_program()

            # Execute the AST
            return self.execute_ast(ast)
        except Exception as e:
            self.logger.error(f"KGML execution failed: {e}")
            self.context.log_execution("program", {"error": str(e)}, None, False)
            raise

    def execute_ast(self, ast) -> KGMLExecutionContext:
        """
        Execute a parsed KGML AST.
        
        Args:
            ast: The parsed KGML program AST
            
        Returns:
            The execution context with results
            
        Raises:
            TypeError: If the AST is not a valid program
        """
        if not hasattr(ast, 'statements'):
            raise TypeError(f"Expected Program AST, got {type(ast)}")

        # Execute each statement in the program
        for statement in ast.statements:
            self.execute_statement(statement)

        return self.context

    def execute_statement(self, statement: ASTNode) -> Any:
        """
        Execute a single KGML statement.
        
        Args:
            statement: The KGML statement to execute
            
        Returns:
            The execution result
            
        Raises:
            TypeError: If the statement type is unknown
        """
        # Determine the statement type and delegate to the appropriate executor
        if isinstance(statement, SimpleCommand):
            return self.command_executor.execute_command(statement)
        elif isinstance(statement, ConditionalCommand):
            return self.control_executor.execute_conditional(statement)
        elif isinstance(statement, LoopCommand):
            return self.control_executor.execute_loop(statement)
        elif isinstance(statement, KGBlock):
            return self.execute_kg_block(statement)
        else:
            raise TypeError(f"Unknown statement type: {type(statement)}")

    def execute_kg_block(self, block: KGBlock) -> Dict[str, Any]:
        """
        Execute a Knowledge Graph block with node and edge declarations.
        
        Args:
            block: The KG block to execute
            
        Returns:
            Dictionary with execution results
            
        Raises:
            NotImplementedError: If KG block execution is not fully implemented
        """
        # Implementation for KG blocks
        results = {"nodes": [], "edges": []}
        
        for declaration in block.declarations:
            if hasattr(declaration, 'uid') and not hasattr(declaration, 'source_uid'):
                # This is a node declaration
                node_id = declaration.uid
                fields = declaration.fields
                
                # Create a node with the specified fields
                # Note: This would need to be implemented properly
                # For now, log that we're creating a node
                self.logger.info(f"Creating node {node_id} with fields: {fields}")
                results["nodes"].append(node_id)
                
            elif hasattr(declaration, 'source_uid') and hasattr(declaration, 'target_uid'):
                # This is an edge declaration
                source_uid = declaration.source_uid
                target_uid = declaration.target_uid
                fields = declaration.fields
                
                # Create an edge with the specified fields
                # Note: This would need to be implemented properly
                # For now, log that we're creating an edge
                self.logger.info(f"Creating edge {source_uid} -> {target_uid} with fields: {fields}")
                results["edges"].append(f"{source_uid} -> {target_uid}")
            else:
                self.logger.warning(f"Unknown declaration type in KG block: {declaration}")
        
        return results
