"""
KGML Control Executor - Handles execution of KGML control flow structures.

This module defines the KGMLControlExecutor class which is responsible for
executing control flow structures (IF►, ELIF►, ELSE►, LOOP►) in KGML programs.
"""

import logging
from typing import Any, List, TYPE_CHECKING

from knowledge.reasoning.dsl.execution.kgml_execution_context import KGMLExecutionContext
from knowledge.reasoning.dsl.kgml_parser import ConditionalCommand, LoopCommand, ASTNode

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from knowledge.reasoning.dsl.execution.kgml_executor import KGMLExecutor


class KGMLControlExecutor:
    """
    Executes control flow statements in KGML programs.

    This class handles the execution of conditional blocks (IF►/ELIF►/ELSE►) and
    loop blocks (LOOP►) by evaluating conditions and executing the appropriate
    statement blocks.
    """

    def __init__(self, executor: 'KGMLExecutor', context: KGMLExecutionContext):
        """
        Initialize the control executor.

        Args:
            executor: The main KGML executor for executing nested statements
            context: Execution context for tracking state
        """
        self.executor = executor
        self.context = context
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_conditional(self, conditional: ConditionalCommand) -> Any:
        """
        Execute a conditional (IF►/ELIF►/ELSE►) statement.

        Args:
            conditional: The conditional command to execute

        Returns:
            True if a block was executed, False otherwise
        """
        # First, evaluate the IF condition
        if_condition_cmd, if_block = conditional.if_clause

        # Execute the condition command (should be an E► command)
        condition_result = self.executor.execute_statement(if_condition_cmd)

        # Determine if condition is truthy
        is_truthy = False
        if isinstance(condition_result, dict):
            is_truthy = condition_result.get("result", False)
        else:
            is_truthy = bool(condition_result)

        self.logger.debug(f"IF condition evaluated to: {is_truthy}")

        # If the condition is True, execute the IF block
        if is_truthy:
            self.logger.debug("Executing IF block")
            for statement in if_block:
                self.executor.execute_statement(statement)
            return True

        # If IF condition was False, check ELIF conditions
        for elif_condition_cmd, elif_block in conditional.elif_clauses:
            elif_result = self.executor.execute_statement(elif_condition_cmd)

            # Determine if condition is truthy
            is_truthy = False
            if isinstance(elif_result, dict):
                is_truthy = elif_result.get("result", False)
            else:
                is_truthy = bool(elif_result)

            self.logger.debug(f"ELIF condition evaluated to: {is_truthy}")

            if is_truthy:
                self.logger.debug("Executing ELIF block")
                for statement in elif_block:
                    self.executor.execute_statement(statement)
                return True

        # If no conditions were True and there's an ELSE block, execute it
        if conditional.else_clause:
            self.logger.debug("Executing ELSE block")
            for statement in conditional.else_clause:
                self.executor.execute_statement(statement)
            return True

        # No blocks were executed
        return False

    def execute_loop(self, loop: LoopCommand) -> int:
        """
        Execute a loop (LOOP►) statement.

        Args:
            loop: The loop command to execute

        Returns:
            Number of iterations executed
        """
        max_iterations = 100  # Safety limit
        iterations = 0

        # Use natural language instruction processing to determine loop behavior
        # Parse the condition to extract loop parameters
        try:
            from knowledge.reasoning.dsl.kgml_instruction_parser import KGMLInstructionParser

            # Get the instruction parser from the executor
            instruction_parser = getattr(self.executor, 'instruction_parser', None)
            if not instruction_parser:
                instruction_parser = KGMLInstructionParser()

            parsed_condition = instruction_parser.parse_natural_language_instruction(
                loop.condition,
                entity_type="NODE",  # Use NODE as default for loop parsing
                command_type="E",  # Treat as evaluation for condition
                context={"operation": "loop_condition"}
            )

            # Create a dummy node to evaluate the loop condition
            from knowledge.graph.kg_models import KGNode
            dummy_node = KGNode(uid="loop_condition", type="LoopCondition")

            should_continue = True

            while should_continue and iterations < max_iterations:
                self.logger.debug(f"Loop iteration {iterations + 1}")

                # Execute the loop body
                for statement in loop.block:
                    self.executor.execute_statement(statement)

                iterations += 1

                # Evaluate whether to continue
                # Pass the current iteration count to the condition
                dummy_node.meta_props["iterations"] = iterations
                dummy_node.meta_props["loop_variables"] = self.context.variables

                # Get the sandbox from the executor
                sandbox = getattr(self.executor, 'sandbox', None)
                if not sandbox:
                    from knowledge.reasoning.dsl.execution.python_sandbox import PythonSandbox
                    sandbox = PythonSandbox()

                # Evaluate the loop condition
                condition_result = sandbox.execute_code(
                    parsed_condition["code"],
                    entity=dummy_node,
                    entity_type="NODE"
                )

                # Check if we should continue
                should_continue = condition_result.get("continue", False)

                # Break if the condition says to stop
                if not should_continue:
                    break

            if iterations >= max_iterations:
                self.logger.warning(f"Loop terminated after reaching max iterations ({max_iterations})")

            return iterations
        except Exception as e:
            self.logger.error(f"Error in loop execution: {e}")
            # Fall back to simpler loop execution
            return self._execute_simple_loop(loop)

    def _execute_simple_loop(self, loop: LoopCommand) -> int:
        """
        Execute a loop using a simplified approach when advanced parsing fails.

        This is a fallback method that executes a loop a predefined number of times
        or until a simple condition is met.

        Args:
            loop: The loop command to execute

        Returns:
            Number of iterations executed
        """
        max_iterations = 100  # Safety limit
        iterations = 0

        # Simple heuristic for iteration count based on the loop condition
        target_iterations = 3  # Default iterations

        # Check if condition contains numeric value
        import re
        numbers = re.findall(r'\d+', loop.condition)
        if numbers:
            try:
                # Use the first number as iteration count if reasonable
                num = int(numbers[0])
                if 0 < num <= max_iterations:
                    target_iterations = num
            except ValueError:
                pass

        # Execute the loop body a fixed number of times
        should_continue = True
        while should_continue and iterations < max_iterations:
            self.logger.debug(f"Simple loop iteration {iterations + 1}")

            # Execute the loop body
            for statement in loop.block:
                self.executor.execute_statement(statement)

            iterations += 1

            # Stop after reaching target iterations
            should_continue = iterations < target_iterations

        return iterations

    def execute_block(self, block: List[ASTNode]) -> List[Any]:
        """
        Execute a block of statements.

        Args:
            block: List of statements to execute

        Returns:
            List of results from executing each statement
        """
        results = []
        for statement in block:
            result = self.executor.execute_statement(statement)
            results.append(result)
        return results