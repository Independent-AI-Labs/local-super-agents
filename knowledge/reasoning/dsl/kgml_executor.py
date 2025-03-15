import logging
from typing import Any, Dict

from kgml_command_functions import KGMLCommandFunctions
# Import dependencies here to avoid circular imports
from kgml_instruction_parser import KGMLInstructionParser
from knowledge.graph.kg_models import KnowledgeGraph
from knowledge.reasoning.dsl.kgml_parser import SimpleCommand, ConditionalCommand, LoopCommand, KGBlock
from python_sandbox import PythonSandbox


class KGMLExecutionContext:
    """
    Context object to keep track of variables and intermediate results during KGML execution.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.results = {}  # Store the results of node evaluations
        self.variables = {}  # Store variables for use in conditional and loop blocks
        self.eval_depth = 0  # Track recursion depth for evaluations
        self.max_eval_depth = 10  # Maximum allowed recursion depth
        self.execution_log = []  # Log of all commands executed and their results

    def log_execution(self, command_type: str, details: Dict[str, Any], result: Any = None, success: bool = True):
        """
        Record the execution of a command along with its result for auditing.
        """
        entry = {
            "command_type": command_type,
            "details": details,
            "result": result,
            "success": success,
            "variables": self.variables.copy()  # Snapshot of variables at this point
        }
        self.execution_log.append(entry)
        logging.debug(f"Execution: {command_type} - Success: {success}")
        return entry


class EnhancedKGMLExecutor:
    """
    Enhanced KGML Executor that integrates natural language instruction parsing
    and secure Python execution with the existing KGML infrastructure.
    """

    def __init__(self, kg: KnowledgeGraph, model_id: str = "qwen2.5-coder:14b"):
        """
        Initialize the enhanced KGML executor.

        Args:
            kg: The Knowledge Graph instance
            model_id: The LLM model ID to use for instruction parsing
        """
        self.kg = kg
        self.model_id = model_id
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.instruction_parser = KGMLInstructionParser()
        self.sandbox = PythonSandbox()
        self.command_functions = KGMLCommandFunctions(kg, self.instruction_parser, self.sandbox)

        # Initialize execution context
        self.context = KGMLExecutionContext(kg)

    def execute(self, kgml_code: str) -> KGMLExecutionContext:
        """
        Execute a KGML program string.

        Args:
            kgml_code: The KGML program to execute

        Returns:
            The execution context with results
        """
        try:
            # Use the existing parser to parse the KGML code
            from knowledge.reasoning.dsl.kgml_parser import tokenize, Parser

            tokens = tokenize(kgml_code)
            parser = Parser(tokens)
            ast = parser.parse_program()

            return self.execute_ast(ast)
        except Exception as e:
            self.logger.error(f"KGML execution failed: {e}")
            self.context.log_execution("program", {"error": str(e)}, None, False)
            raise

    def execute_ast(self, ast):
        """
        Execute a parsed KGML AST.

        Args:
            ast: The parsed KGML program AST

        Returns:
            The execution context with results
        """
        if not hasattr(ast, 'statements'):
            raise TypeError(f"Expected Program AST, got {type(ast)}")

        for statement in ast.statements:
            self.execute_statement(statement)

        return self.context

    def execute_statement(self, statement) -> Any:
        """
        Execute a single KGML statement.

        Args:
            statement: The KGML statement to execute

        Returns:
            The execution result
        """
        # Check the statement type
        if isinstance(statement, SimpleCommand):
            return self.execute_simple_command(statement)
        elif isinstance(statement, ConditionalCommand):
            return self.execute_conditional(statement)
        elif isinstance(statement, LoopCommand):
            return self.execute_loop(statement)
        elif isinstance(statement, KGBlock):
            return self.execute_kg_block(statement)
        else:
            raise TypeError(f"Unknown statement type: {type(statement)}")

    def execute_simple_command(self, cmd: SimpleCommand) -> Any:
        """
        Execute a simple KGML command (C►, U►, D►, E►).

        Args:
            cmd: The command to execute

        Returns:
            The execution result
        """
        cmd_type = cmd.cmd_type.rstrip('►')  # Remove the trailing marker

        try:
            if cmd_type == "C":  # Create command
                return self._execute_create(cmd)
            elif cmd_type == "U":  # Update command
                return self._execute_update(cmd)
            elif cmd_type == "D":  # Delete command
                return self._execute_delete(cmd)
            elif cmd_type == "E":  # Evaluate command
                return self._execute_evaluate(cmd)
            else:
                raise ValueError(f"Unknown command type: {cmd_type}")
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            self.context.log_execution(
                cmd_type,
                {"entity_type": cmd.entity_type, "uid": cmd.uid, "instruction": cmd.instruction},
                None,
                False
            )
            raise

    def _execute_create(self, cmd: SimpleCommand) -> Any:
        """
        Execute a create (C►) command with natural language instruction support.
        """
        details = {
            "entity_type": cmd.entity_type,
            "uid": cmd.uid,
            "instruction": cmd.instruction
        }

        if cmd.entity_type == "NODE":
            # Handle node creation
            result = self.command_functions.create(
                node_type="GenericNode",  # Default type, can be overridden in the instruction
                uid=cmd.uid,
                instruction=cmd.instruction
            )
            self.context.log_execution("C", details, result)
            return result
        elif cmd.entity_type == "LINK":
            # Parse the instruction to determine source and target
            # Default source is the uid in the command
            source = cmd.uid
            target = None

            # Extract target from the instruction
            if "->" in cmd.instruction:
                # Extract target from pattern like "SourceNode -> TargetNode"
                arrow_parts = cmd.instruction.split("->")
                if len(arrow_parts) >= 2:
                    # Target might be the first word after ->
                    target_part = arrow_parts[1].strip()
                    target_words = target_part.split()
                    if target_words:
                        target = target_words[0]
            elif "to" in cmd.instruction.lower().split():
                # Look for pattern like "Link from X to Y"
                parts = cmd.instruction.lower().split()
                if "to" in parts and parts.index("to") < len(parts) - 1:
                    # Find the word after "to"
                    idx = parts.index("to")
                    if idx + 1 < len(parts):
                        target = parts[idx + 1]

            if source and target:
                # Create the link
                result = self.command_functions.link(source, target, cmd.instruction)
                self.context.log_execution("C", details, result)
                return result
            else:
                raise ValueError(f"Could not determine source and target for link creation: {cmd.instruction}")
        else:
            raise ValueError(f"Unknown entity type for create command: {cmd.entity_type}")

    def _execute_update(self, cmd: SimpleCommand) -> Any:
        """
        Execute an update (U►) command with natural language instruction support.
        """
        details = {
            "entity_type": cmd.entity_type,
            "uid": cmd.uid,
            "instruction": cmd.instruction
        }

        if cmd.entity_type == "NODE":
            # Handle node update
            result = self.command_functions.update(
                node_type=None,  # Don't enforce specific type during update
                uid=cmd.uid,
                instruction=cmd.instruction
            )
            self.context.log_execution("U", details, result)
            return result
        elif cmd.entity_type == "LINK":
            # Similar logic as create but for updating links
            # [Implementation similar to _execute_create but for updating]
            # This would need to be implemented similar to the node update
            # I'm omitting the full implementation for brevity
            raise NotImplementedError("Link update not fully implemented in this example")
        else:
            raise ValueError(f"Unknown entity type for update command: {cmd.entity_type}")

    def _execute_delete(self, cmd: SimpleCommand) -> Any:
        """
        Execute a delete (D►) command with natural language instruction support.
        """
        details = {
            "entity_type": cmd.entity_type,
            "uid": cmd.uid,
            "instruction": cmd.instruction
        }

        if cmd.entity_type == "NODE":
            # Handle node deletion
            result = self.command_functions.delete(
                node_type=None,  # Don't enforce specific type during deletion
                uid=cmd.uid,
                instruction=cmd.instruction
            )
            self.context.log_execution("D", details, result)
            return result
        elif cmd.entity_type == "LINK":
            # Similar logic as create but for deleting links
            # [Implementation similar to _execute_create but for deleting]
            # This would need to be implemented similar to the node delete
            # I'm omitting the full implementation for brevity
            raise NotImplementedError("Link deletion not fully implemented in this example")
        else:
            raise ValueError(f"Unknown entity type for delete command: {cmd.entity_type}")

    def _execute_evaluate(self, cmd: SimpleCommand) -> Any:
        """
        Execute an evaluate (E►) command with natural language instruction support.
        """
        if self.context.eval_depth >= self.context.max_eval_depth:
            raise RuntimeError(f"Maximum evaluation depth exceeded: {self.context.max_eval_depth}")

        self.context.eval_depth += 1
        details = {
            "entity_type": cmd.entity_type,
            "uid": cmd.uid,
            "instruction": cmd.instruction
        }

        try:
            if cmd.entity_type == "NODE":
                # Handle node evaluation
                result = self.command_functions.evaluate(
                    node_type=None,  # Don't enforce specific type during evaluation
                    uid=cmd.uid,
                    instruction=cmd.instruction
                )

                # Store the result in context for use in conditionals
                var_name = f"eval_{cmd.uid}"
                self.context.variables[var_name] = result
                self.context.results[cmd.uid] = result

                eval_result = {"status": "evaluated", "node_id": cmd.uid, "result": result}
                self.context.log_execution("E", details, eval_result)

                self.context.eval_depth -= 1
                return result
            else:
                raise ValueError(f"Cannot evaluate entity type: {cmd.entity_type}")
        except Exception as e:
            self.context.eval_depth -= 1
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def execute_conditional(self, conditional: ConditionalCommand) -> Any:
        """
        Execute a conditional (IF►/ELIF►/ELSE►) statement.
        """
        # First, evaluate the IF condition
        if_condition_cmd, if_block = conditional.if_clause

        # Execute the condition command (should be an E► command)
        condition_result = self.execute_statement(if_condition_cmd)

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
                self.execute_statement(statement)
            return True

        # If IF condition was False, check ELIF conditions
        for elif_condition_cmd, elif_block in conditional.elif_clauses:
            elif_result = self.execute_statement(elif_condition_cmd)

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
                    self.execute_statement(statement)
                return True

        # If no conditions were True and there's an ELSE block, execute it
        if conditional.else_clause:
            self.logger.debug("Executing ELSE block")
            for statement in conditional.else_clause:
                self.execute_statement(statement)
            return True

        # No blocks were executed
        return False

    def execute_loop(self, loop: LoopCommand) -> Any:
        """
        Execute a loop (LOOP►) statement with enhanced instruction interpretation.
        """
        max_iterations = 100  # Safety limit
        iterations = 0

        # Use natural language instruction processing to determine loop behavior
        # Parse the condition to extract loop parameters
        try:
            parsed_condition = self.instruction_parser.parse_natural_language_instruction(
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
                    self.execute_statement(statement)

                iterations += 1

                # Evaluate whether to continue
                # Pass the current iteration count to the condition
                dummy_node.meta_props["iterations"] = iterations
                dummy_node.meta_props["loop_variables"] = self.context.variables

                # Evaluate the loop condition
                condition_result = self.sandbox.execute_code(
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
        Fallback simple loop execution.
        """
        max_iterations = 100  # Safety limit
        iterations = 0

        # Simple implementation - just do a few iterations
        should_continue = True

        while should_continue and iterations < max_iterations:
            self.logger.debug(f"Simple loop iteration {iterations + 1}")
            for statement in loop.block:
                self.execute_statement(statement)

            iterations += 1

            # Simple termination - just do 3 iterations
            should_continue = iterations < 3

        return iterations

    def execute_kg_block(self, block: KGBlock) -> Any:
        """
        Execute a KG block with improved entity handling.
        """
        # This method would need implementation similar to the original
        # KGMLExecutor.execute_kg_block but with enhanced instruction processing
        # I'm omitting the full implementation for brevity
        self.logger.info("KG block execution is not fully implemented in this example")
        return {"status": "kg_block_not_implemented"}
