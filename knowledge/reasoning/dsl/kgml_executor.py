import logging
from typing import Any, Dict, Union

from knowledge.graph.kg_models import KGNode, KGEdge, KnowledgeGraph
from knowledge.reasoning.dsl.kgml_parser import parse_kgml, Program, SimpleCommand, ConditionalCommand, LoopCommand

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLExecutor")


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
        logger.debug(f"Execution: {command_type} - Success: {success}")
        return entry


class KGMLExecutor:
    """
    Executes KGML code by processing the AST and manipulating a KnowledgeGraph.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.context = KGMLExecutionContext(kg)

    def execute(self, kgml_code: str) -> KGMLExecutionContext:
        """
        Execute a KGML program string by parsing it to AST and
        then executing each statement.
        """
        try:
            ast = parse_kgml(kgml_code)
            return self.execute_ast(ast)
        except Exception as e:
            logger.error(f"KGML execution failed: {e}")
            self.context.log_execution("program", {"error": str(e)}, None, False)
            raise

    def execute_ast(self, ast: Program) -> KGMLExecutionContext:
        """
        Execute a parsed KGML AST by processing each statement.
        """
        if not isinstance(ast, Program):
            raise TypeError(f"Expected Program AST, got {type(ast)}")

        for statement in ast.statements:
            self.execute_statement(statement)

        return self.context

    def execute_statement(self, statement: Union[SimpleCommand, ConditionalCommand, LoopCommand]) -> Any:
        """
        Execute a single KGML statement which can be a simple command,
        conditional, or loop.
        """
        if isinstance(statement, SimpleCommand):
            return self.execute_simple_command(statement)
        elif isinstance(statement, ConditionalCommand):
            return self.execute_conditional(statement)
        elif isinstance(statement, LoopCommand):
            return self.execute_loop(statement)
        else:
            raise TypeError(f"Unknown statement type: {type(statement)}")

    def execute_simple_command(self, cmd: SimpleCommand) -> Any:
        """
        Execute a simple KGML command (C►, U►, D►, E►, N►).
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
            elif cmd_type == "N":  # Navigate command
                return self._execute_navigate(cmd)
            else:
                raise ValueError(f"Unknown command type: {cmd_type}")
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            self.context.log_execution(
                cmd_type,
                {"entity_type": cmd.entity_type, "uid": cmd.uid, "instruction": cmd.instruction},
                None,
                False
            )
            raise

    def _execute_create(self, cmd: SimpleCommand) -> Any:
        """
        Execute a create (C►) command, creating either a node or a link in the KG.
        """
        details = {
            "entity_type": cmd.entity_type,
            "uid": cmd.uid,
            "instruction": cmd.instruction
        }

        if cmd.entity_type == "NODE":
            # Construct a new KGNode based on the instruction
            node = KGNode(
                uid=cmd.uid,
                type="ReasoningNode",  # Default type
                meta_props={"instruction": cmd.instruction}
            )
            self.kg.add_node(node)
            result = {"status": "created", "node_id": cmd.uid}
            self.context.log_execution("C", details, result)
            return result

        elif cmd.entity_type == "LINK":
            # Parse the instruction to determine source and target
            # Assuming the instruction has format "From X to Y" or similar
            # This is a simplification - in practice, you would use NLP or a more robust approach
            parts = cmd.instruction.split()

            # Very basic parsing of the instruction
            source = cmd.uid  # Using the uid as source by default
            target = None

            # Try to extract source and target from the instruction
            if "To" in parts and parts.index("To") < len(parts) - 1:
                target = parts[parts.index("To") + 1]

            if source and target:
                edge = KGEdge(
                    source_uid=source,
                    target_uid=target,
                    meta_props={"instruction": cmd.instruction}
                )
                self.kg.add_edge(edge)
                result = {"status": "created", "edge": f"{source}->{target}"}
                self.context.log_execution("C", details, result)
                return result
            else:
                raise ValueError(f"Could not determine source and target for link creation: {cmd.instruction}")

        else:
            raise ValueError(f"Unknown entity type for create command: {cmd.entity_type}")

    def _execute_update(self, cmd: SimpleCommand) -> Any:
        """
        Execute an update (U►) command, updating a node or link in the KG.
        """
        details = {
            "entity_type": cmd.entity_type,
            "uid": cmd.uid,
            "instruction": cmd.instruction
        }

        if cmd.entity_type == "NODE":
            # Update the node with new metadata from the instruction
            properties = {"meta_props": {"instruction": cmd.instruction}}
            self.kg.update_node(cmd.uid, properties)
            result = {"status": "updated", "node_id": cmd.uid}
            self.context.log_execution("U", details, result)
            return result

        elif cmd.entity_type == "LINK":
            # Similarly to create, we need to parse the instruction to find source/target
            # For simplicity, assuming uid refers to the source node
            source = cmd.uid
            target = None

            # Basic parsing to extract the target
            parts = cmd.instruction.split()
            if "To" in parts and parts.index("To") < len(parts) - 1:
                target = parts[parts.index("To") + 1]

            if source and target:
                properties = {"meta_props": {"instruction": cmd.instruction}}
                self.kg.update_edge(source, target, properties)
                result = {"status": "updated", "edge": f"{source}->{target}"}
                self.context.log_execution("U", details, result)
                return result
            else:
                raise ValueError(f"Could not determine target for link update: {cmd.instruction}")

        else:
            raise ValueError(f"Unknown entity type for update command: {cmd.entity_type}")

    def _execute_delete(self, cmd: SimpleCommand) -> Any:
        """
        Execute a delete (D►) command, removing a node or link from the KG.
        """
        details = {
            "entity_type": cmd.entity_type,
            "uid": cmd.uid,
            "instruction": cmd.instruction
        }

        if cmd.entity_type == "NODE":
            self.kg.remove_node(cmd.uid)
            result = {"status": "deleted", "node_id": cmd.uid}
            self.context.log_execution("D", details, result)
            return result

        elif cmd.entity_type == "LINK":
            # Parse the instruction to determine the target
            source = cmd.uid
            target = None

            parts = cmd.instruction.split()
            if "Link" in parts and "with" in parts and parts.index("with") < len(parts) - 1:
                target = parts[parts.index("with") + 1]
            elif "To" in parts and parts.index("To") < len(parts) - 1:
                target = parts[parts.index("To") + 1]

            if source and target:
                self.kg.remove_edge(source, target)
                result = {"status": "deleted", "edge": f"{source}->{target}"}
                self.context.log_execution("D", details, result)
                return result
            else:
                raise ValueError(f"Could not determine target for link deletion: {cmd.instruction}")

        else:
            raise ValueError(f"Unknown entity type for delete command: {cmd.entity_type}")

    def _execute_evaluate(self, cmd: SimpleCommand) -> Any:
        """
        Execute an evaluate (E►) command, evaluating a node in the KG.

        The evaluation result is stored in the context and returned.
        This is key for conditional processing.
        """
        # Avoid infinite recursion with a depth counter
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
                # Get the node to evaluate
                node = self.kg.get_node(cmd.uid)
                if not node:
                    raise ValueError(f"Node not found for evaluation: {cmd.uid}")

                # Simple evaluation logic - in a real system, this would invoke
                # functions based on node type, evaluate linked nodes, etc.
                # Here we'll just return a boolean based on simple text analysis

                # Check if the instruction has terms like "success" or "is true"
                result = ("success" in cmd.instruction.lower() or
                          "is true" in cmd.instruction.lower() or
                          "is successful" in cmd.instruction.lower())

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
            logger.error(f"Evaluation failed: {e}")
            raise

    def _execute_navigate(self, cmd: SimpleCommand) -> Any:
        """
        Execute a navigate (N►) command, traversing the KG based on instructions.
        """
        details = {
            "instruction": cmd.instruction,
            "timeout": cmd.timeout
        }

        # This could implement a complex navigation algorithm
        # For now, we'll just simulate navigation with a stub
        result = {"status": "navigated", "path": "simulated_path"}
        self.context.log_execution("N", details, result)
        return result

    def execute_conditional(self, conditional: ConditionalCommand) -> Any:
        """
        Execute a conditional (IF►/ELIF►/ELSE►) statement.
        """
        # First, evaluate the IF condition
        if_condition_cmd, if_block = conditional.if_clause

        # Execute the condition command (should be an E► command)
        condition_result = self.execute_statement(if_condition_cmd)
        logger.debug(f"IF condition evaluated to: {condition_result}")

        # If the condition is True, execute the IF block
        if condition_result:
            logger.debug("Executing IF block")
            for statement in if_block:
                self.execute_statement(statement)
            return True

        # If IF condition was False, check ELIF conditions
        for elif_condition_cmd, elif_block in conditional.elif_clauses:
            elif_result = self.execute_statement(elif_condition_cmd)
            logger.debug(f"ELIF condition evaluated to: {elif_result}")

            if elif_result:
                logger.debug("Executing ELIF block")
                for statement in elif_block:
                    self.execute_statement(statement)
                return True

        # If no conditions were True and there's an ELSE block, execute it
        if conditional.else_clause:
            logger.debug("Executing ELSE block")
            for statement in conditional.else_clause:
                self.execute_statement(statement)
            return True

        # No blocks were executed
        return False

    def execute_loop(self, loop: LoopCommand) -> Any:
        """
        Execute a loop (LOOP►) statement with a maximum iteration safeguard.
        """
        max_iterations = 100  # Safety limit
        iterations = 0

        # Simple implementation of loop condition checking
        # In a real system, this would be more sophisticated
        should_continue = ("while" in loop.condition.lower() or
                           "until" in loop.condition.lower())

        while should_continue and iterations < max_iterations:
            logger.debug(f"Loop iteration {iterations + 1}")
            for statement in loop.block:
                self.execute_statement(statement)

            iterations += 1

            # Simplified termination condition
            # In practice, you'd evaluate the condition based on KG state
            should_continue = iterations < 3  # Just do 3 iterations for demo purposes

        if iterations >= max_iterations:
            logger.warning(f"Loop terminated after reaching max iterations ({max_iterations})")

        return iterations
