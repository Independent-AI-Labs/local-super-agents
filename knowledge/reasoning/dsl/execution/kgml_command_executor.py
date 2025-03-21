"""
KGML Command Executor - Handles execution of simple KGML commands.

This module defines the KGMLCommandExecutor class which is responsible for
executing simple commands (C►, U►, D►, E►) in KGML programs.
"""

import logging
from typing import Any, Optional

from knowledge.graph.kg_models import KnowledgeGraph
from knowledge.reasoning.dsl.execution.kgml_execution_context import KGMLExecutionContext
from knowledge.reasoning.dsl.kgml_instruction_parser import KGMLInstructionParser
from knowledge.reasoning.dsl.kgml_parser import SimpleCommand
from knowledge.reasoning.dsl.execution.python_sandbox import PythonSandbox
from knowledge.reasoning.dsl.kgml_command_functions import KGMLCommandFunctions


class KGMLCommandExecutor:
    """
    Executes simple KGML commands (C►, U►, D►, E►).
    
    This class handles the execution of simple commands by translating them to 
    appropriate operations on the knowledge graph through command functions.
    """

    def __init__(self, 
                 kg: KnowledgeGraph, 
                 instruction_parser: KGMLInstructionParser, 
                 sandbox: PythonSandbox,
                 context: KGMLExecutionContext,
                 command_functions: Optional[KGMLCommandFunctions] = None):
        """
        Initialize the command executor.
        
        Args:
            kg: The Knowledge Graph instance
            instruction_parser: Parser for natural language instructions
            sandbox: Sandbox for executing Python code
            context: Execution context for tracking state
            command_functions: Optional command functions instance (will create if None)
        """
        self.kg = kg
        self.instruction_parser = instruction_parser
        self.sandbox = sandbox
        self.context = context
        
        if command_functions:
            self.command_functions = command_functions
        else:
            self.command_functions = KGMLCommandFunctions(kg, instruction_parser, sandbox)
            
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_command(self, cmd: SimpleCommand) -> Any:
        """
        Execute a simple KGML command.
        
        Args:
            cmd: The command to execute
            
        Returns:
            The execution result
            
        Raises:
            ValueError: If the command type is unknown
        """
        cmd_type = cmd.cmd_type.rstrip('►')  # Remove the trailing marker

        try:
            if cmd_type == "C":  # Create command
                return self.execute_create(cmd)
            elif cmd_type == "U":  # Update command
                return self.execute_update(cmd)
            elif cmd_type == "D":  # Delete command
                return self.execute_delete(cmd)
            elif cmd_type == "E":  # Evaluate command
                return self.execute_evaluate(cmd)
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

    def execute_create(self, cmd: SimpleCommand) -> Any:
        """
        Execute a create (C►) command.
        
        Args:
            cmd: The create command to execute
            
        Returns:
            The creation result
            
        Raises:
            ValueError: If the entity type is unknown
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
            # Handle link creation with support for both syntax options
            source = None
            target = None
            
            # Check if the command uid contains "->" which indicates explicit source and target
            if " -> " in cmd.uid:
                # Format: "a -> b"
                source, target = cmd.uid.split(" -> ", 1)
                source = source.strip()
                target = target.strip()
                # Generate a link ID since one wasn't explicitly provided
                link_id = f"link_{source}_{target}"
                
                # Use the instruction as the link's relation/type
                result = self.command_functions.link(
                    source, 
                    target, 
                    cmd.instruction
                )
                self.context.log_execution("C", details, result)
                return result
            else:
                # Format: link_id "Connect a to b"
                # Attempt to parse source and target from the instruction
                
                # Try specific patterns in the instruction
                instruction_lower = cmd.instruction.lower()
                
                # Check for "link between X and Y" pattern
                if "between" in instruction_lower and " and " in instruction_lower:
                    between_parts = cmd.instruction.split("between", 1)[1]
                    if " and " in between_parts:
                        entities = between_parts.split(" and ", 1)
                        source = entities[0].strip()
                        # Remove trailing text after the target name
                        target = entities[1].strip().split(" ", 1)[0]
                        
                        # Clean up source and target from common text patterns
                        for item in [source, target]:
                            item = item.strip()
                            if item.startswith("the "):
                                item = item[4:].strip()
                            if item.startswith('"') and item.endswith('"'):
                                item = item[1:-1].strip()
                
                # Check for "link from X to Y" pattern
                elif "from" in instruction_lower and " to " in instruction_lower:
                    from_part = cmd.instruction.split("from", 1)[1]
                    if " to " in from_part:
                        to_parts = from_part.split(" to ", 1)
                        source = to_parts[0].strip()
                        # Remove trailing text after the target name
                        target = to_parts[1].strip().split(" ", 1)[0]
                
                # Check for "connect X to Y" pattern
                elif "connect" in instruction_lower and " to " in instruction_lower:
                    connect_part = cmd.instruction.split("connect", 1)[1]
                    if " to " in connect_part:
                        to_parts = connect_part.split(" to ", 1)
                        source = to_parts[0].strip()
                        # Remove trailing text after the target name
                        target = to_parts[1].strip().split(" ", 1)[0]
                
                # Check for "X -> Y" pattern in the instruction
                elif "->" in cmd.instruction:
                    arrow_parts = cmd.instruction.split("->")
                    if len(arrow_parts) >= 2:
                        source = arrow_parts[0].strip()
                        target = arrow_parts[1].strip().split(" ", 1)[0]
                
                # If source and target were successfully extracted
                if source and target:
                    # Clean up source and target to handle quoted node IDs
                    for s in ["'", '"']:
                        if source.startswith(s) and source.endswith(s):
                            source = source[1:-1]
                        if target.startswith(s) and target.endswith(s):
                            target = target[1:-1]
                    
                    # Create the link
                    result = self.command_functions.link(source, target, cmd.instruction)
                    self.context.log_execution("C", details, result)
                    return result
                
                # If parsing failed, use LLM-based extraction as a fallback
                try:
                    # Use the instruction parser to extract source and target
                    parsed = self.instruction_parser.parse_natural_language_instruction(
                        instruction=cmd.instruction,
                        entity_type="LINK",
                        command_type="E",
                        context={"parsing_mode": "extract_link_endpoints"}
                    )
                    
                    # Create a dummy node for execution context
                    from knowledge.graph.kg_models import KGNode
                    dummy_node = KGNode(uid="link_extraction_context")
                    
                    # Execute the parsed code to extract source and target
                    extraction_result = self.sandbox.execute_code(
                        parsed["code"],
                        entity=dummy_node,
                        entity_type="NODE"
                    )
                    
                    if "source" in extraction_result and "target" in extraction_result:
                        source = extraction_result["source"]
                        target = extraction_result["target"]
                        
                        # Create the link
                        result = self.command_functions.link(source, target, cmd.instruction)
                        self.context.log_execution("C", details, result)
                        return result
                except Exception as e:
                    self.logger.warning(f"Failed to extract link endpoints using LLM: {e}")
                
                # If all extraction methods failed
                error_msg = f"Could not determine source and target for link creation: {cmd.instruction}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            raise ValueError(f"Unknown entity type for create command: {cmd.entity_type}")

    def execute_update(self, cmd: SimpleCommand) -> Any:
        """
        Execute an update (U►) command.
        
        Args:
            cmd: The update command to execute
            
        Returns:
            The update result
            
        Raises:
            ValueError: If the entity type is unknown
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
            # Extract source and target from the command UID if it contains "->"
            if " -> " in cmd.uid:
                source, target = cmd.uid.split(" -> ", 1)
                source = source.strip()
                target = target.strip()
                
                # Use default link update logic (would need to be implemented in command_functions)
                # For now, we'll raise a NotImplementedError
                raise NotImplementedError("Link update with explicit source/target not fully implemented")
            else:
                # The UID is the link ID, find the link and update it
                # This would require a method to find a link by ID in the knowledge graph
                # For now, we'll raise a NotImplementedError
                raise NotImplementedError("Link update by ID not fully implemented")
        else:
            raise ValueError(f"Unknown entity type for update command: {cmd.entity_type}")

    def execute_delete(self, cmd: SimpleCommand) -> Any:
        """
        Execute a delete (D►) command.
        
        Args:
            cmd: The delete command to execute
            
        Returns:
            The deletion result
            
        Raises:
            ValueError: If the entity type is unknown
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
            # Extract source and target from the command UID if it contains "->"
            if " -> " in cmd.uid:
                source, target = cmd.uid.split(" -> ", 1)
                source = source.strip()
                target = target.strip()
                
                # Implement link deletion logic
                # We would need to add a delete_link method to command_functions
                # For now, we'll raise a NotImplementedError
                raise NotImplementedError("Link deletion not fully implemented")
            else:
                # The UID is the link ID, find the link and delete it
                # This would require a method to find a link by ID in the knowledge graph
                # For now, we'll raise a NotImplementedError
                raise NotImplementedError("Link deletion by ID not fully implemented")
        else:
            raise ValueError(f"Unknown entity type for delete command: {cmd.entity_type}")

    def execute_evaluate(self, cmd: SimpleCommand) -> Any:
        """
        Execute an evaluate (E►) command.
        
        Args:
            cmd: The evaluate command to execute
            
        Returns:
            The evaluation result
            
        Raises:
            ValueError: If the entity type is not supported for evaluation
        """
        try:
            # Increment the evaluation depth counter
            self.context.increment_eval_depth()
            
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
                    self.context.set_variable(var_name, result)
                    self.context.set_result(cmd.uid, result)

                    eval_result = {"status": "evaluated", "node_id": cmd.uid, "result": result}
                    self.context.log_execution("E", details, eval_result)

                    return result
                else:
                    raise ValueError(f"Cannot evaluate entity type: {cmd.entity_type}")
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                raise
        finally:
            # Always decrement the evaluation depth counter
            self.context.decrement_eval_depth()
