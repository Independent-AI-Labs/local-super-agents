"""
KGML Command Functions - Refactored implementation of KGML command execution.

This module provides functions to execute KGML commands (Create, Update, Delete, Evaluate)
with instruction parsing and sandboxed execution, using a more modular and maintainable
approach with structured error handling.
"""

import datetime
import logging
from typing import Dict, Any, Optional

from knowledge.graph.kg_models import KGNode, KGEdge, KnowledgeGraph
from knowledge.reasoning.dsl.kgml_instruction_parser import KGMLInstructionParser
from knowledge.reasoning.dsl.execution.python_sandbox import PythonSandbox
from knowledge.reasoning.engine.re_models import DataNode
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import LinkRelation


class CommandError(Exception):
    """Exception raised for errors in command execution."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class KGMLCommandFunctions:
    """
    Functions to execute KGML commands (Create, Update, Delete, Evaluate)
    with instruction parsing and sandboxed execution.
    """

    def __init__(self, 
                kg: KnowledgeGraph, 
                instruction_parser: Optional[KGMLInstructionParser] = None, 
                sandbox: Optional[PythonSandbox] = None):
        """
        Initialize command functions with dependencies.
        
        Args:
            kg: The Knowledge Graph instance
            instruction_parser: KGMLInstructionParser instance (or None to create new)
            sandbox: PythonSandbox instance (or None to create new)
        """
        self.kg = kg
        self.instruction_parser = instruction_parser or KGMLInstructionParser()
        self.sandbox = sandbox or PythonSandbox()
        self.logger = logging.getLogger(self.__class__.__name__)

    def create(self, node_type: str, uid: str, instruction: str) -> Dict[str, Any]:
        """
        Create a new node based on natural language instruction.
        
        Args:
            node_type: The type of node to create
            uid: The unique identifier for the node
            instruction: Natural language instruction
            
        Returns:
            Dict with operation results including status and node_id
            
        Raises:
            CommandError: If node creation fails
        """
        try:
            # Log the operation
            self.logger.info(f"Creating node {uid} of type {node_type} with instruction: {instruction}")

            # Parse the instruction
            parsed = self.instruction_parser.parse_natural_language_instruction(
                instruction,
                entity_type="NODE",
                command_type="C",
                context={"default_type": node_type}
            )

            # Create a new node
            now = datetime.datetime.now().isoformat()
            node = KGNode(
                uid=uid,
                type=node_type,
                meta_props={},
                created_at=now,
                updated_at=now
            )

            # Execute the code to set up the node
            result = self.sandbox.execute_code(
                parsed["code"],
                entity=node,
                entity_type="NODE"
            )

            # Add the node to the knowledge graph
            self.kg.add_node(node)

            # Ensure we have a proper result
            if "status" not in result:
                result["status"] = "created"
            if "node_id" not in result:
                result["node_id"] = uid

            return result

        except Exception as e:
            self.logger.error(f"Error creating node: {str(e)}")
            raise CommandError(f"Failed to create node: {str(e)}", {
                "node_id": uid,
                "node_type": node_type,
                "instruction": instruction
            })
            
    def update(self, node_type: Optional[str], uid: str, instruction: str) -> Dict[str, Any]:
        """
        Update an existing node based on natural language instruction.
        
        Args:
            node_type: The type of node to update (optional)
            uid: The unique identifier for the node
            instruction: Natural language instruction
            
        Returns:
            Dict with operation results
            
        Raises:
            CommandError: If node update fails
        """
        try:
            # Log the operation
            self.logger.info(f"Updating node {uid} with instruction: {instruction}")

            # Get the node
            node = self.kg.get_node(uid)
            if not node:
                raise CommandError(f"Node with ID {uid} not found", {
                    "node_id": uid,
                    "instruction": instruction
                })

            # Ensure it's the right type if node_type is specified
            if node_type and node.type != node_type:
                raise CommandError(
                    f"Node with ID {uid} is of type {node.type}, not {node_type}",
                    {"node_id": uid, "actual_type": node.type, "expected_type": node_type}
                )

            # Parse the instruction
            parsed = self.instruction_parser.parse_natural_language_instruction(
                instruction,
                entity_type="NODE",
                command_type="U",
                context={"current_type": node.type}
            )

            # Set the updated_at timestamp
            node.updated_at = datetime.datetime.now().isoformat()

            # Execute the code to update the node
            result = self.sandbox.execute_code(
                parsed["code"],
                entity=node,
                entity_type="NODE"
            )

            # Update the node in the knowledge graph
            self.kg.update_node(uid, node.model_dump())

            # Ensure we have a proper result
            if "status" not in result:
                result["status"] = "updated"
            if "node_id" not in result:
                result["node_id"] = uid

            return result

        except CommandError:
            # Re-raise CommandError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Error updating node: {str(e)}")
            raise CommandError(f"Failed to update node: {str(e)}", {
                "node_id": uid,
                "instruction": instruction
            })
            
    def delete(self, node_type: Optional[str], uid: str, instruction: str) -> Dict[str, Any]:
        """
        Delete a node based on natural language instruction.
        
        Args:
            node_type: The type of node to delete (optional)
            uid: The unique identifier for the node
            instruction: Natural language instruction (may contain conditions)
            
        Returns:
            Dict with operation results
            
        Raises:
            CommandError: If node deletion fails
        """
        try:
            # Log the operation
            self.logger.info(f"Deleting node {uid} with instruction: {instruction}")

            # Get the node
            node = self.kg.get_node(uid)
            if not node:
                raise CommandError(f"Node with ID {uid} not found", {
                    "node_id": uid,
                    "instruction": instruction
                })

            # Ensure it's the right type if node_type is specified
            if node_type and node.type != node_type:
                raise CommandError(
                    f"Node with ID {uid} is of type {node.type}, not {node_type}",
                    {"node_id": uid, "actual_type": node.type, "expected_type": node_type}
                )

            # Parse the instruction if needed for conditional deletion
            should_delete = True

            # If instruction includes conditional logic, parse and evaluate it
            if any(keyword in instruction.lower() for keyword in ["if", "when", "only"]):
                parsed = self.instruction_parser.parse_natural_language_instruction(
                    instruction,
                    entity_type="NODE",
                    command_type="E",
                    context={"operation": "delete"}
                )

                # Execute the code to evaluate the deletion condition
                eval_result = self.sandbox.execute_code(
                    parsed["code"],
                    entity=node,
                    entity_type="NODE"
                )

                # Check if the condition allows deletion
                should_delete = eval_result.get("should_delete", True)

            # Delete the node if conditions allow
            if should_delete:
                self.kg.remove_node(uid)
                return {
                    "status": "deleted",
                    "node_id": uid
                }
            else:
                return {
                    "status": "not_deleted",
                    "reason": "Condition not met",
                    "node_id": uid
                }

        except CommandError:
            # Re-raise CommandError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Error deleting node: {str(e)}")
            raise CommandError(f"Failed to delete node: {str(e)}", {
                "node_id": uid,
                "instruction": instruction
            })
            
    def evaluate(self, node_type: Optional[str], uid: str, instruction: str) -> Dict[str, Any]:
        """
        Evaluate a node based on natural language instruction.
        
        Args:
            node_type: The type of node to evaluate (optional)
            uid: The unique identifier for the node
            instruction: Natural language instruction
            
        Returns:
            Dict with evaluation results
            
        Raises:
            CommandError: If node evaluation fails
        """
        try:
            # Log the operation
            self.logger.info(f"Evaluating node {uid} with instruction: {instruction}")

            # Get the node
            node = self.kg.get_node(uid)
            if not node:
                raise CommandError(f"Node with ID {uid} not found", {
                    "node_id": uid,
                    "instruction": instruction
                })

            # Ensure it's the right type if node_type is specified
            if node_type and node.type != node_type:
                raise CommandError(
                    f"Node with ID {uid} is of type {node.type}, not {node_type}",
                    {"node_id": uid, "actual_type": node.type, "expected_type": node_type}
                )

            # Parse the instruction
            parsed = self.instruction_parser.parse_natural_language_instruction(
                instruction,
                entity_type="NODE",
                command_type="E",
                context={"node_type": node.type}
            )

            # Cast to DataNode if possible to access content
            if not isinstance(node, DataNode):
                # Try to create a temporary DataNode with the same properties
                try:
                    enhanced_node = DataNode(**node.model_dump())
                except Exception:
                    enhanced_node = node
            else:
                enhanced_node = node

            # Execute the code to evaluate the node
            result = self.sandbox.execute_code(
                parsed["code"],
                entity=enhanced_node,
                entity_type="NODE"
            )

            # Add node ID to result if not present
            if "node_id" not in result:
                result["node_id"] = uid

            # Add status if not present
            if "status" not in result:
                result["status"] = "evaluated"

            return result

        except CommandError:
            # Re-raise CommandError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Error evaluating node: {str(e)}")
            raise CommandError(f"Failed to evaluate node: {str(e)}", {
                "node_id": uid,
                "instruction": instruction
            })
            
    def link(self, source_uid: str, target_uid: str, instruction: str) -> Dict[str, Any]:
        """
        Create a link between nodes based on natural language instruction.
        
        Args:
            source_uid: Source node ID
            target_uid: Target node ID
            instruction: Natural language instruction
            
        Returns:
            Dict with operation results
            
        Raises:
            CommandError: If link creation fails
        """
        try:
            # Log the operation
            self.logger.info(f"Creating link from {source_uid} to {target_uid} with instruction: {instruction}")

            # Get the source and target nodes
            source_node = self.kg.get_node(source_uid)
            if not source_node:
                raise CommandError(f"Source node with ID {source_uid} not found", {
                    "source_uid": source_uid,
                    "target_uid": target_uid,
                    "instruction": instruction
                })

            target_node = self.kg.get_node(target_uid)
            if not target_node:
                raise CommandError(f"Target node with ID {target_uid} not found", {
                    "source_uid": source_uid,
                    "target_uid": target_uid,
                    "instruction": instruction
                })

            # Check if the instruction is a direct link relation type (no parsing needed)
            normalized_instruction = instruction.strip()
            if normalized_instruction.startswith('"') and normalized_instruction.endswith('"'):
                normalized_instruction = normalized_instruction[1:-1].strip()
            if normalized_instruction.startswith("'") and normalized_instruction.endswith("'"):
                normalized_instruction = normalized_instruction[1:-1].strip()
                
            # Check if the normalized instruction matches a LinkRelation enum name
            relation_type = None
            try:
                for relation in LinkRelation:
                    if normalized_instruction == relation.name or normalized_instruction == relation.value:
                        relation_type = relation.name
                        break
                        
                if relation_type:
                    # Create a new edge with the explicit relation type
                    now = datetime.datetime.now().isoformat()
                    edge = KGEdge(
                        source_uid=source_uid,
                        target_uid=target_uid,
                        relation=relation_type,
                        meta_props={},
                        created_at=now,
                        updated_at=now
                    )
                    
                    # Add the edge to the knowledge graph
                    self.kg.add_edge(edge)
                    
                    return {
                        "status": "created",
                        "edge": f"{source_uid}->{target_uid}",
                        "relation": relation_type
                    }
            except (AttributeError, ValueError) as e:
                self.logger.debug(f"Instruction '{normalized_instruction}' is not a direct LinkRelation value: {e}")
                # Continue with normal parsing if it's not a direct relation type

            # Parse the instruction using the regular flow
            parsed = self.instruction_parser.parse_natural_language_instruction(
                instruction,
                entity_type="LINK",
                command_type="C",
                context={"source_type": source_node.type, "target_type": target_node.type}
            )

            # Create a new edge
            now = datetime.datetime.now().isoformat()
            edge = KGEdge(
                source_uid=source_uid,
                target_uid=target_uid,
                meta_props={},
                created_at=now,
                updated_at=now
            )

            # Execute the code to set up the edge
            result = self.sandbox.execute_code(
                parsed["code"],
                entity=edge,
                entity_type="LINK"
            )

            # Add the edge to the knowledge graph
            self.kg.add_edge(edge)

            # Ensure we have a proper result
            if "status" not in result:
                result["status"] = "created"
            if "edge" not in result:
                result["edge"] = f"{source_uid}->{target_uid}"

            return result

        except CommandError:
            # Re-raise CommandError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Error creating link: {str(e)}")
            raise CommandError(f"Failed to create link: {str(e)}", {
                "source_uid": source_uid,
                "target_uid": target_uid,
                "instruction": instruction
            })