import datetime
import logging
from typing import Dict, Any

from knowledge.graph.kg_models import KGNode, KGEdge, KnowledgeGraph
from knowledge.reasoning.engine.re_models import DataNode


class KGMLCommandFunctions:
    """
    Functions to execute KGML commands (Create, Update, Delete, Evaluate)
    with instruction parsing and sandboxed execution.
    """

    def __init__(self, kg: KnowledgeGraph, instruction_parser, sandbox):
        """
        Initialize command functions with dependencies.

        Args:
            kg: The Knowledge Graph instance
            instruction_parser: KGMLInstructionParser instance
            sandbox: PythonSandbox instance
        """
        self.kg = kg
        self.instruction_parser = instruction_parser
        self.sandbox = sandbox
        self.logger = logging.getLogger(self.__class__.__name__)

    def create(self, node_type: str, uid: str, instruction: str) -> Dict[str, Any]:
        """
        Create a new node based on instruction.

        Args:
            node_type: The type of node to create
            uid: The unique identifier for the node
            instruction: Natural language instruction

        Returns:
            Dict with operation results
        """
        try:
            # Log the operation
            self.logger.info(f"Creating node {uid} of type {node_type} with instruction: {instruction}")

            # Parse the instruction
            parsed = self.instruction_parser.parse_natural_language_instruction(
                instruction,
                entity_type="NODE",
                command_type="C"
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
            return {
                "status": "error",
                "error": str(e),
                "node_id": uid
            }

    def update(self, node_type: str, uid: str, instruction: str) -> Dict[str, Any]:
        """
        Update an existing node based on instruction.

        Args:
            node_type: The type of node to update
            uid: The unique identifier for the node
            instruction: Natural language instruction

        Returns:
            Dict with operation results
        """
        try:
            # Log the operation
            self.logger.info(f"Updating node {uid} with instruction: {instruction}")

            # Get the node
            node = self.kg.get_node(uid)
            if not node:
                return {
                    "status": "error",
                    "error": f"Node with ID {uid} not found",
                    "node_id": uid
                }

            # Ensure it's the right type if node_type is specified
            if node_type and node.type != node_type:
                return {
                    "status": "error",
                    "error": f"Node with ID {uid} is of type {node.type}, not {node_type}",
                    "node_id": uid
                }

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

        except Exception as e:
            self.logger.error(f"Error updating node: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "node_id": uid
            }

    def delete(self, node_type: str, uid: str, instruction: str) -> Dict[str, Any]:
        """
        Delete a node based on instruction.

        Args:
            node_type: The type of node to delete
            uid: The unique identifier for the node
            instruction: Natural language instruction (may contain conditions)

        Returns:
            Dict with operation results
        """
        try:
            # Log the operation
            self.logger.info(f"Deleting node {uid} with instruction: {instruction}")

            # Get the node
            node = self.kg.get_node(uid)
            if not node:
                return {
                    "status": "error",
                    "error": f"Node with ID {uid} not found",
                    "node_id": uid
                }

            # Ensure it's the right type if node_type is specified
            if node_type and node.type != node_type:
                return {
                    "status": "error",
                    "error": f"Node with ID {uid} is of type {node.type}, not {node_type}",
                    "node_id": uid
                }

            # Parse the instruction if needed for conditional deletion
            # In most cases, delete operation might not need complex parsing
            should_delete = True

            # If instruction includes conditional logic, parse and evaluate it
            if "if" in instruction.lower() or "when" in instruction.lower() or "only" in instruction.lower():
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

        except Exception as e:
            self.logger.error(f"Error deleting node: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "node_id": uid
            }

    def evaluate(self, node_type: str, uid: str, instruction: str) -> Dict[str, Any]:
        """
        Evaluate a node based on instruction.

        Args:
            node_type: The type of node to evaluate
            uid: The unique identifier for the node
            instruction: Natural language instruction

        Returns:
            Dict with evaluation results
        """
        try:
            # Log the operation
            self.logger.info(f"Evaluating node {uid} with instruction: {instruction}")

            # Get the node
            node = self.kg.get_node(uid)
            if not node:
                return {
                    "status": "error",
                    "error": f"Node with ID {uid} not found",
                    "node_id": uid
                }

            # Ensure it's the right type if node_type is specified
            if node_type and node.type != node_type:
                return {
                    "status": "error",
                    "error": f"Node with ID {uid} is of type {node.type}, not {node_type}",
                    "node_id": uid
                }

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

        except Exception as e:
            self.logger.error(f"Error evaluating node: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "node_id": uid
            }

    def link(self, source_uid: str, target_uid: str, instruction: str) -> Dict[str, Any]:
        """
        Create a link between nodes based on instruction.

        Args:
            source_uid: Source node ID
            target_uid: Target node ID
            instruction: Natural language instruction

        Returns:
            Dict with operation results
        """
        try:
            # Log the operation
            self.logger.info(f"Creating link from {source_uid} to {target_uid} with instruction: {instruction}")

            # Get the source and target nodes
            source_node = self.kg.get_node(source_uid)
            if not source_node:
                return {
                    "status": "error",
                    "error": f"Source node with ID {source_uid} not found",
                    "edge": f"{source_uid}->{target_uid}"
                }

            target_node = self.kg.get_node(target_uid)
            if not target_node:
                return {
                    "status": "error",
                    "error": f"Target node with ID {target_uid} not found",
                    "edge": f"{source_uid}->{target_uid}"
                }

            # Parse the instruction
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

        except Exception as e:
            self.logger.error(f"Error creating link: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "edge": f"{source_uid}->{target_uid}"
            }
