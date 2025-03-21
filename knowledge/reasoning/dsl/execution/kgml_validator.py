"""
KGML Validator - Validates KGML programs for semantic correctness.

This module provides a validator for KGML programs, ensuring they follow
semantic rules beyond basic syntax checking.
"""

from typing import Dict, List, Optional, Set, Any

from knowledge.reasoning.dsl.execution.kgml_ast_base import (
    ASTVisitor, Program, SimpleCommand, ConditionalCommand,
    LoopCommand, KGBlock, KGNodeDeclaration, KGEdgeDeclaration
)


class KGMLValidationError(Exception):
    """Exception raised for semantic errors in KGML programs."""
    
    def __init__(self, message: str, node: Any = None):
        self.message = message
        self.node = node
        super().__init__(message)


class KGMLValidator(ASTVisitor):
    """
    Validates KGML programs for semantic correctness.
    
    Traverses the AST and checks for semantic rules such as:
    - Entity type consistency
    - Node references (UIDs exist before being used)
    - Command validity
    - Control structure nesting limits
    """
    
    def __init__(self):
        # Keep track of defined node IDs
        self.defined_nodes: Set[str] = set()
        
        # Keep track of current control structure nesting level
        self.control_nesting_level = 0
        self.max_control_nesting = 5  # Maximum allowed nesting
        
        # Track errors
        self.errors: List[KGMLValidationError] = []
        
        # Entity type validation
        self.valid_entity_types = {"NODE", "LINK"}
        
        # Valid command types
        self.valid_commands = {"C►", "U►", "D►", "E►", "N►"}

    def validate(self, program: Program) -> List[KGMLValidationError]:
        """
        Validate a KGML program.
        
        Args:
            program: The program to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        self.errors = []
        self.defined_nodes = set()
        self.control_nesting_level = 0
        
        # Visit the program to validate it
        program.accept(self)
        
        return self.errors

    def visit_program(self, node: Program):
        """
        Validate a program node.
        
        Args:
            node: The program node to validate
        """
        # Validate each statement
        for statement in node.statements:
            statement.accept(self)

    def visit_simple_command(self, node: SimpleCommand):
        """
        Validate a simple command node.
        
        Args:
            node: The simple command node to validate
        """
        cmd_type = node.get_command_type()
        
        # Check command type
        if cmd_type not in self.valid_commands:
            self.errors.append(
                KGMLValidationError(f"Invalid command type: {cmd_type}", node)
            )
            return
        
        # For N► commands, only validate timeout
        if cmd_type == "N►":
            if node.timeout is not None:
                try:
                    timeout_val = float(node.timeout)
                    if timeout_val <= 0:
                        self.errors.append(
                            KGMLValidationError(f"Timeout must be positive, got: {timeout_val}", node)
                        )
                except ValueError:
                    self.errors.append(
                        KGMLValidationError(f"Invalid timeout value: {node.timeout}", node)
                    )
            return
        
        # Check entity type
        entity_type = node.get_entity_type()
        if entity_type not in self.valid_entity_types:
            self.errors.append(
                KGMLValidationError(f"Invalid entity type: {entity_type}", node)
            )
            return
        
        # Check UID
        uid = node.uid
        if uid is None and cmd_type != "N►":
            self.errors.append(
                KGMLValidationError(f"Missing UID for {cmd_type} command", node)
            )
            return
        
        # For creation commands, register the node ID
        if cmd_type == "C►" and entity_type == "NODE":
            if uid in self.defined_nodes:
                self.errors.append(
                    KGMLValidationError(f"Node with ID '{uid}' already exists", node)
                )
            else:
                self.defined_nodes.add(uid)
        
        # For update, delete, or evaluate commands, check that the node exists
        elif cmd_type in {"U►", "D►", "E►"} and entity_type == "NODE":
            if uid not in self.defined_nodes and not self._is_special_uid(uid):
                self.errors.append(
                    KGMLValidationError(f"Node with ID '{uid}' not defined before use", node)
                )
        
        # For link commands, handle special syntax
        if entity_type == "LINK":
            # Check for a -> b syntax
            if " -> " in uid:
                source, target = uid.split(" -> ", 1)
                source = source.strip()
                target = target.strip()
                
                # Check that source and target nodes exist
                if source not in self.defined_nodes and not self._is_special_uid(source):
                    self.errors.append(
                        KGMLValidationError(f"Source node '{source}' not defined before use", node)
                    )
                
                if target not in self.defined_nodes and not self._is_special_uid(target):
                    self.errors.append(
                        KGMLValidationError(f"Target node '{target}' not defined before use", node)
                    )

    def visit_conditional_command(self, node: ConditionalCommand):
        """
        Validate a conditional command node.
        
        Args:
            node: The conditional command node to validate
        """
        # Check nesting level
        self.control_nesting_level += 1
        if self.control_nesting_level > self.max_control_nesting:
            self.errors.append(
                KGMLValidationError(
                    f"Control structure nesting too deep (max: {self.max_control_nesting})",
                    node
                )
            )
            self.control_nesting_level -= 1
            return
        
        # Validate IF condition
        condition_cmd, block = node.if_clause
        if not isinstance(condition_cmd, SimpleCommand) or condition_cmd.get_command_type() != "E►":
            self.errors.append(
                KGMLValidationError("IF condition must be an evaluate (E►) command", node)
            )
        else:
            condition_cmd.accept(self)
        
        # Validate IF block
        for statement in block:
            statement.accept(self)
        
        # Validate ELIF clauses
        for elif_condition, elif_block in node.elif_clauses:
            if not isinstance(elif_condition, SimpleCommand) or elif_condition.get_command_type() != "E►":
                self.errors.append(
                    KGMLValidationError("ELIF condition must be an evaluate (E►) command", node)
                )
            else:
                elif_condition.accept(self)
            
            for statement in elif_block:
                statement.accept(self)
        
        # Validate ELSE clause
        if node.else_clause:
            for statement in node.else_clause:
                statement.accept(self)
        
        # Restore nesting level
        self.control_nesting_level -= 1

    def visit_loop_command(self, node: LoopCommand):
        """
        Validate a loop command node.
        
        Args:
            node: The loop command node to validate
        """
        # Check nesting level
        self.control_nesting_level += 1
        if self.control_nesting_level > self.max_control_nesting:
            self.errors.append(
                KGMLValidationError(
                    f"Control structure nesting too deep (max: {self.max_control_nesting})",
                    node
                )
            )
            self.control_nesting_level -= 1
            return
        
        # Check condition
        if not node.condition:
            self.errors.append(
                KGMLValidationError("Loop condition cannot be empty", node)
            )
        
        # Validate loop body
        for statement in node.block:
            statement.accept(self)
        
        # Restore nesting level
        self.control_nesting_level -= 1

    def visit_kg_block(self, node: KGBlock):
        """
        Validate a KG block node.
        
        Args:
            node: The KG block node to validate
        """
        # Validate each declaration
        for declaration in node.declarations:
            declaration.accept(self)

    def visit_kg_node_declaration(self, node: KGNodeDeclaration):
        """
        Validate a KG node declaration.
        
        Args:
            node: The KG node declaration to validate
        """
        # Check if node ID already exists
        if node.uid in self.defined_nodes:
            self.errors.append(
                KGMLValidationError(f"Node with ID '{node.uid}' already exists", node)
            )
        else:
            self.defined_nodes.add(node.uid)
        
        # Validate field values
        self._validate_fields(node.fields, "node", node)

    def visit_kg_edge_declaration(self, node: KGEdgeDeclaration):
        """
        Validate a KG edge declaration.
        
        Args:
            node: The KG edge declaration to validate
        """
        # Check that source and target nodes exist
        if node.source_uid not in self.defined_nodes and not self._is_special_uid(node.source_uid):
            self.errors.append(
                KGMLValidationError(f"Source node '{node.source_uid}' not defined before use", node)
            )
        
        if node.target_uid not in self.defined_nodes and not self._is_special_uid(node.target_uid):
            self.errors.append(
                KGMLValidationError(f"Target node '{node.target_uid}' not defined before use", node)
            )
        
        # Validate field values
        self._validate_fields(node.fields, "edge", node)

    def _validate_fields(self, fields: Dict[str, str], entity_kind: str, node: Any):
        """
        Validate field values for nodes and edges.
        
        Args:
            fields: Dictionary of field names to values
            entity_kind: Kind of entity ("node" or "edge")
            node: The AST node being validated
        """
        # Check for required fields based on entity kind
        if entity_kind == "node":
            if "type" not in fields:
                self.errors.append(
                    KGMLValidationError(f"Node declaration missing required 'type' field", node)
                )
        elif entity_kind == "edge":
            if "relation" not in fields:
                self.errors.append(
                    KGMLValidationError(f"Edge declaration missing required 'relation' field", node)
                )

    def _is_special_uid(self, uid: str) -> bool:
        """
        Check if a UID is a special identifier that doesn't need to be defined.
        
        Args:
            uid: The UID to check
            
        Returns:
            True if it's a special UID, False otherwise
        """
        # Special UIDs include built-in functions or auto-generated IDs
        special_prefixes = {"function-", "web_search", "pattern_extraction"}
        return any(uid.startswith(prefix) for prefix in special_prefixes)


def validate_kgml(program: Program) -> List[KGMLValidationError]:
    """
    Validate a KGML program.
    
    Args:
        program: The program to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    validator = KGMLValidator()
    return validator.validate(program)
