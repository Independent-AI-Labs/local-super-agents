"""
KGML AST Base Classes - Define consistent interfaces for AST nodes.

This module provides base classes for the Abstract Syntax Tree (AST) nodes
used in the KGML parser, ensuring a consistent interface and structure.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Iterator


class ASTNode(ABC):
    """
    Abstract base class for all AST nodes.
    
    Provides a common interface for all nodes in the KGML AST,
    including visitor pattern support and attribute access.
    """
    
    @abstractmethod
    def accept(self, visitor):
        """
        Accept a visitor to process this node.
        
        Args:
            visitor: The visitor object
            
        Returns:
            Result of the visitor's visit method
        """
        pass
    
    def get_attributes(self) -> Dict[str, Any]:
        """
        Get all attributes of this node as a dictionary.
        
        Returns:
            Dictionary of attribute names to values
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def has_attribute(self, name: str) -> bool:
        """
        Check if this node has a specific attribute.
        
        Args:
            name: Name of the attribute
            
        Returns:
            True if the attribute exists, False otherwise
        """
        return hasattr(self, name)
    
    def get_attribute(self, name: str, default: Any = None) -> Any:
        """
        Get the value of a specific attribute.
        
        Args:
            name: Name of the attribute
            default: Default value to return if attribute doesn't exist
            
        Returns:
            Value of the attribute or default
        """
        return getattr(self, name, default)
    
    def get_children(self) -> Iterator['ASTNode']:
        """
        Get all child AST nodes of this node.
        
        Returns:
            Iterator of child AST nodes
        """
        for value in self.__dict__.values():
            if isinstance(value, ASTNode):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ASTNode):
                        yield item
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, ASTNode):
                        yield item
    
    def __eq__(self, other):
        """Enable equality comparison between AST nodes."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__


class Statement(ASTNode):
    """Base class for statement AST nodes."""
    
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor to process this statement."""
        pass


class Command(Statement):
    """Base class for command AST nodes."""
    
    @abstractmethod
    def get_command_type(self) -> str:
        """
        Get the type of this command.
        
        Returns:
            Command type string (e.g., "C►", "U►", etc.)
        """
        pass
    
    @abstractmethod
    def get_entity_type(self) -> Optional[str]:
        """
        Get the entity type this command operates on.
        
        Returns:
            Entity type string (e.g., "NODE", "LINK") or None
        """
        pass
    
    @abstractmethod
    def get_instruction(self) -> str:
        """
        Get the instruction text for this command.
        
        Returns:
            Instruction string
        """
        pass


class SimpleCommand(Command):
    """
    Represents a simple KGML command (C►, U►, D►, E►, N►).
    
    Attributes:
        cmd_type: The command type (e.g., "C►", "U►", etc.)
        entity_type: The entity type (e.g., "NODE", "LINK") or None
        uid: UID for the target node/link or None
        instruction: The natural language instruction
        timeout: Optional timeout value for N► commands
    """
    
    def __init__(
            self,
            cmd_type: str,
            entity_type: Optional[str],
            uid: Optional[str],
            instruction: str,
            timeout: Optional[str] = None
    ):
        self.cmd_type = cmd_type
        self.entity_type = entity_type
        self.uid = uid
        self.instruction = instruction
        self.timeout = timeout
    
    def get_command_type(self) -> str:
        """Get the type of this command."""
        return self.cmd_type
    
    def get_entity_type(self) -> Optional[str]:
        """Get the entity type this command operates on."""
        return self.entity_type
    
    def get_instruction(self) -> str:
        """Get the instruction text for this command."""
        return self.instruction
    
    def accept(self, visitor):
        """Accept a visitor to process this command."""
        return visitor.visit_simple_command(self)
    
    def __repr__(self):
        base = f"{self.cmd_type}"
        if self.entity_type is not None and self.uid is not None:
            base += f" {self.entity_type} {self.uid}"
        if self.timeout is not None:
            base += f" (timeout={self.timeout})"
        return f"SimpleCommand({base}, {self.instruction})"


class ControlStructure(Statement):
    """Base class for control structure AST nodes."""
    
    @abstractmethod
    def get_structure_type(self) -> str:
        """
        Get the type of this control structure.
        
        Returns:
            Structure type string (e.g., "IF", "LOOP")
        """
        pass


class ConditionalCommand(ControlStructure):
    """
    Represents a conditional block (IF► / ELIF► / ELSE►).
    
    Attributes:
        if_clause: Tuple of (condition command, block of statements)
        elif_clauses: List of tuples (condition command, block of statements)
        else_clause: Optional block of statements
    """
    
    def __init__(self, if_clause: tuple, elif_clauses: List[tuple], else_clause: Optional[List[Statement]]):
        self.if_clause = if_clause
        self.elif_clauses = elif_clauses
        self.else_clause = else_clause
    
    def get_structure_type(self) -> str:
        """Get the type of this control structure."""
        return "IF"
    
    def accept(self, visitor):
        """Accept a visitor to process this conditional command."""
        return visitor.visit_conditional_command(self)
    
    def __repr__(self):
        return (f"ConditionalCommand(if={self.if_clause}, "
                f"elif={self.elif_clauses}, else={self.else_clause})")


class LoopCommand(ControlStructure):
    """
    Represents a loop block (LOOP►).
    
    Attributes:
        condition: Loop condition as a natural language string
        block: Block of statements to execute
    """
    
    def __init__(self, condition: str, block: List[Statement]):
        self.condition = condition
        self.block = block
    
    def get_structure_type(self) -> str:
        """Get the type of this control structure."""
        return "LOOP"
    
    def accept(self, visitor):
        """Accept a visitor to process this loop command."""
        return visitor.visit_loop_command(self)
    
    def __repr__(self):
        return f"LoopCommand({self.condition}, {self.block})"


class Declaration(ASTNode):
    """Base class for declaration nodes in KG blocks."""
    
    @abstractmethod
    def get_declaration_type(self) -> str:
        """
        Get the type of this declaration.
        
        Returns:
            Declaration type string (e.g., "NODE", "EDGE")
        """
        pass


class KGNodeDeclaration(Declaration):
    """
    Represents a node declaration in a KG block (KGNODE►).
    
    Attributes:
        uid: Unique identifier for the node
        fields: Dictionary of field names to values
    """
    
    def __init__(self, uid: str, fields: Dict[str, str]):
        self.uid = uid
        self.fields = fields
    
    def get_declaration_type(self) -> str:
        """Get the type of this declaration."""
        return "NODE"
    
    def accept(self, visitor):
        """Accept a visitor to process this node declaration."""
        return visitor.visit_kg_node_declaration(self)
    
    def __repr__(self):
        return f"KGNodeDeclaration({self.uid}, {self.fields})"


class KGEdgeDeclaration(Declaration):
    """
    Represents an edge declaration in a KG block (KGLINK►).
    
    Attributes:
        source_uid: Source node unique identifier
        target_uid: Target node unique identifier
        fields: Dictionary of field names to values
    """
    
    def __init__(self, source_uid: str, target_uid: str, fields: Dict[str, str]):
        self.source_uid = source_uid
        self.target_uid = target_uid
        self.fields = fields
    
    def get_declaration_type(self) -> str:
        """Get the type of this declaration."""
        return "EDGE"
    
    def accept(self, visitor):
        """Accept a visitor to process this edge declaration."""
        return visitor.visit_kg_edge_declaration(self)
    
    def __repr__(self):
        return f"KGEdgeDeclaration({self.source_uid}->{self.target_uid}, {self.fields})"


class KGBlock(Statement):
    """
    Represents a Knowledge Graph block (KG►).
    
    Attributes:
        declarations: List of declarations in the block
    """
    
    def __init__(self, declarations: List[Declaration]):
        self.declarations = declarations
    
    def accept(self, visitor):
        """Accept a visitor to process this KG block."""
        return visitor.visit_kg_block(self)
    
    def __repr__(self):
        return f"KGBlock({self.declarations})"


class Program(ASTNode):
    """
    Represents a complete KGML program.
    
    Attributes:
        statements: List of statements in the program
    """
    
    def __init__(self, statements: List[Statement]):
        self.statements = statements
    
    def accept(self, visitor):
        """Accept a visitor to process this program."""
        return visitor.visit_program(self)
    
    def __repr__(self):
        return f"Program({self.statements})"


class ASTVisitor(ABC):
    """
    Abstract base class for AST visitors.
    
    Implements the visitor pattern for traversing and processing AST nodes.
    """
    
    @abstractmethod
    def visit_program(self, node: Program):
        """Visit a program node."""
        pass
    
    @abstractmethod
    def visit_simple_command(self, node: SimpleCommand):
        """Visit a simple command node."""
        pass
    
    @abstractmethod
    def visit_conditional_command(self, node: ConditionalCommand):
        """Visit a conditional command node."""
        pass
    
    @abstractmethod
    def visit_loop_command(self, node: LoopCommand):
        """Visit a loop command node."""
        pass
    
    @abstractmethod
    def visit_kg_block(self, node: KGBlock):
        """Visit a KG block node."""
        pass
    
    @abstractmethod
    def visit_kg_node_declaration(self, node: KGNodeDeclaration):
        """Visit a KG node declaration."""
        pass
    
    @abstractmethod
    def visit_kg_edge_declaration(self, node: KGEdgeDeclaration):
        """Visit a KG edge declaration."""
        pass