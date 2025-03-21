"""
KGML Parser - Enhanced parser implementation for Knowledge Graph Manipulation Language.

This module provides an optimized parser for KGML with improved error handling,
optimized token recognition, and consistent token processing.
"""

import re
from typing import List, Optional, Tuple, Dict, Any, Callable


# ------------------------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------------------------

class KGMLError(Exception):
    """Base exception class for all KGML-related errors."""
    pass


class KGMLParseError(KGMLError):
    """Exception raised for errors during KGML parsing."""
    
    def __init__(self, message: str, position: int, token_value: Optional[str] = None):
        self.position = position
        self.token_value = token_value
        self.message = message
        super().__init__(f"{message} at position {position}" + 
                        (f": '{token_value}'" if token_value else ""))


class KGMLSyntaxError(KGMLParseError):
    """Exception raised for KGML syntax errors."""
    pass


class KGMLTokenError(KGMLParseError):
    """Exception raised for errors during tokenization."""
    pass


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------

class TokenType:
    """Enum-like class defining token types."""
    KEYWORD = "KEYWORD"
    CLOSE = "CLOSE"  # For the closing marker: ◄
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    SYMBOL = "SYMBOL"
    OP = "OP"
    EOF = "EOF"


class Token:
    """Represents a token in the KGML language."""
    
    def __init__(self, type_: str, value: str, pos: int):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


# Updated RESERVED set - consolidate all keywords
RESERVED = {
    # Command keywords
    "C►", "U►", "D►", "E►", "N►",
    # Control keywords
    "IF►", "ELIF►", "ELSE►", "LOOP►", "END►",
    # Structure keywords
    "KG►", "KGNODE►", "KGLINK►"
}

# Optimized TOKEN_REGEX with consolidated keywords
TOKEN_REGEX = re.compile(
    r"""
    (?P<KEYWORD>(?:C|U|D|E|N|IF|ELIF|ELSE|LOOP|END|KG|KGNODE|KGLINK)►)|
    (?P<CLOSE>◄)|
    (?P<NUMBER>\d+(\.\d+)?)|
    (?P<STRING>"([^"\\]|\\.)*")|
    (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)|
    (?P<OP>(==|!=|<=|>=|-\>|[+\-*/<>]))|
    (?P<SYMBOL>[{}(),:;=])|
    (?P<WS>\s+)
    """, re.VERBOSE
)


def tokenize(text: str) -> List[Token]:
    """
    Tokenize a KGML program string.
    
    Args:
        text: The KGML program text to tokenize
        
    Returns:
        List of tokens
        
    Raises:
        KGMLTokenError: If unexpected characters are encountered
    """
    pos = 0
    tokens = []
    while pos < len(text):
        match = TOKEN_REGEX.match(text, pos)
        if not match:
            # Provide more context in the error message
            context_start = max(0, pos - 10)
            context_end = min(len(text), pos + 10)
            context = text[context_start:context_end]
            problematic_char = text[pos] if pos < len(text) else "EOF"
            raise KGMLTokenError(
                f"Unexpected character: '{problematic_char}', context: '...{context}...'", 
                pos, 
                problematic_char
            )
            
        if match.lastgroup == "WS":
            pos = match.end()
            continue
            
        token_type = match.lastgroup
        value = match.group(token_type)
        tokens.append(Token(token_type, value, pos))
        pos = match.end()
        
    tokens.append(Token(TokenType.EOF, "", pos))
    return tokens


# ------------------------------------------------------------------------------
# AST Node Base Classes
# ------------------------------------------------------------------------------

class ASTNode:
    """Base class for all AST nodes."""
    
    def __eq__(self, other):
        """Enable equality comparison between AST nodes."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__


class Statement(ASTNode):
    """Base class for statement nodes."""
    pass


class Command(Statement):
    """Base class for command nodes."""
    pass


class ControlStructure(Statement):
    """Base class for control structure nodes."""
    pass


class Declaration(ASTNode):
    """Base class for declaration nodes in KG blocks."""
    pass


# ------------------------------------------------------------------------------
# AST Node Classes
# ------------------------------------------------------------------------------

class Program(ASTNode):
    """Represents a complete KGML program."""
    
    def __init__(self, statements: List[Statement]):
        self.statements = statements

    def __repr__(self):
        return f"Program({self.statements})"


class SimpleCommand(Command):
    """Represents a simple KGML command (C►, U►, D►, E►, N►)."""
    
    def __init__(
            self,
            cmd_type: str,
            entity_type: Optional[str],
            uid: Optional[str],
            instruction: str,
            timeout: Optional[str] = None
    ):
        self.cmd_type = cmd_type  # e.g. "C►", "U►", etc.
        self.entity_type = entity_type  # "NODE" or "LINK" (None for navigate)
        self.uid = uid  # UID for the target node/link (None for navigate)
        self.instruction = instruction  # The natural language instruction
        self.timeout = timeout  # For N►, if provided

    def __repr__(self):
        base = f"{self.cmd_type}"
        if self.entity_type is not None and self.uid is not None:
            base += f" {self.entity_type} {self.uid}"
        if self.timeout is not None:
            base += f" (timeout={self.timeout})"
        return f"SimpleCommand({base}, {self.instruction})"


class ConditionalCommand(ControlStructure):
    """Represents a conditional block (IF► / ELIF► / ELSE►)."""
    
    def __init__(self, if_clause: Tuple[Command, List[Statement]],
                 elif_clauses: List[Tuple[Command, List[Statement]]],
                 else_clause: Optional[List[Statement]]):
        self.if_clause = if_clause  # Tuple (condition-command, block)
        self.elif_clauses = elif_clauses  # List of tuples (condition-command, block)
        self.else_clause = else_clause  # Block (list of statements) or None

    def __repr__(self):
        return (f"ConditionalCommand(if={self.if_clause}, "
                f"elif={self.elif_clauses}, else={self.else_clause})")


class LoopCommand(ControlStructure):
    """Represents a loop block (LOOP►)."""
    
    def __init__(self, condition: str, block: List[Statement]):
        self.condition = condition  # Loop condition as a natural language string
        self.block = block

    def __repr__(self):
        return f"LoopCommand({self.condition}, {self.block})"


class KGBlock(Statement):
    """Represents a Knowledge Graph block (KG►)."""
    
    def __init__(self, declarations: List[Declaration]):
        self.declarations = declarations

    def __repr__(self):
        return f"KGBlock({self.declarations})"


class KGNodeDeclaration(Declaration):
    """Represents a node declaration in a KG block (KGNODE►)."""
    
    def __init__(self, uid: str, fields: Dict[str, str]):
        self.uid = uid
        self.fields = fields

    def __repr__(self):
        return f"KGNodeDeclaration({self.uid}, {self.fields})"


class KGEdgeDeclaration(Declaration):
    """Represents an edge declaration in a KG block (KGLINK►)."""
    
    def __init__(self, source_uid: str, target_uid: str, fields: Dict[str, str]):
        self.source_uid = source_uid
        self.target_uid = target_uid
        self.fields = fields

    def __repr__(self):
        return f"KGEdgeDeclaration({self.source_uid}->{self.target_uid}, {self.fields})"


# ------------------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------------------

class Parser:
    """Parser for KGML programs."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        """Get the current token."""
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        """Look ahead at a token without consuming it."""
        if self.pos + offset >= len(self.tokens):
            return self.tokens[-1]  # Return EOF token if out of bounds
        return self.tokens[self.pos + offset]

    def eat(self, expected_type: Optional[str] = None, expected_value: Optional[str] = None) -> Token:
        """
        Consume a token, optionally checking its type and value.
        
        Args:
            expected_type: Expected token type
            expected_value: Expected token value
            
        Returns:
            The consumed token
            
        Raises:
            KGMLSyntaxError: If the token doesn't match the expected type or value
        """
        token = self.current_token()
        
        if expected_type and token.type != expected_type:
            raise KGMLSyntaxError(
                f"Expected token type {expected_type} but got {token.type}",
                token.pos,
                token.value
            )
            
        if expected_value and token.value != expected_value:
            raise KGMLSyntaxError(
                f"Expected token value '{expected_value}' but got '{token.value}'",
                token.pos,
                token.value
            )
            
        self.pos += 1
        return token

    def parse_program(self) -> Program:
        """
        Parse a complete KGML program.
        
        Returns:
            Program AST node
        """
        statements = []
        while self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            statements.append(stmt)
        return Program(statements)

    def parse_statement(self) -> Statement:
        """
        Parse a statement.
        
        Returns:
            Statement AST node
            
        Raises:
            KGMLSyntaxError: If the statement syntax is invalid
        """
        token = self.current_token()
        if token.type == TokenType.KEYWORD:
            keyword = token.value
            
            # Command statements
            if keyword in {"C►", "U►", "D►", "E►", "N►"}:
                return self.parse_simple_command()
                
            # Control statements
            elif keyword == "IF►":
                return self.parse_if_command()
            elif keyword == "LOOP►":
                return self.parse_loop_command()
                
            # KG block
            elif keyword == "KG►":
                return self.parse_kg_block()
                
            else:
                raise KGMLSyntaxError(f"Unexpected keyword", token.pos, token.value)
        else:
            raise KGMLSyntaxError(f"Expected a statement keyword", token.pos, token.value)

    def parse_simple_command(self) -> SimpleCommand:
        """
        Parse a simple command.
        
        Returns:
            SimpleCommand AST node
        """
        cmd_token = self.eat(TokenType.KEYWORD)
        cmd = cmd_token.value

        if cmd in {"C►", "U►", "D►", "E►"}:
            # Expect an entity type (NODE or LINK)
            etype_token = self.eat(TokenType.IDENT)
            entity_type = etype_token.value
            if entity_type not in {"NODE", "LINK"}:
                raise KGMLSyntaxError(
                    f"Expected entity type NODE or LINK, got {entity_type}",
                    etype_token.pos,
                    entity_type
                )
                
            # Expect a UID (identifier)
            uid_token = self.eat(TokenType.IDENT)
            uid = uid_token.value
            
            # Expect a STRING for the instruction
            instr_token = self.eat(TokenType.STRING)
            instruction = self._unquote(instr_token.value)
            
            # Expect the closing marker
            self.eat(TokenType.CLOSE, expected_value="◄")
            return SimpleCommand(cmd, entity_type, uid, instruction)
            
        elif cmd == "N►":
            # Navigate command: optionally a timeout NUMBER before the instruction.
            timeout = None
            next_token = self.current_token()
            if next_token.type == TokenType.NUMBER:
                timeout_token = self.eat(TokenType.NUMBER)
                timeout = timeout_token.value
                
            instr_token = self.eat(TokenType.STRING)
            instruction = self._unquote(instr_token.value)
            self.eat(TokenType.CLOSE, expected_value="◄")
            return SimpleCommand(cmd, None, None, instruction, timeout)
            
        else:
            raise KGMLSyntaxError(f"Unknown simple command", cmd_token.pos, cmd)

    def parse_if_command(self) -> ConditionalCommand:
        """
        Parse an IF block.
        
        Returns:
            ConditionalCommand AST node
        """
        self.eat(TokenType.KEYWORD, expected_value="IF►")
        
        # Parse the condition as a full command.
        if_condition_cmd = self.parse_simple_command()
        
        # Parse the IF block until we see an ELIF►, ELSE►, or closing marker (◄)
        if_block = self.parse_block(stop_keywords={"ELIF►", "ELSE►", "◄"})

        # Parse any ELIF clauses.
        elif_clauses = []
        while (self.current_token().type == TokenType.KEYWORD and
               self.current_token().value == "ELIF►"):
            self.eat(TokenType.KEYWORD, expected_value="ELIF►")
            elif_condition_cmd = self.parse_simple_command()
            elif_block = self.parse_block(stop_keywords={"ELIF►", "ELSE►", "◄"})
            elif_clauses.append((elif_condition_cmd, elif_block))

        # Parse the ELSE clause, if present.
        else_clause = None
        if (self.current_token().type == TokenType.KEYWORD and
                self.current_token().value == "ELSE►"):
            self.eat(TokenType.KEYWORD, expected_value="ELSE►")
            else_clause = self.parse_block(stop_keywords={"◄"})

        # Expect the closing marker for the IF block.
        self.eat(TokenType.CLOSE, expected_value="◄")
        return ConditionalCommand((if_condition_cmd, if_block), elif_clauses, else_clause)

    def parse_loop_command(self) -> LoopCommand:
        """
        Parse a LOOP block.
        
        Returns:
            LoopCommand AST node
        """
        self.eat(TokenType.KEYWORD, expected_value="LOOP►")
        loop_cond_token = self.eat(TokenType.STRING)
        loop_condition = self._unquote(loop_cond_token.value)
        self.eat(TokenType.CLOSE, expected_value="◄")
        block = self.parse_block(stop_keywords={"◄"})
        self.eat(TokenType.CLOSE, expected_value="◄")
        return LoopCommand(loop_condition, block)

    def parse_block(self, stop_keywords: set) -> List[Statement]:
        """
        Parse a block of statements.
        
        Args:
            stop_keywords: Set of keywords that terminate the block
            
        Returns:
            List of Statement AST nodes
        """
        stmts = []
        while True:
            curr = self.current_token()
            # If we hit a KEYWORD that is one of the stop tokens or a CLOSE token matching the block end, break.
            if (curr.type == TokenType.KEYWORD and curr.value in stop_keywords) or \
                    (curr.type == TokenType.CLOSE and curr.value in stop_keywords):
                break
                
            if curr.type == TokenType.EOF:
                break
                
            stmts.append(self.parse_statement())
            
        return stmts

    def parse_kg_block(self) -> KGBlock:
        """
        Parse a KG block.
        
        Returns:
            KGBlock AST node
        """
        self.eat(TokenType.KEYWORD, expected_value="KG►")
        declarations = []

        # Parse declarations until we hit the closing marker
        while (self.current_token().type != TokenType.CLOSE and
               self.current_token().type != TokenType.EOF):
            if self.current_token().type == TokenType.KEYWORD:
                if self.current_token().value == "KGNODE►":
                    declarations.append(self.parse_kg_node())
                elif self.current_token().value == "KGLINK►":
                    declarations.append(self.parse_kg_edge())
                else:
                    raise KGMLSyntaxError(
                        f"Unexpected keyword in KG block",
                        self.current_token().pos,
                        self.current_token().value
                    )
            else:
                # Skip any non-keyword tokens (like newlines)
                self.pos += 1

        # Expect the closing marker
        self.eat(TokenType.CLOSE, expected_value="◄")
        return KGBlock(declarations)

    def parse_kg_node(self) -> KGNodeDeclaration:
        """
        Parse a KGNODE declaration.
        
        Returns:
            KGNodeDeclaration AST node
        """
        self.eat(TokenType.KEYWORD, expected_value="KGNODE►")
        uid_token = self.eat(TokenType.IDENT)
        uid = uid_token.value

        # Expect a colon
        self.eat(TokenType.SYMBOL, expected_value=":")

        # Parse the field list
        fields = self._parse_field_list()

        return KGNodeDeclaration(uid, fields)

    def parse_kg_edge(self) -> KGEdgeDeclaration:
        """
        Parse a KGLINK declaration.
        
        Returns:
            KGEdgeDeclaration AST node
        """
        self.eat(TokenType.KEYWORD, expected_value="KGLINK►")
        source_token = self.eat(TokenType.IDENT)
        source_uid = source_token.value

        # Expect the arrow operator
        self.eat(TokenType.OP, expected_value="->")

        target_token = self.eat(TokenType.IDENT)
        target_uid = target_token.value

        # Expect a colon
        self.eat(TokenType.SYMBOL, expected_value=":")

        # Parse the field list
        fields = self._parse_field_list()

        return KGEdgeDeclaration(source_uid, target_uid, fields)

    def _parse_field_list(self) -> Dict[str, str]:
        """
        Parse a list of key-value field assignments.
        
        Returns:
            Dictionary of field names to values
        """
        fields = {}

        # Helper function to parse a single key-value pair
        def parse_key_value() -> Tuple[str, str]:
            # Parse key
            key = self.eat(TokenType.IDENT).value
            
            # Unified approach to accept either OP or SYMBOL for equals sign
            if self.current_token().type == TokenType.OP and self.current_token().value == "=":
                self.eat(TokenType.OP)
            elif self.current_token().type == TokenType.SYMBOL and self.current_token().value == "=":
                self.eat(TokenType.SYMBOL)
            else:
                raise KGMLSyntaxError(
                    f"Expected '=' after field name",
                    self.current_token().pos,
                    self.current_token().value
                )
                
            # Parse value
            value_token = self.eat(TokenType.STRING)
            value = self._unquote(value_token.value)
            
            return key, value

        # Parse the first key-value pair
        key, value = parse_key_value()
        fields[key] = value

        # Parse additional fields separated by commas
        while self.current_token().type == TokenType.SYMBOL and self.current_token().value == ",":
            self.eat(TokenType.SYMBOL, expected_value=",")
            key, value = parse_key_value()
            fields[key] = value

        return fields

    def _unquote(self, s: str) -> str:
        """
        Remove quotes from a string and process escape sequences.
        
        Args:
            s: The quoted string
            
        Returns:
            Unquoted string with escape sequences processed
        """
        if s.startswith('"') and s.endswith('"'):
            return bytes(s[1:-1], "utf-8").decode("unicode_escape")
        return s


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def parse_kgml(kgml_code: str) -> Program:
    """
    Parse KGML code into an AST.
    
    Args:
        kgml_code: The KGML code string
        
    Returns:
        Program AST node
        
    Raises:
        KGMLError: If there's an error during parsing
    """
    try:
        tokens = tokenize(kgml_code)
        parser = Parser(tokens)
        return parser.parse_program()
    except KGMLError as e:
        # Re-raise KGML errors directly
        raise
    except Exception as e:
        # Wrap other exceptions in KGMLError
        raise KGMLError(f"Unexpected error during parsing: {str(e)}")


def format_error_location(kgml_code: str, error_pos: int, context_lines: int = 2) -> str:
    """
    Format a user-friendly error location for KGML errors.
    
    Args:
        kgml_code: The KGML code string
        error_pos: Position of the error
        context_lines: Number of context lines to show
        
    Returns:
        Formatted error location string
    """
    # Get lines before and after the error
    lines = kgml_code.split('\n')
    
    # Find line and column of the error
    line_start = 0
    line_num = 1
    for i, line in enumerate(lines):
        if line_start + len(line) + 1 > error_pos:
            # Found the line containing the error
            col_num = error_pos - line_start + 1
            break
        line_start += len(line) + 1
        line_num += 1
    else:
        # Error is at the end of the file
        line_num = len(lines)
        col_num = len(lines[-1]) + 1
    
    # Build the context display
    start_line = max(1, line_num - context_lines)
    end_line = min(len(lines), line_num + context_lines)
    
    result = []
    result.append(f"Error at line {line_num}, column {col_num}:")
    result.append("")
    
    for i in range(start_line, end_line + 1):
        if i == line_num:
            # Line with the error
            result.append(f"{i:4d} > {lines[i-1]}")
            # Add a caret pointing to the error position
            result.append(f"       {' ' * (col_num - 1)}^")
        else:
            # Context line
            result.append(f"{i:4d}   {lines[i-1]}")
    
    return '\n'.join(result)
