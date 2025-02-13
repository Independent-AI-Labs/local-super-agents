import re
from typing import List, Optional, Tuple


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------

class TokenType:
    KEYWORD = "KEYWORD"
    CLOSE = "CLOSE"  # For the closing marker: ◄
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    SYMBOL = "SYMBOL"
    OP = "OP"
    EOF = "EOF"


class Token:
    def __init__(self, type_: str, value: str, pos: int):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


# Updated reserved keywords per new grammar.
RESERVED = {
    "C►", "U►", "D►", "E►", "N►",
    "IF►", "ELIF►", "ELSE►", "LOOP►", "END►"
}

# Updated TOKEN_REGEX: we now recognize the closing marker "◄"
TOKEN_REGEX = re.compile(
    r"""
    (?P<KEYWORD>(?:C►|U►|D►|E►|N►|IF►|ELIF►|ELSE►|LOOP►|END►))|
    (?P<CLOSE>◄)|
    (?P<NUMBER>\d+(\.\d+)?)|
    (?P<STRING>"([^"\\]|\\.)*")|
    (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)|
    (?P<OP>(==|!=|<=|>=|[+\-*/<>]))|
    (?P<SYMBOL>[{}(),:;])|
    (?P<WS>\s+)
    """, re.VERBOSE
)


def tokenize(text: str) -> List[Token]:
    pos = 0
    tokens = []
    while pos < len(text):
        match = TOKEN_REGEX.match(text, pos)
        if not match:
            raise SyntaxError(f"Unexpected character at pos {pos}: {text[pos]}")
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
# AST Node Classes
# ------------------------------------------------------------------------------

class ASTNode:
    pass


class Program(ASTNode):
    def __init__(self, statements: List[ASTNode]):
        self.statements = statements

    def __repr__(self):
        return f"Program({self.statements})"


# A simple command now includes additional fields for entity type, uid, and an optional timeout.
class SimpleCommand(ASTNode):
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


# A conditional block; it may have multiple clauses (if/elif) and an optional else clause.
class ConditionalCommand(ASTNode):
    def __init__(self, if_clause: Tuple[ASTNode, List[ASTNode]],
                 elif_clauses: List[Tuple[ASTNode, List[ASTNode]]],
                 else_clause: Optional[List[ASTNode]]):
        self.if_clause = if_clause  # Tuple (condition-command, block)
        self.elif_clauses = elif_clauses  # List of tuples (condition-command, block)
        self.else_clause = else_clause  # Block (list of statements) or None

    def __repr__(self):
        return (f"ConditionalCommand(if={self.if_clause}, "
                f"elif={self.elif_clauses}, else={self.else_clause})")


# A loop block.
class LoopCommand(ASTNode):
    def __init__(self, condition: str, block: List[ASTNode]):
        self.condition = condition  # Loop condition as a natural language string
        self.block = block

    def __repr__(self):
        return f"LoopCommand({self.condition}, {self.block})"


# ------------------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def eat(self, expected_type: Optional[str] = None, expected_value: Optional[str] = None) -> Token:
        token = self.current_token()
        if expected_type and token.type != expected_type:
            raise SyntaxError(f"Expected token type {expected_type} at pos {token.pos} but got {token.type}")
        if expected_value and token.value != expected_value:
            raise SyntaxError(f"Expected token value {expected_value} at pos {token.pos} but got {token.value}")
        self.pos += 1
        return token

    def parse_program(self) -> Program:
        statements = []
        while self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            statements.append(stmt)
        return Program(statements)

    def parse_statement(self) -> ASTNode:
        token = self.current_token()
        if token.type == TokenType.KEYWORD:
            if token.value in {"C►", "U►", "D►", "E►", "N►"}:
                return self.parse_simple_command()
            elif token.value == "IF►":
                return self.parse_if_command()
            elif token.value == "LOOP►":
                return self.parse_loop_command()
            else:
                raise SyntaxError(f"Unexpected keyword {token.value} at pos {token.pos}")
        else:
            raise SyntaxError(f"Unexpected token {token} at pos {token.pos}")

    def parse_simple_command(self) -> SimpleCommand:
        """
        Parses a simple command.

        For C►, U►, D►, and E►:
            Command format: KEYWORD ws ENTITY_TYPE ws UID ws STRING ws CLOSE
            where ENTITY_TYPE is either NODE or LINK.

        For N► (navigate):
            Command format: N► ws [ NUMBER ws ] STRING ws CLOSE
        """
        cmd_token = self.eat(TokenType.KEYWORD)
        cmd = cmd_token.value

        if cmd in {"C►", "U►", "D►", "E►"}:
            # Expect an entity type (NODE or LINK)
            etype_token = self.eat(TokenType.IDENT)
            entity_type = etype_token.value
            if entity_type not in {"NODE", "LINK"}:
                raise SyntaxError(f"Expected entity type NODE or LINK at pos {etype_token.pos}, got {entity_type}")
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
            raise SyntaxError(f"Unknown simple command {cmd}")

    def parse_if_command(self) -> ConditionalCommand:
        """
        Parses an IF block with the updated grammar.

        Format:
            IF► <condition_command>
                <if_block>
            { ELIF► <condition_command>
                <elif_block> }
            [ ELSE►
                <else_block> ]
            ◄
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
        Parses a LOOP block with the updated grammar.

        Format:
            LOOP► STRING ◄
                <block>
            ◄
        """
        self.eat(TokenType.KEYWORD, expected_value="LOOP►")
        loop_cond_token = self.eat(TokenType.STRING)
        loop_condition = self._unquote(loop_cond_token.value)
        self.eat(TokenType.CLOSE, expected_value="◄")
        block = self.parse_block(stop_keywords={"◄"})
        self.eat(TokenType.CLOSE, expected_value="◄")
        return LoopCommand(loop_condition, block)

    def parse_block(self, stop_keywords: set) -> List[ASTNode]:
        """
        Parse a block of statements until a keyword token whose value is in stop_keywords is encountered.
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

    def _unquote(self, s: str) -> str:
        if s.startswith('"') and s.endswith('"'):
            return bytes(s[1:-1], "utf-8").decode("unicode_escape")
        return s


# ------------------------------------------------------------------------------
# Helper: Parse full KGML text.
# ------------------------------------------------------------------------------

def parse_kgml(text: str) -> Program:
    tokens = tokenize(text)
    parser = Parser(tokens)
    return parser.parse_program()


# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    example_code = r'''
C► NODE Sensor01 "Create a sensor node" ◄
U► NODE Sensor01 "Update sensor configuration" ◄
D► NODE Sensor01 "Delete sensor node" ◄
E► NODE Sensor01 "Evaluate sensor status" ◄
N► 30 "Navigate graph starting from sensor node" ◄
IF► E► NODE ReasonStep_1 "Check evaluation result is successful" ◄
    C► LINK ActionMeta_1 "To ReasonStep_1" ◄
ELSE►
    U► NODE ReasonStep_1 "Update with error message: Evaluation failed" ◄
◄
LOOP► "while sensor reading != 0" ◄
    U► NODE Sensor01 "Adjust sensor parameters" ◄
◄
'''
    try:
        ast = parse_kgml(example_code)
        print("Parsed AST:")
        print(ast)
    except SyntaxError as e:
        print("Syntax error:", e)
