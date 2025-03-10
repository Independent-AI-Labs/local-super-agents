import re

import pytest

from integration.data.config import KGML_SYSTEM_PROMPT
from integration.net.ollama.ollama_api import prompt_model  # This is the real API call.
from knowledge.reasoning.dsl.kgml_parser import Parser, tokenize

# ------------------------------------------------------------------------------
# Global Constant for Model
# ------------------------------------------------------------------------------
CURRENT_MODEL = "qwen2.5-coder:14b"


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def parser():
    """
    Returns a function that creates a Parser instance for the given KGML text.
    This is a factory fixture that creates a new parser for each test.
    """

    def _create_parser(kgml_text):
        tokens = tokenize(kgml_text)
        return Parser(tokens)

    return _create_parser


@pytest.fixture
def reasoning_stats():
    """
    Holds global reasoning process statistics.
    Keys:
      - total_prompts: Total number of prompts sent.
      - valid_responses: Number of responses that parsed correctly.
      - invalid_responses: Number of responses that failed to parse.
      - errors: List of error messages.
    """
    return {
        "total_prompts": 0,
        "valid_responses": 0,
        "invalid_responses": 0,
        "errors": []
    }


@pytest.fixture
def initial_kg_serialized():
    """
    Returns the initial serialized Knowledge Graph prompt as a plain text string.
    The serialization conforms to the required format:

    KG►
    KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-02-14T13:24:33.347883", message="User inquiry regarding sensor data manage"
    KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Process the current KG and propose the next reasoning step"
    ◄
    """
    return (
        'KG►\n'
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-02-14T13:24:33.347883", message="User inquiry regarding sensor data manage"\n'
        'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Process the current KG and propose the next reasoning step"\n'
        '◄'
    )


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def is_plain_text(prompt: str) -> bool:
    """
    Validate that the prompt is a plain-text string.
    We check that it does not start with '{' (indicating JSON)
    and that it contains expected KGML markers.
    """
    return (not prompt.strip().startswith("{") and
            re.search(r'(KG►|KGNODE►|KGLINK►|C►|U►|D►|E►|N►)', prompt) is not None)


def validate_kgml(kgml_text: str) -> bool:
    """
    Validate that the provided text is valid KGML by attempting to parse it.
    Returns True if parsing succeeds, False otherwise.
    """
    try:
        tokens = tokenize(kgml_text)
        parser = Parser(tokens)
        _ = parser.parse_program()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Parser Tests
# ------------------------------------------------------------------------------

def test_tokenize_valid_kgml():
    """
    Test that tokenization works correctly for valid KGML.
    """
    valid_kgml = 'C► NODE TestNode "Create test node" ◄'
    tokens = tokenize(valid_kgml)
    assert len(tokens) > 0

    # Check for expected tokens
    assert tokens[0].type == "KEYWORD"
    assert tokens[0].value == "C►"
    assert tokens[1].type == "IDENT"
    assert tokens[1].value == "NODE"
    assert tokens[2].type == "IDENT"
    assert tokens[2].value == "TestNode"
    assert tokens[3].type == "STRING"
    assert tokens[3].value == '"Create test node"'
    assert tokens[4].type == "CLOSE"
    assert tokens[4].value == "◄"


def test_parser_valid_kgml(parser):
    """
    Ensure the parser accepts a well-formed KGML string.
    """
    valid_kgml = (
        'C► NODE TestNode "Create test node" ◄\n'
        'IF► E► NODE TestNode "Evaluate test node" ◄\n'
        '    C► NODE TestAlert "Trigger test alert" ◄\n'
        'ELSE►\n'
        '    D► NODE TestNode "Delete test node" ◄\n'
        '◄\n'
    )
    parser_instance = parser(valid_kgml)
    ast = parser_instance.parse_program()
    assert ast is not None
    assert hasattr(ast, "statements")
    assert len(ast.statements) == 2  # Create command and IF statement

    # Check the first statement (Create command)
    assert ast.statements[0].cmd_type == "C►"
    assert ast.statements[0].entity_type == "NODE"
    assert ast.statements[0].uid == "TestNode"
    assert ast.statements[0].instruction == "Create test node"

    # Check the second statement (IF conditional)
    assert hasattr(ast.statements[1], "if_clause")
    condition_cmd, if_block = ast.statements[1].if_clause

    # Check the condition command
    assert condition_cmd.cmd_type == "E►"
    assert condition_cmd.entity_type == "NODE"
    assert condition_cmd.uid == "TestNode"

    # Check the if block
    assert len(if_block) == 1
    assert if_block[0].cmd_type == "C►"
    assert if_block[0].entity_type == "NODE"
    assert if_block[0].uid == "TestAlert"

    # Check the else block
    assert hasattr(ast.statements[1], "else_clause")
    assert len(ast.statements[1].else_clause) == 1
    assert ast.statements[1].else_clause[0].cmd_type == "D►"
    assert ast.statements[1].else_clause[0].entity_type == "NODE"
    assert ast.statements[1].else_clause[0].uid == "TestNode"


def test_tokenize_kg_block():
    """
    Test that tokenization works correctly for KG blocks with nodes and links.
    """
    kg_block = (
        'KG►\n'
        'KGNODE► NodeA : type="TestNode"\n'
        'KGLINK► NodeA -> NodeB : type="TestLink"\n'
        '◄'
    )

    tokens = tokenize(kg_block)
    assert len(tokens) > 0

    # Check for expected tokens
    token_types = [token.type for token in tokens]
    token_values = [token.value for token in tokens]

    # Check for KG block markers
    assert "KEYWORD" in token_types
    assert "KG►" in token_values
    assert "KGNODE►" in token_values
    assert "KGLINK►" in token_values

    # Check for symbols and operators
    assert "SYMBOL" in token_types
    assert ":" in token_values
    assert "OP" in token_types
    assert "->" in token_values


def test_parser_valid_kg_block(parser):
    """
    Ensure the parser accepts a well-formed KG block.
    """
    valid_kg = (
        'KG►\n'
        'KGNODE► NodeA : type="TestNode", description="This is a test node"\n'
        'KGNODE► NodeB : type="TestNode", status="active", priority="high"\n'
        'KGLINK► NodeA -> NodeB : type="TestLink", weight="5"\n'
        '◄\n'
    )
    parser_instance = parser(valid_kg)
    ast = parser_instance.parse_program()
    assert ast is not None
    assert hasattr(ast, "statements")
    assert len(ast.statements) == 1
    assert hasattr(ast.statements[0], "declarations")
    assert len(ast.statements[0].declarations) == 3

    # Verify node declarations
    node_declarations = [d for d in ast.statements[0].declarations if hasattr(d, "uid")]
    assert len(node_declarations) == 2
    assert node_declarations[0].uid == "NodeA"
    assert node_declarations[0].fields["type"] == "TestNode"
    assert node_declarations[0].fields["description"] == "This is a test node"

    # Verify link declaration
    link_declarations = [d for d in ast.statements[0].declarations if hasattr(d, "source_uid")]
    assert len(link_declarations) == 1
    assert link_declarations[0].source_uid == "NodeA"
    assert link_declarations[0].target_uid == "NodeB"
    assert link_declarations[0].fields["type"] == "TestLink"
    assert link_declarations[0].fields["weight"] == "5"


def test_parser_invalid_kgml(parser):
    """
    Ensure the parser raises SyntaxError for a malformed KGML string.
    (e.g. missing the closing marker ◄)
    """
    invalid_kgml = (
        'C► NODE TestNode "Create test node" ◄\n'
        'E► NODE TestNode "Evaluate test node" '  # Missing ◄ here
    )
    parser_instance = parser(invalid_kgml)
    with pytest.raises(SyntaxError):
        _ = parser_instance.parse_program()


def test_parser_invalid_kg_block(parser):
    """
    Ensure the parser raises SyntaxError for a malformed KG block.
    """
    invalid_kg = (
        'KG►\n'
        'KGNODE► NodeA : type="TestNode"\n'
        'KGLINK► NodeA -> '  # Missing target and closing
    )
    parser_instance = parser(invalid_kg)
    with pytest.raises(SyntaxError):
        _ = parser_instance.parse_program()


def test_parser_loop_command(parser):
    """
    Ensure the parser correctly handles LOOP commands.
    """
    loop_kgml = (
        'LOOP► "Iterate through all data points" ◄\n'
        '    C► NODE DataPoint "Create new data point" ◄\n'
        '    E► NODE DataPoint "Validate data point" ◄\n'
        '◄\n'
    )
    parser_instance = parser(loop_kgml)
    ast = parser_instance.parse_program()
    assert ast is not None
    assert hasattr(ast, "statements")
    assert len(ast.statements) == 1
    assert hasattr(ast.statements[0], "condition")
    assert ast.statements[0].condition == "Iterate through all data points"

    # Check the block content
    assert hasattr(ast.statements[0], "block")
    assert len(ast.statements[0].block) == 2

    # Verify the commands inside the loop
    assert ast.statements[0].block[0].cmd_type == "C►"
    assert ast.statements[0].block[0].entity_type == "NODE"
    assert ast.statements[0].block[0].uid == "DataPoint"
    assert ast.statements[0].block[0].instruction == "Create new data point"

    assert ast.statements[0].block[1].cmd_type == "E►"
    assert ast.statements[0].block[1].entity_type == "NODE"
    assert ast.statements[0].block[1].uid == "DataPoint"
    assert ast.statements[0].block[1].instruction == "Validate data point"


# ------------------------------------------------------------------------------
# Integration Tests with Real Prompts
# ------------------------------------------------------------------------------

def test_serialized_prompt_is_plain_text(initial_kg_serialized):
    """
    Verify that the initial serialized KG prompt is plain text
    and contains the required KGML markers.
    """
    assert isinstance(initial_kg_serialized, str)
    assert is_plain_text(initial_kg_serialized)


def test_model_returns_valid_kgml(initial_kg_serialized, reasoning_stats):
    """
    Send the initial KG prompt to the model and verify that the response is valid KGML.
    """
    reasoning_stats["total_prompts"] += 1
    prompt = initial_kg_serialized
    response = prompt_model(prompt, model=CURRENT_MODEL, system_prompt=KGML_SYSTEM_PROMPT)
    print("\n=== Model Response ===\n", response)
    if validate_kgml(response):
        reasoning_stats["valid_responses"] += 1
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append("Model returned invalid KGML.")
    assert validate_kgml(response), "Model response did not parse as valid KGML."


def test_complex_kgml_structures(reasoning_stats):
    """
    Test the model's ability to handle complex KGML structures.
    """
    complex_prompt = (
        'KG►\n'
        'KGNODE► Problem : type="ProblemNode", description="Need to analyze sensor data for anomalies"\n'
        'KGNODE► Context : type="ContextNode", domain="IoT", dataSource="temperature_sensors"\n'
        'KGLINK► Problem -> Context : type="HasContext", priority="high"\n'
        '◄\n'
        'C► NODE Plan "Create a plan to analyze sensor data" ◄\n'
    )

    reasoning_stats["total_prompts"] += 1
    response = prompt_model(complex_prompt, model=CURRENT_MODEL, system_prompt=KGML_SYSTEM_PROMPT)
    print("\n=== Complex Structure Response ===\n", response)

    if validate_kgml(response):
        reasoning_stats["valid_responses"] += 1
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append("Complex structure test: Response not valid KGML.")

    assert validate_kgml(response), "Complex KGML structure should yield valid KGML response."


def test_edge_case_long_prompt(initial_kg_serialized, reasoning_stats):
    """
    Test behavior with a very long prompt simulating many reasoning steps.
    """
    long_prompt = (initial_kg_serialized + "\n") * 5
    response = prompt_model(long_prompt, model=CURRENT_MODEL, system_prompt=KGML_SYSTEM_PROMPT)
    print("\n=== Long Prompt Response ===\n", response)
    if validate_kgml(response):
        reasoning_stats["valid_responses"] += 1
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append("Long prompt test: Response not valid KGML.")
    assert validate_kgml(response), "Long prompt should yield valid KGML response."
