import re

import pytest

from integration.net.ollama.ollama_api import prompt_model  # This is the real API call.
from knowledge.reasoning.dsl.kgml_parser import parse_kgml

# ------------------------------------------------------------------------------
# Global Constant for Model
# ------------------------------------------------------------------------------
CURRENT_MODEL = "qwen2.5-coder:14b"


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

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
    """
    return (
        'KG►\n'
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-02-14T13:24:33.347883", message="User inquiry regarding sensor data manage"\n'
        'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Process the current KG and propose the next reasoning step"'
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
            re.search(r'(KG►|KGNODE►)', prompt) is not None)


def validate_kgml(kgml_text: str) -> bool:
    """
    Validate that the provided text is valid KGML by attempting to parse it.
    Returns True if parsing succeeds, False otherwise.
    """
    try:
        _ = parse_kgml(kgml_text)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Parser Tests
# ------------------------------------------------------------------------------

def test_parser_valid_kgml():
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
    ast = parse_kgml(valid_kgml)
    assert ast is not None
    assert hasattr(ast, "statements")
    assert len(ast.statements) > 0


def test_parser_invalid_kgml():
    """
    Ensure the parser raises SyntaxError for a malformed KGML string.
    (e.g. missing the closing marker ◄)
    """
    invalid_kgml = (
        'C► NODE TestNode "Create test node" ◄\n'
        'E► NODE TestNode "Evaluate test node" '  # Missing ◄ here
    )
    with pytest.raises(SyntaxError):
        _ = parse_kgml(invalid_kgml)


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
    response = prompt_model(prompt, model=CURRENT_MODEL)
    print("\n=== Model Response ===\n", response)
    if validate_kgml(response):
        reasoning_stats["valid_responses"] += 1
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append("Model returned invalid KGML.")
    assert validate_kgml(response), "Model response did not parse as valid KGML."


def test_reasoning_workflow(initial_kg_serialized, reasoning_stats):
    """
    Simulate a multi-step reasoning process:
    1. Start with an initial KG prompt.
    2. Send the prompt to the model.
    3. Parse the response and update the serialized KG.
    4. Repeat for several iterations, tracking statistics.
    """
    num_steps = 3
    current_prompt = initial_kg_serialized
    for step in range(num_steps):
        reasoning_stats["total_prompts"] += 1
        response = prompt_model(current_prompt, model=CURRENT_MODEL)
        print(f"\n=== Step {step + 1} Response ===\n", response)
        if validate_kgml(response):
            reasoning_stats["valid_responses"] += 1
        else:
            reasoning_stats["invalid_responses"] += 1
            reasoning_stats["errors"].append(f"Step {step + 1}: Received invalid KGML response.")
        # Update the prompt by appending the response, simulating evolving reasoning steps.
        current_prompt += "\n" + response

    assert reasoning_stats["total_prompts"] == num_steps
    # Depending on the model, we expect at least one valid response.
    assert reasoning_stats["valid_responses"] >= 1, "Expected at least one valid KGML response."


# def test_empty_prompt(reasoning_stats):
#     """
#     Test behavior when an empty prompt is sent.
#     """
#     empty_prompt = ""
#     response = prompt_model(empty_prompt, model=CURRENT_MODEL)
#     print("\n=== Empty Prompt Response ===\n", response)
#     if not validate_kgml(response):
#         reasoning_stats["invalid_responses"] += 1
#         reasoning_stats["errors"].append("Empty prompt test: Response not valid KGML as expected.")
#     assert not validate_kgml(response), "An empty prompt should not yield valid KGML."
#

def test_edge_case_long_prompt(initial_kg_serialized, reasoning_stats):
    """
    Test behavior with a very long prompt simulating many reasoning steps.
    """
    long_prompt = (initial_kg_serialized + "\n") * 5
    response = prompt_model(long_prompt, model=CURRENT_MODEL)
    print("\n=== Long Prompt Response ===\n", response)
    if validate_kgml(response):
        reasoning_stats["valid_responses"] += 1
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append("Long prompt test: Response not valid KGML.")
    assert validate_kgml(response), "Long prompt should yield valid KGML response."


# ------------------------------------------------------------------------------
# New Iterative Test: Evolving Knowledge Graph Simulation
# ------------------------------------------------------------------------------

def test_iterative_kg_evolution_verbose(initial_kg_serialized, reasoning_stats):
    """
    Simulate an iterative evolution of the Knowledge Graph by sending evolving prompts
    to the LLM using prompt_model. At each iteration:
      - Print the current prompt.
      - Send the prompt to the model and print the response.
      - Validate and parse the KGML response.
      - Print the parsed output.
      - Append the response to the prompt for the next iteration.
      - Print intermediate reasoning statistics.
    This verbose test runs for a larger number of iterations to better trace the evolution.
    """
    num_iterations = 12  # Increased number of iterations
    current_prompt = initial_kg_serialized
    print("\n=== Starting Iterative KG Evolution Verbose Test ===\n", flush=True)

    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Iteration {iteration} ---", flush=True)
        print("Current Prompt:", flush=True)
        print(current_prompt, flush=True)

        # Increment prompt counter and get model response.
        reasoning_stats["total_prompts"] += 1
        response = prompt_model(current_prompt, model=CURRENT_MODEL)
        print("\nModel Response:", flush=True)
        print(response, flush=True)

        # Validate the response.
        if validate_kgml(response):
            reasoning_stats["valid_responses"] += 1
            print("Validation: Response is valid KGML.", flush=True)
        else:
            reasoning_stats["invalid_responses"] += 1
            error_msg = f"Iteration {iteration}: Received invalid KGML response."
            reasoning_stats["errors"].append(error_msg)
            print("Validation Error:", error_msg, flush=True)

        # Attempt to parse the response and print the parsed output.
        try:
            parsed = parse_kgml(response)
            print("\nParsed KGML:", flush=True)
            print(parsed, flush=True)
        except Exception as e:
            reasoning_stats["invalid_responses"] += 1
            error_msg = f"Iteration {iteration}: KGML parsing failed with error: {str(e)}"
            reasoning_stats["errors"].append(error_msg)
            print("Parsing Error:", error_msg, flush=True)
            raise

        # Update the prompt for the next iteration.
        current_prompt = f"{current_prompt}\n{response}"
        print("\nUpdated Prompt:", flush=True)
        print(current_prompt, flush=True)

        # Print intermediate reasoning statistics.
        print("\nIntermediate Reasoning Stats:", flush=True)
        print(reasoning_stats, flush=True)

    print("\n=== Completed Iterative KG Evolution Verbose Test ===\n", flush=True)
    # Final check: ensure that the evolving prompt is not empty.
    assert current_prompt.strip() != "", "The evolving Knowledge Graph prompt should not be empty after iterations."
