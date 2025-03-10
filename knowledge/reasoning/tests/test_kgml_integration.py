"""
Integration Tests for KGML execution and reasoning.

These tests verify the integration between the KGML parser, executor, and LLM-based reasoning.
"""
import logging
import time
from pathlib import Path

from integration.data.config import KGML_SYSTEM_PROMPT
from knowledge.reasoning.dsl.kgml_executor import KGMLExecutor
from knowledge.reasoning.tests.util.kgml_test_helpers import (
    validate_kgml_with_error,
    format_reasoning_summary
)
from knowledge.reasoning.tests.util.kgml_test_logger import KGMLTestLogger
from knowledge.reasoning.tests.util.kgml_test_parameters import *
from knowledge.reasoning.tests.util.kgml_test_reasoning_evaluator import ReasoningEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLIntegrationTests")


# ------------------------------------------------------------------------------
# Test Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def test_logger():
    """
    Provides a test logger for the entire test session.
    """
    test_logger = KGMLTestLogger(base_dir="kgml_test_logs", model_name=CURRENT_MODEL)
    yield test_logger
    # Finalize the test run when the fixture is torn down
    test_logger.end_run()


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_basic_kg_init_and_serialization(knowledge_graph, initial_kg_serialized, test_logger):
    """
    Test that we can initialize a KG from a serialized string and re-serialize it.
    """
    test_logger.start_test("basic_kg_init_and_serialization", {"description": "Basic KG initialization and serialization test"})

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    kg = evaluator.initialize_kg_from_serialized(initial_kg_serialized)

    # Check that nodes were properly created
    assert len(kg.query_nodes()) > 0
    assert kg.get_node("EventMeta_1") is not None
    assert kg.get_node("ActionMeta_1") is not None

    # Check serialization
    serialized = evaluator.serialize_kg()
    assert "EventMeta_1" in serialized
    assert "ActionMeta_1" in serialized
    assert serialized.startswith("KG►")
    assert serialized.endswith("◄")

    test_logger.end_test("basic_kg_init_and_serialization", goal_reached=True, iterations_to_goal=1)


def test_single_kgml_execution(knowledge_graph, test_logger):
    """
    Test that we can execute a single KGML command sequence directly.
    """
    test_logger.start_test("single_kgml_execution", {"description": "Direct KGML execution test"})

    executor = KGMLExecutor(knowledge_graph)

    # Simple KGML program with create and evaluate commands
    kgml_code = (
        'C► NODE TestNode "Create a test node for validation" ◄\n'
        'E► NODE TestNode "Evaluate if test node is successful" ◄'
    )

    start_time = time.time()
    # Execute the KGML
    context = executor.execute(kgml_code)
    end_time = time.time()
    execution_time = end_time - start_time

    # Log the execution
    test_logger.log_request_response(
        test_name="single_kgml_execution",
        iteration=1,
        request=kgml_code,
        response="Direct execution - no response",
        response_time=execution_time,  # Use actual execution time
        is_valid=True,
        has_syntax_errors=False,
        execution_result={
            "success": True,
            "execution_log": context.execution_log,
            "variables": context.variables,
            "results": context.results,
            "execution_time": execution_time  # Include execution time in result
        }
    )

    # Verify execution
    assert len(context.execution_log) == 2  # Two commands executed
    assert knowledge_graph.get_node("TestNode") is not None  # Node was created
    assert "eval_TestNode" in context.variables  # Evaluation result was stored
    assert context.results["TestNode"] is not None  # Result stored in results dict

    test_logger.end_test("single_kgml_execution", goal_reached=True, iterations_to_goal=1)


def test_model_kgml_generation(initial_kg_serialized, reasoning_stats, test_logger):
    """
    Test that the model can generate valid KGML in response to a KG state.
    """
    test_logger.start_test("model_kgml_generation", {"description": "Testing model's ability to generate valid KGML"})

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    evaluator.initialize_kg_from_serialized(initial_kg_serialized)

    # Count the prompt
    reasoning_stats["total_prompts"] += 1

    # Get model response
    response = evaluator.prompt_model_with_kg(initial_kg_serialized, "model_kgml_generation", 1)
    print("\n=== Model Response ===\n", response)

    # Validate and track stats
    is_valid, error_message = validate_kgml_with_error(response)
    if is_valid:
        reasoning_stats["valid_responses"] += 1

        # Execute the KGML
        result = evaluator.execute_kgml(response, "model_kgml_generation", 1)
        reasoning_stats["execution_results"].append(result)

        # Verify the model did something reasonable
        assert result["success"], "KGML execution failed"
        assert len(result["execution_log"]) > 0, "No commands were executed"

        test_logger.end_test("model_kgml_generation", goal_reached=True, iterations_to_goal=1)
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append(f"Model returned invalid KGML: {error_message}")

        test_logger.end_test("model_kgml_generation", goal_reached=False)

    # Assert validity
    assert is_valid, f"Model response was not valid KGML: {error_message}"


def test_multi_step_reasoning(initial_kg_serialized, reasoning_stats, test_logger):
    """
    Test the model's ability to engage in multi-step reasoning through
    iterative KGML generation and execution.
    """
    test_logger.start_test("multi_step_reasoning", {"description": "Testing model's multi-step reasoning capability"})

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    evaluator.initialize_kg_from_serialized(initial_kg_serialized)

    current_kg = initial_kg_serialized
    iterations = 3

    for i in range(iterations):
        print(f"\n=== Iteration {i + 1} ===")
        print("Current KG state:", current_kg)

        # Count the prompt
        reasoning_stats["total_prompts"] += 1

        # Get model response
        response = evaluator.prompt_model_with_kg(current_kg, "multi_step_reasoning", i + 1)
        print(f"\nModel Response {i + 1}:", response)

        # Validate and execute
        is_valid, error_message = validate_kgml_with_error(response)
        if is_valid:
            reasoning_stats["valid_responses"] += 1
            result = evaluator.execute_kgml(response, "multi_step_reasoning", i + 1)
            reasoning_stats["execution_results"].append(result)

            # Update the KG for the next iteration
            current_kg = evaluator.serialize_kg()
        else:
            reasoning_stats["invalid_responses"] += 1
            reasoning_stats["errors"].append(f"Iteration {i + 1}: Invalid KGML: {error_message}")
            break

    # Final checks
    kg_nodes = evaluator.kg.query_nodes()
    kg_growth = len(kg_nodes) > 2

    test_logger.end_test(
        "multi_step_reasoning",
        goal_reached=kg_growth and reasoning_stats["valid_responses"] >= 1,
        iterations_to_goal=iterations
    )

    assert reasoning_stats["valid_responses"] >= 1, "Expected at least one valid response"
    assert len(kg_nodes) > 2, "Expected KG to grow during reasoning"


@pytest.mark.parametrize("difficulty", PROBLEM_DIFFICULTY_LEVELS)
def test_problem_solving_by_difficulty(problem_definitions, reasoning_stats, test_logger, difficulty):
    """
    Test the model's ability to solve problems of different difficulty levels.
    """
    problems = [p for p in problem_definitions if p["difficulty"] == difficulty]
    if not problems:
        pytest.skip(f"No problems defined for difficulty: {difficulty}")

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)

    for problem in problems:
        test_name = f"problem_solving_{difficulty}_{problem['id']}"
        print(f"\n=== Testing Problem: {problem['id']} ({difficulty}) ===")
        print(f"Description: {problem['description']}")

        # Evaluate the model's reasoning on this problem
        result = evaluator.evaluate_reasoning(
            problem,
            max_iterations=MAX_ITERATIONS,
            test_name=test_name
        )
        reasoning_stats["reasoning_success"].append(result)

        # Update global stats
        reasoning_stats["total_prompts"] += result["iterations"]
        reasoning_stats["valid_responses"] += sum(1 for r in result["execution_results"] if r.get("success", False))
        reasoning_stats["invalid_responses"] += sum(1 for r in result["execution_results"] if not r.get("success", False))

        if not result["goal_reached"]:
            reasoning_stats["errors"].append(f"Failed to solve problem {problem['id']}")

        # Print detailed results
        print(f"Goal reached: {result['goal_reached']} in {result['iterations']} iterations")

        # For basic problems, we expect success
        if difficulty == "basic":
            assert result["goal_reached"], f"Failed to solve basic problem: {problem['id']}"

    # Print summary for this difficulty level
    success_count = sum(1 for r in reasoning_stats["reasoning_success"]
                        if r["difficulty"] == difficulty and r["goal_reached"])
    total_count = len([r for r in reasoning_stats["reasoning_success"] if r["difficulty"] == difficulty])

    print(f"\nSummary for {difficulty} problems: {success_count}/{total_count} solved")


def test_comprehensive_reasoning_evaluation(problem_definitions, reasoning_stats, test_logger):
    """
    Comprehensive test that evaluates all problems and reports detailed statistics.
    """
    test_logger.start_test("comprehensive_evaluation", {"description": "Comprehensive evaluation of all problems"})

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    all_results = []

    for problem in problem_definitions:
        test_name = f"comprehensive_{problem['id']}"
        print(f"\n=== Evaluating Problem: {problem['id']} ({problem['difficulty']}) ===")
        result = evaluator.evaluate_reasoning(problem, max_iterations=MAX_ITERATIONS, test_name=test_name)
        all_results.append(result)

    # Compile and print summary
    summary = format_reasoning_summary(all_results, PROBLEM_DIFFICULTY_LEVELS)
    print("\n" + summary)

    # Save summary to file in the test logger directory
    summary_file = Path(test_logger.run_dir) / "comprehensive_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)

    # Store in stats
    reasoning_stats["evaluation_summary"] = summary
    reasoning_stats["reasoning_success"] = all_results

    # Update global stats
    total_iterations = sum(r["iterations"] for r in all_results)
    reasoning_stats["total_prompts"] += total_iterations

    valid_responses = 0
    invalid_responses = 0
    syntax_errors = 0
    execution_errors = 0

    for result in all_results:
        for exec_result in result["execution_results"]:
            if exec_result.get("success", False):
                valid_responses += 1
            else:
                invalid_responses += 1
                if "syntax" in exec_result.get("error", "").lower():
                    syntax_errors += 1
                else:
                    execution_errors += 1

    reasoning_stats["valid_responses"] += valid_responses
    reasoning_stats["invalid_responses"] += invalid_responses
    reasoning_stats["syntax_errors"] = syntax_errors
    reasoning_stats["execution_errors"] = execution_errors

    # Assert reasonable performance overall
    success_rate = sum(1 for r in all_results if r["goal_reached"]) / len(all_results)

    # Update test stats
    test_logger.end_test(
        "comprehensive_evaluation",
        goal_reached=success_rate >= 0.5,
        iterations_to_goal=None
    )

    assert success_rate >= 0.5, "Overall success rate below 50%"


def test_complex_kgml_structures(reasoning_stats, test_logger):
    """
    Test the model's ability to handle complex KGML structures.
    """
    test_logger.start_test("complex_kgml_structures", {"description": "Testing model with complex KGML structures"})

    complex_prompt = (
        'KG►\n'
        'KGNODE► Problem : type="ProblemNode", description="Need to analyze sensor data for anomalies"\n'
        'KGNODE► Context : type="ContextNode", domain="IoT", dataSource="temperature_sensors"\n'
        'KGLINK► Problem -> Context : type="HasContext", priority="high"\n'
        '◄\n'
        'C► NODE Plan "Create a plan to analyze sensor data" ◄\n'
    )

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)

    reasoning_stats["total_prompts"] += 1
    response = evaluator.prompt_model_with_kg(complex_prompt, "complex_kgml_structures", 1)
    print("\n=== Complex Structure Response ===\n", response)

    is_valid, error_message = validate_kgml_with_error(response)
    if is_valid:
        reasoning_stats["valid_responses"] += 1

        # Execute the KGML
        result = evaluator.execute_kgml(response, "complex_kgml_structures", 1)
        reasoning_stats["execution_results"].append(result)

        test_logger.end_test("complex_kgml_structures", goal_reached=True, iterations_to_goal=1)
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append(f"Complex structure test: {error_message}")

        test_logger.end_test("complex_kgml_structures", goal_reached=False)

    assert is_valid, f"Complex KGML structure should yield valid KGML response: {error_message}"
