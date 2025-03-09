import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest

from integration.data.config import KGML_SYSTEM_PROMPT
from integration.net.ollama.ollama_api import prompt_model  # This is the real API call
from knowledge.graph.kg_models import KnowledgeGraph, KGNode
from knowledge.reasoning.dsl.kgml_executor import KGMLExecutor
from knowledge.reasoning.dsl.kgml_parser import parse_kgml
from knowledge.reasoning.tests.util.kgml_test_logger import KGMLTestLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLIntegrationTests")

# ------------------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------------------
CURRENT_MODEL = "qwen2.5-coder:14b"
MAX_ITERATIONS = 10
PROBLEM_DIFFICULTY_LEVELS = ["basic", "intermediate", "advanced"]


# ------------------------------------------------------------------------------
# Test Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def test_logger():
    """
    Provides a test logger for the entire test session.
    """
    logger = KGMLTestLogger(base_dir="kgml_test_logs", model_name=CURRENT_MODEL)
    yield logger
    # Finalize the test run when the fixture is torn down
    logger.end_run()


@pytest.fixture
def reasoning_stats():
    """
    Holds global reasoning process statistics.
    Keys:
      - total_prompts: Total number of prompts sent.
      - valid_responses: Number of responses that parsed correctly.
      - invalid_responses: Number of responses that failed to parse.
      - errors: List of error messages.
      - reasoning_success: List of dictionaries with reasoning success metrics.
      - execution_results: List of execution outcomes.
    """
    return {
        "total_prompts": 0,
        "valid_responses": 0,
        "invalid_responses": 0,
        "errors": [],
        "reasoning_success": [],
        "execution_results": []
    }


@pytest.fixture
def knowledge_graph():
    """
    Create and return a fresh KnowledgeGraph instance for testing.
    """
    return KnowledgeGraph()


@pytest.fixture
def initial_kg_serialized():
    """
    Returns the initial serialized Knowledge Graph prompt as a plain text string.
    The serialization conforms to the required format.
    """
    return (
        'KG►\n'
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-02-14T13:24:33.347883", message="User inquiry regarding sensor data processing"\n'
        'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Process the current KG and propose the next reasoning step"\n'
        '◄'
    )


@pytest.fixture
def problem_definitions():
    """
    Return a set of problem definitions for testing reasoning capabilities.
    Each problem includes:
    - description: A natural language description of the problem
    - initial_kg: Initial state of the knowledge graph
    - goal_condition: Success criteria for the problem
    - difficulty: basic, intermediate, or advanced
    """
    return [
        {
            "id": "simple_sensor_check",
            "description": "Create a reasoning step to check if a sensor is active, then create an alert node if it's not.",
            "initial_kg": (
                'KG►\n'
                'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-02-14T13:24:33.347883", message="Check if Sensor01 is active"\n'
                'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Monitor sensor status and create alert if inactive"\n'
                'KGNODE► Sensor01 : type="SensorNode", status="inactive", last_reading="2025-02-14T13:20:00.000000"\n'
                '◄'
            ),
            "goal_condition": lambda kg: any(node.uid.startswith("Alert") for node in kg.query_nodes()),
            "difficulty": "basic"
        },
        {
            "id": "data_processing_sequence",
            "description": "Create a sequence of data processing steps with dependencies between them.",
            "initial_kg": (
                'KG►\n'
                'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-02-14T14:15:22.123456", message="Process sensor data from multiple sources"\n'
                'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Create a data pipeline with collection, validation, transformation and storage steps"\n'
                'KGNODE► DataSource_1 : type="DataSourceNode", format="CSV", update_frequency="hourly"\n'
                'KGNODE► DataSource_2 : type="DataSourceNode", format="JSON", update_frequency="realtime"\n'
                '◄'
            ),
            "goal_condition": lambda kg: (
                # Check for creation of necessary processing nodes
                    len([n for n in kg.query_nodes() if "Collection" in n.uid]) > 0 and
                    len([n for n in kg.query_nodes() if "Validation" in n.uid]) > 0 and
                    len([n for n in kg.query_nodes() if "Transformation" in n.uid]) > 0 and
                    len([n for n in kg.query_nodes() if "Storage" in n.uid]) > 0 and
                    # Check for proper sequencing through links
                    len(kg.query_edges()) >= 3  # At least 3 edges to connect the 4 processing steps
            ),
            "difficulty": "intermediate"
        },
        {
            "id": "complex_conditional_reasoning",
            "description": "Implement a multi-condition decision tree for sensor data processing with different paths based on data quality and type.",
            "initial_kg": (
                'KG►\n'
                'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-02-14T16:45:12.789012", message="Implement conditional processing for sensor data"\n'
                'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Create decision nodes that route data based on quality metrics and data types"\n'
                'KGNODE► DataQuality_1 : type="QualityMetricNode", completeness="87.5", accuracy="92.3", consistency="78.9"\n'
                'KGNODE► SensorType_1 : type="SensorTypeNode", measurement="temperature", unit="celsius", precision="0.1"\n'
                'KGNODE► SensorType_2 : type="SensorTypeNode", measurement="humidity", unit="percent", precision="0.5"\n'
                '◄'
            ),
            "goal_condition": lambda kg: (
                # Check for decision nodes
                    len([n for n in kg.query_nodes() if "Decision" in n.uid]) >= 2 and
                    # Check for processing paths
                    len([n for n in kg.query_nodes() if "ProcessPath" in n.uid]) >= 3 and
                    # Check for conditional evaluation
                    len([n for n in kg.query_nodes() if "Condition" in n.uid]) >= 2 and
                    # Check for proper linking
                    len(kg.query_edges()) >= 6  # Multiple edges needed for the decision tree
            ),
            "difficulty": "advanced"
        }
    ]


# ------------------------------------------------------------------------------
# Helper Classes
# ------------------------------------------------------------------------------

class ReasoningEvaluator:
    """
    Evaluates the reasoning capabilities of a language model using KGML.
    """

    def __init__(self, model_name: str, system_prompt: str, test_logger: Optional[KGMLTestLogger] = None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.kg = KnowledgeGraph()
        self.executor = KGMLExecutor(self.kg)
        self.test_logger = test_logger

    def initialize_kg_from_serialized(self, serialized_kg: str):
        """
        Initialize the knowledge graph from a serialized string representation.
        """
        # This is a simplified parser for the serialized KG format
        # In a production system, you'd want a more robust parser

        # Clear existing graph
        self.kg = KnowledgeGraph()
        self.executor = KGMLExecutor(self.kg)

        lines = serialized_kg.strip().split('\n')
        if lines[0] != 'KG►' or lines[-1] != '◄':
            raise ValueError("Invalid KG serialization format")

        # Process node definitions
        for line in lines[1:-1]:
            if line.startswith('KGNODE►'):
                # Parse the node definition
                match = re.match(r'KGNODE► (\w+) : (.+)', line)
                if not match:
                    continue

                node_id, props_str = match.groups()

                # Parse the properties
                props = {}
                for prop_match in re.finditer(r'(\w+)="([^"]*)"', props_str):
                    key, value = prop_match.groups()
                    props[key] = value

                # Create and add the node
                node_type = props.pop('type', 'GenericNode')
                node = KGNode(uid=node_id, type=node_type, meta_props=props)
                self.kg.add_node(node)

            elif line.startswith('KGLINK►'):
                # Parse the link definition (not implemented in the original)
                # Would follow a similar pattern to nodes
                pass

        return self.kg

    def serialize_kg(self) -> str:
        """
        Serialize the current knowledge graph to a string representation.
        """
        lines = ['KG►']

        # Add all nodes
        for node in self.kg.query_nodes():
            props_str = f'type="{node.type}"'
            for key, value in node.meta_props.items():
                props_str += f', {key}="{value}"'
            lines.append(f'KGNODE► {node.uid} : {props_str}')

        # Add all edges (not in the original format, but could be added)
        # for edge in self.kg.query_edges():
        #     lines.append(f'KGLINK► {edge.source_uid} -> {edge.target_uid} : type="{edge.type}"')

        lines.append('◄')
        return '\n'.join(lines)

    def prompt_model_with_kg(self, serialized_kg: str, test_name: Optional[str] = None, iteration: Optional[int] = None) -> str:
        """
        Send the serialized KG to the language model and get a KGML response.
        Also tracks response time and logs the interaction if a test logger is available.
        """
        start_time = time.time()
        response = prompt_model(
            serialized_kg,
            model=self.model_name,
            system_prompt=self.system_prompt
        )
        end_time = time.time()
        response_time = end_time - start_time

        # Validate the response syntax
        is_valid = validate_kgml(response)
        has_syntax_errors = not is_valid

        # Log the request-response pair if logger is available
        if self.test_logger and test_name and iteration is not None:
            self.test_logger.log_request_response(
                test_name=test_name,
                iteration=iteration,
                request=serialized_kg,
                response=response,
                response_time=response_time,
                is_valid=is_valid,
                has_syntax_errors=has_syntax_errors
            )

        return response

    def execute_kgml(self, kgml_code: str, test_name: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the KGML code and return the execution results.
        Also logs the execution results if a test logger is available.
        """
        try:
            context = self.executor.execute(kgml_code)
            result = {
                "success": True,
                "execution_log": context.execution_log,
                "variables": context.variables,
                "results": context.results
            }

            # Log execution result if logger is available
            if self.test_logger and test_name and iteration is not None:
                # Update the existing log entry with execution results
                self.test_logger.log_request_response(
                    test_name=test_name,
                    iteration=iteration,
                    request="",  # Empty because we're just updating
                    response="",  # Empty because we're just updating
                    response_time=0.0,  # Not relevant for this update
                    is_valid=True,  # Not relevant for this update
                    has_syntax_errors=False,  # Not relevant for this update
                    execution_result=result
                )

            return result

        except Exception as e:
            logger.error(f"KGML execution failed: {e}")
            error_result = {
                "success": False,
                "error": str(e)
            }

            # Log execution error if logger is available
            if self.test_logger and test_name and iteration is not None:
                # Update the existing log entry with execution error
                self.test_logger.log_request_response(
                    test_name=test_name,
                    iteration=iteration,
                    request="",  # Empty because we're just updating
                    response="",  # Empty because we're just updating
                    response_time=0.0,  # Not relevant for this update
                    is_valid=True,  # We know it parsed, but execution failed
                    has_syntax_errors=False,  # Not relevant for this update
                    execution_result=error_result
                )

            return error_result

    def evaluate_reasoning(self, problem: Dict[str, Any], max_iterations: int = 5, test_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model's reasoning ability on a specific problem.

        Args:
            problem: Problem definition with initial_kg and goal_condition
            max_iterations: Maximum number of reasoning iterations
            test_name: Optional name for test logging

        Returns:
            Dictionary with evaluation results
        """
        # Initialize test logger if provided
        if test_name and self.test_logger:
            self.test_logger.start_test(test_name, {
                "problem_id": problem["id"],
                "difficulty": problem["difficulty"],
                "description": problem["description"]
            })

        # Initialize the KG with the problem definition
        self.initialize_kg_from_serialized(problem["initial_kg"])

        current_kg = problem["initial_kg"]
        iterations = 0
        reached_goal = False
        all_responses = []
        execution_results = []

        while iterations < max_iterations and not reached_goal:
            # Get model response
            iteration_number = iterations + 1
            model_response = self.prompt_model_with_kg(
                current_kg,
                test_name=test_name,
                iteration=iteration_number
            )
            all_responses.append(model_response)

            # Validate and execute the response
            if validate_kgml(model_response):
                # Execute the KGML
                result = self.execute_kgml(
                    model_response,
                    test_name=test_name,
                    iteration=iteration_number
                )
                execution_results.append(result)

                # Check if the goal condition is met
                if problem["goal_condition"](self.kg):
                    reached_goal = True

                # Update the KG for the next iteration
                current_kg = self.serialize_kg()
            else:
                logger.error(f"Invalid KGML response in iteration {iterations + 1}")
                execution_results.append({
                    "success": False,
                    "error": "Invalid KGML syntax"
                })

            iterations += 1

        # End the test if we started one
        if test_name and self.test_logger:
            self.test_logger.end_test(
                test_name=test_name,
                goal_reached=reached_goal,
                iterations_to_goal=iterations if reached_goal else None
            )

        # Prepare evaluation results
        return {
            "problem_id": problem["id"],
            "difficulty": problem["difficulty"],
            "iterations": iterations,
            "goal_reached": reached_goal,
            "responses": all_responses,
            "execution_results": execution_results,
            "final_kg_state": self.serialize_kg()
        }


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


def format_reasoning_summary(evaluation_results: List[Dict[str, Any]]) -> str:
    """
    Format a summary of reasoning evaluation results.
    """
    summary = ["## Reasoning Evaluation Summary"]

    # Overall statistics
    total_problems = len(evaluation_results)
    successful_problems = sum(1 for r in evaluation_results if r["goal_reached"])

    summary.append(f"Total problems: {total_problems}")
    summary.append(f"Successfully solved: {successful_problems} ({successful_problems / total_problems * 100:.1f}%)")

    # Results by difficulty level
    for difficulty in PROBLEM_DIFFICULTY_LEVELS:
        problems_at_level = [r for r in evaluation_results if r["difficulty"] == difficulty]
        if problems_at_level:
            success_at_level = sum(1 for r in problems_at_level if r["goal_reached"])
            success_rate = success_at_level / len(problems_at_level) * 100
            summary.append(f"{difficulty.capitalize()} problems: {success_at_level}/{len(problems_at_level)} ({success_rate:.1f}%)")

    # Individual problem results
    summary.append("\n### Detailed Results")
    for result in evaluation_results:
        status = "✅ SOLVED" if result["goal_reached"] else "❌ FAILED"
        iterations = result["iterations"]
        summary.append(f"{result['problem_id']} ({result['difficulty']}): {status} in {iterations} iterations")

    return "\n".join(summary)


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
    response_time = end_time - start_time

    # Log the execution
    test_logger.log_request_response(
        test_name="single_kgml_execution",
        iteration=1,
        request=kgml_code,
        response="Direct execution - no response",
        response_time=response_time,
        is_valid=True,
        has_syntax_errors=False,
        execution_result={
            "success": True,
            "execution_log": context.execution_log,
            "variables": context.variables,
            "results": context.results
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
    if validate_kgml(response):
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
        reasoning_stats["errors"].append("Model returned invalid KGML")

        test_logger.end_test("model_kgml_generation", goal_reached=False)

    # Assert validity
    assert validate_kgml(response), "Model response was not valid KGML"


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
        if validate_kgml(response):
            reasoning_stats["valid_responses"] += 1
            result = evaluator.execute_kgml(response, "multi_step_reasoning", i + 1)
            reasoning_stats["execution_results"].append(result)

            # Update the KG for the next iteration
            current_kg = evaluator.serialize_kg()
        else:
            reasoning_stats["invalid_responses"] += 1
            reasoning_stats["errors"].append(f"Iteration {i + 1}: Invalid KGML")
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
    summary = format_reasoning_summary(all_results)
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
