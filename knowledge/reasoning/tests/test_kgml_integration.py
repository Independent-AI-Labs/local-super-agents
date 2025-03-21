"""
Integration Tests for KGML Reasoning Capabilities

These tests focus on evaluating the LLM's reasoning capabilities through KGML,
emphasizing research, analysis, and iterative refinement rather than sensor data processing.
"""
import logging
import time
from pathlib import Path

from integration.data.config import KGML_SYSTEM_PROMPT
from knowledge.reasoning.dsl.execution.kgml_executor import KGMLExecutor
from knowledge.reasoning.tests.util.kgml_test_helpers import (
    validate_kgml_with_error,
    format_reasoning_summary
)
from knowledge.reasoning.tests.util.kgml_test_logger import KGMLTestLogger
from knowledge.reasoning.tests.util.kgml_test_parameters import *
from knowledge.reasoning.tests.util.kgml_test_reasoning_evaluator import ReasoningEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLReasoningTests")


# ------------------------------------------------------------------------------
# Test Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def test_logger():
    """
    Provides a test logger for the entire test session.
    """
    test_logger = KGMLTestLogger(base_dir="kgml_reasoning_logs", model_name=CURRENT_MODEL)
    yield test_logger
    # Finalize the test run when the fixture is torn down
    test_logger.end_run()


# ------------------------------------------------------------------------------
# Core Reasoning Tests
# ------------------------------------------------------------------------------

def test_kg_initialization(knowledge_graph, initial_kg_serialized, test_logger):
    """
    Test that we can initialize a Knowledge Graph from a serialized string and re-serialize it.
    """
    test_logger.start_test("kg_initialization", {"description": "Basic KG initialization test"})

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

    test_logger.end_test("kg_initialization", goal_reached=True, iterations_to_goal=1)


def test_simple_kgml_execution(knowledge_graph, test_logger):
    """
    Test direct execution of KGML commands to create and evaluate research nodes.
    """
    test_logger.start_test("simple_kgml_execution", {"description": "Direct KGML execution test"})

    executor = KGMLExecutor(knowledge_graph)

    # KGML program focused on creating a research report
    kgml_code = (
        'C► NODE ResearchTopic "Create a DataNode with content about transformer architecture" ◄\n'
        'C► NODE AnalysisNode "Create an analysis of the transformer architecture topic" ◄\n'
        'C► LINK AnalysisNode -> ResearchTopic "Create a HIERARCHY link showing analysis depends on topic" ◄\n'
        'E► NODE AnalysisNode "Evaluate if the analysis is comprehensive" ◄'
    )

    start_time = time.time()
    # Execute the KGML
    context = executor.execute(kgml_code)
    end_time = time.time()
    execution_time = end_time - start_time

    # Log the execution
    test_logger.log_request_response(
        test_name="simple_kgml_execution",
        iteration=1,
        request=kgml_code,
        response="DIRECT EXECUTION",
        response_time=execution_time,
        is_valid=True,
        has_syntax_errors=False,
        execution_result={
            "success": True,
            "execution_log": context.execution_log,
            "variables": context.variables,
            "results": context.results,
            "execution_time": execution_time
        }
    )

    # Verify execution
    assert len(context.execution_log) == 4  # Four commands executed
    assert knowledge_graph.get_node("ResearchTopic") is not None
    assert knowledge_graph.get_node("AnalysisNode") is not None
    assert "eval_AnalysisNode" in context.variables
    assert context.results["AnalysisNode"] is not None

    test_logger.end_test("simple_kgml_execution", goal_reached=True, iterations_to_goal=1)


def test_model_kgml_generation(initial_kg_serialized, reasoning_stats, test_logger):
    """
    Test the model's ability to generate valid KGML in response to a research request.
    """
    test_logger.start_test("model_kgml_generation", {"description": "Testing model's ability to generate valid KGML for research tasks"})

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

        # Check if any DataNode was created
        has_data_node = any(
            entry["command_type"] == "C" and
            "entity_type" in entry["details"] and
            entry["details"]["entity_type"] == "NODE" and
            "instruction" in entry["details"] and
            "DataNode" in entry["details"]["instruction"]
            for entry in result["execution_log"]
        )

        assert has_data_node, "Model should create at least one DataNode for research"

        test_logger.end_test("model_kgml_generation", goal_reached=True, iterations_to_goal=1)
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append(f"Model returned invalid KGML: {error_message}")

        test_logger.end_test("model_kgml_generation", goal_reached=False)

    # Assert validity
    assert is_valid, f"Model response was not valid KGML: {error_message}"


def test_iterative_reasoning(initial_kg_serialized, reasoning_stats, test_logger):
    """
    Test the model's ability to engage in iterative reasoning to progressively
    refine a research analysis through multiple steps.
    """
    test_logger.start_test("iterative_reasoning", {"description": "Testing model's ability to iteratively refine research analysis"})

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
        response = evaluator.prompt_model_with_kg(current_kg, "iterative_reasoning", i + 1)
        print(f"\nModel Response {i + 1}:", response)

        # Validate and execute
        is_valid, error_message = validate_kgml_with_error(response)
        if is_valid:
            reasoning_stats["valid_responses"] += 1
            result = evaluator.execute_kgml(response, "iterative_reasoning", i + 1)
            reasoning_stats["execution_results"].append(result)

            # Update the KG for the next iteration
            current_kg = evaluator.serialize_kg()

            # Check if the model is refining previous work
            has_links = any(
                entry["command_type"] == "C" and
                "entity_type" in entry["details"] and
                entry["details"]["entity_type"] == "LINK"
                for entry in result["execution_log"]
            )

            if i > 0:  # From second iteration onward, expect refinement
                assert has_links, f"In iteration {i + 1}, model should create links to previous work"
        else:
            reasoning_stats["invalid_responses"] += 1
            reasoning_stats["errors"].append(f"Iteration {i + 1}: Invalid KGML: {error_message}")
            break

    # Final checks
    kg_nodes = evaluator.kg.query_nodes()
    kg_edges = evaluator.kg.query_edges()

    # There should be multiple nodes and edges showing iterative refinement
    assert len(kg_nodes) >= 4, "Expected KG to have multiple nodes showing iterative work"
    assert len(kg_edges) >= 2, "Expected KG to have links showing relationships between iterations"

    test_logger.end_test(
        "iterative_reasoning",
        goal_reached=reasoning_stats["valid_responses"] >= 2,  # At least 2 valid iterations
        iterations_to_goal=iterations
    )


@pytest.mark.parametrize("difficulty", PROBLEM_DIFFICULTY_LEVELS)
def test_reasoning_by_difficulty(problem_definitions, reasoning_stats, test_logger, difficulty):
    """
    Test the model's ability to solve reasoning problems of different difficulty levels.
    """
    problems = [p for p in problem_definitions if p["difficulty"] == difficulty]
    if not problems:
        pytest.skip(f"No problems defined for difficulty: {difficulty}")

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)

    for problem in problems:
        test_name = f"reasoning_{difficulty}_{problem['id']}"
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


def test_conditional_reasoning(reasoning_stats, test_logger):
    """
    Test the model's ability to use conditional reasoning (IF/ELSE structures)
    to make decisions about research approaches.
    """
    test_logger.start_test("conditional_reasoning", {"description": "Testing model's ability to use conditional reasoning structures"})

    # Initial KG with a condition to evaluate
    initial_kg = (
        'KG►\n'
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T13:24:33.347883", message="Analyze the given topic based on complexity"\n'
        'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Determine complexity and apply appropriate analysis method"\n'
        'KGNODE► Topic : type="DataNode", content="Quantum computing applications in cryptography", meta_props={"complexity": "high"}\n'
        '◄'
    )

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    evaluator.initialize_kg_from_serialized(initial_kg)

    reasoning_stats["total_prompts"] += 1
    response = evaluator.prompt_model_with_kg(initial_kg, "conditional_reasoning", 1)
    print("\n=== Conditional Reasoning Response ===\n", response)

    is_valid, error_message = validate_kgml_with_error(response)
    if is_valid:
        reasoning_stats["valid_responses"] += 1
        result = evaluator.execute_kgml(response, "conditional_reasoning", 1)
        reasoning_stats["execution_results"].append(result)

        # Check for conditional structures in the response
        has_conditional = "IF►" in response
        assert has_conditional, "Response should include conditional reasoning (IF structure)"

        # Execute and check result
        # The condition should evaluate topic complexity and branch accordingly
        final_kg = evaluator.serialize_kg()

        # For high complexity topic, expect detailed analysis
        has_detailed_analysis = any(
            node.uid.startswith("DetailedAnalysis") or
            node.uid.startswith("ComplexAnalysis") or
            node.uid.startswith("AdvancedAnalysis")
            for node in evaluator.kg.query_nodes()
        )

        assert has_detailed_analysis, "High complexity topic should trigger detailed analysis path"

        test_logger.end_test("conditional_reasoning", goal_reached=True, iterations_to_goal=1)
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append(f"Conditional reasoning test: {error_message}")
        test_logger.end_test("conditional_reasoning", goal_reached=False)

    assert is_valid, f"Conditional reasoning should yield valid KGML: {error_message}"


def test_complex_structure_handling(reasoning_stats, test_logger):
    """
    Test the model's ability to handle complex KGML structures with nested
    control flow and multiple linked components.
    """
    test_logger.start_test("complex_structure_handling", {"description": "Testing model with complex KGML structures"})

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)

    reasoning_stats["total_prompts"] += 1
    response = evaluator.prompt_model_with_kg(COMPLEX_PROMPT, "complex_structure_handling", 1)
    print("\n=== Complex Structure Response ===\n", response)

    is_valid, error_message = validate_kgml_with_error(response)
    if is_valid:
        reasoning_stats["valid_responses"] += 1
        result = evaluator.execute_kgml(response, "complex_structure_handling", 1)
        reasoning_stats["execution_results"].append(result)

        # Check for advanced structure in response
        has_control_flow = "IF►" in response or "LOOP►" in response
        has_multiple_nodes = len([entry for entry in result["execution_log"]
                                  if entry["command_type"] == "C" and
                                  entry["details"]["entity_type"] == "NODE"]) >= 3

        has_multiple_links = len([entry for entry in result["execution_log"]
                                  if entry["command_type"] == "C" and
                                  entry["details"]["entity_type"] == "LINK"]) >= 2

        assert has_control_flow, "Response should use control flow structures"
        assert has_multiple_nodes, "Response should create multiple nodes"
        assert has_multiple_links, "Response should create multiple links between nodes"

        test_logger.end_test("complex_structure_handling", goal_reached=True, iterations_to_goal=1)
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append(f"Complex structure test: {error_message}")
        test_logger.end_test("complex_structure_handling", goal_reached=False)

    assert is_valid, f"Complex KGML structure should yield valid KGML response: {error_message}"


def test_comprehensive_reasoning_evaluation(problem_definitions, reasoning_stats, test_logger):
    """
    Comprehensive test that evaluates all reasoning problems and reports detailed statistics.
    """
    test_logger.start_test("comprehensive_evaluation", {"description": "Comprehensive evaluation of all reasoning problems"})

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


def test_datanode_content_creation(reasoning_stats, test_logger):
    """
    Test the model's ability to create rich, detailed content in DataNodes.
    """
    test_logger.start_test("datanode_content_creation", {"description": "Testing model's ability to create detailed DataNode content"})

    # Initial KG with a research request
    initial_kg = (
        'KG►\n'
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T13:24:33.347883", message="Create a detailed summary of reinforcement learning algorithms"\n'
        'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Generate comprehensive DataNode content on RL algorithms"\n'
        '◄'
    )

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    evaluator.initialize_kg_from_serialized(initial_kg)

    reasoning_stats["total_prompts"] += 1
    response = evaluator.prompt_model_with_kg(initial_kg, "datanode_content_creation", 1)
    print("\n=== DataNode Content Creation Response ===\n", response)

    is_valid, error_message = validate_kgml_with_error(response)
    if is_valid:
        reasoning_stats["valid_responses"] += 1
        result = evaluator.execute_kgml(response, "datanode_content_creation", 1)
        reasoning_stats["execution_results"].append(result)

        # Check for DataNodes with substantial content
        data_nodes = [node for node in evaluator.kg.query_nodes()
                      if hasattr(node, 'content') and
                      node.content is not None and
                      isinstance(node.content, str)]

        has_substantial_content = any(
            len(str(node.content)) > 1000  # At least 1000 characters
            for node in data_nodes
        )

        has_structured_content = any(
            ("## " in str(node.content) or "**" in str(node.content))  # Has markdown/structure
            for node in data_nodes
        )

        assert len(data_nodes) > 0, "Should create at least one DataNode with content"
        assert has_substantial_content, "At least one DataNode should have substantial content"
        assert has_structured_content, "Content should be structured with sections or formatting"

        # Find the largest content node for the report
        largest_node = max(data_nodes, key=lambda n: len(str(n.content)), default=None)
        if largest_node:
            content_sample = str(largest_node.content)[:500] + "..." if len(str(largest_node.content)) > 500 else str(largest_node.content)
            print(f"\nDataNode Content Sample from {largest_node.uid}:\n{content_sample}")

            # Save the content to a file in the test logger directory
            content_file = Path(test_logger.run_dir) / f"datanode_content_{largest_node.uid}.txt"
            with open(content_file, "w", encoding="utf-8") as f:
                f.write(str(largest_node.content))

        test_logger.end_test("datanode_content_creation", goal_reached=True, iterations_to_goal=1)
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append(f"DataNode content creation test: {error_message}")
        test_logger.end_test("datanode_content_creation", goal_reached=False)

    assert is_valid, f"DataNode content creation should yield valid KGML: {error_message}"


def test_function_node_evaluation(reasoning_stats, test_logger):
    """
    Test the model's ability to create and evaluate FunctionNodes to perform reasoning tasks.
    """
    test_logger.start_test("function_node_evaluation", {"description": "Testing model's ability to use FunctionNodes for reasoning"})

    # Initial KG with a request that requires functional evaluation
    initial_kg = (
        'KG►\n'
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T15:10:23.347883", message="Analyze and classify ML research papers by approach"\n'
        'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Create functions to classify and analyze content by approach"\n'
        'KGNODE► ResearchPapers : type="DataNode", content="1. Deep Reinforcement Learning for Robotic Manipulation\\n2. Attention Is All You Need\\n3. BERT: Pre-training of Deep Bidirectional Transformers\\n4. Proximal Policy Optimization Algorithms\\n5. MuZero: Mastering Go, chess, shogi and Atari without rules"\n'
        '◄'
    )

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    evaluator.initialize_kg_from_serialized(initial_kg)

    reasoning_stats["total_prompts"] += 1
    response = evaluator.prompt_model_with_kg(initial_kg, "function_node_evaluation", 1)
    print("\n=== FunctionNode Evaluation Response ===\n", response)

    is_valid, error_message = validate_kgml_with_error(response)
    if is_valid:
        reasoning_stats["valid_responses"] += 1
        result = evaluator.execute_kgml(response, "function_node_evaluation", 1)
        reasoning_stats["execution_results"].append(result)

        # Check for FunctionNodes in the KG
        function_nodes_created = any(
            "FunctionNode" in entry["details"]["instruction"]
            for entry in result["execution_log"]
            if entry["command_type"] == "C" and "details" in entry
            and "instruction" in entry["details"]
        )

        # Check for evaluation of nodes
        evaluations = [
            entry for entry in result["execution_log"]
            if entry["command_type"] == "E"
        ]

        # Check for output based on function evaluation
        result_nodes = [
            node for node in evaluator.kg.query_nodes()
            if node.uid.startswith("Result") or node.uid.startswith("Classification")
        ]

        assert function_nodes_created, "Model should create FunctionNodes for classification"
        assert len(evaluations) > 0, "Model should evaluate nodes"
        assert len(result_nodes) > 0, "Model should create result nodes from function evaluation"

        test_logger.end_test("function_node_evaluation", goal_reached=True, iterations_to_goal=1)
    else:
        reasoning_stats["invalid_responses"] += 1
        reasoning_stats["errors"].append(f"FunctionNode evaluation test: {error_message}")
        test_logger.end_test("function_node_evaluation", goal_reached=False)

    assert is_valid, f"FunctionNode evaluation should yield valid KGML: {error_message}"


def test_outcome_based_reasoning(reasoning_stats, test_logger):
    """
    Test the model's ability to use OutcomeNodes to assess progress and guide
    further reasoning based on evaluation results.
    """
    test_logger.start_test("outcome_based_reasoning", {"description": "Testing model's ability to perform multi-stage reasoning with outcome assessment"})

    # Initial KG with a request that requires multiple stages and evaluation
    initial_kg = (
        'KG►\n'
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T16:45:23.125678", message="Develop a multi-stage analysis of AGI safety frameworks"\n'
        'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Create a multi-stage analysis with outcome assessment after each stage"\n'
        'KGNODE► EvaluationCriteria : type="DataNode", content="Each stage should be evaluated on: 1) Comprehensiveness, 2) Clarity, 3) Evidence-based reasoning, 4) Balanced perspective", meta_props={"min_score": 0.7}\n'
        '◄'
    )

    evaluator = ReasoningEvaluator(CURRENT_MODEL, KGML_SYSTEM_PROMPT, test_logger)
    evaluator.initialize_kg_from_serialized(initial_kg)

    # This test will run in multiple iterations to observe outcome-based course correction
    current_kg = initial_kg
    iterations = 3
    outcome_nodes_count = 0

    for i in range(iterations):
        print(f"\n=== Outcome Reasoning Iteration {i + 1} ===")

        reasoning_stats["total_prompts"] += 1
        response = evaluator.prompt_model_with_kg(current_kg, "outcome_based_reasoning", i + 1)
        print(f"\nOutcome Reasoning Response {i + 1}:", response)

        is_valid, error_message = validate_kgml_with_error(response)
        if is_valid:
            reasoning_stats["valid_responses"] += 1
            result = evaluator.execute_kgml(response, "outcome_based_reasoning", i + 1)
            reasoning_stats["execution_results"].append(result)

            # Update KG for next iteration
            current_kg = evaluator.serialize_kg()

            # Count OutcomeNodes
            new_outcome_nodes = [
                node for node in evaluator.kg.query_nodes()
                if node.uid.startswith("Outcome") or "outcome" in node.uid.lower()
            ]
            outcome_nodes_count = len(new_outcome_nodes)

            # For iterations after the first, check if the model is responding to previous outcomes
            if i > 0:
                references_outcomes = any(
                    "outcome" in entry["details"]["instruction"].lower() or
                    "evaluation" in entry["details"]["instruction"].lower()
                    for entry in result["execution_log"]
                    if entry["command_type"] in ["E", "C"] and "details" in entry
                    and "instruction" in entry["details"]
                )
                assert references_outcomes, f"In iteration {i + 1}, model should reference previous outcomes"
        else:
            reasoning_stats["invalid_responses"] += 1
            reasoning_stats["errors"].append(f"Outcome reasoning iteration {i + 1}: {error_message}")
            break

    # Final assessment
    has_refinement_based_on_outcomes = any(
        node.uid.startswith("Refined") or
        node.uid.startswith("Improved") or
        "revision" in node.uid.lower() or
        "version" in node.uid.lower()
        for node in evaluator.kg.query_nodes()
    )

    assert outcome_nodes_count > 0, "Should create at least one outcome node"
    assert has_refinement_based_on_outcomes, "Should show refinement based on outcome assessment"

    # Check if outcomes influenced the reasoning process
    outcome_links = [
        edge for edge in evaluator.kg.query_edges()
        if any(term in edge.source_uid.lower() for term in ["outcome", "evaluation", "assessment"]) or
           any(term in edge.target_uid.lower() for term in ["outcome", "evaluation", "assessment"])
    ]

    assert len(outcome_links) > 0, "Outcomes should be linked to other nodes in the reasoning process"

    test_logger.end_test(
        "outcome_based_reasoning",
        goal_reached=is_valid and outcome_nodes_count > 0 and has_refinement_based_on_outcomes,
        iterations_to_goal=iterations
    )
