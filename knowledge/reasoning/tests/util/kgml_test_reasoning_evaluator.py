"""
KGML Reasoning Evaluator - Fixed Version

This module contains the ReasoningEvaluator class with fixes for the timing issue.
"""
import logging
import re
import time
from typing import Dict, Any, Optional

from knowledge.graph.kg_models import KnowledgeGraph, KGNode, KGEdge
from knowledge.reasoning.dsl.execution.kgml_executor import KGMLExecutor
from knowledge.reasoning.tests.util.kgml_test_helpers import validate_kgml_with_error
from knowledge.reasoning.tests.util.kgml_test_logger import KGMLTestLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLIntegrationTests")


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
        Updated to support the new KG format with KGNODE and KGLINK declarations.
        """
        # Clear existing graph
        self.kg = KnowledgeGraph()
        self.executor = KGMLExecutor(self.kg)

        lines = serialized_kg.strip().split('\n')
        if lines[0] != 'KG►' or lines[-1] != '◄':
            raise ValueError("Invalid KG serialization format - missing KG► and ◄ markers")

        current_section = None
        node_defs = {}
        edge_defs = []

        # Process node and link definitions
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

                # Save node definition
                node_type = props.pop('type', 'GenericNode')
                node_defs[node_id] = (node_type, props)

            elif line.startswith('KGLINK►'):
                # Parse the link definition
                match = re.match(r'KGLINK► (\w+) -> (\w+) : (.+)', line)
                if not match:
                    continue

                source_id, target_id, props_str = match.groups()

                # Parse the properties
                props = {}
                for prop_match in re.finditer(r'(\w+)="([^"]*)"', props_str):
                    key, value = prop_match.groups()
                    props[key] = value

                # Save edge definition
                edge_type = props.pop('type', 'GenericLink')
                edge_defs.append((source_id, target_id, edge_type, props))

        # Create nodes first
        for node_id, (node_type, props) in node_defs.items():
            node = KGNode(uid=node_id, type=node_type, meta_props=props)
            self.kg.add_node(node)

        # Then create edges
        for source_id, target_id, edge_type, props in edge_defs:
            if source_id in node_defs and target_id in node_defs:
                edge = KGEdge(source_uid=source_id, target_uid=target_id, type=edge_type, meta_props=props)
                self.kg.add_edge(edge)

        return self.kg

    def serialize_kg(self) -> str:
        """
        Serialize the current knowledge graph to a string representation.
        Updated to include KGLINK syntax for edges.
        """
        lines = ['KG►']

        # Add all nodes
        for node in self.kg.query_nodes():
            props_str = f'type="{node.type}"'
            for key, value in node.meta_props.items():
                props_str += f', {key}="{value}"'
            lines.append(f'KGNODE► {node.uid} : {props_str}')

        # Add all edges using KGLINK syntax
        for edge in self.kg.query_edges():
            props_str = f'type="{edge.type}"'
            for key, value in edge.meta_props.items():
                props_str += f', {key}="{value}"'
            lines.append(f'KGLINK► {edge.source_uid} -> {edge.target_uid} : {props_str}')

        lines.append('◄')
        return '\n'.join(lines)

    def prompt_model_with_kg(self, serialized_kg: str, test_name: Optional[str] = None, iteration: Optional[int] = None) -> str:
        """
        Send the serialized KG to the language model and get a KGML response.
        Also tracks response time and logs the interaction if a test logger is available.
        """
        from integration.net.ollama.ollama_api import prompt_model

        start_time = time.time()
        response = prompt_model(
            serialized_kg,
            model=self.model_name,
            system_prompt=self.system_prompt
        )
        end_time = time.time()
        response_time = end_time - start_time

        # Make sure response_time is never 0 (minimum 0.001s)
        if response_time <= 0:
            response_time = 0.001
            logger.warning(f"Response time was 0 or negative, setting to {response_time}s")

        # Validate the response syntax and get any error
        is_valid, error_message = validate_kgml_with_error(response)
        has_syntax_errors = not is_valid

        # Log the request-response pair if logger is available
        if self.test_logger and test_name and iteration is not None:
            # Include execution result with syntax error if validation failed
            execution_result = None
            if has_syntax_errors:
                execution_result = {
                    "success": False,
                    "error": f"Syntax error: {error_message}"
                }

            self.test_logger.log_request_response(
                test_name=test_name,
                iteration=iteration,
                request=serialized_kg,
                response=response,
                response_time=response_time,
                is_valid=is_valid,
                has_syntax_errors=has_syntax_errors,
                execution_result=execution_result
            )

        return response

    def execute_kgml(self, kgml_code: str, test_name: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the KGML code and return the execution results.
        Logs the execution results without overwriting response timing if a test logger is available.
        """
        start_time = time.time()
        try:
            context = self.executor.execute(kgml_code)
            end_time = time.time()
            execution_time = end_time - start_time

            # Make sure execution_time is never 0 (minimum 0.001s)
            if execution_time <= 0:
                execution_time = 0.001
                logger.warning(f"Execution time was 0 or negative, setting to {execution_time}s")

            result = {
                "success": True,
                "execution_log": context.execution_log,
                "variables": context.variables,
                "results": context.results,
                "execution_time": execution_time
            }

            # Log execution result if logger is available
            if self.test_logger and test_name and iteration is not None:
                # Update without overwriting response time
                self.test_logger.log_request_response(
                    test_name=test_name,
                    iteration=iteration,
                    request="",  # Empty because we're just updating
                    response="",  # Empty because we're just updating
                    response_time=0.0,  # This is ignored in the fixed logger when request is empty
                    is_valid=True,  # Not relevant for this update
                    has_syntax_errors=False,  # Not relevant for this update
                    execution_result=result
                )

            return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            # Make sure execution_time is never 0 (minimum 0.001s)
            if execution_time <= 0:
                execution_time = 0.001
                logger.warning(f"Execution time was 0 or negative, setting to {execution_time}s")

            logger.error(f"KGML execution failed: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }

            # Log execution error if logger is available
            if self.test_logger and test_name and iteration is not None:
                # Update without overwriting response time
                self.test_logger.log_request_response(
                    test_name=test_name,
                    iteration=iteration,
                    request="",  # Empty because we're just updating
                    response="",  # Empty because we're just updating
                    response_time=0.0,  # This is ignored in the fixed logger when request is empty
                    is_valid=True,  # Syntax is valid (execution just failed)
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
            is_valid, error_message = validate_kgml_with_error(model_response)
            if is_valid:
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
                logger.error(f"Invalid KGML response in iteration {iterations + 1}: {error_message}")
                error_result = {
                    "success": False,
                    "error": f"Invalid KGML syntax: {error_message}"
                }
                execution_results.append(error_result)

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
