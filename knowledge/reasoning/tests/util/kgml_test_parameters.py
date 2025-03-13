"""
KGML Test Parameters and Fixtures

This module contains constants, configuration, and fixtures for KGML integration tests.

TODO These need to be loaded from a file!
"""
import pytest

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
    from knowledge.graph.kg_models import KnowledgeGraph
    return KnowledgeGraph()


@pytest.fixture
def initial_kg_serialized():
    """
    Returns the initial serialized Knowledge Graph prompt as a plain text string.
    The serialization conforms to the required format with proper closing markers.
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


@pytest.fixture
def complex_problem_definition():
    return (
        'KG►\n'
        'KGNODE► Problem : type="ProblemNode", description="Need to analyze sensor data for anomalies"\n'
        'KGNODE► Context : type="ContextNode", domain="IoT", dataSource="temperature_sensors"\n'
        'KGLINK► Problem -> Context : type="HasContext", priority="high"\n'
        '◄\n'
        'C► NODE Plan "Create a plan to analyze sensor data" ◄\n'
    )
