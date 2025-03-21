"""
KGML Test Parameters and Fixtures

This module contains constants, configuration, and fixtures for KGML reasoning-focused tests.
"""
import pytest

# ------------------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------------------
CURRENT_MODEL = "qwen2.5-coder:14b"
MAX_ITERATIONS = 10
PROBLEM_DIFFICULTY_LEVELS = ["basic", "intermediate", "advanced"]

# Complex prompt for testing model's ability to handle complex KGML structures
COMPLEX_PROMPT = """
KG►
KGNODE► Instruction : type="DataNode", content="Create a structured analysis of transformer architecture advancements"
KGNODE► Context : type="ContextNode", domain="AI Research", focus="Transformer Models"
KGLINK► Instruction -> Context : relation="HIERARCHY", meta_props={"priority": "high"}
◄

IF► E► NODE Instruction "Check if the instruction requires advanced analysis" ◄
    C► NODE AnalysisPlan "Create a multi-step analysis plan with progressive refinement" ◄
    C► LINK Instruction -> AnalysisPlan "Create a HIERARCHY link to connect instruction to plan" ◄
ELSE►
    C► NODE BasicReport "Create a simple summary report" ◄
◄
"""


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
        'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T13:24:33.347883", message="Create a research report on Transformer architecture advancements"\n'
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
            "id": "basic_research_report",
            "description": "Create a simple research report on the evolution of transformers in NLP.",
            "initial_kg": (
                'KG►\n'
                'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T13:24:33.347883", message="Create a research report on the evolution of transformers in NLP"\n'
                'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Process the request and create a structured report"\n'
                '◄'
            ),
            "goal_condition": lambda kg: any(
                getattr(node, 'type', None) == "DataNode" and
                node.uid.startswith("Report") and
                hasattr(node, 'content') and
                node.content is not None and
                len(str(node.content)) > 500  # Report should have substantial content
                for node in kg.query_nodes()
            ),
            "difficulty": "basic"
        },
        {
            "id": "iterative_refinement",
            "description": "Demonstrate iterative refinement of an AI ethics analysis, with multiple stages of improvement.",
            "initial_kg": (
                'KG►\n'
                'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T14:15:22.123456", message="Create a detailed analysis of ethical considerations in large language models"\n'
                'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Create an initial draft, then refine it iteratively with deeper analysis"\n'
                'KGNODE► InitialContext : type="DataNode", content="Focus on bias, transparency, privacy, and accountability in LLMs"\n'
                '◄'
            ),
            "goal_condition": lambda kg: (
                # Check for multiple revisions of the report
                    len([n for n in kg.query_nodes() if n.uid.startswith("DraftReport")]) >= 1 and
                    len([n for n in kg.query_nodes() if n.uid.startswith("RefinedReport")]) >= 1 and
                    len([n for n in kg.query_nodes() if n.uid.startswith("FinalReport")]) >= 1 and
                    # Check for links showing the refinement process
                    len([e for e in kg.query_edges() if "HIERARCHY" in e.relation]) >= 2
            ),
            "difficulty": "intermediate"
        },
        {
            "id": "hierarchical_concept_mapping",
            "description": "Create a hierarchical concept map of AI research domains with proper relationships and structured metadata.",
            "initial_kg": (
                'KG►\n'
                'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T16:45:12.789012", message="Create a hierarchical concept map of AI research domains"\n'
                'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Identify major domains, subdisciplines, and establish proper hierarchical relationships"\n'
                'KGNODE► RootConcept : type="DataNode", content="Artificial Intelligence Research", meta_props={"level": "root"}\n'
                '◄'
            ),
            "goal_condition": lambda kg: (
                # Check for a minimum number of concept nodes
                    len([n for n in kg.query_nodes() if n.uid.startswith("Concept") and hasattr(n, 'content')]) >= 6 and
                    # Check for hierarchical links between concepts
                    len([e for e in kg.query_edges() if "HIERARCHY" in e.relation]) >= 5 and
                    # Check for at least one concept at each level
                    any([n for n in kg.query_nodes() if n.uid.startswith("Concept") and
                         n.meta_props.get("level") == "root"]) and
                    any([n for n in kg.query_nodes() if n.uid.startswith("Concept") and
                         n.meta_props.get("level") == "domain"]) and
                    any([n for n in kg.query_nodes() if n.uid.startswith("Concept") and
                         n.meta_props.get("level") == "subdomain"])
            ),
            "difficulty": "advanced"
        },
        {
            "id": "multi_perspective_analysis",
            "description": "Create a multi-perspective analysis of AI safety, incorporating different viewpoints and forming a synthesis.",
            "initial_kg": (
                'KG►\n'
                'KGNODE► EventMeta_1 : type="EventMetaNode", timestamp="2025-03-22T10:30:45.123456", message="Create a multi-perspective analysis of AI safety"\n'
                'KGNODE► ActionMeta_1 : type="ActionMetaNode", reference="EventMeta_1", instruction="Analyze AI safety from multiple perspectives and synthesize findings"\n'
                'KGNODE► Perspectives : type="DataNode", content="Consider: academic researchers, industry practitioners, policy makers, and ethicists", meta_props={"min_perspectives": 4}\n'
                '◄'
            ),
            "goal_condition": lambda kg: (
                # Check for perspective nodes
                    len([n for n in kg.query_nodes() if n.uid.startswith("Perspective")]) >= 3 and
                    # Check for a synthesis node
                    any([n for n in kg.query_nodes() if n.uid.startswith("Synthesis") and hasattr(n, 'content')]) and
                    # Check for relationship links between perspectives and synthesis
                    len([e for e in kg.query_edges() if e.target_uid.startswith("Synthesis")]) >= 3
            ),
            "difficulty": "advanced"
        }
    ]
