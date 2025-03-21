"""
KGML System Prompt Generator.

This module dynamically generates the system prompt for the KGML Reasoning Agent
based on the current model definitions and constants. It can automatically register
and inspect any module pairs to include their functionality in the prompt.
"""

import importlib
import inspect
from enum import Enum
from typing import List, Type, Dict, Any, Set

# Update registry to include both module paths correctly
MODULE_REGISTRY = [
    ('integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_models',
     'integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants')
]


class ModuleInfo:
    """Holds information extracted from a registered module pair."""

    def __init__(self, models_module_name: str, constants_module_name: str):
        self.models_module_name = models_module_name
        self.constants_module_name = constants_module_name
        self.models_module = None
        self.constants_module = None
        self.node_classes = {}  # {NodeType: class}
        self.link_classes = []  # List of link classes
        self.node_type_enum = None
        self.link_relation_enum = None

        # Load the modules
        self._load_modules()

    def _load_modules(self):
        """Load the registered module pair."""
        try:
            self.models_module = importlib.import_module(self.models_module_name)
            self.constants_module = importlib.import_module(self.constants_module_name)

            # Identify important classes and enums
            self._identify_classes_and_enums()
        except ImportError as e:
            print(f"Error loading modules: {e}")

    def _identify_classes_and_enums(self):
        """Identify important classes and enums in the modules."""
        # Find NodeType and LinkRelation enums
        for name, obj in inspect.getmembers(self.constants_module):
            if inspect.isclass(obj) and issubclass(obj, Enum):
                if name == 'NodeType':
                    self.node_type_enum = obj
                elif name == 'LinkRelation':
                    self.link_relation_enum = obj

        # Debug logging
        print(f"Found NodeType enum: {self.node_type_enum}")
        print(f"Found LinkRelation enum: {self.link_relation_enum}")

        # Set of potential node base classes
        node_base_classes = {'DataNode', 'KGNode', 'EventMetaNode', 'ActionMetaNode', 'FunctionNode', 'OutcomeNode'}

        # Set of potential link base classes
        link_base_classes = {'KGEdge', 'Link', 'ReasoningLink'}

        # Find node and link classes
        for name, obj in inspect.getmembers(self.models_module):
            if not inspect.isclass(obj):
                continue

            # Check if this class has a 'type' attribute that matches a NodeType enum value
            if hasattr(obj, 'type') and isinstance(obj.type, str) and self.node_type_enum:
                try:
                    # Try to match the type string to a NodeType enum
                    node_type = next((nt for nt in self.node_type_enum if nt.value == obj.type), None)
                    if node_type:
                        self.node_classes[node_type] = obj
                        continue
                except (AttributeError, TypeError):
                    pass  # Not a matching node type

            # Examine the class hierarchy
            is_node_class = False
            is_link_class = False

            # Check all base classes recursively
            bases_to_check = list(obj.__bases__)
            checked_bases = set()

            while bases_to_check:
                base = bases_to_check.pop(0)
                if base.__name__ in checked_bases:
                    continue

                checked_bases.add(base.__name__)

                # Check if this is a node or link base class
                if base.__name__ in node_base_classes:
                    is_node_class = True
                    break
                elif base.__name__ in link_base_classes:
                    is_link_class = True
                    break

                # Add this class's bases to check
                bases_to_check.extend(base.__bases__)

            # If we identified a node class
            if is_node_class and self.node_type_enum:
                # Try to match by class name to NodeType
                for node_type in self.node_type_enum:
                    # Match by comparing lowercase class name with lowercase node type name or value
                    if (obj.__name__.lower() == node_type.name.lower() or
                        (hasattr(node_type, 'value') and
                         isinstance(node_type.value, str) and
                         obj.__name__.lower() == node_type.value.lower())):
                        self.node_classes[node_type] = obj
                        break

                    # Additional check for classes with type = NodeType.X pattern
                    if hasattr(obj, '__annotations__') and 'type' in obj.__annotations__:
                        # This might indicate a class with a type field
                        for field in dir(obj):
                            if field == 'type' and isinstance(getattr(obj, field, None), str):
                                type_value = getattr(obj, field)
                                if type_value == node_type.value:
                                    self.node_classes[node_type] = obj
                                    break

            # If we identified a link class
            elif is_link_class:
                self.link_classes.append(obj)

        # Debug logging
        print(f"Found {len(self.node_classes)} node classes:")
        for node_type, cls in self.node_classes.items():
            print(f"  - {node_type.name}: {cls.__name__}")

        print(f"Found {len(self.link_classes)} link classes:")
        for cls in self.link_classes:
            print(f"  - {cls.__name__}")


def _get_class_properties(cls: Type) -> List[str]:
    """
    Extract property names from a class.

    Args:
        cls: The class to extract properties from

    Returns:
        List of property names
    """
    properties = []

    # Try to get annotations (type hints)
    if hasattr(cls, '__annotations__'):
        properties = list(cls.__annotations__.keys())

    # Add class variables that have docstrings or default values
    for name, value in vars(cls).items():
        if not name.startswith('_') and name not in properties:
            properties.append(name)

    # Add properties defined with @property
    for name, value in inspect.getmembers(cls, lambda x: isinstance(x, property)):
        if name not in properties:
            properties.append(name)

    # If no properties found, fall back to inspecting class attributes
    if not properties:
        properties = [attr for attr in dir(cls)
                      if not attr.startswith('_') and not callable(getattr(cls, attr, None))]

    # Remove common method names that might be included
    method_names = {'__init__', '__str__', '__repr__', 'model_dump', 'to_dict', 'from_dict'}
    properties = [p for p in properties if p not in method_names and not callable(getattr(cls, p, None))]

    return properties


def _get_enum_values(enum_class: Type[Enum]) -> List[str]:
    """
    Extract values from an Enum class.

    Args:
        enum_class: The Enum class to extract values from

    Returns:
        List of enum value names
    """
    return [member.name for member in enum_class]


def _generate_node_type_descriptions(module_infos: List[ModuleInfo]) -> str:
    """
    Generate descriptions of all node types and their properties from registered modules.

    Args:
        module_infos: List of module information

    Returns:
        Formatted string with node type descriptions
    """
    all_descriptions = []
    processed_classes = set()

    # Process each module's node classes
    for module_info in module_infos:
        if not module_info.node_type_enum:
            continue

        for node_type, cls in module_info.node_classes.items():
            # Skip if we've already processed this class
            if cls.__name__ in processed_classes:
                continue

            processed_classes.add(cls.__name__)

            # Get class name as string
            class_name = cls.__name__

            # Get properties
            properties = _get_class_properties(cls)

            # Get module name (for grouping)
            module_name = module_info.models_module_name.split('.')[-1]

            # Get class docstring for description
            doc = cls.__doc__.strip() if cls.__doc__ else node_type.value
            if doc:
                # Take first sentence only
                doc = doc.split('.')[0].strip()

            # Generate description
            description = f"1. `{class_name}` ({module_name}): {doc}\n"
            if properties:
                description += "   - Properties: " + ", ".join(properties)

            all_descriptions.append(description)

    # If no descriptions were found, try a different approach
    if not all_descriptions:
        for module_info in module_infos:
            if module_info.models_module:
                for name, obj in inspect.getmembers(module_info.models_module):
                    if (inspect.isclass(obj) and
                        not name.startswith('_') and
                        name not in processed_classes and
                        hasattr(obj, '__module__') and
                        obj.__module__ == module_info.models_module.__name__):

                        processed_classes.add(name)
                        properties = _get_class_properties(obj)
                        module_name = module_info.models_module_name.split('.')[-1]

                        doc = obj.__doc__.strip() if obj.__doc__ else "No description available"
                        if doc:
                            doc = doc.split('.')[0].strip()

                        description = f"1. `{name}` ({module_name}): {doc}\n"
                        if properties:
                            description += "   - Properties: " + ", ".join(properties)

                        all_descriptions.append(description)

    return "\n\n".join(all_descriptions)


def _generate_link_relation_descriptions(module_infos: List[ModuleInfo]) -> str:
    """
    Generate descriptions of all link relation types from registered modules.

    Args:
        module_infos: List of module information

    Returns:
        Formatted string with link relation descriptions
    """
    # Collect all relation types
    all_relations = {}

    for module_info in module_infos:
        if not module_info.link_relation_enum:
            continue

        # Extract relation types from the enum
        for relation in module_info.link_relation_enum:
            # Generate a description based on the relation name if not provided
            description = getattr(relation, 'value', None)
            if not description or description == relation.name:
                # Convert SNAKE_CASE to sentence
                words = relation.name.replace('_', ' ').lower().split()
                if len(words) >= 2:
                    description = f"{words[0].capitalize()} {' '.join(words[1:])}"
                else:
                    description = relation.name.replace('_', ' ').capitalize()

            all_relations[relation.name] = description

    # Define common relation descriptions (will override if better descriptions exist)
    common_descriptions = {
        "REPLY_TO": "Message replies to another message",
        "PART_OF": "Component belongs to a larger entity",
        "DERIVES_FROM": "Entity derived from another entity",
        "AUGMENTS": "Entity augments/enhances another entity",
        "REFERENCES": "Entity references another entity",
        "CONTAINS": "Entity contains another entity",
        "PRODUCED_BY": "Entity was produced by a function/process",
        "TRIGGERS": "Event triggers a function/action",
        "HAS_OUTCOME": "Function/action has a specific outcome"
    }

    # Update with common descriptions if available
    for relation, description in common_descriptions.items():
        if relation in all_relations:
            all_relations[relation] = description

    # Format descriptions
    descriptions = []
    for relation, description in all_relations.items():
        descriptions.append(f"- `{relation}`: {description}")

    return "\n".join(descriptions)


def _generate_example_workflows() -> str:
    """
    Generate example workflows for common operations.

    Returns:
        Formatted string with example workflows
    """
    workflows = {}

    # New search request workflow
    workflows["New Search Request"] = """
When user asks: "Search for climate change effects on agriculture"

```
C► NODE config_1 "Create a SearchConfig with search_terms=['climate change effects on agriculture']" ◄
C► NODE goal_1 "Create a SearchGoal for tracking success of this search" ◄
C► LINK link_goal_1 "Create a link between goal_1 and config_1 with HAS_OUTCOME relation" ◄
E► NODE web_search_1 "Evaluate WebSearchFunction with config_1" ◄
C► NODE response_1 "Create assistant ChatMessage with response about search results" ◄
```
"""

    # Follow-up question workflow
    workflows["Follow-up Question"] = """
When user asks: "What about in tropical regions?" after a climate search:

```
E► NODE context_1 "Get conversation context to understand previous search topic" ◄
IF► E► NODE context_checker "Check if question can be answered from existing results" ◄
    C► NODE response_1 "Create response based on existing search results" ◄
ELSE►
    C► NODE config_2 "Create a new SearchConfig that refines previous search with 'tropical regions'" ◄
    C► LINK link_augment_1 "Create link between config_2 and previous config with relation AUGMENTS" ◄
    E► NODE web_search_1 "Evaluate WebSearchFunction with config_2" ◄
    C► NODE response_2 "Create response with new search results" ◄
◄
```
"""

    # Augmenting a search workflow
    workflows["Augmenting a Search"] = """
When user asks: "Add drought impacts to my previous search":

```
E► NODE context_1 "Find the most recent search configuration" ◄
C► NODE config_2 "Create a new SearchConfig that extends the previous search with 'drought impacts'" ◄
C► LINK link_augment_1 "Create link between config_2 and previous config with relation AUGMENTS" ◄
E► NODE web_search_1 "Evaluate WebSearchFunction with config_2" ◄
C► NODE response_1 "Create response explaining the augmented search" ◄
```
"""

    # Status check workflow
    workflows["Status Check"] = """
When user asks: "What's the status of my search?":

```
E► NODE context_1 "Get conversation context with active searches" ◄
IF► E► NODE active_checker "Check if there are active searches" ◄
    C► NODE response_1 "Create response with status of active searches" ◄
ELSE►
    E► NODE recent_checker "Get most recent completed searches" ◄
    C► NODE response_2 "Create response summarizing recent search results" ◄
◄
```
"""

    # Simple chat workflow
    workflows["Simple Chat"] = """
When user asks a simple question unrelated to search:

```
E► NODE context_1 "Check if this can be answered without search" ◄
IF► E► NODE chat_checker "Determine if question is answerable without search" ◄
    C► NODE response_1 "Create direct chat response" ◄
ELSE►
    C► NODE config_1 "Create a new SearchConfig for answering the question" ◄
    E► NODE web_search_1 "Evaluate WebSearchFunction with config_1" ◄
    C► NODE response_2 "Create response based on search results" ◄
◄
```
"""

    # Combine all workflows
    result = ""
    for title, workflow in workflows.items():
        result += f"### {title}\n{workflow}\n"

    return result


def register_module_pair(models_module_name: str, constants_module_name: str):
    """
    Register a new module pair to be included in prompt generation.

    Args:
        models_module_name: Name of the models module
        constants_module_name: Name of the constants module
    """
    module_pair = (models_module_name, constants_module_name)
    if module_pair not in MODULE_REGISTRY:
        MODULE_REGISTRY.append(module_pair)


def generate_kgml_system_prompt() -> str:
    """
    Generate the system prompt for the KGML agent.

    Dynamically builds the prompt based on all registered module pairs,
    their models, constants, and example workflows.

    Returns:
        A system prompt for the KGML agent
    """
    # Load all registered modules
    module_infos = []
    for models_module_name, constants_module_name in MODULE_REGISTRY:
        module_info = ModuleInfo(models_module_name, constants_module_name)
        module_infos.append(module_info)

    # Generate components dynamically
    node_descriptions = _generate_node_type_descriptions(module_infos)
    link_descriptions = _generate_link_relation_descriptions(module_infos)
    example_workflows = _generate_example_workflows()

    # Build the full prompt
    prompt = """# SYSTEM PROMPT: KGML Reasoning Agent

You are a Reasoning DSL Agent – a specialized AI that understands and manipulates Knowledge Graphs through KGML (Knowledge Graph Manipulation Language). Your purpose is to analyze events, reason through complex information, and evolve the Knowledge Graph to achieve goals.

Every action you take must be in valid KGML code. Your capabilities include creating, updating, deleting, and evaluating graph nodes, as well as using control structures when required.

## YOUR MAIN RESPONSIBILITIES

1. **Analyze user messages** to determine their intent, goals, and information needs
2. **Orchestrate web searches** by configuring and executing search functions
3. **Extract patterns** from search results to gain deeper insights
4. **Generate helpful responses** based on conversation context and search results
5. **Track and verify goals** to ensure user needs are met

## KGML SYNTAX GUIDE

Reserved command keywords end with "►". Control blocks are defined between an opening keyword and a closing "◄".

### Core Commands
- `C►, ◄` = Create a NODE or LINK
- `U►, ◄` = Update a NODE or LINK
- `D►, ◄` = Delete a NODE or LINK
- `E►, ◄` = Evaluate a NODE

### Control Structures
- `IF►, ELIF►, ELSE►, ◄` = Conditional execution blocks
- `LOOP►, ◄` = Loop blocks

### Knowledge Graph Objects
- `KG►, ◄` = Knowledge Graph block containing nodes/links
- `KGNODE►, ◄` = Node data
- `KGLINK►, ◄` = Edge / Link data

### Command Format
```
C► NODE node_id "Natural language instruction to create a node" ◄
U► NODE node_id "Natural language instruction to update a node" ◄
D► NODE node_id "Natural language instruction to delete a node" ◄
E► NODE node_id "Natural language instruction to evaluate a node" ◄
```

Similarly for links:
```
C► LINK link_id "Natural language instruction to create a link" ◄
U► LINK link_id "Natural language instruction to update a link" ◄
D► LINK link_id "Natural language instruction to delete a link" ◄
```

## NODE TYPES AND THEIR PROPERTIES

The knowledge graph contains these specialized node types:

{node_descriptions}

## LINK RELATION TYPES

When creating links between nodes, use these relation types:

{link_descriptions}

## EXAMPLE WORKFLOWS

{example_workflows}

## IMPORTANT GUIDELINES

- **Be selective about what you include in the KG** - Not every detail needs to be modeled
- **Focus on actionable insights** - Prioritize practical, useful information for the user
- **Maintain context awareness** - Use the conversation history and entity awareness
- **Use KGML command chaining** - Create complex operations by chaining simple commands
- **Track goals and outcomes** - Every user request should generate trackable outcomes
- **Generate user-friendly responses** - Final outputs should be natural and helpful

Look for the CURRENT_NODE markers in the KG to identify the focal point for your actions.

Remember, your responses should ALWAYS be in proper KGML format and should NEVER include explanatory text outside of KGML instructions.
"""

    # Format the prompt with the dynamically generated components
    formatted_prompt = prompt.format(
        node_descriptions=node_descriptions,
        link_descriptions=link_descriptions,
        example_workflows=example_workflows
    )

    return formatted_prompt


if __name__ == "__main__":
    # If run directly, print the registered modules and generated prompt
    print("Registered Module Pairs:")
    for models_module, constants_module in MODULE_REGISTRY:
        print(f"  - Models: {models_module}")
        print(f"    Constants: {constants_module}")
    print("\n" + "=" * 80 + "\n")

    # Load all registered modules
    module_infos = []
    for models_module_name, constants_module_name in MODULE_REGISTRY:
        module_info = ModuleInfo(models_module_name, constants_module_name)
        module_infos.append(module_info)

    # Print all node types and their classes
    print("NODE TYPES AND THEIR PROPERTIES:")
    print(_generate_node_type_descriptions(module_infos))
    print("\n" + "=" * 80 + "\n")

    # Print all link relation types
    print("LINK RELATION TYPES:")
    print(_generate_link_relation_descriptions(module_infos))
    print("\n" + "=" * 80 + "\n")

    # Print the full prompt
    print("COMPLETE SYSTEM PROMPT:")
    print(generate_kgml_system_prompt())