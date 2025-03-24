"""
Prompt templates for architecture-related LLM queries.

This module contains all prompts used for architecture analysis, 
diagram generation, and component extraction.
"""

# Main prompt for architecture document analysis
ARCHITECTURE_ANALYSIS_PROMPT = """
Analyze the following software architecture document with a strong focus on improving code quality.

Be critical and identify:
1. Potential code quality issues
2. Design weaknesses that could lead to technical debt
3. Component coupling and cohesion concerns
4. Violation of SOLID principles
5. Architectural anti-patterns
6. Security vulnerabilities
7. Performance bottlenecks
8. Testing challenges

Document content:
{document_content}

Format your response as a Markdown document with the following sections:
# Critical Architecture Analysis

## Overall Assessment
[Provide a high-level assessment of the architecture]

## Code Quality Concerns
[List and explain code quality issues]

## Design Weaknesses
[List and explain design weaknesses]

## SOLID Principle Violations
[Identify any violations of SOLID principles]

## Security Considerations
[Identify security concerns]

## Performance Considerations
[Identify performance bottlenecks]

## Testability
[Discuss testing challenges]

## Recommended Improvements
[Provide specific, actionable recommendations in order of priority]
"""

# Generic diagram generation prompt
DIAGRAM_GENERATION_PROMPT = """
Generate a diagram for the following architecture document:

{document_content}

Based on the content, identify all components and their relationships.
"""

# Module diagram specific prompt
MODULE_DIAGRAM_PROMPT = """
Create a module diagram in SVG format for the following architecture document:

{document_content}

Components identified:
{components}

Relationships identified:
{relationships}

Create an SVG module diagram that clearly shows the architecture modules, their hierarchical organization, 
and dependencies between them. Use boxes for modules and arrows for dependencies.
Include appropriate colors and styling to make the diagram clear and professional.

Output only valid SVG XML syntax without any explanation or markdown.
"""

# Dataflow diagram specific prompt
DATAFLOW_DIAGRAM_PROMPT = """
Create a data flow diagram in SVG format for the following architecture document:

{document_content}

Components identified:
{components}

Relationships identified:
{relationships}

Create an SVG data flow diagram that shows:
1. Data sources and sinks (external entities)
2. Processes that transform data
3. Data stores
4. Data flows between entities with directional arrows
5. Clear labels on all flows, processes, and entities

Use different shapes for different types of entities:
- Rounded rectangles for processes
- Cylinders for data stores
- Rectangles for external entities
- Arrows with labels for data flows

Output only valid SVG XML syntax without any explanation or markdown.
"""

# Security diagram specific prompt
SECURITY_DIAGRAM_PROMPT = """
Create a security diagram in SVG format for the following architecture document:

{document_content}

Components identified:
{components}

Relationships identified:
{relationships}

Create an SVG security diagram that clearly shows:
1. Trust boundaries (using dotted lines to enclose components)
2. Authentication points (using padlock symbols)
3. Data encryption points (using key symbols)
4. Potential attack vectors (using red warning symbols)
5. Security controls (using shield symbols)
6. User access levels (using different colors)

Include a legend explaining all symbols and colors used.
Make the diagram clear, professional, and easy to understand.

Output only valid SVG XML syntax without any explanation or markdown.
"""

# Mermaid diagram prompt
MERMAID_DIAGRAM_PROMPT = """
Create a Mermaid diagram for the following architecture document:

{document_content}

Generate a comprehensive Mermaid diagram code (using graph TD or flowchart TD syntax) 
that shows all components and their relationships.

Use appropriate node shapes to distinguish between:
- Components/Modules: rectangular nodes
- Services: rounded rectangular nodes
- Databases: cylinder nodes
- External systems: hexagonal nodes

Use color and styling to improve clarity.
Add descriptive labels to relationships between nodes.

Output only valid Mermaid syntax in a code block without any explanation.
"""

# Components extraction prompt
COMPONENTS_EXTRACTION_PROMPT = """
Extract all architectural components from the following document:

{document_content}

For each component, identify:
1. Name
2. Description/purpose
3. Main responsibilities
4. Technologies used (if specified)

Format the output as JSON with the following structure:
[
  {
    "name": "ComponentName",
    "description": "Component description",
    "responsibilities": ["Responsibility 1", "Responsibility 2"],
    "technologies": ["Technology 1", "Technology 2"]
  }
]
"""

# Relationships extraction prompt
RELATIONSHIPS_EXTRACTION_PROMPT = """
Extract all relationships between architectural components from the following document:

{document_content}

For each relationship, identify:
1. Source component
2. Target component
3. Relationship type (depends_on, calls, uses, includes, implements, extends, contains)
4. Description of the interaction

Format the output as JSON with the following structure:
[
  {
    "source": "SourceComponent",
    "target": "TargetComponent",
    "type": "depends_on",
    "description": "Description of how SourceComponent depends on TargetComponent"
  }
]
"""
