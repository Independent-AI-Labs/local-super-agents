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
Generate a Mermaid diagram for the following architecture document:

{document_content}

Based on the content, identify all components and their relationships.
"""

# Module diagram specific prompt
MODULE_DIAGRAM_PROMPT = """
Create a module diagram using Mermaid syntax for the following architecture document:

{document_content}

Components identified:
{components}

Relationships identified:
{relationships}

Create a complex Mermaid flowchart (using flowchart TD syntax) that clearly shows the architecture modules, 
their hierarchical organization, and dependencies between them. Use diverse node shapes for modules and arrows for dependencies.
Include appropriate node styles, colors, and grouping to make the diagram clear and professional.

Follow this complex template style but adapt it to the specific modules in the document:

```mermaid
flowchart TD
    %% Main Module Nodes
    Core((Core Module)):::core --> Components[/"UI Components"/]:::component
    Core --> Services{Service Layer}:::service
    Services -- Implements --> UseCases[Business Logic]:::business
    
    %% Module Hierarchy and Subsystems
    UseCases --> Repository[Repository Layer]:::repository
    Repository --> DataAccess[(Data Access)]:::data
    Repository --> ExternalAPI[External APIs]:::external
    
    %% Dependency Relationships
    Components --> Utils[Utility Functions]:::utility
    Services --> Utils
    Components --> Validation[Validation Module]:::validation
    
    %% Additional Hierarchies
    subgraph Security [Security Layer]
      Auth[Authentication]:::security
      Encrypt[Encryption]:::security
      Auth --> Encrypt
    end
    
    Services --> Security
    
    %% Module Dependencies
    DataAccess --> Config[Configuration]:::config
    ExternalAPI --> Config
    
    %% Clickable Module Documentation
    click Core "https://docs.example.com/core" "Core Module Documentation"
    click Components "https://docs.example.com/components" "Components Documentation"
    
    %% Custom Module Styling
    classDef core fill:#9f6,stroke:#333,stroke-width:2px;
    classDef component fill:#ccf,stroke:#333,stroke-width:2px;
    classDef service fill:#fcf,stroke:#333,stroke-dasharray: 5 5;
    classDef business fill:#cfc,stroke:#333,stroke-width:2px;
    classDef repository fill:#ffc,stroke:#333,stroke-width:2px;
    classDef data fill:#cff,stroke:#333,stroke-dasharray: 2 2;
    classDef external fill:#f9c,stroke:#333,stroke-width:2px;
    classDef utility fill:#fed,stroke:#333,stroke-width:2px;
    classDef validation fill:#9cf,stroke:#333,stroke-width:2px;
    classDef security fill:#fc9,stroke:#333,stroke-width:2px;
    classDef config fill:#f66,stroke:#333,stroke-width:2px;
    
    %% Link Styling 
    linkStyle 0 stroke:#f66, stroke-width:2px, stroke-dasharray: 5 5;
    linkStyle 1 stroke:#00f, stroke-width:2px;
    linkStyle 2 stroke:#0a0, stroke-width:2px, stroke-dasharray: 2 2;
```

Output only valid Mermaid syntax, without any explanation or markdown code blocks.
The output should start with 'flowchart TD' or 'graph TD'.
"""

# Dataflow diagram specific prompt
DATAFLOW_DIAGRAM_PROMPT = """
Create a data flow diagram using Mermaid syntax for the following architecture document:

{document_content}

Components identified:
{components}

Relationships identified:
{relationships}

Create a complex Mermaid flowchart that clearly shows data flows between components.
Show all of these elements with distinctive styling:
1. Data sources and sinks (external entities)
2. Processes that transform data
3. Data stores
4. Data flows between entities with directional arrows
5. Clear labels on all flows, processes, and entities

Follow this complex template style but adapt it to the specific data flows in the document:

```mermaid
flowchart TD
    %% External Data Sources
    User((User)):::external --> |Input Data| Validation[/"Input Validation"/]:::process
    ExternalAPI[[External API]]:::external --> |Raw Data| DataTransform{Data Transformation}:::process
    
    %% Data Transformation Processes
    Validation --> |Validated Data| CoreProcess[Core Processing]:::process
    DataTransform --> |Transformed Data| CoreProcess
    CoreProcess --> |Processed Results| DataAggregation[Data Aggregation]:::process
    
    %% Data Storage
    CoreProcess --> |Store| PrimaryDB[(Primary Database)]:::storage
    PrimaryDB --> |Retrieve| DataAggregation
    DataAggregation --> |Archive| ArchiveDB[(Archive Database)]:::storage
    
    %% Output Flows
    DataAggregation --> |Aggregated Data| ReportGen[Report Generation]:::process
    ReportGen --> |Reports| OutputAPI[[API Endpoint]]:::external
    ReportGen --> |Notifications| NotificationSys[Notification System]:::process
    
    %% Data Flow Subgraphs
    subgraph CachingLayer [Caching System]
        Cache[(Cache Store)]:::cache
        CacheCheck{Cache Valid?}:::decision
        CacheUpdate[Update Cache]:::process
        
        CacheCheck -- Yes --> Cache
        CacheCheck -- No --> CacheUpdate
        CacheUpdate --> Cache
    end
    
    CoreProcess -.-> |Check Cache| CacheCheck
    Cache -.-> |Cached Data| CoreProcess
    
    %% Classification and Styling
    classDef external fill:#8ff,stroke:#333,stroke-width:2px;
    classDef process fill:#fcf,stroke:#333,stroke-width:2px;
    classDef storage fill:#ffc,stroke:#333,stroke-dasharray: 2 2;
    classDef cache fill:#cfc,stroke:#333,stroke-width:2px;
    classDef decision fill:#f9c,stroke:#333,stroke-dasharray: 5 5;
    
    %% Link Styling
    linkStyle 0 stroke:#f00, stroke-width:2px;
    linkStyle 1 stroke:#00f, stroke-width:2px;
    linkStyle 2 stroke:#0a0, stroke-width:2px, stroke-dasharray: 2 2;
    linkStyle 3 stroke:#f0f, stroke-width:2px;
    
    %% Interactive Elements
    click PrimaryDB "https://docs.example.com/database" "Database Documentation"
    click OutputAPI "https://docs.example.com/api" "API Documentation"
```

Output only valid Mermaid syntax, without any explanation or markdown code blocks.
The output should start with 'flowchart TD' or 'graph TD'.
"""

# Security diagram specific prompt
SECURITY_DIAGRAM_PROMPT = """
Create a security diagram using Mermaid syntax for the following architecture document:

{document_content}

Components identified:
{components}

Relationships identified:
{relationships}

Create a complex Mermaid flowchart that clearly shows security aspects of the system including:
1. Trust boundaries (using subgraphs with dotted borders)
2. Authentication points (using specialized node labels)
3. Data encryption points (using specialized node labels)
4. Potential attack vectors (using red edges or nodes)
5. Security controls (using specialized node labels)
6. User access levels (using node colors or styles)

Follow this complex template style but adapt it to the specific security aspects in the document:

```mermaid
flowchart TD
    %% External Actors
    User((End User)):::user --> |HTTPS| LoadBalancer[Load Balancer]:::edge
    Admin((Admin User)):::admin --> |HTTPS + MFA| AdminPortal[Admin Portal]:::adminAccess
    Attacker((Attacker)):::attacker -.-> |Potential Attack Vector| LoadBalancer
    
    %% Edge Security
    LoadBalancer --> |WAF Filtered| APIGateway{API Gateway}:::gateway
    APIGateway --> |JWT Validation| AuthService[Authentication Service]:::authService
    
    %% Authentication Flow
    AuthService --> |Verify Credentials| IdentityProvider[(Identity Provider)]:::idp
    IdentityProvider --> |Token| AuthService
    AuthService --> |Valid Session| ServiceLayer[Service Layer]:::service
    
    %% Trust Boundaries
    subgraph PublicZone [Public DMZ Zone]
        LoadBalancer
        APIGateway
    end
    
    subgraph RestrictedZone [Restricted Zone]
        AuthService
        ServiceLayer
        
        subgraph PrivateZone [Private Data Zone]
            Database[(Encrypted Database)]:::database
            Secrets[(Secrets Vault)]:::vault
        end
    end
    
    %% Data Flows and Encryption Points
    ServiceLayer --> |Encrypted Channel| Database
    ServiceLayer --> |Encrypted Read| Secrets
    
    %% Security Controls
    WAF[Web Application Firewall]:::security --> LoadBalancer
    IDS[Intrusion Detection]:::security --> RestrictedZone
    
    %% Monitoring and Auditing
    ServiceLayer --> |Audit Logs| LogCollection[Log Collection]:::monitoring
    AuthService --> |Auth Logs| LogCollection
    LogCollection --> |Alerts| SIEM[Security Monitoring]:::monitoring
    
    %% Security Classes and Styles
    classDef user fill:#0f6,stroke:#333,stroke-width:2px;
    classDef admin fill:#09f,stroke:#333,stroke-width:2px;
    classDef attacker fill:#f00,stroke:#f00,stroke-width:2px,stroke-dasharray: 5 5;
    classDef edge fill:#fc9,stroke:#333,stroke-width:2px;
    classDef gateway fill:#fcf,stroke:#333,stroke-width:2px;
    classDef authService fill:#9cf,stroke:#333,stroke-width:2px;
    classDef service fill:#cfc,stroke:#333,stroke-width:2px;
    classDef database fill:#ffc,stroke:#333,stroke-dasharray: 2 2;
    classDef vault fill:#f9c,stroke:#333,stroke-width:3px;
    classDef idp fill:#fed,stroke:#333,stroke-width:2px;
    classDef security fill:#c99,stroke:#333,stroke-width:2px;
    classDef monitoring fill:#9fc,stroke:#333,stroke-width:2px;
    classDef adminAccess fill:#99f,stroke:#333,stroke-width:2px;
    
    %% Link Styling
    linkStyle 0 stroke:#0a0, stroke-width:2px;
    linkStyle 1 stroke:#0a0, stroke-width:2px;
    linkStyle 2 stroke:#f00, stroke-width:2px, stroke-dasharray: 3 3;
    linkStyle 3 stroke:#00f, stroke-width:2px;
    
    %% Interactive Elements
    click Database "https://docs.example.com/database" "Database Security Documentation"
    click AuthService "https://docs.example.com/auth" "Authentication Documentation"
```

Output only valid Mermaid syntax, without any explanation or markdown code blocks.
The output should start with 'flowchart TD' or 'graph TD'.
Include a legend explaining the security-related symbols.
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
