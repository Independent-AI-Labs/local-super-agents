"""
Architecture management controller for VibeCheck - enhanced with multi-document support.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from vibecheck.models.architecture import ArchitecturalDocument, ArchitecturalDiagram
from vibecheck.integrations.llm import analyze_architecture
from vibecheck.utils.diagram_utils import generate_diagrams, extract_components_from_architecture, extract_relationships_from_architecture
from vibecheck.utils.file_utils import ensure_directory, write_file, read_file
from vibecheck.utils.cache_utils import AnalysisCache
from vibecheck.utils.architecture_utils import ArchitectureDocumentManager
from vibecheck import config


class ArchitectureController:
    """
    Controller for architecture-related functions including document management
    and diagram generation with multi-document support.
    """

    @staticmethod
    def save_document(project_path: str, doc_name: str, content: str) -> bool:
        """
        Save an architectural document to the project.

        Args:
            project_path: Path to the project
            doc_name: Name of the document
            content: Content of the architectural document

        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Create the document model
            document = ArchitecturalDocument(
                content=content,
                last_modified=datetime.now()
            )
            
            # Ensure the directory exists
            ensure_directory(os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR))
            
            # Save the document - FIXED JSON SERIALIZATION
            document_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.json")
            
            # Convert to dictionary and then to JSON manually
            document_dict = document.dict()
            
            with open(document_path, 'w') as f:
                json.dump(document_dict, f, indent=2, default=str)
            
            # Also save as a markdown file for easier viewing
            md_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.md")
            write_file(md_path, content)
            
            # Invalidate any cached analysis
            AnalysisCache.invalidate_cache(project_path, analysis_type=f"architecture_{doc_name}")
            
            return True
        except Exception as e:
            print(f"Error saving architectural document: {e}")
            return False

    @staticmethod
    def load_document(project_path: str, doc_name: str) -> Optional[ArchitecturalDocument]:
        """
        Load an architectural document from the project.

        Args:
            project_path: Path to the project
            doc_name: Name of the document

        Returns:
            Loaded ArchitecturalDocument or None if not found
        """
        document_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.json")
        
        if not os.path.exists(document_path):
            return None
        
        try:
            with open(document_path, 'r') as f:
                document_data = json.load(f)
                return ArchitecturalDocument.parse_obj(document_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading architectural document: {e}")
            
            # Try to recover from markdown file if available
            md_path = os.path.join(project_path, config.ARCHITECTURE_DOCS_DIR, f"{doc_name}.md")
            if os.path.exists(md_path):
                content = read_file(md_path)
                if content:
                    return ArchitecturalDocument(
                        content=content,
                        last_modified=datetime.fromtimestamp(os.path.getmtime(md_path))
                    )
            
            return None

    @staticmethod
    def generate_diagrams(project_path: str, doc_name: str, document_content: str) -> Dict[str, ArchitecturalDiagram]:
        """
        Generate architectural diagrams from the document.

        Args:
            project_path: Path to the project
            doc_name: Name of the document
            document_content: Content of the architectural document

        Returns:
            Dictionary of generated diagrams by type
        """
        diagrams = {}
        now = datetime.now()
        
        # Check cache first
        cache_key = f"architecture_{doc_name}"
        cached_diagrams = AnalysisCache.get_cached_analysis(
            project_path, cache_key, "diagrams"
        )
        
        if cached_diagrams and isinstance(cached_diagrams, dict):
            # Convert cached diagrams to ArchitecturalDiagram objects
            for diagram_type, svg_content in cached_diagrams.items():
                diagrams[diagram_type] = ArchitecturalDiagram(
                    diagram_type=diagram_type,
                    content=svg_content,
                    generated_at=now
                )
        
        if not diagrams:
            # Generate new diagrams
            try:
                # Extract components and relationships
                components = extract_components_from_architecture(document_content)
                relationships = extract_relationships_from_architecture(document_content)
                
                # Generate SVG diagrams
                svg_diagrams = generate_diagrams(document_content)
                
                # Create ArchitecturalDiagram objects
                for diagram_type, svg_content in svg_diagrams.items():
                    diagrams[diagram_type] = ArchitecturalDiagram(
                        diagram_type=diagram_type,
                        content=svg_content,
                        generated_at=now
                    )
                
                # Cache the generated diagrams
                AnalysisCache.cache_analysis(
                    project_path,
                    cache_key,
                    "diagrams",
                    {k: v.content for k, v in diagrams.items()},
                    ttl_seconds=86400  # 24 hours
                )
            except Exception as e:
                print(f"Error generating diagrams: {e}")
                # Create placeholder diagrams if generation fails
                for diagram_type in ["module", "dataflow", "security"]:
                    diagrams[diagram_type] = ArchitecturalDiagram(
                        diagram_type=diagram_type,
                        content=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300"><text x="50" y="50" font-family="Arial" font-size="16">Placeholder {diagram_type} diagram</text></svg>',
                        generated_at=now
                    )
        
        # Save the diagrams
        ArchitectureController._save_diagrams(project_path, doc_name, diagrams)
        
        return diagrams

    @staticmethod
    def get_diagram(project_path: str, doc_name: str, diagram_type: str) -> Optional[ArchitecturalDiagram]:
        """
        Get a specific architectural diagram.

        Args:
            project_path: Path to the project
            doc_name: Name of the document
            diagram_type: Type of diagram to retrieve

        Returns:
            ArchitecturalDiagram or None if not found
        """
        diagram_path = os.path.join(
            project_path, 
            config.ARCHITECTURE_DIAGRAMS_DIR, 
            f"{doc_name}_{diagram_type}.json"
        )
        
        if not os.path.exists(diagram_path):
            return None
        
        try:
            with open(diagram_path, 'r') as f:
                diagram_data = json.load(f)
                return ArchitecturalDiagram.parse_obj(diagram_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading architectural diagram: {e}")
            
            # Try to recover from SVG file if available
            svg_path = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR, f"{doc_name}_{diagram_type}.svg")
            if os.path.exists(svg_path):
                content = read_file(svg_path)
                if content:
                    return ArchitecturalDiagram(
                        diagram_type=diagram_type,
                        content=content,
                        generated_at=datetime.fromtimestamp(os.path.getmtime(svg_path))
                    )
            
            return None

    @staticmethod
    def get_all_diagrams(project_path: str, doc_name: Optional[str] = None) -> Dict[str, Dict[str, ArchitecturalDiagram]]:
        """
        Get all architectural diagrams for the project or a specific document.

        Args:
            project_path: Path to the project
            doc_name: Optional name of the document

        Returns:
            Dictionary of all diagrams by document and type
        """
        diagrams = {}
        diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
        
        if not os.path.exists(diagrams_dir):
            return diagrams
        
        for filename in os.listdir(diagrams_dir):
            if filename.endswith(".json"):
                # Parse filename to get document name and diagram type
                parts = filename.split("_")
                if len(parts) < 2:
                    continue
                
                doc = "_".join(parts[:-1])  # Handle document names with underscores
                diagram_type = parts[-1].split(".")[0]
                
                # Skip if we're looking for a specific document
                if doc_name and doc != doc_name:
                    continue
                
                # Get the diagram
                diagram = ArchitectureController.get_diagram(project_path, doc, diagram_type)
                if diagram:
                    if doc not in diagrams:
                        diagrams[doc] = {}
                    diagrams[doc][diagram_type] = diagram
        
        return diagrams

    @staticmethod
    def analyze_architecture_document(project_path: str, doc_name: str, document_content: str) -> str:
        """
        Analyze an architecture document with an LLM with enhanced focus on code quality.

        Args:
            project_path: Path to the project
            doc_name: Name of the document
            document_content: Content of the architectural document

        Returns:
            Enhanced critical analysis of the architecture
        """
        # Check cache first
        cache_key = f"architecture_{doc_name}"
        cached_analysis = AnalysisCache.get_cached_analysis(
            project_path, cache_key, "analysis"
        )

        if cached_analysis and isinstance(cached_analysis, str):
            return cached_analysis

        try:
            # Generate new analysis with enhanced critical focus
            # Modify the prompt to the LLM to emphasize code quality
            enhanced_prompt = f"""
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

            Provide a detailed analysis with concrete improvement recommendations.
            """

            # Call the existing analyze_architecture function with enhanced prompt
            analysis = analyze_architecture(enhanced_prompt)

            # Format the analysis to highlight critical aspects
            formatted_analysis = f"""
            # Critical Architecture Analysis

            ## Code Quality Review

            {analysis}

            ## Recommended Improvements

            Based on the analysis, these are the top issues that should be addressed to improve code quality:

            1. [Will be filled by the LLM based on the analysis]
            2. [Will be filled by the LLM based on the analysis]
            3. [Will be filled by the LLM based on the analysis]
            """

            # Cache the enhanced analysis
            AnalysisCache.cache_analysis(
                project_path,
                cache_key,
                "analysis",
                formatted_analysis,
                ttl_seconds=86400  # 24 hours
            )

            # Save analysis to file
            analysis_dir = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR)
            ensure_directory(analysis_dir)
            analysis_path = os.path.join(analysis_dir, f"{doc_name}.json")

            with open(analysis_path, 'w') as f:
                json.dump({
                    'content': formatted_analysis,
                    'generated_at': datetime.now().isoformat()
                }, f, indent=2)

            return formatted_analysis
        except Exception as e:
            print(f"Error analyzing architecture document: {e}")
            return "Error analyzing architecture document. Please try again later."

    @staticmethod
    def get_components_and_relationships(project_path: str, doc_name: str) -> Dict:
        """
        Get the components and relationships from the architecture document.

        Args:
            project_path: Path to the project
            doc_name: Name of the document

        Returns:
            Dictionary with components and relationships
        """
        document = ArchitectureController.load_document(project_path, doc_name)
        if not document or not document.content:
            return {"components": [], "relationships": []}
        
        # Check cache first
        cache_key = f"architecture_{doc_name}"
        cached_data = AnalysisCache.get_cached_analysis(
            project_path, cache_key, "components_and_relationships"
        )
        
        if cached_data and isinstance(cached_data, dict):
            return cached_data
        
        try:
            # Extract components and relationships
            components = extract_components_from_architecture(document.content)
            relationships = extract_relationships_from_architecture(document.content)
            
            result = {
                "components": components,
                "relationships": relationships
            }
            
            # Cache the data
            AnalysisCache.cache_analysis(
                project_path,
                cache_key,
                "components_and_relationships",
                result,
                ttl_seconds=86400  # 24 hours
            )
            
            return result
        except Exception as e:
            print(f"Error extracting components and relationships: {e}")
            return {"components": [], "relationships": []}

    @staticmethod
    def get_mermaid_diagram(project_path: str, doc_name: str) -> str:
        """
        Get a Mermaid diagram representation of the architecture.

        Args:
            project_path: Path to the project
            doc_name: Name of the document

        Returns:
            Mermaid diagram code
        """
        from vibecheck.utils.diagram_utils import get_mermaid_from_components
        
        # Get components and relationships
        data = ArchitectureController.get_components_and_relationships(project_path, doc_name)
        components = data["components"]
        relationships = data["relationships"]
        
        if not components:
            return "graph TD\n    A[No components found] --> B[Add components to your architecture document]"
        
        return get_mermaid_from_components(components, relationships)

    @staticmethod
    def _save_diagrams(project_path: str, doc_name: str, diagrams: Dict[str, ArchitecturalDiagram]) -> None:
        """
        Save architectural diagrams to the project.

        Args:
            project_path: Path to the project
            doc_name: Name of the document
            diagrams: Dictionary of diagrams to save
        """
        # Ensure the directory exists
        diagrams_dir = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR)
        ensure_directory(diagrams_dir)
        
        # Save each diagram
        for diagram_type, diagram in diagrams.items():
            diagram_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.json")
            
            # Convert to dictionary and then to JSON manually
            diagram_dict = diagram.dict()
            
            with open(diagram_path, 'w') as f:
                json.dump(diagram_dict, f, indent=2, default=str)
            
            # Also save the SVG content to a separate file for easy viewing
            svg_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.svg")
            write_file(svg_path, diagram.content)

    @staticmethod
    def get_documents_in_scope(project_path: str) -> List[str]:
        """
        Get the list of documents in the current architecture scope.

        Args:
            project_path: Path to the project

        Returns:
            List of document names in the scope
        """
        return ArchitectureDocumentManager.get_scope(project_path)


# Make sure these directories are defined in config.py
if not hasattr(config, 'ARCHITECTURE_ANALYSIS_DIR'):
    config.ARCHITECTURE_ANALYSIS_DIR = os.path.join(config.VIBECHECK_DIR, 'architecture', 'analysis')
