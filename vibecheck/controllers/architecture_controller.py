"""
Architecture management controller for VibeCheck - enhanced with multi-document support.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from vibecheck.config import MODEL_FOR_ARCH_ANALYSIS
from vibecheck.models.architecture import ArchitecturalDocument, ArchitecturalDiagram
from vibecheck.integrations.llm import analyze_with_llm
from vibecheck.utils.diagram_utils import extract_components_from_architecture, extract_relationships_from_architecture
from vibecheck.utils.file_utils import ensure_directory, write_file, read_file
from vibecheck.utils.cache_utils import AnalysisCache
from vibecheck.utils.architecture_utils import ArchitectureDocumentManager
from vibecheck.constants.architecture_prompts import (
    DIAGRAM_GENERATION_PROMPT, 
    MERMAID_DIAGRAM_PROMPT, 
    MODULE_DIAGRAM_PROMPT,
    DATAFLOW_DIAGRAM_PROMPT,
    SECURITY_DIAGRAM_PROMPT,
    ARCHITECTURE_ANALYSIS_PROMPT
)
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
        Generate architectural diagrams from the document using the prompt model.

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
            # Generate new diagrams using prompt_model
            try:
                # Extract components and relationships for context
                components = extract_components_from_architecture(document_content)
                relationships = extract_relationships_from_architecture(document_content)
                
                # Generate diagrams for each type
                diagram_types = {
                    "module": MODULE_DIAGRAM_PROMPT,
                    "dataflow": DATAFLOW_DIAGRAM_PROMPT,
                    "security": SECURITY_DIAGRAM_PROMPT,
                    "mermaid": MERMAID_DIAGRAM_PROMPT
                }
                
                for diagram_type, prompt_template in diagram_types.items():
                    try:
                        # Prepare prompt with document content
                        prompt = prompt_template.format(
                            document_content=document_content,
                            components=json.dumps(components, indent=2),
                            relationships=json.dumps(relationships, indent=2)
                        )
                        
                        # Call analyze_with_llm to generate the diagram
                        if diagram_type == "mermaid":
                            # For mermaid, we want the raw mermaid code
                            result = analyze_with_llm(document_content, MODEL_FOR_ARCH_ANALYSIS, prompt)
                            # Extract mermaid code from the result
                            mermaid_code = result.strip()
                            # Store mermaid code directly
                            svg_content = mermaid_code
                        else:
                            # For SVG diagrams, prompt for SVG content
                            svg_prompt = f"{prompt}\nPlease output a complete SVG diagram."
                            result = analyze_with_llm(document_content, MODEL_FOR_ARCH_ANALYSIS, svg_prompt)
                            
                            # Extract SVG content (anything between <svg and </svg>)
                            import re
                            svg_match = re.search(r'<svg.*?</svg>', result, re.DOTALL)
                            if svg_match:
                                svg_content = svg_match.group(0)
                            else:
                                # Fallback if no SVG tag found - create minimal valid SVG
                                svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300"><text x="50" y="50" font-family="Arial" font-size="16">{diagram_type.capitalize()} diagram for {doc_name}</text></svg>'
                        
                        # Create ArchitecturalDiagram object
                        diagrams[diagram_type] = ArchitecturalDiagram(
                            diagram_type=diagram_type,
                            content=svg_content,
                            generated_at=now
                        )
                        
                    except Exception as e:
                        print(f"Error generating {diagram_type} diagram: {e}")
                        # Add a placeholder diagram for the failed type
                        diagrams[diagram_type] = ArchitecturalDiagram(
                            diagram_type=diagram_type,
                            content=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300"><text x="50" y="50" font-family="Arial" font-size="16">Failed to generate {diagram_type} diagram: {str(e)}</text></svg>',
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
                for diagram_type in ["module", "dataflow", "security", "mermaid"]:
                    diagrams[diagram_type] = ArchitecturalDiagram(
                        diagram_type=diagram_type,
                        content=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300"><text x="50" y="50" font-family="Arial" font-size="16">Error generating {diagram_type} diagram: {str(e)}</text></svg>',
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
            # Generate new analysis with analyze_with_llm
            system_prompt = ARCHITECTURE_ANALYSIS_PROMPT.format(document_content=document_content)
            
            # Call analyze_with_llm for more control
            analysis = analyze_with_llm(document_content, MODEL_FOR_ARCH_ANALYSIS, system_prompt)
            
            # Cache the enhanced analysis
            AnalysisCache.cache_analysis(
                project_path,
                cache_key,
                "analysis",
                analysis,
                ttl_seconds=86400  # 24 hours
            )

            # Save analysis to file
            analysis_dir = os.path.join(project_path, config.ARCHITECTURE_ANALYSIS_DIR)
            ensure_directory(analysis_dir)
            analysis_path = os.path.join(analysis_dir, f"{doc_name}.json")

            with open(analysis_path, 'w') as f:
                json.dump({
                    'content': analysis,
                    'generated_at': datetime.now().isoformat()
                }, f, indent=2)

            return analysis
        except Exception as e:
            print(f"Error analyzing architecture document: {e}")
            return f"Error analyzing architecture document: {str(e)}. Please try again later."

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
            # Extract components and relationships using analyze_with_llm for more accurate results
            system_prompt = f"""
            Extract components and their relationships from the following architecture document.
            
            Output a JSON object with two keys:
            1. "components" - an array of objects with "name" and "description" fields
            2. "relationships" - an array of objects with "source", "target", and "type" fields
            
            The relationship types should be one of: "depends_on", "calls", "uses", "includes", "implements", "extends", "contains"
            
            Format the output as valid JSON only.
            """
            
            result = analyze_with_llm(document.content, MODEL_FOR_ARCH_ANALYSIS, system_prompt)
            
            # Extract JSON from the result
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group(0))
                    components = extracted_data.get("components", [])
                    relationships = extracted_data.get("relationships", [])
                except json.JSONDecodeError:
                    # Fallback to traditional extraction methods
                    components = extract_components_from_architecture(document.content)
                    relationships = extract_relationships_from_architecture(document.content)
            else:
                # Fallback to traditional extraction methods
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
            # Fallback to traditional extraction methods
            try:
                components = extract_components_from_architecture(document.content)
                relationships = extract_relationships_from_architecture(document.content)
                return {"components": components, "relationships": relationships}
            except:
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
        # Get document content
        document = ArchitectureController.load_document(project_path, doc_name)
        if not document or not document.content:
            return "graph TD\n    A[No document content found] --> B[Add content to your architecture document]"
        
        # Check cache for existing mermaid diagram
        cache_key = f"architecture_{doc_name}"
        cached_diagrams = AnalysisCache.get_cached_analysis(project_path, cache_key, "diagrams")
        if cached_diagrams and isinstance(cached_diagrams, dict) and "mermaid" in cached_diagrams:
            return cached_diagrams["mermaid"]
        
        # Generate a new mermaid diagram using prompt_model
        try:
            system_prompt = MERMAID_DIAGRAM_PROMPT.format(document_content=document.content)
            mermaid_code = analyze_with_llm(document.content, MODEL_FOR_ARCH_ANALYSIS, system_prompt)
            
            # Clean up the response to extract just the mermaid code
            import re
            mermaid_match = re.search(r'```mermaid\n(.*?)\n```', mermaid_code, re.DOTALL)
            if mermaid_match:
                cleaned_mermaid = mermaid_match.group(1)
            else:
                # Try to find any code block
                code_match = re.search(r'```\w*\n(.*?)\n```', mermaid_code, re.DOTALL)
                if code_match:
                    cleaned_mermaid = code_match.group(1)
                else:
                    # Use the whole response if no code block is found
                    cleaned_mermaid = mermaid_code.strip()
            
            # Store in cache
            if cached_diagrams and isinstance(cached_diagrams, dict):
                cached_diagrams["mermaid"] = cleaned_mermaid
                AnalysisCache.cache_analysis(
                    project_path,
                    cache_key,
                    "diagrams",
                    cached_diagrams,
                    ttl_seconds=86400  # 24 hours
                )
            else:
                AnalysisCache.cache_analysis(
                    project_path,
                    cache_key,
                    "diagrams",
                    {"mermaid": cleaned_mermaid},
                    ttl_seconds=86400  # 24 hours
                )
            
            return cleaned_mermaid
            
        except Exception as e:
            print(f"Error generating mermaid diagram: {e}")
            # Fallback to generating from components
            try:
                # Get components and relationships
                data = ArchitectureController.get_components_and_relationships(project_path, doc_name)
                components = data["components"]
                relationships = data["relationships"]
                
                from vibecheck.utils.diagram_utils import get_mermaid_from_components
                return get_mermaid_from_components(components, relationships)
            except Exception as e:
                print(f"Fallback mermaid generation failed: {e}")
                return "graph TD\n    A[Error generating diagram] --> B[Please try again]"

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
