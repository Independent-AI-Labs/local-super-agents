"""
Architecture management controller for VibeCheck - enhanced with Mermaid diagram support.
"""

import json
import os
import re
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
    MODULE_DIAGRAM_PROMPT,
    DATAFLOW_DIAGRAM_PROMPT,
    SECURITY_DIAGRAM_PROMPT,
    ARCHITECTURE_ANALYSIS_PROMPT
)
from vibecheck.constants.architecture_templates import (
    MERMAID_DIRECT_TEMPLATE,
    DIAGRAM_ERROR_TEMPLATE,
    MERMAID_FALLBACK_TEMPLATE,
    NO_DIAGRAM_SELECTED_TEMPLATE,
    NO_DOCUMENT_FIRST_TEMPLATE,
    GENERATING_DIAGRAMS_TEMPLATE
)
from vibecheck import config


class ArchitectureController:
    """
    Controller for architecture-related functions including document management
    and Mermaid diagram generation with multi-document support.
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
        Generate architectural Mermaid diagrams from the document using the prompt model.

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
            for diagram_type, mermaid_content in cached_diagrams.items():
                diagrams[diagram_type] = ArchitecturalDiagram(
                    diagram_type=diagram_type,
                    content=mermaid_content,
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
                    "security": SECURITY_DIAGRAM_PROMPT
                }
                
                for diagram_type, prompt_template in diagram_types.items():
                    try:
                        # Prepare prompt with document content
                        prompt = prompt_template.format(
                            document_content=document_content,
                            components=json.dumps(components, indent=2),
                            relationships=json.dumps(relationships, indent=2)
                        )
                        
                        # Call analyze_with_llm to generate the Mermaid diagram
                        result = analyze_with_llm(document_content, MODEL_FOR_ARCH_ANALYSIS, prompt)
                        
                        # Extract Mermaid code from the result
                        mermaid_code = ArchitectureController._extract_mermaid_code(result)
                        
                        # Create ArchitecturalDiagram object with Mermaid content
                        diagrams[diagram_type] = ArchitecturalDiagram(
                            diagram_type=diagram_type,
                            content=mermaid_code,
                            generated_at=now
                        )
                        
                    except Exception as e:
                        print(f"Error generating {diagram_type} diagram: {e}")
                        # Add a placeholder diagram for the failed type
                        diagrams[diagram_type] = ArchitecturalDiagram(
                            diagram_type=diagram_type,
                            content=f'graph TD\n    A[Error] -->|Failed to generate| B[{diagram_type} diagram]\n    B -->|Error message| C["{str(e)}"]',
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
                        content=f'graph TD\n    A[Error] -->|Failed to generate| B[{diagram_type} diagram]\n    B -->|Error message| C["{str(e)}"]',
                        generated_at=now
                    )
        
        # Save the diagrams
        ArchitectureController._save_diagrams(project_path, doc_name, diagrams)
        
        return diagrams

    @staticmethod
    def _extract_mermaid_code(result: str) -> str:
        """
        Extract Mermaid code from LLM response.
        
        Args:
            result: LLM response text
            
        Returns:
            Clean Mermaid code
        """
        # First, try to extract code from markdown code block with mermaid tag
        code_block_pattern = r'```(?:mermaid)?\s*((?:graph|flowchart|sequenceDiagram|classDiagram|erDiagram|gantt|pie|gitGraph|stateDiagram)[\s\S]*?)```'
        code_match = re.search(code_block_pattern, result, re.DOTALL)
        
        if code_match:
            return code_match.group(1).strip()
        
        # If no mermaid code block found, try to find code block without specific language
        generic_code_block = r'```\s*((?:graph|flowchart|sequenceDiagram|classDiagram|erDiagram|gantt|pie|gitGraph|stateDiagram)[\s\S]*?)```'
        generic_match = re.search(generic_code_block, result, re.DOTALL)
        
        if generic_match:
            return generic_match.group(1).strip()
        
        # If no code block found, try to find just graph TD or flowchart TD syntax
        # This regex captures everything from the "graph/flowchart" keyword to the end
        graph_pattern = r'((?:graph|flowchart)[ \t]+(?:TB|TD|BT|RL|LR)[\s\S]*)'
        graph_match = re.search(graph_pattern, result, re.DOTALL)
        
        if graph_match:
            # Take the whole content from the match to the end
            return graph_match.group(1).strip()
        
        # If still no match, create a simple flowchart with a single node
        # This is a fallback for when the LLM doesn't provide proper Mermaid syntax
        return 'flowchart TD\n    A[Component] --> B[Related Component]'

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
            
            # Try to recover from Mermaid file if available
            mermaid_path = os.path.join(project_path, config.ARCHITECTURE_DIAGRAMS_DIR, f"{doc_name}_{diagram_type}.mmd")
            if os.path.exists(mermaid_path):
                content = read_file(mermaid_path)
                if content:
                    return ArchitecturalDiagram(
                        diagram_type=diagram_type,
                        content=content,
                        generated_at=datetime.fromtimestamp(os.path.getmtime(mermaid_path))
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
            # Extract components with a simplified approach
            # This is a manual extraction to avoid LLM parsing issues
            component_regex = r'(?:Component|Module|Service|Class|Entity|System)\s+([A-Za-z0-9_]+)'
            components_matches = re.findall(component_regex, document.content)
            
            # Create a list of component dictionaries
            components = []
            for name in components_matches:
                # Try to find a description
                desc_regex = rf'{name}\s*[:=\-]\s*([^\n\.]+)'
                desc_match = re.search(desc_regex, document.content)
                description = desc_match.group(1).strip() if desc_match else f"{name} component"
                
                # Add to components list
                if name not in [c.get('name') for c in components]:
                    components.append({
                        "name": name,
                        "description": description
                    })
            
            # If no components found with regex, add a default component
            if not components and "Component 1" in document.content:
                components = [
                    {"name": "Component 1", "description": "Description of component 1"},
                    {"name": "Component 2", "description": "Description of component 2"}
                ]
            elif not components:
                # Extract any capitalized words as potential components
                cap_words_regex = r'\b([A-Z][a-z]+)\b'
                cap_words = re.findall(cap_words_regex, document.content)
                for word in cap_words[:5]:  # Limit to 5 components
                    if word not in [c.get('name') for c in components]:
                        components.append({
                            "name": word,
                            "description": f"{word} component"
                        })
            
            # Extract relationships - look for patterns like "X communicates with Y"
            relationship_patterns = [
                (r'([A-Za-z0-9_]+)\s+communicates\s+with\s+([A-Za-z0-9_]+)', 'communicates_with'),
                (r'([A-Za-z0-9_]+)\s+depends\s+on\s+([A-Za-z0-9_]+)', 'depends_on'),
                (r'([A-Za-z0-9_]+)\s+calls\s+([A-Za-z0-9_]+)', 'calls'),
                (r'([A-Za-z0-9_]+)\s+uses\s+([A-Za-z0-9_]+)', 'uses'),
                (r'([A-Za-z0-9_]+)\s+includes\s+([A-Za-z0-9_]+)', 'includes'),
                (r'([A-Za-z0-9_]+)\s+implements\s+([A-Za-z0-9_]+)', 'implements'),
                (r'([A-Za-z0-9_]+)\s+extends\s+([A-Za-z0-9_]+)', 'extends'),
                (r'([A-Za-z0-9_]+)\s+contains\s+([A-Za-z0-9_]+)', 'contains'),
                (r'([A-Za-z0-9_]+)\s+->+\s+([A-Za-z0-9_]+)', 'connects_to')
            ]
            
            relationships = []
            for pattern, rel_type in relationship_patterns:
                matches = re.findall(pattern, document.content)
                for match in matches:
                    source, target = match
                    relationships.append({
                        "source": source,
                        "target": target,
                        "type": rel_type
                    })
            
            # If no relationships found, create a default relationship if we have components
            if not relationships and len(components) >= 2:
                relationships = [{
                    "source": components[0]["name"],
                    "target": components[1]["name"],
                    "type": "communicates_with"
                }]
            
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
            # Create basic components and relationships as fallback
            fallback_components = [
                {"name": "Component 1", "description": "First component"},
                {"name": "Component 2", "description": "Second component"}
            ]
            fallback_relationships = [{
                "source": "Component 1",
                "target": "Component 2",
                "type": "communicates_with"
            }]
            return {
                "components": fallback_components,
                "relationships": fallback_relationships
            }

    @staticmethod
    def get_mermaid_diagram_html(project_path: str, doc_name: str, diagram_type: str) -> str:
        """
        Get HTML for a Mermaid diagram with embedded rendering.

        Args:
            project_path: Path to the project
            doc_name: Name of the document
            diagram_type: Type of diagram to retrieve

        Returns:
            HTML with embedded Mermaid diagram
        """
        # Get the diagram
        diagram = ArchitectureController.get_diagram(project_path, doc_name, diagram_type)
        
        if not diagram or not diagram.content:
            # Return placeholder if no diagram found
            return DIAGRAM_ERROR_TEMPLATE.format(
                diagram_type=diagram_type,
                error_message=f"No diagram found for {doc_name}."
            )
        
        # Format the Mermaid code for HTML embedding
        mermaid_code = diagram.content.strip()
        
        # Generate a unique ID for this diagram
        import hashlib
        import time
        diagram_id = hashlib.md5(f"{doc_name}_{diagram_type}_{time.time()}".encode()).hexdigest()[:8]
        
        # Apply some sanitization to ensure the Mermaid code is valid
        # Remove any backticks or markdown formatting that might be present
        clean_code = mermaid_code
        if clean_code.startswith("```mermaid"):
            clean_code = clean_code.replace("```mermaid", "", 1)
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3]
        
        # Ensure code starts with proper syntax
        if not any(clean_code.strip().startswith(prefix) for prefix in 
                  ["graph ", "flowchart ", "sequenceDiagram", "classDiagram", "erDiagram"]):
            clean_code = f"flowchart TD\n{clean_code}"
        
        # Remove complex customizations that might cause rendering issues
        clean_code = re.sub(r'%{.*?}%', '', clean_code)  # Remove Mermaid directives
        clean_code = re.sub(r'style\s+\w+\s+[^;]+;', '', clean_code)  # Remove style directives
        clean_code = re.sub(r'linkStyle\s+\d+\s+[^;]+;', '', clean_code)  # Remove link style directives
        
        try:
            # Generate simplified Mermaid code for reliable rendering
            simplified_code = "flowchart TD\n"
            
            # Extract component nodes and relationships
            node_pattern = r'(\w+)(\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|(\[\([^)]*\)\])|(\[\[[^\]]*\]\]))'
            nodes = re.findall(node_pattern, clean_code)
            
            # Add nodes to simplified code
            added_nodes = set()
            for node_match in nodes:
                node_id = node_match[0]
                if node_id not in added_nodes:
                    node_label = re.findall(r'[\(\[\{]([^\]\}\)]+)', node_match[1])
                    label = node_label[0] if node_label else node_id
                    simplified_code += f"    {node_id}[{label}]\n"
                    added_nodes.add(node_id)
            
            # Extract relationships
            rel_pattern = r'(\w+)\s*--?>?\s*(\w+)'
            relationships = re.findall(rel_pattern, clean_code)
            
            # Add relationships to simplified code
            for rel in relationships:
                source, target = rel
                if source in added_nodes and target in added_nodes:
                    simplified_code += f"    {source} --> {target}\n"
            
            # Use the simplified code if it contains both nodes and relationships
            # Otherwise, use the original code
            if len(added_nodes) > 1 and relationships:
                clean_code = simplified_code
            
            # Try to create encoded version for Mermaid Live Editor link
            import base64
            import zlib
            encoded_diagram = base64.urlsafe_b64encode(
                zlib.compress(clean_code.encode('utf-8'), 9)
            ).decode('ascii')
            
            # Generate simplified HTML that will work reliably
            html = f"""
            <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: white; margin: 10px 0;">
              <div style="text-align: center;">
                <pre style="text-align: left; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{clean_code}</pre>
                <p style="font-style: italic; margin-top: 10px; color: #666;">
                  To view this diagram interactively, copy the code above and paste it into 
                  <a href="https://mermaid.live" target="_blank">Mermaid Live Editor</a>
                </p>
              </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            print(f"Error formatting Mermaid HTML: {e}")
            # Ultra-simple fallback - just show the code in a pre tag
            sanitized_code = clean_code.replace('{', '&#123;').replace('}', '&#125;')
            return f"""
            <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: white; margin: 10px 0;">
              <h3 style="margin-top: 0;">{diagram_type.capitalize()} Diagram</h3>
              <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{sanitized_code}</pre>
            </div>
            """

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
            
            # Also save the Mermaid content to a separate file for easy viewing
            mermaid_path = os.path.join(diagrams_dir, f"{doc_name}_{diagram_type}.mmd")
            write_file(mermaid_path, diagram.content)


# Make sure these directories are defined in config.py
if not hasattr(config, 'ARCHITECTURE_ANALYSIS_DIR'):
    config.ARCHITECTURE_ANALYSIS_DIR = os.path.join(config.VIBECHECK_DIR, 'architecture', 'analysis')
