"""
Utility functions for diagram generation from architecture documents.

This module provides utilities for generating various types of diagrams
from architecture documents, extracting components and relationships,
and converting between different diagram formats.
"""

import json
import re
from typing import Dict, List, Any

from vibecheck.config import MODEL_FOR_ARCH_ANALYSIS
from vibecheck.integrations.llm import analyze_with_llm
from vibecheck.constants.architecture_prompts import (
    COMPONENTS_EXTRACTION_PROMPT,
    RELATIONSHIPS_EXTRACTION_PROMPT
)


def extract_components_from_architecture(document_content: str) -> List[Dict[str, Any]]:
    """
    Extract components from an architecture document.

    Args:
        document_content: Content of the architecture document

    Returns:
        List of component dictionaries with name and description
    """
    if not document_content:
        return []

    try:
        # Use analyze_with_llm with the components extraction prompt
        system_prompt = COMPONENTS_EXTRACTION_PROMPT.format(document_content=document_content)
        result = analyze_with_llm(document_content, MODEL_FOR_ARCH_ANALYSIS, system_prompt)
        
        # Extract the JSON from the response
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            try:
                components = json.loads(json_match.group(0))
                # Ensure consistent format with at least name and description
                cleaned_components = []
                for comp in components:
                    if isinstance(comp, dict) and "name" in comp:
                        cleaned_comp = {
                            "name": comp["name"],
                            "description": comp.get("description", "")
                        }
                        # Add other fields if present
                        if "responsibilities" in comp:
                            cleaned_comp["responsibilities"] = comp["responsibilities"]
                        if "technologies" in comp:
                            cleaned_comp["technologies"] = comp["technologies"]
                        cleaned_components.append(cleaned_comp)
                return cleaned_components
            except json.JSONDecodeError:
                # Fall back to regex extraction
                return _extract_components_regex(document_content)
        else:
            # Fall back to regex extraction
            return _extract_components_regex(document_content)
    except Exception as e:
        print(f"Error extracting components with prompt model: {e}")
        # Fall back to regex extraction
        return _extract_components_regex(document_content)


def extract_relationships_from_architecture(document_content: str) -> List[Dict[str, str]]:
    """
    Extract relationships from an architecture document.

    Args:
        document_content: Content of the architecture document

    Returns:
        List of relationship dictionaries with source, target, and type
    """
    if not document_content:
        return []

    try:
        # Use analyze_with_llm with the relationships extraction prompt
        system_prompt = RELATIONSHIPS_EXTRACTION_PROMPT.format(document_content=document_content)
        result = analyze_with_llm(document_content, MODEL_FOR_ARCH_ANALYSIS, system_prompt)
        
        # Extract the JSON from the response
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            try:
                relationships = json.loads(json_match.group(0))
                # Ensure consistent format with source, target, and type
                cleaned_relationships = []
                for rel in relationships:
                    if isinstance(rel, dict) and "source" in rel and "target" in rel:
                        cleaned_rel = {
                            "source": rel["source"],
                            "target": rel["target"],
                            "type": rel.get("type", "depends_on")
                        }
                        # Add description if present
                        if "description" in rel:
                            cleaned_rel["description"] = rel["description"]
                        cleaned_relationships.append(cleaned_rel)
                return cleaned_relationships
            except json.JSONDecodeError:
                # Fall back to regex extraction
                return _extract_relationships_regex(document_content)
        else:
            # Fall back to regex extraction
            return _extract_relationships_regex(document_content)
    except Exception as e:
        print(f"Error extracting relationships with prompt model: {e}")
        # Fall back to regex extraction
        return _extract_relationships_regex(document_content)


def get_mermaid_from_components(components: List[Dict[str, Any]], relationships: List[Dict[str, str]]) -> str:
    """
    Generate a Mermaid diagram from components and relationships.

    Args:
        components: List of component dictionaries
        relationships: List of relationship dictionaries

    Returns:
        Mermaid diagram code
    """
    if not components:
        return "graph TD\n    A[No components found] --> B[Add components to your architecture document]"

    # Start with the graph declaration
    mermaid_code = "graph TD\n"
    
    # Add component nodes
    for component in components:
        name = component["name"]
        desc = component.get("description", "")
        # Create a sanitized ID for the node
        node_id = re.sub(r'[^a-zA-Z0-9]', '', name)
        # Add the node with a description tooltip
        mermaid_code += f"    {node_id}[\"{name}\"] -- \"{desc}\" --- {node_id}\n"
    
    # Add relationships
    for relationship in relationships:
        source = relationship["source"]
        target = relationship["target"]
        rel_type = relationship.get("type", "depends_on").replace("_", " ")
        
        # Create sanitized IDs
        source_id = re.sub(r'[^a-zA-Z0-9]', '', source)
        target_id = re.sub(r'[^a-zA-Z0-9]', '', target)
        
        # Add the relationship
        mermaid_code += f"    {source_id} -- \"{rel_type}\" --> {target_id}\n"
    
    return mermaid_code


def _extract_components_regex(document_content: str) -> List[Dict[str, Any]]:
    """
    Fallback method to extract components using regex.

    Args:
        document_content: Content of the architecture document

    Returns:
        List of component dictionaries with name and description
    """
    components = []
    
    # Look for components section
    components_section_match = re.search(
        r'(?:##?\s*Components|##?\s*Modules)(.*?)(?:##|$)', 
        document_content, 
        re.DOTALL
    )
    
    if components_section_match:
        components_section = components_section_match.group(1)
        
        # Look for list items (- Component: Description format)
        component_matches = re.findall(
            r'-\s*([\w\s]+)[:\-–—]\s*(.*?)(?=\n-|\n\n|$)', 
            components_section, 
            re.DOTALL
        )
        
        for name, desc in component_matches:
            components.append({
                "name": name.strip(),
                "description": desc.strip()
            })
        
        # If no components found with the above pattern, try another pattern
        if not components:
            component_matches = re.findall(
                r'-\s*([\w\s]+)\s*(?:\n\s+(.+?))?(?=\n-|\n\n|$)', 
                components_section, 
                re.DOTALL
            )
            
            for name, desc in component_matches:
                components.append({
                    "name": name.strip(),
                    "description": desc.strip() if desc else ""
                })
    
    # Look for component headings if we still have no components
    if not components:
        component_headings = re.findall(
            r'###\s*([\w\s]+)[:\r\n](.*?)(?=###|$)', 
            document_content, 
            re.DOTALL
        )
        
        for name, desc in component_headings:
            if 'Component' in name or 'Module' in name:
                continue  # Skip section headings
            components.append({
                "name": name.strip(),
                "description": desc.strip()
            })
    
    return components


def _extract_relationships_regex(document_content: str) -> List[Dict[str, str]]:
    """
    Fallback method to extract relationships using regex.

    Args:
        document_content: Content of the architecture document

    Returns:
        List of relationship dictionaries with source, target, and type
    """
    relationships = []
    
    # Look for relationships section
    relationships_section_match = re.search(
        r'(?:##?\s*Relationships|##?\s*Dependencies)(.*?)(?:##|$)', 
        document_content, 
        re.DOTALL
    )
    
    if relationships_section_match:
        relationships_section = relationships_section_match.group(1)
        
        # Look for list items with relationships (- Component1 relates to Component2)
        relationship_patterns = [
            r'-\s*([\w\s]+)\s*(communicates with|depends on|uses|includes|implements|extends|contains)\s*([\w\s]+)',
            r'-\s*([\w\s]+)\s*(?:-+>|→|-->)\s*([\w\s]+)'
        ]
        
        for pattern in relationship_patterns:
            matches = re.findall(pattern, relationships_section, re.IGNORECASE)
            
            for match in matches:
                if len(match) == 3:  # Pattern with explicit relationship type
                    source, rel_type, target = match
                    relationships.append({
                        "source": source.strip(),
                        "target": target.strip(),
                        "type": rel_type.strip().replace(" ", "_")
                    })
                elif len(match) == 2:  # Pattern with arrow but no explicit type
                    source, target = match
                    relationships.append({
                        "source": source.strip(),
                        "target": target.strip(),
                        "type": "depends_on"
                    })
    
    # If no relationships found with the above patterns, try a more generic approach
    if not relationships:
        # Look for sentences describing relationships
        relationship_terms = [
            r'([\w\s]+)\s+(communicates with|depends on|uses|includes|implements|extends|contains)\s+([\w\s]+)',
            r'([\w\s]+)\s+(?:is connected to|interacts with)\s+([\w\s]+)'
        ]
        
        for pattern in relationship_terms:
            matches = re.findall(pattern, document_content, re.IGNORECASE)
            
            for match in matches:
                if len(match) == 3:
                    source, rel_type, target = match
                    relationships.append({
                        "source": source.strip(),
                        "target": target.strip(),
                        "type": rel_type.strip().replace(" ", "_")
                    })
                elif len(match) == 2:
                    source, target = match
                    relationships.append({
                        "source": source.strip(),
                        "target": target.strip(),
                        "type": "interacts_with"
                    })
    
    return relationships
