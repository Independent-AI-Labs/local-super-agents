"""
Diagram utilities for VibeCheck.

This module provides utilities for generating and manipulating architectural diagrams.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from vibecheck.integrations.llm import generate_diagrams_from_architecture


def extract_components_from_architecture(architecture_doc: str) -> List[Dict[str, str]]:
    """
    Extract component information from an architectural document.

    Args:
        architecture_doc (str): The architectural document

    Returns:
        List[Dict[str, str]]: A list of components with their names and descriptions
    """
    components = []
    
    # Look for component definitions using several patterns
    
    # Pattern 1: Headers with "Component" in them
    component_headers = re.finditer(r'#+\s*(.*?Component.*?)\s*\n', architecture_doc)
    for match in component_headers:
        header = match.group(1)
        start_pos = match.end()
        
        # Find the next header or the end of the document
        next_header = re.search(r'#+\s*', architecture_doc[start_pos:])
        end_pos = start_pos + next_header.start() if next_header else len(architecture_doc)
        
        # Extract the description
        description = architecture_doc[start_pos:end_pos].strip()
        
        # Extract the component name
        component_name = re.sub(r'Component\s*:?\s*', '', header).strip()
        if not component_name:
            continue
        
        components.append({
            'name': component_name,
            'description': description
        })
    
    # Pattern 2: Lists or tables of components
    component_lists = re.finditer(r'(?:Component|Module)s?:?\s*\n(?:\s*[-*]\s*(.*?)(?:\s*[:–-]\s*(.*?))?\s*\n)+', architecture_doc, re.IGNORECASE)
    for match in component_lists:
        list_text = match.group(0)
        
        # Extract each list item
        for item_match in re.finditer(r'\s*[-*]\s*(.*?)(?:\s*[:–-]\s*(.*?))?\s*\n', list_text):
            component_name = item_match.group(1).strip()
            description = item_match.group(2).strip() if item_match.group(2) else ""
            
            # Skip if we already found this component
            if any(c['name'] == component_name for c in components):
                continue
            
            components.append({
                'name': component_name,
                'description': description
            })
    
    # Pattern 3: Architectural sections with component names
    sections = re.finditer(r'#+\s*(.+?)\s*\n(.*?)(?=\n#+\s|$)', architecture_doc, re.DOTALL)
    for match in sections:
        section_name = match.group(1)
        section_content = match.group(2)
        
        # Skip if this looks like a component we already found
        if any(c['name'] == section_name for c in components):
            continue
        
        # Check if this section might be a component
        component_indicators = ['module', 'component', 'service', 'manager', 'controller', 'provider', 'handler']
        if any(indicator in section_name.lower() for indicator in component_indicators):
            components.append({
                'name': section_name,
                'description': section_content.strip()
            })
    
    return components


def extract_relationships_from_architecture(architecture_doc: str) -> List[Dict[str, str]]:
    """
    Extract relationship information from an architectural document.

    Args:
        architecture_doc (str): The architectural document

    Returns:
        List[Dict[str, str]]: A list of relationships with source, target, and type
    """
    relationships = []
    
    # Find component names first to identify potential relationships
    components = extract_components_from_architecture(architecture_doc)
    component_names = [c['name'] for c in components]
    
    # Pattern 1: Direct mentions of relationships
    for comp1 in component_names:
        for comp2 in component_names:
            if comp1 == comp2:
                continue
            
            # Check for different relationship patterns
            patterns = [
                # A interacts with B
                rf'{re.escape(comp1)}\s+(?:interacts|communicates|connects|talks|interfaces)\s+with\s+{re.escape(comp2)}',
                # A depends on B
                rf'{re.escape(comp1)}\s+(?:depends|relies)\s+on\s+{re.escape(comp2)}',
                # A calls B
                rf'{re.escape(comp1)}\s+(?:calls|invokes|uses|initiates)\s+{re.escape(comp2)}',
                # A sends data to B
                rf'{re.escape(comp1)}\s+(?:sends|provides|writes|pushes|outputs)\s+(?:data|information|events|messages)?\s+(?:to|for)\s+{re.escape(comp2)}',
                # A receives data from B
                rf'{re.escape(comp1)}\s+(?:receives|gets|reads|consumes)\s+(?:data|information|events|messages)?\s+from\s+{re.escape(comp2)}',
            ]
            
            for pattern in patterns:
                if re.search(pattern, architecture_doc, re.IGNORECASE):
                    relationship_type = 'interacts_with'
                    if 'depends' in pattern or 'relies' in pattern:
                        relationship_type = 'depends_on'
                    elif 'calls' in pattern or 'uses' in pattern:
                        relationship_type = 'calls'
                    elif 'sends' in pattern or 'provides' in pattern:
                        relationship_type = 'sends_data_to'
                    elif 'receives' in pattern or 'gets' in pattern:
                        relationship_type = 'receives_data_from'
                    
                    relationships.append({
                        'source': comp1,
                        'target': comp2,
                        'type': relationship_type
                    })
    
    # Pattern 2: Diagrams in text format
    diagram_patterns = [
        # ASCII arrows
        r'(\w+)\s*-+>\s*(\w+)',
        r'(\w+)\s*<-+\s*(\w+)',
        r'(\w+)\s*--+\s*(\w+)',
        # Text diagrams with relationship labels
        r'(\w+)\s*-\s*\[(.*?)\]\s*->\s*(\w+)',
    ]
    
    for pattern in diagram_patterns:
        matches = re.finditer(pattern, architecture_doc)
        for match in matches:
            if len(match.groups()) == 2:
                source, target = match.groups()
                rel_type = 'connected_to'
                
                # Correctly orient "from" and "to" based on arrow direction
                if '<-' in match.group(0):
                    source, target = target, source
            else:
                source, rel_type, target = match.groups()
                rel_type = rel_type.strip().replace(' ', '_').lower()
            
            # Ensure the components exist
            if source in component_names and target in component_names:
                relationships.append({
                    'source': source,
                    'target': target,
                    'type': rel_type
                })
    
    # Deduplicate relationships
    unique_relationships = []
    for rel in relationships:
        if not any(
            r['source'] == rel['source'] and 
            r['target'] == rel['target'] and 
            r['type'] == rel['type'] 
            for r in unique_relationships
        ):
            unique_relationships.append(rel)
    
    return unique_relationships


def create_module_diagram_svg(components: List[Dict[str, str]], relationships: List[Dict[str, str]]) -> str:
    """
    Create a module diagram in SVG format.

    Args:
        components (List[Dict[str, str]]): List of components
        relationships (List[Dict[str, str]]): List of relationships

    Returns:
        str: SVG diagram
    """
    # In a real implementation, we would generate a proper SVG diagram
    # For now, we'll use a simple placeholder implementation
    
    # Calculate diagram dimensions
    width = 800
    height = max(400, len(components) * 80)
    
    # Start SVG
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#000" />
        </marker>
    </defs>
    <rect width="{width}" height="{height}" fill="#f8f8f8" />
    <text x="10" y="20" font-family="Arial" font-size="16" font-weight="bold">Module Diagram</text>
    """
    
    # Calculate component positions in a circular layout
    center_x = width / 2
    center_y = height / 2
    radius = min(center_x, center_y) * 0.7
    
    # Create component shapes
    component_positions = {}
    for i, component in enumerate(components):
        angle = (i / len(components)) * 2 * 3.14159
        x = center_x + radius * max(0, min(1, 0.99 * (0.5 - 0.5 * len(components)))) * (0.5 - i / len(components)) * (0.5 - i / len(components)) * (-1 if i % 2 == 0 else 1) + radius * 0.85 * (-1 if len(components) > 7 and i % 2 == 0 else 1) * (i % 3 - 1) / 3
        y = center_y + radius * max(0, min(1, 0.99 * (0.5 - 0.5 * len(components)))) * (0.5 - i / len(components)) * (0.5 - i / len(components)) * (-1 if i % 2 == 1 else 1) + radius * 0.85 * (-1 if len(components) > 7 and i % 2 == 1 else 1) * (i % 3 - 1) / 3
        
        name = component['name']
        component_positions[name] = (x, y)
        
        # Draw component box
        box_width = 120
        box_height = 60
        
        svg += f"""
        <rect x="{x - box_width/2}" y="{y - box_height/2}" width="{box_width}" height="{box_height}" 
              rx="5" ry="5" fill="#e0e8ff" stroke="#0066cc" stroke-width="2" />
        <text x="{x}" y="{y}" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle">{name}</text>
        """
    
    # Create relationship arrows
    for relationship in relationships:
        source = relationship['source']
        target = relationship['target']
        rel_type = relationship['type']
        
        if source in component_positions and target in component_positions:
            # Get positions
            x1, y1 = component_positions[source]
            x2, y2 = component_positions[target]
            
            # Calculate arrow points
            angle = 0
            try:
                angle = (180 / 3.14159) * (0.5 * 3.14159 + (1 if y2 - y1 > 0 else -1) * abs(((y2 - y1) / ((x2 - x1) or 0.001))))
            except:
                pass
            
            # Draw the arrow
            svg += f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="1.5" 
                  marker-end="url(#arrow)" stroke-dasharray="{5 if 'depends' in rel_type else ''}" />
            """
            
            # Add relationship label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label = rel_type.replace('_', ' ')
            
            svg += f"""
            <rect x="{mid_x - 40}" y="{mid_y - 10}" width="80" height="20" rx="5" ry="5" 
                  fill="white" fill-opacity="0.8" />
            <text x="{mid_x}" y="{mid_y + 5}" font-family="Arial" font-size="10" 
                  text-anchor="middle">{label}</text>
            """
    
    # Close SVG
    svg += "</svg>"
    
    return svg


def create_dataflow_diagram_svg(components: List[Dict[str, str]], relationships: List[Dict[str, str]]) -> str:
    """
    Create a dataflow diagram in SVG format.

    Args:
        components (List[Dict[str, str]]): List of components
        relationships (List[Dict[str, str]]): List of relationships

    Returns:
        str: SVG diagram
    """
    # Filter relationships to only include data flow
    data_flow_relationships = [
        r for r in relationships 
        if 'data' in r['type'] or 'sends' in r['type'] or 'receives' in r['type']
    ]
    
    # If no data flow relationships were found, include all relationships
    if not data_flow_relationships:
        data_flow_relationships = relationships
    
    # Calculate diagram dimensions
    width = 800
    height = max(400, len(components) * 80)
    
    # Start SVG
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#000" />
        </marker>
    </defs>
    <rect width="{width}" height="{height}" fill="#f8f8f8" />
    <text x="10" y="20" font-family="Arial" font-size="16" font-weight="bold">Data Flow Diagram</text>
    """
    
    # Layout components in a grid
    cols = max(1, min(4, round(len(components) ** 0.5)))
    rows = max(1, (len(components) + cols - 1) // cols)
    
    cell_width = width / (cols + 1)
    cell_height = height / (rows + 1)
    
    # Create component shapes
    component_positions = {}
    for i, component in enumerate(components):
        row = i // cols
        col = i % cols
        
        x = cell_width * (col + 1)
        y = cell_height * (row + 1)
        
        name = component['name']
        component_positions[name] = (x, y)
        
        # Draw component as ellipse for data flow diagram
        ellipse_width = 140
        ellipse_height = 70
        
        svg += f"""
        <ellipse cx="{x}" cy="{y}" rx="{ellipse_width/2}" ry="{ellipse_height/2}" 
                 fill="#e8f4e8" stroke="#006600" stroke-width="2" />
        <text x="{x}" y="{y}" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle">{name}</text>
        """
    
    # Create data flow arrows
    for relationship in data_flow_relationships:
        source = relationship['source']
        target = relationship['target']
        rel_type = relationship['type']
        
        if source in component_positions and target in component_positions:
            # Get positions
            x1, y1 = component_positions[source]
            x2, y2 = component_positions[target]
            
            # Calculate control points for curved arrows
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Add some curvature
            curve_strength = 30
            dx = x2 - x1
            dy = y2 - y1
            curve_x = mid_x + curve_strength * (-dy / ((dx**2 + dy**2) ** 0.5))
            curve_y = mid_y + curve_strength * (dx / ((dx**2 + dy**2) ** 0.5))
            
            # Draw the curved arrow
            svg += f"""
            <path d="M{x1},{y1} Q{curve_x},{curve_y} {x2},{y2}" fill="none" stroke="#060" stroke-width="2" 
                  marker-end="url(#arrow)" />
            """
            
            # Add data label
            label = rel_type.replace('_', ' ')
            
            svg += f"""
            <rect x="{curve_x - 45}" y="{curve_y - 10}" width="90" height="20" rx="5" ry="5" 
                  fill="white" fill-opacity="0.8" />
            <text x="{curve_x}" y="{curve_y + 5}" font-family="Arial" font-size="10" 
                  text-anchor="middle">{label}</text>
            """
    
    # Close SVG
    svg += "</svg>"
    
    return svg


def create_security_diagram_svg(components: List[Dict[str, str]], relationships: List[Dict[str, str]]) -> str:
    """
    Create a security diagram in SVG format.

    Args:
        components (List[Dict[str, str]]): List of components
        relationships (List[Dict[str, str]]): List of relationships

    Returns:
        str: SVG diagram
    """
    # Calculate diagram dimensions
    width = 800
    height = 600
    
    # Start SVG
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#000" />
        </marker>
    </defs>
    <rect width="{width}" height="{height}" fill="#f8f8f8" />
    <text x="10" y="20" font-family="Arial" font-size="16" font-weight="bold">Security Diagram</text>
    """
    
    # Simulate security zones
    # In a real implementation, we would analyze the architecture to determine security zones
    zones = [
        {"name": "External Zone", "color": "#fff0f0", "border": "#ff6666", "y": 100, "height": 120},
        {"name": "DMZ", "color": "#ffffd0", "border": "#cc9900", "y": 250, "height": 120},
        {"name": "Internal Zone", "color": "#f0fff0", "border": "#66cc66", "y": 400, "height": 120}
    ]
    
    # Draw security zones
    for zone in zones:
        svg += f"""
        <rect x="50" y="{zone['y']}" width="{width - 100}" height="{zone['height']}" 
              fill="{zone['color']}" stroke="{zone['border']}" stroke-width="2" stroke-dasharray="5,2" />
        <text x="{width - 80}" y="{zone['y'] + 20}" font-family="Arial" font-size="14" 
              font-weight="bold" text-anchor="end">{zone['name']}</text>
        """
    
    # Assign components to zones
    # In a real implementation, we would analyze the architecture to assign components to zones
    component_zones = {}
    for i, component in enumerate(components):
        # Simple assignment algorithm - distribute components across zones
        zone_index = min(i % len(zones), len(zones) - 1)
        component_zones[component['name']] = zones[zone_index]
    
    # Calculate component positions within zones
    component_positions = {}
    zone_component_counts = {}
    
    for component, zone in component_zones.items():
        if zone['name'] not in zone_component_counts:
            zone_component_counts[zone['name']] = 0
        
        zone_component_counts[zone['name']] += 1
    
    zone_component_indices = {}
    for zone in zones:
        zone_component_indices[zone['name']] = 0
    
    for component_name, zone in component_zones.items():
        count = zone_component_counts[zone['name']]
        index = zone_component_indices[zone['name']]
        
        # Calculate position within zone
        x = 100 + (width - 200) * (index + 1) / (count + 1)
        y = zone['y'] + zone['height'] / 2
        
        component_positions[component_name] = (x, y)
        zone_component_indices[zone['name']] += 1
    
    # Draw components
    for component_name, (x, y) in component_positions.items():
        # Draw component box
        box_width = 120
        box_height = 60
        
        svg += f"""
        <rect x="{x - box_width/2}" y="{y - box_height/2}" width="{box_width}" height="{box_height}" 
              rx="5" ry="5" fill="#ffffff" stroke="#333333" stroke-width="2" />
        <text x="{x}" y="{y}" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle">{component_name}</text>
        """
    
    # Draw relationships, highlighting those that cross security zones
    for relationship in relationships:
        source = relationship['source']
        target = relationship['target']
        
        if source in component_positions and target in component_positions:
            source_pos = component_positions[source]
            target_pos = component_positions[target]
            
            # Determine if the relationship crosses security zones
            source_zone = component_zones.get(source, {}).get('name', '')
            target_zone = component_zones.get(target, {}).get('name', '')
            
            crosses_zones = source_zone != target_zone
            
            # Set arrow style based on whether it crosses zones
            stroke_color = "#ff0000" if crosses_zones else "#333333"
            stroke_width = "2" if crosses_zones else "1.5"
            
            # Draw the arrow
            svg += f"""
            <line x1="{source_pos[0]}" y1="{source_pos[1]}" x2="{target_pos[0]}" y2="{target_pos[1]}" 
                  stroke="{stroke_color}" stroke-width="{stroke_width}" marker-end="url(#arrow)" />
            """
            
            # Add security warning for cross-zone relationships
            if crosses_zones:
                mid_x = (source_pos[0] + target_pos[0]) / 2
                mid_y = (source_pos[1] + target_pos[1]) / 2
                
                svg += f"""
                <circle cx="{mid_x}" cy="{mid_y}" r="10" fill="#ffcccc" stroke="#ff0000" stroke-width="1" />
                <text x="{mid_x}" y="{mid_y + 4}" font-family="Arial" font-size="12" font-weight="bold" 
                      text-anchor="middle" dominant-baseline="middle">!</text>
                <text x="{mid_x + 15}" y="{mid_y - 10}" font-family="Arial" font-size="10" 
                      text-anchor="start" fill="#cc0000">Zone crossing</text>
                """
    
    # Close SVG
    svg += "</svg>"
    
    return svg


def generate_diagrams(architecture_doc: str) -> Dict[str, str]:
    """
    Generate all types of diagrams from an architecture document.

    Args:
        architecture_doc (str): The architecture document

    Returns:
        Dict[str, str]: A dictionary of diagram types to SVG content
    """
    try:
        # First, try to use LLM to generate diagrams
        diagrams = generate_diagrams_from_architecture(architecture_doc)
        
        # Check if we have valid SVG diagrams
        valid_diagrams = True
        for diagram_type, svg_content in diagrams.items():
            if not svg_content or not svg_content.startswith('<svg') or not svg_content.endswith('</svg>'):
                valid_diagrams = False
                break
        
        if valid_diagrams:
            return diagrams
    except Exception as e:
        print(f"Error generating diagrams with LLM: {e}")
    
    # Fallback to manual diagram generation
    components = extract_components_from_architecture(architecture_doc)
    relationships = extract_relationships_from_architecture(architecture_doc)
    
    module_diagram = create_module_diagram_svg(components, relationships)
    dataflow_diagram = create_dataflow_diagram_svg(components, relationships)
    security_diagram = create_security_diagram_svg(components, relationships)
    
    return {
        "module": module_diagram,
        "dataflow": dataflow_diagram,
        "security": security_diagram
    }


def svg_to_png(svg_content: str, output_path: str, width: int = 800, height: int = 600) -> bool:
    """
    Convert an SVG to a PNG file.

    Args:
        svg_content (str): The SVG content
        output_path (str): The output PNG file path
        width (int, optional): The PNG width. Defaults to 800.
        height (int, optional): The PNG height. Defaults to 600.

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Check if cairosvg is available
        import cairosvg
        
        # Convert SVG to PNG
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            output_width=width,
            output_height=height
        )
        
        return True
    except ImportError:
        print("cairosvg is not installed. PNG conversion not available.")
        return False
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return False


def get_mermaid_from_components(components: List[Dict[str, str]], relationships: List[Dict[str, str]]) -> str:
    """
    Generate a Mermaid diagram from components and relationships.

    Args:
        components (List[Dict[str, str]]): List of components
        relationships (List[Dict[str, str]]): List of relationships

    Returns:
        str: Mermaid diagram code
    """
    mermaid = "graph TD\n"
    
    # Add components
    for i, component in enumerate(components):
        name = component['name']
        # Replace spaces with underscores for Mermaid IDs
        id_name = name.replace(' ', '_')
        mermaid += f"    {id_name}[{name}]\n"
    
    # Add relationships
    for relationship in relationships:
        source = relationship['source'].replace(' ', '_')
        target = relationship['target'].replace(' ', '_')
        rel_type = relationship['type']
        
        # Map relationship type to Mermaid arrow style
        arrow_style = "-->"
        if "depends" in rel_type:
            arrow_style = "-.->|depends on|"
        elif "calls" in rel_type:
            arrow_style = "-->|calls|"
        elif "sends" in rel_type:
            arrow_style = "-->|sends data|"
        elif "receives" in rel_type:
            # Reverse the direction for "receives"
            source, target = target, source
            arrow_style = "-->|sends data|"
        
        mermaid += f"    {source} {arrow_style} {target}\n"
    
    return mermaid
