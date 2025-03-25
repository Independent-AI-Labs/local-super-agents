"""
Mermaid utility functions for diagram rendering in VibeCheck.

This module provides utility functions for converting mermaid diagrams to SVG
using the mermaid-py library.
"""

from mermaid import Graph, Mermaid

"""
Local Mermaid to SVG renderer using pyppeteer.

This module provides utility functions to render Mermaid diagrams as SVG locally
without relying on external services.
"""

import asyncio
import time
from typing import Optional

import pyppeteer


class LocalMermaidRenderer:
    """
    Renderer for converting Mermaid code to SVG using pyppeteer.
    """

    _browser = None
    _page = None
    _initialized = False

    @classmethod
    async def initialize(cls):
        """Initialize the browser if not already initialized."""
        if not cls._initialized:
            print("[DEBUG] Initializing local Mermaid renderer")
            cls._browser = await pyppeteer.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            cls._page = await cls._browser.newPage()
            await cls._page.setViewport({'width': 1200, 'height': 1200})
            await cls._page.setContent(HTML_TEMPLATE)
            await cls._page.addScriptTag({
                'url': 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js'
            })
            await cls._page.evaluate('''
                () => {
                    window.mermaid.initialize({
                        startOnLoad: false,
                        securityLevel: 'loose',
                        theme: 'default'
                    });
                }
            ''')
            cls._initialized = True
            print("[DEBUG] Local Mermaid renderer initialized")

    @classmethod
    async def close(cls):
        """Close the browser if it's open."""
        if cls._browser:
            await cls._browser.close()
            cls._browser = None
            cls._page = None
            cls._initialized = False
            print("[DEBUG] Local Mermaid renderer closed")

    @classmethod
    async def render_to_svg(cls, mermaid_code: str) -> Optional[str]:
        """
        Render Mermaid code to SVG.

        Args:
            mermaid_code: Mermaid diagram code to render

        Returns:
            SVG content as string, or None if rendering failed
        """
        start_time = time.time()
        print(f"[DEBUG] Rendering Mermaid to SVG locally")

        try:
            # Initialize if needed
            await cls.initialize()

            # Sanitize the Mermaid code
            mermaid_code = mermaid_code.replace('\\', '\\\\').replace('`', '\\`').replace('\n', '\\n')

            # Render the diagram
            svg_content = await cls._page.evaluate(f'''
                async () => {{
                    try {{
                        const container = document.getElementById('container');
                        container.innerHTML = '';

                        // Create diagram div
                        const div = document.createElement('div');
                        div.className = 'mermaid';
                        div.textContent = `{mermaid_code}`;
                        container.appendChild(div);

                        // Render the diagram
                        await window.mermaid.init(undefined, div);

                        // Get SVG content
                        const svg = container.querySelector('svg');
                        if (!svg) {{
                            throw new Error('No SVG element found after rendering');
                        }}

                        // Ensure SVG has width and height
                        if (!svg.hasAttribute('width')) {{
                            svg.setAttribute('width', '100%');
                        }}
                        if (!svg.hasAttribute('height')) {{
                            svg.setAttribute('height', '100%');
                        }}

                        // Add viewport if missing
                        if (!svg.hasAttribute('viewBox')) {{
                            const bbox = svg.getBBox();
                            svg.setAttribute('viewBox', `0 0 ${{bbox.width + 100}} ${{bbox.height + 100}}`);
                        }}

                        return svg.outerHTML;
                    }} catch (error) {{
                        return `<svg xmlns="http://www.w3.org/2000/svg" width="500" height="100" viewBox="0 0 500 100">
                            <text x="10" y="30" fill="red">Error rendering diagram: ${{error.message}}</text>
                        </svg>`;
                    }}
                }}
            ''')

            render_time = time.time() - start_time
            print(f"[PROFILE] Local SVG rendering took {render_time:.2f} seconds")

            # Validate SVG content
            if svg_content and svg_content.startswith('<svg'):
                return svg_content
            else:
                print(f"[ERROR] Invalid SVG content generated: {svg_content[:100]}...")
                return None

        except Exception as e:
            print(f"[ERROR] Error rendering Mermaid to SVG locally: {e}")
            render_time = time.time() - start_time
            print(f"[PROFILE] Failed local SVG rendering took {render_time:.2f} seconds")
            return None


# Simple HTML template for rendering Mermaid diagrams
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        #container {
            width: 1000px;
            height: 1000px;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div id="container"></div>
</body>
</html>
"""


# Synchronous wrapper function for easier integration
def render_mermaid_to_svg(mermaid_code: str) -> Optional[str]:
    """
    Render Mermaid code to SVG synchronously.

    Args:
        mermaid_code: Mermaid diagram code to render

    Returns:
        SVG content as string, or None if rendering failed
    """
    start_time = time.time()

    try:
        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Render the diagram
        svg_content = loop.run_until_complete(LocalMermaidRenderer.render_to_svg(mermaid_code))

        # Cleanup - don't close the browser to allow reuse
        total_time = time.time() - start_time
        print(f"[PROFILE] Total synchronous rendering took {total_time:.2f} seconds")

        return svg_content
    except Exception as e:
        print(f"[ERROR] Error in synchronous Mermaid rendering: {e}")
        return None
    finally:
        if 'loop' in locals() and loop.is_running():
            loop.close()


def mermaid_to_svg(mermaid_code: str, width: Optional[int] = 800, height: Optional[int] = 600) -> str:
    """
    Convert Mermaid code to SVG using mermaid-py.
    
    Args:
        mermaid_code: Mermaid diagram code
        width: Width of the SVG
        height: Height of the SVG
        
    Returns:
        SVG content as a string
    """
    try:
        # Clean up mermaid code
        mermaid_code = clean_mermaid_code(mermaid_code)

        # Create a Graph object from the mermaid code
        graph = Graph("architecture_diagram", mermaid_code)

        # Create a Mermaid object with the specified dimensions
        mermaid = Mermaid(graph, width=width, height=height)

        # Get the SVG content
        svg_content = mermaid.svg_response.text

        return svg_content
    except Exception as e:
        # Fallback to provide a simple error SVG
        print(f"Error converting mermaid to SVG: {e}")
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
            <rect width="100%" height="100%" fill="white"/>
            <text x="10" y="20" font-family="Arial" font-size="14" fill="red">
                Error rendering diagram: {str(e)}
            </text>
            <text x="10" y="40" font-family="Arial" font-size="12">
                Raw mermaid code:
            </text>
            <text x="10" y="60" font-family="monospace" font-size="10">
                {mermaid_code[:100]}...
            </text>
        </svg>'''


def clean_mermaid_code(mermaid_code: str) -> str:
    """
    Clean and format mermaid code to ensure it's valid.
    
    Args:
        mermaid_code: Raw mermaid code
        
    Returns:
        Cleaned mermaid code
    """
    # Remove markdown code blocks if present
    if mermaid_code.strip().startswith("```mermaid"):
        lines = mermaid_code.strip().split("\n")
        mermaid_code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Check if the diagram has valid syntax
    if not any(keyword in mermaid_code for keyword in ["graph ", "flowchart ", "sequenceDiagram", "classDiagram", "stateDiagram"]):
        # Add flowchart TD by default if not present
        mermaid_code = "flowchart TD\n" + mermaid_code

    return mermaid_code


def save_mermaid_diagram(mermaid_code: str, output_path: str, width: Optional[int] = 800, height: Optional[int] = 600) -> bool:
    """
    Save a mermaid diagram as an SVG file.
    
    Args:
        mermaid_code: Mermaid diagram code
        output_path: Path to save the SVG file
        width: Width of the SVG
        height: Height of the SVG
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Clean up mermaid code
        mermaid_code = clean_mermaid_code(mermaid_code)

        # Create a Graph object from the mermaid code
        graph = Graph("architecture_diagram", mermaid_code)

        # Create a Mermaid object with the specified dimensions
        mermaid = Mermaid(graph, width=width, height=height)

        # Save the SVG to the specified path
        mermaid.to_svg(output_path)

        return True
    except Exception as e:
        print(f"Error saving mermaid diagram: {e}")
        return False


def render_mermaid_html(mermaid_code: str) -> str:
    """
    Render mermaid code as HTML for direct display in a browser.
    
    Args:
        mermaid_code: Mermaid diagram code
        
    Returns:
        HTML content for rendering the mermaid diagram
    """
    # Clean up mermaid code and ensure it's properly formatted
    mermaid_code = clean_mermaid_code(mermaid_code)

    # Create HTML with the mermaid diagram and CDN script
    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            fontFamily: 'Arial',
            fontSize: 14,
            securityLevel: 'loose'
        }});
    </script>
    """

    return html


# Example usage
if __name__ == "__main__":
    mermaid_code = """
    graph TD
        A[Start] --> B{Is it working?}
        B -->|Yes| C[Great!]
        B -->|No| D[Debug]
        D --> B
    """

    svg_output = render_mermaid_to_svg(mermaid_code)

    if svg_output:
        with open('test_diagram.svg', 'w') as f:
            f.write(svg_output)
        print("SVG file saved as test_diagram.svg")
    else:
        print("Failed to render SVG")

    # Close the browser when done with all rendering
    asyncio.get_event_loop().run_until_complete(LocalMermaidRenderer.close())
