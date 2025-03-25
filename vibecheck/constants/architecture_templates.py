"""
HTML templates for architecture module in VibeCheck.

This module contains HTML templates used for rendering diagrams,
error messages, and other UI elements in the architecture tab.
"""

# ===== Mermaid Diagram HTML Template =====
MERMAID_HTML_TEMPLATE = """
<div class="mermaid-container" style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: white; overflow: auto;">
  <div class="mermaid-diagram" style="display: flex; justify-content: center;">
    <div class="mermaid" style="font-family: 'Courier New', Courier, monospace;">
{diagram_code}
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@9.4.3/dist/mermaid.min.js" onload="initMermaid()"></script>
  <script>
    function initMermaid() {{
      if (typeof window.mermaidInitialized === 'undefined') {{
        window.mermaidInitialized = true;
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'default',
          securityLevel: 'loose',
          flowchart: {{ htmlLabels: true, curve: 'basis' }},
          er: {{ layoutDirection: 'TB', entityPadding: 15 }},
          sequence: {{ actorMargin: 50, messageMargin: 25 }}
        }});
      }}

      // Force render on this specific diagram
      try {{
        mermaid.init(undefined, document.querySelectorAll('.mermaid:not(.mermaid-processed)'));
        document.querySelectorAll('.mermaid').forEach(el => {{
          el.classList.add('mermaid-processed');
        }});
      }} catch (e) {{
        console.error('Mermaid rendering error:', e);
      }}
    }}

    // Try to render right away
    setTimeout(function() {{
      if (typeof mermaid !== 'undefined') {{
        initMermaid();
      }}
    }}, 100);

    // Also try on window load
    window.addEventListener('load', function() {{
      setTimeout(function() {{
        if (typeof mermaid !== 'undefined') {{
          initMermaid();
        }}
      }}, 200);
    }});
  </script>
</div>
"""

# ===== Default Document Content =====
DEFAULT_DOCUMENT_TEMPLATE = """# {doc_name}

## System Overview

Describe your system here...

## Components

- Component 1: Description of component 1
- Component 2: Description of component 2

## Relationships

- Component 1 communicates with Component 2
"""

# Template for Mermaid diagram with direct script injection
MERMAID_DIRECT_TEMPLATE = """
<div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: white; margin: 10px 0;">
  <div id="mermaid-diagram-{diagram_id}" style="text-align: center;">
    <pre class="mermaid" style="display: none;">
{diagram_code}
    </pre>
    <div id="mermaid-render-{diagram_id}"></div>
  </div>

  <script>
    (function() {
      // Define the render function
      function renderMermaidDiagram() {
        try {
          // Check if mermaid is loaded
          if (typeof mermaid === 'undefined') {
            // Load mermaid script if not already loaded
            var script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/mermaid@9.4.3/dist/mermaid.min.js';
            script.onload = function() {
              initMermaid();
            };
            document.head.appendChild(script);
          } else {
            initMermaid();
          }
        } catch (e) {
          console.error('Error rendering mermaid diagram:', e);
          document.getElementById('mermaid-render-{diagram_id}').innerHTML = 
            '<div style="color: red; padding: 10px; border: 1px solid #ffcccc; background-color: #ffeeee; border-radius: 5px;">' +
            '<p><strong>Error rendering diagram:</strong> ' + e.message + '</p>' +
            '<p>Raw diagram code:</p>' +
            '<pre style="background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto;">' + 
            document.querySelector('#mermaid-diagram-{diagram_id} pre.mermaid').textContent + '</pre></div>';
        }
      }

      // Initialize mermaid and render the diagram
      function initMermaid() {
        mermaid.initialize({{
          startOnLoad: false,
          theme: 'default',
          securityLevel: 'loose',
          flowchart: {{ htmlLabels: true, curve: 'basis' }},
          themeVariables: {{ primaryColor: '#007bff' }}
        }});
        
        // Get the diagram code
        var diagramCode = document.querySelector('#mermaid-diagram-{diagram_id} pre.mermaid').textContent.trim();
        
        // Render the diagram
        try {{
          var insertSvg = function(svgCode) {{
            document.getElementById('mermaid-render-{diagram_id}').innerHTML = svgCode;
          }};
          
          mermaid.render('mermaid-svg-{diagram_id}', diagramCode, insertSvg);
        }} catch (e) {{
          console.error('Mermaid rendering error:', e);
          document.getElementById('mermaid-render-{diagram_id}').innerHTML = 
            '<div style="color: red; padding: 10px; border: 1px solid #ffcccc; background-color: #ffeeee; border-radius: 5px;">' +
            '<p><strong>Diagram syntax error:</strong> ' + e.message + '</p>' +
            '<p>Raw diagram code:</p>' +
            '<pre style="background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto;">' + diagramCode + '</pre></div>';
        }}
      }}
      
      // Call the render function
      if (document.readyState === 'complete' || document.readyState === 'interactive') {{
        setTimeout(renderMermaidDiagram, 100);
      }} else {{
        document.addEventListener('DOMContentLoaded', renderMermaidDiagram);
      }}
      
      // Also try on window load and after a short delay
      window.addEventListener('load', function() {{
        setTimeout(renderMermaidDiagram, 500);
      }});
    }})();
  </script>
</div>
"""

# Error template for diagram generation failure
DIAGRAM_ERROR_TEMPLATE = """
<div style="padding: 15px; border: 1px solid #ffcccc; border-radius: 8px; background-color: #ffeeee; margin: 10px 0;">
  <h3 style="color: #cc0000; margin-top: 0;">Error Generating {diagram_type} Diagram</h3>
  <p><strong>Error:</strong> {error_message}</p>
  <p>Please try regenerating the diagram or check your document for potential issues.</p>
</div>
"""

# Fallback template for when Mermaid rendering fails
MERMAID_FALLBACK_TEMPLATE = """
<div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: white; margin: 10px 0;">
  <h3 style="margin-top: 0;">Mermaid Diagram ({diagram_type})</h3>
  <p><em>Diagram could not be rendered. Below is the raw Mermaid code:</em></p>
  <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto;">{diagram_code}</pre>
  <a href="https://mermaid.live/edit#pako:{encoded_diagram}" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; margin-top: 10px;">
    Open in Mermaid Live Editor
  </a>
</div>
"""

# No diagram selected placeholder
NO_DIAGRAM_SELECTED_TEMPLATE = """
<div style="padding: 20px; border: 1px dashed #ccc; border-radius: 8px; background-color: #f9f9f9; margin: 10px 0; text-align: center;">
  <p style="font-size: 16px; color: #666;">Select a document and diagram type to view.</p>
  <p style="font-size: 14px; color: #999;">If no diagrams are available, click "Generate Diagrams".</p>
</div>
"""

# No document selected placeholder
NO_DOCUMENT_FIRST_TEMPLATE = """
<div style="padding: 20px; border: 1px dashed #ccc; border-radius: 8px; background-color: #f9f9f9; margin: 10px 0; text-align: center;">
  <p style="font-size: 16px; color: #666;">Please select a document first.</p>
</div>
"""

# Generation in progress template
GENERATING_DIAGRAMS_TEMPLATE = """
<div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f5f5f5; margin: 10px 0; text-align: center;">
  <div style="display: inline-block; width: 40px; height: 40px; border: 4px solid #dddddd; border-top: 4px solid #3498db; border-radius: 50%; animation: spin 2s linear infinite; margin-bottom: 10px;"></div>
  <p style="font-size: 16px; color: #666;">Generating diagrams...</p>
  <p style="font-size: 14px; color: #999;">This may take a moment.</p>
  <style>
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</div>
"""
