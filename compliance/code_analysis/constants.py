# Constants for Dependency Analyzer

# Exclusion Patterns
DEFAULT_SOURCE_EXCLUDES = [
    r'test_',
    r'_test',
    r'\.pyc$',
    r'__pycache__',
    r'/\.',  # Hidden files/directories
    r'__init__\.py$'  # Exclude __init__.py files
]

# Comprehensive exclusion patterns to filter out non-project code
NON_SOURCE_CODE_EXCLUDES = [
    # Virtual environments and package managers
    r'site-packages',
    r'env/',
    r'venv/',
    r'\.venv/',
    r'conda',
    r'anaconda',
    r'miniconda',
    r'\.eggs',

    # Build and distribution directories
    r'build/',
    r'dist/',
    r'\.egg-info',

    # System and library paths
    r'python\d+\.\d+',  # Python version directories
    r'lib/python\d+\.\d+',
    r'Lib/site-packages',
    r'lib64/python\d+\.\d+',

    # Common system and library exclusions
    r'/usr/lib',
    r'/usr/local/lib',
    r'/opt/',
    r'\.tox/',
    r'\.mypy_cache',
    r'\.pytest_cache'
]

# Built-in Python modules to exclude
BUILTIN_MODULES = [
    'os', 're', 'sys', 'typing', 'collections', 'abc', 'asyncio', 'dataclasses',
    'datetime', 'enum', 'functools', 'io', 'itertools', 'logging', 'math',
    'pathlib', 'random', 'statistics', 'string', 'time', 'warnings',
    'contextlib', 'json', 'pickle', 'traceback', 'urllib', 'http', 'socket'
]

# Messages
DEFAULT_INCLUDE_PATTERN = [r'\.py$']

# Error Messages
ERROR_INVALID_DIRECTORY = "Invalid directory: {}"
ERROR_NO_FILES_FOUND = "No Python files found to analyze."
ERROR_NO_DEPENDENCIES = "No dependencies could be found to visualize."
ERROR_PARSING_FILE = "Error parsing {}: {}"
ERROR_RESOLVING_IMPORT = "Error resolving import {} in {}: {}"
ERROR_PROCESSING_FILE = "Error processing {}: {}"
ERROR_COMPUTING_LAYOUT = "Error computing graph layout: {}"
ERROR_VISUALIZATION = "Error creating visualization: {}"

# Report Templates
REPORT_HEADER = "\n--- Dependency Analysis Report ---"
REPORT_TOTAL_FILES = "Total Files Analyzed: {}"
REPORT_TOTAL_DEPENDENCIES = "Total Dependencies: {}"
REPORT_MOST_DEPENDENT = "\nMost Dependent Files:"
REPORT_MOST_DEPENDENCIES = "\nFiles with Most Dependencies:"
REPORT_FILE_FORMAT = "  - {}: {} {} dependencies"

# Visualization Color Constants
COLOR_DARK_BACKGROUND = "rgb(18, 18, 30)"
COLOR_DARK_PAPER = "rgb(28, 28, 40)"
COLOR_EDGE = "rgba(150, 150, 150, 0.35)"
COLOR_NODE_LINE = "rgba(200, 200, 200, 0.5)"
COLOR_TEXT_LIGHT = "#e0e0e0"
COLOR_PROXIMITY_LABEL = "#999999"

# Bright pastel palette for dark backgrounds
COLORS_BRIGHT_PALETTE = [
    'rgb(130, 204, 221)',  # Bright Blue
    'rgb(129, 222, 135)',  # Bright Green
    'rgb(255, 165, 165)',  # Bright Peach
    'rgb(197, 134, 255)',  # Bright Purple
    'rgb(255, 223, 97)',  # Bright Yellow
    'rgb(117, 230, 218)',  # Bright Teal
    'rgb(255, 166, 158)',  # Bright Salmon
    'rgb(178, 235, 242)',  # Bright Sky
    'rgb(255, 140, 184)',  # Bright Pink
    'rgb(165, 165, 255)',  # Bright Lavender
]

# Visualization layout settings
VIZ_TITLE = ""
VIZ_CONNECTIONS_THRESHOLD = 10
VIZ_MIN_FONT_SIZE = 12
VIZ_MAX_FONT_SIZE = 24
VIZ_FONT_GROWTH_RATE = 2  # Increase font size by 1pt for every X additional connections
VIZ_NODE_MIN_SIZE = 4  # Increased from 7
VIZ_NODE_SIZE_FACTOR = 8  # Increased from 6 for faster growth
VIZ_PROXIMITY_THRESHOLD = 100  # Pixel radius for showing proximity labels
VIZ_EDGE_WIDTH = 32  # Increased from 1.5 to 32
VIZ_NODE_LINE_WIDTH = 1

# JavaScript template for mouse proximity labels
JS_PROXIMITY_TEMPLATE = """
<script>
  // When the plot is fully rendered
  document.addEventListener('DOMContentLoaded', function() {
    var plotDiv = document.getElementById('gd');
    if (!plotDiv) return;  // Exit if plot not found

    // Distance threshold for showing labels (in pixels)
    var proximityThreshold = %d;

    // Flag to check if we're in 3D scene
    var isInScene = false;

    // Store node data for proximity calculations
    var nodeData = %s;

    // Create a container for proximity labels
    var labelContainer = document.createElement('div');
    labelContainer.style.position = 'absolute';
    labelContainer.style.pointerEvents = 'none';
    labelContainer.style.zIndex = 1000;
    plotDiv.appendChild(labelContainer);

    // Track mouse position
    plotDiv.addEventListener('mousemove', function(event) {
      // Only show proximity labels when inside the 3D scene
      var sceneDiv = plotDiv.querySelector('.scene');
      if (!sceneDiv) return;

      var sceneRect = sceneDiv.getBoundingClientRect();
      isInScene = (
        event.clientX >= sceneRect.left && 
        event.clientX <= sceneRect.right && 
        event.clientY >= sceneRect.top && 
        event.clientY <= sceneRect.bottom
      );

      if (!isInScene) {
        // Clear labels when mouse leaves the scene
        labelContainer.innerHTML = '';
        return;
      }

      // Get relative position within the scene
      var mouseX = event.clientX - sceneRect.left;
      var mouseY = event.clientY - sceneRect.top;

      // Clear previous labels
      labelContainer.innerHTML = '';

      // In 3D, we need the camera position to properly calculate proximity
      // For simplicity, we'll use 2D distance on screen
      var camera = plotDiv._fullLayout.scene._scene.glplot.camera;

      // Check each node
      for (var i = 0; i < nodeData.length; i++) {
        var node = nodeData[i];

        // Skip nodes that already have permanent labels (connections > %d)
        if (node.connections > %d) continue;

        // Project 3D point to 2D screen coordinates
        var projected = camera.project([node.x, node.y, node.z]);

        // Convert normalized coordinates to screen coordinates
        var screenX = sceneRect.width * (projected[0] + 1) / 2;
        var screenY = sceneRect.height * (1 - (projected[1] + 1) / 2);

        // Calculate distance from mouse to node
        var distance = Math.sqrt(
          Math.pow(screenX - mouseX, 2) + 
          Math.pow(screenY - mouseY, 2)
        );

        // If within threshold, create a label
        if (distance < proximityThreshold) {
          var label = document.createElement('div');
          label.textContent = node.filename;

          // Style label
          label.style.position = 'absolute';
          label.style.left = (sceneRect.left + screenX) + 'px';
          label.style.top = (sceneRect.top + screenY - 15) + 'px';
          label.style.color = '%s';
          label.style.opacity = (1 - distance / proximityThreshold) * 0.8;
          label.style.fontSize = '11px';
          label.style.textShadow = '1px 1px 2px rgba(0,0,0,0.7)';
          label.style.pointerEvents = 'none';
          label.style.transformOrigin = 'center bottom';
          label.style.transform = 'translate(-50%%, -100%%)';

          labelContainer.appendChild(label);
        }
      }
    });

    // Hide labels when mouse leaves the plot
    plotDiv.addEventListener('mouseleave', function() {
      labelContainer.innerHTML = '';
    });
  });
</script>
"""

# Plot config options
PLOT_CONFIG = {
    'responsive': True,
    'scrollZoom': True,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'dependency_graph',
        'height': 1200,
        'width': 1800,
        'scale': 2
    }
}

# Centrality Metric Headers
MOST_INFLUENTIAL_HEADER = "\nMost Influential Files (by Eigenvector Centrality):"
MOST_DEPENDENT_HEADER = "\nMost Dependent Files (Incoming Dependencies):"
MOST_DEPENDENCIES_HEADER = "\nFiles with Most Dependencies (Outgoing Dependencies):"

# Visualization Annotations
CENTRALITY_EXPLANATION = (
    'Centrality Metrics Explained:<br><br>'
    '• Degree Centrality: How connected<br>'
    '  a file is overall<br><br>'
    '• In-Degree Centrality: How many<br>'
    '  other files depend on this file<br><br>'
    '• Out-Degree Centrality: How many<br>'
    '  dependencies this file has<br><br>'
    '• Eigenvector Centrality: Influence<br>'
    '  of a file based on its connections'
)

# Centrality Color Scale Texts
CENTRALITY_COLOR_TEXTS = [
    'Low Influence',
    'Moderate Influence',
    'High Influence'
]