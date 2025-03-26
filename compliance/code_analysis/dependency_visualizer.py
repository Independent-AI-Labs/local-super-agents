import argparse
import math
import os
import typing
from collections import defaultdict

import networkx as nx
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from compliance.code_analysis.dependency_analyzer import DependencyAnalyzer
from compliance.code_analysis.enhanced_clustering_mixin import EnhancedClusteringMixin

from constants import (
    # Error messages
    ERROR_VISUALIZATION,

    # Colors
    COLOR_DARK_BACKGROUND,
    COLOR_DARK_PAPER,
    COLOR_EDGE,
    COLOR_NODE_LINE,
    COLOR_TEXT_LIGHT,
    COLOR_PROXIMITY_LABEL,
    COLORS_BRIGHT_PALETTE,

    # Visualization settings
    VIZ_TITLE,
    VIZ_CONNECTIONS_THRESHOLD,
    VIZ_MIN_FONT_SIZE,
    VIZ_MAX_FONT_SIZE,
    VIZ_FONT_GROWTH_RATE,
    VIZ_NODE_MIN_SIZE,
    VIZ_NODE_SIZE_FACTOR,
    VIZ_PROXIMITY_THRESHOLD,
    VIZ_EDGE_WIDTH,
    VIZ_NODE_LINE_WIDTH,

    # Report templates
    REPORT_TOTAL_FILES,
    REPORT_TOTAL_DEPENDENCIES,
    REPORT_MOST_DEPENDENT,
    REPORT_MOST_DEPENDENCIES,
    REPORT_FILE_FORMAT,

    # JavaScript template
    JS_PROXIMITY_TEMPLATE,

    # Plot config
    PLOT_CONFIG, ERROR_NO_DEPENDENCIES
)


class DependencyVisualizer(EnhancedClusteringMixin):
    """
    Specialized class for visualizing dependency graphs with enhanced aesthetics.
    """

    @staticmethod
    def get_pastel_colors(n_colors: int) -> typing.List[str]:
        """
        Generate a list of visually appealing pastel colors optimized for dark backgrounds.

        Args:
            n_colors: Number of colors to generate

        Returns:
            List of pastel colors in RGB format
        """
        # Use predefined bright pastel palette
        pastel_palette = COLORS_BRIGHT_PALETTE

        # If we need more colors than in our palette, generate them
        if n_colors > len(pastel_palette):
            import colorsys

            additional_colors = []
            for i in range(n_colors - len(pastel_palette)):
                # Generate HSV colors with high saturation and value for pastel-like colors
                h = i / (n_colors - len(pastel_palette))
                s = 0.4  # Lower saturation for pastel effect
                v = 0.95  # High value for lightness

                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                additional_colors.append(f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})')

            return pastel_palette + additional_colors

        return pastel_palette[:n_colors]

    @classmethod
    def apply_force_directed_layout(cls,
                                    graph: nx.DiGraph,
                                    clusters: typing.Dict[str, int],
                                    seed: int = 42) -> typing.Dict[str, typing.Tuple[float, float, float]]:
        """
        Apply a modified force-directed layout with cluster-based positioning.

        Args:
            graph: NetworkX graph
            clusters: Dictionary mapping nodes to cluster indices
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping nodes to 3D coordinates
        """
        # First get a basic layout
        basic_layout = nx.spring_layout(graph, dim=3, seed=seed)

        # Group nodes by cluster
        cluster_groups = defaultdict(list)
        for node, cluster in clusters.items():
            cluster_groups[cluster].append(node)

        # Create cluster centroids with spacing
        np.random.seed(seed)
        centroid_coords = {}

        for cluster_idx in cluster_groups:
            # Create a more organic distribution of cluster centroids
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.5, 1.0)

            # Spherical coordinates for a more natural 3D distribution
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            centroid_coords[cluster_idx] = (x, y, z)

        # Now adjust node positions based on clusters
        layout = {}
        for node, (x, y, z) in basic_layout.items():
            cluster = clusters.get(node, 0)
            centroid = centroid_coords[cluster]

            # Pull nodes toward their cluster centroid with jitter
            cluster_pull = 0.7  # How strongly nodes are pulled to cluster center
            jitter = np.random.uniform(0.05, 0.15, 3)  # Random variation

            new_x = x * (1 - cluster_pull) + centroid[0] * cluster_pull + jitter[0]
            new_y = y * (1 - cluster_pull) + centroid[1] * cluster_pull + jitter[1]
            new_z = z * (1 - cluster_pull) + centroid[2] * cluster_pull + jitter[2]

            layout[node] = (new_x, new_y, new_z)

        return layout

    @classmethod
    def visualize_dependencies(
            cls,
            dependency_graph: nx.DiGraph,
            base_path: str,
            output_path: str = None,
            n_clusters: int = 6,
            connectivity_weight: float = 0.3
    ) -> None:
        """
        Create an aesthetically pleasing 3D visualization of the dependency graph.

        Args:
            dependency_graph: NetworkX dependency graph to visualize
            base_path: Base path of the project for relative path computation
            output_path: Path to save the HTML visualization
            n_clusters: Number of clusters for node grouping
            connectivity_weight: Weight given to graph connectivity vs source tree structure
        """
        # Skip if no nodes
        if len(dependency_graph) == 0:
            print("No dependencies found to visualize.")
            return

        try:
            # Use enhanced clustering based on source tree structure
            node_clusters = cls.cluster_nodes_by_source_tree(
                dependency_graph,
                base_path,
                n_clusters=n_clusters,
                connectivity_weight=connectivity_weight
            )

            # Apply custom layout
            layout = cls.apply_force_directed_layout(dependency_graph, node_clusters)

            # Generate pastel colors for clusters
            cluster_colors = cls.get_pastel_colors(n_clusters)

            # Prepare visualization data
            edge_x, edge_y, edge_z = [], [], []
            node_x, node_y, node_z = [], [], []
            node_text, node_sizes, node_colors = [], [], []

            # Add cluster information to tracking
            cluster_names = {}
            for node, cluster in node_clusters.items():
                # Get directory for naming the cluster
                try:
                    rel_path = os.path.relpath(node, base_path)
                    directory = os.path.dirname(rel_path)

                    # Use directory for cluster naming
                    if directory and cluster not in cluster_names:
                        # Use the most common directory as the cluster name
                        if directory in cluster_names.values():
                            # If this directory is already used, make it more specific
                            cluster_names[cluster] = f"{directory} (group {cluster + 1})"
                        else:
                            cluster_names[cluster] = directory
                except ValueError:
                    pass

            # Ensure all clusters have names
            for cluster in set(node_clusters.values()):
                if cluster not in cluster_names:
                    cluster_names[cluster] = f"Cluster {cluster + 1}"

            # Process nodes
            for node, (x, y, z) in layout.items():
                # Compute node metrics for hover text
                in_degree = dependency_graph.in_degree(node)
                out_degree = dependency_graph.out_degree(node)
                total_degree = in_degree + out_degree

                node_x.append(x)
                node_y.append(y)
                node_z.append(z)

                # Try to get relative path for display
                try:
                    rel_path = os.path.relpath(node, base_path)
                except ValueError:
                    rel_path = node

                # Get cluster info
                cluster = node_clusters.get(node, 0)
                cluster_name = cluster_names.get(cluster, f"Cluster {cluster + 1}")

                # Add cluster info to hover text
                hover_text = (
                    f"File: {rel_path}<br>"
                    f"Cluster: {cluster_name}<br>"
                    f"Incoming Dependencies: {in_degree}<br>"
                    f"Outgoing Dependencies: {out_degree}<br>"
                    f"Total Connections: {total_degree}"
                )
                node_text.append(hover_text)

                # Enhanced node size calculation for more prominent nodes
                # Use a more aggressive growth function with square root for better scaling
                node_size = VIZ_NODE_MIN_SIZE + VIZ_NODE_SIZE_FACTOR * math.sqrt(total_degree + 1)
                node_sizes.append(node_size)

                # Node color based on cluster
                node_colors.append(cluster_colors[cluster])

            # Process edges with better aesthetics
            for edge in dependency_graph.edges():
                start = layout[edge[0]]
                end = layout[edge[1]]

                # Add a slight curve to edges for better visualization
                # Calculate midpoint with a small random offset for organic look
                midpoint_offset = [
                    (np.random.random() - 0.5) * 0.1,
                    (np.random.random() - 0.5) * 0.1,
                    (np.random.random() - 0.5) * 0.1
                ]

                mid_x = (start[0] + end[0]) / 2 + midpoint_offset[0]
                mid_y = (start[1] + end[1]) / 2 + midpoint_offset[1]
                mid_z = (start[2] + end[2]) / 2 + midpoint_offset[2]

                # Add points for a curved line (start -> mid -> end)
                edge_x.extend([start[0], mid_x, end[0], None])
                edge_y.extend([start[1], mid_y, end[1], None])
                edge_z.extend([start[2], mid_z, end[2], None])

            # Create Plotly traces for edges and nodes
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(width=VIZ_EDGE_WIDTH, color=COLOR_EDGE),
                hoverinfo='none',
                mode='lines',
                opacity=0.1  # Set opacity to 0.1 as requested
            )

            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=VIZ_NODE_LINE_WIDTH, color=COLOR_NODE_LINE),
                    opacity=0.85
                )
            )

            # Create labels for nodes with high connectivity
            label_x, label_y, label_z = [], [], []
            label_text, label_sizes = [], []

            for node, (x, y, z) in layout.items():
                total_connections = dependency_graph.in_degree(node) + dependency_graph.out_degree(node)

                # Only add labels for nodes with more than threshold connections
                if total_connections > VIZ_CONNECTIONS_THRESHOLD:
                    label_x.append(x)
                    label_y.append(y)
                    label_z.append(z)

                    # Extract just the filename for the label
                    filename = os.path.basename(node)
                    label_text.append(filename)

                    # Scale the font size based on the number of connections
                    # Start at min font size for threshold+1 connections, and increase by 1pt for every additional connections
                    connection_bonus = (total_connections - VIZ_CONNECTIONS_THRESHOLD) // VIZ_FONT_GROWTH_RATE
                    font_size = VIZ_MIN_FONT_SIZE + connection_bonus
                    font_size = min(font_size, VIZ_MAX_FONT_SIZE)  # Cap at max font size
                    label_sizes.append(font_size)

            # Create a trace for the labels
            labels_trace = go.Scatter3d(
                x=label_x, y=label_y, z=label_z,
                mode='text',
                text=label_text,
                hoverinfo='none',
                textposition='top center',
                textfont=dict(
                    size=label_sizes,
                    color='white'
                )
            )

            # Create cluster legend
            # Add a textbox showing cluster-to-directory mapping
            legend_text = "Source Tree Clusters:<br>"
            for cluster_id, cluster_name in sorted(cluster_names.items()):
                color = cluster_colors[cluster_id]
                legend_text += f"<span style='color:{color}'>■</span> {cluster_name}<br>"

            # Layout configuration with dark theme aesthetics
            fig_layout = go.Layout(
                title={
                    'text': VIZ_TITLE,
                    'font': {'size': 24, 'family': 'Arial, sans-serif', 'color': COLOR_TEXT_LIGHT},
                    'y': 0.95
                },
                scene=dict(
                    xaxis=dict(
                        showticklabels=False,
                        title='',
                        showgrid=False,
                        zeroline=False,
                        showbackground=False
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        title='',
                        showgrid=False,
                        zeroline=False,
                        showbackground=False
                    ),
                    zaxis=dict(
                        showticklabels=False,
                        title='',
                        showgrid=False,
                        zeroline=False,
                        showbackground=False
                    ),
                    bgcolor=COLOR_DARK_BACKGROUND
                ),
                showlegend=False,
                margin=dict(l=0, r=0, b=0, t=50),
                paper_bgcolor=COLOR_DARK_PAPER,
                annotations=[
                    # Add cluster legend annotation
                    dict(
                        x=0.01,
                        y=0.99,
                        xref="paper",
                        yref="paper",
                        text=legend_text,
                        showarrow=False,
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color=COLOR_TEXT_LIGHT
                        ),
                        align="left",
                        bgcolor="rgba(28, 28, 40, 0.7)",
                        bordercolor="rgba(200, 200, 200, 0.3)",
                        borderwidth=1,
                        borderpad=6,
                        opacity=0.9
                    )
                ]
            )

            # Create and save/display figure
            fig = go.Figure(data=[edge_trace, node_trace, labels_trace], layout=fig_layout)

            # Add custom JavaScript for mouse proximity labels
            # Prepare node data for JavaScript
            node_data_for_js = []
            for node, (x, y, z) in layout.items():
                total_connections = dependency_graph.in_degree(node) + dependency_graph.out_degree(node)
                filename = os.path.basename(node)
                node_data_for_js.append({
                    'x': x, 'y': y, 'z': z,
                    'filename': filename,
                    'connections': total_connections
                })

            # Convert the node data to JSON
            node_data_json = str(node_data_for_js).replace("'", '"')

            # Format the JavaScript template, replacing all placeholders
            formatted_hover_script = JS_PROXIMITY_TEMPLATE % (
                VIZ_PROXIMITY_THRESHOLD,  # proximityThreshold
                node_data_json,  # nodeData JSON
                VIZ_CONNECTIONS_THRESHOLD,  # first connections threshold
                VIZ_CONNECTIONS_THRESHOLD,  # second connections threshold
                COLOR_PROXIMITY_LABEL  # label text color
            )

            # Add the script to the figure
            fig.add_annotation(
                text=formatted_hover_script,
                showarrow=False,
                visible=False
            )

            # Add a cleaner camera view
            fig.update_layout(
                scene_camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1)
                )
            )

            if output_path:
                # Create HTML with mouse proximity feature
                html_content = pio.to_html(
                    fig,
                    include_plotlyjs=True,
                    full_html=True,
                    config=PLOT_CONFIG
                )

                # Write to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                print(f"Interactive 3D graph saved to {output_path}")
            else:
                fig.show()

        except Exception as e:
            print(f"{ERROR_VISUALIZATION}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Command-line interface for dependency visualization.
    """

    # Create argument parser
    parser = argparse.ArgumentParser(description='Python Dependency Visualizer')
    parser.add_argument(
        'base_path',
        nargs='?',
        default=os.getcwd(),
        help='Base directory to analyze (default: current working directory)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Path to save the dependency graph visualization'
    )
    parser.add_argument(
        '--no-filter-sources',
        action='store_true',
        help='Disable filtering of non-source code files'
    )
    parser.add_argument(
        '--include-builtin',
        action='store_true',
        help='Include built-in Python modules in dependency analysis'
    )
    parser.add_argument(
        '--clusters',
        '-c',
        type=int,
        default=6,
        help='Number of visual clusters to form (default: 6)'
    )
    parser.add_argument(
        '--connectivity-weight',
        '-w',
        type=float,
        default=0.3,
        help='Weight given to graph connectivity vs source tree structure (0.0-1.0, default: 0.3)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate connectivity weight
    if args.connectivity_weight < 0.0 or args.connectivity_weight > 1.0:
        print("Connectivity weight must be between 0.0 and 1.0. Using default value of 0.3.")
        args.connectivity_weight = 0.3

    try:
        # Initialize analyzer
        analyzer = DependencyAnalyzer(
            base_path=args.base_path,
            filter_sources=not args.no_filter_sources,
            ignore_builtin_modules=not args.include_builtin,
            verbose=args.verbose
        )

        # Analyze dependencies
        dependency_graph = analyzer.analyze_dependencies()

        # Check if any dependencies were found
        if len(dependency_graph.edges()) == 0:
            print(ERROR_NO_DEPENDENCIES)
            return

        # Print dependency report
        report = analyzer.generate_dependency_report()
        print(f"\n{REPORT_TOTAL_FILES.format(report['total_files'])}")
        print(f"{REPORT_TOTAL_DEPENDENCIES.format(report['total_dependencies'])}")

        print(REPORT_MOST_DEPENDENT)
        for filepath, metrics in report['most_dependent_files']:
            try:
                rel_path = os.path.relpath(filepath, args.base_path)
                print(REPORT_FILE_FORMAT.format(rel_path, metrics['in_degree'], "incoming"))
            except ValueError:
                # Handle case where paths are on different drives
                print(REPORT_FILE_FORMAT.format(filepath, metrics['in_degree'], "incoming"))

        print(REPORT_MOST_DEPENDENCIES)
        for filepath, metrics in report['most_dependencies']:
            try:
                rel_path = os.path.relpath(filepath, args.base_path)
                print(REPORT_FILE_FORMAT.format(rel_path, metrics['out_degree'], "outgoing"))
            except ValueError:
                # Handle case where paths are on different drives
                print(REPORT_FILE_FORMAT.format(filepath, metrics['out_degree'], "outgoing"))

        # Determine output path
        output_path = args.output or os.path.join(args.base_path, 'dependency_graph.html')

        # Visualize dependencies
        print(f"\nGenerating visualization with source tree clustering...")
        print(f"Using {args.clusters} clusters with connectivity weight of {args.connectivity_weight}")

        DependencyVisualizer.visualize_dependencies(
            dependency_graph,
            base_path=args.base_path,
            output_path=output_path,
            n_clusters=args.clusters,
            connectivity_weight=args.connectivity_weight
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
