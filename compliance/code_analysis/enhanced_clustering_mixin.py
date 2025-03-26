import os
from collections import defaultdict

import networkx as nx
import numpy as np


class EnhancedClusteringMixin:
    """
    Mixin to enhance the DependencyVisualizer with improved clustering
    based on source tree structure and logical proximity.
    """

    @staticmethod
    def compute_path_similarity(path1, path2, base_path):
        """
        Compute similarity between two file paths based on their location in source tree.

        Args:
            path1: First file path
            path2: Second file path
            base_path: Base path of the project

        Returns:
            Float value representing similarity (0-1 range, higher means more similar)
        """
        # Convert absolute paths to relative paths
        try:
            rel_path1 = os.path.relpath(path1, base_path)
            rel_path2 = os.path.relpath(path2, base_path)
        except ValueError:
            # Handle case when paths are on different drives
            return 0.0

        # Split paths into components
        components1 = rel_path1.split(os.sep)
        components2 = rel_path2.split(os.sep)

        # Calculate common prefix length
        common_prefix_len = 0
        for c1, c2 in zip(components1, components2):
            if c1 == c2:
                common_prefix_len += 1
            else:
                break

        # Calculate max possible common length
        max_common_len = min(len(components1), len(components2))

        # Similarity is ratio of common prefix to path length
        # Scale it to favor even small commonalities
        if max_common_len == 0:
            return 0.0

        # Base similarity on common prefix
        path_similarity = common_prefix_len / max_common_len

        # Boost similarity for files in the same directory
        if os.path.dirname(rel_path1) == os.path.dirname(rel_path2):
            path_similarity += 0.3

        # Additional boost for files with similar names (e.g., test files)
        filename1 = os.path.basename(rel_path1)
        filename2 = os.path.basename(rel_path2)

        # Check if filenames share common prefixes or suffixes
        name1 = os.path.splitext(filename1)[0]
        name2 = os.path.splitext(filename2)[0]

        # Look for common patterns in names
        if (name1.startswith(name2) or name2.startswith(name1) or
                name1.endswith(name2) or name2.endswith(name1)):
            path_similarity += 0.2

        # Cap similarity at 1.0
        return min(path_similarity, 1.0)

    @classmethod
    def create_source_tree_similarity_matrix(cls, graph, base_path):
        """
        Create a similarity matrix between nodes based on their location in source tree.

        Args:
            graph: NetworkX graph of dependencies
            base_path: Base path of the project

        Returns:
            Numpy array with similarity values
        """
        nodes = list(graph.nodes())
        n_nodes = len(nodes)

        # Create node index mapping
        node_indices = {node: i for i, node in enumerate(nodes)}

        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_nodes, n_nodes))

        # Compute pairwise similarities
        for i, node1 in enumerate(nodes):
            # Self-similarity is maximum
            similarity_matrix[i, i] = 1.0

            for j in range(i + 1, n_nodes):
                node2 = nodes[j]

                # Compute path-based similarity
                sim = cls.compute_path_similarity(node1, node2, base_path)

                # Make similarity matrix symmetric
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix, node_indices

    @classmethod
    def cluster_nodes_by_source_tree(cls, graph, base_path, n_clusters=8, connectivity_weight=0.3):
        """
        Cluster nodes based on source tree structure and connectivity.

        Args:
            graph: NetworkX graph
            base_path: Base path of the project
            n_clusters: Number of clusters to form
            connectivity_weight: Weight given to graph connectivity (vs source tree)

        Returns:
            Dictionary mapping node IDs to cluster indices
        """
        from sklearn.cluster import AgglomerativeClustering

        # If graph is too small, reduce clusters
        if len(graph) < n_clusters:
            n_clusters = max(2, len(graph) // 2)

        # Get source tree similarity matrix
        path_similarity, node_indices = cls.create_source_tree_similarity_matrix(graph, base_path)

        # Get graph connectivity matrix
        adjacency_matrix = nx.to_numpy_array(graph, nodelist=list(node_indices.keys()))

        # Normalize adjacency matrix
        if adjacency_matrix.sum() > 0:  # Avoid division by zero
            adjacency_matrix = adjacency_matrix / adjacency_matrix.sum()

        # Combine path similarity and graph connectivity
        # The higher the connectivity_weight, the more emphasis on graph structure
        # The lower, the more emphasis on file system structure
        combined_similarity = (1 - connectivity_weight) * path_similarity + connectivity_weight * adjacency_matrix

        # Convert to distance matrix (1 - similarity)
        distance_matrix = 1 - combined_similarity

        try:
            # Apply hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            ).fit(distance_matrix)

            # Map nodes to clusters
            node_to_cluster = {node: cluster for node, cluster in
                               zip(node_indices.keys(), clustering.labels_)}

            return node_to_cluster

        except Exception as e:
            print(f"Clustering error: {e}. Using fallback clustering method.")

            # Fallback: Use directory-based clustering
            return cls.fallback_directory_clustering(graph, base_path, n_clusters)

    @classmethod
    def fallback_directory_clustering(cls, graph, base_path, n_clusters):
        """
        Simple fallback clustering based on directories.

        Args:
            graph: NetworkX graph
            base_path: Base path of the project
            n_clusters: Number of clusters to aim for

        Returns:
            Dictionary mapping node IDs to cluster indices
        """
        # Group files by directory
        dir_groups = defaultdict(list)

        for node in graph.nodes():
            try:
                # Get directory relative to base path
                rel_path = os.path.relpath(node, base_path)
                directory = os.path.dirname(rel_path)

                # Use directory as group key
                dir_groups[directory].append(node)
            except ValueError:
                # Handle case when paths are on different drives
                dir_groups["_other"].append(node)

        # If we have too many directories, merge smaller ones
        if len(dir_groups) > n_clusters:
            # Sort directories by size (number of files)
            sorted_dirs = sorted(dir_groups.items(), key=lambda x: len(x[1]))

            # Merge smallest directories until we have n_clusters
            while len(sorted_dirs) > n_clusters:
                smallest = sorted_dirs.pop(0)  # Remove smallest
                second_smallest = sorted_dirs.pop(0)  # Remove second smallest

                # Merge the two smallest
                merged = (
                    f"{second_smallest[0]}_merged",
                    second_smallest[1] + smallest[1]
                )

                # Add merged group back and re-sort
                sorted_dirs.append(merged)
                sorted_dirs.sort(key=lambda x: len(x[1]))

            # Convert back to dict
            dir_groups = {dir_name: files for dir_name, files in sorted_dirs}

        # Assign cluster numbers
        node_to_cluster = {}
        for cluster_idx, (_, nodes) in enumerate(dir_groups.items()):
            for node in nodes:
                node_to_cluster[node] = cluster_idx

        return node_to_cluster
