from typing import List, Tuple
from nltk import ngrams


def generate_ngrams(word: str, min_length: int = 5) -> List[str]:
    """
    Generates n-grams from a given word based on its length.

    Args:
        word (str): The word to generate n-grams from.
        min_length (int): The minimum length of the word to generate n-grams. Defaults to 5.

    Returns:
        List[str]: A list of generated n-grams.
    """
    n = 3
    if len(word) >= 7:
        n = 4
    if len(word) >= 9:
        n = 5

    return [''.join(gram) for gram in ngrams(word, n)] if len(word) >= min_length else [word]


def extract_matches_with_context(
        match_lines: List[int], context_size_lines: int, content: str, line_indices: List[int]
) -> List[str]:
    """
    Extracts matches with context from the content based on the given line indices.

    Args:
        match_lines (List[int]): The lines that contain matches.
        context_size_lines (int): The number of lines to include as context.
        content (str): The content to search within.
        line_indices (List[int]): The line indices of the content.

    Returns:
        List[str]: A list of strings containing the matches with context.
    """
    matches_with_context = []
    content_lines = content.splitlines()
    total_lines = len(content_lines)

    for cluster_start, cluster_end in cluster_match_lines(match_lines, context_size_lines):
        context_start = max(0, cluster_start - context_size_lines)
        context_end = min(total_lines, cluster_end + context_size_lines + 1)

        numbered_content = [
            f"|{i + 1}|    {content_lines[i]}"
            for i in range(context_start, context_end)
        ]
        matches_with_context.append("\n".join(numbered_content))

    return matches_with_context


def cluster_match_lines(match_lines: List[int], context_size_lines: int) -> List[Tuple[int, int]]:
    """
    Clusters match lines that are close to each other.

    Args:
        match_lines (List[int]): The lines that contain matches.
        context_size_lines (int): The number of lines to include as context.

    Returns:
        List[Tuple[int, int]]: A list of tuples containing the start and end of each cluster.
    """
    if not match_lines:
        return []

    clusters = []
    cluster_start = match_lines[0]
    prev_line = cluster_start

    for line in match_lines[1:]:
        if line - prev_line > context_size_lines * 2:
            clusters.append((cluster_start, prev_line))
            cluster_start = line
        prev_line = line

    clusters.append((cluster_start, prev_line))
    return clusters
