from typing import List, Optional, Tuple

from spellchecker import SpellChecker

from retrieval.hype.search.file_search import search_structured_data
from retrieval.hype.data.models import UnifiedSearchResult
from retrieval.hype.util.search_util import find_candidate_word_clusters


# TODO 'AND' searching.
def search_structured_data_file(
        file_path: str,
        metadata_dir: str,
        search_term_strings: List[str],
        exact_match: bool = False,
        case_sensitive: bool = False,
        spelling_autocorrect: bool = True,
        spellchecker: SpellChecker = None,
        min_matches: int = 1,
        derivative_search: bool = False,
) -> Tuple[List[UnifiedSearchResult], Optional[List[UnifiedSearchResult]]]:
    """
    Search for terms in a structured data file with an optional derivative search.

    Args:
        file_path (str): Path to the data file to be searched.
        metadata_dir (str): Directory containing metadata files.
        search_term_strings (List[str]): List of search strings.
        exact_match (bool): Whether to search for exact matches.
        case_sensitive (bool): Whether to search for case-sensitive matches.
        spelling_autocorrect (bool): Whether to autocorrect the spelling of the search terms.
        spellchecker (SpellChecker): The SpellChecker instance to use. Load your custom dictionaries here.
        derivative_search (bool): Whether to perform a derivative search based on initial results.

    Returns:
        Tuple[List[UnifiedSearchResult], Optional[List[UnifiedSearchResult]]]:
        - initial_results: Results from the initial search.
        - derivative_results: Results from the derivative search if performed, otherwise None.
    """
    # Perform initial search
    initial_results = search_structured_data(file_path, metadata_dir, search_term_strings, min_matches)

    derivative_results = None
    if derivative_search:
        # TODO Derivative search based on frequent word clusters from the initial results.
        # derivative_results = perform_derivative_search(initial_results, file_path, metadata_dir)
        pass

    return initial_results, derivative_results


def perform_derivative_search(
        initial_results: List[UnifiedSearchResult],
        file_path: str,
        metadata_dir: str
) -> List[UnifiedSearchResult]:
    """
    Perform a derivative search based on the initial search results.

    Args:
        initial_results (List[UnifiedSearchResult]): Initial search results to derive the next search terms.
        file_path (str): Path to the data file to be searched.
        offsets_file (str): Path to the offsets file for block manage.
        metadata_dir (str): Directory containing metadata files.

    Returns:
        List[UnifiedSearchResult]: Results from the derivative search.
    """
    # Find candidate clusters to use as new search terms
    candidate_clusters = find_candidate_word_clusters(initial_results)[:1]

    # Perform the derivative search with the new terms
    return search_structured_data(file_path, metadata_dir, candidate_clusters)
