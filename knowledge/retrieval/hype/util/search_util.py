import csv
import heapq
import re
import time

from math import sqrt
from collections import defaultdict, Counter
from itertools import combinations

from typing import List, Tuple, Any, Optional, Set

import ahocorasick
from nltk import word_tokenize
from nltk.corpus import stopwords

from spellchecker import SpellChecker

from knowledge.retrieval.hype.data.models import SearchTerm, UnifiedSearchResult
from knowledge.retrieval.hype.util.text_util import generate_ngrams

SPELLCHECKER = None

if SPELLCHECKER is None:
    SPELLCHECKER = SpellChecker()
    # SPELLCHECKER.word_frequency.load_text_file('C:\\Users\\vdonc\\user_reviews.freq')

ORDER_SEPARATOR = '::'


def preprocess_search_term_list(
        search_term_strings: List[str],
        exact_match: bool = False,
        case_sensitive: bool = False,
        spelling_autocorrect: bool = True,
        spellchecker: SpellChecker = None,
        ref_ids: List[str] = None,
) -> List[SearchTerm]:
    """
    Preprocess a list of search strings into search terms.

    Args:
        search_term_strings (List[str]): List of raw search strings.
        exact_match (bool): Whether to search for exact matches.
        case_sensitive (bool): Whether to search for case-sensitive matches.
        spelling_autocorrect (bool): Whether to autocorrect the spelling of the search terms.
        spellchecker (SpellChecker): A spell checker object used to perform the search.
        ref_ids (List[str]): List of reference ids.

    Returns:
        List[str]: List of preprocessed search terms.
    """
    search_terms = []

    if not exact_match:
        for search_term_string in search_term_strings:
            search_terms.extend(preprocess_search_term(search_term_string, case_sensitive, spelling_autocorrect, spellchecker, ref_ids))
    else:
        search_terms = parse_search_terms(search_term_strings)

    return search_terms


def preprocess_search_term(
        search_term_string: str,
        case_sensitive: bool = False,
        spelling_autocorrect: bool = True,
        spellchecker: SpellChecker = None,
        ref_ids: List[str] | None = None
) -> List[SearchTerm]:
    """
    Preprocesses the search string by spell-checking, generating n-grams, and normalizing it.

    :param search_term_string: The raw search string input.
    :param case_sensitive: Whether the search for this term should be case-sensitive.
    :param spelling_autocorrect: Whether to use spell-checking & autocorrect.
    :param spellchecker: The SpellChecker instance to use. Load your custom dictionaries here.
    :param ref_ids: A list of reference IDs to attach to the search terms.

    :return: A list of preprocessed search terms.
    """
    # Start with the original search string. This will become a search term of order 0.
    # 0 signifies the highest order of magnitude when scores are calculated (denoted by the whole number).
    #
    # These can be either set by the user or are automatically determined by the system when generating
    # search term variations, ngrams, and the such.
    #
    # Search terms of the 1st, 2nd, 3rd, etc. order will have progressively smaller magnitudes with the number signifying decimal places.
    # E.g. 0.1, 0.01, 0.001, etc.
    search_term_strings = [append_order(search_term_string, 0)]

    words = re.findall(r'\w+', search_term_string)
    corrected = []

    # One important note here is that it's not feasible to handle all capitalization variations.
    # This limitation comes from the Aho-Corasick search algorithm itself.
    # TODO One possible solution for this is to keep a searchable "de-capitalization map" (basically a patch or diff)
    #  that patches the files in memory on search-time.
    # TODO Another option would be to re-work the Aho-Corasick implementation, but that would still mean a
    #  significant performance trade-off when searching.
    for word in words:
        if spelling_autocorrect:
            if spellchecker is None:
                spellchecker = SPELLCHECKER

            spellchecked = spellchecker.correction(word)
            if spellchecked is not None and word != spellchecked:
                corrected.append(spellchecked)

    # "Fuzzy Search" functionality is implemented via overlapping ngram matching.
    if len(corrected) > 0:
        append_with_ngrams(" ".join(corrected), search_term_strings)

    if not case_sensitive:
        if search_term_string.lower() != search_term_string:
            append_with_ngrams(search_term_string.lower(), search_term_strings)

        if search_term_string.upper() != search_term_string:
            append_with_ngrams(search_term_string.upper(), search_term_strings)

        if search_term_string.title() != search_term_string:
            append_with_ngrams(search_term_string.title(), search_term_strings)

    return parse_search_terms(search_term_strings, ref_ids)


def append_with_ngrams(search_term_string: str, search_term_strings: List[str]):
    search_term_strings.append(append_order(search_term_string, 1))
    search_term_strings.extend([append_order(ngram, 2) for ngram in generate_ngrams(search_term_string)])


def parse_search_terms(search_term_strings: List[str], ref_ids: List[str] | None = None) -> List[SearchTerm]:
    parsed_search_terms = []

    for search_term_string in search_term_strings:
        split = search_term_string.split(f"{ORDER_SEPARATOR}")
        if len(split) == 2:
            parsed_search_terms.append(SearchTerm(order=int(split[1]), text=split[0], ref_ids=ref_ids))
            # print(f"ADDED {split[0]}{ORDER_SEPARATOR}{int(split[1])}")
        else:
            parsed_search_terms.append(SearchTerm(order=0, text=split[0], ref_ids=ref_ids))
            # print(f"ADDED 0{ORDER_SEPARATOR}{split[0]}")

    return parsed_search_terms


def append_order(search_term_string: str, order: int) -> str:
    # Adopting this format for specifying the order allows for easy query-level control over scoring.
    # TODO This will be extended by a customizable / scriptable scoring system.
    return f"{search_term_string}{ORDER_SEPARATOR}{order}"


def build_automaton(search_terms: List[SearchTerm]) -> ahocorasick.Automaton:
    """
    Build an Aho-Corasick automaton for keyword matching.

    Args:
        search_terms (List[str]): List of n-grams, words, or terms to search for.

    Returns:
        ahocorasick.Automaton: Aho-Corasick automaton for the keywords.
    """
    automaton = ahocorasick.Automaton()

    for search_term in search_terms:
        automaton.add_word(search_term.text, search_term)

    automaton.make_automaton()

    return automaton


def build_automaton_from_strings(search_term_strings: List[str]) -> ahocorasick.Automaton:
    """
    Build an Aho-Corasick automaton for keyword matching.

    Args:
        search_term_strings (List[str]): List of n-grams, words, or terms to search for.

    Returns:
        ahocorasick.Automaton: Aho-Corasick automaton for the keywords.
    """
    automaton = ahocorasick.Automaton()

    for search_term_string in search_term_strings:
        automaton.add_word(search_term_string, search_term_string)

    automaton.make_automaton()

    return automaton


def simple_aho_corasick_match(automaton: ahocorasick.Automaton, text: str) -> List[int]:
    """
    Perform pattern matching using the Aho-Corasick algorithm.

    Args:
        automaton (ahocorasick.Automaton): Aho-Corasick automaton for pattern matching.
        text (str): Text to search in.

    Returns:
        List[int]: List of starting positions where patterns were found.
    """
    matches = []

    for end_index, search_term in automaton.iter(text):
        # Covers basic PyAhoCorasick usage.
        if isinstance(search_term, str):
            matches.append(end_index - len(search_term) + 1)
        else:
            matches.append(end_index - len(search_term.text) + 1)

    return matches


def aho_corasick_match(automaton: ahocorasick.Automaton, text: str, min_score: float = 1) -> Tuple[List[Tuple[int, Any]], float]:
    """
    Perform pattern matching using the Aho-Corasick algorithm.

    Args:
        automaton (ahocorasick.Automaton): Aho-Corasick automaton for pattern matching.
        text (str): Text to search in.

    Returns:
        List[Tuple[int, str]]: List of tuples containing the starting positions and the matched objects.
    """
    matches = []
    search_terms = []

    for end_index, search_term in automaton.iter(text):
        matches.append((end_index - len(search_term.text) + 1, search_term))
        search_terms.append(search_term)

    if len(search_terms) > 0:
        # print(f"MATCHES: {len(matches)}")
        score = calculate_advanced_search_score(search_terms, len(text), [pos for pos, _ in matches])
        if score >= min_score:
            return matches, score

    return [], 0


def get_pattern_match_length(pattern: re.Pattern) -> int:
    length = 0
    # Compile the pattern
    compiled_pattern = re.compile(pattern)
    # Traverse through each part of the pattern
    for token in compiled_pattern.pattern:
        if token == '-':  # Literal character '-'
            length += 1
        elif token == '{':
            # Look ahead for numbers inside the curly braces
            start = compiled_pattern.pattern.find('{') + 1
            end = compiled_pattern.pattern.find('}')
            if start != -1 and end != -1:
                num = int(compiled_pattern.pattern[start:end])
                length += num
        elif token == '[':
            # Move to the closing bracket ']'
            while compiled_pattern.pattern[length] != ']':
                length += 1
            length += 1  # For closing ']'
    return length


def calculate_search_score(matched_search_terms: List[SearchTerm], document_length: int, match_positions: List[int]) -> float:
    """
    Calculate the search score based on matched terms, taking into account unique terms, term order, and match distribution.

    :param matched_search_terms: List of matched search terms.
    :param document_length: The length of the document (e.g., in characters or tokens).
    :param match_positions: The positions of the matched terms within the document.
    :return: The calculated search score.
    """
    score = 0.0
    unique_terms = set()
    term_counts_by_order = defaultdict(int)

    # Score accumulation based on term order
    for search_term in matched_search_terms:
        unique_terms.add(search_term.text)
        term_counts_by_order[search_term.order] += 1
        score += 10 / (10 ** (search_term.order + 1))

    # Unique terms multiplier
    unique_term_multiplier = len(unique_terms) ** 0.5  # Using square root to reduce the impact of large numbers of unique terms
    score *= unique_term_multiplier

    # Distribution factor
    if match_positions and document_length > 0:
        # Calculate distribution spread of matches across the document
        match_density = len(match_positions) / document_length
        normalized_distribution = min(1.0, match_density)  # Caps at 1 to prevent over-boosting
        distribution_factor = 1 + normalized_distribution
    else:
        distribution_factor = 1.0

    # Apply distribution factor
    score *= distribution_factor

    return score


def calculate_improved_search_score(
        matched_search_terms: List[SearchTerm],
        document_length: int,
        match_positions: List[int],
        partial_match_penalty: float = 0.9,
        proximity_window: int = 50
) -> float:
    """
    An improved scoring function that balances order weighting, unique term boosting,
    coverage ratio, proximity of matches, and distribution across the document.

    :param matched_search_terms: List of matched search terms (with 'order' and 'text').
    :param document_length: The length of the document (e.g., in characters or tokens).
    :param match_positions: The positions (indexes) of each match within the document.
    :param partial_match_penalty: A factor [0,1] that can penalize partial or fuzzy matches.
                                  If you only have exact matches, you can set this to 1.0.
    :param proximity_window: Controls how big a "cluster" is in terms of match indices.

    :return: The calculated search score (float).
    """
    # Early exit if no matches
    if not matched_search_terms:
        return 0.0

    score = 0.0

    # ---- 1) ORDER WEIGHTING: Milder exponential weighting ----
    # Example: if order=0 => weighting=1, order=1 => 0.5, order=2 => 0.33, etc.
    term_counts_by_order = defaultdict(int)
    for search_term in matched_search_terms:
        term_counts_by_order[search_term.order] += 1
        # Weighted by partial match factor (if you differentiate exact vs partial)
        order_weight = 1.0 / (search_term.order + 1.0)
        score += order_weight * partial_match_penalty

    # ---- 2) UNIQUE TERM BOOST ----
    unique_terms = set(t.text for t in matched_search_terms)
    unique_count = len(unique_terms)
    # Use sqrt to gently reward multiple unique terms.
    unique_term_multiplier = sqrt(unique_count)
    score *= unique_term_multiplier

    # ---- 3) COVERAGE FACTOR ----
    # Coverage = fraction of matched characters (or tokens) over total doc length
    # If you track matched substring lengths, you could sum them up instead of len(match_positions).
    if document_length > 0:
        coverage_ratio = len(match_positions) / float(document_length)
        # Example: linearly boosting coverage, or you might weight coverage more heavily
        coverage_boost = 1.0 + coverage_ratio
    else:
        coverage_boost = 1.0

    score *= coverage_boost

    # ---- 4) PROXIMITY FACTOR ----
    # If matches are clustered, we assume more contextual relevance.
    # We'll compare pairwise distances and see how many matches fall within a proximity window.
    if match_positions:
        match_positions_sorted = sorted(match_positions)
        cluster_count = 1
        cluster_start = match_positions_sorted[0]

        for pos in match_positions_sorted[1:]:
            if (pos - cluster_start) <= proximity_window:
                # still in the same cluster
                continue
            else:
                # start a new cluster
                cluster_count += 1
                cluster_start = pos

        # Fewer clusters => more concentrated => higher proximity factor
        proximity_factor = 1.0 + (1.0 / cluster_count)
    else:
        proximity_factor = 1.0

    score *= proximity_factor

    # ---- 5) DISTRIBUTION FACTOR ----
    # Original logic: slight boost if matches are scattered across doc,
    # but cap at 1.0 for the distribution ratio, then add 1 for a 2x max.
    if match_positions and document_length > 0:
        match_density = len(match_positions) / float(document_length)
        normalized_distribution = min(1.0, match_density)
        distribution_factor = 1.0 + normalized_distribution
    else:
        distribution_factor = 1.0

    score *= distribution_factor

    return score


def calculate_advanced_search_score(
        matched_search_terms: List["SearchTerm"],
        document_length: int,
        match_positions: List[int],
        partial_match_penalty: float = 0.9,
        proximity_window: int = 50,
        ignore_known_noise: bool = True,
        noise_markers: Optional[List[str]] = None
) -> float:
    """
    A unified scoring function that combines:
      1) Order weighting      (lower weight for higher-order terms)
      2) Unique term boosting (sqrt of the number of unique terms)
      3) Coverage ratio       (matches/length)
      4) Proximity clustering (fewer clusters => higher relevance)
      5) Distribution factor  (spreading matches across doc => slight boost)
      6) Partial-match penalty (for fuzzy matches)

    :param matched_search_terms:
        A list of SearchTerm objects. Each has:
          - 'order': an integer indicating the term’s priority (0 = highest).
          - 'text': the actual matched text.
          - 'ref_ids': optional metadata (unused here).
    :param document_length:
        The total length of the document (characters or other measure).
    :param match_positions:
        The character positions (or indices) where each match occurred.
        If you have multiple matches of the same term, you can list them all.
    :param partial_match_penalty:
        [0..1], a factor that penalizes partial/fuzzy matches. Default 0.9.
    :param proximity_window:
        The maximum distance between match positions to consider them
        in one “cluster.” A smaller window => more clusters => lower score.
    :param ignore_known_noise:
        If True, tries to detect known “noise” hits (like 'aria' in '[font=Arial]')
        and skip or penalize them. If False, all matches are counted equally.
    :param noise_markers:
        A list of known “noise” substrings or contexts. For instance,
        [ "font=Arial", "color=", "style=", "<style", "<script", etc. ]
        You can expand or modify to skip/penalize these matches.

    :return:
        A single float “score” that tries to reflect the overall relevance.
    """
    # Early exit if no matches
    if not matched_search_terms:
        return 0.0

    # -------------------------------------------------------------------
    # 0) Pre-process: Possibly skip or penalize known “noise” matches.
    #    You might also do a single pass to remove matches that appear in
    #    obviously noisy contexts, e.g., if the substring “aria” is inside
    #    "[font=Arial]" or "<style>" or "color=...".
    # -------------------------------------------------------------------
    if ignore_known_noise and noise_markers:
        filtered_terms = []
        filtered_positions = []
        for i, st in enumerate(matched_search_terms):
            # Heuristic: if st.text in noise_markers => skip or reduce weight
            # Or, check if the match_positions[i] region is in a markup substring
            # This logic is VERY naive. In a real scenario, you’d parse around
            # match_positions[i] to see if you’re inside [font=].
            # For simplicity, we just do “if text is in noise list => skip match”
            # You could also do partial penalties instead of skipping.
            if any(marker in st.text.lower() for marker in noise_markers):
                # skip or penalize. We’ll skip for demonstration.
                continue
            else:
                filtered_terms.append(st)
                filtered_positions.append(match_positions[i])

        matched_search_terms = filtered_terms
        match_positions = filtered_positions

        if not matched_search_terms:
            return 0.0

    # -------------------------------------------------------------------
    # 1) ORDER WEIGHTING with partial-match penalty
    #    Example: weight = (1 / (order + 1)) * partial_match_penalty
    # -------------------------------------------------------------------
    score = 0.0
    for st in matched_search_terms:
        weight = 1.0 / (st.order + 1.0)
        # Multiply by partial match penalty if applicable
        # If you do differentiate “exact” from “partial,” you’d store that in st.
        score += weight * partial_match_penalty

    # -------------------------------------------------------------------
    # 2) UNIQUE TERM BOOST
    #    sqrt(#unique terms) => “gently” reward multiple distinct matches
    # -------------------------------------------------------------------
    unique_terms = set(st.text for st in matched_search_terms)
    unique_count = len(unique_terms)
    unique_term_multiplier = sqrt(unique_count)
    score *= unique_term_multiplier

    # -------------------------------------------------------------------
    # 3) COVERAGE BOOST
    #    coverage_ratio = (# of match positions) / (document_length)
    #    final factor = 1 + coverage_ratio
    # -------------------------------------------------------------------
    if document_length > 0:
        coverage_ratio = len(match_positions) / float(document_length)
        coverage_boost = 1.0 + coverage_ratio
    else:
        coverage_boost = 1.0
    score *= coverage_boost

    # -------------------------------------------------------------------
    # 4) PROXIMITY FACTOR
    #    Fewer clusters => more “contextual” => higher factor
    #    e.g. factor = 1.0 + (1.0 / cluster_count)
    # -------------------------------------------------------------------
    if match_positions:
        match_positions_sorted = sorted(match_positions)
        cluster_count = 1
        cluster_start = match_positions_sorted[0]

        for pos in match_positions_sorted[1:]:
            if (pos - cluster_start) <= proximity_window:
                # still same cluster
                continue
            else:
                # new cluster
                cluster_count += 1
                cluster_start = pos
        proximity_factor = 1.0 + (1.0 / cluster_count)
    else:
        proximity_factor = 1.0
    score *= proximity_factor

    # -------------------------------------------------------------------
    # 5) DISTRIBUTION FACTOR
    #    Slight bonus if matches are well spread across the doc.
    #    match_density = (# of matches) / doc_length
    #    normalized <= 1
    #    final factor = 1 + normalized_distribution
    # -------------------------------------------------------------------
    if match_positions and document_length > 0:
        match_density = len(match_positions) / float(document_length)
        normalized_distribution = min(1.0, match_density)
        distribution_factor = 1.0 + normalized_distribution
    else:
        distribution_factor = 1.0
    score *= distribution_factor

    return score


def find_candidate_word_clusters(
        search_results: List[UnifiedSearchResult],
        min_words: int = 2,
        min_word_length: int = 5,
        max_results: int = 4,
        max_words: int = 4,
        timeout: int = 30
) -> List[str]:
    """
    Find candidate word clusters from search results with improved efficiency using a min-heap.

    :param search_results: List of UnifiedSearchResult objects.
    :param min_words: Minimum number of words in a cluster.
    :param min_word_length: Minimum length of each word.
    :param max_results: Maximum number of search results to process.
    :param max_words: Maximum number of words in a cluster.
    :param timeout: Maximum execution time in seconds.
    :return: List of top 10 word clusters as strings, sorted by score in descending order.
    """
    stop_words = set(stopwords.words('english'))
    word_freq = Counter()
    start_time = time.time()

    def process_text(text: str) -> List[str]:
        """
        Processes text to tokenize, filter stopwords, and update word frequencies.

        :param text: The text to process.
        :return: List of filtered and tokenized words.
        """
        words = [word.lower() for word in word_tokenize(text)
                 if word.isalnum() and word.lower() not in stop_words and len(word) >= min_word_length]
        word_freq.update(words)
        return words

    all_words = set()
    for i, result in enumerate(search_results[:max_results]):
        if time.time() - start_time > timeout:
            print(f"Timeout reached after processing {i + 1} results.")
            break
        if result.file_matches:
            all_words.update(process_text(result.file_matches.title))
            for match in result.file_matches.matches_with_context:
                all_words.update(process_text(match))
        elif result.structured_matches:
            all_words.update(process_text(result.structured_matches.item_content))
        print(f"Processed result {i + 1}/{min(len(search_results), max_results)}")

    all_words = sorted(all_words, key=lambda w: -word_freq[w])  # Sort words by frequency, descending
    print(f"Total unique words: {len(all_words)}")

    def score_combination(combo: Tuple[str]) -> float:
        """
        Scores a combination of words based on their frequencies.

        :param combo: The combination of words to score.
        :return: The calculated score for the combination.
        """
        return sum(word_freq[word] for word in combo) / len(combo)

    top_combinations = []
    processed = 0

    for i in range(min_words, min(len(all_words) + 1, max_words + 1)):
        for combo in combinations(all_words[:50], i):  # Only consider top 50 most frequent words
            if time.time() - start_time > timeout:
                print(f"Timeout reached after processing {processed} combinations.")
                return [' '.join(combo) for _, combo in sorted(top_combinations, reverse=True)]

            score = score_combination(combo)
            if len(top_combinations) < 10:
                heapq.heappush(top_combinations, (score, combo))
            elif score > top_combinations[0][0]:
                heapq.heapreplace(top_combinations, (score, combo))

            processed += 1
            if processed % 10000 == 0:
                print(f"Processed {processed} combinations")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    return [' '.join(combo) for _, combo in sorted(top_combinations, reverse=True)]


def extract_headers(filepath: str) -> Optional[List[str]]:
    """
    Extracts the headers from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        Optional[List[str]]: A list of headers if present, otherwise None.
    """
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader, None)
    return headers


def compile_patterns(item_start_pattern_regex_str: str, item_break_sequence: str = None) -> Tuple[re.Pattern, re.Pattern]:
    """
    Given a regular expression in string format, this function returns two compiled regex patterns:
    one for string matching and one for binary data matching.

    Args:
    regex_str (str): The regular expression pattern in string format.

    Returns:
    tuple: A tuple containing two compiled regex patterns (string_pattern, binary_pattern).
    """
    # Currently, the binary pattern is only used when initially splitting large files, so we need to concatenate the
    # item break sequence in order to make sure we get valid file segment breakpoints and no false positives
    # of unescaped item break sequences inside items.
    # (Pretty common in the wild with e.g. CSV dumps containing long text fields)
    if item_break_sequence is not None:
        item_start_pattern_regex_str = f"{item_break_sequence}{item_start_pattern_regex_str}"

    # Compile the pattern for string data (str type)
    string_pattern = re.compile(item_start_pattern_regex_str)

    # Compile the pattern for binary data (bytes type), using re.ASCII flag
    binary_pattern = re.compile(item_start_pattern_regex_str.encode('utf-8'), re.ASCII)

    return string_pattern, binary_pattern
