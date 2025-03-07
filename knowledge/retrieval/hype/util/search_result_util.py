import mmap
import os
import queue
import time

from multiprocessing import Queue
from typing import List, Tuple, Dict, Any, Union

from knowledge.retrieval.hype.data.models import UnifiedSearchResult, CommonData, FileMatch, SearchTerm, StructuredMatch
from knowledge.retrieval.hype.util.file_util import get_line_number
from knowledge.retrieval.hype.util.search_util import build_automaton, aho_corasick_match, calculate_advanced_search_score
from knowledge.retrieval.hype.util.text_util import extract_matches_with_context


def search_file(content: str, search_terms: List[SearchTerm], min_score: float = 1) -> Tuple[List[Tuple[int, Any]], float]:
    """
    Search for terms in a given content using Aho-Corasick automaton.
    """
    automaton = build_automaton(search_terms)
    return aho_corasick_match(automaton, content, min_score)


def collect_and_merge_results(results_queue: Union[Queue, queue.Queue]) -> List[UnifiedSearchResult]:
    """
    Efficiently collect and merge results from the queue, preserving all original data.
    """
    merged_results: Dict[str, UnifiedSearchResult] = {}
    total_blocks = 0

    while not results_queue.empty():
        results: List[UnifiedSearchResult] = results_queue.get()
        total_blocks += 1

        for result in results:
            key = f"{result.common.uri}:{result.structured_matches.item_index}"
            if key not in merged_results:
                merged_results[key] = result.model_copy(deep=True)
            else:
                if result.structured_matches:
                    merged_results[key].structured_matches.item_content = result.structured_matches.item_content
                else:
                    merged_results[key].file_matches.line_numbers.extend(result.file_matches.line_numbers)
                    merged_results[key].file_matches.matches_with_context.extend(result.file_matches.matches_with_context)

                merged_results[key].common.matched_search_terms.extend(result.common.matched_search_terms)
                for pattern, count in result.common.search_term_match_counts.items():
                    merged_results[key].common.search_term_match_counts[pattern] = (
                            merged_results[key].common.search_term_match_counts.get(pattern, 0) + count
                    )

    return sorted(merged_results.values(), key=lambda x: x.common.score, reverse=True)


def build_search_result_for_file(
        file_path: str,
        search_term_matches: List[Tuple[int, SearchTerm]],
        score: float,
        line_indices: List[int],
        content: str,
        context_size_lines: int,
        and_search: bool,
        search_term_count: int,
        timing: Dict
) -> Tuple[List[UnifiedSearchResult], Dict[str, float]]:
    """
    Build search results from matches, extracting context and calculating scores.
    """
    start_time = time.time()

    if not search_term_matches:
        return [], timing

    search_term_match_counts = {}
    line_numbers = [get_line_number(match_index, line_indices) for match_index, _ in search_term_matches]

    for match_index, search_term in search_term_matches:
        search_term_match_counts[search_term.text] = search_term_match_counts.get(search_term.text, 0) + 1

    if and_search and len(search_term_match_counts) < search_term_count:
        return [], timing

    context_start = time.time()

    content_lines = content.splitlines()
    title = "\n".join(content_lines[:min(5, len(content_lines))]).strip()

    file_matches = FileMatch(
        line_numbers=line_numbers,
        title=title,
        matches_with_context=extract_matches_with_context(line_numbers, context_size_lines, content, line_indices)
    )

    common_data = CommonData(
        uri=file_path,
        search_term_match_counts=search_term_match_counts,
        matched_search_terms=[search_term[1] for search_term in search_term_matches],
        score=score
    )

    result = UnifiedSearchResult(
        file_matches=file_matches,
        common=common_data
    )

    timing['context'] = time.time() - context_start
    timing['total'] = time.time() - start_time

    return [result], timing


def process_structured_matches(
        search_term_matches: List[Tuple[int, SearchTerm]],
        min_score: float,
        metadata_filename: str,
        block_data_string: str,
        file_path: str
) -> List[UnifiedSearchResult]:
    """
    Process matches and create search results for structured data directly.
    """
    with mmap.mmap(os.open(metadata_filename, os.O_RDONLY), 0, access=mmap.ACCESS_READ) as metadata_mm:
        unique_item_results = {}

        for match_index, search_term in search_term_matches:
            item_start, item_end, item_index = binary_search_metadata(metadata_mm, match_index)

            if item_index != -1:
                if item_index not in unique_item_results:
                    # Create a search result directly if not already in the unique_results.
                    unique_item_results[item_index] = UnifiedSearchResult(
                        structured_matches=StructuredMatch(
                            item_index=item_index,
                            item_content=block_data_string[item_start:item_end]
                        ),
                        common=CommonData(
                            uri=file_path
                        )
                    )

                # Update the search result.
                result = unique_item_results[item_index]

                result.common.search_term_match_counts[search_term.text] = result.common.search_term_match_counts.get(search_term.text, 0) + 1
                result.common.matched_search_terms.append(search_term)
            else:
                print(f"Warning: ITEM NOT FOUND FOR MATCH POSITION {match_index}!")
                continue

    # print(unique_item_results)

    final_results = []

    for unique_result in unique_item_results.values():
        unique_result.common.score = calculate_advanced_search_score(unique_result.common.matched_search_terms, len(block_data_string),
                                                                     [pos for pos, _ in search_term_matches])

        if unique_result.common.score >= min_score:
            # print(unique_result.common.score)
            final_results.append(unique_result)

    return final_results


def binary_search_metadata(mm: mmap.mmap, target_index: int, line_length: int = 37) -> Tuple[int, int, int]:
    """
    Perform a binary search on a memory-mapped metadata file to find the row containing the target index.
    """
    num_lines = mm.size() // line_length

    low, high = 0, num_lines - 1
    row_start, row_end, row_index = -1, -1, -1

    while low <= high:
        mid = (low + high) // 2
        line_offset = mid * line_length

        mm.seek(line_offset)
        line = mm.read(line_length).decode('utf-8', errors='ignore').strip()

        try:
            start, end, row_idx = map(int, line.split(','))
        except ValueError:
            continue

        if start <= target_index < end:
            row_start, row_end, row_index = start, end, row_idx
            break
        elif target_index < start:
            high = mid - 1
        else:
            low = mid + 1

    return row_start, row_end, row_index
