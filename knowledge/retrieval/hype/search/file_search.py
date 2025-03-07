import collections
import hashlib
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import Manager, Queue
from typing import List, Dict

import psutil

from knowledge.retrieval.hype.data.models import UnifiedSearchResult, SearchTerm
from knowledge.retrieval.hype.indexing.structured_data_mapper import execute_memory_managed_block_processing_threads
from knowledge.retrieval.hype.util.file_util import build_file_tree, extract_file_content, load_offsets, categorize_files, read_mmap
from knowledge.retrieval.hype.util.search_result_util import (
    collect_and_merge_results,
    build_search_result_for_file,
    search_file, process_structured_matches
)
from knowledge.retrieval.hype.util.search_util import build_automaton, aho_corasick_match, preprocess_search_term_list, parse_search_terms


def search_in_structured_file_block(
        file_path: str,
        start_offset: int,
        end_offset: int,
        block_id: int,
        search_terms: List[SearchTerm],
        metadata_dir: str,
        results_queue: Queue,
        min_score: float = 1
) -> Dict[str, float]:
    """
    Optimized search for patterns in a file block and return the results, with timing measurements.
    """
    timings = {}
    start_total = time.time()

    # print(f"Searching block ID {block_id}")

    block_data_string = read_mmap(file_path, start_offset, end_offset)
    timings['file_open_and_mmap'] = time.time() - start_total

    # Build automaton and find matches
    start_time = time.time()
    search_automaton = build_automaton(search_terms)
    block_matches, block_score = aho_corasick_match(search_automaton, block_data_string, min_score)
    timings['aho_corasick_match'] = time.time() - start_time

    if len(block_matches) > 0:
        # Process matches
        start_time = time.time()
        metadata_filename = os.path.join(metadata_dir, f"metadata_{block_id}.csv")

        search_results = process_structured_matches(block_matches, min_score, metadata_filename, block_data_string, file_path)

        # print(f"MATCHES: {len(block_matches)} RESULTS: {len(search_results)}")

        timings['process_matches'] = time.time() - start_time
        timings['total_time'] = time.time() - start_total

        if search_results:
            # print(search_results)
            results_queue.put(search_results)

    timings['process_matches'] = time.time() - start_time
    timings['total_time'] = time.time() - start_total

    return timings


def search_structured_data(
        file_path: str,
        metadata_dir: str,
        search_term_strings: List[str],
        min_score: float = 1
) -> List[UnifiedSearchResult]:
    # The below approach is not efficient enough.
    # """
    # Search for terms in a structured data file using Aho-Corasick with memory-aware multiprocessing.
    # """
    # # Full path to the file becomes the UUID for the metadata subdir.
    # metadata_dir = os.path.join(metadata_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())
    #
    # offsets_file = os.path.join(metadata_dir, "offsets.csv")
    #
    # block_offsets = load_offsets(offsets_file)
    #
    # with Manager() as manager:
    #     results_queue = manager.Queue()
    #
    #     procs = [Process(target=search_in_structured_file_block, args=(
    #         file_path, start_offset, end_offset, block_id, search_terms, metadata_dir, results_queue
    #     )) for block_id, (start_offset, end_offset) in enumerate(block_offsets)]
    #
    #     execute_memory_managed_block_processing(block_offsets, procs)
    #
    #     return collect_and_merge_results(results_queue)
    search_terms = preprocess_search_term_list(search_term_strings)

    return search_structured_data_threaded(file_path, metadata_dir, search_terms, min_score)


def search_structured_data_threaded(
        file_path: str,
        metadata_dir: str,
        search_terms: List[SearchTerm],
        min_score: float = 1
) -> List[UnifiedSearchResult]:
    """
    Search for terms in a structured data file using Aho-Corasick with memory-aware threading.
    """
    # Full path to the file becomes the UUID for the metadata subdir.
    metadata_dir = os.path.join(metadata_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())

    offsets_file = os.path.join(metadata_dir, "offsets.csv")

    block_offsets = load_offsets(offsets_file)

    with Manager() as manager:
        results_queue = manager.Queue()
        threads = [threading.Thread(target=search_in_structured_file_block, args=(
            file_path, start_offset, end_offset, block_id, search_terms, metadata_dir, results_queue, min_score
        )) for block_id, (start_offset, end_offset) in enumerate(block_offsets)]

        execute_memory_managed_block_processing_threads(block_offsets, threads)

        return collect_and_merge_results(results_queue)


def search_structured_data_batched(
        file_path: str,
        metadata_dir: str,
        search_term_strings: List[str],
        min_score: float = 1,
        free_memory_usage_limit: float = 0.75,
        memory_overhead_factor: int = 1,
) -> List[UnifiedSearchResult]:
    """
    Search for terms in a structured data file using a hybrid multi-manage / threading approach.
    """
    # Full path to the file becomes the UUID for the metadata subdir.
    file_metadata_dir = os.path.join(metadata_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())
    offsets_file = os.path.join(file_metadata_dir, "offsets.csv")
    block_offsets = load_offsets(offsets_file)

    search_terms = preprocess_search_term_list(search_term_strings)

    num_physical_cores = psutil.cpu_count(logical=False)
    if len(block_offsets) / 2 > num_physical_cores:
        chunk_size = len(block_offsets) // num_physical_cores
    else:
        # Revert back to the threaded implementation as the performance gains will not be significant enough to overcome the
        # process creation overhead...
        print("(Relatively) small file. Reverting back to Threaded search...")
        return search_structured_data_threaded(file_path, metadata_dir, search_terms, min_score)

    chunks = [block_offsets[i:i + chunk_size] for i in range(0, len(block_offsets), chunk_size)]

    with Manager() as manager:
        results_queue = manager.Queue()

        batch_search_blocks(
            file_path,
            search_terms,
            chunks,
            metadata_dir,
            results_queue,
            min_score,
            free_memory_usage_limit,
            memory_overhead_factor
        )

        return collect_and_merge_results(results_queue)


def batch_search(
        file_path: str,
        block_offset_batch: List[tuple],
        search_terms: List[SearchTerm],
        metadata_dir: str,
        parent_result_queue: Queue,
        offset_offset: int,  # I'm sorry about this.
        min_score: float = 1
):
    """
    Process a chunk of blocks using threading within a process.
    """
    file_metadata_dir = os.path.join(metadata_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())

    for i, (start, end) in enumerate(block_offset_batch):
        search_in_structured_file_block(file_path, start, end, i + offset_offset, search_terms, file_metadata_dir, parent_result_queue, min_score)


def batch_search_blocks(
        file_path: str,
        search_terms: List[SearchTerm],
        block_offset_chunks: list,
        metadata_dir: str,
        result_queue: Queue,
        min_score: float,
        free_memory_usage_limit: float = 0.75,
        memory_overhead_factor: int = 1,
        batch_size: int = 2  # Number of completed processes to wait for before submitting new ones
) -> None:
    """
    A hybrid memory-managed block manage function using both process pooling
    and memory checks to manage large tasks efficiently.

    :param block_offset_chunks: List of tuples containing start and end offsets for blocks.
    :param task_function: The function to execute for each block.
    :param args_tuple: Arguments to pass to the task_function.
    :param free_memory_usage_limit: Fraction of free memory to consider safe for new task submission.
    :param memory_overhead_factor: Memory overhead factor for task estimation.
    :param batch_size: Number of completed processes to wait for before submitting new tasks.
    """
    start_time = time.time()
    max_procs = psutil.cpu_count(logical=False)

    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        futures = []
        running_procs = 0
        last_submitted = 0

        while last_submitted < len(block_offset_chunks) or running_procs > 0:
            # Monitor and handle completed tasks
            if running_procs > 0:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.remove(future)
                    running_procs -= 1

            # Submit new tasks if possible
            while running_procs < max_procs and last_submitted < len(block_offset_chunks):
                offsets_range = block_offset_chunks[last_submitted]

                start = offsets_range[0][0]
                end = offsets_range[-1][1]

                # Check memory before starting a new process
                required_memory = (end - start) * memory_overhead_factor
                available_memory = psutil.virtual_memory().available * free_memory_usage_limit

                if required_memory > available_memory:
                    print(f"Waiting for memory to free up... Required: {required_memory}, Available: {available_memory}")
                    time.sleep(0.1)  # Wait and re-check memory availability
                    break  # Exit the inner while loop to re-check all conditions

                # Submit task to the executor
                # print(f"Starting process {last_submitted}... Available memory: {available_memory}")
                future = executor.submit(batch_search, file_path, block_offset_chunks[last_submitted], search_terms, metadata_dir, result_queue,
                                         last_submitted * len(block_offset_chunks[0]), min_score)
                futures.append(future)
                running_procs += 1
                last_submitted += 1

            # Wait until at least `batch_size` tasks complete if we can't submit more
            if running_procs >= max_procs or (last_submitted >= len(block_offset_chunks) and running_procs > 0):
                # print(f"Waiting for at least {batch_size} processes to finish...")
                done, not_done = wait(futures, return_when=FIRST_COMPLETED, timeout=1)
                completed_count = len(done)

                if completed_count < batch_size:
                    # print("Still waiting, fewer than batch size completed.")
                    time.sleep(0.1)
                    continue

        # Wait for all remaining futures to complete
        for future in futures:
            future.result()

    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"{sum([len(blocks) for blocks in block_offset_chunks])} blocks processed in {process_time:.2f} ms.")


def process_file_list(
        file_list: List[str],
        search_term_strings: List[str],
        context_size_lines: int,
        results_queue: Queue,
        min_score: float = 1,
        and_search: bool = False,
        exact_matches_only: bool = True,
) -> None:
    if exact_matches_only:
        search_terms = parse_search_terms(search_term_strings)
    else:
        search_terms = preprocess_search_term_list(search_term_strings)

    all_results = []
    timing_totals = {'extraction': 0, 'automaton': 0, 'matching': 0, 'context': 0, 'total': 0}
    file_count = 0
    search_time = 0

    for file_path in file_list:
        search_start_time = time.time()

        extract_start = time.time()
        content, line_indices, is_utf8 = extract_file_content(file_path)
        extract_end = time.time()
        timing_totals['extraction'] += extract_end - extract_start

        file_matches, score = search_file(content, search_terms, min_score)

        if len(file_matches) == 0:
            continue

        results, timing = build_search_result_for_file(file_path, file_matches, score, line_indices, content, context_size_lines, and_search, len(search_terms),
                                                       timing_totals)

        all_results.extend(results)
        search_time += time.time() - search_start_time
        file_count += 1

    # print(round(timing_totals['extraction']))

    results_queue.put(all_results)


# TODO IMPORTANT: "Fuzzy" searching a large document corpus can have a significant performance impact.
#  With more search terms and a low min_matches value it has the potential to generate  A  L O T  of results.
#  Only set exact_matches_only to "False" if you have tons of RAM (and patience).
def bulk_search_files(
        root_dir: str,
        search_term_strings: List[str],
        context_size_lines: int = 16,
        large_file_size_threshold: int = 512 * 1024 * 1024,  # 512MB
        min_score: float = 1,
        and_search: bool = False,
        exact_matches_only: bool = True
) -> List[UnifiedSearchResult]:
    # start_time = time.time()
    # print(f"Starting bulk search in {root_dir}")

    # tree_build_start = time.time()
    root = build_file_tree(root_dir)
    # tree_build_end = time.time()
    # print(f"File tree build time: {tree_build_end - tree_build_start:.4f} seconds")

    large_files = []
    small_files = []

    # categorize_start = time.time()
    categorize_files(large_files, small_files, root, large_file_size_threshold)
    # categorize_end = time.time()
    # print(f"File categorization time: {categorize_end - categorize_start:.4f} seconds")
    # print(f"Number of large files: {len(large_files)}")
    # print(f"Number of small files: {len(small_files)}")

    results_queue = queue.Queue()
    threads = []

    def create_thread(target, args):
        t = threading.Thread(target=target, args=args)
        threads.append(t)
        return t

    for file_path in large_files:
        # TODO Case
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ['.csv', '.tsv']:
            create_thread(target=search_structured_data, args=(
                file_path, f"{file_path}.offsets", f"{file_path}.metadata", search_term_strings, results_queue
            ))

    # prep_start = time.time()
    # The thread count does not really matter outside of free-threaded Python, as long as there are enough threads
    # to compensate for the I/O waits (saturate the CPU utilization).
    cpu_count = psutil.cpu_count(logical=False)
    files_per_thread = max(1, len(small_files) // cpu_count)
    for i in range(0, len(small_files), files_per_thread):
        file_chunk = small_files[i:i + files_per_thread]
        create_thread(target=process_file_list,
                      args=(file_chunk, search_term_strings, context_size_lines, results_queue, min_score, and_search, exact_matches_only))

    # prep_end = time.time()
    # print(f"Thread setup time: {prep_end - prep_start:.4f} seconds.")

    # process_start = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    # process_end = time.time()
    # print(f"File manage time: {process_end - process_start:.4f} seconds")

    # collect_start = time.time()
    all_results = []
    while not results_queue.empty():
        all_results.extend(results_queue.get())
    # collect_end = time.time()
    # print(f"Result collection time: {collect_end - collect_start:.4f} seconds")

    sorted_results = sorted(all_results, key=lambda x: x.common.score, reverse=True)

    # end_time = time.time()
    # print(f"Total bulk search time: {end_time - start_time:.4f} seconds")
    # print(f"Total results found: {len(sorted_results)}")

    return sorted_results


class WorkStealingQueue:
    def __init__(self, num_workers=None):
        if num_workers is None:
            num_workers = os.cpu_count()

        self.num_workers = num_workers
        self.queues = [collections.deque() for _ in range(num_workers)]
        self.locks = [threading.Lock() for _ in range(num_workers)]
        self.worker_indices = {}  # Map thread ID to worker index
        self.next_worker = 0
        self.global_lock = threading.Lock()

    def get_worker_index(self):
        thread_id = threading.get_ident()
        if thread_id not in self.worker_indices:
            with self.global_lock:
                self.worker_indices[thread_id] = self.next_worker
                self.next_worker = (self.next_worker + 1) % self.num_workers
        return self.worker_indices[thread_id]

    def push(self, item):
        worker_idx = self.get_worker_index()
        with self.locks[worker_idx]:
            self.queues[worker_idx].append(item)

    def pop(self):
        # First try to get work from our own queue
        worker_idx = self.get_worker_index()
        with self.locks[worker_idx]:
            if self.queues[worker_idx]:
                return self.queues[worker_idx].pop()

        # Try to steal work from other queues
        for i in range(self.num_workers):
            idx = (worker_idx + i + 1) % self.num_workers
            with self.locks[idx]:
                if self.queues[idx]:
                    return self.queues[idx].pop()

        # No work found
        return None

    def push_many(self, items):
        # Distribute items across queues
        chunks = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(items):
            chunks[i % self.num_workers].append(item)

        for i, chunk in enumerate(chunks):
            if chunk:
                with self.locks[i]:
                    self.queues[i].extend(chunk)

    def is_empty(self):
        for i in range(self.num_workers):
            with self.locks[i]:
                if self.queues[i]:
                    return False
        return True


# TODO TEST!!!
def bulk_search_files_optimized(
        root_dir: str,
        search_term_strings: List[str],
        context_size_lines: int = 16,
        large_file_size_threshold: int = 512 * 1024 * 1024,
        min_score: float = 1,
        and_search: bool = False,
        exact_matches_only: bool = True
) -> List[UnifiedSearchResult]:
    # Build file tree and categorize files
    root = build_file_tree(root_dir)
    large_files = []
    small_files = []
    categorize_files(large_files, small_files, root, large_file_size_threshold)

    # Create work-stealing queue
    work_queue = WorkStealingQueue()

    # Process small files with work-stealing queue
    cpu_count = psutil.cpu_count(logical=False)

    # Instead of fixed chunks, add all files to the work queue
    work_queue.push_many(small_files)

    results_queue = queue.Queue()
    threads = []

    def worker():
        while not work_queue.is_empty():
            file_path = work_queue.pop()
            if file_path is None:
                break

            # Process single file
            process_single_file(file_path, search_term_strings, context_size_lines,
                                results_queue, min_score, and_search, exact_matches_only)

    # Start worker threads
    for _ in range(cpu_count):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)

    # Process large files separately
    for file_path in large_files:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in ['.csv', '.tsv']:
            thread = threading.Thread(target=search_structured_data, args=(
                file_path, f"{file_path}.offsets", f"{file_path}.metadata",
                search_term_strings, results_queue
            ))
            thread.start()
            threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results
    all_results = []
    while not results_queue.empty():
        all_results.extend(results_queue.get())

    # Sort and return results
    return sorted(all_results, key=lambda x: x.common.score, reverse=True)


def process_single_file(file_path, search_term_strings, context_size_lines, results_queue, min_score, and_search, exact_matches_only):
    """Process a single file and put results in the queue"""
    try:
        content, line_indices, is_utf8 = extract_file_content(file_path)

        if exact_matches_only:
            search_terms = parse_search_terms(search_term_strings)
        else:
            search_terms = preprocess_search_term_list(search_term_strings)

        file_matches, score = search_file(content, search_terms, min_score)

        if len(file_matches) == 0:
            return

        results, _ = build_search_result_for_file(
            file_path, file_matches, score, line_indices, content,
            context_size_lines, and_search, len(search_terms), {}
        )

        results_queue.put(results)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
