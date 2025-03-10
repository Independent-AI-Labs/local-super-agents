import concurrent
import hashlib
import os
import platform
import re
import subprocess
import sys
import time

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Tuple

import psutil

from retrieval.hype.search.file_search import bulk_search_files, search_structured_data, search_structured_data_batched
from retrieval.hype.indexing.structured_data_mapper import split_structured_data_file, index_structured_data_batched, DEFAULT_ITEM_BREAK_SEQUENCE_STR

from retrieval.hype.util.search_util import extract_headers, compile_patterns
from retrieval.tests.util import deploy_test_resources, check_system_specs, print_search_results

if not os.path.exists(os.path.join("res", "test_resources")):
    deploy_test_resources()

sys.set_int_max_str_digits(128000)


def main():
    specs = check_system_specs()

    structured_data_file = os.path.join("res", "test_resources", "user_reviews.csv")
    structured_data_file_partial = os.path.join("res", "test_resources", "user_reviews_partial.csv")
    metadata_dir = os.path.join("res", "metadata")
    file_metadata_dir = os.path.join(metadata_dir, hashlib.sha256(structured_data_file.encode("utf-8")).hexdigest())
    partial_metadata_dir = os.path.join(metadata_dir, hashlib.sha256(structured_data_file_partial.encode("utf-8")).hexdigest())
    bulk_search_dir = os.path.join("res", "test_resources", "eu_energy_law_sample", )

    # # Remove any metadata from previous runs.
    # if os.path.exists(metadata_dir):
    #     shutil.rmtree(metadata_dir, ignore_errors=False, onerror=None)

    if not os.path.exists(metadata_dir):
        # Hardcoded message for the initial test data.
        DEFAULT_LOGGER.log_debug("First run / initial setup required.\n\nStarted indexing of ~14GB (CSV) data / >70M unique items (rows).\n")
        index_data_file(structured_data_file, metadata_dir)
        index_data_file(structured_data_file_partial, metadata_dir)

    score = 0

    print("Running tests...\n")

    structured_throughput, structured_summary = search_structured(
        structured_data_file, metadata_dir, file_metadata_dir, structured_data_file_partial, partial_metadata_dir)
    score += structured_throughput

    bulk_throughput, bulk_summary = search_bulk(bulk_search_dir)
    score += bulk_throughput

    score_msg = f"\n🏅 PERFORMANCE SCORE: {round(score, 1)}"

    with open(os.path.join("res", f"results_{time.time()}.txt"), 'w', encoding="utf-8") as f:
        f.write(f"{specs}{structured_summary}{bulk_summary}{score_msg}")

    print(score_msg)


# If applicable, this pattern needs to be known beforehand, or inferred during runtime using an LLM (WIP).
ITEM_START_PATTERN_STR = '"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",'


def index_data_file(data_file: str, metadata_dir: str):
    start_time = time.time()

    # TODO Compiled patterns can be cached and re-used.
    string_pattern, binary_pattern = compile_patterns(ITEM_START_PATTERN_STR, DEFAULT_ITEM_BREAK_SEQUENCE_STR)

    # Calculate default number of file blocks.
    block_offsets = split_structured_data_file(data_file, metadata_dir, binary_pattern)
    split_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"File split into {len(block_offsets)} blocks in {split_time:.2f} ms.")

    index_structured_data_batched(data_file, block_offsets, metadata_dir, string_pattern)

    print("\n========================================\n")


def search_structured(data_file: str, metadata_dir: str, file_metadata_dir: str, partial_data_file: str = None, partial_metadata_dir: str = None) -> Tuple[
    float, str]:
    # Misspelled on purpose.
    search_term_strings = ["Bulgraia"]

    start_time = time.time()

    # We don't need the entire file to benchmark the single-core search.
    if partial_data_file:
        single_search_results = search_structured_data(partial_data_file, metadata_dir, search_term_strings, min_score=1)
    else:
        single_search_results = search_structured_data(data_file, metadata_dir, search_term_strings, min_score=1)

    single_search_time = (time.time() - start_time)

    start_time = time.time()

    # Only large data files (>10GB) significantly benefit from the batched multi-manage search method.
    # Expect gains of about 500-600% at 10x parallelization (10 physical cores).
    # Maximum efficiency can be achieved with greater (>50GB) and streaming (WIP) workloads.
    # Despite this entire module being one big search optimization itself, performance tuning is very much still an ongoing thing.
    # TODO Spawning C processes or utilizing threads with shared memory has a lower overhead. That is a possible optimization route.
    #  That will however require a lot of code to be translated to Cython / C++... Free-threaded Python seems like a good alternative (still untested however).
    #  In addition, the gains due to process creation overhead reduction will not matter when operating in a worker process.
    #  A huey / redis multi-process implementation is currently WIP.
    search_results = search_structured_data_batched(data_file, metadata_dir, search_term_strings, min_score=1)

    search_time = (time.time() - start_time) + 1  # Convert to milliseconds

    print()

    print_search_results(single_search_results, 10, extract_headers(data_file))

    # assert len(single_search_results) == len(search_results)

    single_timing_msg = f"🏁 Partial single-core structured data search completed in {single_search_time * 1000:.2f} ms."
    multi_timing_msg = f"🏁 Multi-core structured data search completed in {search_time * 1000:.2f} ms."

    print(single_timing_msg)
    print(multi_timing_msg)
    print(f"🏁 {len(single_search_results) + len(search_results)} results total.")

    throughput_single_mb, single_summary = calculate_structured_search_throughput(partial_data_file, partial_metadata_dir, single_search_time, 1)

    total_cores = psutil.cpu_count(logical=False)
    throughput_mb, multi_summary = calculate_structured_search_throughput(data_file, file_metadata_dir, search_time, total_cores)

    print("\n========================================\n")

    total_throughput = throughput_single_mb + throughput_mb
    summary = f"\n{single_timing_msg}\n{multi_timing_msg}\n{single_summary}\n{multi_summary}"

    return total_throughput, summary


def single_bulk_search(search_dir):
    search_terms = [
        "LNG facility",
        "Bulgaria energy",
        "natural gas supply",
        "renewable infrastructure",
        "smart energy systems",
        "grid modernization",
        "battery storage",
        "energy resilience",
        "sustainability impact",
        "climate finance"
    ]
    search_results = bulk_search_files(search_dir, search_terms, min_score=128, and_search=False)
    return search_results


def search_bulk(
        bulk_search_dir: str,
        max_procs: int = psutil.cpu_count(logical=False),
        total_procs: int = psutil.cpu_count(logical=False)
) -> Tuple[float, str]:
    start_time = time.time()

    single_bulk_search(bulk_search_dir)

    single_search_time = (time.time() - start_time)  # Convert to milliseconds.

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        future_to_search = {executor.submit(single_bulk_search, bulk_search_dir): _ for _ in range(total_procs)}

        all_results = []
        first_result_printed = False  # Flag to track the first result.
        for future in concurrent.futures.as_completed(future_to_search):
            search_results = future.result()
            all_results.extend(search_results)

            # Only print on the first search, as we're searching the same dataset over and over again.
            # TODO Add variable search terms.
            if not first_result_printed:
                print_search_results(all_results)
                first_result_printed = True

    search_time = (time.time() - start_time)  # Convert to milliseconds
    single_timing_msg = f"🏁 File search completed in {single_search_time * 1000:.2f} s."
    multi_timing_msg = f"🏁 Bulk file search completed in {search_time * 1000:.2f} s.\n"

    print(single_timing_msg)
    print(multi_timing_msg)
    print(f"🏁 {len(all_results)} results total.")

    throughput_single_mb, single_summary = calculate_bulk_search_throughput(bulk_search_dir, single_search_time, 1)
    throughput_mb, multi_summary = calculate_bulk_search_throughput(bulk_search_dir, search_time, total_procs)

    # benchmark_bulk_search(root_directory, search_terms, min_matches=16, and_search=True)

    throughput = throughput_single_mb + throughput_mb

    summary = f"\n\n{single_timing_msg}\n{multi_timing_msg}\n{single_summary}\n{multi_summary}"

    return throughput, summary


def calculate_structured_search_throughput(
        data_file: str,
        file_metadata_dir: str,
        search_time: float,
        total_procs: int = 1
) -> Tuple[float, str]:
    throughput = (os.path.getsize(data_file) / 1024 / 1024) / search_time

    with open(os.path.join(file_metadata_dir, "offsets.csv"), 'r', 'utf-8') as file:
        last_line = file.readlines()[-1]
        last_item_index_offset = int(last_line.split(',')[-1])

    # Pattern to match the item metadata files.
    metadata_pattern = re.compile(r'metadata_(\d+)\.csv')

    # Get all files in the directory and filter those matching the metadata pattern.
    metadata_files = [
        filename for filename in os.listdir(file_metadata_dir)
        if metadata_pattern.match(filename)
    ]

    # Sort the metadata files based on the number in their filename.
    metadata_files.sort(key=lambda x: int(metadata_pattern.search(x).group(1)))

    with open(os.path.join(file_metadata_dir, metadata_files[-1]), 'r', 'utf-8') as file:
        last_line = file.readlines()[-1]
        last_item_index = int(last_line.split(',')[-1])

    row_throughput = (last_item_index + last_item_index_offset) / search_time
    summary = ""

    if total_procs == 1:
        single_raw_throughput_msg = f"\n🏁 SINGLE-CORE RAW DATA THROUGHPUT >>> {round(throughput, 2)} MB/s"
        summary += single_raw_throughput_msg
        print(single_raw_throughput_msg)

        single_item_throughput_msg = f"🏁 SINGLE-CORE ITEM/ROW THROUGHPUT >>> {round(row_throughput, 2)} Items/s"
        summary += "\n" + single_item_throughput_msg
        print(single_item_throughput_msg)
    else:
        multi_raw_throughput_msg = f"\n🏁 ACHIEVED ALL PHY-CORE ({total_procs} PROCS) RAW DATA THROUGHPUT >>> {round(throughput, 2)} MB/s"
        summary += "\n" + multi_raw_throughput_msg
        print(multi_raw_throughput_msg)

        multi_raw_throughput_msg = f"🏁 ACHIEVED ALL PHY-CORE ({total_procs} PROCS) ITEM/ROW THROUGHPUT >>> {round(row_throughput, 2)} Items/s"
        summary += "\n" + multi_raw_throughput_msg
        print(multi_raw_throughput_msg)

    return throughput, summary


def calculate_bulk_search_throughput(
        bulk_search_dir: str,
        search_time: float,
        total_procs: int = 1
) -> Tuple[float, str]:
    throughput_single = (sum(file.stat().st_size for file in Path(bulk_search_dir).rglob('*')) / 1024 / 1024) / search_time
    file_throughput_single = len(list(Path(bulk_search_dir).rglob('*'))) / search_time

    throughput_multi = (sum(file.stat().st_size for file in Path(bulk_search_dir).rglob('*')) / 1024 / 1024 * total_procs) / search_time
    file_throughput_multi = (len(list(Path(bulk_search_dir).rglob('*'))) * total_procs) / search_time

    summary = ""

    if total_procs == 1:
        single_raw_throughput_msg = f"\n🏁 SINGLE-CORE RAW DATA THROUGHPUT >>> {round(throughput_single, 2)} MB/s"
        single_doc_throughput_msg = f"🏁 SINGLE-CORE DOCUMENT THROUGHPUT >>> {round(file_throughput_single, 2)} Documents/s"

        summary += single_raw_throughput_msg
        summary += "\n" + single_doc_throughput_msg

        print(single_raw_throughput_msg)
        print(single_doc_throughput_msg)

        return throughput_single, summary
    else:
        multi_raw_throughput_msg = (f"\n🏁 ALL PHY-CORE ({total_procs} PROCS) RAW DATA THROUGHPUT >>> {round(throughput_multi, 2)} MB/s "
                                    f"(SINGLE-CORE AVG: {round(throughput_single, 2)} MB/s)")
        multi_doc_throughput_msg = (f"🏁 ALL PHY-CORE ({total_procs} PROCS) DOCUMENT THROUGHPUT >>> {round(file_throughput_multi, 2)} Documents/s "
                                    f"(SINGLE-CORE AVG: {round(file_throughput_single, 2)} Documents/s)\n")

        summary += "\n" + multi_raw_throughput_msg
        summary += "\n" + multi_doc_throughput_msg

        print(multi_raw_throughput_msg)
        print(multi_doc_throughput_msg)

        return throughput_multi, summary


def benchmark_bulk_search(root_directory: str, search_terms: List[str], min_matches: int, and_search: bool,
                          num_runs: int = 4) -> Dict[str, Any]:
    total_time = 0
    all_results = []

    for i in range(num_runs):
        start_time = time.time()
        results = bulk_search_files(root_directory, search_terms, min_score=min_matches, and_search=and_search)
        end_time = time.time()

        run_time = end_time - start_time
        total_time += run_time
        all_results.append(results)

        print(f"\nRun {i + 1}: {run_time:.4f} seconds")

    average_time = total_time / num_runs
    print(f"\nAverage time over {num_runs} runs: {average_time:.4f} seconds")

    return {
        "average_time": average_time,
        "all_results": all_results,
        "num_runs": num_runs
    }


if __name__ == "__main__":
    main()
