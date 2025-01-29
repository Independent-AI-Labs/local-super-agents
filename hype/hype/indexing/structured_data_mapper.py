import csv
import hashlib
import os
import gc
import re
import mmap
import string
import time

from multiprocessing import Process

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from threading import Thread

from typing import List, Tuple, Dict

import psutil

from hype.hype.util.search_util import build_automaton_from_strings, simple_aho_corasick_match, get_pattern_match_length

DEFAULT_ITEM_BREAK_SEQUENCE_STR = '\r\n'


def get_metadata_for_position(metadata_filepath: str, position: int) -> Tuple[int, int, int]:
    """
    Retrieve item metadata for a given character position.

    :param metadata_filepath: Path to the metadata file.
    :param position: The character position to locate.
    :return: A tuple containing the index of the row, start index, and end index of the matched row.
    """
    with open(metadata_filepath, 'r+b') as f:
        # Memory-map the file, size 0 means whole file
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return find_item_metadata(mm, position)


def find_item_metadata(memory_map: mmap.mmap, position: int) -> Tuple[int, int, int]:
    """
    Find the item metadata for a given character position using a memory-mapped file with fixed-size rows.

    :param memory_map: Memory-mapped file object for the metadata.
    :param position: The character position to locate.
    :return: A tuple containing (index, start, end) of the row containing the position.
    """
    # Determine row size from the first line
    first_newline = memory_map.find(b'\n')
    if first_newline == -1:
        return -1, -1, -1  # No newline found, invalid format

    row_size = first_newline + 1  # Include the newline character in the row size

    total_rows = memory_map.size() // row_size
    left, right = 0, total_rows - 1

    while left <= right:
        mid = (left + right) // 2
        row_start = mid * row_size

        # Read the fixed-size row
        row_data = memory_map[row_start:row_start + row_size].decode('utf-8').strip()
        start, end, index = map(int, row_data.split(','))

        if start <= position <= end:
            return index, start, end
        elif position < start:
            right = mid - 1
        else:
            left = mid + 1

    return -1, -1, -1


def detect_line_break_sequence(sample_lines: List[str], default_sequence: str = '\r\n\"') -> str:
    """
    Detect the row break sequence from a sample of lines.

    :param sample_lines: A list of lines from the CSV to analyze.
    :param default_sequence: The default row break sequence to use if none is found.
    :return: The detected row break sequence.
    """
    # Check the sample lines for the common row break sequence
    for line in sample_lines:
        if default_sequence in line:
            return default_sequence
    # If no sequence is found, return the default
    return default_sequence


PATTERN_CACHE: Dict[str, Tuple[re.Pattern, re.Pattern]] = {}


def calculate_total_blocks(file_size: int, available_memory: int, sub_blocks: int = os.cpu_count() // 2) -> int:
    """
    Calculate the total number of blocks based on file size, available memory, and number of physical CPUs.

    :param file_size: Size of the file in bytes.
    :param available_memory: Amount of available memory in bytes.
    :param sub_blocks: How many sub-blocks to produce per individual block.

    In order to reduce the maximum memory footprint of certain operations, such as data indexing, we need to carefully
    segment the file. This aims to strike a careful balance during parallel processing.

    You can experiment with setting the sub_blocks parameter to higher or lower values as many factors can influence
    the end performance:
     - Physical / Logical CPUs ratio
     - Individual time spent per block vs. the overhead of creating more processes

     More physical cores are always beneficial, but having a lot of cores in general will always have the most impact on
     file processing speed.

    Some systems benefit from heavy parallelization more than others.
    Having more blocks can also benefit distributed applications.

    :return: Total number of blocks to process.
    """
    block_load_memory_limit = available_memory // sub_blocks
    total_blocks = max(1, file_size // block_load_memory_limit)

    return (total_blocks * sub_blocks) - 1


def split_structured_data_file(
        file_path: str,
        metadata_dir: str,
        item_start_validation_pattern_str: re.Pattern = None,
        free_memory_usage_limit: float = .75
) -> List[Tuple[int, int]]:
    """
    Find byte offsets for each block in the file by matching the UID pattern.

    :param file_path: Path to the file to split.
    :param metadata_dir: Path to the metadata directory.
    :param item_start_validation_pattern_str: Pattern to identify the start of each item.
    :param free_memory_usage_limit: Fraction of available memory to use.
    :return: List of tuples, each containing start and end offsets for a block.
    """
    file_size = os.path.getsize(file_path)
    available_memory = int(psutil.virtual_memory().available * free_memory_usage_limit)

    total_blocks = calculate_total_blocks(file_size, available_memory)
    block_size = file_size // total_blocks
    offsets = []

    # Full path to the file becomes the UUID for the metadata subdir.
    metadata_dir = os.path.join(metadata_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())

    os.makedirs(metadata_dir, exist_ok=True)

    with open(file_path, 'rb') as file:
        # Memory-map the file for efficient access
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Check for header
            start_offset = 0

            lines = ""

            for i in range(32):
                lines = f"{lines}{mm.readline()}"

            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(lines)

            mm.seek(start_offset)

            if has_header:
                header_length = len(mm.readline().decode('utf-8'))
                mm.seek(start_offset)

                start_offset = header_length
                print(f"Header found. Starting from byte offset: {start_offset}")

            current_position = start_offset

            while current_position < file_size:
                # Check if the remaining bytes are less than a full block size
                remaining_bytes = file_size - current_position
                if remaining_bytes <= block_size:
                    # Write the last offset and conclude the loop
                    offsets.append((current_position, file_size))
                    break

                # Guess where the block row breaks would be
                end_position = min(current_position + block_size, file_size)

                # Initialize search window
                search_radius = 1024 * 1024  # 1 MB
                match = None

                while not match and end_position < file_size:
                    search_start = max(current_position, end_position - search_radius)
                    search_end = min(file_size, end_position + search_radius)

                    # Read the search window
                    mm.seek(search_start)
                    search_window = mm.read(search_end - search_start)

                    # Search for the UID pattern in the search window
                    match = item_start_validation_pattern_str.search(search_window)

                    if not match:
                        # Increase search window
                        search_radius *= 2
                        end_position = min(file_size, end_position + block_size)

                if match:
                    # Calculate the actual byte positions
                    match_start = search_start + match.start()

                    # Store the offsets for the block
                    offsets.append((current_position, match_start))
                    # print(f"Valid block end found at byte offset: {match_start}")

                    # Update the current position to start after the match
                    current_position = match_start + len(DEFAULT_ITEM_BREAK_SEQUENCE_STR)
                else:
                    # If no valid match is found, move to the end of the file
                    offsets.append((current_position, file_size))
                    print(f"No valid UID found after byte offset: {current_position}")
                    break

    # Save the offsets to a CSV file
    with open(os.path.join(metadata_dir, "offsets.csv"), "w", encoding="utf-8") as metadata_file:
        for i, offset in enumerate(offsets):
            if i == len(offsets) - 1:
                metadata_file.write(f"{offset[0]},{offset[1]}")
            else:
                metadata_file.write(f"{offset[0]},{offset[1]}\n")

    return offsets


# Function to manage memory and spawn processes
def index_structured_data(file_path: str, block_offsets: list, metadata_output_dir: str, item_start_validation_pattern_str: re.Pattern = None):
    """
    Process CSV chunks while managing memory usage to avoid exceeding the specified RAM threshold.

    :param file_path: Path to the structured data file.
    :param block_offsets: List of tuples (start_offset, end_offset) for each chunk.
    :param metadata_output_dir: Directory to store metadata output files.
    """
    # Full path to the file becomes the UUID for the metadata subdir.
    metadata_output_dir = os.path.join(metadata_output_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())

    procs = prepare_structured_data_indexing_procs(file_path, block_offsets, metadata_output_dir, item_start_validation_pattern_str)
    execute_memory_managed_block_processing(block_offsets, procs)

    update_offsets(file_path, metadata_output_dir)


def index_structured_data_threaded(file_path: str, block_offsets: list, metadata_output_dir: str, item_start_validation_pattern_str: re.Pattern = None):
    """
    Process CSV chunks while managing memory usage to avoid exceeding the specified RAM threshold.

    :param file_path: Path to the structured data file.
    :param block_offsets: List of tuples (start_offset, end_offset) for each chunk.
    :param metadata_output_dir: Directory to store metadata output files.
    """
    # Full path to the file becomes the UUID for the metadata subdir.
    metadata_output_dir = os.path.join(metadata_output_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())

    procs = prepare_structured_data_indexing_threads(file_path, block_offsets, metadata_output_dir, item_start_validation_pattern_str)
    execute_memory_managed_block_processing_threads(block_offsets, procs)

    update_offsets(file_path, metadata_output_dir)


def index_structured_data_batched(
        file_path: str,
        block_offsets: list,
        metadata_output_dir: str,
        item_start_validation_pattern_str: re.Pattern = None,
        free_memory_usage_limit: float = 0.75
):
    """
    Process CSV chunks while managing memory usage to avoid exceeding the specified RAM threshold.

    :param file_path: Path to the CSV file.
    :param block_offsets: List of tuples (start_offset, end_offset) for each chunk.
    :param metadata_output_dir: Directory to store metadata output files.
    """
    # Chunk the block offsets to create fewer, longer-running processes
    num_physical_cores = psutil.cpu_count(logical=False)
    if len(block_offsets) / 2 > num_physical_cores:
        chunk_size = len(block_offsets) // num_physical_cores
    else:
        # Revert back to the threaded implementation as the performance gains will not be significant enough to overcome the
        # process creation overhead...
        print("Small file. Reverting back to non-batched indexing...")
        return index_structured_data(file_path, block_offsets, metadata_output_dir, item_start_validation_pattern_str)

    block_offset_chunks = [block_offsets[i:i + chunk_size] for i in range(0, len(block_offsets), chunk_size)]

    # Execute using memory-managed block processing
    batch_index_blocks(
        file_path,
        block_offset_chunks,
        metadata_output_dir,
        item_start_validation_pattern_str,
        free_memory_usage_limit,
    )

    # Once processing is complete, update the offsets
    update_offsets(file_path, os.path.join(metadata_output_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest()))


def batch_index_blocks(
        file_path: str,
        block_offset_chunks: List[List[Tuple[int, int]]],
        metadata_dir: str,
        item_start_validation_pattern: re.Pattern,
        free_memory_usage_limit: float = 0.75,
        memory_overhead_factor: int = 1
) -> None:
    """
    A hybrid memory-managed block processing function using both process pooling
    and memory checks to manage large tasks efficiently.

    :param block_offset_chunks: List of tuples containing start and end offsets for blocks.
    :param free_memory_usage_limit: Fraction of free memory to consider safe for new task submission.
    :param memory_overhead_factor: Memory overhead factor for task estimation.
    :param batch_size: Number of completed processes to wait for before submitting new tasks.
    """
    start_time = time.time()
    num_physical_cores = psutil.cpu_count(logical=False)
    max_procs = num_physical_cores

    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        futures = []
        last_submitted = 0

        # Ensure no more processes are created than there are physical cores
        while last_submitted < len(block_offset_chunks):
            # Submit tasks only if there are free CPUs
            while len(futures) < max_procs and last_submitted < len(block_offset_chunks):
                offsets_range = block_offset_chunks[last_submitted]

                start = offsets_range[0][0]
                end = offsets_range[-1][1]

                # Check memory before starting a new process
                required_memory = (end - start) * memory_overhead_factor
                available_memory = psutil.virtual_memory().available * free_memory_usage_limit

                if required_memory > available_memory:
                    print(f"Waiting for memory to free up... Required: {required_memory}, Available: {available_memory}")
                    time.sleep(0.1)  # Wait and re-check memory availability
                    continue  # Skip to the next iteration of the outer loop

                # Submit task to the executor
                # print(f"Starting process {last_submitted}... Available memory: {available_memory}")
                future = executor.submit(
                    batch_index,
                    file_path,
                    block_offset_chunks[last_submitted],
                    metadata_dir,
                    item_start_validation_pattern,
                    last_submitted * len(block_offset_chunks[0])
                )
                futures.append(future)
                last_submitted += 1

            # Wait for at least one task to complete before submitting more
            if futures:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.remove(future)

        # Wait for all remaining futures to complete
        for future in futures:
            future.result()

    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"{sum([len(blocks) for blocks in block_offset_chunks])} blocks processed in {process_time:.2f} ms.\n")


def batch_index(
        file_path: str,
        block_offset_chunk: List[Tuple[int, int]],
        metadata_dir: str,
        item_start_validation_pattern: re.Pattern,
        offset_offset: int
) -> None:
    """
    Function to handle indexing within a process, processing each chunk sequentially.
    """
    file_metadata_dir = os.path.join(metadata_dir, hashlib.sha256(file_path.encode("utf-8")).hexdigest())

    for i, (start, end) in enumerate(block_offset_chunk):
        index_structured_data_file(
            file_path,
            start,
            end,
            i + offset_offset,
            file_metadata_dir,
            DEFAULT_ITEM_BREAK_SEQUENCE_STR,
            item_start_validation_pattern
        )


def update_offsets(file_path: str, metadata_output_dir: str):
    """
    Update the offsets file based on the metadata files generated during processing.

    :param file_path: The path to the original file being processed.
    :param metadata_output_dir: The directory where metadata files are stored.
    """
    # Pattern to match the metadata files
    metadata_pattern = re.compile(r'metadata_(\d+)\.csv')

    # List to store the last integers found in the metadata files
    last_integers = []

    # Get all files in the directory and filter those matching the metadata pattern
    metadata_files = [
        filename for filename in os.listdir(metadata_output_dir)
        if metadata_pattern.match(filename)
    ]

    # Sort the metadata files based on the number in their filename
    metadata_files.sort(key=lambda x: int(metadata_pattern.search(x).group(1)))

    # Iterate over sorted metadata files to collect last integers
    for filename in metadata_files:
        metadata_file_path = os.path.join(metadata_output_dir, filename)
        with open(metadata_file_path, 'r') as file:
            last_line = file.readlines()[-1]
            last_integer = int(last_line.split(',')[-1])  # Get the last integer from the last line
            last_integers.append(last_integer)

    # Calculate cumulative offsets
    cumulative_offsets = []
    current_offset = 0
    for last_integer in last_integers:
        cumulative_offsets.append(current_offset)
        current_offset += last_integer

    # Full path to the file becomes the UUID for the metadata subdir.
    # metadata_subdir = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    # metadata_output_dir = os.path.join(metadata_output_dir, metadata_subdir)

    # Read the current offsets.csv
    offsets_file_path = os.path.join(metadata_output_dir, 'offsets.csv')
    offsets_data = []

    if os.path.exists(offsets_file_path):
        with open(offsets_file_path, 'r') as file:
            reader = csv.reader(file)
            offsets_data = [row for row in reader]

    # Update the offsets.csv with new offsets
    new_offsets_data = []
    for i, row in enumerate(offsets_data):
        if i < len(cumulative_offsets):
            # Prepare a new row with three elements: start, end, and cumulative offset
            new_row = row[:2] + [str(cumulative_offsets[i])]
        else:
            # If there's no cumulative offset, just retain the existing row structure
            new_row = row
        new_offsets_data.append(new_row)

    # Print new_offsets_data for debugging purposes
    print(new_offsets_data)

    # Write the updated offsets data back to offsets.csv
    with open(offsets_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_offsets_data)

    print("Offsets file updated successfully.")


def is_valid_header(header: list) -> bool:
    """
    Check if a given list of strings represents a valid CSV header.

    :param header: List of header strings.
    :return: True if valid, False otherwise.
    """
    valid_chars = set(string.ascii_letters)  # Allow only alphabetic characters

    for field in header:
        # Check if all characters in the field are alphabetic
        if not all(char in valid_chars for char in field):
            return False
        # Check for spaces
        if ' ' in field:
            return False
        # Check field length
        if len(field) > 255:
            return False

    return True


def prepare_structured_data_indexing_procs(data_file: str, block_offsets: list, metadata_output_dir: str,
                                           item_start_validation_pattern_str: re.Pattern = None) -> list:
    """
    Check if the first line is a header and start processes for processing CSV blocks.

    :param data_file: Path to the CSV file.
    :param block_offsets: List of tuples containing start and end offsets for each block.
    :param metadata_output_dir: Directory to store metadata files.
    :return: List of created processes.
    """
    procs = []

    # Start a new process for each block
    for i, (start, end) in enumerate(block_offsets):
        procs.append(Process(target=index_structured_data_file,
                             args=(data_file, start, end, i, metadata_output_dir, DEFAULT_ITEM_BREAK_SEQUENCE_STR, item_start_validation_pattern_str)))

    return procs


def prepare_structured_data_indexing_threads(data_file: str, block_offsets: list, metadata_output_dir: str,
                                             item_start_validation_pattern_str: re.Pattern = None) -> list:
    """
    Check if the first line is a header and start processes for processing CSV blocks.

    :param data_file: Path to the CSV file.
    :param block_offsets: List of tuples containing start and end offsets for each block.
    :param metadata_output_dir: Directory to store metadata files.
    :return: List of created processes.
    """
    threads = []

    # Start a new process for each block
    for i, (start, end) in enumerate(block_offsets):
        threads.append(Thread(target=index_structured_data_file,
                              args=(data_file, start, end, i, metadata_output_dir, DEFAULT_ITEM_BREAK_SEQUENCE_STR, item_start_validation_pattern_str)))

    return threads


def index_structured_data_file(
        file_path: str,
        start_offset: int,
        end_offset: int,
        block_id: int,
        output_dir: str = "metadata_chunks",
        item_break_sequence: str = DEFAULT_ITEM_BREAK_SEQUENCE_STR,
        item_start_validation_pattern: re.Pattern = None,
        write_buffer_size_bytes: int = 1024 * 1024
) -> None:
    """
    Process a chunk of the file and generate metadata for each row, writing results to a file.

    :param file_path: Path to the structured data file.
    :param start_offset: The starting byte offset in the file.
    :param end_offset: The ending byte offset in the file.
    :param block_id: An identifier for the chunk being processed.
    :param output_dir: Directory to store metadata files.
    :param item_break_sequence: The sequence used to detect potential item ends.
    :param item_start_validation_pattern: The regex pattern used to validate item starts in cases where item break
    sequences not always represent item ends.
    :param write_buffer_size_bytes: The size of the batch for writing metadata lines.
    """

    # Record the total start time
    total_start_time = time.time()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare metadata file
    metadata_filename = os.path.join(output_dir, f"metadata_{block_id}.csv")
    with open(metadata_filename, "w", encoding="utf-8") as metadata_file:
        with open(file_path, 'r+b') as f:
            # Memory-map the file
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Read the block from the file
            mm.seek(start_offset)

            block_data_bytes = mm.read(end_offset - start_offset)

            # Decode the block data to a string for processing
            block_data_string = block_data_bytes.decode('utf-8', errors='ignore')

            del block_data_bytes
            gc.collect()

            # Build the automaton for detecting item breaks.
            automaton = build_automaton_from_strings([item_break_sequence])

            # Use Aho-Corasick to find all matches of the item break sequence.
            potential_item_breaks = simple_aho_corasick_match(automaton, block_data_string)

            # Start timing UID pattern matching.
            uid_start_time = time.time()

            # Process each potential row break to find valid rows
            last_valid_start = 0
            batch = []

            total_items = 0

            for index, break_position in enumerate(potential_item_breaks):
                # Check if the match is followed by a valid item start sequence.
                next_start_index_candidate = break_position + len(item_break_sequence)

                if item_start_validation_pattern is not None:
                    next_string_start = block_data_string[
                                        next_start_index_candidate - 2:next_start_index_candidate + get_pattern_match_length(
                                            item_start_validation_pattern)]

                if item_start_validation_pattern is None or item_start_validation_pattern.match(next_string_start):
                    total_items += 1

                    # Valid row break found, add to batch
                    start_index = last_valid_start
                    end_index = break_position

                    # print(f"====== {block_data_string[end_index - 32:end_index]}")

                    # TODO This needs to be replaced with a user-configurable fixed-size metadata parameter list.
                    # Currently it is hardcoded to annotate the start and end string indices of items inside
                    # structured data files, as well as the sequential item number / index inside the data structure.
                    batch.append(
                        f"{str(start_index).zfill(12)},{str(end_index).zfill(12)},{str(total_items).zfill(9)}\n")
                    last_valid_start = end_index + len(item_break_sequence)
                    # print(block_data_string[start_index:end_index])

                    # Write batch if it reaches the batch size limit
                    if len(batch) >= write_buffer_size_bytes:
                        metadata_file.writelines(batch)
                        batch = []

            # Handle the last row in the block
            if last_valid_start < len(block_data_string):
                start_index = last_valid_start
                end_index = len(block_data_string)
                batch.append(f"{str(start_index).zfill(12)},{str(end_index).zfill(12)},{str(total_items).zfill(9)}\n")

            # Record the UID matching time
            uid_matching_time = time.time() - uid_start_time

            # Write any remaining lines in the batch
            if batch:
                metadata_file.writelines(batch)

            # Clean up to free memory
            del block_data_string
            gc.collect()

            # Unmap the file
            mm.close()

    # Calculate total processing time
    total_processing_time = time.time() - total_start_time

    # Calculate the percentage of time spent on UID matching
    uid_matching_percentage = (uid_matching_time / total_processing_time) * 100

    # print(f"Processed block {block_id} from {start_offset} to {end_offset}")
    # print(f"UID matching took {uid_matching_percentage:.2f}% of the total processing time.")


def execute_memory_managed_block_processing(
        block_offsets: list,
        procs: List[Process],
        free_memory_usage_limit: float = .75,
        memory_overhead_factor: int = 1
):
    max_procs = psutil.cpu_count(logical=False)

    # Measure the time to process each chunk
    start_time = time.time()

    running_procs = 0
    initial_load = 0
    initial_load_index = len(block_offsets)

    last_joined = 0

    for i, (start, end) in enumerate(block_offsets):
        if initial_load + (end - start) * memory_overhead_factor < psutil.virtual_memory().available * free_memory_usage_limit:
            initial_load += (end - start) * memory_overhead_factor
        else:
            initial_load_index = i
            break

    for i, (start, end) in enumerate(block_offsets):
        if running_procs >= max_procs:
            joined_count = 0
            for pix, process in enumerate(procs[last_joined:i]):
                process.join()
                last_joined = pix
                running_procs -= 1
                joined_count += 1

        if i > initial_load_index or (end - start) * memory_overhead_factor > psutil.virtual_memory().available * free_memory_usage_limit:
            print(f"{i} waiting for free memory...")

            if i == initial_load_index + 1:
                # Wait for a bit, as we need an accurate RAM reading.
                time.sleep(1)

            # Wait for available memory if the current block can't fit.
            while (end - start) * memory_overhead_factor > psutil.virtual_memory().available * free_memory_usage_limit:
                time.sleep(0.1)

        # print(f"Starting process {i}... {psutil.virtual_memory().available}")
        procs[i].start()
        running_procs += 1

    # Wait for all subprocesses to finish
    for process in procs:
        process.join()

    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"{len(block_offsets)} blocks processed in {process_time:.2f} ms.\n")


def execute_memory_managed_block_processing_threads(
        block_offsets: list,
        threads: List[Thread],
        free_memory_usage_limit: float = .75
):
    # Measure the time to process each chunk
    start_time = time.time()

    initial_load = 0
    initial_load_index = len(block_offsets)

    # print("Calculating initial load...")
    for i, (start, end) in enumerate(block_offsets):
        if initial_load + end - start < psutil.virtual_memory().available * free_memory_usage_limit:
            initial_load += end - start
        else:
            initial_load_index = i
            break

    for i, (start, end) in enumerate(block_offsets):
        if i > initial_load_index or (end - start) > psutil.virtual_memory().available * free_memory_usage_limit:

            if i == initial_load_index + 1:
                # Wait for a bit, as we need an accurate RAM reading.
                time.sleep(1)

            # Wait for available memory if the current block can't fit.
            while (end - start) > psutil.virtual_memory().available * free_memory_usage_limit:
                time.sleep(0.1)

        # print(f"Starting thread {i}")
        threads[i].start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"{len(threads)} blocks processed in {process_time:.2f} ms.")
