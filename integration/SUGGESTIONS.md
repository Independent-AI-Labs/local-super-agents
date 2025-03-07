# Search System Optimization Guide

This comprehensive guide outlines practical improvements to optimize the performance, reliability, and efficiency of your web and file search systems. Each section includes detailed code examples and implementation steps.

## Table of Contents

1. [Memory Management Optimizations](#1-memory-management-optimizations)
2. [Search Algorithm Enhancements](#2-search-algorithm-enhancements)
3. [Parallelization Improvements](#3-parallelization-improvements)
4. [Indexing Strategy Improvements](#4-indexing-strategy-improvements)
5. [Web Scraping Enhancements](#5-web-scraping-enhancements)
6. [Error Handling and Robustness](#6-error-handling-and-robustness)
7. [Configuration and Tuning](#7-configuration-and-tuning)

## 1. Memory Management Optimizations

### 1.1 Memory-Mapped File Pool

Currently, your system creates and destroys memory-mapped objects frequently, which can lead to memory fragmentation. Implementing a reusable pool of memory-mapped objects can significantly reduce overhead.

```python
# Add to file_search.py
class MemoryMappedFilePool:
    def __init__(self, max_pool_size=10):
        self.pool = {}
        self.max_pool_size = max_pool_size
        self.access_times = {}
        self._lock = threading.RLock()

    def get_mmap(self, file_path, offset, size):
        with self._lock:
            key = (file_path, offset, size)
            if key in self.pool:
                mm = self.pool[key]
                self.access_times[key] = time.time()
                return mm
            
            # If pool is full, evict least recently used
            if len(self.pool) >= self.max_pool_size:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                self.pool[lru_key].close()
                del self.pool[lru_key]
                del self.access_times[lru_key]
            
            # Create new mmap
            with open(file_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ, offset=offset)
                self.pool[key] = mm
                self.access_times[key] = time.time()
                return mm
    
    def close_all(self):
        with self._lock:
            for mm in self.pool.values():
                mm.close()
            self.pool.clear()
            self.access_times.clear()

# Global instance
mmap_pool = MemoryMappedFilePool()

# Modify read_mmap in file_util.py to use the pool
def read_mmap(file_path: str, start_offset: int, end_offset: int) -> str:
    """
    Reads a specific portion of a file using a pooled memory mapping.
    """
    mm = mmap_pool.get_mmap(file_path, 0, os.path.getsize(file_path))
    mm.seek(start_offset)
    block_data = mm.read(end_offset - start_offset)
    block_data_string = block_data.decode('utf-8', errors='ignore')
    # Don't close the mmap as it's managed by the pool
    return block_data_string
```

### 1.2 Adaptive Block Size Calculation

Improve the `calculate_total_blocks` function to consider file content density:

```python
def calculate_total_blocks(file_path, file_size, available_memory, sub_blocks=os.cpu_count() // 2):
    """
    Calculate optimal block sizes based on file content sampling.
    """
    # Sample the file to determine content density
    sample_size = min(1024 * 1024, file_size // 10)  # 1MB or 10% of file
    density_factor = 1.0
    
    with open(file_path, 'rb') as f:
        # Take samples from beginning, middle, and end
        samples = []
        f.seek(0)
        samples.append(f.read(sample_size // 3))
        
        f.seek(file_size // 2 - sample_size // 6)
        samples.append(f.read(sample_size // 3))
        
        f.seek(file_size - sample_size // 3)
        samples.append(f.read(sample_size // 3))
    
    # Analyze compression ratio as proxy for content density
    import zlib
    original_size = sum(len(s) for s in samples)
    compressed_size = sum(len(zlib.compress(s)) for s in samples)
    
    # Adjust density factor based on compression ratio
    if original_size > 0:
        compression_ratio = compressed_size / original_size
        if compression_ratio < 0.3:  # Highly compressible = sparse content
            density_factor = 0.7  # Need fewer blocks
        elif compression_ratio > 0.7:  # Not very compressible = dense content
            density_factor = 1.3  # Need more blocks
    
    # Calculate block size using density-adjusted algorithm
    block_load_memory_limit = int(available_memory // (sub_blocks * density_factor))
    total_blocks = max(1, file_size // block_load_memory_limit)
    
    return int((total_blocks * sub_blocks) - 1)
```

### 1.3 Chrome Driver Pooling for Web Scraping

Replace the individual Chrome driver creation with a pool in `surfer.py`:

```python
class ChromeDriverPool:
    def __init__(self, max_drivers=4):
        self.max_drivers = max_drivers
        self.available_drivers = queue.Queue()
        self.used_drivers = set()
        self._lock = threading.RLock()
    
    def get_driver(self):
        with self._lock:
            if not self.available_drivers.empty():
                driver = self.available_drivers.get()
                self.used_drivers.add(driver)
                return driver
            
            if len(self.used_drivers) < self.max_drivers:
                driver = self._create_new_driver()
                self.used_drivers.add(driver)
                return driver
            
            # Wait for a driver to be returned
            while self.available_drivers.empty():
                time.sleep(0.1)
            driver = self.available_drivers.get()
            self.used_drivers.add(driver)
            return driver
    
    def return_driver(self, driver):
        with self._lock:
            if driver in self.used_drivers:
                self.used_drivers.remove(driver)
                # Clear cookies and reset state
                driver.delete_all_cookies()
                driver.execute_script("window.localStorage.clear();")
                driver.execute_script("window.sessionStorage.clear();")
                self.available_drivers.put(driver)
    
    def _create_new_driver(self):
        # Existing Chrome initialization code from init_web_driver()
        options = webdriver.ChromeOptions()
        options.binary_location = CHROME_PATH
        # ... (rest of the options from init_web_driver)
        driver = webdriver.Chrome(options=options)
        return driver
    
    def close_all(self):
        with self._lock:
            while not self.available_drivers.empty():
                driver = self.available_drivers.get()
                driver.quit()
            
            for driver in list(self.used_drivers):
                driver.quit()
            self.used_drivers.clear()

# Initialize the pool once
chrome_driver_pool = ChromeDriverPool()

# Modify scrape_urls to use the pool
def scrape_urls(urls):
    driver = chrome_driver_pool.get_driver()
    try:
        initial_window = driver.current_window_handle
        results = []
        
        # Your existing scraping code...
        
        return results
    finally:
        chrome_driver_pool.return_driver(driver)
```

## 2. Search Algorithm Enhancements

### 2.1 Optimized Aho-Corasick with Fail States

The current Aho-Corasick implementation can be enhanced with proper fail states and suffix links:

```python
class OptimizedAhoCorasick:
    def __init__(self):
        self.root = {}
        self.fail_links = {}
        self.output = {}
        self.finalized = False
    
    def add_pattern(self, pattern, value=None):
        if self.finalized:
            raise ValueError("Cannot add patterns after finalization")
        
        node = self.root
        for char in pattern:
            node = node.setdefault(char, {})
        
        if value is None:
            value = pattern
        self.output[id(node)] = value
    
    def build_fail_links(self):
        """Build the fail links and output links using BFS"""
        queue = []
        # Initialize the fail links for depth 1 nodes
        for char, node in self.root.items():
            self.fail_links[id(node)] = self.root
            queue.append(node)
        
        # BFS to build the rest of the fail links
        while queue:
            current = queue.pop(0)
            
            for char, child in current.items():
                queue.append(child)
                
                # Find the fail link for this child
                fail_state = self.fail_links[id(current)]
                
                while fail_state is not self.root and char not in fail_state:
                    fail_state = self.fail_links[id(fail_state)]
                
                if char in fail_state:
                    self.fail_links[id(child)] = fail_state[char]
                else:
                    self.fail_links[id(child)] = self.root
                
                # Add output links
                if id(self.fail_links[id(child)]) in self.output:
                    if id(child) not in self.output:
                        self.output[id(child)] = []
                    elif not isinstance(self.output[id(child)], list):
                        self.output[id(child)] = [self.output[id(child)]]
                    
                    fail_outputs = self.output[id(self.fail_links[id(child)])]
                    if isinstance(fail_outputs, list):
                        self.output[id(child)].extend(fail_outputs)
                    else:
                        self.output[id(child)].append(fail_outputs)
        
        self.finalized = True
    
    def search(self, text):
        if not self.finalized:
            self.build_fail_links()
        
        current = self.root
        results = []
        
        for i, char in enumerate(text):
            # Follow the fail links until we find a match or reach the root
            while current is not self.root and char not in current:
                current = self.fail_links[id(current)]
            
            if char in current:
                current = current[char]
            else:
                continue  # No match, stay at root
            
            # Check if this state contains any output
            if id(current) in self.output:
                output = self.output[id(current)]
                if isinstance(output, list):
                    for pattern in output:
                        results.append((i - len(pattern) + 1, pattern))
                else:
                    results.append((i - len(output) + 1, output))
        
        return results

# Replace build_automaton in search_util.py
def build_optimized_automaton(search_terms):
    """
    Build an optimized Aho-Corasick automaton for keyword matching.
    """
    automaton = OptimizedAhoCorasick()
    
    for search_term in search_terms:
        automaton.add_pattern(search_term.text, search_term)
    
    automaton.build_fail_links()
    return automaton

# Modify aho_corasick_match to use the optimized automaton
def optimized_aho_corasick_match(automaton, text, min_score=1):
    matches = []
    search_terms = []
    
    for pos, search_term in automaton.search(text):
        matches.append((pos, search_term))
        search_terms.append(search_term)
    
    if len(search_terms) > 0:
        score = calculate_advanced_search_score(search_terms, len(text), [pos for pos, _ in matches])
        if score >= min_score:
            return matches, score
    
    return [], 0
```

### 2.2 Improved Scoring Algorithm

Optimize the scoring algorithm to be more performant and relevant:

```python
def calculate_optimized_search_score(matched_search_terms, document_length, match_positions):
    """
    An optimized scoring function that reduces computational complexity.
    """
    if not matched_search_terms:
        return 0.0
    
    # Use hash map for O(1) lookup instead of linear search
    term_weights = {}
    unique_terms = set()
    
    # Pre-compute term weights - O(n) where n is number of matches
    for term in matched_search_terms:
        term_text = term.text
        unique_terms.add(term_text)
        
        # Use simple weight formula based on term order
        weight = 1.0 / (term.order + 1.0)
        
        if term_text in term_weights:
            term_weights[term_text] += weight
        else:
            term_weights[term_text] = weight
    
    # Calculate basic score from term weights - O(k) where k is unique terms
    base_score = sum(term_weights.values()) * (len(unique_terms) ** 0.5)
    
    # Skip complex operations for very long documents
    if document_length > 100000:
        return base_score * (1.0 + min(1.0, len(match_positions) / float(document_length)))
    
    # Calculate proximity bonus - O(m log m) where m is match positions
    if match_positions:
        match_positions = sorted(match_positions)
        
        # Find clusters with simple linear scan - O(m)
        cluster_count = 1
        proximity_window = 50
        cluster_start = match_positions[0]
        
        for pos in match_positions[1:]:
            if pos - cluster_start > proximity_window:
                cluster_count += 1
                cluster_start = pos
        
        proximity_factor = 1.0 + (1.0 / cluster_count)
    else:
        proximity_factor = 1.0
    
    # Calculate final score with coverage and proximity factors
    coverage_ratio = min(1.0, len(match_positions) / float(document_length)) if document_length > 0 else 0
    
    return base_score * (1.0 + coverage_ratio) * proximity_factor
```

### 2.3 Optimized Binary Search for Metadata

Improve the binary search with an interpolation-based initial approximation:

```python
def optimized_binary_search_metadata(mm, target_index, line_length=37):
    """
    Perform an optimized binary search with interpolation for initial approximation.
    """
    file_size = mm.size()
    num_lines = file_size // line_length
    
    if num_lines <= 0:
        return -1, -1, -1
    
    # Read the first and last line to get the range
    mm.seek(0)
    first_line = mm.read(line_length).decode('utf-8', errors='ignore').strip()
    first_start, _, _ = map(int, first_line.split(','))
    
    mm.seek(file_size - line_length)
    last_line = mm.read(line_length).decode('utf-8', errors='ignore').strip()
    last_start, last_end, _ = map(int, last_line.split(','))
    
    # If target is outside the range, return early
    if target_index < first_start or target_index >= last_end:
        return -1, -1, -1
    
    # Interpolation for initial guess (if range is valid)
    if last_start > first_start:
        position_ratio = (target_index - first_start) / (last_start - first_start)
        mid = int(position_ratio * (num_lines - 1))
        mid = max(0, min(mid, num_lines - 1))  # Clamp to valid range
    else:
        mid = num_lines // 2  # Fallback to binary search midpoint
    
    # Binary search from the interpolated position
    low, high = 0, num_lines - 1
    row_start, row_end, row_index = -1, -1, -1
    
    while low <= high:
        line_offset = mid * line_length
        mm.seek(line_offset)
        line = mm.read(line_length).decode('utf-8', errors='ignore').strip()
        
        try:
            start, end, row_idx = map(int, line.split(','))
        except (ValueError, IndexError):
            # Handle malformed line - adjust search space and continue
            mid = (low + high) // 2
            continue
        
        if start <= target_index < end:
            row_start, row_end, row_index = start, end, row_idx
            break
        elif target_index < start:
            high = mid - 1
        else:
            low = mid + 1
        
        mid = (low + high) // 2
    
    return row_start, row_end, row_index
```

## 3. Parallelization Improvements

### 3.1 Dynamic Thread Allocation for Web Search

Implement adaptive thread allocation based on system load:

```python
def adaptive_thread_pool(min_workers=2, max_workers=16):
    """
    Create a thread pool that adjusts the number of workers based on system load.
    """
    # Check system load
    cpu_count = os.cpu_count()
    load_avg = os.getloadavg()[0] / cpu_count if hasattr(os, 'getloadavg') else 0.5
    
    # Calculate optimal worker count
    if load_avg < 0.3:  # System is not busy
        workers = max(min_workers, int(cpu_count * 0.75))
    elif load_avg < 0.7:  # System is moderately busy
        workers = max(min_workers, int(cpu_count * 0.5))
    else:  # System is very busy
        workers = max(min_workers, int(cpu_count * 0.25))
    
    # Cap at max_workers
    workers = min(workers, max_workers)
    
    return ThreadPoolExecutor(max_workers=workers)

# Modify search_web function in surfer.py
def search_web(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None):
    # Create a dynamic thread pool
    thread_pool = adaptive_thread_pool()
    
    try:
        # Perform web searches in parallel using adaptive ThreadPoolExecutor
        futures = [thread_pool.submit(single_search_with_backoff, term) for term in search_terms]
        results = [future.result() for future in as_completed(futures)]
        
        # Rest of the function remains the same...
    finally:
        thread_pool.shutdown()
    
    # Rest of the function...

# Add backoff strategy for rate-limited websites
def single_search_with_backoff(search_term: str, max_retries=3, max_results=3):
    retries = 0
    backoff_time = 1  # Initial backoff time in seconds
    
    while retries < max_retries:
        try:
            results = get_request(f"{SEARXNG_BASE_URL}/search?q={search_term}", {}, {})
            
            if "error" in results and "Too Many Requests" in results:
                # Detected rate limiting
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                retries += 1
                continue
            
            urls_to_scrape, most_common_words = extract_urls_and_common_words(results)
            return scrape_urls(urls_to_scrape[:max_results]), most_common_words
        
        except Exception as e:
            retries += 1
            time.sleep(backoff_time)
            backoff_time *= 2
    
    # If all retries failed, return empty results
    return [], []
```

### 3.2 Work-Stealing Queue for File Search

Implement a work-stealing queue for better load balancing:

```python
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

# Modify bulk_search_files to use work-stealing queue
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
```

### 3.3 Asynchronous Web Scraping

Replace synchronous web requests with asynchronous ones for better I/O concurrency:

```python
# Add to surfer.py
import aiohttp
import asyncio

async def async_get_request(url, headers={}, params={}):
    """
    Asynchronous version of the get_request function.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.ClientResponseError as e:
        return json.dumps({"error": str(e), "content": ""})
    except Exception as e:
        return json.dumps({"error": str(e), "content": ""})

async def async_single_search(search_term: str, max_results: int = 3):
    """
    Asynchronous version of single_search function.
    """
    results = await async_get_request(f"{SEARXNG_BASE_URL}/search?q={search_term}", {}, {})
    urls_to_scrape, most_common_words = extract_urls_and_common_words(results)
    
    # Scrape urls concurrently
    scraped_results = await async_scrape_urls(urls_to_scrape[:max_results])
    return scraped_results, most_common_words

async def async_scrape_urls(urls):
    """
    Asynchronous version of scrape_urls that uses a shared Chrome instance.
    """
    driver = None
    try:
        driver = chrome_driver_pool.get_driver()
        initial_window = driver.current_window_handle
        
        # Tasks list for concurrent processing
        tasks = []
        for index, url in enumerate(urls):
            # Open a new tab
            driver.execute_script(f"window.open('{url}', 'tab{index}');")
        
        results = []
        
        # Process each tab
        for window in driver.window_handles:
            if window == initial_window:
                continue
            
            # Switch to the new tab
            driver.switch_to.window(window)
            
            # Start a new thread for scrolling the page to the bottom while it loads
            scroll_to_bottom(driver)
            
            url = driver.current_url
            
            try:
                readable_text = extract_relevant_content(driver.page_source)
                links = categorize_links(url, driver.page_source)
                
                # Handle CAPTCHA and cookies
                if ("verify" in readable_text.lower() and "human" in readable_text.lower()) or (
                        "accept" in readable_text.lower() and "cookies" in readable_text.lower()):
                    time.sleep(.5)
                    readable_text = extract_relevant_content(driver.page_source)
                
                sparse_text = re.sub(r'\n+', '\n', readable_text).strip()
                deduplicated_text = '\n'.join(sorted(set(sparse_text.splitlines()), key=sparse_text.splitlines().index))
            
            except Exception as e:
                print(e)
                deduplicated_text = "! ERROR ACCESSING WEBSITE!"
                links = ""
            
            results.append(
                f"!# WEBSITE CONTENTS FOR URL={url}:\n\n"
                f"{deduplicated_text}\n\n"
                f"!# END OF WEBSITE CONTENTS FOR URL={url}\n\n"
            )
            
            driver.close()
        
        return results
    finally:
        if driver:
            chrome_driver_pool.return_driver(driver)

# New search_web function using asyncio
async def async_search_web(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None):
    """
    Asynchronous version of search_web function.
    """
    discovered_patterns = []
    
    # Use semaphore to limit concurrent searches
    semaphore = asyncio.Semaphore(8)  # Limit to 8 concurrent searches
    
    async def limited_search(term):
        async with semaphore:
            return await async_single_search(term)
    
    # Gather all search tasks
    tasks = [limited_search(term) for term in search_terms]
    results = await asyncio.gather(*tasks)
    
    # Process results
    filtered_data = []
    for idx, result in enumerate(results):
        search_result, top_common_words = result
        for word in top_common_words:
            if word not in semantic_patterns and word not in discovered_patterns:
                semantic_patterns.append(word)
                discovered_patterns.append(word)
        
        for website in search_result:
            filtered_data.append(website)
    
    return filtered_data, discovered_patterns

# Function to run the async search from synchronous code
def search_web_async(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None):
    """
    Wrapper to run async_search_web from synchronous code.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_search_web(search_terms, semantic_patterns, instructions))
    finally:
        loop.close()
```

## 4. Indexing Strategy Improvements

### 4.1 Persistent Search Index

Implement a persistent index system to avoid reprocessing the same files repeatedly:

```python
import sqlite3
import pickle
import hashlib
import zlib

class PersistentSearchIndex:
    def __init__(self, index_db_path="search_index.db"):
        self.index_db_path = index_db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        # Create files table - stores file metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            file_path TEXT UNIQUE,
            size INTEGER,
            last_modified INTEGER,
            checksum TEXT
        )
        ''')
        
        # Create content_index table - stores the extracted content
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_index (
            file_id TEXT PRIMARY KEY,
            content BLOB,
            line_indices BLOB,
            is_utf8 INTEGER,
            FOREIGN KEY (file_id) REFERENCES files(file_id)
        )
        ''')
        
        # Create inverted index table - maps terms to documents
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS inverted_index (
            term TEXT,
            file_id TEXT,
            positions BLOB,
            PRIMARY KEY (term, file_id),
            FOREIGN KEY (file_id) REFERENCES files(file_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_file_id(self, file_path):
        """Generate a unique identifier for a file."""
        return hashlib.sha256(file_path.encode('utf-8')).hexdigest()
    
    def get_file_checksum(self, file_path):
        """Calculate a checksum of the file to detect changes."""
        h = hashlib.sha256()
        with open(file_path, 'rb') as file:
            # Read in chunks to handle large files
            for chunk in iter(lambda: file.read(4096), b''):
                h.update(chunk)
        return h.hexdigest()
    
    def is_file_indexed(self, file_path):
        """Check if a file is already indexed and up-to-date."""
        try:
            file_id = self.get_file_id(file_path)
            file_stat = os.stat(file_path)
            current_size = file_stat.st_size
            current_mtime = int(file_stat.st_mtime)
            
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT size, last_modified, checksum FROM files WHERE file_id = ?",
                (file_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                indexed_size, indexed_mtime, indexed_checksum = row
                
                # Quick check based on size and mtime
                if indexed_size == current_size and indexed_mtime == current_mtime:
                    return True
                
                # If size or mtime changed, perform more expensive checksum verification
                current_checksum = self.get_file_checksum(file_path)
                return indexed_checksum == current_checksum
            
            return False
        except Exception as e:
            print(f"Error checking if file is indexed: {e}")
            return False
    
    def add_file_to_index(self, file_path, content, line_indices, is_utf8, terms_positions):
        """Add or update a file in the index."""
        try:
            file_id = self.get_file_id(file_path)
            file_stat = os.stat(file_path)
            
            # Compress content to save space
            compressed_content = zlib.compress(pickle.dumps(content))
            compressed_line_indices = zlib.compress(pickle.dumps(line_indices))
            
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Add or update file metadata
            cursor.execute(
                '''
                INSERT OR REPLACE INTO files
                (file_id, file_path, size, last_modified, checksum)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    file_id,
                    file_path,
                    file_stat.st_size,
                    int(file_stat.st_mtime),
                    self.get_file_checksum(file_path)
                )
            )
            
            # Add or update content index
            cursor.execute(
                '''
                INSERT OR REPLACE INTO content_index
                (file_id, content, line_indices, is_utf8)
                VALUES (?, ?, ?, ?)
                ''',
                (
                    file_id,
                    compressed_content,
                    compressed_line_indices,
                    1 if is_utf8 else 0
                )
            )
            
            # Remove old term-document mappings
            cursor.execute("DELETE FROM inverted_index WHERE file_id = ?", (file_id,))
            
            # Add new term-document mappings
            for term, positions in terms_positions.items():
                compressed_positions = zlib.compress(pickle.dumps(positions))
                cursor.execute(
                    '''
                    INSERT INTO inverted_index
                    (term, file_id, positions)
                    VALUES (?, ?, ?)
                    ''',
                    (term, file_id, compressed_positions)
                )
            
            # Commit transaction
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error adding file to index: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            return False
    
    def search_index(self, search_terms, and_search=False):
        """
        Search the index for files containing the given terms.
        Returns a list of (file_path, positions) tuples.
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            if and_search:
                # AND search: find files containing all search terms
                placeholders = ','.join(['?' for _ in search_terms])
                query = f'''
                SELECT f.file_path, i.term, i.positions
                FROM files f
                JOIN inverted_index i ON f.file_id = i.file_id
                WHERE i.term IN ({placeholders})
                AND f.file_id IN (
                    SELECT file_id
                    FROM inverted_index
                    WHERE term IN ({placeholders})
                    GROUP BY file_id
                    HAVING COUNT(DISTINCT term) = ?
                )
                '''
                params = search_terms + search_terms + [len(search_terms)]
            else:
                # OR search: find files containing any search term
                placeholders = ','.join(['?' for _ in search_terms])
                query = f'''
                SELECT f.file_path, i.term, i.positions
                FROM files f
                JOIN inverted_index i ON f.file_id = i.file_id
                WHERE i.term IN ({placeholders})
                '''
                params = search_terms
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Group results by file_path
            file_results = {}
            for file_path, term, compressed_positions in results:
                if file_path not in file_results:
                    file_results[file_path] = {}
                
                positions = pickle.loads(zlib.decompress(compressed_positions))
                file_results[file_path][term] = positions
            
            conn.close()
            
            return [(file_path, positions_dict) for file_path, positions_dict in file_results.items()]
        except Exception as e:
            print(f"Error searching index: {e}")
            if 'conn' in locals():
                conn.close()
            return []
    
    def get_indexed_content(self, file_path):
        """
        Retrieve the indexed content and line indices for a file.
        Returns (content, line_indices, is_utf8) or None if not found.
        """
        try:
            file_id = self.get_file_id(file_path)
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT content, line_indices, is_utf8 FROM content_index WHERE file_id = ?",
                (file_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                compressed_content, compressed_line_indices, is_utf8 = row
                content = pickle.loads(zlib.decompress(compressed_content))
                line_indices = pickle.loads(zlib.decompress(compressed_line_indices))
                return content, line_indices, bool(is_utf8)
            
            return None
        except Exception as e:
            print(f"Error retrieving indexed content: {e}")
            if 'conn' in locals():
                conn.close()
            return None
```

### 4.2 Bloom Filter for Pre-filtering

Add a Bloom filter as a pre-filtering step to quickly eliminate files that don't contain search terms:

```python
import bitarray
import mmh3  # MurmurHash3 for fast hashing

class BloomFilter:
    def __init__(self, capacity, error_rate=0.001):
        """
        Initialize a Bloom filter.
        
        Args:
            capacity: Expected number of elements
            error_rate: False positive probability (default: 0.1%)
        """
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal filter size and number of hash functions
        self.size = self._calculate_size(capacity, error_rate)
        self.hash_count = self._calculate_hash_count(self.size, capacity)
        
        # Initialize bit array
        self.bit_array = bitarray.bitarray(self.size)
        self.bit_array.setall(0)
    
    def _calculate_size(self, capacity, error_rate):
        """Calculate optimal size of bit array."""
        return int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
    
    def _calculate_hash_count(self, size, capacity):
        """Calculate optimal number of hash functions."""
        return int(size / capacity * math.log(2))
    
    def _get_hash_values(self, item):
        """Get hash values for the item using different seeds."""
        hash_values = []
        item_str = str(item).encode('utf-8')
        for i in range(self.hash_count):
            hash_values.append(mmh3.hash(item_str, i) % self.size)
        return hash_values
    
    def add(self, item):
        """Add an item to the Bloom filter."""
        for bit_position in self._get_hash_values(item):
            self.bit_array[bit_position] = 1
    
    def contains(self, item):
        """
        Check if an item might be in the Bloom filter.
        False means definitely not in the filter.
        True means possibly in the filter (with false positive rate).
        """
        for bit_position in self._get_hash_values(item):
            if not self.bit_array[bit_position]:
                return False
        return True
    
    def union(self, other):
        """Compute union of two Bloom filters of the same size."""
        if self.size != other.size:
            raise ValueError("Bloom filters must have the same size")
        
        result = BloomFilter(self.capacity, self.error_rate)
        result.bit_array = self.bit_array | other.bit_array
        return result
    
    def intersection(self, other):
        """Compute intersection of two Bloom filters of the same size."""
        if self.size != other.size:
            raise ValueError("Bloom filters must have the same size")
        
        result = BloomFilter(self.capacity, self.error_rate)
        result.bit_array = self.bit_array & other.bit_array
        return result
    
    def serialize(self):
        """Serialize the Bloom filter to bytes."""
        return pickle.dumps({
            'capacity': self.capacity,
            'error_rate': self.error_rate,
            'size': self.size,
            'hash_count': self.hash_count,
            'bit_array': self.bit_array.tobytes()
        })
    
    @classmethod
    def deserialize(cls, data):
        """Deserialize bytes to a Bloom filter."""
        obj = pickle.loads(data)
        bloom = cls(obj['capacity'], obj['error_rate'])
        bloom.size = obj['size']
        bloom.hash_count = obj['hash_count']
        bloom.bit_array = bitarray.bitarray()
        bloom.bit_array.frombytes(obj['bit_array'])
        return bloom

# Add Bloom filter integration to the PersistentSearchIndex
class EnhancedSearchIndex(PersistentSearchIndex):
    def __init__(self, index_db_path="search_index.db"):
        super().__init__(index_db_path)
        self._init_bloom_filter_table()
    
    def _init_bloom_filter_table(self):
        """Initialize the Bloom filter table."""
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bloom_filters (
            file_id TEXT PRIMARY KEY,
            bloom_filter BLOB,
            FOREIGN KEY (file_id) REFERENCES files(file_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def build_bloom_filter(self, terms):
        """Build a Bloom filter from a set of terms."""
        bloom = BloomFilter(len(terms) * 10)  # 10x capacity for future terms
        for term in terms:
            bloom.add(term)
        return bloom
    
    def add_file_to_index(self, file_path, content, line_indices, is_utf8, terms_positions):
        """Override to add Bloom filter."""
        success = super().add_file_to_index(file_path, content, line_indices, is_utf8, terms_positions)
        
        if success:
            try:
                file_id = self.get_file_id(file_path)
                
                # Create Bloom filter from the terms
                bloom = self.build_bloom_filter(terms_positions.keys())
                serialized_bloom = bloom.serialize()
                
                conn = sqlite3.connect(self.index_db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO bloom_filters
                    (file_id, bloom_filter)
                    VALUES (?, ?)
                    ''',
                    (file_id, serialized_bloom)
                )
                
                conn.commit()
                conn.close()
                
                return True
            except Exception as e:
                print(f"Error adding Bloom filter: {e}")
                return False
        
        return success
    
    def pre_filter_files(self, search_terms):
        """
        Use Bloom filters to quickly identify files that might contain the search terms.
        Returns a list of file_ids that potentially have matches.
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_id, bloom_filter FROM bloom_filters")
            candidate_files = []
            
            for file_id, serialized_bloom in cursor.fetchall():
                bloom = BloomFilter.deserialize(serialized_bloom)
                
                # Check if all terms (AND search) or any term (OR search) might be in the filter
                if any(bloom.contains(term) for term in search_terms):
                    candidate_files.append(file_id)
            
            conn.close()
            return candidate_files
        except Exception as e:
            print(f"Error pre-filtering files: {e}")
            if 'conn' in locals():
                conn.close()
            return []
```

### 4.3 Two-tier Indexing System

Implement a two-tier indexing system with directory-level and file-level indices:

```python
class TwoTierSearchIndex:
    def __init__(self, index_root="search_index"):
        self.index_root = index_root
        self.directory_index_path = os.path.join(index_root, "directory_index.db")
        os.makedirs(index_root, exist_ok=True)
        self._init_directory_index()
    
    def _init_directory_index(self):
        """Initialize the directory index database."""
        conn = sqlite3.connect(self.directory_index_path)
        cursor = conn.cursor()
        
        # Create directories table - stores directory metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS directories (
            dir_id TEXT PRIMARY KEY,
            dir_path TEXT UNIQUE,
            last_modified INTEGER,
            file_count INTEGER,
            total_size INTEGER
        )
        ''')
        
        # Create directory terms table - maps terms to directories
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS directory_terms (
            term TEXT,
            dir_id TEXT,
            frequency INTEGER,
            PRIMARY KEY (term, dir_id),
            FOREIGN KEY (dir_id) REFERENCES directories(dir_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_directory_id(self, dir_path):
        """Generate a unique identifier for a directory."""
        return hashlib.sha256(dir_path.encode('utf-8')).hexdigest()
    
    def get_directory_last_modified(self, dir_path):
        """Get the latest modification time of any file in the directory tree."""
        last_modified = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    last_modified = max(last_modified, int(os.stat(file_path).st_mtime))
                except (OSError, IOError):
                    pass
        return last_modified
    
    def get_directory_stats(self, dir_path):
        """Get file count and total size of a directory."""
        file_count = 0
        total_size = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_count += 1
                    total_size += os.path.getsize(file_path)
                except (OSError, IOError):
                    pass
        return file_count, total_size
    
    def update_directory_index(self, dir_path, term_frequencies):
        """Update the directory index with term frequencies."""
        try:
            dir_id = self.get_directory_id(dir_path)
            last_modified = self.get_directory_last_modified(dir_path)
            file_count, total_size = self.get_directory_stats(dir_path)
            
            conn = sqlite3.connect(self.directory_index_path)
            cursor = conn.cursor()
            
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Add or update directory metadata
            cursor.execute(
                '''
                INSERT OR REPLACE INTO directories
                (dir_id, dir_path, last_modified, file_count, total_size)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (dir_id, dir_path, last_modified, file_count, total_size)
            )
            
            # Remove old term-directory mappings
            cursor.execute("DELETE FROM directory_terms WHERE dir_id = ?", (dir_id,))
            
            # Add new term-directory mappings
            for term, frequency in term_frequencies.items():
                cursor.execute(
                    '''
                    INSERT INTO directory_terms
                    (term, dir_id, frequency)
                    VALUES (?, ?, ?)
                    ''',
                    (term, dir_id, frequency)
                )
            
            # Commit transaction
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error updating directory index: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            return False
    
    def get_file_index_path(self, dir_id):
        """Get the path to the file index database for a directory."""
        return os.path.join(self.index_root, f"{dir_id}_files.db")
    
    def create_file_index(self, dir_path):
        """Create a new file index for a directory."""
        dir_id = self.get_directory_id(dir_path)
        file_index_path = self.get_file_index_path(dir_id)
        
        # Create a new EnhancedSearchIndex instance
        file_index = EnhancedSearchIndex(file_index_path)
        return file_index
    
    def get_file_index(self, dir_path):
        """Get the file index for a directory."""
        dir_id = self.get_directory_id(dir_path)
        file_index_path = self.get_file_index_path(dir_id)
        
        if os.path.exists(file_index_path):
            return EnhancedSearchIndex(file_index_path)
        else:
            return self.create_file_index(dir_path)
    
    def search(self, search_terms, directories=None):
        """
        Search for terms across all indexed directories or specific ones.
        Returns a list of (file_path, positions) tuples.
        """
        all_results = []
        
        try:
            conn = sqlite3.connect(self.directory_index_path)
            cursor = conn.cursor()
            
            # Find directories that contain the search terms
            if directories:
                dir_ids = [self.get_directory_id(d) for d in directories]
                placeholders = ','.join(['?' for _ in dir_ids])
                dir_condition = f"AND dir_id IN ({placeholders})"
                dir_params = dir_ids
            else:
                dir_condition = ""
                dir_params = []
            
            term_placeholders = ','.join(['?' for _ in search_terms])
            query = f'''
            SELECT DISTINCT d.dir_id, d.dir_path
            FROM directories d
            JOIN directory_terms dt ON d.dir_id = dt.dir_id
            WHERE dt.term IN ({term_placeholders})
            {dir_condition}
            '''
            
            cursor.execute(query, search_terms + dir_params)
            candidate_dirs = cursor.fetchall()
            conn.close()
            
            # Search each candidate directory's file index
            for dir_id, dir_path in candidate_dirs:
                file_index = EnhancedSearchIndex(self.get_file_index_path(dir_id))
                
                # Use Bloom filters to pre-filter files
                candidate_file_ids = file_index.pre_filter_files(search_terms)
                
                if candidate_file_ids:
                    # Search the filtered files
                    results = file_index.search_index(search_terms)
                    all_results.extend(results)
            
            return all_results
        except Exception as e:
            print(f"Error searching two-tier index: {e}")
            if 'conn' in locals():
                conn.close()
            return []
```

## 5. Web Scraping Enhancements

### 5.1 Improved Content Extraction with NLP

Enhance content extraction with natural language processing to identify primary content:

```python
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class EnhancedContentExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            min_df=2     # Ignore terms that appear in fewer than 2 documents
        )
    
    def extract_content(self, html_doc, url):
        """
        Extract relevant content from HTML using NLP techniques.
        
        Args:
            html_doc: HTML document as string
            url: URL of the document for context
        
        Returns:
            str: Extracted content
        """
        # First use basic extraction to get candidate content
        soup = self._remove_unwanted_elements(BeautifulSoup(html_doc, 'html.parser'))
        
        # Extract all text blocks (paragraphs, divs, etc.)
        blocks = []
        block_elements = soup.find_all(['p', 'div', 'article', 'section', 'main'])
        
        for block in block_elements:
            text = block.get_text(strip=True)
            if len(text) > 100:  # Only consider substantial blocks
                blocks.append(text)
        
        if not blocks:
            # Fallback to basic extraction if no substantial blocks found
            return self._basic_extraction(soup)
        
        # Use NLP to identify the main content
        return self._extract_main_content(blocks, url)
    
    def _remove_unwanted_elements(self, soup):
        """Remove navigation, headers, footers, scripts, etc."""
        for tag in ['head', 'nav', 'footer', 'header', 'aside', 'script', 'style', 'iframe', 
                    'noscript', 'button', 'input', 'meta', 'form', 'svg', 'path']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with common navigation or sidebar classes/IDs
        nav_patterns = ['nav', 'menu', 'sidebar', 'footer', 'header', 'banner', 'ad', 'widget']
        for pattern in nav_patterns:
            for element in soup.find_all(class_=lambda c: c and pattern in c.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda i: i and pattern in i.lower()):
                element.decompose()
        
        return soup
    
    def _basic_extraction(self, soup):
        """Basic content extraction fallback method."""
        main_content = soup.find_all(['article', 'main', 'div'])
        relevant_text = '\n'.join(content.get_text(strip=True) for content in main_content)
        return relevant_text
    
    def _extract_main_content(self, blocks, url):
        """
        Use NLP techniques to identify the main content.
        
        1. Calculate TF-IDF scores for each block
        2. Use sentence density and length as features
        3. Score blocks based on semantic relevance to the URL
        """
        if len(blocks) == 1:
            return blocks[0]
        
        # Compute TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(blocks)
            tfidf_scores = tfidf_matrix.toarray().sum(axis=1)
        except ValueError:
            # If vectorization fails, fall back to length-based scoring
            tfidf_scores = [len(block) for block in blocks]
        
        # Calculate sentence density (sentences per character)
        sentence_density = []
        for block in blocks:
            sentences = sent_tokenize(block)
            density = len(sentences) / (len(block) + 1)  # Add 1 to avoid division by zero
            sentence_density.append(density)
        
        # Extract domain from URL for relevance scoring
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            domain_terms = set(domain.lower().split('.')[:-1])  # Ignore TLD
        except:
            domain_terms = set()
        
        # Score blocks based on domain relevance
        domain_relevance = []
        for block in blocks:
            words = set(w.lower() for w in block.split() if w.lower() not in self.stop_words)
            relevance = len(words.intersection(domain_terms)) / (len(words) + 1)
            domain_relevance.append(relevance)
        
        # Combine scores with appropriate weights
        combined_scores = [
            (0.5 * tfidf + 0.3 * density + 0.2 * relevance + 0.1 * len(block))
            for tfidf, density, relevance, block in zip(tfidf_scores, sentence_density, domain_relevance, blocks)
        ]
        
        # Sort blocks by score in descending order
        ranked_blocks = [block for _, block in sorted(
            zip(combined_scores, blocks), key=lambda x: x[0], reverse=True
        )]
        
        # Return top blocks that together constitute a substantial amount of content
        output = []
        total_length = 0
        target_length = sum(len(block) for block in blocks) * 0.7  # Target 70% of total content
        
        for block in ranked_blocks:
            output.append(block)
            total_length += len(block)
            if total_length >= target_length:
                break
        
        return '\n\n'.join(output)

# Replace extract_relevant_content in web_util.py
def enhanced_extract_relevant_content(html_doc: str, url: str) -> str:
    """
    Extracts relevant content from an HTML document using NLP techniques.
    
    Args:
        html_doc (str): The HTML document.
        url (str): The URL of the document.
    
    Returns:
        str: The extracted relevant content.
    """
    extractor = EnhancedContentExtractor()
    return extractor.extract_content(html_doc, url)
```

### 5.2 Machine Learning-based Relevance Scoring

Implement ML-based relevance scoring for extracted content:

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class RelevanceScoringModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            max_features=10000
        )
    
    def train(self, query_document_pairs):
        """
        Train the model on a set of query-document pairs with relevance scores.
        
        Args:
            query_document_pairs: List of (query, document, score) tuples
        """
        queries = [pair[0] for pair in query_document_pairs]
        documents = [pair[1] for pair in query_document_pairs]
        scores = [pair[2] for pair in query_document_pairs]
        
        # Fit vectorizer on both queries and documents
        all_texts = queries + documents
        self.vectorizer.fit(all_texts)
        
        # Store reference data
        self.reference_queries = queries
        self.reference_documents = documents
        self.reference_scores = scores
        
        # Pre-compute document vectors
        self.document_vectors = self.vectorizer.transform(documents)
    
    def score_relevance(self, query, document):
        """
        Score the relevance of a document to a query.
        
        Args:
            query: Search query string
            document: Document content string
        
        Returns:
            float: Relevance score between 0 and 1
        """
        # Transform query and document to vectors
        query_vector = self.vectorizer.transform([query])
        doc_vector = self.vectorizer.transform([document])
        
        # Calculate direct cosine similarity
        direct_similarity = cosine_similarity(query_vector, doc_vector)[0][0]
        
        # Find similarities to reference queries and documents
        query_similarities = cosine_similarity(query_vector, self.vectorizer.transform(self.reference_queries))[0]
        doc_similarities = cosine_similarity(doc_vector, self.document_vectors)[0]
        
        # Find the most similar reference pairs
        combined_similarities = query_similarities * doc_similarities
        top_indices = np.argsort(combined_similarities)[-5:]  # Top 5 similar pairs
        
        # Weighted average of reference scores
        weights = combined_similarities[top_indices]
        weights = weights / (np.sum(weights) + 1e-10)  # Normalize weights
        weighted_score = np.sum(weights * np.array([self.reference_scores[i] for i in top_indices]))
        
        # Combine direct similarity with reference-based score
        final_score = 0.7 * direct_similarity + 0.3 * weighted_score
        
        return max(0, min(1, final_score))  # Clamp to [0, 1]

# Function to rank search results by relevance
def rank_search_results(search_term, results):
    """
    Rank search results by relevance to the search query.
    
    Args:
        search_term: The original search query
        results: List of search result content
    
    Returns:
        List of (result, score) tuples sorted by relevance
    """
    # Initialize the model
    model = RelevanceScoringModel()
    
    # Create synthetic training data (in a real implementation, use actual relevance scores)
    synthetic_pairs = []
    for i, result in enumerate(results[:5]):
        # Extract a sample from each result for training
        sample = result[:500] if len(result) > 500 else result
        # Assign synthetic relevance score based on position (higher is better)
        score = 1.0 - (i / 5)
        synthetic_pairs.append((search_term, sample, score))
    
    # Train the model on synthetic data
    model.train(synthetic_pairs)
    
    # Score all results
    scored_results = []
    for result in results:
        score = model.score_relevance(search_term, result)
        scored_results.append((result, score))
    
    # Sort by score in descending order
    return sorted(scored_results, key=lambda x: x[1], reverse=True)

# Integrate into the web search function
def search_web_with_relevance(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None):
    """
    Search the web and rank results by relevance.
    """
    # Get raw search results
    raw_results, discovered_patterns = search_web(search_terms, semantic_patterns, instructions)
    
    # Rank results for each search term
    ranked_results = []
    for term in search_terms:
        term_results = rank_search_results(term, raw_results)
        ranked_results.extend([result for result, _ in term_results[:3]])  # Take top 3 for each term
    
    # Remove duplicates while preserving order
    seen = set()
    filtered_results = []
    for result in ranked_results:
        result_hash = hash(result[:100])  # Use first 100 chars as hash
        if result_hash not in seen:
            seen.add(result_hash)
            filtered_results.append(result)
    
    return filtered_results, discovered_patterns
```

### 5.3 Semantic HTML Parsing

Add semantic HTML tag recognition for better context awareness:

```python
class SemanticHTMLParser:
    def __init__(self):
        # Define semantic HTML5 tags and their importance
        self.semantic_tags = {
            'article': 10,      # Highest importance - self-contained composition
            'main': 9,          # Main content of the document
            'section': 8,       # Thematic grouping of content
            'h1': 7,            # Primary heading
            'h2': 6,            # Secondary heading
            'h3': 5,            # Tertiary heading
            'p': 4,             # Paragraph
            'ul': 3,            # Unordered list
            'ol': 3,            # Ordered list
            'li': 2,            # List item
            'blockquote': 4,    # Quotation
            'figure': 3,        # Self-contained content
            'figcaption': 2,    # Caption for figure
            'details': 3,       # Details/summary widget
            'summary': 3,       # Summary part of details
            'time': 2,          # Time designation
            'mark': 2,          # Marked/highlighted text
            'dl': 3,            # Description list
            'dt': 2,            # Term in description list
            'dd': 2,            # Description in description list
            'table': 3,         # Table
            'th': 2,            # Table header
            'tr': 1,            # Table row
            'td': 1             # Table cell
        }
    
    def parse(self, html_doc):
        """
        Parse HTML content with awareness of semantic tags.
        
        Args:
            html_doc: HTML document as string
        
        Returns:
            dict: Structured content with semantic information
        """
        soup = BeautifulSoup(html_doc, 'html.parser')
        
        # Extract semantic structure
        semantic_structure = self._extract_semantic_structure(soup)
        
        # Extract main content based on semantic structure
        main_content = self._extract_main_content(semantic_structure)
        
        # Extract headings hierarchy
        headings = self._extract_headings(soup)
        
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        return {
            'main_content': main_content,
            'headings': headings,
            'metadata': metadata,
            'semantic_structure': semantic_structure
        }
    
    def _extract_semantic_structure(self, soup):
        """Extract the semantic structure of the document."""
        structure = []
        
        # Process body or root if no body
        body = soup.body or soup
        
        # Process direct children of body
        for child in body.children:
            if child.name in self.semantic_tags:
                structure.append(self._process_semantic_element(child))
        
        return structure
    
    def _process_semantic_element(self, element, depth=0):
        """Process a semantic element and its children recursively."""
        if not element.name:
            return None
        
        importance = self.semantic_tags.get(element.name, 0)
        text_content = element.get_text(strip=True)
        
        # Skip elements with no text
        if not text_content:
            return None
        
        result = {
            'tag': element.name,
            'importance': importance,
            'depth': depth,
            'text': text_content,
            'attrs': dict(element.attrs) if element.attrs else {},
            'children': []
        }
        
        # Process children recursively
        for child in element.children:
            if child.name in self.semantic_tags:
                child_data = self._process_semantic_element(child, depth + 1)
                if child_data:
                    result['children'].append(child_data)
        
        return result
    
    def _extract_main_content(self, semantic_structure):
        """Extract main content based on semantic importance."""
        main_content = []
        
        # Helper function to traverse the structure and find important content
        def extract_content(structure, min_importance=3):
            for item in structure:
                if item and 'importance' in item and item['importance'] >= min_importance:
                    main_content.append(item['text'])
                
                if item and 'children' in item and item['children']:
                    extract_content(item['children'], min_importance)
        
        extract_content(semantic_structure)
        return '\n\n'.join(main_content)
    
    def _extract_headings(self, soup):
        """Extract the hierarchy of headings."""
        headings = []
        for i in range(1, 7):  # h1 through h6
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text(strip=True)
                })
        
        return headings
    
    def _extract_metadata(self, soup):
        """Extract metadata from the document."""
        metadata = {}
        
        # Extract title
        title_tag = soup.title
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        # Extract structured data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                json_data = json.loads(script.string)
                metadata['structured_data'] = json_data
            except (json.JSONDecodeError, TypeError):
                pass
        
        return metadata

# Integrate the semantic parser into web content extraction
def extract_web_content_with_semantics(html_doc, url):
    """
    Extract web content using semantic HTML parsing.
    
    Args:
        html_doc: HTML document as string
        url: URL of the document
    
    Returns:
        dict: Structured content with semantic information
    """
    # Basic cleanup first
    soup = remove_unwanted_elements(BeautifulSoup(html_doc, 'html.parser'))
    cleaned_html = str(soup)
    
    # Apply semantic parsing
    parser = SemanticHTMLParser()
    semantic_data = parser.parse(cleaned_html)
    
    # Add URL to metadata
    if 'metadata' not in semantic_data:
        semantic_data['metadata'] = {}
    semantic_data['metadata']['url'] = url
    
    return semantic_data

# Function to format semantic data for searching and display
def format_semantic_content(semantic_data):
    """
    Format semantic data into a searchable and displayable string.
    
    Args:
        semantic_data: Structured semantic data
    
    Returns:
        str: Formatted content
    """
    parts = []
    
    # Add title
    if 'metadata' in semantic_data and 'title' in semantic_data['metadata']:
        parts.append(f"TITLE: {semantic_data['metadata']['title']}")
    
    # Add headings in hierarchy
    if 'headings' in semantic_data and semantic_data['headings']:
        parts.append("HEADINGS:")
        for heading in semantic_data['headings']:
            indent = "  " * (heading['level'] - 1)
            parts.append(f"{indent}{heading['text']}")
    
    # Add main content
    if 'main_content' in semantic_data:
        parts.append("CONTENT:")
        parts.append(semantic_data['main_content'])
    
    # Add metadata
    if 'metadata' in semantic_data:
        parts.append("METADATA:")
        for key, value in semantic_data['metadata'].items():
            if key != 'structured_data' and not isinstance(value, dict):
                parts.append(f"  {key}: {value}")
    
    return "\n\n".join(parts)
```

## 6. Error Handling and Robustness

### 6.1 Comprehensive Error Recovery

Implement more robust error handling and recovery mechanisms:

```python
import logging
import traceback
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("search_system")

class ErrorRecoveryState:
    """Class to maintain recovery state across retries."""
    def __init__(self):
        self.retry_count = 0
        self.last_error = None
        self.checkpoints = {}
    
    def checkpoint(self, name, data):
        """Save a checkpoint."""
        self.checkpoints[name] = data
    
    def get_checkpoint(self, name):
        """Get a checkpoint."""
        return self.checkpoints.get(name)
    
    def increment_retry(self, error):
        """Increment retry count and save error."""
        self.retry_count += 1
        self.last_error = error

def with_error_recovery(max_retries=3, checkpoint_frequency=None):
    """
    Decorator for functions that need error recovery.
    
    Args:
        max_retries: Maximum number of retries
        checkpoint_frequency: How often to save checkpoints (in seconds)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            recovery_state = ErrorRecoveryState()
            last_checkpoint_time = time.time()
            
            while recovery_state.retry_count <= max_retries:
                try:
                    # If this is a retry, log it
                    if recovery_state.retry_count > 0:
                        logger.info(f"Retry {recovery_state.retry_count}/{max_retries} for {func.__name__}")
                    
                    # If we have checkpoints, restore from the last one
                    if 'result_so_far' in recovery_state.checkpoints:
                        kwargs['_recovery_state'] = recovery_state
                    
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Success! Return the result
                    return result
                
                except Exception as e:
                    recovery_state.increment_retry(e)
                    
                    # Log the error
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    
                    # If we've exceeded max retries, re-raise the exception
                    if recovery_state.retry_count > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Calculate backoff time
                    backoff_time = min(2 ** recovery_state.retry_count, 60)  # Max 60 seconds
                    logger.info(f"Backing off for {backoff_time} seconds")
                    time.sleep(backoff_time)
        
        return wrapper
    
    return decorator

# Apply error recovery to bulk_search_files
@with_error_recovery(max_retries=3)
def bulk_search_files_with_recovery(
        root_dir: str,
        search_term_strings: List[str],
        context_size_lines: int = 16,
        large_file_size_threshold: int = 512 * 1024 * 1024,
        min_score: float = 1,
        and_search: bool = False,
        exact_matches_only: bool = True,
        _recovery_state: ErrorRecoveryState = None
) -> List[UnifiedSearchResult]:
    """
    Enhanced version of bulk_search_files with error recovery.
    """
    # If we have a recovery state with checkpoints, use them
    if _recovery_state and 'processed_files' in _recovery_state.checkpoints:
        processed_files = _recovery_state.get_checkpoint('processed_files')
        all_results = _recovery_state.get_checkpoint('results_so_far')
        logger.info(f"Resuming search from checkpoint with {len(processed_files)} processed files")
    else:
        processed_files = set()
        all_results = []
    
    try:
        # Build file tree
        root = build_file_tree(root_dir)
        
        # Categorize files
        large_files = []
        small_files = []
        categorize_files(large_files, small_files, root, large_file_size_threshold)
        
        # Filter out already processed files
        small_files = [f for f in small_files if f not in processed_files]
        large_files = [f for f in large_files if f not in processed_files]
        
        logger.info(f"Processing {len(small_files)} small files and {len(large_files)} large files")
        
        # Create queues
        results_queue = queue.Queue()
        small_files_queue = queue.Queue()
        for file in small_files:
            small_files_queue.put(file)
        
        # Process small files
        threads = []
        cpu_count = psutil.cpu_count(logical=False)
        
        for _ in range(cpu_count):
            thread = threading.Thread(
                target=process_files_worker,
                args=(small_files_queue, results_queue, search_term_strings, context_size_lines, 
                      min_score, and_search, exact_matches_only, processed_files)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Process large files
        for file_path in large_files:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension in ['.csv', '.tsv']:
                    # Process structured data file
                    results = search_structured_data(
                        file_path, f"{file_path}.offsets", f"{file_path}.metadata", 
                        search_term_strings, min_score
                    )
                    results_queue.put(results)
                
                # Mark file as processed
                processed_files.add(file_path)
                
                # Save checkpoint
                if _recovery_state:
                    _recovery_state.checkpoint('processed_files', processed_files)
                    
                    # Collect intermediate results
                    intermediate_results = list(all_results)
                    while not results_queue.empty():
                        intermediate_results.extend(results_queue.get())
                    
                    _recovery_state.checkpoint('results_so_far', intermediate_results)
            
            except Exception as e:
                logger.error(f"Error processing large file {file_path}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Wait for all threads to complete
        small_files_queue.join()
        for thread in threads:
            thread.join(timeout=1)
        
        # Collect all results
        while not results_queue.empty():
            all_results.extend(results_queue.get())
        
        # Sort and return results
        return sorted(all_results, key=lambda x: x.common.score, reverse=True)
    
    except Exception as e:
        logger.error(f"Error in bulk_search_files: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Save checkpoint before re-raising
        if _recovery_state:
            _recovery_state.checkpoint('processed_files', processed_files)
            _recovery_state.checkpoint('results_so_far', all_results)
        
        raise

def process_files_worker(file_queue, results_queue, search_terms, context_size, 
                        min_score, and_search, exact_matches, processed_files):
    """Worker function to process files from the queue."""
    while True:
        try:
            # Get a file from the queue
            file_path = file_queue.get(block=False)
        except queue.Empty:
            # No more files to process
            break
        
        try:
            # Process the file
            content, line_indices, is_utf8 = extract_file_content(file_path)
            
            if exact_matches:
                search_terms_obj = parse_search_terms(search_terms)
            else:
                search_terms_obj = preprocess_search_term_list(search_terms)
            
            file_matches, score = search_file(content, search_terms_obj, min_score)
            
            if file_matches:
                results, _ = build_search_result_for_file(
                    file_path, file_matches, score, line_indices, content, 
                    context_size, and_search, len(search_terms), {}
                )
                
                results_queue.put(results)
            
            # Mark file as processed
            processed_files.add(file_path)
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.debug(traceback.format_exc())
        
        finally:
            # Mark the task as done
            file_queue.task_done()
```

### 6.2 Circuit Breaker for External Services

Add circuit breaker pattern for external service dependencies:

```python
from enum import Enum
import time
import threading

class CircuitState(Enum):
    CLOSED = 1  # Normal operation, requests flow through
    OPEN = 2    # Circuit is open, requests fail fast
    HALF_OPEN = 3  # Testing if service is back up

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30, name="default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.lock = threading.RLock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == CircuitState.OPEN:
                    # Check if recovery timeout has elapsed
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        logger.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                        self.state = CircuitState.HALF_OPEN
                    else:
                        logger.warning(f"Circuit {self.name} is OPEN, failing fast")
                        raise CircuitBreakerOpenError(f"Circuit {self.name} is open")
            
            try:
                result = func(*args, **kwargs)
                
                # If the call succeeded and we were in HALF_OPEN, close the circuit
                with self.lock:
                    if self.state == CircuitState.HALF_OPEN:
                        logger.info(f"Circuit {self.name} transitioning from HALF_OPEN to CLOSED")
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                
                return result
            
            except Exception as e:
                with self.lock:
                    self.last_failure_time = time.time()
                    
                    # If we're already in OPEN or HALF_OPEN state, stay in OPEN state
                    if self.state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
                        logger.warning(f"Failure in {self.name} circuit while in {self.state} state")
                        self.state = CircuitState.OPEN
                    else:
                        # We're in CLOSED state, increment failure count
                        self.failure_count += 1
                        logger.warning(f"Failure #{self.failure_count} in {self.name} circuit")
                        
                        if self.failure_count >= self.failure_threshold:
                            logger.warning(f"Circuit {self.name} transitioning from CLOSED to OPEN")
                            self.state = CircuitState.OPEN
                
                raise
        
        return wrapper

class CircuitBreakerOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    pass

# Create circuit breakers for different services
searx_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=60, name="searx")
web_scraper_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=120, name="web_scraper")

# Apply circuit breakers to web search functions
@searx_circuit
def get_request_with_circuit_breaker(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> str:
    """
    Sends a GET request with circuit breaker protection.
    """
    return get_request(url, headers, params)

@web_scraper_circuit
def scrape_urls_with_circuit_breaker(urls):
    """
    Scrapes URLs with circuit breaker protection.
    """
    return scrape_urls(urls)

# Modified search_web function using circuit breakers
def search_web_with_circuit_breakers(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None):
    """
    Search the web with circuit breaker protection.
    """
    discovered_patterns = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for term in search_terms:
                futures.append(executor.submit(single_search_with_circuit_breakers, term))
            
            results = []
            for future in as_completed(futures):
                try:
                    search_result, top_common_words = future.result()
                    results.append((search_result, top_common_words))
                except CircuitBreakerOpenError as e:
                    logger.error(f"Circuit breaker error: {str(e)}")
                    # Fallback to alternative source or cached data
                    continue
            
            for search_result, top_common_words in results:
                for word in top_common_words:
                    if word not in semantic_patterns and word not in discovered_patterns:
                        semantic_patterns.append(word)
                        discovered_patterns.append(word)
                
                for idx, website in enumerate(search_result):
                    file_path = os.path.join(temp_dir, f'search_result_{idx}.txt')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(website + "\n")
            
            filtered_data = filter_scraped_data(temp_dir, semantic_patterns, instructions)
    
    finally:
        if os.path.exists(temp_dir):
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)
    
    return filtered_data, discovered_patterns

def single_search_with_circuit_breakers(search_term: str, max_results: int = 3):
    """
    Single search with circuit breaker protection.
    """
    try:
        results = get_request_with_circuit_breaker(f"{SEARXNG_BASE_URL}/search?q={search_term}", {}, {})
        urls_to_scrape, most_common_words = extract_urls_and_common_words(results)
        return scrape_urls_with_circuit_breaker(urls_to_scrape[:max_results]), most_common_words
    except CircuitBreakerOpenError:
        # If the circuit is open, try to use a fallback search engine
        try:
            # This is just a placeholder - in a real implementation you'd use a different search engine
            logger.info(f"Using fallback search engine for term: {search_term}")
            time.sleep(1)  # Simulate fallback search
            return [], []
        except Exception as e:
            logger.error(f"Fallback search also failed: {str(e)}")
            return [], []
```

## 7. Configuration and Tuning

### 7.1 Self-tuning Parameters

Implement self-tuning search parameters based on system resources and search patterns:

```python
# TODO
```