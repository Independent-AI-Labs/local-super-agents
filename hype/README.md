# High-Yield Pattern Extraction (HYPE)
Memory and CPU-optimized semantic search & data annotation for the edge.


---

##  Features

HYPE offers a comprehensive set of features to **optimize data processing, indexing, and retrieval** for various document types and sizes. These features work in concert to provide a robust, adaptable, and efficient solution for advanced data analysis and search operations.

### 1. Data Pre-Processing & Indexing

#### 1.1 Large File Segmentation & Small File Batching

HYPE provides a **resource-aware** approach to file handling, optimizing the processing of both large individual files and collections of small files.

- **Large Files**  
  Large files are split into manageable chunks that can be processed concurrently, making full but controlled use of system RAM (without forcing the system into SWAP). This design scales with available system RAM and CPU cores:
  
  ```python
  def split_structured_data_file(
      file_path: str,
      metadata_dir: str,
      item_start_validation_pattern_str: re.Pattern = None,
      free_memory_usage_limit: float = .75
  ) -> List[Tuple[int, int]]:
      file_size = os.path.getsize(file_path)
      available_memory = int(psutil.virtual_memory().available * free_memory_usage_limit)

      total_blocks = calculate_total_blocks(file_size, available_memory)
      block_size = file_size // total_blocks

      offsets = []

      with open(file_path, 'rb') as file:
          with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
              current_position = 0
              # Implementation details for finding valid block boundaries...
              ...
      return offsets
  ```

  ```python
  def calculate_total_blocks(
      file_size: int,
      available_memory: int,
      sub_blocks: int = os.cpu_count() // 2
  ) -> int:
      """
      Calculate the total number of blocks based on file size, available memory, and number of physical CPUs.
      ...
      """
      block_load_memory_limit = available_memory // sub_blocks
      total_blocks = max(1, file_size // block_load_memory_limit)
      return total_blocks * sub_blocks
  ```

  ```python
  def execute_memory_managed_block_processing(
      block_offsets: list,
      procs: List[Process],
      free_memory_usage_limit: float = .75
  ) -> None:
      ...
      # Initially, load as many blocks as available memory allows.
      initial_load = 0
      initial_load_index = len(block_offsets)

      for i, (start, end) in enumerate(block_offsets):
          if initial_load + end - start < psutil.virtual_memory().available * free_memory_usage_limit:
              initial_load += end - start
          else:
              initial_load_index = i
              break

      for i, (start, end) in enumerate(block_offsets):
          if i > initial_load_index or (end - start) > psutil.virtual_memory().available * free_memory_usage_limit:
              if i == initial_load_index + 1:
                  time.sleep(1)  # Wait for more accurate RAM reading

              while (end - start) > psutil.virtual_memory().available * free_memory_usage_limit:
                  time.sleep(0.1)

          procs[i].start()
      ...
  ```

- **Small Files**  
  When processing **many small files**, the overhead of handling each file individually can be reduced using standard Python multi-threading—well-suited to I/O-bound operations:

  ```python
  cpu_count = os.cpu_count() // 2
  files_per_thread = max(1, len(small_files) // cpu_count)

  for i in range(0, len(small_files), files_per_thread):
      file_chunk = small_files[i : i + files_per_thread]
      create_thread(
          target=process_file_list,
          args=(
              file_chunk,
              search_term_strings,
              context_size_lines,
              results_queue,
              min_score,
              and_search,
              exact_matches_only,
          )
      )
  ```

This design enables **parallel I/O** operations and scales linearly with CPU resources in production environments. For more CPU-intensive tasks, HYPE also supports multi-processing (and multi-hosting/distributed deployments).

#### 1.2 Structured Data Item Caching & Indexing

HYPE implements **efficient caching and indexing** for structured data (e.g., CSV rows, JSONL items). The following example indexes structured data at the character level:

```python
def index_structured_data_file(
    file_path: str,
    start_offset: int,
    end_offset: int,
    block_id: int,
    output_dir: str = "metadata_chunks",
    item_break_sequence: str = DEFAULT_LINE_BREAK_SEQUENCE_STR,
    item_start_validation_pattern: re.Pattern = None,
    write_buffer_size_bytes: int = 1024 * 1024
) -> None:
    ...
    # Use Aho-Corasick to find matches of the item break sequence.
    potential_item_breaks = simple_aho_corasick_match(automaton, block_data_string)
    ...

    for index, break_position in enumerate(potential_item_breaks):
        ...
        if item_start_validation_pattern is None or item_start_validation_pattern.match(string_start):
            ...
            batch.append(
                f"{str(start_index).zfill(12)},{str(end_index).zfill(12)},{str(total_items).zfill(9)}\n"
            )
            last_valid_end = end_index + len(item_break_sequence) - 1
    ...
```

Such an approach can handle **noisy or malformed data** commonly found in the wild. It is conceptually similar to the **column indexing** used by database engines, often applied when large volumes of sequential data need to be retrieved quickly.

In the near future, an LLM-driven **auto-detection** of line-break sequences and validation patterns will allow fully automated indexing of raw data, with minimal human intervention.

---

### 2. Fast Automated Data Annotation & Retrieval (WIP)

HYPE supports **persistent binary-searchable metadata** for low-latency, low-memory data retrieval. For example, a simple CSV-based metadata file might look like:

```
000000000000,000000000217,0000000001
000000000219,000000000541,0000000002
000000000543,000000000804,0000000003
...
```

Each line stores fixed-width start index, end index, and row number for a CSV item in the target file. This format facilitates extremely fast lookups for any given character position without loading large data structures into memory.

```python
def get_metadata_for_position(
    metadata_filepath: str, 
    position: int
) -> Tuple[int, int, int]:
    """
    Retrieve item metadata for a given character position.
    """
    with open(metadata_filepath, 'r+b') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return find_item_metadata(mm, position)
```

```python
def find_item_metadata(
    memory_map: mmap.mmap,
    position: int
) -> Tuple[int, int, int]:
    # Determine row size from the first line.
    first_newline = memory_map.find(b'\n')
    if first_newline == -1:
        return -1, -1, -1

    row_size = first_newline + 1
    total_rows = memory_map.size() // row_size
    left, right = 0, total_rows - 1

    while left <= right:
        mid = (left + right) // 2
        row_start = mid * row_size

        row_data = memory_map[row_start : row_start + row_size].decode('utf-8').strip()
        start, end, index = map(int, row_data.split(','))

        if start <= position <= end:
            return index, start, end
        elif position < start:
            right = mid - 1
        else:
            left = mid + 1

    return -1, -1, -1
```

#### 2.1 Annotation Engine (Coming Soon)

Planned annotation features include:

- **Custom Field Declarations**  
- **Python-Scriptable Text & Binary Matching Rules**  
- **Built-In Parallelization**  

---

### 3.  Document Content Extraction & Indexing

HYPE can optionally handle content extraction for various document types:

```python
def extract_file_content(file_path: str) -> Tuple[str, List[int]]:
    file_extension = os.path.splitext(file_path)[1].lower()

    if any(file_path.endswith(ext) for ext in TEXT_BASED_EXTENSIONS):
        return extract_text_and_line_indices(file_path)
    elif file_extension == '.docx':
        return extract_docx_content(file_path)
    elif file_extension == '.pdf':
        return extract_pdf_content(file_path)
    ...
```

- **Plain-text Files**  
  For typical text-based file formats, no special manipulation is required.
  
- **HTML Files**  
  Treated as code by default, with optional HTML-specific extraction utilities in a separate library.

- **XLSX Files**  
  Planned for future support (requires additional handling for formulas and multi-sheet structures).

- **PDF, DOCX, etc.**  
  Basic extraction implemented or in progress.

This entire functionality will be offloaded to [Apache Tika](https://tika.apache.org/) for more robust document parsing in a following release.

---

### 4. Spelling Correction Frequency Dictionary Creation (WIP)

HYPE can automatically build **frequency dictionaries** to aid in search-term spelling corrections:

```python
def generate_frequency_dict(
    corpus_files: List[str],
    output_file: str = 'frequency_dict.txt',
    chunk_size: int = 1024 * 1024
) -> Dict[str, int]:
    word_counter = Counter()

    def process_chunk(chunk: str):
        words = re.findall(r'\b\w+\b', chunk.lower())
        word_counter.update(words)

    for file_path in corpus_files:
        if not os.path.isfile(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        with open(file_path, 'r+b') as file:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

            for i in range(0, len(mm), chunk_size):
                chunk = mm[i : i + chunk_size].decode('utf-8', errors='ignore')
                process_chunk(chunk)

            mm.close()

    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in word_counter.most_common():
            f.write(f"{word} {count}\n")

    return dict(word_counter)
```

This approach enables a “reverse elastic-search” style method for efficient, **context-aware** spelling correction.

---

### 5. Sparse Word / Lead Character Map Generation (TBD)

A planned feature for further optimization of data retrieval and analytics.

---

### 6. Line Numbering

Although it appears trivial, line numbering in Python can be costly in massive-scale processing. HYPE handles it via a **compiled Cython** module:

```python
def extract_text_and_line_indices(str file_path):
    cdef:
        bytes content
        const unsigned char * content_ptr
        Py_ssize_t content_len, pos = 0
        list indices = [0]
        str decoded_content
        int is_utf8 = 1

    with open(file_path, 'rb') as f:
        content = f.read()

    content_ptr = <const unsigned char *> PyBytes_AS_STRING(content)
    content_len = PyBytes_GET_SIZE(content)

    while pos < content_len:
        if content_ptr[pos] == ord('\n'):
            indices.append(pos + 1)
        pos += 1

    # Attempt UTF-8 decoding, fall back to Latin-1
    try:
        decoded_content = PyUnicode_DecodeUTF8(<const char *> content_ptr, content_len, "ignore")
    except UnicodeDecodeError:
        is_utf8 = 0
        decoded_content = PyUnicode_DecodeLatin1(<const char *> content_ptr, content_len, "ignore")

    return decoded_content, indices, is_utf8
```

---

### 7. Storage Engine Translation Layers (WIP)

#### 7.1 WiredTiger (MongoDB) Integration

HYPE includes initial work for **direct reading and querying** of `.wt` (WiredTiger) files:

```python
WT_PAGE_HEADER_SIZE = 28
WT_CELL_HEADER_SIZE = 4
WT_BSON_DOC_MARKER = b'\x03'

def read_wiredtiger_file(file_path: str) -> List[Dict[str, Any]]:
    decoded_documents = []
    with open(file_path, 'rb') as wt_file:
        with mmap.mmap(wt_file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            offset = 0
            while offset < mm.size():
                page_type, _, page_size = struct.unpack('>BHI', mm[offset:offset+7])
                if page_type == 1:  # Leaf page
                    offset += WT_PAGE_HEADER_SIZE
                    while offset < mm.size():
                        # Read and decode cell data...
                        ...
                else:
                    offset += page_size
    return decoded_documents
```

```python
def query_wiredtiger_file(file_path: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Query documents in a WiredTiger file based on a simple query.
    """
    documents = read_wiredtiger_file(file_path)

    def match_document(doc: Dict[str, Any], q: Dict[str, Any]) -> bool:
        for key, value in q.items():
            if key not in doc or doc[key] != value:
                return False
        return True
    
    return [doc for doc in documents if match_document(doc, query)]
```

Once completed, this integration will enable **offline** querying of MongoDB data without running a MongoDB instance, making it ideal for systems where memory consumption and overall resource usage must be minimized.

#### 7.2 Future Database Engine Support (TBD)

Plans are in place to support direct or semi-direct data extraction from:

- **SQLite**  
- **PostgreSQL**  
- **MySQL / MariaDB**

#### 7.3 Database Query Generation (WIP)

For databases requiring external API or SQL connections, **LLM-driven query generation** (in SQL, Mongo, etc.) will be provided in a future release.

---

### 8. Semantic Search & Query Augmentation

#### 8.1 Query Pre-Processing

HYPE supports **advanced query pre-processing** steps:

- **Spelling Correction** (with custom dictionaries)  
- **“Adjacent” Search Term Generation** using LLMs (WIP)  
- **N-gram Generation** for fuzzy matching  
- **Semantic Variations** (TBD)

```python
def preprocess_search_term(
    search_term_string: str,
    case_sensitive: bool = False,
    spelling_autocorrect: bool = True,
    spellchecker: SpellChecker = None,
    ref_ids: List[str] | None = None
) -> List[SearchTerm]:
    search_term_strings = [append_order(search_term_string, 0)]
    ...

    if spelling_autocorrect:
        ...
        corrected = []
        for word in words:
            spellchecked = spellchecker.correction(word)
            ...
        if len(corrected) > 0:
            append_with_ngrams(" ".join(corrected), search_term_strings)

    # Case variations
    if not case_sensitive:
        if search_term_string.lower() != search_term_string:
            append_with_ngrams(search_term_string.lower(), search_term_strings)
        if search_term_string.upper() != search_term_string:
            append_with_ngrams(search_term_string.upper(), search_term_strings)
        if search_term_string.title() != search_term_string:
            append_with_ngrams(search_term_string.title(), search_term_strings)

    return parse_search_terms(search_term_strings, ref_ids)
```

#### 8.2 Result Post-Processing & Scoring

HYPE includes a **flexible scoring system** for search results, factoring in term order, frequency, and distribution:

```python
def calculate_search_score(
    matched_search_terms: List[SearchTerm],
    document_length: int,
    match_positions: List[int]
) -> float:
    ...
    # Score accumulation based on term order
    for search_term in matched_search_terms:
        score += 10 / (10 ** (search_term.order + 1))
    ...
    # Distribution factor
    if match_positions and document_length > 0:
        match_density = len(match_positions) / document_length
        distribution_factor = 1 + min(1.0, match_density)
        score *= distribution_factor
    ...
```

Future work includes:

- **Derivative Search** (candidate word clusters or LLM-generated complementary terms)  
- **Context-Aware Summarization**  

---

### 9. Large & Bulk File Searching

HYPE uses **optimized strategies** for large-scale search:

```python
def bulk_search_files(
    root_dir: str,
    search_term_strings: List[str],
    context_size_lines: int = 16,
    large_file_size_threshold: int = 512 * 1024 * 1024,  # 512MB
    min_score: float = 1,
    and_search: bool = False,
    exact_matches_only: bool = True
) -> List[UnifiedSearchResult]:
    ...
    cpu_count = psutil.cpu_count(logical=False)
    files_per_thread = max(1, len(small_files) // cpu_count)

    for i in range(0, len(small_files), files_per_thread):
        file_chunk = small_files[i : i + files_per_thread]
        create_thread(
            target=process_file_list,
            args=(
                file_chunk,
                search_term_strings,
                context_size_lines,
                results_queue,
                min_score,
                and_search,
                exact_matches_only,
            )
        )
```

### 10. Structured Data Searching

HYPE also provides a **comprehensive** search approach for structured data using a **combined** multi-process/multi-thread strategy:

```python
def search_structured_data_batched(
    file_path: str,
    metadata_dir: str,
    search_term_strings: List[str],
    min_score: float = 1,
    free_memory_usage_limit: float = 0.75,
    memory_overhead_factor: int = 1,
) -> List[UnifiedSearchResult]:
    ...
    search_terms = preprocess_search_term_list(search_term_strings)
    num_physical_cores = psutil.cpu_count(logical=False)

    if len(block_offsets) / 2 > num_physical_cores:
        chunk_size = len(block_offsets) // num_physical_cores
    else:
        return search_structured_data_threaded(
            file_path, metadata_dir, search_terms, min_score
        )

    chunks = [block_offsets[i : i + chunk_size] for i in range(0, len(block_offsets), chunk_size)]
    with Manager() as manager:
        results_queue = manager.Queue()
        ...
        return collect_and_merge_results(results_queue)
```

---

### 11. Future Enhancements

Planned improvements include:

1. **Binary Data Search** (e.g., “binary” Aho-Corasick for non-text files)  
2. **Enhanced Distributed Processing** (using tools like `huey`, `redis`)  
 **Expanded Machine Learning Integrations**  
   - Thinc or other advanced NLP libraries  
   - Model generation and inference for specialized NLP tasks  
4. **Real-Time Indexing and Search** for continuous data streams  
5. **Advanced Visualization Tools** for search results and data patterns  

---