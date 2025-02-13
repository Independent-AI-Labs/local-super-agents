import bisect
import gc
import mmap
import os

from typing import Tuple, List

import PyPDF2
import docx

from retrieval.hype.data.file_type_support import TEXT_BASED_EXTENSIONS
from retrieval.hype.data.models import FileNode
from retrieval.hype.indexing.line_indexing import extract_text_and_line_indices


def build_file_tree(directory: str) -> FileNode:
    """
    Build a file tree structure for a given directory.

    Args:
        directory (str): The root directory to start building the tree.

    Returns:
        FileNode: The root node of the file tree.
    """
    root = FileNode(directory, 0)
    for dir_path, dir_names, filenames in os.walk(directory):
        current_node = root
        path_parts = os.path.relpath(dir_path, directory).split(os.sep)
        for part in path_parts:
            if part == '.':
                continue
            if part not in current_node.children:
                current_node.children[part] = FileNode(os.path.join(current_node.path, part), 0)
            current_node = current_node.children[part]

        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            file_size = os.path.getsize(file_path)
            current_node.children[filename] = FileNode(file_path, file_size)
            current_node.size += file_size

    return root


def categorize_files(large_files: list, small_files: list, node: FileNode, large_file_size_threshold: int) -> None:
    """
    Categorizes files into large and small based on a size threshold.

    :param large_files: List to store paths of large files.
    :param small_files: List to store paths of small files.
    :param node: The root node of the file tree to categorize.
    :param large_file_size_threshold: Size threshold to distinguish large files from small files.
    """
    if not node.children:  # It's a file
        if node.size > large_file_size_threshold:
            large_files.append(node.path)
        else:
            small_files.append(node.path)
    else:  # It's a directory
        for child in node.children.values():
            categorize_files(large_files, small_files, child, large_file_size_threshold)


def get_line_number(index: int, line_indices: List[int]) -> int:
    """
    Get the line number for a given index using binary search.

    Args:
        index (int): The index to find the line number for.
        line_indices (List[int]): List of line start indices.

    Returns:
        int: Line number (1-based).
    """
    return bisect.bisect_right(line_indices, index)


def load_offsets(offsets_file: str) -> list:
    """
    Load start and end offsets from an offsets file.

    :param offsets_file: Path to the offsets file.
    :return: List of tuples containing (start_offset, end_offset) for each chunk.
    """
    offsets = []

    with open(offsets_file, 'r', encoding='utf-8') as f:
        for line in f:
            start_offset, end_offset, item_offset = map(int, line.strip().split(','))
            offsets.append((start_offset, end_offset))

    return offsets


def read_mmap(file_path: str, start_offset: int, end_offset: int) -> str:
    """
    Reads a specific portion of a file using memory mapping (mmap).

    :param file_path: The path to the file.
    :param start_offset: The start offset (in bytes) to read from.
    :param end_offset: The end offset (in bytes) to read to.
    :return: The string content read from the specified file portion.
    """
    with open(file_path, 'r+b', buffering=0) as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm.seek(start_offset)
        block_data = mm.read(end_offset - mm.tell())
        block_data_string = block_data.decode('utf-8', errors='ignore')
        mm.close()
    gc.collect()
    return block_data_string


# TODO Implement everything below this line using Apache Tika!
#  ==============================================================
def extract_file_content(file_path: str) -> Tuple[str, List[int], bool]:
    """
    Extract content from various file types and return it as a string along with line indices.

    Supported file types: TXT, CSV, TSV, JSON, HTML, DOCX, PDF, and various source code formats.

    Args:
        file_path (str): Path to the file.

    Returns:
        Tuple[str, List[int]]: A tuple containing the extracted content as a string and a list of line start indices.

    Raises:
        IsADirectoryError: If the given path is a directory.
    """
    if os.path.isdir(file_path):
        raise IsADirectoryError(f"{file_path} is a directory.")

    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if any(file_path.endswith(ext) for ext in TEXT_BASED_EXTENSIONS):
            return extract_text_and_line_indices(file_path)
        elif file_extension == '.docx':
            return extract_docx_content(file_path)
        elif file_extension == '.pdf':
            return extract_pdf_content(file_path)
        else:
            print(f"To Be Implemented: Support for {file_extension} files")
            return "", [], False
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return "", [], False


def extract_docx_content(file_path: str) -> Tuple[str, List[int], bool]:
    """
    Extract text content and line indices from a DOCX file.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        Tuple[str, List[int]]: Extracted content as a string and list of line start indices.
    """
    doc = docx.Document(file_path)
    full_text = []
    line_indices = [0]
    current_index = 0
    for para in doc.paragraphs:
        full_text.append(para.text)
        current_index += len(para.text) + 1  # +1 for the newline
        line_indices.append(current_index)
    return '\n'.join(full_text), line_indices, True


def extract_pdf_content(file_path: str) -> Tuple[str, List[int], bool]:
    """
    Extract text content and line indices from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        Tuple[str, List[int]]: Extracted content as a string and list of line start indices.
    """
    with open(file_path, 'rb', buffering=0) as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = []
        line_indices = [0]
        current_index = 0
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            page_text = page.extractText()
            text.append(page_text)
            current_index += len(page_text) + 1  # +1 for the newline
            line_indices.append(current_index)
    return '\n'.join(text), line_indices, True
