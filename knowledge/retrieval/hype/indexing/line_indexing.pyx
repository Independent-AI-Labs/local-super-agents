# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.unicode cimport PyUnicode_DecodeUTF8, PyUnicode_DecodeLatin1

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_text_and_line_indices(str file_path):
    cdef:
        bytes content
        const unsigned char * content_ptr
        Py_ssize_t content_len, pos = 0
        list indices = [0]
        str decoded_content
        int is_utf8 = 1  # Assume UTF-8 initially

    # Read file content
    with open(file_path, 'rb') as f:
        content = f.read()

    content_ptr = <const unsigned char *> PyBytes_AS_STRING(content)
    content_len = PyBytes_GET_SIZE(content)

    while pos < content_len:
        if content_ptr[pos] == ord('\n'):
            indices.append(pos + 1)
        pos += 1

    # Try decoding as UTF-8
    try:
        decoded_content = PyUnicode_DecodeUTF8(<const char *> content_ptr, content_len, "ignore")
    except UnicodeDecodeError:
        # Fallback to Latin-1
        is_utf8 = 0
        decoded_content = PyUnicode_DecodeLatin1(<const char *> content_ptr, content_len, "ignore")

    return decoded_content, indices, is_utf8
