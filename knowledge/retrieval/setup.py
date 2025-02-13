from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "line_indexing",
        ["retrieval\\indexing\\line_indexing.pyx"],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)
