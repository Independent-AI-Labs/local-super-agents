#!/bin/bash
set -e

# Activate the pipelines environment
conda activate pipelines

# Build the Python extension in place
python setup.py build_ext --inplace

# Move the compiled .so file to the desired folder
# The wildcard pattern handles different Python versions and platforms
mv line_indexing.cpython-*.so hype/indexing/

# Display success message
echo "Build and move operation completed successfully!"
