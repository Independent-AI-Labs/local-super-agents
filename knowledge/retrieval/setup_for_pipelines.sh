#!/bin/bash
set -e

# Activate the pipelines environment
conda activate pipelines

# Install dependencies from requirements.txt
pip install -r requirements-hype.txt

# Run the rebuild.sh script
bash ./rebuild_for_pipelines.sh

# Display success message
echo "Installation and rebuild process completed successfully!"
