#!/bin/bash
set -e

echo "Creating and setting up vibecheck conda environment..."

# Create a new conda environment named vibecheck with Python 3.10
conda create -n vibecheck python=3.10 -y

# Activate the vibecheck environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vibecheck

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Change to the knowledge/retrieval directory and run setup
cd ../knowledge/retrieval/

# Run the setup script in the retrieval directory
bash ./run_to_setup.sh

# Return to the original directory
cd ../../vibecheck/

echo ""
echo "VibeCheck environment setup completed successfully!"
echo "To activate the environment, run: conda activate vibecheck"
echo "To start the application use the appropriate run script!"
