#!/bin/bash

echo "Starting VibeCheck application..."

# Activate the vibecheck conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate vibecheck; then
    echo "Failed to activate vibecheck environment."
    echo "Please make sure you've run setup.sh first."
    exit 1
fi

# Run the VibeCheck application
echo "Running VibeCheck app..."
python -m vibecheck.app

# Capture the exit code
VIBECHECK_EXIT=$?

# Return the exit code
if [ $VIBECHECK_EXIT -ne 0 ]; then
    echo "VibeCheck application exited with error code: $VIBECHECK_EXIT"
    exit $VIBECHECK_EXIT
fi

echo "VibeCheck application closed."
