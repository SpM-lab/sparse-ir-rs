#!/bin/bash
# Build conda package
# This script finds conda in PATH or common locations

set -e

# Find conda executable
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
elif [ -f "/opt/miniconda3/bin/conda" ]; then
    CONDA_CMD="/opt/miniconda3/bin/conda"
elif [ -f "/opt/anaconda3/bin/conda" ]; then
    CONDA_CMD="/opt/anaconda3/bin/conda"
else
    echo "Error: conda not found. Please install conda or add it to PATH." >&2
    exit 1
fi

echo "Using conda: $CONDA_CMD"
"$CONDA_CMD" build conda-recipe
