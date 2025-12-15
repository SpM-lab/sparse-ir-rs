#!/bin/bash
# Setup conda path
# This script adds conda to PATH if not already present

# Find conda directory
if command -v conda &> /dev/null; then
    # conda is already in PATH
    exit 0
elif [ -d "$HOME/miniconda3/bin" ]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
elif [ -d "$HOME/anaconda3/bin" ]; then
    export PATH="$HOME/anaconda3/bin:$PATH"
elif [ -d "/opt/miniconda3/bin" ]; then
    export PATH="/opt/miniconda3/bin:$PATH"
elif [ -d "/opt/anaconda3/bin" ]; then
    export PATH="/opt/anaconda3/bin:$PATH"
fi
