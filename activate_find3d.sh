#!/bin/bash
# Find3D Environment Activation Script
# Usage: source activate_find3d.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Add uv to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "✓ Find3D environment activated!"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  torch-geometric: $(python -c 'import torch_geometric; print(torch_geometric.__version__)')"
echo ""
echo "Environment location: $SCRIPT_DIR/.venv"
