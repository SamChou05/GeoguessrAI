#!/bin/bash
# Install dependencies for GeoGuessr AI training

echo "Installing dependencies for GeoGuessr AI..."
echo ""

# Core ML libraries
pip install torch torchvision torchaudio

# Data processing
pip install numpy pandas scikit-learn

# Visualization
pip install matplotlib seaborn

# Utilities
pip install requests

# Optional: For S3 support (if needed later)
# pip install boto3

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "To verify installation:"
echo "  python3 -c 'import torch; print(\"PyTorch:\", torch.__version__)'"

