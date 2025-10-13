#!/bin/bash
set -e

echo "🔧 Setting up environment..."

# Install ultralytics first
echo "📦 Installing ultralytics..."
pip install ultralytics

# Uninstall the problematic opencv-python
echo "🗑️  Removing opencv-python..."
pip uninstall -y opencv-python || true

# Install the headless version
echo "✅ Installing opencv-python-headless..."
pip install opencv-python-headless

echo "🎉 Setup complete! Starting training..."

# Run the training script with all arguments passed to this script
python train.py "$@"
