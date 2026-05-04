#!/bin/bash
set -e

echo "Starting NAS-FCOS Pipeline..."

# System dependencies (Ubuntu 22.04)
echo "Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip python3-dev build-essential \
    libgl1-mesa-glx libglib2.0-0

# Virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run full pipeline (data streams from HuggingFace, no download needed)
echo "Running NAS-FCOS pipeline..."
python run.py

echo "Done. Results are in the 'outputs/' folder."
