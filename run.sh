#!/bin/bash

set -e  # stop on error

echo "Creating virtual environment..."
python3 -m venv rl_venv

echo "Activating virtual environment..."
source rl_venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Phase 1.1: Image Classification - Baseline Testing."
cd "Image Classification"
cd "Baseline-simplest"


python run.py
echo "Phase 1.1 complete"

cd ..
cd "nas_with_skip"
python run.py

echo "Phase 1.2 complete"
