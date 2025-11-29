#!/bin/bash
# Setup script for Chinese Rap Lyrics Analysis Pipeline

set -e

echo "=========================================="
echo "Chinese Rap Lyrics Analysis Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if version is >= 3.8
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize jieba
echo ""
echo "Initializing jieba dictionary..."
python -c "import jieba; jieba.initialize()"

# Create data directory structure
echo ""
echo "Creating data directory structure..."
mkdir -p data/transformed_data
mkdir -p data/transformed_data/visualizations

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Place your lyrics file at: data/transformed_data/all_lyrics.txt"
echo "3. Run the pipeline: python pipeline.py --base-dir ./data"
echo ""
echo "For more information, see README.md"
