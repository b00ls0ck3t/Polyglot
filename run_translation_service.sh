#!/bin/bash

echo "=========================================="
echo "Starting Translation Service"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found"
    echo "Run ./setup.sh first"
    exit 1
fi

# Load environment variables if .env exists
if [ -f ".env" ]; then
    source .env
fi

# Run the translation service
echo "Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python translation_demo.py
