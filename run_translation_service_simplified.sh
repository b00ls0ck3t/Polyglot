#!/bin/bash

echo "=========================================="
echo "Starting Translation Service (Simplified)"
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

# Check for DeepL API key
if [ -z "$DEEPL_API_KEY" ]; then
    echo "⚠ DeepL API key not set"
    echo ""
    read -p "Enter your DeepL API key: " api_key
    if [ -n "$api_key" ]; then
        export DEEPL_API_KEY="$api_key"
        echo "export DEEPL_API_KEY=\"$api_key\"" >> .env
        echo "✓ API key saved to .env"
    fi
    echo ""
fi

# Run the translation service
echo "Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python translation_demo_simplified.py
