#!/bin/bash

echo "=========================================="
echo "Starting Audio Processing Pipeline"
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

# Check if translation service is running
if ! curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "⚠ Warning: Translation service not detected at http://localhost:8000"
    echo "  The pipeline will run but translations won't be sent anywhere."
    echo "  Start the translation service in another terminal with:"
    echo "  ./run_translation_service.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the audio pipeline
echo "Starting audio pipeline..."
echo "Press Ctrl+C to stop"
echo ""

python audio_pipeline.py
