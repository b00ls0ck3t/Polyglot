#!/bin/bash
set -e

echo "=========================================="
echo "Czech → English Translation Pipeline"
echo "Complete Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check for required tools
echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.8 or later"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ Git not found${NC}"
    echo "Please install git"
    exit 1
fi
echo -e "${GREEN}✓ Git found${NC}"

if ! command -v make &> /dev/null; then
    echo -e "${RED}✗ Make not found${NC}"
    echo "Please install build tools (Xcode Command Line Tools on macOS)"
    exit 1
fi
echo -e "${GREEN}✓ Make found${NC}"

# Check for portaudio (needed for pyaudio)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}⚠ Homebrew not found - pyaudio may fail to install${NC}"
    else
        if ! brew list portaudio &> /dev/null 2>&1; then
            echo "Installing portaudio via Homebrew..."
            brew install portaudio
        fi
        echo -e "${GREEN}✓ Portaudio available${NC}"
    fi
fi

echo ""
echo "=========================================="
echo "Step 1: Setting up Python environment"
echo "=========================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Upgrade pip
echo "Upgrading pip..."
./venv/bin/pip3 install --upgrade pip > /dev/null

echo ""
echo "=========================================="
echo "Step 2: Installing Python dependencies"
echo "=========================================="

echo "Installing packages (this may take a few minutes)..."
./venv/bin/pip3 install -r requirements.txt

echo -e "${GREEN}✓ Python dependencies installed${NC}"

echo ""
echo "=========================================="
echo "Step 3: Installing whisper.cpp"
echo "=========================================="

WHISPER_DIR="$HOME/whisper.cpp"

if [ -d "$WHISPER_DIR" ]; then
    echo "whisper.cpp directory already exists at $WHISPER_DIR"
    read -p "Rebuild whisper.cpp? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping whisper.cpp build"
    else
        cd "$WHISPER_DIR"
        echo "Pulling latest changes..."
        git pull
        echo "Building whisper.cpp..."
        make clean
        make -j
        cd -
        echo -e "${GREEN}✓ whisper.cpp rebuilt${NC}"
    fi
else
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
    
    cd "$WHISPER_DIR"
    echo "Building whisper.cpp (optimized for M1)..."
    make -j
    cd -
    
    echo -e "${GREEN}✓ whisper.cpp installed at $WHISPER_DIR${NC}"
fi

echo ""
echo "=========================================="
echo "Step 4: Downloading Whisper model"
echo "=========================================="

MODEL="large-v2"
MODEL_FILE="$WHISPER_DIR/models/ggml-$MODEL.bin"

if [ -f "$MODEL_FILE" ]; then
    echo -e "${GREEN}✓ Model $MODEL already downloaded${NC}"
else
    echo "Downloading $MODEL model (~3GB, this will take a few minutes)..."
    cd "$WHISPER_DIR"
    bash ./models/download-ggml-model.sh "$MODEL"
    cd -
    
    if [ -f "$MODEL_FILE" ]; then
        echo -e "${GREEN}✓ Model downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Model download failed${NC}"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Step 5: Setting up Pyannote"
echo "=========================================="

echo ""
echo -e "${YELLOW}Pyannote requires a HuggingFace token.${NC}"
echo ""
echo "To get your token:"
echo "1. Go to https://huggingface.co/settings/tokens"
echo "2. Create a new token (or copy existing)"
echo "3. Accept the model terms at:"
echo "   https://huggingface.co/pyannote/speaker-diarization-3.1"
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN environment variable not set."
    read -p "Enter your HuggingFace token (or press Enter to skip): " HF_INPUT
    
    if [ -n "$HF_INPUT" ]; then
        export HF_TOKEN="$HF_INPUT"
        
        # Add to .env file for persistence
        echo "export HF_TOKEN=\"$HF_INPUT\"" > .env
        echo -e "${GREEN}✓ Token saved to .env file${NC}"
        echo "  Run 'source .env' in future sessions"
    else
        echo -e "${YELLOW}⚠ Skipping pyannote setup - diarization will be disabled${NC}"
    fi
else
    echo -e "${GREEN}✓ HF_TOKEN already set${NC}"
fi

echo ""
echo "=========================================="
echo "Step 6: Verifying installation"
echo "=========================================="

echo "Checking whisper.cpp..."
if [ -f "$WHISPER_DIR/main" ]; then
    echo -e "${GREEN}✓ whisper.cpp executable found${NC}"
else
    echo -e "${RED}✗ whisper.cpp executable not found${NC}"
    exit 1
fi

echo "Checking Python packages..."
python3 -c "import fastapi, websockets, pyannote.audio, pyaudio, numpy, torch" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All Python packages installed${NC}"
else
    echo -e "${RED}✗ Some Python packages missing${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To run the pipeline:"
echo ""
echo "  1. Start the translation service:"
echo "     ${GREEN}./run_translation_service.sh${NC}"
echo ""
echo "  2. In another terminal, start the audio pipeline:"
echo "     ${GREEN}./run_audio_pipeline.sh${NC}"
echo ""
echo "  3. Set your DeepL API key in the web UI at:"
echo "     ${GREEN}http://localhost:8000${NC}"
echo ""
echo "Note: You can also run just the audio pipeline for testing"
echo "      (it will work without the translation service)"
echo ""
echo "=========================================="
echo ""