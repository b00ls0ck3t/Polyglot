#!/bin/bash

echo "=========================================="
echo "Git Setup & GitHub Upload"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "audio_pipeline.py" ]; then
    echo "Error: audio_pipeline.py not found"
    echo "Please run this script from the project directory"
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git not installed"
    echo "Install with: brew install git"
    exit 1
fi

echo "Step 1: Initialize git repository"
if [ -d ".git" ]; then
    echo "  Git repository already initialized"
else
    git init
    echo "  [OK] Git repository initialized"
fi

echo ""
echo "Step 2: Configure git (if needed)"
if [ -z "$(git config user.name)" ]; then
    read -p "Enter your name: " git_name
    git config user.name "$git_name"
fi

if [ -z "$(git config user.email)" ]; then
    read -p "Enter your email: " git_email
    git config user.email "$git_email"
fi

echo "  Git configured:"
echo "    Name: $(git config user.name)"
echo "    Email: $(git config user.email)"

echo ""
echo "Step 3: Add files to git"

# Copy necessary files if not present
if [ ! -f ".gitignore" ]; then
    echo "  Creating .gitignore..."
    # .gitignore should already be in outputs directory
fi

# Add all project files
git add .gitignore
git add audio_pipeline.py
git add translation_demo_simplified.py
git add setup.sh
git add run_translation_service_simplified.sh
git add run_audio_pipeline.sh
git add requirements.txt
git add README.md
git add CONFIGURATION_GUIDE.md
git add IMPROVEMENT_SUGGESTIONS.md
git add QUICKSTART.txt

echo "  [OK] Files staged"

echo ""
echo "Step 4: Create initial commit"
if git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "  No changes to commit"
else
    git commit -m "Initial commit: Polyglot - Real-time speech translation

- Real-time transcription with whisper.cpp
- Speaker diarization with pyannote.audio
- DeepL translation with speaker-aware batching
- Web UI with live updates
- Two profiles: SPEED and ACCURACY"
    echo "  [OK] Initial commit created"
fi

echo ""
echo "Step 5: GitHub setup"
echo ""
echo "To upload to GitHub:"
echo ""
echo "1. Go to https://github.com/new"
echo "2. Create a new repository (e.g., 'czech-english-translation')"
echo "3. DO NOT initialize with README (you already have one)"
echo "4. Copy the repository URL"
echo ""
read -p "Enter your GitHub repository URL (or press Enter to skip): " repo_url

if [ -n "$repo_url" ]; then
    echo ""
    echo "Adding GitHub remote..."
    git remote add origin "$repo_url" 2>/dev/null || git remote set-url origin "$repo_url"
    
    echo "Pushing to GitHub..."
    git branch -M main
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "[OK] Successfully pushed to GitHub!"
        echo "=========================================="
        echo ""
        echo "Your repository is now at:"
        echo "$repo_url"
    else
        echo ""
        echo "[WARN] Push failed. You may need to authenticate."
        echo ""
        echo "If using HTTPS, you need a Personal Access Token:"
        echo "1. Go to https://github.com/settings/tokens"
        echo "2. Generate new token (classic)"
        echo "3. Select 'repo' scope"
        echo "4. Use token as password when prompted"
        echo ""
        echo "Or use SSH:"
        echo "1. Set up SSH key: https://docs.github.com/en/authentication"
        echo "2. Change remote: git remote set-url origin git@github.com:username/repo.git"
        echo "3. Push again: git push -u origin main"
    fi
else
    echo ""
    echo "Skipped GitHub upload."
    echo ""
    echo "To upload later:"
    echo "1. Create repo on GitHub"
    echo "2. git remote add origin <url>"
    echo "3. git push -u origin main"
fi

echo ""
echo "=========================================="
echo "Git Setup Complete"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  git status              - Check what changed"
echo "  git add <file>          - Stage changes"
echo "  git commit -m 'message' - Commit changes"
echo "  git push                - Push to GitHub"
echo "  git log                 - View commit history"
echo ""