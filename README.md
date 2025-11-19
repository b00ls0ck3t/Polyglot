# Czech → English Real-Time Translation Pipeline

Complete end-to-end system: Audio capture → Czech transcription → Speaker diarization → English translation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AUDIO PIPELINE                          │
│  (audio_pipeline.py)                                        │
│                                                             │
│  ┌─────────────┐      ┌──────────────┐                    │
│  │ Microphone  │──────>│ Audio Queue  │                    │
│  │   Input     │  5s   │   (chunks)   │                    │
│  └─────────────┘ chunks└──────────────┘                    │
│                              │                              │
│                              ▼                              │
│  ┌────────────────────────────────────────────┐            │
│  │   PARALLEL PROCESSING                      │            │
│  │                                            │            │
│  │  ┌──────────────┐    ┌─────────────────┐ │            │
│  │  │ whisper.cpp  │    │ pyannote.audio  │ │            │
│  │  │              │    │                 │ │            │
│  │  │ Transcribe   │    │  Diarization    │ │            │
│  │  │ Czech → Text │    │  Speaker ID     │ │            │
│  │  └──────────────┘    └─────────────────┘ │            │
│  │         │                     │           │            │
│  └─────────┼─────────────────────┼───────────┘            │
│            │                     │                         │
│            └──────────┬──────────┘                         │
│                       ▼                                     │
│            ┌──────────────────┐                            │
│            │  Merge Results   │                            │
│            │  Text + Speaker  │                            │
│            └──────────────────┘                            │
│                       │                                     │
│                       │ WebSocket                           │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              TRANSLATION SERVICE                            │
│  (translation_demo.py)                                      │
│                                                             │
│  ┌──────────────┐       ┌─────────────┐                   │
│  │  WebSocket   │──────>│   DeepL     │                   │
│  │   Server     │       │     API     │                   │
│  └──────────────┘       └─────────────┘                   │
│         │                      │                           │
│         │                      ▼                           │
│         │          ┌──────────────────┐                   │
│         │          │ Czech → English  │                   │
│         │          │   Translation    │                   │
│         │          └──────────────────┘                   │
│         │                      │                           │
│         └──────────────────────┘                           │
│                    │                                        │
│                    │ Broadcast                              │
│                    ▼                                        │
│  ┌────────────────────────────────────┐                   │
│  │         Web Interface               │                   │
│  │  ┌─────────────────────────────┐   │                   │
│  │  │  Czech Transcript Pane      │   │                   │
│  │  └─────────────────────────────┘   │                   │
│  │  ┌─────────────────────────────┐   │                   │
│  │  │  English Translation Pane   │   │                   │
│  │  └─────────────────────────────┘   │                   │
│  └────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## One-Command Setup

```bash
./setup.sh
```

This single script does **everything**:
1. ✓ Checks prerequisites (Python, Git, Make)
2. ✓ Creates Python virtual environment
3. ✓ Installs all Python dependencies
4. ✓ Clones and builds whisper.cpp (M1 optimized)
5. ✓ Downloads Whisper large-v2 model (~3GB)
6. ✓ Configures pyannote with HuggingFace token
7. ✓ Verifies installation

**Time:** ~10-15 minutes (mostly model download)

## Running Everything

### Two-Terminal Setup (Recommended)

**Terminal 1 - Translation Service:**
```bash
./run_translation_service.sh
```

**Terminal 2 - Audio Pipeline:**
```bash
./run_audio_pipeline.sh
```

That's it! Speak in Czech, see English translations in real-time at `http://localhost:8000`

## What You Get

### Features
- ✅ **Real-time audio capture** from microphone
- ✅ **whisper.cpp transcription** - Fast, accurate Czech → text
- ✅ **pyannote diarization** - Automatic speaker identification  
- ✅ **DeepL translation** - High-quality Czech → English
- ✅ **Web UI** - Split-pane display with timestamps
- ✅ **All offline** except DeepL API calls

### Performance (M1 Mac)
- Transcription: ~0.3x real-time with large-v2
- Diarization: ~0.5x real-time
- Total latency: 5-7 seconds

## Prerequisites

### Required
- macOS with Homebrew
- Python 3.8+
- Xcode Command Line Tools: `xcode-select --install`

### API Keys (Free Tier)
1. **DeepL**: https://www.deepl.com/pro-api (500k chars/month)
2. **HuggingFace**: https://huggingface.co/settings/tokens
   - Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1

## Configuration

Edit `audio_pipeline.py`:
```python
WHISPER_MODEL = "large-v2"  # or "medium" for faster
CHUNK_DURATION = 5  # seconds of audio per chunk
```

**Model options:**
- `large-v2` - Best accuracy for Czech (recommended)
- `medium` - 2x faster, slightly less accurate
- `small` - Not recommended for Czech

## Project Structure

```
.
├── setup.sh                      # ONE COMMAND TO RULE THEM ALL
├── run_translation_service.sh    # Start backend + UI
├── run_audio_pipeline.sh         # Start audio processing
├── audio_pipeline.py             # Whisper + pyannote integration
├── translation_demo.py           # DeepL + WebSocket + Web UI
└── requirements.txt              # Python deps
```

## Testing Without Translation

Run just the audio pipeline to test transcription:

```bash
./run_audio_pipeline.sh
```

It will work standalone and print Czech transcriptions + speakers to console.

## Troubleshooting

**"Build failed"**
```bash
xcode-select --install  # Install build tools
./setup.sh              # Run again
```

**"pyaudio install failed"**
```bash
brew install portaudio  # Install audio library
./setup.sh              # Run again
```

**"No microphone"**
- System Preferences → Security → Microphone → Allow Terminal

**"Slow transcription"**
- Use `medium` model instead of `large-v2`
- Increase `CHUNK_DURATION` to 10 seconds

## Architecture Details

### Audio Pipeline (audio_pipeline.py)
- **AudioProcessor**: Main orchestrator
- **WhisperTranscriber**: Wraps whisper.cpp binary
- **SpeakerDiarizer**: Wraps pyannote.audio
- Parallel processing: transcription + diarization run simultaneously
- WebSocket client: sends results to translation service

### Translation Service (translation_demo.py)
- **FastAPI**: HTTP + WebSocket server
- **TranslationManager**: DeepL API integration
- **Web UI**: Real-time split-pane interface
- Broadcasts to all connected clients

### Data Flow
```
Mic → PyAudio → Queue → WAV file
  ↓
[whisper.cpp] → Czech text
[pyannote]    → Speaker labels
  ↓
Merge → WebSocket → Translation Service
  ↓
DeepL API → English text → Web UI
```

## Environment Variables

```bash
# Required for diarization
export HF_TOKEN="your_huggingface_token"

# Optional overrides
export WHISPER_PATH="/custom/path/to/whisper.cpp"
```

The setup script will prompt for HF_TOKEN and save to `.env` file.

## Dependencies

### Python
- fastapi, uvicorn, websockets, httpx
- pyannote.audio, torch, torchaudio
- pyaudio, numpy

### External
- whisper.cpp (built from source, installed to ~/whisper.cpp)

## Performance Tips

1. **Lower latency**: Reduce `CHUNK_DURATION` to 3-4s
2. **Faster processing**: Use `medium` model
3. **Better accuracy**: Use `large-v3` model
4. **Save battery**: Use `medium` + increase chunk duration

## What's Offline vs Online

**Offline (runs locally):**
- ✅ Audio capture
- ✅ whisper.cpp transcription
- ✅ pyannote diarization
- ✅ Web UI display

**Online (needs internet):**
- 🌐 DeepL translation API

**Note:** Models download once during setup, then work offline.

## Future Enhancements

- [ ] Offline translation (NLLB or similar)
- [ ] Voice activity detection (skip silence)
- [ ] Export transcripts
- [ ] Native app (no web browser needed)
- [ ] Multiple target languages

## License

MIT - Use freely

## Credits

- whisper.cpp: Georgi Gerganov
- pyannote.audio: Hervé Bredin (CNRS)
- DeepL API
- FastAPI framework
