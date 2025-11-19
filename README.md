# Polyglot

Real-time Czech to English speech translation with speaker diarization.

## What it does

Captures audio from your microphone, transcribes Czech speech using whisper.cpp, identifies speakers with pyannote, buffers text intelligently, and translates to English via DeepL. All processing happens locally except translation.

## Quick start

```bash
./setup.sh

export DEEPL_API_KEY="your_key"
export HF_TOKEN="your_hf_token"

# Terminal 1
./run_translation_service_simplified.sh

# Terminal 2
./run_audio_pipeline.sh

# Browser
open http://localhost:8000
```

## Requirements

- macOS with M1 or Intel
- Python 3.8+
- DeepL API key (free tier: 500k chars/month)
- HuggingFace token (free)

## How it works

Audio flows through: microphone -> whisper.cpp -> pyannote -> buffer -> DeepL -> web UI.

Instead of translating every chunk immediately, the system buffers consecutive chunks from the same speaker. Translation happens when the speaker changes, 60 seconds pass, 2000 characters accumulate, or 5 seconds of silence occur. This gives DeepL full context for better translations and reduces API calls.

Czech transcriptions appear in real-time. English translations appear in batches.

## Two profiles

**SPEED** (default): medium model, 4s chunks, no diarization, ~5-6s latency. Good for real-time conversations.

**ACCURACY**: large-v2 model, 10-20s chunks, speaker diarization enabled, ~12-15s latency. Good for podcasts and recordings.

Switch by editing `ACTIVE_PROFILE` in audio_pipeline.py.

## Configuration

Edit audio_pipeline.py:
- `MAX_BUFFER_TIME = 60` - seconds before force flush
- `MAX_BUFFER_CHARS = 2000` - characters before force flush  
- `SILENCE_FLUSH_TIME = 5` - seconds of silence before flush
- `VAD_THRESHOLD = 0.5` - voice activity detection sensitivity

## Performance

M1 Mac with large-v2: ~0.4x real-time processing. 10 seconds of audio processes in 4 seconds. Diarization adds ~0.5x. VAD filters silence. Total latency depends on chunk duration and buffering.

## Files

- `audio_pipeline.py` - transcription, diarization, buffering
- `translation_demo_simplified.py` - translation service and web UI
- `setup.sh` - installs everything
- `requirements.txt` - python dependencies

## Limitations

Czech only (easily extended). Requires internet for translation. Speaker diarization needs 10-15s chunks to work well. Not perfect - expect ~95% transcription accuracy.

## API costs

DeepL free tier: 500k characters/month (~10 hours of transcription). After that: $25 per million characters. Speaker-aware batching reduces API calls by ~70%.

## Dependencies

- whisper.cpp (local transcription, M1 optimized)
- pyannote.audio (speaker diarization, runs locally)
- DeepL API (translation, cloud)
- FastAPI (web server)
- Silero VAD (voice activity detection)

## License

MIT