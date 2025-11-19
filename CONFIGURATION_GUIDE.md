# CONFIGURATION PROFILES GUIDE

## Two Optimized Profiles

### SPEED Profile (Default)
**Best for: Real-time conversations, live transcription**

Settings:
- Model: medium
- Chunk duration: 4 seconds
- Speaker diarization: Disabled
- VAD threshold: 0.5

Performance:
- Processing time: ~1-2s per chunk
- Total latency: ~5-6s
- Real-time factor: ~0.3x
- Can keep up with normal speech

Trade-offs:
- No speaker labels
- Slightly less accurate than large-v2
- Still excellent Czech transcription quality

---

### ACCURACY Profile
**Best for: Podcast transcription, meeting recordings, multiple speakers**

Settings:
- Model: large-v2
- Chunk duration: 10 seconds
- Speaker diarization: Enabled
- VAD threshold: 0.5

Performance:
- Processing time: ~3-5s per chunk
- Total latency: ~12-15s
- Real-time factor: ~0.4x
- Cannot keep up with fast speech

Benefits:
- Best transcription accuracy
- Automatic speaker identification
- Better handling of accents/unclear speech

---

## How to Switch Profiles

Edit `audio_pipeline.py` line 43:

```python
# For speed:
ACTIVE_PROFILE = "SPEED"

# For accuracy:
ACTIVE_PROFILE = "ACCURACY"
```

Then restart the audio pipeline.

---

## Updated Frontend

The new simplified translation service:
- ✅ Shows Czech input and English translation side-by-side
- ✅ Displays connection status
- ✅ Shows telemetry (translation count, average time)
- ✅ Processing time per entry
- ✅ Entry count per pane
- ❌ Removed: API key input widget (set via env variable)
- ❌ Removed: Manual Czech input form (audio pipeline only)

To use:
1. Set DEEPL_API_KEY environment variable
2. Run `./run_translation_service_simplified.sh`
3. Run `./run_audio_pipeline.sh`
4. Open http://localhost:8000

---

## Telemetry Added

### Console (audio_pipeline.py):
- Profile info on startup
- VAD processing time
- Transcription processing time  
- Real-time factor (RTF)
- "." dots when no speech detected

Example output:
```
⚡ Processing chunk (4s audio, VAD: 12ms)... ✓ 1.3s (RTF: 0.33x)
[SPEAKER_01] Czech: Dobrý den, jak se máte?
```

### Web UI:
- Connection status indicator
- Total translation count
- Average translation time
- Entry count per pane
- Processing time per entry

---

## Environment Variables

Required:
```bash
export DEEPL_API_KEY="your_key_here"
export HF_TOKEN="your_hf_token"  # Only needed for ACCURACY profile
```

Add to ~/.zshrc for persistence:
```bash
echo 'export DEEPL_API_KEY="your_key"' >> ~/.zshrc
echo 'export HF_TOKEN="your_token"' >> ~/.zshrc
source ~/.zshrc
```

---

## Quick Start

```bash
# Set API keys (one time)
export DEEPL_API_KEY="your_deepl_key"
export HF_TOKEN="your_hf_token"

# Terminal 1: Translation service
./run_translation_service_simplified.sh

# Terminal 2: Audio pipeline
./run_audio_pipeline.sh

# Open browser
open http://localhost:8000
```

---

## Performance Comparison

| Profile  | Model    | Chunk | Diarization | Latency | Can Keep Up? |
|----------|----------|-------|-------------|---------|--------------|
| SPEED    | medium   | 4s    | No          | ~5-6s   | ✅ Yes       |
| ACCURACY | large-v2 | 10s   | Yes         | ~12-15s | ⚠️ Barely    |

---

## Recommendations

**For podcasts/recordings:** Use ACCURACY profile
- You're not in a hurry
- Speaker identification is valuable
- Best transcription quality matters

**For live conversations:** Use SPEED profile  
- Need to keep up with speech
- Can manually note speakers
- "Good enough" accuracy is fine

**For meetings:** Try ACCURACY first
- If it falls behind, switch to SPEED
- Can post-process for speaker ID later
