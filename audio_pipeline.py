#!/usr/bin/env python3
"""
Polyglot - Real-Time Speech Translation
Whisper.cpp -> Pyannote Diarization -> DeepL Translation
"""

import warnings
# Filter out torchaudio deprecation warnings (can't fix - library issue)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._internal.module_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.pipelines.speaker_verification")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain.utils.torch_audio_backend")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.tasks.segmentation.mixins")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*std.*degrees of freedom.*")

import asyncio
import json
import wave
import threading
import subprocess
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from queue import Queue
import sys

import numpy as np
import websockets
from pyannote.audio import Pipeline
import pyaudio
import torch

# ============================================================
# CONFIGURATION PROFILES
# ============================================================
# Switch between profiles by changing ACTIVE_PROFILE

PROFILES = {
    "SPEED": {
        "name": "Speed Priority (Real-time capable)",
        "whisper_model": "medium",
        "chunk_duration": 4,
        "enable_diarization": False,
        "vad_threshold": 0.5,
        "description": "Fast transcription, no speaker labels, ~5-6s total latency"
    },
    "ACCURACY": {
        "name": "Accuracy Priority (Speaker identification)",
        "whisper_model": "large-v2", 
        "chunk_duration": 10,
        "enable_diarization": True,
        "vad_threshold": 0.5,
        "description": "Best quality, speaker labels, ~12-15s total latency"
    }
}

# SELECT YOUR PROFILE HERE:
ACTIVE_PROFILE = "SPEED"  # Change to "ACCURACY" for speaker diarization

# Load active configuration
CONFIG = PROFILES[ACTIVE_PROFILE]
WHISPER_MODEL = CONFIG["whisper_model"]
CHUNK_DURATION = CONFIG["chunk_duration"]
ENABLE_DIARIZATION = CONFIG["enable_diarization"]
VAD_THRESHOLD = CONFIG["vad_threshold"]

# ============================================================

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
WEBSOCKET_URL = "ws://localhost:8000/ws"

# Speaker-aware translation batching
MAX_BUFFER_TIME = 60  # seconds - max time to hold before translating
MAX_BUFFER_CHARS = 2000  # characters - max size before translating
SILENCE_FLUSH_TIME = 5  # seconds - flush after this much silence

@dataclass
class TranscriptionSegment:
    text: str
    speaker: str
    start_time: float
    end_time: float


class WhisperTranscriber:
    """Handles whisper.cpp transcription"""
    
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.whisper_path = None
        self.model_path = None
        
    def setup(self):
        """Find whisper.cpp executable and model"""
        # Check if whisper.cpp is available
        whisper_locations = [
            str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
            "/usr/local/bin/whisper-cli",
            str(Path.home() / "whisper.cpp" / "build" / "bin" / "main"),
            "/usr/local/bin/whisper-cpp",
            str(Path.home() / "whisper.cpp" / "main"),
            "./whisper.cpp/main",
            "whisper-cpp"
        ]
        
        for location in whisper_locations:
            if Path(location).exists() or self._command_exists(location):
                self.whisper_path = location
                break
        
        if not self.whisper_path:
            raise RuntimeError(
                "whisper.cpp not found. Please install it first.\n"
                "See setup instructions in the README."
            )
        
        # Find model file
        model_locations = [
            f"/usr/local/share/whisper/ggml-{self.model_name}.bin",
            str(Path.home() / "whisper.cpp" / "models" / f"ggml-{self.model_name}.bin"),
            f"./models/ggml-{self.model_name}.bin"
        ]
        
        for location in model_locations:
            if Path(location).exists():
                self.model_path = location
                break
        
        if not self.model_path:
            raise RuntimeError(
                f"Whisper model '{self.model_name}' not found.\n"
                "Run the download script first."
            )
        
        print(f"[OK] Whisper.cpp found at: {self.whisper_path}")
        print(f"[OK] Model found at: {self.model_path}")
    
    def _command_exists(self, cmd):
        """Check if command exists in PATH"""
        try:
            subprocess.run([cmd, "--help"], capture_output=True, timeout=1)
            return True
        except:
            return False
    
    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file using whisper.cpp"""
        try:
            # Run whisper.cpp
            cmd = [
                self.whisper_path,
                "-m", self.model_path,
                "-f", audio_file,
                "-l", "cs",  # Czech language
                "-nt",  # No timestamps in output
                "-np",  # No progress
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"Whisper error: {result.stderr}")
                return ""
            
            # Extract transcription from output
            # whisper.cpp outputs to stderr by default
            output = result.stderr if result.stderr else result.stdout
            
            # Parse output (format: "[timestamp] text")
            lines = output.strip().split('\n')
            transcription = []
            for line in lines:
                if line.strip() and not line.startswith('['):
                    transcription.append(line.strip())
            
            return ' '.join(transcription)
            
        except subprocess.TimeoutExpired:
            print("Whisper transcription timed out")
            return ""
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return ""


class SpeakerDiarizer:
    """Handles speaker diarization with pyannote"""
    
    def __init__(self):
        self.pipeline = None
        
    def setup(self):
        """Initialize pyannote pipeline"""
        if not ENABLE_DIARIZATION:
            print("[WARN] Speaker diarization disabled (SPEED profile)")
            self.pipeline = None
            return
            
        try:
            # Load pretrained pipeline
            # Requires HuggingFace token set in environment or passed here
            import os
            hf_token = os.environ.get('HF_TOKEN')
            
            if not hf_token:
                print("[WARN] HF_TOKEN not set - skipping speaker diarization")
                self.pipeline = None
                return
            
            print("⏳ Loading speaker diarization (this takes ~30s first time)...")
            
            # Try new API first (token), fall back to old API (use_auth_token)
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
            except TypeError:
                # Old API version
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            
            print("[OK] Pyannote diarization pipeline loaded")
            
        except Exception as e:
            print(f"[WARN] Pyannote setup failed: {e}")
            print("  Continuing without speaker diarization...")
            self.pipeline = None
    
    def diarize(self, audio_file: str) -> List[tuple]:
        """
        Perform speaker diarization
        Returns: List of (start, end, speaker_label) tuples
        """
        if not self.pipeline:
            return []
        
        try:
            # Run diarization
            diarization = self.pipeline(audio_file)
            
            # Extract speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append((turn.start, turn.end, speaker))
            
            return segments
            
        except Exception as e:
            print(f"Diarization error: {e}")
            return []


class VoiceActivityDetector:
    """Silero VAD for detecting speech in audio"""
    
    def __init__(self, threshold: float = VAD_THRESHOLD):
        self.model = None
        self.threshold = threshold
        self.utils = None
        
    def setup(self):
        """Load Silero VAD model"""
        try:
            # Load Silero VAD model from torch hub
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            print("[OK] Silero VAD loaded")
        except Exception as e:
            print(f"[WARN] VAD setup failed: {e}")
            print("  Continuing without VAD (may have more false positives)...")
            self.model = None
    
    def contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech
        Returns: True if speech detected, False otherwise
        """
        if self.model is None:
            return True  # If VAD not available, process everything
        
        try:
            # Convert to float32 and normalize to [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Silero VAD needs 512 samples at 16kHz (32ms windows)
            window_size = 512
            speech_probs = []
            
            # Process audio in 512-sample windows
            for i in range(0, len(audio_float), window_size):
                window = audio_float[i:i + window_size]
                
                # Skip incomplete windows at the end
                if len(window) < window_size:
                    break
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(window)
                
                # Get speech probability for this window
                speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()
                speech_probs.append(speech_prob)
            
            # Return True if any window has speech above threshold
            if speech_probs:
                max_prob = max(speech_probs)
                return max_prob > self.threshold
            
            return False
            
        except Exception as e:
            print(f"VAD error: {e}")
            return True  # On error, process the chunk


@dataclass
class SpeakerBuffer:
    """Buffer for accumulating text from a single speaker"""
    speaker: Optional[str]
    text_chunks: List[str]
    start_time: float
    last_update: float
    
    def add_chunk(self, text: str):
        """Add a text chunk to the buffer"""
        self.text_chunks.append(text)
        self.last_update = time.time()
    
    def get_full_text(self) -> str:
        """Get concatenated text from all chunks"""
        return " ".join(self.text_chunks)
    
    def get_char_count(self) -> int:
        """Get total character count"""
        return len(self.get_full_text())
    
    def get_duration(self) -> float:
        """Get time since buffer started"""
        return time.time() - self.start_time
    
    def get_idle_time(self) -> float:
        """Get time since last update"""
        return time.time() - self.last_update
    
    def should_flush(self) -> bool:
        """Check if buffer should be flushed based on time/size limits"""
        return (
            self.get_duration() >= MAX_BUFFER_TIME or
            self.get_char_count() >= MAX_BUFFER_CHARS or
            self.get_idle_time() >= SILENCE_FLUSH_TIME
        )


class AudioProcessor:
    """Main audio processing pipeline"""
    
    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.vad = VoiceActivityDetector()
        self.audio_queue = Queue()
        self.running = False
        self.websocket = None
        self.current_buffer: Optional[SpeakerBuffer] = None
        self.no_speech_time = 0.0
        self.last_speech_time = time.time()
        
    async def setup(self):
        """Initialize all components"""
        print("Setting up audio processing pipeline...")
        self.transcriber.setup()
        self.vad.setup()
        self.diarizer.setup()
        print("[OK] Pipeline ready\n")
    
    def save_audio_chunk(self, audio_data: np.ndarray) -> str:
        """Save audio chunk to temporary WAV file"""
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False
        )
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        
        return temp_file.name
    
    def assign_speakers_to_text(
        self, 
        text: str, 
        audio_duration: float,
        diarization_segments: List[tuple]
    ) -> str:
        """
        Assign speaker to transcribed text based on diarization
        Simple heuristic: use speaker who spoke most during this chunk
        """
        if not diarization_segments:
            return None
        
        # Count duration per speaker
        speaker_durations = {}
        for start, end, speaker in diarization_segments:
            duration = end - start
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        # Return speaker with most time
        if speaker_durations:
            return max(speaker_durations.items(), key=lambda x: x[1])[0]
        
        return None
    
    async def flush_buffer(self, reason: str = ""):
        """Flush current buffer and send for translation"""
        if not self.current_buffer or not self.current_buffer.text_chunks:
            return
        
        full_text = self.current_buffer.get_full_text()
        speaker = self.current_buffer.speaker
        duration = self.current_buffer.get_duration()
        
        print(f"\n[FLUSH] Flushing buffer: {len(self.current_buffer.text_chunks)} chunks, "
              f"{self.current_buffer.get_char_count()} chars, {duration:.1f}s"
              f"{' (' + reason + ')' if reason else ''}")
        
        # Send to translation
        await self.send_for_translation(full_text, speaker)
        
        # Print to console
        speaker_label = f"[{speaker}] " if speaker else ""
        print(f"{speaker_label}Czech: {full_text}")
        
        # Clear buffer
        self.current_buffer = None
    
    def should_create_new_buffer(self, speaker: Optional[str]) -> bool:
        """Check if we should start a new buffer"""
        if not self.current_buffer:
            return True
        
        # Speaker changed - this is the main trigger
        if speaker != self.current_buffer.speaker:
            return True
        
        # Don't check buffer limits here - they're checked separately
        return False
    
    async def process_audio_chunk(self, audio_data: np.ndarray, duration: float):
        """Process a chunk of audio through the pipeline"""
        start_time = time.time()
        
        # Check for speech using VAD
        vad_start = time.time()
        has_speech = self.vad.contains_speech(audio_data)
        vad_time = time.time() - vad_start
        
        if not has_speech:
            print(".", end="", flush=True)  # Show activity without speech
            
            # Track silence time
            self.no_speech_time = time.time() - self.last_speech_time
            
            # Flush buffer if we've had enough silence
            if self.no_speech_time >= SILENCE_FLUSH_TIME:
                await self.flush_buffer(reason="silence timeout")
                self.no_speech_time = 0
            
            return
        
        # Reset silence tracking
        self.last_speech_time = time.time()
        self.no_speech_time = 0
        
        print(f"\n[PROC] Processing chunk ({duration}s audio, VAD: {vad_time*1000:.0f}ms)...", end="", flush=True)
        
        # Save audio to temp file
        audio_file = self.save_audio_chunk(audio_data)
        
        try:
            # Run transcription and diarization in parallel
            transcription_start = time.time()
            transcription_task = asyncio.to_thread(
                self.transcriber.transcribe, audio_file
            )
            diarization_task = asyncio.to_thread(
                self.diarizer.diarize, audio_file
            )
            
            # Wait for both to complete
            czech_text, diarization = await asyncio.gather(
                transcription_task,
                diarization_task
            )
            transcription_time = time.time() - transcription_start
            
            if not czech_text.strip():
                print(" [no speech detected]")
                return
            
            # Assign speaker
            speaker = self.assign_speakers_to_text(
                czech_text, duration, diarization
            )
            
            # Calculate total processing time
            total_time = time.time() - start_time
            rtf = total_time / duration  # Real-time factor
            
            print(f" [OK] {transcription_time:.1f}s (RTF: {rtf:.2f}x)")
            print(f"  + Chunk: \"{czech_text[:60]}{'...' if len(czech_text) > 60 else ''}\"")
            
            # Send transcription to UI immediately (real-time display)
            await self.send_transcription_only(czech_text, speaker)
            
            # Check if we need to start new buffer due to speaker change
            if self.should_create_new_buffer(speaker):
                # Flush existing buffer first
                if self.current_buffer:
                    await self.flush_buffer(reason="speaker change")
                
                # Start new buffer
                self.current_buffer = SpeakerBuffer(
                    speaker=speaker,
                    text_chunks=[],
                    start_time=time.time(),
                    last_update=time.time()
                )
                print(f"  -> New buffer started for {speaker or 'unknown speaker'}")
            
            # Add chunk to current buffer
            self.current_buffer.add_chunk(czech_text)
            print(f"  -> Buffer: {self.current_buffer.get_char_count()} chars, "
                  f"{len(self.current_buffer.text_chunks)} chunks, "
                  f"{self.current_buffer.get_duration():.1f}s")
            
            # Check if buffer should be flushed due to time/size limits
            if self.current_buffer.should_flush():
                reason = []
                if self.current_buffer.get_duration() >= MAX_BUFFER_TIME:
                    reason.append(f"time limit {MAX_BUFFER_TIME}s")
                if self.current_buffer.get_char_count() >= MAX_BUFFER_CHARS:
                    reason.append(f"char limit {MAX_BUFFER_CHARS}")
                if self.current_buffer.get_idle_time() >= SILENCE_FLUSH_TIME:
                    reason.append(f"silence {SILENCE_FLUSH_TIME}s")
                
                await self.flush_buffer(reason=" | ".join(reason))
            
        finally:
            # Cleanup temp file
            Path(audio_file).unlink(missing_ok=True)
    
    async def send_for_translation(self, czech_text: str, speaker: Optional[str]):
        """Send transcription to translation service via WebSocket"""
        if not self.websocket:
            return
        
        try:
            message = {
                "type": "translate",
                "czech_text": czech_text,
                "speaker": speaker
            }
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            print(f"\n[WARN] WebSocket error: {e}")
            # Try to reconnect
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
            print("  Attempting to reconnect...")
            await self.connect_websocket()
    
    async def send_transcription_only(self, czech_text: str, speaker: Optional[str]):
        """Send Czech transcription to UI immediately (no translation yet)"""
        if not self.websocket:
            return
        
        try:
            message = {
                "type": "transcription",
                "czech_text": czech_text,
                "speaker": speaker
            }
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            print(f"\n[WARN] WebSocket error sending transcription: {e}")
    
    async def connect_websocket(self):
        """Connect to translation service with retry"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.websocket = await websockets.connect(WEBSOCKET_URL)
                print(f"[OK] Connected to translation service at {WEBSOCKET_URL}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[WARN] Connection attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"[WARN] Could not connect to translation service after {max_retries} attempts")
                    print(f"  Error: {e}")
                    print("  Continuing without live translation...")
                    self.websocket = None
    
    def capture_audio_thread(self):
        """Capture audio from microphone in separate thread"""
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024
        )
        
        print(f"\n[REC] Recording audio (Ctrl+C to stop)...")
        print(f"   Processing in {CHUNK_DURATION}s chunks\n")
        
        chunk_samples = SAMPLE_RATE * CHUNK_DURATION
        buffer = []
        
        try:
            while self.running:
                data = stream.read(1024, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                buffer.extend(audio_data)
                
                # When we have enough samples, queue for processing
                if len(buffer) >= chunk_samples:
                    chunk = np.array(buffer[:chunk_samples])
                    self.audio_queue.put(chunk)
                    buffer = buffer[chunk_samples:]
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    async def process_queue(self):
        """Process queued audio chunks"""
        while self.running:
            try:
                # Check for new audio chunks
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get()
                    await self.process_audio_chunk(audio_chunk, CHUNK_DURATION)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Processing error: {e}")
    
    async def run(self):
        """Main processing loop"""
        await self.setup()
        await self.connect_websocket()
        
        self.running = True
        
        # Start audio capture in separate thread
        capture_thread = threading.Thread(
            target=self.capture_audio_thread,
            daemon=True
        )
        capture_thread.start()
        
        try:
            # Process audio queue
            await self.process_queue()
        except KeyboardInterrupt:
            print("\n\n[STOP] Stopping...")
        finally:
            self.running = False
            
            # Flush any remaining buffer
            if self.current_buffer:
                print("\n[FLUSH] Flushing final buffer...")
                await self.flush_buffer(reason="shutdown")
            
            if self.websocket:
                await self.websocket.close()


async def main():
    print("=" * 60)
    print("Polyglot - Real-Time Speech Translation")
    print("=" * 60)
    print()
    print(f"Active Profile: {CONFIG['name']}")
    print(f"  • Model: {WHISPER_MODEL}")
    print(f"  • Chunk duration: {CHUNK_DURATION}s")
    print(f"  • Speaker diarization: {'Enabled' if ENABLE_DIARIZATION else 'Disabled'}")
    print(f"  • {CONFIG['description']}")
    print()
    print("To switch profiles, edit ACTIVE_PROFILE in audio_pipeline.py")
    print("=" * 60)
    print()
    
    processor = AudioProcessor()
    await processor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
        sys.exit(0)