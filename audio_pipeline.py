#!/usr/bin/env python3
"""
Polyglot - Real-Time Speech Translation
Whisper.cpp -> Diarization -> Translation Service
"""

from __future__ import annotations

import warnings
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
from typing import Optional, List, Dict, Any
from queue import Queue
import sys

import numpy as np
import websockets
import pyaudio
import torch

from config_loader import get_config
from diarization_factory import create_diarizer


class WhisperTranscriber:
    """Handles whisper.cpp transcription"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.whisper_path = None
        self.model_path = None

    def setup(self):
        whisper_locations = [
            str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
            "/usr/local/bin/whisper-cli",
            str(Path.home() / "whisper.cpp" / "build" / "bin" / "main"),
            str(Path.home() / "whisper.cpp" / "main"),
            "/usr/local/bin/whisper-cpp",
            "./whisper.cpp/main",
            "whisper-cpp",
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

        model_locations = [
            f"/usr/local/share/whisper/ggml-{self.model_name}.bin",
            str(Path.home() / "whisper.cpp" / "models" / f"ggml-{self.model_name}.bin"),
            f"./models/ggml-{self.model_name}.bin",
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

    def _command_exists(self, cmd: str) -> bool:
        try:
            subprocess.run([cmd, "--help"], capture_output=True, timeout=1)
            return True
        except Exception:
            return False

    def transcribe(self, audio_file: str) -> str:
        try:
            cmd = [
                self.whisper_path,
                "-m", self.model_path,
                "-f", audio_file,
                "-l", "cs",
                "-nt",
                "-np",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"Whisper error: {result.stderr}")
                return ""

            # Transcription goes to stdout; stderr contains diagnostic messages
            output = result.stdout if result.stdout.strip() else result.stderr

            lines = output.strip().split('\n')
            transcription = [
                line.strip()
                for line in lines
                if line.strip() and not line.startswith('[')
            ]

            return ' '.join(transcription)

        except subprocess.TimeoutExpired:
            print("Whisper transcription timed out")
            return ""
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return ""


class VoiceActivityDetector:
    """Silero VAD for detecting speech in audio"""

    def __init__(self, threshold: float, sample_rate: int):
        self.model = None
        self.threshold = threshold
        self.sample_rate = sample_rate

    def setup(self):
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            print("[OK] Silero VAD loaded")
        except Exception as e:
            print(f"[WARN] VAD setup failed: {e}")
            print("  Continuing without VAD (may have more false positives)...")
            self.model = None

    def contains_speech(self, audio_data: np.ndarray) -> bool:
        if self.model is None:
            return True

        try:
            audio_float = audio_data.astype(np.float32) / 32768.0
            window_size = 512
            speech_probs = []

            for i in range(0, len(audio_float), window_size):
                window = audio_float[i:i + window_size]
                if len(window) < window_size:
                    break
                speech_prob = self.model(torch.from_numpy(window), self.sample_rate).item()
                speech_probs.append(speech_prob)

            return max(speech_probs) > self.threshold if speech_probs else False

        except Exception as e:
            print(f"VAD error: {e}")
            return True


@dataclass
class SpeakerBuffer:
    """Buffer for accumulating text from a single speaker"""
    speaker: Optional[str]
    text_chunks: List[str]
    start_time: float
    last_update: float
    max_time: float
    max_chars: int
    silence_flush: float

    def add_chunk(self, text: str):
        self.text_chunks.append(text)
        self.last_update = time.time()

    def get_full_text(self) -> str:
        return " ".join(self.text_chunks)

    def get_char_count(self) -> int:
        return len(self.get_full_text())

    def get_duration(self) -> float:
        return time.time() - self.start_time

    def get_idle_time(self) -> float:
        return time.time() - self.last_update

    def should_flush(self) -> bool:
        return (
            self.get_duration() >= self.max_time
            or self.get_char_count() >= self.max_chars
            or self.get_idle_time() >= self.silence_flush
        )


class AudioProcessor:
    """Main audio processing pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.sample_rate = config['audio']['sample_rate']
        self.channels = config['audio']['channels']
        self.chunk_duration = config['transcription']['chunk_duration']
        self.websocket_url = config['api']['websocket_url']
        self.buffer_config = config['buffering']

        self.transcriber = WhisperTranscriber(model_name=config['transcription']['model'])
        self.diarizer = create_diarizer(
            method=config['diarization']['method'],
            config=config['diarization'],
        )
        self.vad = VoiceActivityDetector(
            threshold=config['vad']['threshold'],
            sample_rate=self.sample_rate,
        )

        self.audio_queue: Queue = Queue()
        self.running = False
        self.websocket = None
        self.current_buffer: Optional[SpeakerBuffer] = None
        self.last_speech_time = time.time()

    async def setup(self):
        print("Setting up audio processing pipeline...")
        self.transcriber.setup()
        self.vad.setup()
        self.diarizer.setup()
        print("[OK] Pipeline ready\n")

    def _new_buffer(self, speaker: Optional[str]) -> SpeakerBuffer:
        now = time.time()
        return SpeakerBuffer(
            speaker=speaker,
            text_chunks=[],
            start_time=now,
            last_update=now,
            max_time=self.buffer_config['max_time'],
            max_chars=self.buffer_config['max_chars'],
            silence_flush=self.buffer_config['silence_flush'],
        )

    def save_audio_chunk(self, audio_data: np.ndarray) -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return temp_file.name

    def assign_speaker(self, diarization_segments: List[tuple]) -> Optional[str]:
        if not diarization_segments:
            return None
        speaker_durations: Dict[str, float] = {}
        for start, end, speaker in diarization_segments:
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + (end - start)
        return max(speaker_durations, key=speaker_durations.get)

    async def flush_buffer(self, reason: str = ""):
        if not self.current_buffer or not self.current_buffer.text_chunks:
            return

        full_text = self.current_buffer.get_full_text()
        speaker = self.current_buffer.speaker

        print(
            f"\n[FLUSH] {len(self.current_buffer.text_chunks)} chunks, "
            f"{self.current_buffer.get_char_count()} chars"
            f"{' (' + reason + ')' if reason else ''}"
        )

        await self.send_for_translation(full_text, speaker)

        speaker_label = f"[{speaker}] " if speaker else ""
        print(f"{speaker_label}Czech: {full_text}")

        self.current_buffer = None

    async def process_audio_chunk(self, audio_data: np.ndarray):
        start_time = time.time()

        has_speech = self.vad.contains_speech(audio_data)

        if not has_speech:
            print(".", end="", flush=True)
            if time.time() - self.last_speech_time >= self.buffer_config['silence_flush']:
                await self.flush_buffer(reason="silence timeout")
            return

        self.last_speech_time = time.time()
        print(f"\n[PROC] Processing {self.chunk_duration}s chunk...", end="", flush=True)

        audio_file = self.save_audio_chunk(audio_data)

        try:
            czech_text, diarization = await asyncio.gather(
                asyncio.to_thread(self.transcriber.transcribe, audio_file),
                asyncio.to_thread(self.diarizer.diarize, audio_file),
            )

            if not czech_text.strip():
                print(" [no speech detected]")
                return

            speaker = self.assign_speaker(diarization)
            rtf = (time.time() - start_time) / self.chunk_duration

            print(f" [OK] RTF: {rtf:.2f}x")
            print(f"  + \"{czech_text[:60]}{'...' if len(czech_text) > 60 else ''}\"")

            await self.send_transcription_only(czech_text, speaker)

            if not self.current_buffer or speaker != self.current_buffer.speaker:
                if self.current_buffer:
                    await self.flush_buffer(reason="speaker change")
                self.current_buffer = self._new_buffer(speaker)
                print(f"  -> Buffer started for {speaker or 'unknown speaker'}")

            self.current_buffer.add_chunk(czech_text)
            print(
                f"  -> Buffer: {self.current_buffer.get_char_count()} chars, "
                f"{len(self.current_buffer.text_chunks)} chunks"
            )

            if self.current_buffer.should_flush():
                reasons = []
                if self.current_buffer.get_duration() >= self.buffer_config['max_time']:
                    reasons.append("time limit")
                if self.current_buffer.get_char_count() >= self.buffer_config['max_chars']:
                    reasons.append("char limit")
                if self.current_buffer.get_idle_time() >= self.buffer_config['silence_flush']:
                    reasons.append("silence")
                await self.flush_buffer(reason=" | ".join(reasons))

        finally:
            Path(audio_file).unlink(missing_ok=True)

    async def send_for_translation(self, czech_text: str, speaker: Optional[str]):
        if not self.websocket:
            return
        try:
            await self.websocket.send(json.dumps({
                "type": "translate",
                "czech_text": czech_text,
                "speaker": speaker,
            }))
        except Exception as e:
            print(f"\n[WARN] WebSocket send error: {e}")
            self.websocket = None
            await self.connect_websocket()

    async def send_transcription_only(self, czech_text: str, speaker: Optional[str]):
        if not self.websocket:
            return
        try:
            await self.websocket.send(json.dumps({
                "type": "transcription",
                "czech_text": czech_text,
                "speaker": speaker,
            }))
        except Exception as e:
            print(f"\n[WARN] WebSocket send error: {e}")

    async def connect_websocket(self):
        for attempt in range(3):
            try:
                self.websocket = await websockets.connect(self.websocket_url)
                print(f"[OK] Connected to translation service at {self.websocket_url}")
                return
            except Exception as e:
                if attempt < 2:
                    print(f"[WARN] Connection attempt {attempt + 1} failed, retrying in 2s...")
                    await asyncio.sleep(2)
                else:
                    print(f"[WARN] Could not connect to translation service: {e}")
                    print("  Continuing without live translation...")
                    self.websocket = None

    def capture_audio_thread(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
        )

        print(f"\n[REC] Recording audio (Ctrl+C to stop)...")
        print(f"   Processing in {self.chunk_duration}s chunks\n")

        chunk_samples = self.sample_rate * self.chunk_duration
        buffer: List[int] = []

        try:
            while self.running:
                data = stream.read(1024, exception_on_overflow=False)
                buffer.extend(np.frombuffer(data, dtype=np.int16))

                if len(buffer) >= chunk_samples:
                    self.audio_queue.put(np.array(buffer[:chunk_samples]))
                    buffer = buffer[chunk_samples:]
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def process_queue(self):
        while self.running:
            if not self.audio_queue.empty():
                await self.process_audio_chunk(self.audio_queue.get())
            else:
                await asyncio.sleep(0.1)

    async def run(self):
        await self.setup()
        await self.connect_websocket()

        self.running = True

        capture_thread = threading.Thread(target=self.capture_audio_thread, daemon=True)
        capture_thread.start()

        try:
            await self.process_queue()
        except KeyboardInterrupt:
            print("\n\n[STOP] Stopping...")
        finally:
            self.running = False
            if self.current_buffer:
                print("\n[FLUSH] Flushing final buffer...")
                await self.flush_buffer(reason="shutdown")
            if self.websocket:
                await self.websocket.close()


async def main():
    config = get_config()

    print("=" * 60)
    print("Polyglot - Real-Time Speech Translation")
    print("=" * 60)
    print()
    print(f"  Model:        {config['transcription']['model']}")
    print(f"  Chunk:        {config['transcription']['chunk_duration']}s")
    print(f"  Diarization:  {config['diarization']['method']}")
    print(f"  VAD:          {config['vad']['threshold']}")
    print()

    processor = AudioProcessor(config)
    await processor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
        sys.exit(0)
