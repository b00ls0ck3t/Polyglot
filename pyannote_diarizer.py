"""
Pyannote speaker diarization implementation
"""

from __future__ import annotations

import os
from typing import List, Optional


class PyannoteDiarizer:
    """Speaker diarization using pyannote.audio"""

    def __init__(self):
        self.pipeline = None

    def setup(self):
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print("[WARN] HF_TOKEN not set - speaker diarization disabled")
            return

        try:
            from pyannote.audio import Pipeline
            print("Loading pyannote diarization pipeline (first time: ~30s)...")
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
            except TypeError:
                # Older pyannote.audio API
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            print("[OK] Pyannote pipeline loaded")
        except Exception as e:
            print(f"[WARN] Pyannote setup failed: {e}")
            print("  Continuing without speaker diarization...")
            self.pipeline = None

    def diarize(self, audio_file: str) -> List[tuple]:
        if not self.pipeline:
            return []

        try:
            diarization = self.pipeline(audio_file)
            return [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
        except Exception as e:
            print(f"Diarization error: {e}")
            return []
