"""
Diarization factory - creates appropriate diarizer based on config
"""

from typing import Protocol, List
import os


class DiarizationProtocol(Protocol):
    """Protocol that all diarizers must implement"""
    
    def setup(self) -> None:
        """Initialize the diarizer"""
        ...
    
    def diarize(self, audio_file: str) -> List[tuple]:
        """
        Perform diarization on audio file
        Returns: List of (start, end, speaker_label) tuples
        """
        ...


class NoDiarizer:
    """Dummy diarizer that returns no speaker information"""
    
    def setup(self):
        print("[INFO] Speaker diarization disabled")
    
    def diarize(self, audio_file: str) -> List[tuple]:
        return []


def create_diarizer(method: str, config: dict = None) -> DiarizationProtocol:
    """
    Factory function to create appropriate diarizer
    
    Args:
        method: 'ecapa', 'pyannote', or 'none'
        config: Configuration dict with diarization settings
    
    Returns:
        Diarizer instance
    """
    config = config or {}
    
    if method == "ecapa":
        from ecapa_diarizer import ECAPADiarizer
        
        pending_threshold = config.get('pending_threshold', 0.4)
        min_pending_samples = config.get('min_pending_samples', 5)
        
        return ECAPADiarizer(
            pending_threshold=pending_threshold,
            min_pending_samples=min_pending_samples
        )
    
    elif method == "pyannote":
        # Import the existing pyannote diarizer class
        # We'll need to extract it from audio_pipeline.py
        from pyannote_diarizer import PyannnoteDiarizer
        return PyannnoteDiarizer()
    
    elif method == "none":
        return NoDiarizer()
    
    else:
        raise ValueError(f"Unknown diarization method: {method}. "
                        f"Choose from: ecapa, pyannote, none")


def get_available_methods() -> List[str]:
    """Get list of available diarization methods"""
    methods = ["none"]
    
    # Check if ECAPA dependencies are available
    try:
        import speechbrain
        methods.append("ecapa")
    except ImportError:
        pass
    
    # Check if pyannote is available
    try:
        from pyannote.audio import Pipeline
        if os.environ.get('HF_TOKEN'):
            methods.append("pyannote")
    except ImportError:
        pass
    
    return methods
