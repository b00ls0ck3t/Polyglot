"""
ECAPA-TDNN Speaker Diarization Implementation
Lightweight alternative to pyannote for real-time use
"""

import numpy as np
import torch
from typing import Optional, List, Dict
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering


@dataclass
class SpeakerProfile:
    """Profile for a known speaker"""
    speaker_id: str
    embeddings: List[np.ndarray]
    
    def get_representative_embedding(self) -> np.ndarray:
        """Get median embedding as speaker representative"""
        if not self.embeddings:
            return None
        return np.median(self.embeddings, axis=0)
    
    def add_embedding(self, embedding: np.ndarray):
        """Add new embedding to speaker profile"""
        self.embeddings.append(embedding)
        # Keep only last N embeddings to prevent memory bloat
        if len(self.embeddings) > 100:
            self.embeddings = self.embeddings[-100:]


class ECAPADiarizer:
    """
    Speaker diarization using SpeechBrain's ECAPA-TDNN
    Faster and more suitable for real-time than pyannote
    """
    
    def __init__(self, pending_threshold: float = 0.4, min_pending_samples: int = 5):
        self.classifier = None
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.pending_embeddings: List[np.ndarray] = []
        self.pending_threshold = pending_threshold
        self.min_pending_samples = min_pending_samples
        self.next_speaker_id = 0
        
    def setup(self):
        """Initialize ECAPA-TDNN model"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            print("[PROC] Loading ECAPA-TDNN model (first time: ~100MB download)...")
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/ecapa"
            )
            print("[OK] ECAPA-TDNN speaker diarization loaded")
        except Exception as e:
            print(f"[WARN] ECAPA setup failed: {e}")
            print("  Install with: pip install speechbrain")
            print("  Continuing without speaker diarization...")
            self.classifier = None
    
    def extract_embedding(self, audio_file: str) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio file"""
        if self.classifier is None:
            return None
        
        try:
            import torchaudio
            
            # Load audio
            signal, fs = torchaudio.load(audio_file)
            
            # Resample if needed
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.classifier.encode_batch(signal)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"[WARN] Embedding extraction failed: {e}")
            return None
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2)
    
    def identify_speaker(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        """
        Identify speaker from embedding
        Returns: (speaker_id, confidence)
        """
        if embedding is None:
            return None, 0.0
        
        # No speakers yet - this is first speaker
        if not self.speakers:
            speaker_id = self._create_new_speaker(embedding)
            return speaker_id, 1.0
        
        # Compare with all known speakers
        best_match = None
        best_similarity = -1.0
        
        for speaker_id, profile in self.speakers.items():
            rep_embedding = profile.get_representative_embedding()
            if rep_embedding is None:
                continue
            
            similarity = self.cosine_similarity(embedding, rep_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id
        
        # High confidence match
        if best_similarity > 0.7:
            self.speakers[best_match].add_embedding(embedding)
            return best_match, best_similarity
        
        # Medium confidence - add to pending
        elif best_similarity > self.pending_threshold:
            self.pending_embeddings.append(embedding)
            
            # Try to promote pending to new speaker
            if len(self.pending_embeddings) >= self.min_pending_samples:
                new_speaker = self._try_promote_pending()
                if new_speaker:
                    return new_speaker, 0.6
            
            return "pending", best_similarity
        
        # Low confidence - definitely new speaker
        else:
            speaker_id = self._create_new_speaker(embedding)
            return speaker_id, 0.5
    
    def _create_new_speaker(self, embedding: np.ndarray) -> str:
        """Create new speaker profile"""
        speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
        self.next_speaker_id += 1
        
        self.speakers[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            embeddings=[embedding]
        )
        
        print(f"  [INFO] New speaker detected: {speaker_id}")
        return speaker_id
    
    def _try_promote_pending(self) -> Optional[str]:
        """
        Try to cluster pending embeddings and promote largest cluster to new speaker
        Returns: new speaker_id if successful, None otherwise
        """
        if len(self.pending_embeddings) < self.min_pending_samples:
            return None
        
        try:
            # Cluster pending embeddings
            embeddings_array = np.array(self.pending_embeddings)
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.6,
                metric='cosine',
                linkage='average'
            )
            
            labels = clustering.fit_predict(embeddings_array)
            
            # Find largest cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_cluster_label = unique_labels[np.argmax(counts)]
            largest_cluster_size = np.max(counts)
            
            # Only promote if cluster is big enough
            if largest_cluster_size >= self.min_pending_samples:
                # Get embeddings from largest cluster
                cluster_embeddings = embeddings_array[labels == largest_cluster_label]
                
                # Create new speaker
                speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
                self.next_speaker_id += 1
                
                self.speakers[speaker_id] = SpeakerProfile(
                    speaker_id=speaker_id,
                    embeddings=list(cluster_embeddings)
                )
                
                # Remove clustered embeddings from pending
                self.pending_embeddings = [
                    emb for i, emb in enumerate(self.pending_embeddings)
                    if labels[i] != largest_cluster_label
                ]
                
                print(f"  [INFO] Promoted pending to: {speaker_id} ({largest_cluster_size} samples)")
                return speaker_id
            
        except Exception as e:
            print(f"[WARN] Pending promotion failed: {e}")
        
        return None
    
    def diarize(self, audio_file: str) -> List[tuple]:
        """
        Perform speaker diarization on audio file
        Returns: List of (start, end, speaker_label) tuples
        For compatibility with pyannote interface
        """
        if self.classifier is None:
            return []
        
        # Extract embedding
        embedding = self.extract_embedding(audio_file)
        if embedding is None:
            return []
        
        # Identify speaker
        speaker_id, confidence = self.identify_speaker(embedding)
        
        if speaker_id == "pending":
            return []  # Don't assign speaker yet
        
        # Return dummy segment covering whole audio
        # (We don't have timestamp info from ECAPA alone)
        return [(0.0, 1.0, speaker_id)]
