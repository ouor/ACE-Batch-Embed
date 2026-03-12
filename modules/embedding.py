from __future__ import annotations

from dataclasses import dataclass

import librosa
import torch
# pip install muq
from muq import MuQMuLan

from modules.types import EmbeddedTrack, EncodedTrack


@dataclass
class MuQMuLanEmbedder:
    """Independent MuQ-MuLan embedder for audio files."""

    model_id: str = "OpenMuQ/MuQ-MuLan-large"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        self._model = MuQMuLan.from_pretrained(self.model_id).to(self.device).eval()

    def embed_audio(self, encoded: EncodedTrack) -> EmbeddedTrack:
        """Load audio as 24k mono and return embedding vector."""
        wav, _ = librosa.load(encoded.encoded_path.as_posix(), sr=24000, mono=True)
        wav_tensor = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            emb = self._model(wavs=wav_tensor)

        vector = emb.squeeze(0).detach().cpu().float().tolist()
        return EmbeddedTrack(
            track_id=encoded.track_id,
            prompt=encoded.prompt,
            encoded_path=encoded.encoded_path,
            duration_sec=encoded.duration_sec,
            embedding=vector,
        )
