"""Shared data types for generation, encoding, and embedding pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EncodedTrack:
    """Audio track metadata required as input for embedding."""

    track_id: str
    prompt: str
    encoded_path: Path
    duration_sec: float


@dataclass
class EmbeddedTrack:
    """Embedded audio track containing vector representation."""

    track_id: str
    prompt: str
    encoded_path: Path
    duration_sec: float
    embedding: list[float]
