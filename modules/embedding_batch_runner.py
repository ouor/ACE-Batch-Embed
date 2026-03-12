"""Batch helpers for converting generated audio outputs into MuQ embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from modules.prompt_batch_runner import BatchItemResult
from modules.types import EmbeddedTrack, EncodedTrack


def collect_encoded_tracks(results: list[BatchItemResult]) -> list[EncodedTrack]:
    """Collect valid generated audio files from batch generation results."""
    encoded_tracks: list[EncodedTrack] = []
    for item in results:
        if not item.success:
            continue
        for audio in item.audios:
            path_value = str(audio.get("path") or "").strip()
            if not path_value:
                continue
            encoded_path = Path(path_value)
            if not encoded_path.exists():
                logger.warning("Generated audio path does not exist: {}", encoded_path)
                continue
            duration_sec = float(audio.get("params", {}).get("duration", 0.0) or 0.0)
            encoded_tracks.append(
                EncodedTrack(
                    track_id=f"batch_{item.index}",
                    prompt=item.prompt,
                    encoded_path=encoded_path,
                    duration_sec=duration_sec,
                )
            )
    return encoded_tracks


def embed_tracks(
    tracks: list[EncodedTrack],
    model_id: str,
    device: str,
) -> list[EmbeddedTrack]:
    """Embed generated audio tracks with MuQ-MuLan."""
    if not tracks:
        return []

    from modules.embedding import MuQMuLanEmbedder

    embedder = MuQMuLanEmbedder(model_id=model_id, device=device)
    return [embedder.embed_audio(track) for track in tracks]


def write_embedding_report(report_path: Path, embedded_tracks: list[EmbeddedTrack]) -> None:
    """Persist embedded vectors and metadata to a JSON report file."""
    payload = [asdict(item) for item in embedded_tracks]
    for item in payload:
        item["encoded_path"] = str(item["encoded_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
