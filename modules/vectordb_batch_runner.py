"""Batch helpers for indexing embedded tracks into Qdrant."""

from __future__ import annotations

from loguru import logger

from modules.types import EmbeddedTrack


def upsert_embedded_tracks_to_qdrant(
    embedded_tracks: list[EmbeddedTrack],
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str,
    object_key_by_track_id: dict[str, str] | None = None,
) -> int:
    """Upsert embedded tracks to Qdrant and return upserted count."""
    if not embedded_tracks:
        return 0

    vector_size = len(embedded_tracks[0].embedding)
    if vector_size <= 0:
        raise ValueError("Embedded track vector is empty.")

    from modules.vectordb import QdrantIndexer

    indexer = QdrantIndexer(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        vector_size=vector_size,
    )
    upserted_count = 0
    for track in embedded_tracks:
        if len(track.embedding) != vector_size:
            logger.warning(
                "Skipping track with inconsistent vector size: track_id={}, expected={}, actual={}",
                track.track_id,
                vector_size,
                len(track.embedding),
            )
            continue
        object_key = (
            object_key_by_track_id.get(track.track_id)
            if object_key_by_track_id is not None
            else track.encoded_path.as_posix()
        )
        if not object_key:
            logger.warning("Skipping track without object_key: track_id={}", track.track_id)
            continue
        indexer.upsert(embedded=track, object_key=object_key)
        upserted_count += 1
    return upserted_count
