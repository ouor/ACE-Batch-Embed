"""Batch helpers for persisting embedding metadata into SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from modules.types import EmbeddedTrack


def write_embeddings_to_timestamped_sqlite(
    embedded_tracks: list[EmbeddedTrack],
    output_dir: str | Path,
    object_key_by_track_id: dict[str, str] | None = None,
) -> Path:
    """Create timestamped SQLite and insert embedding rows."""
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = base_dir / f"embedding_{timestamp}.sqlite3"

    with sqlite3.connect(db_path.as_posix()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS track_embeddings (
                uuid TEXT NOT NULL,
                prompt TEXT NOT NULL,
                duration REAL NOT NULL,
                object_key TEXT NOT NULL,
                emdedding TEXT NOT NULL
            )
            """
        )
        rows = []
        for track in embedded_tracks:
            object_key = (
                object_key_by_track_id.get(track.track_id)
                if object_key_by_track_id is not None
                else track.encoded_path.as_posix()
            )
            rows.append(
                (
                    track.track_id,
                    track.prompt,
                    float(track.duration_sec),
                    object_key or "",
                    json.dumps(track.embedding, ensure_ascii=False),
                )
            )
        conn.executemany(
            """
            INSERT INTO track_embeddings (uuid, prompt, duration, object_key, emdedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return db_path
