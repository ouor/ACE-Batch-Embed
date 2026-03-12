"""Unit tests for SQLite batch export helpers."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from modules.sqlite_batch_runner import write_embeddings_to_timestamped_sqlite
from modules.types import EmbeddedTrack


class TestSqliteBatchRunner(unittest.TestCase):
    """Verify timestamped SQLite creation and row insertion."""

    def test_write_embeddings_to_timestamped_sqlite_creates_table_and_rows(self):
        """SQLite file should contain requested columns and serialized embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracks = [
                EmbeddedTrack(
                    track_id="u1",
                    prompt="prompt one",
                    encoded_path=Path(tmpdir) / "a.flac",
                    duration_sec=12.5,
                    embedding=[0.1, 0.2],
                ),
                EmbeddedTrack(
                    track_id="u2",
                    prompt="prompt two",
                    encoded_path=Path(tmpdir) / "b.flac",
                    duration_sec=8.0,
                    embedding=[0.3, 0.4],
                ),
            ]
            db_path = write_embeddings_to_timestamped_sqlite(
                embedded_tracks=tracks,
                output_dir=tmpdir,
                object_key_by_track_id={"u1": "music/u1.flac", "u2": "music/u2.flac"},
            )

            self.assertTrue(db_path.exists())
            self.assertTrue(db_path.name.startswith("embedding_"))
            with sqlite3.connect(db_path.as_posix()) as conn:
                rows = conn.execute(
                    "SELECT uuid, prompt, duration, object_key, emdedding FROM track_embeddings ORDER BY uuid"
                ).fetchall()

            self.assertEqual(
                [
                    ("u1", "prompt one", 12.5, "music/u1.flac", "[0.1, 0.2]"),
                    ("u2", "prompt two", 8.0, "music/u2.flac", "[0.3, 0.4]"),
                ],
                rows,
            )


if __name__ == "__main__":
    unittest.main()
