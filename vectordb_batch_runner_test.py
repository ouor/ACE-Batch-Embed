"""Unit tests for Qdrant batch indexing helpers."""

import types
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.types import EmbeddedTrack
from modules.vectordb_batch_runner import upsert_embedded_tracks_to_qdrant


class TestVectorDbBatchRunner(unittest.TestCase):
    """Verify Qdrant upsert orchestration behavior."""

    def test_upsert_embedded_tracks_to_qdrant_uses_indexer(self):
        """All valid vectors should be upserted with path-based object keys."""
        fake_module = types.ModuleType("modules.vectordb")
        captured = {"items": []}

        class _FakeIndexer:
            def __init__(self, url: str, api_key: str, collection_name: str, vector_size: int):
                self.url = url
                self.api_key = api_key
                self.collection_name = collection_name
                self.vector_size = vector_size

            def upsert(self, embedded, object_key: str):
                captured["items"].append((embedded.track_id, object_key))

        fake_module.QdrantIndexer = _FakeIndexer
        embedded_tracks = [
            EmbeddedTrack(
                track_id="t1",
                prompt="p1",
                encoded_path=Path("/tmp/a.flac"),
                duration_sec=1.0,
                embedding=[0.1, 0.2],
            ),
            EmbeddedTrack(
                track_id="t2",
                prompt="p2",
                encoded_path=Path("/tmp/b.flac"),
                duration_sec=2.0,
                embedding=[0.3, 0.4],
            ),
        ]

        with patch.dict("sys.modules", {"modules.vectordb": fake_module}):
            count = upsert_embedded_tracks_to_qdrant(
                embedded_tracks=embedded_tracks,
                qdrant_url="http://localhost:6333",
                qdrant_api_key="k",
                collection_name="c",
            )

        self.assertEqual(2, count)
        self.assertEqual(("t1", "/tmp/a.flac"), captured["items"][0])
        self.assertEqual(("t2", "/tmp/b.flac"), captured["items"][1])

    def test_upsert_embedded_tracks_to_qdrant_skips_inconsistent_vector_size(self):
        """Vectors with mismatched dimensions should be skipped."""
        fake_module = types.ModuleType("modules.vectordb")

        class _FakeIndexer:
            def __init__(self, url: str, api_key: str, collection_name: str, vector_size: int):
                self.vector_size = vector_size

            def upsert(self, embedded, object_key: str):
                pass

        fake_module.QdrantIndexer = _FakeIndexer
        embedded_tracks = [
            EmbeddedTrack(
                track_id="t1",
                prompt="p1",
                encoded_path=Path("/tmp/a.flac"),
                duration_sec=1.0,
                embedding=[0.1, 0.2],
            ),
            EmbeddedTrack(
                track_id="t2",
                prompt="p2",
                encoded_path=Path("/tmp/b.flac"),
                duration_sec=2.0,
                embedding=[0.3],
            ),
        ]

        with patch.dict("sys.modules", {"modules.vectordb": fake_module}):
            count = upsert_embedded_tracks_to_qdrant(
                embedded_tracks=embedded_tracks,
                qdrant_url="http://localhost:6333",
                qdrant_api_key="k",
                collection_name="c",
            )

        self.assertEqual(1, count)


if __name__ == "__main__":
    unittest.main()
