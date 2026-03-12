"""Unit tests for embedding batch orchestration helpers."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.embedding_batch_runner import (
    collect_encoded_tracks,
    embed_tracks,
    write_embedding_report,
)
from modules.prompt_batch_runner import BatchItemResult
from modules.types import EmbeddedTrack


class TestEmbeddingBatchRunner(unittest.TestCase):
    """Verify conversion and persistence behavior for embedding batch steps."""

    def test_collect_encoded_tracks_filters_invalid_items(self):
        """Only successful batch outputs with real audio paths should be collected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "a.flac"
            audio_file.write_bytes(b"test")
            results = [
                BatchItemResult(
                    index=1,
                    prompt="prompt 1",
                    success=True,
                    status_message="ok",
                    error=None,
                    audios=[{"path": str(audio_file), "params": {"duration": 12.5}}],
                ),
                BatchItemResult(
                    index=2,
                    prompt="prompt 2",
                    success=False,
                    status_message="failed",
                    error="err",
                    audios=[{"path": str(audio_file)}],
                ),
                BatchItemResult(
                    index=3,
                    prompt="prompt 3",
                    success=True,
                    status_message="ok",
                    error=None,
                    audios=[{"path": str(Path(tmpdir) / "missing.flac")}],
                ),
            ]

            tracks = collect_encoded_tracks(results)

            self.assertEqual(1, len(tracks))
            self.assertEqual("batch_1", tracks[0].track_id)
            self.assertEqual(12.5, tracks[0].duration_sec)

    def test_embed_tracks_uses_embedder_contract(self):
        """Embedding call should pass every encoded track through the embedder."""
        fake_module = types.ModuleType("modules.embedding")

        class _FakeEmbedder:
            def __init__(self, model_id: str, device: str):
                self.model_id = model_id
                self.device = device

            def embed_audio(self, encoded):
                return EmbeddedTrack(
                    track_id=encoded.track_id,
                    prompt=encoded.prompt,
                    encoded_path=encoded.encoded_path,
                    duration_sec=encoded.duration_sec,
                    embedding=[0.1, 0.2],
                )

        fake_module.MuQMuLanEmbedder = _FakeEmbedder
        tracks = collect_encoded_tracks(
            [
                BatchItemResult(
                    index=1,
                    prompt="prompt",
                    success=True,
                    status_message="ok",
                    error=None,
                    audios=[],
                )
            ]
        )
        self.assertEqual([], tracks)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "a.flac"
            audio_file.write_bytes(b"test")
            tracks = collect_encoded_tracks(
                [
                    BatchItemResult(
                        index=1,
                        prompt="prompt",
                        success=True,
                        status_message="ok",
                        error=None,
                        audios=[{"path": str(audio_file), "params": {}}],
                    )
                ]
            )

            with patch.dict("sys.modules", {"modules.embedding": fake_module}):
                embedded = embed_tracks(tracks, model_id="mid", device="cpu")

            self.assertEqual(1, len(embedded))
            self.assertEqual([0.1, 0.2], embedded[0].embedding)

    def test_write_embedding_report_serializes_paths(self):
        """Embedding report should serialize ``Path`` fields as strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "embedding.json"
            embedded = [
                EmbeddedTrack(
                    track_id="t1",
                    prompt="p1",
                    encoded_path=Path(tmpdir) / "a.flac",
                    duration_sec=1.0,
                    embedding=[1.0],
                )
            ]

            write_embedding_report(report_path, embedded)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(1, len(payload))
            self.assertTrue(isinstance(payload[0]["encoded_path"], str))


if __name__ == "__main__":
    unittest.main()
