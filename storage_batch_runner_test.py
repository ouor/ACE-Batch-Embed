"""Unit tests for storage batch upload helpers."""

import types
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.storage_batch_runner import (
    normalize_region_name,
    upload_embedded_tracks_to_storage,
)
from modules.types import EmbeddedTrack


class TestStorageBatchRunner(unittest.TestCase):
    """Verify S3-compatible upload orchestration behavior."""

    def test_normalize_region_name_supports_auto_and_empty(self):
        """Empty/auto regions should resolve to boto3-safe default."""
        self.assertEqual("us-east-1", normalize_region_name(""))
        self.assertEqual("us-east-1", normalize_region_name("auto"))
        self.assertEqual("ap-northeast-2", normalize_region_name("ap-northeast-2"))

    def test_upload_embedded_tracks_to_storage_returns_object_key_map(self):
        """Uploader should return track_id to object_key mappings."""
        fake_module = types.ModuleType("modules.storage")
        captured = {"keys": []}

        class _FakeUploader:
            def __init__(
                self,
                endpoint_url: str,
                access_key_id: str,
                secret_access_key: str,
                bucket: str,
                region_name: str,
                key_prefix: str,
            ):
                self.key_prefix = key_prefix

            def upload_file(self, local_path: Path, track_id: str) -> str:
                key = f"{self.key_prefix}/{track_id}{local_path.suffix}"
                captured["keys"].append(key)
                return key

        fake_module.S3CompatibleUploader = _FakeUploader
        tracks = [
            EmbeddedTrack(
                track_id="t1",
                prompt="p1",
                encoded_path=Path("/tmp/a.flac"),
                duration_sec=1.0,
                embedding=[1.0],
            )
        ]

        with patch.dict("sys.modules", {"modules.storage": fake_module}):
            object_keys = upload_embedded_tracks_to_storage(
                embedded_tracks=tracks,
                endpoint_url="http://s3.local",
                access_key_id="ak",
                secret_access_key="sk",
                bucket="b",
                region_name="auto",
                key_prefix="music/",
            )

        self.assertEqual({"t1": "music/t1.flac"}, object_keys)
        self.assertEqual(["music/t1.flac"], captured["keys"])


if __name__ == "__main__":
    unittest.main()
