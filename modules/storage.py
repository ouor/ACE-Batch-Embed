from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import boto3


@dataclass
class S3CompatibleUploader:
    """Upload artifacts to S3-compatible storage using boto3."""

    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    region_name: str = "us-east-1"
    key_prefix: str = "music"

    def __post_init__(self) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
        )

    def upload_file(self, local_path: Path, track_id: str) -> str:
        """Upload file and return object key."""
        object_key = f"{self.key_prefix}/{track_id}{local_path.suffix}"
        self._client.upload_file(local_path.as_posix(), self.bucket, object_key)
        return object_key
