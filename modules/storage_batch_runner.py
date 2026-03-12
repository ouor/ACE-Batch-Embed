"""Batch helpers for uploading embedded tracks to S3-compatible storage."""

from __future__ import annotations

from modules.types import EmbeddedTrack


def normalize_region_name(raw_region: str | None) -> str:
    """Normalize region input from env/CLI to a valid boto3 region."""
    value = (raw_region or "").strip()
    if not value or value.lower() == "auto":
        return "us-east-1"
    return value


def upload_embedded_tracks_to_storage(
    embedded_tracks: list[EmbeddedTrack],
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str,
    region_name: str = "us-east-1",
    key_prefix: str = "music",
) -> dict[str, str]:
    """Upload embedded track source audio files and return ``track_id -> object_key``."""
    if not embedded_tracks:
        return {}

    from modules.storage import S3CompatibleUploader

    normalized_prefix = (key_prefix or "music").strip().strip("/")
    if not normalized_prefix:
        normalized_prefix = "music"

    uploader = S3CompatibleUploader(
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        bucket=bucket,
        region_name=normalize_region_name(region_name),
        key_prefix=normalized_prefix,
    )
    object_keys: dict[str, str] = {}
    for track in embedded_tracks:
        object_keys[track.track_id] = uploader.upload_file(track.encoded_path, track.track_id)
    return object_keys
