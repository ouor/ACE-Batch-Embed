from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http import models

from modules.types import EmbeddedTrack


@dataclass
class QdrantIndexer:
    """Upsert vectors to Qdrant with required payload fields."""

    url: str
    api_key: str
    collection_name: str
    vector_size: int

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme and parsed.hostname:
            default_port = 443 if parsed.scheme == "https" else 80
            self._client = QdrantClient(
                host=parsed.hostname,
                port=parsed.port or default_port,
                https=parsed.scheme == "https",
                api_key=self.api_key,
            )
        else:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        names = {c.name for c in self._client.get_collections().collections}
        if self.collection_name in names:
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(self, embedded: EmbeddedTrack, object_key: str) -> None:
        """Upsert one vector with uuid, duration, prompt, object_key payload."""
        payload = {
            "uuid": embedded.track_id,
            "duration": embedded.duration_sec,
            "prompt": embedded.prompt,
            "object_key": object_key,
        }
        point = models.PointStruct(
            id=embedded.track_id,
            vector=embedded.embedding,
            payload=payload,
        )
        self._client.upsert(collection_name=self.collection_name, points=[point])
