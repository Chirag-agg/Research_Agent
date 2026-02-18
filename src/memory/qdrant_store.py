"""Qdrant-backed vector store implementation."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class QdrantStore(VectorStore):
    """VectorStore implementation using Qdrant as the backend."""

    def __init__(self, collection: str = "research_memory"):
        self.collection = collection
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info("Qdrant client initialized (cloud) at %s", url)
        else:
            self.client = QdrantClient(host="localhost", port=6333)
            logger.info("Qdrant client initialized (local) at http://localhost:6333")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if missing and validate vector size when present."""
        exists = self.client.collection_exists(self.collection)
        if exists:
            try:
                info = self.client.get_collection(self.collection)
                current_size = info.config.params.vectors.size  # type: ignore[attr-defined]
                if current_size != self.embedding_dim:
                    logger.warning(
                        "Qdrant collection '%s' vector size (%s) != EMBEDDING_DIM (%s)",
                        self.collection,
                        current_size,
                        self.embedding_dim,
                    )
            except Exception as e:
                logger.warning("Could not validate Qdrant collection '%s': %s", self.collection, e)
            return

        logger.info("Creating Qdrant collection '%s' (size=%s, distance=cosine)", self.collection, self.embedding_dim)
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.embedding_dim, distance=qm.Distance.COSINE),
        )

    async def _to_thread(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    async def upsert(self, id: str, vector: List[float], payload: Dict) -> None:
        await self._to_thread(
            self.client.upsert,
            collection_name=self.collection,
            points=[qm.PointStruct(id=id, vector=vector, payload=payload)],
        )

    async def search(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        """Search nearest vectors and return score, payload, and derived confidence."""
        result = await self._to_thread(
            self.client.search,
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
        )

        def _to_confidence(score: float) -> float:
            # Qdrant cosine scores are similarity; clamp to [0,1] for downstream weighting
            try:
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.0

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "confidence": _to_confidence(hit.score),
                "payload": hit.payload,
            }
            for hit in result
        ]

    async def delete(self, id: str) -> None:
        await self._to_thread(
            self.client.delete,
            collection_name=self.collection,
            points_selector=qm.PointIdsList(points=[id]),
        )
