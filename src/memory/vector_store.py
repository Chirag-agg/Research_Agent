"""Abstract vector store interface for semantic storage.

Defines the minimal async contract for inserting, searching, and deleting
vectors with associated payloads, enabling backend-agnostic implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class VectorStore(ABC):
    """Abstract base class for vector-backed semantic storage."""

    @abstractmethod
    async def upsert(self, id: str, vector: List[float], payload: Dict) -> None:
        """Insert or update a vector and its payload."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        """Return the top_k nearest vectors with their payloads."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Remove a vector and its payload from the store."""
        raise NotImplementedError
