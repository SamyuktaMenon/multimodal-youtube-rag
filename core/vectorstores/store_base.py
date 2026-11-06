"""
Base Vector Store abstraction.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from core.chunking.chunker_base import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract interface for vector database implementations."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Index chunks with their corresponding dense embeddings."""
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for top-k similar chunks with cosine similarity score in [0.0, 1.0].
        Higher score means more similar.
        """
        pass

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
    ) -> List[DocumentChunk]:
        """Convenience method returning just chunks."""
        results = self.similarity_search_with_score(query_embedding, k=k)
        return [chunk for chunk, _ in results]

    @abstractmethod
    def count(self) -> int:
        """Total number of indexed vectors."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the store."""
        pass

