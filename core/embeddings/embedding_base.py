"""
Base Embedding Model abstraction.
"""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Abstract base class for all text embedding providers."""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        pass

