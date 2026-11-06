"""
Memory Vector Store: Pure NumPy vector database implementation.
"""
from typing import List, Tuple
import numpy as np

from core.chunking.chunker_base import DocumentChunk
from core.vectorstores.store_base import BaseVectorStore


class MemoryVectorStore(BaseVectorStore):
    """
    Lightweight, deterministic in-memory vector store using pure NumPy cosine similarity.
    Requires no native C++ bindings, making it 100% portable.
    """

    def __init__(self):
        super().__init__(name="Memory-NumPy")
        self.chunks: List[DocumentChunk] = []
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        if not chunks or not embeddings:
            return

        new_vecs = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-9
        normalized_vecs = new_vecs / norms

        if self.vectors.size == 0:
            self.vectors = normalized_vecs
        else:
            self.vectors = np.vstack([self.vectors, normalized_vecs])

        self.chunks.extend(chunks)

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.vectors.size == 0 or not self.chunks:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-9
        q_normed = q_vec / q_norm

        # Cosine similarity dot product
        similarities = np.dot(self.vectors, q_normed)

        top_indices = np.argsort(similarities)[::-1][:k]
        results: List[Tuple[DocumentChunk, float]] = []

        for idx in top_indices:
            score = float(max(0.0, min(1.0, (similarities[idx] + 1.0) / 2.0)))
            results.append((self.chunks[idx], score))

        return results

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

