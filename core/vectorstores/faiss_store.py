"""
FAISS Vector Store implementation.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.chunking.chunker_base import DocumentChunk
from core.vectorstores.store_base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    High-performance vector index powered by Facebook AI Similarity Search (FAISS).
    Uses IndexFlatIP on L2-normalized vectors for exact cosine similarity search.
    """

    def __init__(self, metric: str = "cosine"):
        super().__init__(name="FAISS")
        self.metric = metric
        self.index = None
        self.dimension = 0
        self.chunks: List[DocumentChunk] = []

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        if not chunks or not embeddings:
            return

        import faiss

        vecs = np.array(embeddings, dtype=np.float32)
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vecs)

        num_vecs, dim = vecs.shape
        self.dimension = dim

        if self.index is None:
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index = faiss.IndexFlatL2(dim)

        self.index.add(vecs)
        self.chunks.extend(chunks)
        logger.info("Indexed %d chunks into FAISS store (Total: %d)", num_vecs, self.count())

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None or not self.chunks:
            return []

        import faiss

        q_vec = np.array([query_embedding], dtype=np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(q_vec)

        actual_k = min(k, len(self.chunks))
        distances, indices = self.index.search(q_vec, actual_k)

        results: List[Tuple[DocumentChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            # Convert metric distance to normalized score [0, 1]
            if self.metric == "cosine":
                score = float(max(0.0, min(1.0, (dist + 1.0) / 2.0)))
            else:
                score = float(1.0 / (1.0 + dist))
            results.append((chunk, score))

        return results

    def count(self) -> int:
        return self.index.ntotal if self.index else 0

    def clear(self) -> None:
        self.index = None
        self.chunks = []

