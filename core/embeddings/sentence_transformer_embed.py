"""
SentenceTransformers Embedding provider for local HuggingFace embeddings.
"""
import logging
from typing import List, Optional
import numpy as np

from core.embeddings.embedding_base import BaseEmbedding

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Local embedding model using SentenceTransformers (e.g. all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self._model_name = model_name
        self._load_model()
        dim = 384
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            dim = self.model.get_sentence_embedding_dimension()
        super().__init__(model_name=model_name, dimension=dim)

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self._model_name)
            except Exception as e:
                logger.info("SentenceTransformer model %s using pseudo-embedding fallback: %s", self._model_name, e)
                self.model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.tolist()
            except Exception as e:
                logger.error("SentenceTransformer embed_documents failed: %s", e)

        # Fallback to pseudo-deterministic hash vectors if model failed
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.model is not None:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                return embedding.tolist()
            except Exception as e:
                logger.error("SentenceTransformer embed_query failed: %s", e)
        return self._hash_vector(text)

    def _hash_vector(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding fallback."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(self.dimension)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.tolist()
