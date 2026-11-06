"""
Google Gemini and OpenAI Embedding providers, and TF-IDF fallback.
"""
import logging
import os
from typing import List, Optional
import numpy as np

from core.embeddings.embedding_base import BaseEmbedding

logger = logging.getLogger(__name__)


class GoogleEmbedding(BaseEmbedding):
    """
    Google Gemini text-embedding provider using official google-genai SDK.
    """

    def __init__(self, model_name: str = "text-embedding-004", api_key: Optional[str] = None):
        super().__init__(model_name=model_name, dimension=768)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning("Could not initialize google-genai client: %s", e)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.client:
            try:
                results = []
                for text in texts:
                    res = self.client.models.embed_content(
                        model=self.model_name,
                        contents=text,
                    )
                    results.append(res.embedding.values)
                return results
            except Exception as e:
                logger.warning("Google embedding call failed: %s", e)

        # Fallback
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.client:
            try:
                res = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                return res.embedding.values
            except Exception as e:
                logger.warning("Google query embedding call failed: %s", e)
        return self._hash_vector(text)

    def _hash_vector(self, text: str) -> List[float]:
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(self.dimension)
        return (vec / (np.linalg.norm(vec) + 1e-9)).tolist()


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI text-embedding provider (text-embedding-3-small / text-embedding-ada-002).
    """

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        super().__init__(model_name=model_name, dimension=1536)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning("Could not initialize OpenAI client: %s", e)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.client:
            try:
                res = self.client.embeddings.create(input=texts, model=self.model_name)
                return [d.embedding for d in res.data]
            except Exception as e:
                logger.warning("OpenAI embedding call failed: %s", e)
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.client:
            try:
                res = self.client.embeddings.create(input=[text], model=self.model_name)
                return res.data[0].embedding
            except Exception as e:
                logger.warning("OpenAI query embedding call failed: %s", e)
        return self._hash_vector(text)

    def _hash_vector(self, text: str) -> List[float]:
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(self.dimension)
        return (vec / (np.linalg.norm(vec) + 1e-9)).tolist()


class TFIDFEmbedding(BaseEmbedding):
    """
    Lightweight, fast scikit-learn TF-IDF embedding provider.
    Zero external network calls, perfect for offline benchmarking.
    """

    def __init__(self, dimension: int = 256):
        super().__init__(model_name="tfidf-dense-projection", dimension=dimension)
        self.vectorizer = None
        self.fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=self.dimension, stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        self.fitted = True

        # Pad if unique terms < dimension
        if tfidf_matrix.shape[1] < self.dimension:
            pad_width = self.dimension - tfidf_matrix.shape[1]
            tfidf_matrix = np.pad(tfidf_matrix, ((0, 0), (0, pad_width)))

        # Normalize
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True) + 1e-9
        normalized = tfidf_matrix / norms
        return normalized.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self.fitted or self.vectorizer is None:
            return [0.0] * self.dimension
        query_vec = self.vectorizer.transform([text]).toarray()
        if query_vec.shape[1] < self.dimension:
            pad_width = self.dimension - query_vec.shape[1]
            query_vec = np.pad(query_vec, ((0, 0), (0, pad_width)))
        norm = np.linalg.norm(query_vec) + 1e-9
        return (query_vec / norm)[0].tolist()

