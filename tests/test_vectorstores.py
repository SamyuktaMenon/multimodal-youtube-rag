"""
Unit tests for FAISS, ChromaDB, MemoryStore, and Comparator.
"""
import unittest
import numpy as np

from core.chunking.chunker_base import DocumentChunk
from core.vectorstores.chroma_store import ChromaDBVectorStore
from core.vectorstores.comparator import VectorStoreComparator
from core.vectorstores.faiss_store import FAISSVectorStore
from core.vectorstores.memory_store import MemoryVectorStore


class TestVectorStores(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            DocumentChunk("c1", "Attention is all you need for transformers.", 0.0, 10.0, "test"),
            DocumentChunk("c2", "Vector databases index embeddings using HNSW and Flat indices.", 10.0, 20.0, "test"),
            DocumentChunk("c3", "BM25 uses term frequency and inverse document frequency.", 20.0, 30.0, "test"),
            DocumentChunk("c4", "Cross-encoders rerank passages with full cross-attention.", 30.0, 40.0, "test"),
        ]
        # 4 random 128-dim embeddings
        rng = np.random.RandomState(42)
        self.embeddings = rng.randn(4, 128).tolist()
        self.query_embedding = rng.randn(128).tolist()

    def test_faiss_store(self):
        store = FAISSVectorStore(metric="cosine")
        store.add_documents(self.chunks, self.embeddings)
        self.assertEqual(store.count(), 4)

        results = store.similarity_search_with_score(self.query_embedding, k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0][0], DocumentChunk)
        self.assertGreaterEqual(results[0][1], 0.0)

    def test_chroma_store(self):
        store = ChromaDBVectorStore(collection_name="test_unittest_coll")
        store.clear()
        store.add_documents(self.chunks, self.embeddings)
        self.assertEqual(store.count(), 4)

        results = store.similarity_search_with_score(self.query_embedding, k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0][0], DocumentChunk)

    def test_memory_store(self):
        store = MemoryVectorStore()
        store.add_documents(self.chunks, self.embeddings)
        self.assertEqual(store.count(), 4)

        results = store.similarity_search_with_score(self.query_embedding, k=2)
        self.assertEqual(len(results), 2)

    def test_comparator(self):
        comparator = VectorStoreComparator()
        report = comparator.run_benchmark(
            chunks=self.chunks,
            embeddings=self.embeddings,
            k=2,
            num_query_repeats=2,
        )
        self.assertIn("summary_df", report)
        self.assertIn("metrics", report)
        self.assertIn("faiss_index_ms", report["metrics"])


if __name__ == "__main__":
    unittest.main()

